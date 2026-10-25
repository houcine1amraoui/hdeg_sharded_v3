"""Forward-only 2x2 feature/topology intervention on saved DBRL intermediates."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import uuid
import numpy as np
import yaml

LEVELS = ('Z', 'S', 'S_tilde', 'g')
QUOTAS = dict(valid_binary=4, valid_continuous=4, availability_loss=4,
              availability_recovery=4, first_observation=2)
OFFSETS = [1, 15, 30]
ATOL, RTOL = 1e-7, 1e-5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def state_digest(model):
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        a = t.detach().cpu().contiguous().numpy()
        h.update(json.dumps([name, str(a.dtype), list(a.shape)], separators=(',', ':')).encode())
        h.update(a.tobytes())
    return h.hexdigest()


CONDITIONS = ('XX','YX','XY','YY')  # First letter: features; second: topology.


def decomposition(xx, yx, xy, yy):
    xx,yx,xy,yy = [np.asarray(a,dtype=np.float64) for a in (xx,yx,xy,yy)]
    total=yy-xx
    effects=dict(native=total, feature_at_X_graph=yx-xx, feature_at_Y_graph=yy-xy,
                 topology_at_X_features=xy-xx, topology_at_Y_features=yy-yx,
                 interaction=yy-yx-xy+xx)
    metrics={k:dict(mse=float(np.square(v).mean()), max_abs=float(np.abs(v).max()),
                    per_device_mse=np.square(v).mean(axis=-1).tolist()) for k,v in effects.items()}
    paths={}
    for name,f,t in [('features_then_topology',effects['feature_at_X_graph'],effects['topology_at_Y_features']),
                     ('topology_then_features',effects['feature_at_Y_graph'],effects['topology_at_X_features'])]:
        error=float(np.abs(total-f-t).max())
        cross=float(2*np.mean(f*t))
        energy_error=abs(float(np.square(total).mean())-float(np.square(f).mean())-float(np.square(t).mean())-cross)
        require(error<=1e-12 and energy_error<=1e-12,'Decomposition closure failed')
        paths[name]=dict(vector_closure_max_abs=error, twice_mean_cross_product=cross, mse_closure_abs_error=energy_error)
    return dict(effects=metrics,paths=paths)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def validate_arrays(a):
    for b in ('X','Y'):
        for name in ('H','Z'):
            v=a[b+'_'+name]
            require(v.shape==(60,23,64) and v.dtype==np.float32 and np.isfinite(v).all(),'Invalid '+b+'_'+name)
        for name in ('builder_edges','effective_edges'):
            v=a[b+'_'+name]
            require(v.shape==(60,23,23) and v.dtype==np.bool_,'Invalid adjacency '+b+'_'+name)
            require(np.array_equal(v,v.transpose(0,2,1)),'Expected symmetric graph')
        expected=a[b+'_builder_edges'].copy()
        expected[:,np.arange(23),np.arange(23)]=True
        require(np.array_equal(expected,a[b+'_effective_edges']),'Effective topology differs from self-loop contract')


def run(args,config,out,report):
    for name in ('attribution_report','hierarchy_report'):
        require(getattr(args,name).is_file(),f'--{name} must name an existing file; received {str(getattr(args,name))!r}')
    proof=json.loads(args.attribution_report.read_text());hierarchy=json.loads(args.hierarchy_report.read_text())
    require(proof['status']=='COMPLETE' and proof['protocol']=='source_dbrl_timing_attribution_v1','Expected completed attribution')
    require(hierarchy['status']=='PASS' and hierarchy['protocol']=='source_hierarchy_forward_v1','Expected hierarchy PASS')
    require(proof['source_hashes']['hierarchy_report']==sha(args.hierarchy_report),'Hierarchy reference differs')
    require(proof['controls']=='PASS' and proof['source_files_unchanged'] and proof['parameters_and_buffers_unchanged'], 'Attribution gates failed')
    require(all(v==0 for v in proof['replay_max_abs_errors'].values()) and
            all(v==0 for v in proof['repeat_max_abs_errors'].values()),'Exact prior replay/repeat required')
    for r in (proof,hierarchy):
        require(r['device']=='cpu' and r['batch_size']==8 and r['optimizer_steps']==r['backward_calls']==0,'Execution contract differs')
        for k in ('hdeg','preprocessing','seed'):
            require(config.get(k)==r['configuration'].get(k),'Configuration differs: '+k)
    deps=dict(attribution_report=args.attribution_report,hierarchy_report=args.hierarchy_report,
              initialization=args.hierarchy_report.parent/'initialization.pt',
              arrays=args.attribution_report.parent/'dbrl_internal_outputs.npz',
              pairs=args.attribution_report.parent/'pair_attribution.json',
              selection=args.attribution_report.parent/'consumed_selection.json',
              runner=Path(__file__),config=Path('configs/config.yaml'))
    for k in ('arrays','pairs','selection'):
        require(sha(deps[k])==proof['output_hashes'][deps[k].name],'Attribution artifact differs: '+k)
    require(sha(deps['selection'])==proof['source_hashes']['selection'],'Selection lineage differs')
    require(sha(deps['initialization'])==proof['initialization_file_sha256']==hierarchy['initialization_file_sha256'],'Initialization file differs')
    selection=json.loads(deps['selection'].read_text());pairs=json.loads(deps['pairs'].read_text())
    require(selection['protocol']=='source_timing_selection_v1' and selection['status']=='FROZEN_SELECTION' and
            selection['total_pairs']==60 and len(pairs)==60 and [p['pair_id'] for p in pairs]==list(range(60)), 'Invalid frozen pair contract')
    require(selection['manifest_sha256']==proof['input_manifest_sha256']==hierarchy['manifest_sha256'],'Input lineage differs')
    # Saved intermediates are the intervention inputs. No raw-data or encoder rerun is needed.
    with np.load(deps['arrays'],allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
    validate_arrays(a)
    import torch
    import torch_geometric
    from torch_geometric.data import Data,Batch
    from src.models.hdeg.graph_reasoner import GraphReasoner
    deps['GraphReasoner']=Path(inspect.getfile(GraphReasoner))
    hashes={k:sha(p) for k,p in deps.items()}
    require(hashes['GraphReasoner']==proof['source_hashes']['GraphReasoner']==hierarchy['source_hashes']['GraphReasoner'],
            'GraphReasoner source differs')
    report.update(source_hashes=hashes,input_manifest_sha256=proof['input_manifest_sha256'],
                  initialization_file_sha256=hashes['initialization'],
                  framework_versions=dict(torch=torch.__version__,torch_geometric=torch_geometric.__version__,numpy=np.__version__),
                  reference_framework_versions=proof['framework_versions'])
    torch.set_num_threads(1);torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True)
    state=torch.load(deps['initialization'],map_location='cpu',weights_only=True)
    class SavedState:
        def state_dict(self):return state
    full_digest=state_digest(SavedState())
    require(full_digest==proof['initialization_state_sha256']==hierarchy['initialization_state_sha256'],'Full state differs')
    p=proof['model_parameters']
    require(p==hierarchy['model_parameters']['dbrl'] and p['embedding_dim']==64 and
            p['graph_symmetric'] is True and p['graph_self_loops'] is True,'DBRL contract differs')
    model=GraphReasoner(embedding_dim=64,heads=p['graph_heads'],dropout=p['graph_dropout'])
    prefix='dbrl.graph_reasoner.'
    model.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
    model.eval();model.requires_grad_(False);before=state_digest(model)
    require(model.gat.add_self_loops is True and model.gat.edge_dim is None,'GAT contract differs')
    report.update(initialization_state_sha256=full_digest,graph_reasoner_state_sha256=before)
    outputs={k:[] for k in CONDITIONS};replay=dict(X=0.,Y=0.);repeats={k:0. for k in CONDITIONS}

    def evaluate(features,topology,start,stop):
        graphs=[]
        for i in range(start,stop):
            # Lexical source/target order matches GraphBuilder's torch.unique edges.
            edge=np.stack(np.nonzero(a[topology+'_builder_edges'][i])).astype(np.int64)
            graphs.append(Data(x=torch.from_numpy(a[features+'_H'][i].copy()),
                               edge_index=torch.from_numpy(edge),num_nodes=23))
        batch=Batch.from_data_list(graphs)
        report['graph_reasoner_forward_calls']+=1
        z=model(batch)
        require(tuple(z.shape)==(stop-start,23,64) and torch.isfinite(z).all().item() and not z.requires_grad,'Invalid intervention output')
        return z.numpy().copy()

    with torch.inference_mode():
        for start in range(0,60,8):
            stop=min(start+8,60);current={}
            # Native corners first; a failed replay blocks crossed corners in this batch.
            for condition in ('XX','YY','YX','XY'):
                value=evaluate(condition[0],condition[1],start,stop)
                repeated=evaluate(condition[0],condition[1],start,stop)
                repeats[condition]=max(repeats[condition],float(np.abs(value.astype(float)-repeated.astype(float)).max()))
                require(np.array_equal(value,repeated),'Exact repeat failed: '+condition)
                if condition in ('XX','YY'):
                    branch=condition[0];old=a[branch+'_Z'][start:stop]
                    replay[branch]=max(replay[branch],float(np.abs(value.astype(float)-old.astype(float)).max()))
                    require(np.array_equal(value,old),'Exact native Z replay failed: '+condition)
                current[condition]=value;outputs[condition].append(value)
            for j,i in enumerate(range(start,stop)):
                if np.array_equal(a['X_effective_edges'][i],a['Y_effective_edges'][i]):
                    require(np.array_equal(current['XX'][j],current['XY'][j]) and np.array_equal(current['YX'][j],current['YY'][j]),
                            'Equal-topology control failed')
                if np.array_equal(a['X_H'][i],a['Y_H'][i]):
                    require(np.array_equal(current['XX'][j],current['YX'][j]) and np.array_equal(current['XY'][j],current['YY'][j]),
                            'Equal-feature control failed')
            print('Intervened pairs:',stop,'/60',flush=True)
    outputs={k:np.concatenate(v) for k,v in outputs.items()}
    results=[]
    for i,row in enumerate(pairs):
        identity={k:v for k,v in row.items() if k in ('pair_id','event_id','family','index','destination_offset','channel','channel_index','event_row','from_status','to_status')}
        results.append(dict(identity,**decomposition(*(outputs[k][i] for k in CONDITIONS)),
                            effective_edge_symmetric_difference=row['effective_edges']['symmetric_difference']))
    require(state_digest(model)==before and all(p.grad is None for p in model.parameters()),'Model state changed or gradients exist')
    require(all(sha(path)==hashes[k] for k,path in deps.items()),'Source artifact changed')
    write(out/'pair_interventions.json',results);write(out/'consumed_selection.json',selection)
    np.savez_compressed(out/'fixed_topology_outputs.npz',**outputs)
    report.update(status='COMPLETE',pair_count=60,native_replay_max_abs_errors=replay,repeat_max_abs_errors=repeats,
                  parameters_and_buffers_unchanged=True,source_files_unchanged=True,structural_controls='PASS',
                  output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project_root_dir')
    p.add_argument('--attribution_report',required=True,type=Path);p.add_argument('--hierarchy_report',required=True,type=Path)
    args=p.parse_args()
    with open('configs/config.yaml') as f:config=yaml.safe_load(f)
    if args.project_root_dir:config['project_root_dir']=args.project_root_dir
    from src.utils.get_folders_utils import get_processed_folder
    out=Path(get_processed_folder(config))/'v2_validation'/'fixed_topology_intervention'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status='BLOCKED',protocol='source_fixed_topology_v1',configuration=config,
                device='cpu',batch_size=8,graph_reasoner_forward_calls=0,encoder_forward_calls=0,
                optimizer_steps=0,backward_calls=0,scientific_claim_decision='NOT_EVALUATED',
                condition_order=list(CONDITIONS),condition_definition='First letter: temporal features; second letter: graph topology',
                limitations=['Computational intervention within this fixed untrained model, not causal user-behavior evidence.',
                'All 60 frozen pairs retained; overlapping placements are not independent samples.',
                'Graph attention is recomputed under each condition; connectivity is fixed, attention weights are not.',
                'Vector effects are path-dependent and interacting; MSE contributions are not additive without cross terms.',
                'Saved intermediates are verified; raw inputs and encoder are not rerun.',
                'No missing-node removal, top-k redesign, optimization or downstream hierarchy evaluation.'])
    try:run(args,config,out,report)
    except Exception as exc:report.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    write(out/'fixed_topology_report.json',report)
    print('Fixed-topology intervention:',report['status']);print('Report:',out/'fixed_topology_report.json')
    if 'error' in report:print(report['error'])
    return 0 if report['status']=='COMPLETE' else 2

if __name__=='__main__':raise SystemExit(main())
