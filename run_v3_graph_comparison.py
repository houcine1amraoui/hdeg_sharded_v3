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


CONDITIONS=('A_reference','B_eligibility','C_dense','D_candidate')
EPSILONS=(1e-4,1e-5,1e-6,1e-7)


def dense_edges(eligible):
    edge=np.outer(eligible,eligible).astype(bool)
    np.fill_diagonal(edge,True)
    return edge


def check_edges(edge,eligible,restricted):
    require(edge.shape==(23,23) and edge.dtype==np.bool_ and np.array_equal(edge,edge.T),'Invalid graph')
    require(np.all(np.diag(edge)),'Missing self loop')
    if restricted:
        allowed=dense_edges(eligible)
        require(not np.any(edge & ~allowed),'Prohibited cross-device edge')


def increments(x,y):
    delta=y.astype(float)-x.astype(float)
    return dict(mse=float(np.square(delta).mean()),max_abs=float(np.abs(delta).max()),
                per_device_mse=np.square(delta).mean(axis=-1).tolist())


def run(args,config,out,report):
    from src.data_preparation import input_state_windows as windows
    for name in ('attribution_report','hierarchy_report'):
        require(getattr(args,name).is_file(),f'--{name} must name a file; received {str(getattr(args,name))!r}')
    require((args.input_dir/'manifest.json').is_file(),'--input_dir must contain manifest.json')
    proof=json.loads(args.attribution_report.read_text());hierarchy=json.loads(args.hierarchy_report.read_text())
    require(proof['status']=='COMPLETE' and proof['protocol']=='source_dbrl_timing_attribution_v1','Expected completed attribution')
    require(hierarchy['status']=='PASS' and hierarchy['protocol']=='source_hierarchy_forward_v1','Expected hierarchy PASS')
    require(sha(args.hierarchy_report)==proof['source_hashes']['hierarchy_report'],'Hierarchy reference differs')
    for r in (proof,hierarchy):
        require(r['parameters_and_buffers_unchanged'] and r['optimizer_steps']==r['backward_calls']==0,'Reference changed')
        require(r['device']=='cpu' and r['batch_size']==8,'CPU/batch-8 required')
        for key in ('hdeg','preprocessing','seed'):
            require(config.get(key)==r['configuration'].get(key),'Configuration differs: '+key)
    deps=dict(attribution_report=args.attribution_report,hierarchy_report=args.hierarchy_report,
        initialization=args.hierarchy_report.parent/'initialization.pt',
        arrays=args.attribution_report.parent/'dbrl_internal_outputs.npz',
        pairs=args.attribution_report.parent/'pair_attribution.json',
        selection=args.attribution_report.parent/'consumed_selection.json',
        runner=Path(__file__),config=Path('configs/config.yaml'),window_loader=Path(windows.__file__))
    for key in ('arrays','pairs','selection'):
        require(sha(deps[key])==proof['output_hashes'][deps[key].name],'Artifact differs: '+key)
    require(sha(deps['initialization'])==proof['initialization_file_sha256']==hierarchy['initialization_file_sha256'],'Initialization differs')
    pairs=json.loads(deps['pairs'].read_text());selection=json.loads(deps['selection'].read_text())
    require(len(pairs)==60 and [r['pair_id'] for r in pairs]==list(range(60)) and selection['total_pairs']==60,'Pair budget differs')
    expected=[]
    for event in selection['selected_events']:
        expected.extend(p['index'] for p in event['placements'])
    expected.extend(c['index'] for c in selection['controls'])
    require(expected==[p['index'] for p in pairs],'Frozen order differs')
    with np.load(deps['arrays'],allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
    validate_arrays(a)
    ds=windows.InputStateWindowDataset(args.input_dir,30)
    require(ds.policy_version==windows.SOURCE_POLICY and ds.manifest_sha256==proof['input_manifest_sha256']==
            selection['manifest_sha256']==hierarchy['manifest_sha256'],'Input lineage differs')
    require(ds.metadata_hashes==proof['metadata_hashes']==hierarchy['metadata_hashes'],'Input metadata differs')
    eligibility={b:[] for b in ('X','Y')}
    for row in pairs:
        block=ds.read_rows(row['index'],row['index']+31)
        eligibility['X'].append(block['status'][29]==0);eligibility['Y'].append(block['status'][30]==0)
    eligibility={b:np.stack(v) for b,v in eligibility.items()}
    import torch
    import torch_geometric
    from torch_geometric.data import Data,Batch
    from src.models.hdeg.graph_reasoner import GraphReasoner
    from src.common.graph.graph_builder import GraphBuilder
    for cls in (GraphReasoner,GraphBuilder):deps[cls.__name__]=Path(inspect.getfile(cls))
    hashes={k:sha(p) for k,p in deps.items()}
    for key in ('window_loader','GraphReasoner','GraphBuilder'):
        require(hashes[key]==proof['source_hashes'][key]==hierarchy['source_hashes'][key],'Executed source differs: '+key)
    report.update(source_hashes=hashes,input_manifest_sha256=ds.manifest_sha256,metadata_hashes=ds.metadata_hashes,
        framework_versions=dict(torch=torch.__version__,torch_geometric=torch_geometric.__version__,numpy=np.__version__),
        reference_framework_versions=proof['framework_versions'])
    torch.set_num_threads(1);torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True)
    state=torch.load(deps['initialization'],map_location='cpu',weights_only=True)
    class SavedState:
        def state_dict(self):return state
    digest=state_digest(SavedState())
    require(digest==proof['initialization_state_sha256']==hierarchy['initialization_state_sha256'],'Saved state differs')
    p=proof['model_parameters'];require(p==hierarchy['model_parameters']['dbrl'],'Model parameters differ')
    require(p['graph_top_k']==15 and p['graph_self_loops'] is True and p['graph_symmetric'] is True and p['embedding_dim']==64,'Unsupported graph contract')
    model=GraphReasoner(64,heads=p['graph_heads'],dropout=p['graph_dropout'])
    prefix='dbrl.graph_reasoner.'
    model.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
    model.eval();model.requires_grad_(False);before=state_digest(model);gat=model.gat
    require(gat.heads==1 and not gat.concat and gat.add_self_loops and gat.edge_dim is None and
            getattr(gat,'res',None) is None and gat.bias is not None,'GAT formula contract differs')
    report.update(initialization_state_sha256=digest,initialization_file_sha256=hashes['initialization'],
        graph_reasoner_state_sha256=before,relation_parameters=dict(negative_slope=gat.negative_slope,heads=gat.heads,
        concat=gat.concat,dropout=gat.dropout,residual=False,edge_dim=None))
    graphs={}
    with torch.inference_mode():
        for b in ('X','Y'):
            graphs['A_reference_'+b]=a[b+'_builder_edges'].copy()
            graphs['C_dense_'+b]=np.ones((60,23,23),bool)
            graphs['D_candidate_'+b]=np.stack([dense_edges(e) for e in eligibility[b]])
            masked=[]
            for i,e in enumerate(eligibility[b]):
                ids=np.flatnonzero(e);edge=np.eye(23,dtype=bool)
                if len(ids):
                    builder=GraphBuilder(top_k=min(15,len(ids)),self_loops=True,symmetric=True)
                    graph=builder.build(torch.from_numpy(a[b+'_H'][i,ids].copy()).unsqueeze(0))
                    ix=graph.edge_index.numpy();edge[ids[ix[0]],ids[ix[1]]]=True
                masked.append(edge)
            graphs['B_eligibility_'+b]=np.stack(masked)
            for condition in CONDITIONS:
                for i,edge in enumerate(graphs[condition+'_'+b]):
                    # Original builder loops are finalized by GAT, as in the reference.
                    effective=edge.copy();np.fill_diagonal(effective,True)
                    check_edges(effective,eligibility[b][i],condition in ('B_eligibility','D_candidate'))
    attention_checks=[]
    def evaluate(h,edges,inspect_attention=False):
        items=[]
        for features,edge in zip(h,edges):
            ix=np.stack(np.nonzero(edge)).astype(np.int64)
            items.append(Data(x=torch.from_numpy(features.copy()),edge_index=torch.from_numpy(ix),num_nodes=23))
        batch=Batch.from_data_list(items);report['graph_reasoner_forward_calls']+=1
        z=model(batch)
        require(tuple(z.shape)==(len(h),23,64) and torch.isfinite(z).all().item() and not z.requires_grad,'Invalid model output')
        if inspect_attention:
            report['attention_probe_gat_calls']+=1
            output,(ix,alpha)=gat(batch.x,batch.edge_index,return_attention_weights=True)
            require(torch.equal(output.reshape_as(z),z),'Attention probe changes output')
            require(torch.isfinite(alpha).all().item() and (alpha>=0).all().item(),'Invalid attention')
            total=torch.zeros((len(h)*23,1));total.index_add_(0,ix[1],alpha)
            error=float((total-1).abs().max());require(error<=1e-6,'Attention not normalized over incoming edges')
            require(torch.equal(ix[0]//23,ix[1]//23),'Cross-sample attention edge')
            effective=edges.copy();effective[:,np.arange(23),np.arange(23)]=True
            observed=np.zeros_like(effective);src,dst=ix.numpy();observed[src//23,src%23,dst%23]=True
            require(np.array_equal(observed,effective) and ix.shape[1]==int(effective.sum()),'Attention edge set/duplicates differ')
            attention_checks.append(dict(samples=len(h),normalization_max_abs_error=error,
                                         minimum_weight=float(alpha.min()),maximum_weight=float(alpha.max())))
        return z.numpy().copy()
    outputs={c+'_'+b:[] for c in CONDITIONS for b in ('X','Y')};replay={};repeat={}
    with torch.inference_mode():
        for start in range(0,60,8):
            stop=min(start+8,60)
            for c in CONDITIONS:
                for b in ('X','Y'):
                    key=c+'_'+b;z=evaluate(a[b+'_H'][start:stop],graphs[key][start:stop],c=='D_candidate')
                    again=evaluate(a[b+'_H'][start:stop],graphs[key][start:stop])
                    require(np.array_equal(z,again),'Exact repeat failed: '+key);repeat[key]=0.
                    if c=='A_reference':
                        require(np.array_equal(z,a[b+'_Z'][start:stop]),'Exact reference replay failed: '+b);replay[b]=0.
                    outputs[key].append(z)
            print('Compared pairs:',stop,'/60',flush=True)
        outputs={k:np.concatenate(v) for k,v in outputs.items()}
        rng=np.random.default_rng(20261025);directions=rng.standard_normal((60,23,64)).astype(np.float32)
        directions/=np.max(np.abs(directions),axis=(1,2),keepdims=True)
        perturbations=[];perturbed_outputs={}
        for eps in EPSILONS:
            h=(a['X_H']+np.float32(eps)*directions).astype(np.float32);parts=[]
            for start in range(0,60,8):parts.append(evaluate(h[start:start+8],graphs['D_candidate_X'][start:start+8]))
            z=np.concatenate(parts);perturbed_outputs[str(eps)]=z
            perturbations.append(dict(epsilon=eps,pairs=[dict(pair_id=i,
                input_change=increments(a['X_H'][i],h[i]),
                output_change=increments(outputs['D_candidate_X'][i],z[i])) for i in range(60)]))
        # One frozen permutation, all 60 X windows: eight additional calls.
        perm=np.random.default_rng(20261026).permutation(23);inverse=np.argsort(perm);permutation_errors=[]
        for start in range(0,60,8):
            stop=min(start+8,60);h=a['X_H'][start:stop][:,perm]
            edge=graphs['D_candidate_X'][start:stop][:,perm][:,:,perm]
            z=evaluate(h,edge)[:,inverse];base=outputs['D_candidate_X'][start:stop]
            require(np.allclose(z,base,atol=1e-7,rtol=1e-5),'Permutation equivariance failed')
            permutation_errors.append(float(np.abs(z.astype(float)-base.astype(float)).max()))
    results=[]
    for i,row in enumerate(pairs):
        identity={k:v for k,v in row.items() if k in ('pair_id','event_id','family','index','destination_offset','channel','channel_index')}
        entry=dict(identity,eligible_X=int(eligibility['X'][i].sum()),eligible_Y=int(eligibility['Y'][i].sum()),
                   eligibility_changed=bool(np.any(eligibility['X'][i]!=eligibility['Y'][i])),conditions={})
        for c in CONDITIONS:
            x,y=outputs[c+'_X'][i],outputs[c+'_Y'][i]
            if row['event_id'] is None:require(np.array_equal(x,y),'Identical-window control failed: '+c)
            entry['conditions'][c]=dict(increment=increments(x,y),
                builder_edges_X=int(graphs[c+'_X'][i].sum()),builder_edges_Y=int(graphs[c+'_Y'][i].sum()))
        results.append(entry)
    groups=[]
    for family,offset in dict.fromkeys((r['family'],r['destination_offset']) for r in pairs):
        ids=[i for i,r in enumerate(pairs) if (r['family'],r['destination_offset'])==(family,offset)]
        groups.append(dict(family=family,offset=offset,count=len(ids),conditions={c:{b:float(outputs[c+'_'+b][ids].astype(float).var(axis=0).mean())
            for b in ('X','Y')} for c in CONDITIONS}))
    require(state_digest(model)==before and all(v.grad is None for v in model.parameters()),'State changed/gradients exist')
    ds.verify_unchanged();require(all(sha(path)==hashes[k] for k,path in deps.items()),'Dependency changed')
    write(out/'pair_comparison.json',results);write(out/'variation_groups.json',groups)
    write(out/'perturbation_results.json',perturbations);write(out/'consumed_selection.json',selection)
    np.savez_compressed(out/'comparison_outputs.npz',**outputs,**{'edges_'+k:v for k,v in graphs.items()},X_eligible=eligibility['X'],Y_eligible=eligibility['Y'])
    np.savez_compressed(out/'perturbation_outputs.npz',directions=directions,**perturbed_outputs)
    report.update(status='COMPLETE',native_replay_max_abs_errors=replay,repeat_max_abs_errors=repeat,
        attention_checks=attention_checks,permutation=perm.tolist(),permutation_max_abs_errors=permutation_errors,
        controls='PASS',parameters_and_buffers_unchanged=True,source_files_unchanged=True,
        output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()})


def main():
    parser=argparse.ArgumentParser(description='Frozen eligibility/dense graph comparison; no training')
    parser.add_argument('--project_root_dir');parser.add_argument('--input_dir',required=True,type=Path)
    parser.add_argument('--attribution_report',required=True,type=Path);parser.add_argument('--hierarchy_report',required=True,type=Path)
    args=parser.parse_args()
    with open('configs/config.yaml') as f:config=yaml.safe_load(f)
    if args.project_root_dir:config['project_root_dir']=args.project_root_dir
    from src.utils.get_folders_utils import get_processed_folder
    out=Path(get_processed_folder(config))/'v2_validation'/'graph_stability_comparison'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    report=dict(status='BLOCKED',protocol='source_graph_stability_comparison_v1',configuration=config,
        device='cpu',batch_size=8,pair_count=60,graph_reasoner_forward_calls=0,attention_probe_gat_calls=0,
        encoder_forward_calls=0,optimizer_steps=0,backward_calls=0,
        planned_budget=dict(comparison_calls=128,perturbation_calls=32,permutation_calls=8,attention_probe_calls=16,total_gat_calls=184),
        perturbation_seed=20261025,permutation_seed=20261026,epsilons=list(EPSILONS),
        permutation_tolerance=dict(atol=1e-7,rtol=1e-5),scientific_claim_decision='NOT_EVALUATED',
        limitations=['Development diagnostic on one untrained initialization, not learned behavior discrimination.',
        'Endpoint eligibility discontinuities are explicit; continuity probes keep eligibility fixed.',
        'No post-hoc stability threshold; perturbation outputs are descriptive, not an automatic stability PASS.',
        'Dense attention may dilute device-specific information; local and downstream usefulness remains untested.',
        'Attention is recomputed for each condition; no new parameters or training.',
        'All-nonvalid behavior is exercised by existing all-missing controls; each self-only neighborhood has attention one.'])
    try:run(args,config,out,report)
    except Exception as exc:report.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    write(out/'graph_comparison_report.json',report)
    print('Graph stability comparison:',report['status']);print('Report:',out/'graph_comparison_report.json')
    if 'error' in report:print(report['error'])
    return 0 if report['status']=='COMPLETE' else 2
if __name__=='__main__':raise SystemExit(main())
