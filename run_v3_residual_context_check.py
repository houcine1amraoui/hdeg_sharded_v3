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
    for name in ('attribution_report','hierarchy_report','comparison_report'):
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
    # Dense baseline and offline analytical references are additional immutable inputs.
    cp=json.loads(args.comparison_report.read_text())
    require(cp['status']=='COMPLETE' and cp['protocol']=='source_graph_stability_comparison_v1','Expected completed graph comparison')
    require(cp['source_hashes']['attribution_report']==sha(args.attribution_report) and
            cp['initialization_state_sha256']==digest,'Comparison lineage differs')
    require(cp['configuration']['hdeg']==config['hdeg'],'Comparison configuration differs')
    require(cp['source_hashes']['GraphReasoner']==hashes['GraphReasoner'],'Comparison model source differs')
    deps.update(comparison_report=args.comparison_report,
                comparison_arrays=args.comparison_report.parent/'comparison_outputs.npz',
                perturbation_arrays=args.comparison_report.parent/'perturbation_outputs.npz',
                analysis_report=args.analysis_dir/'analysis_report.json',
                analytical_arrays=args.analysis_dir/'analytical_representations.npz')
    for key in ('comparison_arrays','perturbation_arrays'):
        require(sha(deps[key])==cp['output_hashes'][deps[key].name],'Comparison output differs: '+key)
    ar=json.loads(deps['analysis_report'].read_text())
    require(ar['status']=='COMPLETE' and ar['protocol']=='offline_residual_context_v1' and
            ar['gamma']==1.0 and ar['initialization_state_sha256']==digest,'Offline analysis contract differs')
    with np.load(deps['comparison_arrays'],allow_pickle=False) as f:baseline={k:f[k].copy() for k in f.files}
    with np.load(deps['perturbation_arrays'],allow_pickle=False) as f:perturb={k:f[k].copy() for k in f.files}
    with np.load(deps['analytical_arrays'],allow_pickle=False) as f:analytical={k:f[k].copy() for k in f.files}
    hashes={k:sha(path) for k,path in deps.items()};report['source_hashes']=hashes
    W=state['dbrl.graph_reasoner.gat.lin.weight'].numpy().astype(float)
    bias=state['dbrl.graph_reasoner.gat.bias'].numpy().astype(float)
    for b in ('X','Y'):
        require(np.array_equal(eligibility[b],baseline[b+'_eligible']),'Eligibility differs')
        require(np.array_equal(analytical[b+'_H'],a[b+'_H']) and
                np.array_equal(analytical[b+'_C'],baseline['D_candidate_'+b]),'Offline source arrays differ')
        expected=a[b+'_H'].astype(float)+baseline['D_candidate_'+b].astype(float)-(a[b+'_H'].astype(float)@W.T+bias)
        require(np.allclose(expected,analytical[b+'_R'],atol=1e-12,rtol=0),'Analytical formula differs')
    graphs={b:np.stack([dense_edges(e) for e in eligibility[b]]) for b in ('X','Y')}
    for b in ('X','Y'):
        require(np.array_equal(graphs[b],baseline['edges_D_candidate_'+b]),'Dense candidate graph differs')
    def evaluate(h,e):
        edges=np.stack([dense_edges(v) for v in e]);items=[]
        for features,edge in zip(h,edges):
            ix=np.stack(np.nonzero(edge)).astype(np.int64)
            items.append(Data(x=torch.from_numpy(features.copy()),edge_index=torch.from_numpy(ix),num_nodes=23))
        batch=Batch.from_data_list(items);report['graph_reasoner_forward_calls']+=1
        c=model(batch);H=torch.from_numpy(h.copy())
        # Same saved GAT projection, computed in float32; no new parameters.
        local=gat.lin(batch.x).reshape_as(H)+gat.bias
        q=c-local;raw=H+q
        has_peer=torch.from_numpy(e & (e.sum(axis=1,keepdims=True)>1))
        residual=torch.where(has_peer.unsqueeze(-1),raw,H)
        applied=torch.where(has_peer.unsqueeze(-1),q,torch.zeros_like(q))
        for value in (c,local,raw,residual,applied):
            require(tuple(value.shape)==(len(h),23,64) and torch.isfinite(value).all().item() and not value.requires_grad,'Invalid output')
        require(torch.equal(residual[~has_peer],H[~has_peer]),'Exact local fallback failed')
        return {k:v.numpy().copy() for k,v in dict(C=c,L=local,R_raw=raw,R=residual,Q=applied).items()}
    collected={};repeat_errors={};analytic_errors={};fallback_counts={}
    with torch.inference_mode():
        for start in range(0,60,8):
            stop=min(start+8,60)
            for b in ('X','Y'):
                first=evaluate(a[b+'_H'][start:stop],eligibility[b][start:stop])
                second=evaluate(a[b+'_H'][start:stop],eligibility[b][start:stop])
                for key,value in first.items():
                    require(np.array_equal(value,second[key]),'Exact repeat failed: '+b+'_'+key)
                    repeat_errors[b+'_'+key]=0.;collected.setdefault(b+'_'+key,[]).append(value)
                require(np.array_equal(first['C'],baseline['D_candidate_'+b][start:stop]),'Exact dense-context replay failed: '+b)
                for key in ('R','R_raw'):
                    ref=analytical[b+'_R'][start:stop]
                    error=float(np.abs(first[key].astype(float)-ref).max())
                    analytic_errors[b+'_'+key]=max(analytic_errors.get(b+'_'+key,0.),error)
                    require(np.allclose(first[key],ref,atol=1e-7,rtol=1e-5),'Float32/analytical discrepancy exceeds tolerance: '+b+'_'+key)
            print('Residual pairs checked:',stop,'/60',flush=True)
        outputs={k:np.concatenate(v) for k,v in collected.items()}
        directions=perturb['directions']
        require(directions.shape==(60,23,64) and directions.dtype==np.float32 and np.isfinite(directions).all(),'Invalid perturbation directions')
        rng=np.random.default_rng(20261025);expected=rng.standard_normal((60,23,64)).astype(np.float32)
        expected/=np.max(np.abs(expected),axis=(1,2),keepdims=True)
        require(np.array_equal(directions,expected),'Frozen perturbation directions differ')
        perturbations=[];perturb_outputs={}
        for eps in EPSILONS:
            hp=(a['X_H']+np.float32(eps)*directions).astype(np.float32);parts=[]
            for start in range(0,60,8):
                stop=min(start+8,60);value=evaluate(hp[start:stop],eligibility['X'][start:stop])
                require(np.array_equal(value['C'],perturb[str(eps)][start:stop]),'Perturbed dense-context replay failed')
                parts.append(value['R'])
            rp=np.concatenate(parts);perturb_outputs[str(eps)]=rp
            perturbations.append(dict(epsilon=eps,pairs=[dict(pair_id=i,input_change=increments(a['X_H'][i],hp[i]),
                output_change=increments(outputs['X_R'][i],rp[i])) for i in range(60)]))
        perm=np.random.default_rng(20261026).permutation(23);inv=np.argsort(perm);permutation_errors=[]
        for start in range(0,60,8):
            stop=min(start+8,60);v=evaluate(a['X_H'][start:stop][:,perm],eligibility['X'][start:stop][:,perm])['R'][:,inv]
            ref=outputs['X_R'][start:stop];error=float(np.abs(v.astype(float)-ref.astype(float)).max())
            require(np.allclose(v,ref,atol=1e-7,rtol=1e-5),'Permutation test failed');permutation_errors.append(error)
    rows=[];distinction=[]
    for i,pair in enumerate(pairs):
        r={k:v for k,v in pair.items() if k in ('pair_id','event_id','family','index','destination_offset','channel','channel_index')}
        r['increments']={key:increments(outputs['X_'+key][i],outputs['Y_'+key][i]) for key in ('C','R')}
        r['increments']['H']=increments(a['X_H'][i],a['Y_H'][i])
        if pair['event_id'] is None:require(r['increments']['R']['mse']==0,'Identical-window residual control failed')
        rows.append(r)
        for b in ('X','Y'):
            e=eligibility[b][i];self_only=~(e & (e.sum()>1))
            fallback_counts[b]=fallback_counts.get(b,0)+int(self_only.sum())
            for subset,mask in [('eligible',e),('noneligible',~e)]:
                if mask.sum()<2:continue
                distinction.append(dict(pair_id=i,side=b,subset=subset,nodes=int(mask.sum()),
                    mean_feature_variance={name:float(value[mask].astype(float).var(axis=0).mean()) for name,value in
                    [('H',a[b+'_H'][i]),('C',outputs[b+'_C'][i]),('R',outputs[b+'_R'][i])]}))
    require(state_digest(model)==before and all(p.grad is None for p in model.parameters()),'Model changed or gradients exist')
    ds.verify_unchanged();require(all(sha(path)==hashes[k] for k,path in deps.items()),'Dependency changed')
    write(out/'pair_metrics.json',rows);write(out/'device_distinction.json',distinction)
    write(out/'perturbation_results.json',perturbations);write(out/'consumed_selection.json',selection)
    np.savez_compressed(out/'residual_context_outputs.npz',**outputs,X_eligible=eligibility['X'],Y_eligible=eligibility['Y'])
    np.savez_compressed(out/'perturbation_outputs.npz',**perturb_outputs)
    report.update(status='COMPLETE',dense_context_replay='EXACT',perturbed_context_replay='EXACT',
        repeat_max_abs_errors=repeat_errors,analytical_max_abs_errors=analytic_errors,
        permutation_max_abs_errors=permutation_errors,exact_fallback_node_instances=fallback_counts,
        controls='PASS',parameters_and_buffers_unchanged=True,source_files_unchanged=True,
        output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()})


def main():
    parser=argparse.ArgumentParser(description='Bounded float32 residual-context check; no training')
    parser.add_argument('--project_root_dir');parser.add_argument('--input_dir',required=True,type=Path)
    for key in ('attribution_report','hierarchy_report','comparison_report','analysis_dir'):
        parser.add_argument('--'+key,required=True,type=Path)
    args=parser.parse_args()
    with open('configs/config.yaml') as f:config=yaml.safe_load(f)
    if args.project_root_dir:config['project_root_dir']=args.project_root_dir
    from src.utils.get_folders_utils import get_processed_folder
    out=Path(get_processed_folder(config))/'v2_validation'/'residual_context_check'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    report=dict(status='BLOCKED',protocol='source_residual_context_check_v1',configuration=config,
        device='cpu',batch_size=8,gamma=1.,formula='H + C - (WH+b); self-only nodes return H exactly',
        graph_reasoner_forward_calls=0,encoder_forward_calls=0,optimizer_steps=0,backward_calls=0,
        planned_budget=dict(base_and_repeat=32,perturbations=32,permutation=8,total_graph_reasoner_calls=72),
        analytical_tolerance=dict(atol=1e-7,rtol=1e-5),permutation_tolerance=dict(atol=1e-7,rtol=1e-5),
        epsilons=list(EPSILONS),perturbation_seed=20261025,permutation_seed=20261026,
        scientific_claim_decision='NOT_EVALUATED',limitations=[
        'Single initialization and same 60 development pairs; no learned behavior-discrimination claim.',
        'Gamma fixed at one; standard model is unchanged; no training or checkpoint writes.',
        'Self-only fallback explicitly returns H, including a sole eligible node.',
        'Float32 residual is compared with float64 analytical arrays; bitwise equality is not expected.',
        'Continuity probes keep eligibility fixed; endpoint eligibility discontinuities remain.',
        'Larger device variance does not establish useful relational or behavioral information.'])
    try:run(args,config,out,report)
    except Exception as exc:report.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    write(out/'residual_context_report.json',report)
    print('Residual-context check:',report['status']);print('Report:',out/'residual_context_report.json')
    if 'error' in report:print(report['error'])
    return 0 if report['status']=='COMPLETE' else 2
if __name__=='__main__':raise SystemExit(main())
