"""Bounded forward-only DBRL -> BSE -> BIL -> EBRL -> standard HBF check."""
import argparse
from datetime import datetime,timezone
import hashlib
import inspect
import json
from pathlib import Path
import uuid
import numpy as np
import yaml

LEVELS=('Z','S','S_tilde','g')
STATE_IDS=('access_entry','occupancy_presence','physical_disturbance','thermal_environment',
 'switch_controlled_appliance_activity','lighting_activity','air_quality','water_heating','humidification')
EDGES=((0,1),(1,4),(1,5),(1,7),(4,3),(7,3),(8,6))
STRATA=('all_missing','partial_quiet','valid_value_transition','status_transition')


def require(c,msg):
    if not c:raise ValueError(msg)


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def rolling_sum(x,width):
    c=np.concatenate(([0],np.cumsum(x,dtype=np.int64)))
    return c[width:]-c[:-width]


def select_pairs(values,status):
    require(values.shape==status.shape and values.shape[1]==23 and len(values)>=31,'Unexpected row arrays')
    missing=rolling_sum(np.all(status==3,axis=1),31)==31
    any_valid=rolling_sum(np.any(status==0,axis=1),31)>0
    any_nonvalid=rolling_sum(np.any(status!=0,axis=1),31)>0
    delta_status=np.any(status[1:]!=status[:-1],axis=1)
    delta_value=np.any((status[1:]==0)&(status[:-1]==0)&(values[1:]!=values[:-1]),axis=1)
    sc=rolling_sum(delta_status,30)>0;vc=rolling_sum(delta_value,30)>0
    masks=dict(all_missing=missing,partial_quiet=~sc&~vc&any_valid&any_nonvalid,
               valid_value_transition=~sc&vc,status_transition=sc)
    selected=[];counts={}
    for name in STRATA:
        candidates=np.flatnonzero(masks[name]);counts[name]=len(candidates)
        require(len(candidates)>=8,'Fewer than eight eligible pairs in stratum: '+name)
        for index in candidates[np.linspace(0,len(candidates)-1,8,dtype=np.int64)]:
            i=int(index)
            selected.append(dict(index=i,stratum=name,valid_fraction=float(np.mean(status[i:i+31]==0)),
               missing_fraction=float(np.mean(status[i:i+31]==3)),
               newly_arriving_status_transition=bool(delta_status[i+29]),
               newly_arriving_valid_value_transition=bool(delta_value[i+29])))
    require(len({x['index'] for x in selected})==32,'Duplicate selected indices')
    return selected,counts


def state_digest(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().contiguous().numpy()
        h.update(json.dumps([name,str(a.dtype),list(a.shape)],separators=(',',':')).encode());h.update(a.tobytes())
    return h.hexdigest()


def variation(a):
    a=a.astype(np.float64).reshape(len(a),-1)
    return dict(samples=len(a),mean_feature_variance=float(a.var(axis=0).mean()),
                mean_squared_norm=float(np.square(a).sum(axis=1).mean()),
                maximum_absolute_value=float(np.abs(a).max()))


def run(args,config,out,report):
    import torch
    import torch_geometric
    from src.data_preparation import input_state_windows as windows
    from src.common.graph.semantics import load_behavioral_state_config
    from src.models.hdeg.dbrl import DBRL
    from src.models.hdeg.bse import BehavioralStateEstimator
    from src.models.hdeg.bil import BehavioralInteractionLearner
    from src.models.hdeg.ebrl import EcosystemBehavioralRepresentationLearner
    from src.models.hdeg.hbf import HierarchicalBehavioralForecaster
    from src.models.hdeg.graph_reasoner import GraphReasoner
    from src.common.graph.graph_builder import GraphBuilder
    proof=json.loads(args.window_report.read_text())
    require(proof['status']=='PASS' and proof['policy_version']==windows.SOURCE_POLICY,'Expected source-window PASS report')
    require(proof['source_files_unchanged'] is True and proof['model_forwards']==proof['optimizer_steps']==0,'Unexpected loader verification')
    require(config['preprocessing']['window_size']==30,'Window size must be 30')
    ds=windows.InputStateWindowDataset(args.input_dir,30)
    require(ds.policy_version==windows.SOURCE_POLICY and ds.manifest_sha256==proof['manifest_sha256'],'Input policy/manifest differs')
    require(ds.metadata_hashes==proof['metadata_hashes'],'Metadata differs from loader PASS')
    require(windows.sha256(Path(windows.__file__))==proof['loader_sha256'],'Loader source differs from PASS report')
    require(ds.num_rows<=3_000_000 and len(ds.shards)<=512,'Row/shard bounds exceeded')
    deps={'config':Path('configs/config.yaml'),'runner':Path(__file__),'window_report':args.window_report,
          'window_loader':Path(windows.__file__),'semantic_config':args.semantic_config or Path(config['project_root_dir'])/'configs/hdeg/behavioral_states.yaml',
          'semantic_loader':Path(inspect.getfile(load_behavioral_state_config))}
    for cls in [DBRL,BehavioralStateEstimator,BehavioralInteractionLearner,EcosystemBehavioralRepresentationLearner,
                HierarchicalBehavioralForecaster,GraphReasoner,GraphBuilder]:deps[cls.__name__]=Path(inspect.getfile(cls))
    hashes={k:windows.sha256(p) for k,p in deps.items()}
    write(out/'devices.json',ds.devices)
    semantics=load_behavioral_state_config(deps['semantic_config'],out/'devices.json')
    require(list(semantics.device_ids())==ds.devices and semantics.state_ids()==STATE_IDS,'Semantic order differs')
    values=np.empty((ds.num_rows,23),dtype=np.float32);status=np.empty((ds.num_rows,23),dtype=np.uint8)
    for i,e in enumerate(ds.shards):
        a=ds.load_shard(i);values[e['start']:e['stop']]=a['values'];status[e['start']:e['stop']]=a['status']
    selected,counts=select_pairs(values,status)
    for row in selected:row['first_timestamp']=str(ds.first+np.timedelta64(row['index'],'s'))
    del values,status
    selection=dict(protocol='source_hierarchy_forward_v1',manifest_sha256=ds.manifest_sha256,
         candidate_counts=counts,selected_pairs=selected,selection_rule='8 evenly ranked starts per disjoint stratum; overlapping pairs allowed')
    write(out/'selection_manifest.json',selection)
    settings=config.get('source_hierarchy_verification',{})
    batch_size=args.batch_size if args.batch_size is not None else settings.get('batch_size',8)
    require(type(batch_size) is int and 1<=batch_size<=8,'batch_size must be 1..8')
    device_name=args.device or settings.get('device','cpu')
    require(device_name in ['cpu','cuda'] and (device_name=='cpu' or torch.cuda.is_available()),'Requested device unavailable')
    seed=config.get('seed',42);require(type(seed) is int and 0<=seed<2**32,'Invalid seed')
    p=config['hdeg']['dbrl'];require(p.get('input_dim',6)==6,'DBRL input_dim must be 6')
    kwargs={k:p.get(k,v) for k,v in dict(hidden_dim=64,embedding_dim=64,gru_layers=1,dropout=0.,
        graph_top_k=15,graph_self_loops=False,graph_symmetric=False,graph_heads=1,graph_dropout=0.).items()}
    kwargs['input_dim']=6
    bse_heads=config['hdeg'].get('bse',{}).get('num_heads',1)
    bp=config['hdeg'].get('bil',{});require(bp.get('num_heads',1)==1,'BIL supports one head')
    bil_kwargs=dict(negative_slope=bp.get('negative_slope',.2),dropout=bp.get('dropout',0.))
    hbf_hidden=config['hdeg']['hbf']['dynamics_hidden_dim']
    require(kwargs['embedding_dim']==64 and 1<=kwargs['hidden_dim']<=256 and
            1<=kwargs['gru_layers']<=4 and 1<=kwargs['graph_heads']<=8 and
            1<=bse_heads<=8 and 1<=hbf_hidden<=512,'Model dimensions exceed compatibility budget')
    torch.set_num_threads(1);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends,'cudnn'):
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    d=kwargs['embedding_dim'];device=torch.device(device_name)
    # Initialize on CPU, then transfer, so device transfer does not change initialization order.
    model=torch.nn.ModuleDict(dict(dbrl=DBRL(**kwargs),bse=BehavioralStateEstimator(9,d,num_heads=bse_heads),
        bil=BehavioralInteractionLearner(d,num_states=9,**bil_kwargs),
        ebrl=EcosystemBehavioralRepresentationLearner(d,num_states=9),
        hbf=HierarchicalBehavioralForecaster(num_devices=23,num_states=9,embedding_dim=d,dynamics_hidden_dim=hbf_hidden)))
    initialization={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save(initialization,out/'initialization.pt')
    model.to(device).eval();before=state_digest(model)
    mask=semantics.torch_mask(device=device);original_mask=mask.clone()
    require(torch.equal(model['bil'].edge_index,torch.tensor(EDGES,device=device).t().contiguous()),'BIL edges differ')
    report.update(source_hashes=hashes,manifest_sha256=ds.manifest_sha256,metadata_hashes=ds.metadata_hashes,
         initialization_state_sha256=before,initialization_file_sha256=windows.sha256(out/'initialization.pt'),
         device=device_name,seed=seed,batch_size=batch_size,framework_versions=dict(torch=torch.__version__,torch_geometric=torch_geometric.__version__,numpy=np.__version__),
         model_parameters=dict(dbrl=kwargs,bse=dict(num_heads=bse_heads),bil=bil_kwargs,hbf=dict(dynamics_hidden_dim=hbf_hidden)),
         state_ids=list(STATE_IDS),compatibility_mask=mask.cpu().tolist(),big_edges=list(EDGES),batches=[])
    def encode(x):
        z=model['dbrl'](x);s=model['bse'](z,mask);st=model['bil'](s)
        return dict(Z=z,S=s,S_tilde=st,g=model['ebrl'](st))
    arrays={f'{branch}_{level}':[] for branch in ['X','Y','forecast'] for level in LEVELS}
    with torch.inference_mode():
        for start in range(0,32,batch_size):
            indices=[r['index'] for r in selected[start:start+batch_size]]
            batch=ds.get_batch(indices);x=torch.from_numpy(batch['X']).to(device);y=torch.from_numpy(batch['Y']).to(device)
            current=encode(x);pred=model['hbf'](*(current[k] for k in LEVELS));target=encode(y)
            repeat=encode(x);repeat_pred=model['hbf'](*(repeat[k] for k in LEVELS))
            entry=dict(indices=indices,levels={})
            for level,shape in zip(LEVELS,[(len(indices),23,d),(len(indices),9,d),(len(indices),9,d),(len(indices),d)]):
                for branch in [current,pred,target,repeat,repeat_pred]:
                    require(tuple(branch[level].shape)==shape and torch.isfinite(branch[level]).all().item(),'Shape/finite check failed: '+level)
                    require(not branch[level].requires_grad,'Autograd unexpectedly enabled')
                errors={}
                for name,a,b in [('representation',current[level],repeat[level]),('forecast',pred[level],repeat_pred[level])]:
                    errors[name]=float((a-b).abs().max())
                    require(torch.allclose(a,b,atol=1e-7,rtol=1e-5),'Evaluation repeatability failed: '+level)
                entry['levels'][level]=dict(shape=list(shape),repeat_max_abs_error=errors)
                for name,branch in [('X',current),('Y',target),('forecast',pred)]:arrays[name+'_'+level].append(branch[level].cpu().numpy().copy())
            report['batches'].append(entry);print('Checked pairs:',start+len(indices),'/32',flush=True)
    arrays={k:np.concatenate(v) for k,v in arrays.items()}
    np.savez_compressed(out/'forward_outputs.npz',**arrays)
    metrics={}
    for stratum in STRATA:
        take=np.array([r['stratum']==stratum for r in selected]);metrics[stratum]={}
        for level in LEVELS:
            x,y,p=[arrays[b+'_'+level][take] for b in ['X','Y','forecast']]
            metrics[stratum][level]=dict(current=variation(x),next=variation(y),forecast=variation(p),
                  mean_squared_representation_increment=float(np.square(y.astype(float)-x.astype(float)).mean()))
    require(state_digest(model)==before,'Parameters or buffers changed')
    require(torch.equal(mask,original_mask),'Semantic mask changed')
    require(all(p.grad is None for p in model.parameters()),'Unexpected parameter gradients')
    ds.verify_unchanged();require(all(windows.sha256(p)==hashes[k] for k,p in deps.items()),'Dependency changed')
    report.update(status='PASS',variation_by_stratum=metrics,parameters_and_buffers_unchanged=True,
        semantic_mask_unchanged=True,source_and_artifact_recheck_passed=True,
        encoder_forward_calls=3*len(report['batches']),hbf_forward_calls=2*len(report['batches']),
        output_hashes={p.name:windows.sha256(p) for p in out.iterdir() if p.is_file()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project_root_dir')
    p.add_argument('--input_dir',required=True,type=Path);p.add_argument('--window_report',required=True,type=Path)
    p.add_argument('--semantic_config',type=Path);p.add_argument('--device',choices=['cpu','cuda']);p.add_argument('--batch_size',type=int)
    args=p.parse_args()
    with open('configs/config.yaml') as f:config=yaml.safe_load(f)
    if args.project_root_dir:config['project_root_dir']=args.project_root_dir
    from src.utils.get_folders_utils import get_processed_folder
    out=Path(get_processed_folder(config))/'v2_validation'/'source_hierarchy_compatibility'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status='BLOCKED',protocol='source_hierarchy_forward_v1',configuration=config,
       policy_version='cu_source_input_state_v1',optimizer_steps=0,backward_calls=0,checkpoint_loaded=False,
       initialization='fresh seeded CPU construction; saved state dictionary',model_mode='eval_inference_mode',
       repeat_tolerance=dict(atol=1e-7,rtol=1e-5),scientific_claim_decision='NOT_EVALUATED',
       limitations=['Compatibility only: no behavior-discrimination or forecasting-skill conclusion.',
          'All-missing windows are diagnostic controls, not approved optimization samples.',
          'Variation describes eight selected pairs per stratum, not independent episodes or population estimates.',
          'Status transitions may coincide with value transitions; raw physical behavior is not isolated.',
          'Hard top-k ties and device/runtime differences may affect results; initialization state is saved.',
          'Zero representation variation is reported, not treated as an automatic compatibility failure.',
          'Existing semantic assignments are checked structurally, not scientifically validated.'])
    try:run(args,config,out,report)
    except Exception as exc:report.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    write(out/'source_hierarchy_compatibility_report.json',report)
    print('Source hierarchy compatibility:',report['status']);print('Report:',out/'source_hierarchy_compatibility_report.json')
    if 'error' in report:print(report['error'])
    return 0 if report['status']=='PASS' else 2


if __name__=='__main__':raise SystemExit(main())
