"""Two fixed train-only optimization arms on the frozen transition selection."""
from __future__ import annotations
import argparse
import inspect
import hashlib
import copy
from collections import Counter
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch_geometric
import yaml
from src.data_preparation import input_state_windows as windows
from src.common.graph.semantics import load_behavioral_state_config
from src.models.hdeg.dbrl import DBRL
from src.models.hdeg.bse import BehavioralStateEstimator
from src.models.hdeg.bil import BehavioralInteractionLearner
from src.models.hdeg.ebrl import EcosystemBehavioralRepresentationLearner
from src.models.hdeg.hbf import HierarchicalBehavioralForecaster
from src.models.hdeg.mo import ModelOptimization
from src.utils.get_folders_utils import get_processed_folder

require = windows.require
LEVELS = ('Z', 'S', 'S_tilde', 'g')
GROUPS = ('dbrl', 'bse', 'bil', 'ebrl')
STATE_IDS = ('access_entry', 'occupancy_presence', 'physical_disturbance',
             'thermal_environment', 'switch_controlled_appliance_activity',
             'lighting_activity', 'air_quality', 'water_heating', 'humidification')
EDGES = ((0, 1), (1, 4), (1, 5), (1, 7), (4, 3), (7, 3), (8, 6))



def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def state_hash(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().contiguous().numpy()
        h.update(json.dumps([name,str(a.dtype),list(a.shape)]).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def errors(x,y,f):
    x,y,f=(a.astype(np.float64) for a in (x,y,f))
    return ((x-y)**2).reshape(len(x),-1).mean(1), ((f-y)**2).reshape(len(y),-1).mean(1)


def summary(x,y,f):
    p,e=errors(x,y,f)
    result=dict(n=len(x),persistence_mean=float(p.mean()),forecast_mean=float(e.mean()),
        persistence_median=float(np.median(p)),forecast_median=float(np.median(e)),
        persistence_max=float(p.max()),forecast_max=float(e.max()),
        mean_paired_difference=float((e-p).mean()),wins=int((e<p).sum()),
        ties=int((e==p).sum()),losses=int((e>p).sum()),zero_persistence=int((p==0).sum()),
        skill= float(1-e.sum()/p.sum()) if p.sum()>0 else None,
        skill_denominator=float(p.sum()))
    for side,a in [('X',x),('Y',y)]:
        a=a.astype(np.float64).reshape(len(a),-1)
        v=((a-a.mean(0))**2).mean(0)
        result[side+'_variation']=float(v.mean())
        result[side+'_rms']=float(np.sqrt((a*a).mean()))
        result[side+'_constant_coordinate_fraction']=float((a==a[:1]).all(0).mean())
    v=result['Y_variation']
    result['persistence_to_Y_variation']=float(p.mean()/v) if v>0 else None
    return result


STRATA=('quiet','older_value_only','older_status_only','older_mixed',
        'new_value_only','new_status_only','new_mixed')
EVALUATIONS=(0,16,32,64,128)


def split_pairs(rows,digest):
    """Input-only assignment; full SHA256 priority, index breaks ties."""
    assignments=[];batches=[]
    for portion in range(8):
        groups={}
        for stratum in STRATA:
            candidates=[r['index'] for r in rows if r['portion']==portion and r['stratum']==stratum]
            quota=8 if stratum=='quiet' else 4
            require(len(candidates)==quota,'Invalid split quota')
            def key(i):
                return (int(hashlib.sha256(f'hdeg_transition_optimization_split_v1|{digest}|42|{i}'.encode('ascii')).hexdigest(),16),i)
            ranked=sorted(candidates,key=key);half=quota//2
            groups[stratum]=ranked[:half]
            for rank,i in enumerate(ranked):
                assignments.append(dict(index=i,portion=portion,stratum=stratum,
                    role='update' if rank<half else 'probe',priority_hex=f'{key(i)[0]:064x}'))
        for b in range(2):
            batches.append(groups['quiet'][2*b:2*b+2]+[groups[st][b] for st in STRATA[1:]])
    update=[i for batch in batches for i in batch]
    require(len(update)==len(set(update))==128,'Update schedule is not unique')
    require(len(assignments)==256 and len({r['index'] for r in assignments})==256,'Split indices not unique')
    return dict(assignments=sorted(assignments,key=lambda r:r['index']),batches=batches,
        selection_manifest_sha256=digest,cycles=8,updates_per_arm=128,
        rule='full SHA256 unsigned integer; tie index; first half update within portion/stratum')


def append_json(path,obj):
    with path.open('a') as f:
        f.write(json.dumps(obj,allow_nan=False)+'\n');f.flush()


def safe_ratio(a,b):
    return float(a/b) if b>0 else None


def drift_summary(data,initial,idx,level):
    entry=summary(*(data[side][level][idx] for side in ('X','Y','prediction')))
    base=summary(*(initial[side][level][idx] for side in ('X','Y','prediction')))
    entry['initial_persistence_mean']=base['persistence_mean']
    entry['initial_forecast_mean']=base['forecast_mean']
    for side in ('X','Y'):
        a=data[side][level][idx].astype(np.float64)
        b=initial[side][level][idx].astype(np.float64)
        drift=float(((a-b)**2).mean())
        entry[side+'_drift_mse']=drift
        entry[side+'_variation_retention']=safe_ratio(entry[side+'_variation'],base[side+'_variation'])
        entry[side+'_rms_retention']=safe_ratio(entry[side+'_rms'],base[side+'_rms'])
        entry[side+'_drift_to_initial_variation']=safe_ratio(drift,base[side+'_variation'])
    return entry


def parameter_updates(model,previous,initial):
    groups={}
    for name,p in model.named_parameters():
        part=name.split('.')
        group=part[0]
        if group=='hbf':group+='.'+(part[1] if part[1].endswith('prediction_head') else 'shared_backbone')
        now=p.detach().double();old=previous[name].double();origin=initial[name].double()
        g=groups.setdefault(group,dict(update_sq=0.,displacement_sq=0.,changed_entries=0,entries=0))
        g['update_sq']+=float(((now-old)**2).sum())
        g['displacement_sq']+=float(((now-origin)**2).sum())
        g['changed_entries']+=int((now!=old).sum());g['entries']+=p.numel()
    for g in groups.values():
        g['update_norm']=g.pop('update_sq')**.5;g['displacement_norm']=g.pop('displacement_sq')**.5
        g['changed_fraction']=g['changed_entries']/g['entries']
    return groups


def execute_arms(initial_model,encode_fn,load_batch,rows,split,initial,integrity,out,report):
    """Fixed engine. Injectable data/model boundaries support independent synthetic tests."""
    indices=[r['index'] for r in rows]
    roles={r['index']:r['role'] for r in split['assignments']}
    positions={(role,st):[j for j,r in enumerate(rows) if roles[r['index']]==role and r['stratum']==st]
               for role in ('update','probe') for st in STRATA}
    initial_hash=state_hash(initial_model)
    report['arms']={}
    for arm in ('A_hbf_only','B_joint'):
        folder=out/arm;folder.mkdir()
        model=copy.deepcopy(initial_model)
        require(state_hash(model)==initial_hash,'Arm initialization differs')
        for name,p in model.named_parameters():p.requires_grad_(arm=='B_joint' or name.startswith('hbf.'))
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4,
            betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False)
        origin={n:p.detach().clone() for n,p in model.named_parameters()}
        frozen={n:t.detach().clone() for n,t in model.state_dict().items() if not n.startswith('hbf.')}
        result=dict(status='RUNNING',completed_updates=0,initial_state_sha256=state_hash(model),evaluations=[])
        report['arms'][arm]=result
        def check_frozen():
            if arm=='A_hbf_only':
                require(all(torch.equal(model.state_dict()[n],t) for n,t in frozen.items()),'Frozen encoder state changed')
        def evaluate(step):
            integrity();model.eval();data={s:{k:[] for k in LEVELS} for s in ('X','Y','prediction')}
            with torch.no_grad():
                for start in range(0,256,8):
                    batch=load_batch(indices[start:start+8])
                    x=encode_fn(model,batch['X']);y=encode_fn(model,batch['Y'])
                    pred=model['hbf'](*(x[k] for k in LEVELS))
                    for side,values in zip(data,(x,y,pred)):
                        for k in LEVELS:
                            require(values[k].shape==y[k].shape and torch.isfinite(values[k]).all().item(),'Invalid evaluation output')
                            data[side][k].append(values[k].cpu().numpy())
            data={s:{k:np.concatenate(v) for k,v in c.items()} for s,c in data.items()}
            if step==0 or arm=='A_hbf_only':
                for side in ('X','Y'):
                    for k in LEVELS:require(np.array_equal(data[side][k],initial[side][k]),'Fixed/initial representations changed')
            if step==0:
                for k in LEVELS:require(np.array_equal(data['prediction'][k],initial['prediction'][k]),'Initial HBF differs')
            metrics={role:{st:{k:drift_summary(data,initial,positions[role,st],k) for k in LEVELS}
                       for st in STRATA} for role in ('update','probe')}
            pairs=[]
            for j,row in enumerate(rows):
                entry=dict(row,role=roles[row['index']],levels={})
                for k in LEVELS:
                    p,f=errors(*(data[s][k][j:j+1] for s in ('X','Y','prediction')))
                    p0,_=errors(*(initial[s][k][j:j+1] for s in ('X','Y','prediction')))
                    entry['levels'][k]=dict(persistence_mse=float(p[0]),forecast_mse=float(f[0]),initial_persistence_mse=float(p0[0]))
                    for side in ('X','Y'):
                        entry['levels'][k][side+'_drift_mse']=float(((data[side][k][j].astype(float)-initial[side][k][j].astype(float))**2).mean())
                pairs.append(entry)
            write_json(folder/f'evaluation_{step:03d}.json',dict(step=step,metrics=metrics,pairs=pairs))
            if step==128:
                with (folder/'final_representations.npz').open('xb') as f:
                    np.savez_compressed(f,indices=np.array(indices),**{s+'_'+k:a for s,c in data.items() for k,a in c.items()})
            check_frozen();result['evaluations'].append(step)
            print(f'{arm}: evaluation at update {step}',flush=True)
        evaluate(0)
        for step in range(1,129):
            report['active_arm']=arm;report['attempted_update']=step
            batch_indices=split['batches'][(step-1)%16]
            require(all(roles[i]=='update' for i in batch_indices),'Probe entered updates')
            batch=load_batch(batch_indices);model.train()
            if arm=='A_hbf_only':
                for name,module in model.items():
                    if name!='hbf':module.eval()
            optimizer.zero_grad(set_to_none=True)
            previous={n:p.detach().clone() for n,p in model.named_parameters()}
            x=encode_fn(model,batch['X']);y=encode_fn(model,batch['Y'])
            if arm=='B_joint':require(all(y[k].requires_grad for k in LEVELS),'Live target detached')
            pred=model['hbf'](*(x[k] for k in LEVELS))
            for c in (x,y,pred):require(all(torch.isfinite(c[k]).all().item() for k in LEVELS),'Nonfinite training output')
            losses={k:(pred[k]-y[k]).square().mean() for k in LEVELS}
            loss=sum(losses.values());require(torch.isfinite(loss).item(),'Nonfinite loss')
            loss.backward();report['backward_calls']+=1
            for name,p in model.named_parameters():
                if p.requires_grad:require(p.grad is not None and torch.isfinite(p.grad).all().item(),'Invalid gradient: '+name)
                else:require(p.grad is None,'Frozen parameter received gradient')
            optimizer.step();report['optimizer_steps']+=1;result['completed_updates']=step
            require(all(torch.isfinite(p).all().item() for p in model.state_dict().values()),'Nonfinite model state')
            check_frozen()
            append_json(folder/'steps.jsonl',dict(step=step,indices=batch_indices,
                loss=float(loss.detach()),level_losses={k:float(v.detach()) for k,v in losses.items()},
                parameter_updates=parameter_updates(model,previous,origin)))
            if step in EVALUATIONS:evaluate(step)
        integrity();check_frozen()
        with (folder/'final_model.pt').open('xb') as f:
            torch.save(dict(model_state_dict=model.state_dict(),arm=arm,updates=128,
                initial_state_sha256=initial_hash,split_manifest_sha256=windows.sha256(out/'split_manifest.json')),f)
        result.update(status='COMPLETE',final_state_sha256=state_hash(model),
            checkpoint_sha256=windows.sha256(folder/'final_model.pt'),
            frozen_encoder_unchanged=True if arm=='A_hbf_only' else None)
    require(state_hash(initial_model)==initial_hash,'Initial model was mutated')
    report['status']='COMPLETE'


def run(config,processed,args,report,out):
    settings=config.get('transition_optimization',{})
    require(set(settings)<= {'batch_size','device','updates','learning_rate'},'Unknown transition_optimization setting')
    require(type(settings.get('batch_size',8)) is int and settings.get('batch_size',8)==8,'Batch size must be eight')
    require(settings.get('updates',128)==128 and type(settings.get('updates',128)) is int,'Updates must be 128')
    require(settings.get('learning_rate',1e-4)==1e-4,'Learning rate must be 1e-4')
    device_name=args.device or settings.get('device','cpu')
    require(device_name=='cpu','Protocol requires CPU')
    proof_path=Path(args.pilot_dir)/'pilot_report.json'
    proof=json.loads(proof_path.read_text())
    require(proof['status']=='COMPLETED' and proof['protocol']=='v2_input_state_optimization_pilot_v1'
        and proof['optimizer_steps']==100 and proof['device']=='cpu','Expected completed CPU pilot')
    dataset=windows.InputStateWindowDataset(args.input_dir,30)
    require(config['preprocessing']['window_size']==30,'Window size must be 30')
    require(dataset.manifest_sha256==proof['manifest_sha256'],'Prepared manifest differs')
    require(config['preprocessing']==proof['configuration']['preprocessing']==dataset.preparation['configuration']['preprocessing'],'Preprocessing differs')
    indices=proof['selected_window_indices']
    require(len(indices)==32 and indices==sorted(set(indices)) and all(type(i) is int and 0<=i<len(dataset) for i in indices),'Invalid anchor indices')
    seed=config.get('seed',42)
    require(type(seed) is int and seed==proof['seed']==42,'Seed must be 42')
    semantic_path = Path(config['project_root_dir']) / 'configs/hdeg/behavioral_states.yaml'
    dependencies = {'config': Path('configs/config.yaml'), 'pilot_report': proof_path,
                    'runner': Path(__file__), 'window_loader': Path(windows.__file__),
                    'devices': processed / 'devices.json', 'timestamps': processed / 'timestamps.npz',
                    'selected_features': Path('configs/CU_selected_features.txt'),
                    'registry': Path(config['preprocessing'].get('input_state_registry', 'configs/CU_input_state_registry.json')),
                    'semantic_config': semantic_path, 'semantic_loader': Path(inspect.getfile(load_behavioral_state_config))}
    for key, cls in zip(GROUPS, (DBRL, BehavioralStateEstimator, BehavioralInteractionLearner,
                                EcosystemBehavioralRepresentationLearner)):
        dependencies[key] = Path(inspect.getfile(cls))
    from src.common.graph.graph_builder import GraphBuilder
    from src.models.hdeg.graph_reasoner import GraphReasoner
    dependencies['graph_builder'] = Path(inspect.getfile(GraphBuilder))
    dependencies['graph_reasoner'] = Path(inspect.getfile(GraphReasoner))
    dependencies['hbf'] = Path(inspect.getfile(HierarchicalBehavioralForecaster))
    dependencies['mo'] = Path(inspect.getfile(ModelOptimization))
    hashes = {k: windows.sha256(p) for k, p in dependencies.items()}
    for key in ('hbf', 'mo', 'dbrl', 'bse', 'bil', 'ebrl', 'semantic_config', 'semantic_loader', 'window_loader', 'graph_builder', 'graph_reasoner', 'devices', 'timestamps', 'selected_features', 'registry'):
        require(hashes[key] == proof['source_hashes'][key], f'Dependency changed since pilot: {key}')
    for key in ('devices', 'timestamps', 'selected_features', 'registry'):
        require(hashes[key] == dataset.metadata['source_hashes'][key], f'Prepared dependency differs: {key}')
    semantics = load_behavioral_state_config(semantic_path, dependencies['devices'])
    require(list(semantics.device_ids()) == dataset.devices, 'Semantic device order differs')
    require(semantics.state_ids() == STATE_IDS, 'State order does not match the frozen nine-state BIG')
    p = config['hdeg']['dbrl']
    kwargs = {k: p.get(k, default) for k, default in dict(hidden_dim=64, embedding_dim=64,
        gru_layers=1, dropout=0.0, graph_top_k=15, graph_self_loops=False, graph_symmetric=False,
        graph_heads=1, graph_dropout=0.0).items()}
    kwargs['input_dim'] = 6
    require(kwargs == proof['model_parameters']['dbrl'] and p.get('input_dim', 6) == 6, 'DBRL settings differ')
    bse_heads = config['hdeg'].get('bse', {}).get('num_heads', 1)
    require(type(bse_heads) is int and bse_heads > 0, 'BSE num_heads must be positive integer')
    # Current BIL implements a single frozen-graph attention mechanism, not multihead attention.
    require(config['hdeg'].get('bil', {}).get('num_heads', 1) == 1, 'Current BIL supports one head')
    bil_settings = config['hdeg'].get('bil', {})
    bil_kwargs = dict(negative_slope=bil_settings.get('negative_slope', 0.2),
                      dropout=bil_settings.get('dropout', 0.0))
    require({'num_heads': bse_heads} == proof['model_parameters']['bse'] and bil_kwargs == proof['model_parameters']['bil'],
            'BSE/BIL settings differ from pilot')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    d = kwargs['embedding_dim']
    model = torch.nn.ModuleDict({
        'dbrl': DBRL(**kwargs),
        'bse': BehavioralStateEstimator(9, d, num_heads=bse_heads),
        'bil': BehavioralInteractionLearner(d, num_states=9, **bil_kwargs),
        'ebrl': EcosystemBehavioralRepresentationLearner(d, num_states=9)
    }).to(device).eval()
    mask = semantics.torch_mask(device=device)
    expected_edges = torch.tensor(EDGES, device=device).t().contiguous()
    require(torch.equal(model['bil'].edge_index, expected_edges), 'BIL edge topology differs')
    hbf_hidden = config['hdeg']['hbf']['dynamics_hidden_dim']
    model['hbf'] = HierarchicalBehavioralForecaster(num_devices=23, num_states=9,
                    embedding_dim=d, dynamics_hidden_dim=hbf_hidden).to(device)
    mo_cfg = config['hdeg'].get('mo', {})
    weights = {key: mo_cfg.get(key, 1.0) for key in ('lambda_z', 'lambda_s', 'lambda_s_tilde', 'lambda_g')}
    require(all(type(v) in (int,float) and v == 1.0 for v in weights.values()), 'All four loss weights must be one')
    require({'dynamics_hidden_dim':hbf_hidden} == proof['model_parameters']['hbf'] and weights == proof['model_parameters']['mo'], 'HBF/objective configuration differs')
    require(kwargs['dropout'] == kwargs['graph_dropout'] == bil_kwargs['dropout'] == 0, 'Protocol requires zero dropout')
    model['mo'] = ModelOptimization(**weights).to(device)
    model.eval()
    def encode(x):
        z = model['dbrl'](x)
        s = model['bse'](z, mask)
        st = model['bil'](s)
        return dict(Z=z, S=s, S_tilde=st, g=model['ebrl'](st))
    dataset.verify_unchanged()
    report.update(source_hashes=hashes, manifest_sha256=dataset.manifest_sha256,
                  input_dir=str(dataset.directory.resolve()), seed=seed,device=device_name,
                  model_parameters=proof['model_parameters'], selected_window_indices=indices,
                  framework_versions={'torch':torch.__version__,'torch_geometric':torch_geometric.__version__,'numpy':np.__version__})

    census_path=Path(args.census_dir)/'transition_coverage_report.json'
    selection_path=Path(args.census_dir)/'selection_manifest.json'
    census=json.loads(census_path.read_text());selection=json.loads(selection_path.read_text())
    require(census['status']=='COMPLETE' and census['protocol']=='v2_transition_coverage_census_v1','Expected completed census')
    require(census['manifest_sha256']==selection['manifest_sha256']==dataset.manifest_sha256,'Census manifest differs')
    require(selection['protocol']=='v2_transition_coverage_selection_v1' and selection['seed']==42 and selection['window_size']==30 and selection['target_shift']==1,'Selection protocol differs')
    require(selection['pilot_report_sha256']==census['source_hashes']['pilot_report']==hashes['pilot_report'],'Pilot provenance differs')
    for key in ('devices','timestamps','selected_features','registry','window_loader'):
        require(census['source_hashes'][key]==hashes[key],'Census dependency differs: '+key)
    dependencies.update(census_report=census_path,selection_manifest=selection_path)
    for name,digest in census['output_hashes'].items():
        require(Path(name).name==name,'Invalid output filename')
        path=Path(args.census_dir)/name
        require(windows.sha256(path)==digest,'Census output hash differs: '+name)
        dependencies['census_'+name]=path
    for name in ('initialization_representations.npz','evaluations.jsonl','protocol.json'):
        path=Path(args.pilot_dir)/name
        require(windows.sha256(path)==proof['result_files'][name],'Pilot artifact differs: '+name)
        dependencies[name]=path
    require(proof['initialization_representations_sha256']==proof['result_files']['initialization_representations.npz'],'Initialization digest differs')
    hashes={k:windows.sha256(p) for k,p in dependencies.items()}
    rows=selection['rows'];selected=[r['index'] for r in rows]
    strata=('quiet','older_value_only','older_status_only','older_mixed','new_value_only','new_status_only','new_mixed')
    require(len(selected)==256 and selected==sorted(set(selected)) and all(type(i) is int and 0<=i<len(dataset) for i in selected),'Invalid selected indices')
    require(all(b-a>=31 for a,b in zip(selected,selected[1:])),'Source spans overlap')
    bounds=[j*len(dataset)//8 for j in range(9)]
    require(selection['chronological_bounds']==bounds,'Chronological bounds differ')
    for part in range(8):
        for st in strata:
            group=[r for r in rows if r['portion']==part and r['stratum']==st]
            require(len(group)==(8 if st=='quiet' else 4),'Quota mismatch')
            require(all(bounds[part]<=r['index']<bounds[part+1] for r in group),'Portion mismatch')
    diagnostic_dir=Path(args.diagnostic_dir)
    diagnostic_path=diagnostic_dir/'diagnostic_report.json'
    diagnostic=json.loads(diagnostic_path.read_text())
    require(diagnostic['status']=='PASS' and diagnostic['protocol']=='v2_transition_diagnostic_v1','Successful diagnostic required')
    require(diagnostic['manifest_sha256']==dataset.manifest_sha256 and diagnostic['selected_window_indices']==selected,'Diagnostic inputs differ')
    for k in hashes:
        if k in diagnostic['source_hashes'] and k not in ('config','runner'):
            require(hashes[k]==diagnostic['source_hashes'][k],'Diagnostic provenance differs: '+k)
    require(diagnostic['source_hashes']['selection_manifest']==windows.sha256(selection_path),'Diagnostic selection differs')
    dependencies['diagnostic_report']=diagnostic_path
    for name,digest in {**diagnostic['output_hashes'],'protocol.json':diagnostic['protocol_sha256']}.items():
        require(Path(name).name==name,'Invalid diagnostic filename')
        path=diagnostic_dir/name
        require(windows.sha256(path)==digest,'Diagnostic artifact differs: '+name)
        dependencies['diagnostic_'+name]=path
    hashes={k:windows.sha256(p) for k,p in dependencies.items()}
    split=split_pairs(rows,windows.sha256(selection_path))
    write_json(out/'split_manifest.json',split)
    report.update(source_hashes=hashes,manifest_sha256=dataset.manifest_sha256,
        protocol='v2_transition_optimization_v1',configuration=config,
        split_manifest_sha256=windows.sha256(out/'split_manifest.json'),
        reference_framework_versions=diagnostic['framework_versions'])
    write_json(out/'protocol.json',dict(protocol=report['protocol'],configuration=config,source_hashes=hashes,
        model_parameters=proof['model_parameters'],split_manifest_sha256=report['split_manifest_sha256'],
        seed=42,device='cpu',batch_size=8,updates_per_arm=128,evaluations=list(EVALUATIONS),
        optimizer=dict(name='Adam',lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0,amsgrad=False,foreach=False),
        arms=dict(A_hbf_only='Frozen encoder',B_joint='Live shared encoder; no detach'),
        loss='Sum of four coordinate-mean batch MSEs; equal level weights',
        gate=dict(rtol=1e-5,atol=1e-7),limitations=report['limitations']))
    def integrity():
        dataset.verify_unchanged()
        require(all(windows.sha256(p)==hashes[k] for k,p in dependencies.items()),'Dependency changed')
    def load_batch(which):
        batch=dataset.get_batch(which)
        require(np.array_equal(batch['X'][:,1:],batch['Y'][:,:-1]),'Pair overlap differs')
        require(np.all(batch['Y_timestamps']-batch['X_timestamps']==np.timedelta64(1,'s')),'Target shift differs')
        return {side:torch.from_numpy(batch[side]) for side in ('X','Y')}
    def encode_model(m,x):
        z=m['dbrl'](x);s=m['bse'](z,mask);st=m['bil'](s)
        return dict(Z=z,S=s,S_tilde=st,g=m['ebrl'](st))
    initial={s:{k:[] for k in LEVELS} for s in ('X','Y','prediction')}
    state_before=state_hash(model)
    with torch.no_grad():
        for start in range(0,256,8):
            batch=load_batch(selected[start:start+8])
            x=encode_model(model,batch['X']);y=encode_model(model,batch['Y'])
            pred=model['hbf'](*(x[k] for k in LEVELS))
            for side,c in zip(initial,(x,y,pred)):
                for k in LEVELS:
                    require(torch.isfinite(c[k]).all().item(),'Nonfinite initialization')
                    initial[side][k].append(c[k].numpy())
    require(state_hash(model)==state_before,'Initialization evaluation changed model')
    initial={s:{k:np.concatenate(v) for k,v in c.items()} for s,c in initial.items()}
    checks={};report['step_zero_checks']=checks
    def compare(name,a,b):
        require(a.shape==b.shape and np.isfinite(b).all(),'Invalid reference '+name)
        checks[name]=dict(passed=bool(np.allclose(a,b,rtol=1e-5,atol=1e-7)),
            maximum_absolute_difference=float(np.max(np.abs(a.astype(float)-b.astype(float)))))
        require(checks[name]['passed'],'Step-zero mismatch: '+name)
    with np.load(diagnostic_dir/'representations.npz',allow_pickle=False) as ref:
        require(ref['indices'].tolist()==selected,'Reference index order differs')
        for side in initial:
            for k in LEVELS:compare(side+'_'+k,initial[side][k],ref[side+'_'+k])
    metrics=json.loads((diagnostic_dir/'pair_metrics.json').read_text())
    require([r['index'] for r in metrics]==selected,'Pair metric order differs')
    for row,expected in zip(rows,metrics):
        require(all(expected[k]==v for k,v in row.items()),'Reference pair annotation differs')
    for k in LEVELS:
        p,f=errors(*(initial[s][k] for s in ('X','Y','prediction')))
        compare('persistence_'+k,p,np.array([r['levels'][k]['persistence_mse'] for r in metrics]))
        compare('forecast_'+k,f,np.array([r['levels'][k]['forecast_mse'] for r in metrics]))
    report['initial_state_sha256']=state_before
    integrity()
    with (out/'initialization_representations.npz').open('xb') as f:
        np.savez_compressed(f,indices=np.array(selected),**{s+'_'+k:a for s,c in initial.items() for k,a in c.items()})
    with (out/'initial_model.pt').open('xb') as f:torch.save(model.state_dict(),f)
    execute_arms(model,encode_model,load_batch,metrics,split,initial,integrity,out,report)
    integrity()
    report['source_and_artifact_recheck_passed']=True
    report['output_hashes']={str(p.relative_to(out)):windows.sha256(p) for p in out.rglob('*') if p.is_file()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root_dir')
    for name in ('input_dir','pilot_dir','census_dir','diagnostic_dir'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',choices=['cpu'])
    args=parser.parse_args()
    with open('configs/config.yaml') as f:config=yaml.safe_load(f)
    if args.project_root_dir:config['project_root_dir']=args.project_root_dir
    processed=Path(get_processed_folder(config))
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=processed/'v2_validation'/'transition_optimization'/run_id
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status='BLOCKED',optimizer_steps=0,backward_calls=0,
        limitations=['Enriched training-period development probes; not untouched validation or generalization.',
        'Joint targets move; raw arm losses are not equivalent forecasting quality measures.',
        'Nonoverlapping spans do not establish independence; no inferential tests.',
        'Fixed budget does not establish convergence or impossibility of learning.'])
    try:run(config,processed,args,report,out)
    except (ValueError,KeyError,TypeError,OSError,IndexError,RuntimeError) as exc:
        report.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    write_json(out/'comparison_report.json',report)
    print(f'Transition optimization: {report["status"]}\nReport: {out/"comparison_report.json"}')
    if 'error' in report:print(report['error'])
    return 0 if report['status']=='COMPLETE' else 2


if __name__=='__main__':raise SystemExit(main())
