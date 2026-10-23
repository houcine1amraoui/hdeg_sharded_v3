"""Frozen-encoder persistence-initialized output-residual HBF experiment."""
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


HEADS=('device_prediction_head','state_prediction_head','context_prediction_head','global_prediction_head')
VARIANT='output_residual_hbf_zero_heads_frozen_encoder_v1'


def group_name(name):
    return name.split('.')[1] if name.startswith('hbf.') and name.split('.')[1] in HEADS else 'shared_backbone'


def initialize_corrections(model):
    """Only the experiment's in-memory copy is changed."""
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith('hbf.'))
        p.grad=None
    with torch.no_grad():
        for name in HEADS:
            head=getattr(model['hbf'],name)
            head.weight.zero_();head.bias.zero_()
    require(all(torch.count_nonzero(getattr(model['hbf'],h).weight)==0 and
                torch.count_nonzero(getattr(model['hbf'],h).bias)==0 for h in HEADS),'Heads not zero')
    model.eval()


def predict(model,x):
    correction=model['hbf'](*(x[k] for k in LEVELS))
    return {k:x[k]+correction[k] for k in LEVELS},correction


def correction_metrics(x,y,pred,raw_correction):
    """Use effective float64 prediction-X correction for the exact error identity."""
    x,y,pred,raw=(a.astype(np.float64) for a in (x,y,pred,raw_correction))
    c=pred-x;d=y-x
    energy=float((c*c).mean());alignment=float((c*d).mean())
    difference=float(((pred-y)**2).mean()-(d*d).mean())
    require(np.isclose(difference,energy-2*alignment,rtol=1e-10,atol=1e-16),'Correction identity failed')
    return dict(correction_rms=energy**.5,increment_rms=float(np.sqrt((d*d).mean())),
        mean_correction=c.mean(0).reshape(-1).tolist(),
        mean_correction_rms=float(np.sqrt((c.mean(0)**2).mean())),
        correction_energy=energy,twice_increment_alignment=2*alignment,
        forecast_minus_persistence=difference,raw_head_correction_rms=float(np.sqrt((raw*raw).mean())),
        addition_roundoff_rms=float(np.sqrt(((raw-c)**2).mean())))


def execute_residual(model,encode,load_batch,rows,split,initial,arm_a,integrity,out,report):
    indices=[r['index'] for r in rows];roles={r['index']:r['role'] for r in split['assignments']}
    update=np.array([roles[i]=='update' for i in indices])
    require(update.sum()==128,'Expected 128 update pairs')
    target_means={k:initial['Y'][k][update].astype(float).mean(0) for k in LEVELS}
    with (out/'update_target_means.npz').open('xb') as f:np.savez_compressed(f,**target_means)
    frozen={n:t.detach().clone() for n,t in model.state_dict().items() if not n.startswith('hbf.')}
    origin={n:p.detach().clone() for n,p in model.named_parameters()}
    optimizer=torch.optim.Adam(model['hbf'].parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,
        weight_decay=0,amsgrad=False,foreach=False)
    positions={(role,st):[j for j,r in enumerate(rows) if roles[r['index']]==role and r['stratum']==st]
        for role in ('update','probe') for st in STRATA}
    def check_frozen():
        require(all(torch.equal(model.state_dict()[n],v) for n,v in frozen.items()),'Frozen state changed')
        require(all(p.grad is None for n,p in model.named_parameters() if not n.startswith('hbf.')),'Encoder accumulated gradients')
    def evaluate(step):
        integrity();model.eval()
        values={s:{k:[] for k in LEVELS} for s in ('X','Y','prediction','correction')}
        with torch.no_grad():
            for start in range(0,256,8):
                b=load_batch(indices[start:start+8]);x=encode(model,b['X']);y=encode(model,b['Y'])
                pred,c=predict(model,x)
                for side,collection in zip(values,(x,y,pred,c)):
                    for k in LEVELS:
                        require(collection[k].shape==y[k].shape and torch.isfinite(collection[k]).all().item(),'Invalid evaluation output')
                        values[side][k].append(collection[k].cpu().numpy())
        data={s:{k:np.concatenate(v) for k,v in c.items()} for s,c in values.items()}
        for side in ('X','Y'):
            for k in LEVELS:require(np.array_equal(data[side][k],initial[side][k]),'Encoder representations changed')
        if step==0:
            for k in LEVELS:
                require(np.count_nonzero(data['correction'][k])==0,'Initial correction nonzero')
                require(np.array_equal(data['prediction'][k],data['X'][k]),'Initial prediction is not persistence')
                p,f=errors(data['X'][k],data['Y'][k],data['prediction'][k])
                require(np.array_equal(p,f),'Initial errors differ from persistence')
            report['exact_persistence_initialization_passed']=True
        metrics={role:{} for role in ('update','probe')}
        pair_records=[dict(r,role=roles[r['index']],levels={}) for r in rows]
        for k in LEVELS:
            p,f=errors(*(data[s][k] for s in ('X','Y','prediction')))
            mean_error=((data['Y'][k].astype(float)-target_means[k])**2).reshape(256,-1).mean(1)
            a_error=np.array([r['levels'][k]['forecast_mse'] for r in arm_a[step]['pairs']])
            for j,row in enumerate(pair_records):
                row['levels'][k]=dict(persistence_mse=float(p[j]),forecast_mse=float(f[j]),
                    standard_arm_a_mse=float(a_error[j]),update_target_mean_mse=float(mean_error[j]))
            for (role,st),idx in positions.items():
                m=summary(*(data[s][k][idx] for s in ('X','Y','prediction')))
                m.update(correction_metrics(*(data[s][k][idx] for s in ('X','Y','prediction','correction'))))
                m.update(X_drift_mse=0.,Y_drift_mse=0.,encoder_representations_exactly_unchanged=True)
                for label,baseline in [('standard_arm_a',a_error),('update_target_mean',mean_error)]:
                    b=baseline[idx];e=f[idx]
                    m[label]=dict(mean_mse=float(b.mean()),mean_paired_difference=float((e-b).mean()),
                        wins=int((e<b).sum()),ties=int((e==b).sum()),losses=int((e>b).sum()))
                if st=='quiet':
                    require(np.array_equal(data['X'][k][idx],data['Y'][k][idx]),'Quiet input representation changed across time')
                    m['quiet_correction_cost']=m['forecast_mean']
                metrics[role].setdefault(st,{})[k]=m
        write_json(out/f'evaluation_{step:03d}.json',dict(step=step,metrics=metrics,pairs=pair_records))
        if step in (0,128):
            with (out/f'representations_{step:03d}.npz').open('xb') as f:
                np.savez_compressed(f,indices=np.array(indices),**{s+'_'+k:a for s,c in data.items() for k,a in c.items()})
        check_frozen();report['completed_evaluations'].append(step)
        print(f'Residual HBF: evaluation at update {step}',flush=True)
    evaluate(0)
    report['variant_initial_state_sha256']=state_hash(model)
    with (out/'initial_variant_model.pt').open('xb') as f:
        torch.save(dict(variant=VARIANT,forward_semantics='prediction=current+standard_hbf_output',model_state_dict=model.state_dict()),f)
    for step in range(1,129):
        report['attempted_update']=step
        ids=split['batches'][(step-1)%16]
        require(all(roles[i]=='update' for i in ids),'Probe entered update batch')
        b=load_batch(ids);model.eval();model['hbf'].train()
        with torch.no_grad():x=encode(model,b['X']);y=encode(model,b['Y'])
        optimizer.zero_grad(set_to_none=True)
        previous={n:p.detach().clone() for n,p in model.named_parameters()}
        pred,c=predict(model,x)
        for collection in (x,y,pred,c):require(all(torch.isfinite(collection[k]).all().item() for k in LEVELS),'Nonfinite training output')
        losses={k:(pred[k]-y[k]).square().mean() for k in LEVELS}
        loss=sum(losses.values());require(torch.isfinite(loss).item(),'Nonfinite loss')
        loss.backward();report['backward_calls']+=1
        gradients={}
        for name,p in model.named_parameters():
            if not name.startswith('hbf.'):
                require(p.grad is None,'Frozen encoder has gradients');continue
            g=gradients.setdefault(group_name(name),dict(missing_tensors=0,zero_tensors=0,nonzero_tensors=0,norm_sq=0.))
            if p.grad is None:g['missing_tensors']+=1;continue
            require(torch.isfinite(p.grad).all().item(),'Nonfinite gradient: '+name)
            g['zero_tensors' if torch.count_nonzero(p.grad)==0 else 'nonzero_tensors']+=1
            g['norm_sq']+=float(p.grad.double().square().sum())
        for g in gradients.values():g['norm']=g.pop('norm_sq')**.5
        require(all(g['missing_tensors']==0 for g in gradients.values()),'Missing HBF gradient tensor')
        if step==1:require(gradients['shared_backbone']['norm']==0,'Expected zero first-step backbone gradient')
        optimizer.step();report['optimizer_steps']+=1
        require(all(torch.isfinite(t).all().item() for t in model.state_dict().values()),'Nonfinite model state')
        check_frozen()
        append_json(out/'steps.jsonl',dict(step=step,indices=ids,loss=float(loss.detach()),
            level_losses={k:float(v.detach()) for k,v in losses.items()},gradients=gradients,
            parameter_updates=parameter_updates(model,previous,origin)))
        if step in EVALUATIONS:evaluate(step)
    integrity();check_frozen()
    with (out/'final_variant_model.pt').open('xb') as f:
        torch.save(dict(variant=VARIANT,forward_semantics='prediction=current+standard_hbf_output',
            model_state_dict=model.state_dict(),updates=128,split_manifest_sha256=report['split_manifest_sha256']),f)
    report.update(status='COMPLETE',frozen_encoder_unchanged=True,final_state_sha256=state_hash(model))


def run(config,processed,args,report,out):
    settings=config.get('frozen_residual_transition',{})
    require(set(settings)<= {'batch_size','device','updates','learning_rate'},'Unknown frozen_residual_transition setting')
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
    comparison_dir=Path(args.comparison_dir)
    comparison_path=comparison_dir/'comparison_report.json'
    comparison=json.loads(comparison_path.read_text())
    require(comparison['status']=='COMPLETE' and comparison['protocol']=='v2_transition_optimization_v1','Completed comparison required')
    require(comparison['manifest_sha256']==dataset.manifest_sha256,'Comparison manifest differs')
    for k in hashes:
        if k in comparison['source_hashes'] and k not in ('config','runner'):
            require(hashes[k]==comparison['source_hashes'][k],'Comparison provenance differs: '+k)
    require(comparison['source_hashes']['diagnostic_report']==hashes['diagnostic_report'],'Diagnostic reference differs')
    dependencies['comparison_report']=comparison_path
    for name,digest in comparison['output_hashes'].items():
        relative=Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts,'Invalid comparison artifact path')
        path=comparison_dir/relative
        require(windows.sha256(path)==digest,'Comparison artifact differs: '+name)
        dependencies['comparison_'+name]=path
    split_path=comparison_dir/'split_manifest.json'
    split=json.loads(split_path.read_text())
    require(windows.sha256(split_path)==comparison['split_manifest_sha256'],'Split hash differs')
    require(split==split_pairs(rows,windows.sha256(selection_path)),'Frozen split or schedule differs')
    # Preserve the original bytes, rather than generating a new split.
    with (out/'split_manifest.json').open('xb') as f:f.write(split_path.read_bytes())
    state=torch.load(comparison_dir/'initial_model.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    require(state_hash(model)==comparison['initial_state_sha256'],'Original checkpoint state differs')
    report['original_state_sha256']=state_hash(model)
    report['loaded_checkpoint_sha256']=windows.sha256(comparison_dir/'initial_model.pt')
    hashes={k:windows.sha256(p) for k,p in dependencies.items()}
    report.update(source_hashes=hashes,manifest_sha256=dataset.manifest_sha256,
        protocol='v2_frozen_residual_transition_v1',variant=VARIANT,configuration=config,
        split_manifest_sha256=windows.sha256(split_path),checkpoint_loaded=True,
        reference_framework_versions=comparison['framework_versions'])
    write_json(out/'protocol.json',dict(protocol=report['protocol'],variant=VARIANT,
        forward_semantics='prediction=current+standard_hbf_output; all four final correction heads zero-initialized',
        configuration=config,source_hashes=hashes,model_parameters=proof['model_parameters'],
        original_checkpoint_sha256=report['loaded_checkpoint_sha256'],
        split_manifest_sha256=report['split_manifest_sha256'],updates=128,batch_size=8,evaluations=list(EVALUATIONS),
        optimizer=dict(name='Adam',lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0,amsgrad=False,foreach=False),
        encoder='Frozen evaluation mode',loss='Sum of four coordinate-mean batch MSEs; equal weights',
        gate=dict(rtol=1e-5,atol=1e-7,exact_persistence=True),
        correction_identity='Effective prediction-X in float64; raw head correction and addition roundoff also recorded',
        first_step_backbone_gradient='Expected present and zero; nonzero gradients are not required',limitations=report['limitations']))
    def integrity():
        dataset.verify_unchanged()
        require(all(windows.sha256(p)==hashes[k] for k,p in dependencies.items()),'Source changed')
    def load_batch(which):
        b=dataset.get_batch(which)
        require(np.array_equal(b['X'][:,1:],b['Y'][:,:-1]),'Pair overlap differs')
        require(np.all(b['Y_timestamps']-b['X_timestamps']==np.timedelta64(1,'s')),'Pair shift differs')
        return {s:torch.from_numpy(b[s]) for s in ('X','Y')}
    def encode_model(m,x):
        z=m['dbrl'](x);s=m['bse'](z,mask);st=m['bil'](s)
        return dict(Z=z,S=s,S_tilde=st,g=m['ebrl'](st))
    initial={s:{k:[] for k in LEVELS} for s in ('X','Y','prediction')}
    model.eval()
    with torch.no_grad():
        for start in range(0,256,8):
            b=load_batch(selected[start:start+8]);x=encode_model(model,b['X']);y=encode_model(model,b['Y'])
            pred=model['hbf'](*(x[k] for k in LEVELS))
            for side,c in zip(initial,(x,y,pred)):
                for k in LEVELS:
                    require(torch.isfinite(c[k]).all().item(),'Nonfinite original output')
                    initial[side][k].append(c[k].numpy())
    initial={s:{k:np.concatenate(v) for k,v in c.items()} for s,c in initial.items()}
    require(state_hash(model)==comparison['initial_state_sha256'],'Original model mutated during verification')
    checks={};report['original_checkpoint_checks']=checks
    with np.load(comparison_dir/'initialization_representations.npz',allow_pickle=False) as ref:
        require(ref['indices'].tolist()==selected,'Reference indices differ')
        for side in initial:
            for k in LEVELS:
                a=initial[side][k];b=ref[side+'_'+k]
                require(a.shape==b.shape and np.isfinite(b).all(),'Invalid original reference')
                checks[side+'_'+k]=dict(passed=bool(np.allclose(a,b,rtol=1e-5,atol=1e-7)),
                    maximum_absolute_difference=float(np.abs(a.astype(float)-b.astype(float)).max()))
                require(checks[side+'_'+k]['passed'],'Original checkpoint reproduction mismatch: '+side+'_'+k)
    arm_a={t:json.loads((comparison_dir/'A_hbf_only'/f'evaluation_{t:03d}.json').read_text()) for t in EVALUATIONS}
    for t,e in arm_a.items():
        require(e['step']==t and [r['index'] for r in e['pairs']]==selected,'Arm A alignment differs')
    metrics=arm_a[0]['pairs']
    for row,record in zip(rows,metrics):require(all(record[k]==v for k,v in row.items()),'Pair metadata differs')
    original_nonheads={n:t.clone() for n,t in model.state_dict().items() if not any(n.startswith('hbf.'+h+'.') for h in HEADS)}
    initialize_corrections(model)
    require(all(torch.equal(model.state_dict()[n],t) for n,t in original_nonheads.items()),'Non-head initialization changed')
    integrity()
    execute_residual(model,encode_model,load_batch,metrics,split,initial,arm_a,integrity,out,report)
    integrity()
    report['source_and_artifact_recheck_passed']=True
    report['output_hashes']={str(p.relative_to(out)):windows.sha256(p) for p in out.rglob('*') if p.is_file()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root_dir')
    for name in ('input_dir','pilot_dir','census_dir','diagnostic_dir','comparison_dir'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',choices=['cpu'])
    args=parser.parse_args()
    with open('configs/config.yaml') as f:config=yaml.safe_load(f)
    if args.project_root_dir:config['project_root_dir']=args.project_root_dir
    processed=Path(get_processed_folder(config))
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=processed/'v2_validation'/'frozen_residual_transition'/run_id
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status='BLOCKED',optimizer_steps=0,backward_calls=0,completed_evaluations=[],
        limitations=['Enriched training-period probes, not untouched validation or population performance.',
        'Output parameterization and head initialization change together.',
        'Fixed budget does not establish convergence; no best-step selection.',
        'Use variant forward semantics when loading saved states; never load as standard absolute-output HBF.'])
    try:run(config,processed,args,report,out)
    except (ValueError,KeyError,TypeError,OSError,IndexError,RuntimeError) as exc:
        report.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    write_json(out/'residual_report.json',report)
    print(f'Frozen residual HBF: {report["status"]}\nReport: {out/"residual_report.json"}')
    if 'error' in report:print(report['error'])
    return 0 if report['status']=='COMPLETE' else 2


if __name__=='__main__':raise SystemExit(main())
