"""Bounded train-only transition diagnostic at verified original initialization."""
from __future__ import annotations
import argparse
import inspect
import hashlib
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


def run(config,processed,args,report,out):
    settings=config.get('transition_diagnostic',{})
    require(set(settings)<= {'batch_size','device'},'Unknown transition_diagnostic setting')
    require(type(settings.get('batch_size',8)) is int and settings.get('batch_size',8)==8,'Batch size must be eight')
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
    report.update(source_hashes=hashes,selected_window_indices=selected,configuration=config,
        protocol='v2_transition_diagnostic_v1',model_state='UNTRAINED_ORIGINAL_INITIALIZATION',
        batch_size=8,anchor_pairs=32,diagnostic_pairs=256,rtol=1e-5,atol=1e-7)
    report['reference_framework_versions']=proof['framework_versions']
    write_json(out/'protocol.json',dict(protocol=report['protocol'],configuration=config,
        source_hashes=hashes,manifest_sha256=dataset.manifest_sha256,seed=42,
        model_parameters=proof['model_parameters'],device='cpu',batch_size=8,
        selected_window_indices=selected,anchor_window_indices=indices,
        hierarchy_forward_budget=72,hbf_forward_budget=36,optimizer_steps=0,backward_calls=0,
        initialization_tolerance=dict(rtol=1e-5,atol=1e-7),
        reduction='float64, per-pair coordinate mean; equal weight per pair within stratum',
        denominator_policy='zero -> null; no epsilon; strict wins/ties/losses',
        variation='between-pair population variance per aligned coordinate, then coordinate mean',
        model_state='UNTRAINED_ORIGINAL_INITIALIZATION',limitations=report['limitations']))
    report['protocol_sha256']=windows.sha256(out/'protocol.json')
    before=state_hash(model);report['initial_state_sha256']=before
    def evaluate(which,annotations=None):
        collections={side:{k:[] for k in LEVELS} for side in ('X','Y','prediction')}
        with torch.inference_mode():
            for start in range(0,len(which),8):
                batch=dataset.get_batch(which[start:start+8])
                require(np.array_equal(batch['X'][:,1:],batch['Y'][:,:-1]),'Pair overlap differs')
                require(np.all(batch['Y_timestamps']-batch['X_timestamps']==np.timedelta64(1,'s')),'Pair shift differs')
                if annotations is not None:
                    for j,row in enumerate(annotations[start:start+8]):
                        v=np.concatenate((batch['X'][j,:,:,0],batch['Y'][j,-1:,:,0]))
                        status=np.concatenate((batch['X'][j,:,:,1:],batch['Y'][j,-1:,:,1:])).argmax(-1)
                        cv=(status[:-1]==0)&(status[1:]==0)&(v[:-1]!=v[1:])
                        cs=status[:-1]!=status[1:]
                        whole=int(cv.any())+2*int(cs.any());new=int(cv[-1].any())+2*int(cs[-1].any())
                        label=whole+3*(new>0)
                        require(row['stratum']==strata[label] and row['newest_composition']==('none','value_only','status_only','mixed')[new],'Pair classification differs')
                        require(row['older_changes_present']==bool(cv[:-1].any() or cs[:-1].any()),'Older flag differs')
                        require(row['valid_value_change_cells']==int(cv.sum()) and row['status_transition_cells']==int(cs.sum()),'Transition counts differ')
                        row['contains_unknown']=bool((status==1).any())
                        row['contains_unavailable']=bool((status==2).any())
                x=encode(torch.from_numpy(batch['X']));report['hierarchy_forward_calls']+=1
                y=encode(torch.from_numpy(batch['Y']));report['hierarchy_forward_calls']+=1
                pred=model['hbf'](*(x[k] for k in LEVELS));report['hbf_forward_calls']+=1
                for side,values in zip(collections,(x,y,pred)):
                    for k in LEVELS:
                        require(values[k].shape==y[k].shape and torch.isfinite(values[k]).all().item(),'Invalid output: '+k)
                        collections[side][k].append(values[k].cpu().numpy())
                print(f'Evaluated {start+8}/{len(which)} pairs',flush=True)
        return {s:{k:np.concatenate(v) for k,v in c.items()} for s,c in collections.items()}
    anchor=evaluate(indices)
    reference=np.load(Path(args.pilot_dir)/'initialization_representations.npz',allow_pickle=False)
    step0=[json.loads(line) for line in (Path(args.pilot_dir)/'evaluations.jsonl').read_text().splitlines() if json.loads(line)['step']==0]
    require(len(step0)==1,'Expected one step-zero evaluation')
    checks={}
    report['initialization_checks']=checks
    def compare(name,a,b):
        a=np.asarray(a);b=np.asarray(b)
        require(a.shape==b.shape and np.isfinite(b).all(),'Invalid anchor reference '+name)
        checks[name]=dict(mismatched_entries=int((~np.isclose(a,b,rtol=1e-5,atol=1e-7)).sum()),max_absolute_difference=float(np.max(np.abs(a.astype(float)-b.astype(float)))),passed=bool(np.allclose(a,b,rtol=1e-5,atol=1e-7)))
        require(checks[name]['passed'],'Initialization mismatch: '+name)
    for k in LEVELS:
        for side in ('X','Y'):compare(side+'_'+k,anchor[side][k],reference[side+'_'+k])
        p,f=errors(anchor['X'][k],anchor['Y'][k],anchor['prediction'][k])
        compare('forecast_'+k,f,step0[0]['levels'][k]['forecast_mse_per_window'])
        compare('persistence_'+k,p,step0[0]['levels'][k]['persistence_mse_per_window'])
    reference.close();report['initialization_checks']=checks
    data=evaluate(selected,rows)
    records=[];summaries={}
    for k in LEVELS:
        p,f=errors(data['X'][k],data['Y'][k],data['prediction'][k])
        for j,row in enumerate(rows):
            if k==LEVELS[0]:records.append(dict(row,levels={}))
            records[j]['levels'][k]=dict(persistence_mse=float(p[j]),forecast_mse=float(f[j]),difference=float(f[j]-p[j]))
            if row['stratum']=='quiet':require(np.array_equal(data['X'][k][j],data['Y'][k][j]),'Quiet pair encodings differ')
    groups={st:[i for i,r in enumerate(rows) if r['stratum']==st] for st in strata}
    for composition in ('value_only','status_only','mixed'):
        groups['new_mixed/newest_'+composition]=[i for i,r in enumerate(rows) if r['stratum']=='new_mixed' and r['newest_composition']==composition]
    for name,idx in groups.items():
        summaries[name]={k:summary(*(data[side][k][idx] for side in ('X','Y','prediction'))) for k in LEVELS} if idx else dict(n=0)
    require(state_hash(model)==before,'Parameters or buffers changed')
    dataset.verify_unchanged()
    require(all(windows.sha256(p)==hashes[k] for k,p in dependencies.items()),'Dependency changed during execution')
    require(report['hierarchy_forward_calls']==72 and report['hbf_forward_calls']==36,'Execution budget differs')
    write_json(out/'pair_metrics.json',records)
    write_json(out/'stratum_metrics.json',summaries)
    with (out/'representations.npz').open('xb') as f:
        np.savez_compressed(f,indices=np.array(selected),**{s+'_'+k:a for s,c in data.items() for k,a in c.items()})
    report.update(status='PASS',parameters_and_buffers_unchanged=True,final_state_sha256=state_hash(model),
        source_and_artifact_recheck_passed=True,quiet_pair_identity_passed=True,
        output_hashes={n:windows.sha256(out/n) for n in ('pair_metrics.json','stratum_metrics.json','representations.npz')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root_dir')
    parser.add_argument('--input_dir',required=True)
    parser.add_argument('--pilot_dir',required=True)
    parser.add_argument('--census_dir',required=True)
    parser.add_argument('--device',choices=['cpu'])
    args=parser.parse_args()
    with open('configs/config.yaml') as f:config=yaml.safe_load(f)
    if args.project_root_dir:config['project_root_dir']=args.project_root_dir
    processed=Path(get_processed_folder(config))
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=processed/'v2_validation'/'transition_diagnostic'/run_id
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status='BLOCKED',optimizer_steps=0,backward_calls=0,checkpoint_loaded=False,
        hierarchy_forward_calls=0,hbf_forward_calls=0,
        limitations=['Untrained hierarchy; not learned forecasting skill.',
            'Training-only enriched diagnostic; not population performance or generalization.',
            'Nonoverlapping spans do not establish independence; no inferential tests.',
            'Quiet prepared values/statuses do not establish physical inactivity or freshness.'])
    try:run(config,processed,args,report,out)
    except (ValueError,KeyError,TypeError,OSError,IndexError,RuntimeError) as exc:
        report.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    write_json(out/'diagnostic_report.json',report)
    print(f'Transition diagnostic: {report["status"]}\nReport: {out/"diagnostic_report.json"}')
    if 'error' in report:print(report['error'])
    return 0 if report['status']=='PASS' else 2


if __name__=='__main__':raise SystemExit(main())
