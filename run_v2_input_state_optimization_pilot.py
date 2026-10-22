"""Fixed 100-step train-only six-feature HDEG optimization diagnostic."""
from __future__ import annotations
import argparse
import inspect
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


EVALUATION_STEPS = (0, 4, 20, 40, 60, 80, 100)
LOSS_NAMES = dict(Z='L_Z', S='L_S', S_tilde='L_S_tilde', g='L_G')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
    temporary.replace(path)


def append_json(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')
        f.flush()


def ratio(numerator, denominator):
    # Preserve absolute denominator and never substitute an epsilon.
    return {'value': numerator / denominator if denominator != 0 else None,
            'denominator': denominator,
            'status': 'DEFINED' if denominator != 0 else 'ZERO_DENOMINATOR'}


def group_name(name):
    if not name.startswith('hbf.'):
        return name.split('.')[0]
    part = name.split('.')[1]
    if part in ('device_prediction_head', 'state_prediction_head',
                'context_prediction_head', 'global_prediction_head'):
        return 'hbf.' + part
    return 'hbf.shared_backbone'


def parameter_measurements(model, previous, initial):
    groups = {}
    for name, p in model.named_parameters():
        now = p.detach().double()
        old = previous[name].double()
        origin = initial[name].double()
        delta = now-old
        g = groups.setdefault(group_name(name), dict(update_sq=0., parameter_before_sq=0.,
            cumulative_sq=0., initial_sq=0., max_abs_change=0., changed_entries=0, entries=0))
        g['update_sq'] += float(delta.square().sum())
        g['parameter_before_sq'] += float(old.square().sum())
        g['cumulative_sq'] += float((now-origin).square().sum())
        g['initial_sq'] += float(origin.square().sum())
        g['max_abs_change'] = max(g['max_abs_change'], float(delta.abs().max()))
        g['changed_entries'] += int((now != old).sum())
        g['entries'] += p.numel()
    result = {}
    for name, g in groups.items():
        update, displacement = g['update_sq']**0.5, g['cumulative_sq']**0.5
        result[name] = dict(update_norm=update,
            update_to_parameter_norm=ratio(update, g['parameter_before_sq']**0.5),
            cumulative_displacement_norm=displacement,
            displacement_to_initial_norm=ratio(displacement, g['initial_sq']**0.5),
            max_abs_change=g['max_abs_change'], changed_entries=g['changed_entries'],
            entries=g['entries'], changed_fraction=g['changed_entries']/g['entries'])
    return result


def representation_metrics(current, observed, predicted, initial, baseline=None):
    result = {}
    for level in LEVELS:
        a, b, f = (collection[level].double() for collection in (current, observed, predicted))
        # Axis zero contains windows. Never pool nodes into the sample axis.
        entry = dict(forecast_mse=float((f-b).square().mean()),
            persistence_mse=float((a-b).square().mean()),
            X_variation=float(a.var(dim=0, unbiased=False).mean()),
            Y_variation=float(b.var(dim=0, unbiased=False).mean()),
            X_rms=float(a.square().mean().sqrt()), Y_rms=float(b.square().mean().sqrt()),
            X_drift_mse=float((a-initial['X'][level].double()).square().mean()),
            target_drift_mse=float((b-initial['Y'][level].double()).square().mean()))
        relative = ratio(entry['forecast_mse'], entry['persistence_mse'])
        entry['persistence_relative_skill'] = {**relative,
            'value': None if relative['value'] is None else 1-relative['value']}
        entry['ratios_to_initial'] = {k: ratio(v, (baseline[level][k] if baseline is not None else v))
                                      for k,v in entry.items() if isinstance(v, float)}
        entry['forecast_mse_per_window'] = (f-b).square().reshape(len(b),-1).mean(1).tolist()
        entry['persistence_mse_per_window'] = (a-b).square().reshape(len(b),-1).mean(1).tolist()
        result[level] = entry
    return result


def execute_schedule(model, encode, paired_batches, integrity_check, out, report):
    """Fixed 100-step engine; receives immutable selected input batches only."""
    require(len(paired_batches) == 4 and all(len(b['indices']) == 8 for b in paired_batches),
            'Protocol requires four fixed batches of eight')
    params = list(model.named_parameters())
    initial_parameters = {n:p.detach().clone() for n,p in params}
    initial_buffers = {n:b.detach().clone() for n,b in model.named_buffers()}
    optimizer = torch.optim.Adam([p for _,p in params], lr=1e-4, betas=(0.9,0.999),
                                 eps=1e-8, weight_decay=0, amsgrad=False, foreach=False)
    report['optimizer'] = dict(name='Adam', lr=1e-4, betas=[0.9,0.999], eps=1e-8,
                              weight_decay=0, amsgrad=False, foreach=False, fresh_state=True)
    initial_representations = None
    baseline = None
    def forecast(reps):
        return model['hbf'](*(reps[k] for k in LEVELS))
    def evaluate(step):
        nonlocal initial_representations, baseline
        integrity_check(full=True)
        model.eval()
        collections = [{k:[] for k in LEVELS} for _ in range(3)]
        with torch.no_grad():
            for batch in paired_batches:
                x = encode(batch['X'])
                y = encode(batch['Y'])
                prediction = forecast(x)
                for dest, source in zip(collections, (x,y,prediction)):
                    for level in LEVELS:
                        require(torch.isfinite(source[level]).all().item(), f'Nonfinite evaluation {level}')
                        require(source[level].shape == y[level].shape, f'Evaluation shape differs: {level}')
                        dest[level].append(source[level].detach().cpu())
        x,y,prediction = [{k:torch.cat(v) for k,v in c.items()} for c in collections]
        if initial_representations is None:
            initial_representations = {'X':x, 'Y':y}
            with (out/'initialization_representations.npz').open('xb') as f:
                np.savez_compressed(f, **{side+'_'+k:v.numpy() for side,c in initial_representations.items() for k,v in c.items()})
            report['initialization_representations_sha256'] = windows.sha256(out/'initialization_representations.npz')
        metrics = representation_metrics(x,y,prediction,initial_representations,baseline)
        if baseline is None:
            baseline = metrics
        record = dict(step=step, levels=metrics,
                      weighted_total_loss=sum(v['forecast_mse'] for v in metrics.values()))
        append_json(out/'evaluations.jsonl', record)
        report['completed_evaluation_steps'].append(step)
        require(all(torch.equal(b,initial_buffers[n]) for n,b in model.named_buffers()), 'Model buffer changed')
        write_json(out/'pilot_report.json',report)
        print(f'Evaluation after {step} updates: total MSE={record["weighted_total_loss"]:.8g}',flush=True)
    evaluate(0)
    report['status'] = 'RUNNING'
    for step in range(1,101):
        report['attempted_step'] = step
        report['phase'] = 'before_update'
        integrity_check(full=False)
        batch = paired_batches[(step-1)%4]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x = batch['X'].detach().requires_grad_(True)
        y = batch['Y'].detach().requires_grad_(True)
        current = encode(x)
        observed = encode(y)
        for value in observed.values():
            require(value.requires_grad, 'Target branch detached')
            value.retain_grad()
        predicted = forecast(current)
        objectives = model['mo'](observed,predicted).as_dict()
        require(all(v.ndim == 0 and torch.isfinite(v).item() for v in objectives.values()), 'Nonfinite/scalar loss failure')
        objectives['L_HDEG'].backward()
        missing = [n for n,p in params if p.grad is None]
        nonfinite = [n for n,p in params if p.grad is not None and not torch.isfinite(p.grad).all().item()]
        gradient_norms = {n:None if p.grad is None or n in nonfinite else float(p.grad.detach().double().norm()) for n,p in params}
        report['last_gradient_check'] = dict(step=step,missing=missing,nonfinite=nonfinite)
        require(not missing and not nonfinite, f'Missing/nonfinite parameter gradients: {missing}, {nonfinite}')
        for name,value in [('X',x),('Y',y),*observed.items()]:
            require(value.grad is not None and torch.isfinite(value.grad).all().item(), f'Broken/nonfinite gradient: {name}')
        previous = {n:p.detach().clone() for n,p in params}
        record = dict(step=step,cycle=(step-1)//4+1,batch_index=(step-1)%4,indices=batch['indices'],
                      losses_pre_update={k:float(v.detach()) for k,v in objectives.items()},
                      gradient_norms_pre_update=gradient_norms,
                      input_gradient_norms={k:float(v.grad.detach().double().norm()) for k,v in [('X',x),('Y',y)]},
                      missing_gradients=missing,nonfinite_gradients=nonfinite)
        report['phase'] = 'optimizer_step_in_progress'
        optimizer.step()
        # Count successful calls even when subsequent numerical validation fails.
        report['optimizer_steps'] = step
        report['phase'] = 'after_update'
        require(all(torch.isfinite(p).all().item() for _,p in params), 'Nonfinite parameter after update')
        require(all(torch.equal(b,initial_buffers[n]) for n,b in model.named_buffers()), 'Model buffer changed')
        record['parameter_updates'] = parameter_measurements(model,previous,initial_parameters)
        grouped = {}
        for n,value in gradient_norms.items():
            group = group_name(n)
            grouped[group] = grouped.get(group,0.) + value**2
        record['group_gradient_norms_pre_update'] = {k:v**0.5 for k,v in grouped.items()}
        append_json(out/'steps.jsonl',record)
        report['completed_step_records'] = step
        write_json(out/'pilot_report.json',report)
        if step in EVALUATION_STEPS:
            evaluate(step)
    integrity_check(full=True)
    report.update(status='COMPLETED',phase='finished',buffers_unchanged=True,
                  source_and_artifact_recheck_passed=True)


def prepare_and_run(config, processed, args, report, out):
    require(isinstance(config.get('hdeg'), dict) and 'dbrl' in config['hdeg'],
            'configs/config.yaml must contain hdeg.dbrl; check the current working directory')
    require(args.input_dir.strip() and (Path(args.input_dir) / 'manifest.json').is_file(),
            '--input_dir must name the prepared row directory containing manifest.json')
    settings = config.get('input_state_optimization_pilot', {})
    batch_size = settings.get('batch_size',8)
    require(set(settings) <= {'batch_size','updates','learning_rate','device'}, 'Unknown pilot execution setting')
    require(type(settings.get('updates',100)) is int and settings.get('updates',100) == 100, 'Fixed budget is 100 updates')
    require(type(settings.get('learning_rate',1e-4)) in (int,float) and settings.get('learning_rate',1e-4) == 1e-4, 'Fixed learning rate is 1e-4')
    device_name = args.device or settings.get('device', 'cpu')
    require(type(batch_size) is int and batch_size == 8, 'Fixed batch size is eight')
    require(device_name in ('cpu', 'cuda'), 'device must be cpu or cuda')
    require(device_name != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
    proof_path = Path(args.compatibility_report)
    proof = json.loads(proof_path.read_text())
    require(proof['status'] == 'PASS' and proof['protocol'] == 'v2_hbf_objective_compatibility_v1',
            'A successful HBF/objective compatibility report is required')
    require(proof['optimizer_steps'] == 0 and proof['parameters_and_buffers_unchanged'] is True,
            'Expected a fresh-initialization HBF/objective compatibility report')
    require(proof['target_gradient_policy'] == 'LIVE_SHARED_ENCODER_NO_DETACH', 'Target policy differs')
    dataset = windows.InputStateWindowDataset(args.input_dir, config['preprocessing']['window_size'])
    require(dataset.manifest_sha256 == proof['manifest_sha256'], 'Prepared manifest differs from compatibility report')
    require(config['preprocessing'] == proof['configuration']['preprocessing'] ==
            dataset.preparation['configuration']['preprocessing'], 'Preprocessing settings differ')
    require(dataset.window_size == proof['configuration']['preprocessing']['window_size'], 'Window size differs from compatibility report')
    indices = proof['selected_window_indices']
    require(len(indices) == proof['budget']['windows'] == 32 and
            all(type(i) is int and 0 <= i < len(dataset) for i in indices) and
            indices == sorted(set(indices)), 'Expected exactly the 32 distinct verified training windows')
    seed = config.get('seed', 42)
    require(type(seed) is int and seed == proof['seed'] == 42, 'Seed differs from compatibility check')
    semantic_path = Path(config['project_root_dir']) / 'configs/hdeg/behavioral_states.yaml'
    dependencies = {'config': Path('configs/config.yaml'), 'compatibility_report': proof_path,
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
        require(hashes[key] == proof['source_hashes'][key], f'Dependency changed since compatibility PASS: {key}')
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
            'BSE/BIL settings differ from compatibility check')
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
    artifact_paths = [dataset.manifest_path] + [dataset.directory/name for name in dataset.metadata_hashes]
    artifact_paths += [dataset.directory/e['file'] for e in dataset.shards]
    watched = list(dependencies.values()) + artifact_paths
    stamps = {str(p):windows.stamp(p) for p in watched}
    mask_before = mask.clone()
    def integrity_check(full=False):
        require(all(windows.stamp(p) == stamps[str(p)] for p in watched), 'Source or input artifact changed')
        require(torch.equal(mask,mask_before), 'Semantic mask changed')
        if full:
            dataset.verify_unchanged()
            require(all(windows.sha256(p) == hashes[k] for k,p in dependencies.items()), 'Dependency hash changed')
    paired_batches = []
    for start in range(0,32,8):
        batch = dataset.get_batch(indices[start:start+8])
        require(np.array_equal(batch['X'][:,1:],batch['Y'][:,:-1]), 'X/Y overlap differs')
        require(np.all(batch['Y_timestamps']-batch['X_timestamps'] == np.timedelta64(1,'s')), 'Pair shift differs')
        paired_batches.append({'indices':indices[start:start+8],
                              'X':torch.from_numpy(batch['X']).to(device),
                              'Y':torch.from_numpy(batch['Y']).to(device)})
    protocol = dict(protocol=report['protocol'],configuration=config,source_hashes=hashes,
                    manifest_sha256=dataset.manifest_sha256,seed=42,device=device_name,
                    updates=100,batch_size=8,cycles=25,evaluation_steps=list(EVALUATION_STEPS),
                    selected_window_indices=indices,
                    batch_indices=[b['indices'] for b in paired_batches],
                    target_gradient_policy=report['target_gradient_policy'],
                    optimizer=dict(name='Adam',lr=1e-4,betas=[0.9,0.999],eps=1e-8,weight_decay=0,amsgrad=False,foreach=False),
                    loss_weights=weights,model_parameters=proof['model_parameters'],
                    semantic_state_ids=list(STATE_IDS),big_edges=list(EDGES),compatibility_mask=mask.cpu().tolist(),
                    representation_variance='population variance across 32 windows per coordinate, then coordinate mean',
                    denominator_policy='zero -> null plus explicit denominator; no epsilon',
                    measurement_set_role='same windows used for optimization; no generalization assessment')
    write_json(out/'protocol.json',protocol)
    report['protocol_sha256'] = windows.sha256(out/'protocol.json')
    execute_schedule(model,encode,paired_batches,integrity_check,out,report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root_dir')
    parser.add_argument('--input_dir',required=True)
    parser.add_argument('--compatibility_report',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'])
    args = parser.parse_args()
    with open('configs/config.yaml') as f:
        config = yaml.safe_load(f)
    if args.project_root_dir:
        config['project_root_dir'] = args.project_root_dir
    processed = Path(get_processed_folder(config))
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out = processed/'v2_validation'/'input_state_optimization_pilot'/run_id
    out.mkdir(parents=True,exist_ok=False)
    report = dict(protocol='v2_input_state_optimization_pilot_v1',status='PREPARING',
        target_gradient_policy='LIVE_SHARED_ENCODER_NO_DETACH',configuration=config,
        configuration_path=str(Path('configs/config.yaml').resolve()),optimizer_steps=0,
        completed_step_records=0,attempted_step=0,completed_evaluation_steps=[],checkpoint_loaded=False,
        scientific_claim_decision='NOT_EVALUATED',
        limitations=['Repeated optimization of the same 32 training pairs; no generalization claim.',
                     'Live targets can move or contract; loss reduction alone is insufficient.',
                     'Variation is across windows at fixed node/coordinate identity.',
                     'No collapse threshold, best-step selection, budget extension, or old-checkpoint reuse.'])
    try:
        prepare_and_run(config,processed,args,report,out)
    except KeyboardInterrupt:
        report.update(status='INTERRUPTED',error='KeyboardInterrupt; no automatic resume')
    except (ValueError,TypeError,KeyError,OSError,RuntimeError,IndexError) as exc:
        report.update(status='STOPPED' if report['attempted_step'] else 'BLOCKED',error=f'{type(exc).__name__}: {exc}')
    report['result_files'] = {p.name:windows.sha256(p) for p in out.iterdir() if p.is_file() and p.name != 'pilot_report.json' and not p.name.endswith('.tmp')}
    write_json(out/'pilot_report.json',report)
    print(f'Pilot status: {report["status"]}; optimizer steps: {report["optimizer_steps"]}/100\nReport: {out/"pilot_report.json"}')
    if 'error' in report:
        print(report['error'])
    return 0 if report['status'] == 'COMPLETED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
