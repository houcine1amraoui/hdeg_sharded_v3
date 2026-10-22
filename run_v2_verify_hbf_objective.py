"""Bounded six-feature HBF and live-target training-objective compatibility."""
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


def gradients(loss, named, allow_missing=False):
    values = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
    result = {}
    for (name, _), grad in zip(named, values):
        require(allow_missing or grad is not None, f'Disconnected gradient: {name}')
        if grad is not None:
            require(torch.isfinite(grad).all().item(), f'Nonfinite gradient: {name}')
        result[name] = grad
    return result


def norms(grads):
    return {k: None if v is None else float(v.detach().double().norm()) for k, v in grads.items()}


def verify(config, processed, args, report):
    require(isinstance(config.get('hdeg'), dict) and 'dbrl' in config['hdeg'],
            'configs/config.yaml must contain hdeg.dbrl; check the current working directory')
    require(args.input_dir.strip() and (Path(args.input_dir) / 'manifest.json').is_file(),
            '--input_dir must name the prepared row directory containing manifest.json')
    settings = config.get('input_state_hbf_objective_verification', {})
    batch_size = args.batch_size if args.batch_size is not None else settings.get('batch_size', 8)
    device_name = args.device or settings.get('device', 'cpu')
    require(type(batch_size) is int and 1 <= batch_size <= 16, 'batch_size must be 1..16')
    require(device_name in ('cpu', 'cuda'), 'device must be cpu or cuda')
    require(device_name != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
    proof_path = Path(args.hierarchy_report)
    proof = json.loads(proof_path.read_text())
    require(proof['status'] == 'PASS' and proof['protocol'] == 'v2_six_feature_hierarchy_compatibility_v1',
            'A successful six-feature hierarchy report is required')
    require(proof['optimizer_steps'] == 0 and proof['parameters_and_buffers_unchanged'] is True,
            'Expected a fresh-initialization DBRL compatibility report')
    dataset = windows.InputStateWindowDataset(args.input_dir, config['preprocessing']['window_size'])
    require(dataset.manifest_sha256 == proof['manifest_sha256'], 'Prepared manifest differs from hierarchy report')
    require(config['preprocessing'] == proof['configuration']['preprocessing'] ==
            dataset.preparation['configuration']['preprocessing'], 'Preprocessing settings differ')
    require(dataset.window_size == proof['configuration']['preprocessing']['window_size'], 'Window size differs from hierarchy report')
    indices = proof['selected_window_indices']
    require(len(indices) == proof['budget']['windows'] == 32 and
            all(type(i) is int and 0 <= i < len(dataset) for i in indices) and
            indices == sorted(set(indices)), 'Expected exactly the 32 distinct verified training windows')
    seed = config.get('seed', 42)
    require(type(seed) is int and seed == proof['seed'], 'Seed differs from hierarchy check')
    semantic_path = Path(config['project_root_dir']) / 'configs/hdeg/behavioral_states.yaml'
    dependencies = {'config': Path('configs/config.yaml'), 'hierarchy_report': proof_path,
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
    for key in ('dbrl', 'bse', 'bil', 'ebrl', 'semantic_config', 'semantic_loader', 'window_loader', 'graph_builder', 'graph_reasoner', 'devices', 'timestamps', 'selected_features', 'registry'):
        require(hashes[key] == proof['source_hashes'][key], f'Dependency changed since hierarchy PASS: {key}')
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
            'BSE/BIL settings differ from hierarchy check')
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
    model['mo'] = ModelOptimization(**weights).to(device)
    model.eval()
    def encode(x):
        z = model['dbrl'](x)
        s = model['bse'](z, mask)
        st = model['bil'](s)
        return dict(Z=z, S=s, S_tilde=st, g=model['ebrl'](st))
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    original_mask = mask.clone()
    report.update(source_hashes=hashes, manifest_sha256=dataset.manifest_sha256,
                  selected_window_indices=indices, seed=seed, device=device_name,
                  budget={'windows': 32, 'batch_size': batch_size},
                  model_parameters={'dbrl': kwargs, 'bse': {'num_heads': bse_heads}, 'bil': bil_kwargs, 'hbf': {'dynamics_hidden_dim': hbf_hidden}, 'mo': weights},
                  semantic_state_ids=list(STATE_IDS), semantic_state_names=list(semantics.state_names()),
                  compatibility_mask=mask.cpu().tolist(), big_edges=list(EDGES),
                  feature_order=['value'] + list(windows.STATUS_CODES),
                  framework_versions={'torch': torch.__version__, 'torch_geometric': torch_geometric.__version__,
                                      'numpy': np.__version__}, batches=[])
    head_names = dict(Z='device_prediction_head', S='state_prediction_head',
                      S_tilde='context_prediction_head', g='global_prediction_head')
    loss_names = dict(Z='L_Z', S='L_S', S_tilde='L_S_tilde', g='L_G')
    weight_buffers = ('lambda_Z', 'lambda_S', 'lambda_S_tilde', 'lambda_G')
    params = list(model.named_parameters())
    for start in range(0, 32, batch_size):
        batch = dataset.get_batch(indices[start:start+batch_size])
        require(np.array_equal(batch['X'][:,1:], batch['Y'][:,:-1]), 'X/Y one-row overlap differs')
        require(np.all(batch['Y_timestamps'] - batch['X_timestamps'] == np.timedelta64(1, 's')),
                'X/Y timestamp shift must be exactly one second')
        x = torch.from_numpy(batch['X']).to(device).requires_grad_(True)
        y = torch.from_numpy(batch['Y']).to(device).requires_grad_(True)
        current = encode(x)
        predicted = model['hbf'](*(current[k] for k in LEVELS))
        observed = encode(y)  # Deliberately LIVE: same encoder; no detach/no_grad.
        objectives = model['mo'](observed, predicted).as_dict()
        entry = {'indices': indices[start:start+batch_size], 'levels': {}}
        shapes = ((len(x),23,d), (len(x),9,d), (len(x),9,d), (len(x),d))
        for level, shape in zip(LEVELS, shapes):
            for branch in (current, observed, predicted):
                require(tuple(branch[level].shape) == shape and torch.isfinite(branch[level]).all().item(),
                        f'Invalid {level} representation or forecast')
            loss = objectives[loss_names[level]]
            reference = (predicted[level].double()-observed[level].double()).square().mean()
            require(torch.isclose(loss.double(), reference, rtol=1e-5, atol=1e-8).item(), 'MSE reduction differs')
            named = [('input.X', x), ('input.Y', y), ('target', observed[level]), ('forecast', predicted[level])] + params
            grads = gradients(loss, named, allow_missing=True)
            # Other forecast heads must be absent from this level-specific loss.
            for name, grad in grads.items():
                unused_head = any(name.startswith('hbf.'+head+'.') for lev, head in head_names.items() if lev != level)
                require((grad is None) == unused_head, f'Unexpected gradient connectivity: {level}/{name}')
            require(all(float(grads[k].detach().double().norm()) > 0 for k in ('input.X','input.Y','target','forecast')),
                    f'Zero branch/interface gradient for {level}')
            direct = 2*(observed[level]-predicted[level])/observed[level].numel()
            require(torch.allclose(grads['target'], direct, atol=1e-8, rtol=1e-5), 'Target derivative differs')
            require(torch.allclose(grads['forecast'], -direct, atol=1e-8, rtol=1e-5), 'Forecast derivative differs')
            entry['levels'][level] = {'shape': list(shape), 'loss': float(loss.detach()),
                                      'reference_mse_float64': float(reference.detach()), 'gradient_norms': norms(grads)}
        total = objectives['L_HDEG']
        weighted = sum(getattr(model['mo'], w).double()*objectives[loss_names[l]].double()
                       for l,w in zip(LEVELS,weight_buffers))
        require(torch.isclose(total.double(), weighted, atol=1e-8, rtol=1e-5).item(), 'Weighted objective differs')
        named = [('input.X', x), ('input.Y', y)] + params
        live_grads = gradients(total, named)
        forecast_only = model['mo']({k:v.detach() for k,v in observed.items()}, predicted).L_HDEG
        target_only = model['mo'](observed, {k:v.detach() for k,v in predicted.items()}).L_HDEG
        pg = gradients(forecast_only, named, allow_missing=True)
        tg = gradients(target_only, named, allow_missing=True)
        require(pg['input.Y'] is None and tg['input.X'] is None, 'Branch isolation failed')
        require(pg['input.X'] is not None and tg['input.Y'] is not None, 'Branch disconnected')
        require(all(tg[n] is None for n,_ in params if n.startswith('hbf.')), 'Target branch reaches HBF')
        errors = {}
        for name, live in live_grads.items():
            split = (pg[name] if pg[name] is not None else torch.zeros_like(live)) + (tg[name] if tg[name] is not None else torch.zeros_like(live))
            require(torch.allclose(live, split, rtol=1e-4, atol=1e-7), f'Gradient branch sum differs: {name}')
            errors[name] = float((live-split).detach().abs().max())
        live_norms = norms(live_grads)
        require(live_norms['input.X'] > 0 and live_norms['input.Y'] > 0, 'Total loss branch has zero gradient')
        for group in (*GROUPS, 'hbf'):
            require(sum(v for n,v in live_norms.items() if n.startswith(group+'.')) > 0,
                    f'Total loss has zero aggregate gradient in {group}')
        entry.update(total_loss=float(total.detach()), total_gradient_norms=live_norms,
                     forecast_branch_gradient_norms=norms(pg), target_branch_gradient_norms=norms(tg),
                     branch_sum_max_abs_errors=errors)
        report['batches'].append(entry)
        print(f'Checked HBF/objective batch {len(report["batches"])} ({start+len(x)}/32 pairs)', flush=True)
    require(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()), 'Parameters or buffers changed')
    require(torch.equal(mask, original_mask), 'Semantic mask changed')
    dataset.verify_unchanged()
    require(all(windows.sha256(p) == hashes[k] for k, p in dependencies.items()), 'Dependency changed during execution')
    report.update(status='PASS', parameters_and_buffers_unchanged=True, semantic_mask_unchanged=True,
                  encoder_forward_calls=2*len(report['batches']), hbf_forward_calls=len(report['batches']),
                  gradient_calls=7*len(report['batches']),
                  source_and_artifact_recheck_passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root_dir')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--hierarchy_report', required=True)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    with open('configs/config.yaml') as f:
        config = yaml.safe_load(f)
    if args.project_root_dir:
        config['project_root_dir'] = args.project_root_dir
    processed = Path(get_processed_folder(config))
    report = dict(protocol='v2_hbf_objective_compatibility_v1', status='BLOCKED',
                  target_gradient_policy='LIVE_SHARED_ENCODER_NO_DETACH', model_mode='eval_with_autograd',
                  configuration=config, configuration_path=str(Path('configs/config.yaml').resolve()),
                  optimizer_steps=0, checkpoint_loaded=False, scientific_claim_decision='NOT_EVALUATED',
                  limitations=['Fresh initialization; bounded training inputs; zero optimizer steps; no forecasting-skill claim.',
                               'Live-target MSE can decrease through target movement or collapse; finite gradients do not establish objective adequacy.',
                               'Hard top-k selection is discrete; no gradient-through-selection claim.',
                               'Availability-aware inputs do not isolate physical behavior.',
                               'Existing semantic assignments are reused; their scientific adequacy is not validated.',
                               'Individual zero gradients are reported; group nonzero checks do not prove every parameter is active.'])
    try:
        verify(config, processed, args, report)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, IndexError) as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    out = processed / 'v2_validation' / 'hbf_objective_compatibility' / run_id
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'hbf_objective_compatibility_report.json'
    with path.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(f'HBF/objective compatibility: {report["status"]}\nReport: {path}')
    if 'error' in report:
        print(report['error'])
    return 0 if report['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
