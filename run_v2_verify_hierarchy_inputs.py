"""Bounded train-only six-feature DBRL -> BSE -> BIL -> EBRL compatibility."""
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
from src.utils.get_folders_utils import get_processed_folder

require = windows.require
LEVELS = ('Z', 'S', 'S_tilde', 'g')
GROUPS = ('dbrl', 'bse', 'bil', 'ebrl')
STATE_IDS = ('access_entry', 'occupancy_presence', 'physical_disturbance',
             'thermal_environment', 'switch_controlled_appliance_activity',
             'lighting_activity', 'air_quality', 'water_heating', 'humidification')
EDGES = ((0, 1), (1, 4), (1, 5), (1, 7), (4, 3), (7, 3), (8, 6))


def gradient_probe(output, inputs, names):
    """A separate terminal probe per level prevents shorter paths hiding a break."""
    weights = torch.linspace(0.5, 1.5, output.numel(), device=output.device).reshape_as(output)
    loss = (weights * output).mean()
    gradients = torch.autograd.grad(loss, inputs, retain_graph=True, allow_unused=True)
    norms = {}
    for name, grad in zip(names, gradients):
        require(grad is not None, f'Disconnected gradient: {name}')
        require(torch.isfinite(grad).all().item(), f'Nonfinite gradient: {name}')
        norms[name] = float(grad.detach().double().norm())
    return float(loss.detach()), norms


def verify(config, processed, args, report):
    require(isinstance(config.get('hdeg'), dict) and 'dbrl' in config['hdeg'],
            'configs/config.yaml must contain hdeg.dbrl; check the current working directory')
    require(args.input_dir.strip() and (Path(args.input_dir) / 'manifest.json').is_file(),
            '--input_dir must name the prepared row directory containing manifest.json')
    settings = config.get('input_state_hierarchy_verification', {})
    batch_size = args.batch_size if args.batch_size is not None else settings.get('batch_size', 8)
    device_name = args.device or settings.get('device', 'cpu')
    require(type(batch_size) is int and 1 <= batch_size <= 16, 'batch_size must be 1..16')
    require(device_name in ('cpu', 'cuda'), 'device must be cpu or cuda')
    require(device_name != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
    proof_path = Path(args.dbrl_report)
    proof = json.loads(proof_path.read_text())
    require(proof['status'] == 'PASS' and proof['protocol'] == 'v2_dbrl_six_feature_compatibility_v1',
            'A successful six-feature DBRL report is required')
    require(proof['optimizer_steps'] == 0 and proof['parameters_unchanged'] is True,
            'Expected a fresh-initialization DBRL compatibility report')
    dataset = windows.InputStateWindowDataset(args.input_dir, config['preprocessing']['window_size'])
    require(dataset.manifest_sha256 == proof['manifest_sha256'], 'Prepared manifest differs from DBRL report')
    require(config['preprocessing'] == proof['configuration']['preprocessing'] ==
            dataset.preparation['configuration']['preprocessing'], 'Preprocessing settings differ')
    require(dataset.window_size == proof['window_size'], 'Window size differs from DBRL report')
    indices = proof['selected_window_indices']
    require(len(indices) == proof['budget']['windows'] == 32 and
            all(type(i) is int and 0 <= i < len(dataset) for i in indices) and
            indices == sorted(set(indices)), 'Expected exactly the 32 distinct verified training windows')
    seed = config.get('seed', 42)
    require(type(seed) is int and seed == proof['seed'], 'Seed differs from DBRL check')
    semantic_path = Path(config['project_root_dir']) / 'configs/hdeg/behavioral_states.yaml'
    dependencies = {'config': Path('configs/config.yaml'), 'dbrl_report': proof_path,
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
    hashes = {k: windows.sha256(p) for k, p in dependencies.items()}
    for key in ('dbrl', 'window_loader', 'graph_builder', 'graph_reasoner', 'devices', 'timestamps', 'selected_features', 'registry'):
        require(hashes[key] == proof['source_hashes'][key], f'Dependency changed since DBRL PASS: {key}')
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
    require(kwargs == proof['model_parameters'] and p.get('input_dim', 6) == 6, 'DBRL settings differ')
    bse_heads = config['hdeg'].get('bse', {}).get('num_heads', 1)
    require(type(bse_heads) is int and bse_heads > 0, 'BSE num_heads must be positive integer')
    # Current BIL implements a single frozen-graph attention mechanism, not multihead attention.
    require(config['hdeg'].get('bil', {}).get('num_heads', 1) == 1, 'Current BIL supports one head')
    bil_settings = config['hdeg'].get('bil', {})
    bil_kwargs = dict(negative_slope=bil_settings.get('negative_slope', 0.2),
                      dropout=bil_settings.get('dropout', 0.0))
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
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    original_mask = mask.clone()
    report.update(source_hashes=hashes, manifest_sha256=dataset.manifest_sha256,
                  selected_window_indices=indices, seed=seed, device=device_name,
                  budget={'windows': 32, 'batch_size': batch_size},
                  model_parameters={'dbrl': kwargs, 'bse': {'num_heads': bse_heads}, 'bil': bil_kwargs},
                  semantic_state_ids=list(STATE_IDS), semantic_state_names=list(semantics.state_names()),
                  compatibility_mask=mask.cpu().tolist(), big_edges=list(EDGES),
                  feature_order=['value'] + list(windows.STATUS_CODES),
                  framework_versions={'torch': torch.__version__, 'torch_geometric': torch_geometric.__version__,
                                      'numpy': np.__version__}, batches=[])
    for start in range(0, 32, batch_size):
        x = torch.from_numpy(dataset.get_batch(indices[start:start+batch_size])['X']).to(device).requires_grad_(True)
        z = model['dbrl'](x)
        s = model['bse'](z, mask)
        st = model['bil'](s)
        g = model['ebrl'](st)
        outputs = (z, s, st, g)
        expected_shapes = ((len(x),23,d), (len(x),9,d), (len(x),9,d), (len(x),d))
        entry = {'indices': indices[start:start+batch_size], 'levels': {}}
        for level, (name, output, shape) in enumerate(zip(LEVELS, outputs, expected_shapes)):
            require(tuple(output.shape) == shape and torch.isfinite(output).all().item(), f'Invalid {name} output')
            params = [(n, p) for n, p in model.named_parameters() if n.split('.')[0] in GROUPS[:level+1]]
            # Include intermediate outputs so every upstream interface is explicitly exercised.
            names = ['input.X'] + ['interface.'+n for n in LEVELS[:level]] + [n for n, _ in params]
            tensors = [x] + list(outputs[:level]) + [p for _, p in params]
            loss, norms = gradient_probe(output, tensors, names)
            require(norms['input.X'] > 0, f'{name} has zero input gradient')
            for interface in LEVELS[:level]:
                require(norms['interface.'+interface] > 0, f'{name} has zero gradient at {interface}')
            group_norms = {group: sum(v for n, v in norms.items() if n.startswith(group+'.'))
                           for group in GROUPS[:level+1]}
            require(all(v > 0 for v in group_norms.values()), f'{name} has a zero-gradient module group')
            entry['levels'][name] = {'shape': list(shape), 'min': float(output.detach().min()),
                'max': float(output.detach().max()), 'probe': loss, 'gradient_norms': norms,
                'group_gradient_norm_sums': group_norms}
        report['batches'].append(entry)
        print(f'Checked hierarchical batch {len(report["batches"])} ({start+len(x)}/32 windows)', flush=True)
    require(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()), 'Parameters or buffers changed')
    require(torch.equal(mask, original_mask), 'Semantic mask changed')
    dataset.verify_unchanged()
    require(all(windows.sha256(p) == hashes[k] for k, p in dependencies.items()), 'Dependency changed during execution')
    report.update(status='PASS', parameters_and_buffers_unchanged=True, semantic_mask_unchanged=True,
                  hierarchical_forward_calls=len(report['batches']), gradient_probe_calls=4*len(report['batches']),
                  source_and_artifact_recheck_passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root_dir')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--dbrl_report', required=True)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    with open('configs/config.yaml') as f:
        config = yaml.safe_load(f)
    if args.project_root_dir:
        config['project_root_dir'] = args.project_root_dir
    processed = Path(get_processed_folder(config))
    report = dict(protocol='v2_six_feature_hierarchy_compatibility_v1', status='BLOCKED',
                  configuration=config, configuration_path=str(Path('configs/config.yaml').resolve()),
                  optimizer_steps=0, checkpoint_loaded=False, scientific_claim_decision='NOT_EVALUATED',
                  limitations=['Fresh initialization; bounded training inputs only; no forecasting or optimization.',
                               'Gradient probes establish connectivity, not representation quality or convergence.',
                               'Hard top-k selection is discrete; no gradient-through-selection claim.',
                               'Availability-aware inputs do not isolate physical behavior.',
                               'Existing semantic assignments are reused; their scientific adequacy is not validated.',
                               'Individual zero gradients are reported; group nonzero checks do not prove every parameter is active.'])
    try:
        verify(config, processed, args, report)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, IndexError) as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    out = processed / 'v2_validation' / 'hierarchy_input_compatibility' / run_id
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'hierarchy_input_compatibility_report.json'
    with path.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(f'Hierarchical input compatibility: {report["status"]}\nReport: {path}')
    if 'error' in report:
        print(report['error'])
    return 0 if report['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
