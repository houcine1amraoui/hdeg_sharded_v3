"""Frozen-event, saved-initialization timing diagnostic. No optimization."""
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


def changes(block):
    v, s = block['values'], block['status']
    return (s[1:] != s[:-1]) | ((s[1:] == 0) & (s[:-1] == 0) & (v[1:] != v[:-1]))


def validate_selection(selection, ds):
    """Recheck frozen identities and 60-row isolation without selecting new events."""
    require(selection['protocol'] == 'source_timing_selection_v1' and
            selection['status'] == 'FROZEN_SELECTION', 'Not a frozen timing selection')
    require(selection['offsets'] == OFFSETS and selection['total_pairs'] == 60, 'Timing budget differs')
    require(Counter(e['family'] for e in selection['selected_events']) == QUOTAS, 'Event quotas differ')
    require(Counter(c['type'] for c in selection['controls']) == dict(quiet=4, all_missing=2), 'Control quotas differ')
    occupied, rows = [], []
    for event_id, event in enumerate(selection['selected_events']):
        e, c = event['event_row'], event['channel_index']
        require(type(e) is int and type(c) is int and 0 <= c < 23 and 30 <= e <= ds.num_rows-30,
                'Invalid event coordinates')
        require(event['channel'] == ds.devices[c], 'Channel order differs')
        a, b = e-30, e+30
        require((event['support_start'], event['support_stop']) == (a, b), 'Event support differs')
        require(not any(a < y and x < b for x, y in occupied), 'Event supports overlap')
        occupied.append((a, b))
        block = ds.read_rows(a, b)
        require(np.argwhere(changes(block)).tolist() == [[29, c]], 'Event no longer isolated')
        old, new = int(block['status'][29, c]), int(block['status'][30, c])
        require((old, new) == (event['from_status'], event['to_status']), 'Status direction differs')
        family = event['family']
        first = ds.metadata['channel_diagnostics'][event['channel']]['no_prior_record_rows']
        kind = ds.metadata['registry']['channels'][c]['type']
        if family.startswith('valid_'):
            require(old == new == 0 and family == 'valid_'+kind, 'Valid-event type differs')
            require(float(block['values'][30, c])-float(block['values'][29, c]) == event['normalized_delta'],
                    'Event magnitude differs')
        elif family == 'first_observation':
            require(e == first and old == 3 and new != 3, 'Not the first observed grid state')
        else:
            require((old, new) == ((0, 2) if family == 'availability_loss' else (2, 0)), 'Availability family differs')
        require([p['destination_offset'] for p in event['placements']] == OFFSETS, 'Placements differ')
        for p in event['placements']:
            j, i = p['destination_offset'], p['index']
            require(i == e-j, 'Placement index differs')
            require(np.argwhere(changes(ds.read_rows(i, i+31))).tolist() == [[j-1, c]], 'Placement transition differs')
            rows.append(dict(pair_id=len(rows), event_id=event_id, family=family, index=i,
                             destination_offset=j, channel=event['channel'], channel_index=c,
                             event_row=e, from_status=old, to_status=new))
    for control in selection['controls']:
        i = control['index']
        require(type(i) is int and 0 <= i < len(ds), 'Control out of bounds')
        require(not any(i < y and x < i+31 for x, y in occupied), 'Control support overlaps')
        occupied.append((i, i+31))
        block = ds.read_rows(i, i+31)
        require(not changes(block).any(), 'Control has transitions')
        require(np.all(block['status'] == 3) if control['type'] == 'all_missing' else np.any(block['status'] == 0),
                'Control category differs')
        rows.append(dict(pair_id=len(rows), event_id=None, family=control['type'], index=i, destination_offset=None))
    require(len({r['index'] for r in rows}) == 60, 'Duplicate pairs')
    return rows


def metrics(x, y, p):
    x, y, p = [np.asarray(a, dtype=np.float64) for a in (x, y, p)]
    delta, correction = y-x, p-x
    den = float(np.linalg.norm(delta)*np.linalg.norm(correction))
    return dict(persistence_mse=float(np.square(delta).mean()),
                hbf_mse=float(np.square(y-p).mean()),
                hbf_minus_persistence_mse=float(np.square(y-p).mean()-np.square(delta).mean()),
                increment_l2=float(np.linalg.norm(delta)), increment_max_abs=float(np.abs(delta).max()),
                correction_mse=float(np.square(correction).mean()),
                correction_alignment=None if den == 0 else float(np.clip(np.sum(delta*correction)/den, -1, 1)),
                current_mean_square=float(np.square(x).mean()), target_mean_square=float(np.square(y).mean()))


def summarize(rows, arrays):
    pairs = []
    for i, row in enumerate(rows):
        entry = dict(row, levels={})
        for level in LEVELS:
            entry['levels'][level] = metrics(*(arrays[b+'_'+level][i] for b in ('X', 'Y', 'forecast')))
        pairs.append(entry)
    contrasts = []
    for eid in sorted({r['event_id'] for r in rows if r['event_id'] is not None}):
        by_offset = {r['destination_offset']: r for r in pairs if r['event_id'] == eid}
        entry = dict(event_id=eid, family=by_offset[1]['family'], channel=by_offset[1]['channel'], levels={})
        for level in LEVELS:
            entry['levels'][level] = {}
            for early in (1, 15):
                a, b = by_offset[30]['levels'][level], by_offset[early]['levels'][level]
                entry['levels'][level]['offset_30_minus_'+str(early)] = {
                    key: a[key]-b[key] for key in ('persistence_mse', 'hbf_mse', 'correction_mse')}
        contrasts.append(entry)
    groups = []
    keys = list(dict.fromkeys((r['family'], r['destination_offset']) for r in rows))
    for family, offset in keys:
        ids = [i for i, r in enumerate(rows) if (r['family'], r['destination_offset']) == (family, offset)]
        group = dict(family=family, destination_offset=offset, pairs=len(ids), levels={})
        for level in LEVELS:
            m = [pairs[i]['levels'][level] for i in ids]
            g = {key: dict(mean=float(np.mean([v[key] for v in m])), median=float(np.median([v[key] for v in m])),
                          minimum=min(v[key] for v in m), maximum=max(v[key] for v in m))
                 for key in ('persistence_mse', 'hbf_mse', 'correction_mse')}
            for branch in ('X', 'Y', 'forecast'):
                a = arrays[branch+'_'+level][ids].astype(np.float64).reshape(len(ids), -1)
                g[branch+'_mean_feature_variance'] = float(a.var(axis=0).mean())
            group['levels'][level] = g
        groups.append(group)
    return dict(pair_metrics=pairs, within_event_contrasts=contrasts, descriptive_groups=groups)


def run(args, config, out, report):
    from src.data_preparation import input_state_windows as windows
    selection = json.loads(args.selection_manifest.read_text())
    feasibility_path = args.selection_manifest.parent/'timing_event_feasibility_report.json'
    feasibility = json.loads(feasibility_path.read_text())
    proof = json.loads(args.hierarchy_report.read_text())
    require(feasibility['status'] == 'COMPLETE' and feasibility['feasibility'] == 'FULL_QUOTAS' and
            feasibility['selection_manifest_sha256'] == windows.sha256(args.selection_manifest), 'Selection feasibility/hash differs')
    require(proof['status'] == 'PASS' and proof['protocol'] == 'source_hierarchy_forward_v1' and
            selection['hierarchy_report_sha256'] == windows.sha256(args.hierarchy_report), 'Hierarchy reference differs')
    require(proof['parameters_and_buffers_unchanged'] and proof['source_and_artifact_recheck_passed'], 'Hierarchy provenance gate failed')
    for key in ('hdeg', 'preprocessing', 'seed'):
        require(config.get(key) == proof['configuration'].get(key) == selection['configuration'].get(key),
                'Configuration differs: '+key)
    base = args.hierarchy_report.parent
    deps = dict(selection=args.selection_manifest, feasibility=feasibility_path, hierarchy_report=args.hierarchy_report,
                initialization=base/'initialization.pt', reference_pairs=base/'selection_manifest.json',
                reference_outputs=base/'forward_outputs.npz', config=Path('configs/config.yaml'), runner=Path(__file__),
                window_loader=Path(windows.__file__))
    require(selection['initialization_file_sha256'] == proof['initialization_file_sha256'] == windows.sha256(deps['initialization']),
            'Initialization file differs')
    require(selection['initialization_state_sha256'] == proof['initialization_state_sha256'], 'Initialization state reference differs')
    for key in ('reference_pairs', 'reference_outputs'):
        require(windows.sha256(deps[key]) == proof['output_hashes'][deps[key].name], 'Replay reference differs: '+key)
    ds = windows.InputStateWindowDataset(args.input_dir, 30)
    require(ds.policy_version == windows.SOURCE_POLICY and
            ds.manifest_sha256 == selection['manifest_sha256'] == proof['manifest_sha256'], 'Input lineage differs')
    require(ds.metadata_hashes == proof['metadata_hashes'], 'Input metadata differs')
    rows = validate_selection(selection, ds)
    report.update(input_checks='PASS', pair_count=len(rows), input_manifest_sha256=ds.manifest_sha256,
                  metadata_hashes=ds.metadata_hashes, selection_manifest_sha256=windows.sha256(args.selection_manifest))
    # Torch is deliberately imported after the input and frozen-selection gates.
    import torch
    import torch_geometric
    from src.common.graph.semantics import load_behavioral_state_config
    from src.models.hdeg.dbrl import DBRL
    from src.models.hdeg.bse import BehavioralStateEstimator
    from src.models.hdeg.bil import BehavioralInteractionLearner
    from src.models.hdeg.ebrl import EcosystemBehavioralRepresentationLearner
    from src.models.hdeg.hbf import HierarchicalBehavioralForecaster
    from src.models.hdeg.graph_reasoner import GraphReasoner
    from src.common.graph.graph_builder import GraphBuilder
    deps['semantic_config'] = args.semantic_config or Path(config['project_root_dir'])/'configs/hdeg/behavioral_states.yaml'
    deps['semantic_loader'] = Path(inspect.getfile(load_behavioral_state_config))
    for cls in (DBRL, BehavioralStateEstimator, BehavioralInteractionLearner,
                EcosystemBehavioralRepresentationLearner, HierarchicalBehavioralForecaster, GraphReasoner, GraphBuilder):
        deps[cls.__name__] = Path(inspect.getfile(cls))
    hashes = {k: windows.sha256(p) for k, p in deps.items()}
    for key in ('window_loader', 'semantic_config', 'semantic_loader', 'DBRL', 'BehavioralStateEstimator',
                'BehavioralInteractionLearner', 'EcosystemBehavioralRepresentationLearner',
                'HierarchicalBehavioralForecaster', 'GraphReasoner', 'GraphBuilder'):
        require(hashes[key] == proof['source_hashes'][key], 'Executed source differs: '+key)
    report['source_hashes'] = hashes
    require(proof['device'] == 'cpu' and proof['batch_size'] == 8, 'This diagnostic requires the saved CPU/batch-8 contract')
    torch.set_num_threads(1)
    torch.manual_seed(proof['seed'])
    torch.use_deterministic_algorithms(True)
    p = proof['model_parameters']; d = p['dbrl']['embedding_dim']
    require(d == 64, 'Embedding dimension differs')
    model = torch.nn.ModuleDict(dict(dbrl=DBRL(**p['dbrl']),
        bse=BehavioralStateEstimator(9, d, **p['bse']),
        bil=BehavioralInteractionLearner(d, num_states=9, **p['bil']),
        ebrl=EcosystemBehavioralRepresentationLearner(d, num_states=9),
        hbf=HierarchicalBehavioralForecaster(num_devices=23, num_states=9, embedding_dim=d, **p['hbf'])))
    model.load_state_dict(torch.load(deps['initialization'], map_location='cpu', weights_only=True), strict=True)
    model.eval()
    model.requires_grad_(False)
    before = state_digest(model)
    require(before == proof['initialization_state_sha256'], 'Loaded state digest differs')
    write(out/'devices.json', ds.devices)
    semantics = load_behavioral_state_config(deps['semantic_config'], out/'devices.json')
    require(list(semantics.device_ids()) == ds.devices and list(semantics.state_ids()) == proof['state_ids'], 'Semantic order differs')
    mask = semantics.torch_mask(device=torch.device('cpu')); saved_mask = mask.clone()
    require(mask.tolist() == proof['compatibility_mask'], 'Semantic mask differs')
    require(model['bil'].edge_index.tolist() == np.array(proof['big_edges']).T.tolist(), 'BIL edges differ')
    report.update(initialization_file_sha256=hashes['initialization'], initialization_state_sha256=before,
                  framework_versions=dict(torch=torch.__version__, torch_geometric=torch_geometric.__version__, numpy=np.__version__),
                  reference_framework_versions=proof['framework_versions'], model_parameters=p)

    def encode(x):
        report['encoder_forward_calls'] += 1
        z = model['dbrl'](x); s = model['bse'](z, mask); st = model['bil'](s)
        return dict(Z=z, S=s, S_tilde=st, g=model['ebrl'](st))

    def forecast(x):
        report['hbf_forward_calls'] += 1
        return model['hbf'](*(x[k] for k in LEVELS))

    def forward(indices):
        arrays = {b+'_'+k: [] for b in ('X', 'Y', 'forecast') for k in LEVELS}
        repeat_errors = {k: dict(representation=0.0, forecast=0.0) for k in LEVELS}
        with torch.inference_mode():
            for start in range(0, len(indices), 8):
                batch = ds.get_batch(indices[start:start+8]); x = torch.from_numpy(batch['X']); y = torch.from_numpy(batch['Y'])
                current = encode(x); pred = forecast(current); target = encode(y)
                repeat = encode(x); repeat_pred = forecast(repeat)
                for level, shape in zip(LEVELS, ((len(x), 23, d), (len(x), 9, d), (len(x), 9, d), (len(x), d))):
                    for branch in (current, target, pred, repeat, repeat_pred):
                        require(tuple(branch[level].shape) == shape and torch.isfinite(branch[level]).all().item(), 'Invalid forward output: '+level)
                        require(not branch[level].requires_grad, 'Unexpected autograd')
                    for name, a, b in (('representation', current[level], repeat[level]), ('forecast', pred[level], repeat_pred[level])):
                        repeat_errors[level][name] = max(repeat_errors[level][name], float((a-b).abs().max()))
                        require(torch.allclose(a, b, atol=ATOL, rtol=RTOL), 'Repeatability failed: '+level)
                    for name, branch in (('X', current), ('Y', target), ('forecast', pred)):
                        arrays[name+'_'+level].append(branch[level].numpy().copy())
                print('Forward pairs:', min(start+8, len(indices)), '/', len(indices), flush=True)
        report.setdefault('repeat_checks', []).append(dict(pairs=len(indices), maximum_absolute_errors=repeat_errors))
        return {k: np.concatenate(v) for k, v in arrays.items()}

    reference = json.loads(deps['reference_pairs'].read_text())
    require(reference['manifest_sha256'] == ds.manifest_sha256 and len(reference['selected_pairs']) == 32, 'Invalid replay selection')
    replay = forward([r['index'] for r in reference['selected_pairs']])
    report['replay_gate'] = {}
    with np.load(deps['reference_outputs'], allow_pickle=False) as saved:
        for key, a in replay.items():
            b = saved[key]
            require(a.shape == b.shape and np.isfinite(b).all(), 'Invalid replay reference: '+key)
            difference = float(np.abs(a.astype(float)-b.astype(float)).max())
            report['replay_gate'][key] = dict(max_abs_difference=difference, passed=bool(np.allclose(a, b, atol=ATOL, rtol=RTOL)))
            require(report['replay_gate'][key]['passed'], 'Saved-output replay failed: '+key)
        report['replay_increment_max_abs_difference'] = {
            k: float(np.abs((replay['Y_'+k].astype(float)-replay['X_'+k].astype(float))-
                            (saved['Y_'+k].astype(float)-saved['X_'+k].astype(float))).max()) for k in LEVELS}
    report['replay_gate_status'] = 'PASS'
    arrays = forward([r['index'] for r in rows])
    results = summarize(rows, arrays)
    # Identity controls use an explicit numerical tolerance, not an epsilon in ratios.
    controls = [i for i, r in enumerate(rows) if r['event_id'] is None]
    for level in LEVELS:
        require(np.allclose(arrays['X_'+level][controls], arrays['Y_'+level][controls], atol=ATOL, rtol=RTOL),
                'Identical-window control failed: '+level)
    require(state_digest(model) == before, 'Parameters or buffers changed')
    require(torch.equal(mask, saved_mask) and all(p.grad is None for p in model.parameters()), 'Mask changed or gradients exist')
    ds.verify_unchanged()
    require(all(windows.sha256(p) == hashes[k] for k, p in deps.items()), 'Dependency changed during diagnostic')
    write(out/'pair_metrics.json', results['pair_metrics'])
    write(out/'within_event_contrasts.json', results['within_event_contrasts'])
    write(out/'descriptive_groups.json', results['descriptive_groups'])
    write(out/'consumed_selection.json', selection)
    np.savez_compressed(out/'forward_outputs.npz', **arrays)
    report.update(status='COMPLETE', parameters_and_buffers_unchanged=True, source_files_unchanged=True,
                  semantic_mask_unchanged=True, identical_window_controls='PASS',
                  output_hashes={p.name: windows.sha256(p) for p in out.iterdir() if p.is_file()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project_root_dir')
    p.add_argument('--input_dir', required=True, type=Path)
    p.add_argument('--selection_manifest', required=True, type=Path)
    p.add_argument('--hierarchy_report', required=True, type=Path)
    p.add_argument('--semantic_config', type=Path)
    args = p.parse_args()
    with open('configs/config.yaml') as f:
        config = yaml.safe_load(f)
    if args.project_root_dir:
        config['project_root_dir'] = args.project_root_dir
    from src.utils.get_folders_utils import get_processed_folder
    out = Path(get_processed_folder(config))/'v2_validation'/'timing_diagnostic'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='BLOCKED', protocol='source_timing_forward_v1', configuration=config,
                  device='cpu', batch_size=8, optimizer_steps=0, backward_calls=0,
                  encoder_forward_calls=0, hbf_forward_calls=0, repeat_tolerance=dict(atol=ATOL, rtol=RTOL),
                  scientific_claim_decision='NOT_EVALUATED', limitations=[
        'Untrained saved initialization: HBF errors are descriptive, not learned forecasting skill.',
        'Within-event placements overlap and are paired observations, not independent samples.',
        'Moving the window changes pre/post-event composition as well as event position.',
        'Cross-family channels, magnitudes and contexts are not matched; no causal ranking.',
        'Continuous events cover four temperature channels, not PM2.5.',
        'First observation and availability transitions are not evidence of user activity.',
        'Prepared-grid timing is not independently verified raw event or annotation timing.',
        'No confidence intervals, significance tests, training decisions or behavior-discrimination claim.'])
    try:
        run(args, config, out, report)
    except Exception as exc:
        report.update(status='BLOCKED', error=f'{type(exc).__name__}: {exc}')
    write(out/'timing_diagnostic_report.json', report)
    print('Timing diagnostic:', report['status'])
    print('Report:', out/'timing_diagnostic_report.json')
    if 'error' in report:
        print(report['error'])
    return 0 if report['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
