"""Read-only DBRL temporal/topology attribution on frozen timing pairs."""
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


def array_delta(x, y):
    delta = y.astype(np.float64)-x.astype(np.float64)
    return dict(mse=float(np.square(delta).mean()), max_abs=float(np.abs(delta).max()),
                per_device_mse=np.square(delta).mean(axis=-1).tolist())


def edge_delta(x, y):
    x, y = x.astype(bool), y.astype(bool)
    union = int((x | y).sum()); intersection = int((x & y).sum())
    return dict(x_edges=int(x.sum()), y_edges=int(y.sum()), added=int((y & ~x).sum()),
                removed=int((x & ~y).sum()), symmetric_difference=int((x ^ y).sum()),
                jaccard=1.0 if union == 0 else intersection/union,
                changed_source_nodes=np.flatnonzero(np.any(x != y, axis=1)).tolist(),
                changed_target_nodes=np.flatnonzero(np.any(x != y, axis=0)).tolist())


def pair_metrics(rows, arrays):
    result = []
    for i, row in enumerate(rows):
        r = dict(row, temporal=array_delta(arrays['X_H'][i], arrays['Y_H'][i]),
                 post_graph=array_delta(arrays['X_Z'][i], arrays['Y_Z'][i]))
        for name in ('neighbors', 'builder_edges', 'effective_edges'):
            r[name] = edge_delta(arrays['X_'+name][i], arrays['Y_'+name][i])
        r['cosine_max_abs_change'] = float(np.abs(arrays['X_cosine'][i].astype(float)-arrays['Y_cosine'][i].astype(float)).max())
        r['boundary_margins'] = {b: dict(minimum=float(arrays[b+'_margin'][i].min()),
            exact_tie_nodes=np.flatnonzero(arrays[b+'_margin'][i] == 0).tolist(),
            per_device=arrays[b+'_margin'][i].tolist()) for b in ('X','Y')}
        # Descriptive comparison, not an additive decomposition or causal attribution.
        r['post_minus_temporal_mse'] = r['post_graph']['mse']-r['temporal']['mse']
        result.append(r)
    return result


def run(args, config, out, report):
    from src.data_preparation import input_state_windows as windows
    for name in ('timing_report','hierarchy_report'):
        p = getattr(args, name)
        require(p.is_file(), f'--{name} must name a file; received {str(p)!r}')
    require((args.input_dir/'manifest.json').is_file(), '--input_dir must contain manifest.json')
    timing = json.loads(args.timing_report.read_text()); proof = json.loads(args.hierarchy_report.read_text())
    require(timing['status'] == 'COMPLETE' and timing['protocol'] == 'source_timing_forward_v1' and
            timing['replay_gate_status'] == 'PASS', 'Expected completed timing diagnostic with replay PASS')
    require(proof['status'] == 'PASS' and proof['protocol'] == 'source_hierarchy_forward_v1', 'Expected hierarchy PASS')
    require(timing['source_hashes']['hierarchy_report'] == windows.sha256(args.hierarchy_report), 'Hierarchy reference differs')
    for r in (timing, proof):
        require(r['optimizer_steps'] == r['backward_calls'] == 0 and r['parameters_and_buffers_unchanged'], 'Reference state changed')
        require(r['device'] == 'cpu' and r['batch_size'] == 8, 'CPU/batch-8 reference required')
        for k in ('hdeg','preprocessing','seed'):
            require(config.get(k) == r['configuration'].get(k), 'Configuration differs: '+k)
    deps = dict(timing_report=args.timing_report, hierarchy_report=args.hierarchy_report,
                selection=args.timing_report.parent/'consumed_selection.json',
                reference_pairs=args.timing_report.parent/'pair_metrics.json',
                reference_outputs=args.timing_report.parent/'forward_outputs.npz',
                initialization=args.hierarchy_report.parent/'initialization.pt',
                runner=Path(__file__), config=Path('configs/config.yaml'), window_loader=Path(windows.__file__))
    for k in ('selection','reference_pairs','reference_outputs'):
        require(windows.sha256(deps[k]) == timing['output_hashes'][deps[k].name], 'Timing artifact differs: '+k)
    require(windows.sha256(deps['initialization']) == timing['initialization_file_sha256'] == proof['initialization_file_sha256'],
            'Saved initialization file differs')
    selection = json.loads(deps['selection'].read_text())
    require(windows.sha256(deps['selection']) == timing['selection_manifest_sha256'], 'Selection differs')
    ds = windows.InputStateWindowDataset(args.input_dir, 30)
    require(ds.policy_version == windows.SOURCE_POLICY and ds.manifest_sha256 == selection['manifest_sha256'] ==
            timing['input_manifest_sha256'] == proof['manifest_sha256'], 'Input lineage differs')
    require(ds.metadata_hashes == timing['metadata_hashes'] == proof['metadata_hashes'], 'Metadata differs')
    rows = validate_selection(selection, ds)
    old_pairs = json.loads(deps['reference_pairs'].read_text())
    require(rows == [{k:v for k,v in r.items() if k != 'levels'} for r in old_pairs], 'Pair identities/order differ')
    import torch
    import torch_geometric
    from src.models.hdeg.dbrl import DBRL
    from src.models.hdeg.graph_reasoner import GraphReasoner
    from src.common.graph.graph_builder import GraphBuilder
    for cls in (DBRL, GraphReasoner, GraphBuilder):
        deps[cls.__name__] = Path(inspect.getfile(cls))
    hashes = {k:windows.sha256(p) for k,p in deps.items()}
    for k in ('window_loader','DBRL','GraphReasoner','GraphBuilder'):
        require(hashes[k] == timing['source_hashes'][k] == proof['source_hashes'][k], 'Executed source differs: '+k)
    report.update(source_hashes=hashes, input_manifest_sha256=ds.manifest_sha256, metadata_hashes=ds.metadata_hashes,
                  framework_versions=dict(torch=torch.__version__, torch_geometric=torch_geometric.__version__, numpy=np.__version__),
                  reference_framework_versions=timing['framework_versions'])
    torch.set_num_threads(1); torch.manual_seed(proof['seed']); torch.use_deterministic_algorithms(True)
    state = torch.load(deps['initialization'], map_location='cpu', weights_only=True)
    # Reuse the exact full-hierarchy canonical digest without constructing unrelated modules.
    class SavedState:
        def state_dict(self): return state
    require(state_digest(SavedState()) == timing['initialization_state_sha256'] == proof['initialization_state_sha256'],
            'Full initialization state digest differs')
    kwargs = proof['model_parameters']['dbrl']
    require(kwargs == timing['model_parameters']['dbrl'] and kwargs['embedding_dim'] == 64 and
            kwargs['input_dim'] == 6 and kwargs['graph_top_k'] == 15 and kwargs['graph_self_loops'] is True and
            kwargs['graph_symmetric'] is True, 'DBRL contract differs')
    model = DBRL(**kwargs)
    model.load_state_dict({k[len('dbrl.'):]:v for k,v in state.items() if k.startswith('dbrl.')}, strict=True)
    model.eval(); model.requires_grad_(False); before = state_digest(model)
    require(model.graph_reasoner.gat.add_self_loops is True and model.graph_reasoner.gat.edge_dim is None,
            'GAT self-loop/edge-attribute contract differs')
    report.update(initialization_file_sha256=hashes['initialization'], initialization_state_sha256=state_digest(SavedState()),
                  dbrl_state_sha256=before, model_parameters=kwargs,
                  topology_note='Builder directed edges are symmetrized; GAT replaces self loops with one per node. Cosine edge_attr is not passed to GAT.')
    captured = {}

    def capture(module, inputs):
        graph = inputs[0]; b = graph.num_graphs
        h = graph.x.reshape(b,23,64)
        normalized = torch.nn.functional.normalize(h,p=2,dim=-1)
        sim = normalized @ normalized.transpose(-1,-2)
        scores, indices = torch.topk(sim,k=15,dim=-1,largest=True,sorted=True)
        sorted_scores = sim.sort(dim=-1,descending=True).values
        neighbors = torch.zeros((b,23,23),dtype=torch.bool)
        neighbors.scatter_(2,indices,True)
        expected = neighbors | neighbors.transpose(1,2)
        edges = torch.zeros_like(expected)
        src,dst = graph.edge_index
        require(torch.equal(src//23,dst//23), 'Cross-sample graph edge')
        edges[src//23,src%23,dst%23] = True
        require(torch.equal(edges,expected), 'Captured graph does not match independently recovered top-k selection')
        effective = edges.clone()
        effective[:,torch.arange(23),torch.arange(23)] = True
        captured.clear()
        for name,a in dict(H=h, cosine=sim, margin=sorted_scores[:,:,14]-sorted_scores[:,:,15],
                           neighbor_indices=indices, neighbors=neighbors, builder_edges=edges, effective_edges=effective).items():
            captured[name] = a.detach().cpu().numpy().copy()

    handle = model.graph_reasoner.register_forward_pre_hook(capture)
    arrays = {}; repeat_errors = {}; replay_errors = {}
    with np.load(deps['reference_outputs'],allow_pickle=False) as reference:
        try:
            with torch.inference_mode():
                for start in range(0,60,8):
                    batch = ds.get_batch([r['index'] for r in rows[start:start+8]])
                    stop = start+len(batch['X'])
                    for branch in ('X','Y'):
                        x = torch.from_numpy(batch[branch])
                        z = model(x); report['dbrl_forward_calls'] += 1
                        first = dict(captured, Z=z.numpy().copy())
                        repeated = model(x); report['dbrl_forward_calls'] += 1
                        second = dict(captured,Z=repeated.numpy().copy())
                        require(tuple(z.shape) == (len(x),23,64) and not z.requires_grad, 'Invalid DBRL output')
                        for name,a in first.items():
                            require(np.isfinite(a).all(), 'Nonfinite capture: '+name)
                            b = second[name]
                            error = float(np.abs(a.astype(float)-b.astype(float)).max())
                            repeat_errors[name] = max(repeat_errors.get(name,0.),error)
                            require(np.array_equal(a,b), 'Exact instrumentation repeat failed: '+name)
                            arrays.setdefault(branch+'_'+name,[]).append(a)
                        old = reference[branch+'_Z'][start:stop]
                        require(old.shape == first['Z'].shape and np.isfinite(old).all(), 'Invalid saved Z')
                        error = float(np.abs(first['Z'].astype(float)-old.astype(float)).max())
                        replay_errors[branch] = max(replay_errors.get(branch,0.),error)
                        # Attribution concerns tiny changes: require exact observed replay, not merely allclose.
                        require(np.array_equal(first['Z'],old), 'Exact saved-Z replay failed: '+branch)
                    print('Attributed pairs:',stop,'/60',flush=True)
        finally:
            handle.remove()
    arrays = {k:np.concatenate(v) for k,v in arrays.items()}
    result = pair_metrics(rows,arrays)
    for i,row in enumerate(rows):
        if row['event_id'] is None:
            for name in ('H','Z','neighbors','builder_edges','effective_edges'):
                require(np.array_equal(arrays['X_'+name][i],arrays['Y_'+name][i]), 'Quiet/missing control differs: '+name)
    require(state_digest(model) == before and all(p.grad is None for p in model.parameters()), 'State changed or gradients exist')
    ds.verify_unchanged()
    require(all(windows.sha256(p) == hashes[k] for k,p in deps.items()), 'Dependency changed')
    write(out/'pair_attribution.json',result); write(out/'consumed_selection.json',selection)
    np.savez_compressed(out/'dbrl_internal_outputs.npz',**arrays)
    report.update(status='COMPLETE', replay_max_abs_errors=replay_errors, repeat_max_abs_errors=repeat_errors,
                  parameters_and_buffers_unchanged=True, source_files_unchanged=True, controls='PASS', pair_count=60,
                  output_hashes={p.name:windows.sha256(p) for p in out.iterdir() if p.is_file()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project_root_dir'); p.add_argument('--input_dir',required=True,type=Path)
    p.add_argument('--timing_report',required=True,type=Path);p.add_argument('--hierarchy_report',required=True,type=Path)
    args=p.parse_args()
    with open('configs/config.yaml') as f:config=yaml.safe_load(f)
    if args.project_root_dir:config['project_root_dir']=args.project_root_dir
    from src.utils.get_folders_utils import get_processed_folder
    out=Path(get_processed_folder(config))/'v2_validation'/'dbrl_timing_attribution'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status='BLOCKED',protocol='source_dbrl_timing_attribution_v1',configuration=config,
        device='cpu',batch_size=8,dbrl_forward_calls=0,optimizer_steps=0,backward_calls=0,
        scientific_claim_decision='NOT_EVALUATED',limitations=[
        'Single untrained initialization and 18 selected events; no learned behavior-discrimination claim.',
        'Observational stage attribution, not a causal or additive decomposition: no fixed-topology intervention.',
        'Pre/post MSE scales need not be comparable; no amplification ratio is used.',
        'Exact repeatability does not rule out deterministic rounding or near-tie sensitivity.',
        'Effective topology is derived from verified GAT self-loop settings; attention weights are not measured.',
        'Window placement changes pre/post-event composition; within-event placements overlap.',
        'No change to standard model, prior runners, frozen selection or initialization.'])
    try:run(args,config,out,report)
    except Exception as exc:report.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    write(out/'dbrl_attribution_report.json',report)
    print('DBRL attribution:',report['status']);print('Report:',out/'dbrl_attribution_report.json')
    if 'error' in report:print(report['error'])
    return 0 if report['status']=='COMPLETE' else 2


if __name__=='__main__':raise SystemExit(main())
