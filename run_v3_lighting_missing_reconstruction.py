"""Read-only, training-bounded reconstruction of two lighting channels' missing values."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
import yaml

CHANNELS = ('light.hue_light_bulb_pid_040', 'light.053_lumiman_smart_bulb_2')
NS = 1_000_000_000
CHUNK = 100_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def stamp(path):
    s = Path(path).stat()
    return [s.st_size, s.st_mtime_ns]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def times(series):
    require(not series.str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$', regex=True).any(),
            'Timezone-bearing timestamps require reviewed conversion')
    t = pd.to_datetime(series, format='mixed', errors='coerce')
    require(not t.isna().any(), 'Unparseable timestamp')
    return t.astype('datetime64[ns]').astype('int64').to_numpy()


def default_missing(tokens):
    # Ask the installed public CSV parser, preserving literal strings separately.
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['id', 'state'])
    for i, token in enumerate(tokens):
        writer.writerow([i, token])
    buf.seek(0)
    parsed = pd.read_csv(buf, dtype={'state': str}, keep_default_na=True)
    return {token: bool(pd.isna(v)) for token, v in zip(tokens, parsed.state)}


def number(token, codes):
    # Current master policy comparison, NOT the unavailable historical state map.
    if token in {'on', 'off'}:
        return float(token == 'on')
    if token in codes:
        return float(codes[token])
    try:
        n = float(token)
        return n if np.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def reconstruct(events, lo, hi, queries, codes):
    """Resample first; propagate donor indices, not assumed numeric state codes.

    Each event has timestamp_ns, states, examples. Duplicate conflicting states
    are barriers: they may be carried but never count as explained values.
    """
    require(lo <= hi and lo % NS == hi % NS == 0, 'Invalid grid bounds')
    require((hi - lo) // NS + 1 <= 3_000_000, 'Training grid exceeds 3 million seconds')
    events = sorted(events, key=lambda e: e['timestamp_ns'])
    require(len({e['timestamp_ns'] for e in events}) == len(events), 'Events must be grouped by timestamp')
    tokens = sorted({s for e in events for s in e['states']})
    na = default_missing(tokens) if tokens else {}
    missing = np.array([len(e['states']) == 1 and na[e['states'][0]] for e in events], dtype=bool)
    ts = np.array([e['timestamp_ns'] for e in events], dtype=np.int64)
    grid = np.arange(lo, hi + NS, NS, dtype=np.int64)
    latest = np.searchsorted(ts, grid, side='right') - 1
    valid = latest >= 0
    if len(events):
        valid &= ~missing[np.maximum(latest, 0)]
    # Conflicts are retained as uncertain donors, never skipped as if known missing.
    donor_pos = np.where(valid, np.arange(len(grid)), -1)
    donor_pos = np.maximum.accumulate(donor_pos)
    nonmissing_positions = np.flatnonzero(valid)
    if len(nonmissing_positions):
        donor_pos[donor_pos < 0] = nonmissing_positions[0]  # explicit leading bfill
    nonmissing_events = np.flatnonzero(~missing)
    results = []

    def describe(i):
        if i < 0:
            return None
        e = events[int(i)]
        return dict(timestamp=str(pd.Timestamp(e['timestamp_ns'])),
                    timestamp_ns=e['timestamp_ns'], states=e['states'], examples=e['examples'])

    for q in queries:
        require(lo <= q <= hi and q % NS == 0, 'Query outside training grid')
        k = (q - lo) // NS
        raw_i = int(latest[k])
        j = int(donor_pos[k])
        donor_i = int(latest[j]) if j >= 0 else -1
        p = np.searchsorted(ts[nonmissing_events], q, side='right') - 1
        previous_i = int(nonmissing_events[p]) if p >= 0 else -1
        row = dict(timestamp=str(pd.Timestamp(q)), timestamp_ns=q,
                   latest_raw=describe(raw_i), latest_preceding_nonmissing_raw=describe(previous_i),
                   fill_donor=describe(donor_i), reconstructed_value=None)
        if donor_i < 0:
            row['fill_method'] = 'UNRESOLVED_NO_TRAINING_GRID_DONOR'
        else:
            row['fill_method'] = 'BFILL' if j > k else ('NONE' if valid[k] else 'FFILL')
            row['donor_grid_timestamp'] = str(pd.Timestamp(int(grid[j])))
            row['donor_raw_age_seconds'] = (q - ts[donor_i]) / NS
            row['uses_future_donor'] = j > k
            if len(events[donor_i]['states']) != 1:
                row['reconstruction_status'] = 'UNRESOLVED_DUPLICATE_CONFLICT'
            else:
                row['reconstructed_value'] = number(events[donor_i]['states'][0], codes)
                row['reconstruction_status'] = ('MAPPED' if row['reconstructed_value'] is not None
                                                 else 'UNRESOLVED_STATE_MAPPING')
        row.setdefault('reconstruction_status', 'UNRESOLVED_NO_DONOR')
        row['latest_raw_parses_as_missing'] = bool(raw_i >= 0 and missing[raw_i])
        row['latest_nonmissing_raw_is_fill_donor'] = previous_i == donor_i if donor_i >= 0 else None
        results.append(row)
    return results, na


def expected_hash(prior, basename):
    values = [v for p, v in prior['source_hashes'].items() if Path(p).name == basename]
    require(len(values) == 1, 'Prior source hash missing or ambiguous: ' + basename)
    return values[0]


def run(args, config, out, report):
    prior = read_json(args.audit_report)
    require(prior['audit_status'] == 'COMPLETE', 'Prior raw/master audit is not complete')
    evidence_path = args.audit_report.parent / 'sample_evidence.json'
    require(sha(evidence_path) == prior['output_hashes']['sample_evidence.json'], 'Prior evidence hash differs')
    evidence = read_json(evidence_path)
    selected = [e for e in evidence if e['entity'] in CHANNELS and e.get('raw_states') == ['NULL']
                and e['status'] == 'COMPARABLE']
    require(0 < len(selected) <= 128, 'Expected 1–128 frozen NULL samples')
    require({e['entity'] for e in selected} == set(CHANNELS), 'Both lighting channels must have NULL samples')
    require(len({(e['entity'], e['master_timestamp_ns']) for e in selected}) == len(selected), 'Duplicate sample keys')
    manifest_path = args.input_dir / 'manifest.json'
    manifest = read_json(manifest_path)
    require(sha(manifest_path) == expected_hash(prior, 'manifest.json'), 'Training manifest changed')
    require(manifest['status'] == 'COMPLETE' and manifest['split'] == 'train', 'Expected completed train manifest')
    registry_path = Path(__file__).resolve().parent / 'configs/CU_input_state_registry.json'
    registry = read_json(registry_path)
    require(sha(registry_path) == expected_hash(prior, registry_path.name) == manifest['registry_sha256'], 'Registry hash changed')
    require(manifest['devices'] == [c['name'] for c in registry['channels']], 'Channel order mismatch')
    lo, hi = [int(pd.Timestamp(x).value) for x in prior['training_bounds']]
    require(lo == pd.Timestamp(manifest['shards'][0]['first_timestamp']).value and
            hi == pd.Timestamp(manifest['shards'][-1]['last_timestamp']).value, 'Training bounds mismatch')
    require(0 <= (hi - lo) // NS < 3_000_000, 'Training bound exceeds limit')
    root = Path(config['project_root_dir'])
    raw = args.raw_dir or root / 'data/raw'
    master = args.master or root / 'data/CUMaster.csv'
    files = sorted(raw.glob('CUCyPhy*.csv'))
    require([p.name for p in files] == sorted(f['name'] for f in prior['raw_files']), 'Raw inventory changed')
    require(1 <= len(files) <= 16 and sum(p.stat().st_size for p in files) <= 4_000_000_000, 'Raw scan bound exceeded')
    paths = files + [master, manifest_path, registry_path, args.audit_report, evidence_path,
                     Path('configs/config.yaml'), Path(__file__)]
    stamps = {str(p.resolve()): stamp(p) for p in paths}
    hashes = {}
    for p in paths:
        print('Hashing:', p.name, flush=True)
        hashes[str(p.resolve())] = sha(p)
        if p in files or p == master:
            require(hashes[str(p.resolve())] == expected_hash(prior, p.name), 'Source content changed: ' + p.name)
    report.update(source_hashes=hashes, source_stamps_before=stamps, training_bounds=prior['training_bounds'],
                  selected_sample_count=len(selected), selected_counts=dict(Counter(e['entity'] for e in selected)),
                  pandas_version=pd.__version__, numpy_version=np.__version__)
    # Only two channels are retained in memory; overlapping exports are deduplicated.
    grouped = {e: {} for e in CHANNELS}
    retained_count = 0
    for p in files:
        print('Scanning raw:', p.name, flush=True)
        for chunk in pd.read_csv(p, usecols=['entity_id', 'state_id', 'state', 'last_updated'],
                                 dtype=str, keep_default_na=False, na_filter=False,
                                 encoding='utf-8-sig', chunksize=CHUNK):
            chunk = chunk[chunk.entity_id.isin(CHANNELS)].copy()
            if chunk.empty:
                continue
            chunk['_t'] = times(chunk.last_updated)
            chunk = chunk[chunk['_t'] <= hi]  # no future-period donor values
            retained_count += len(chunk)
            require(retained_count <= 1_000_000, 'Two-channel raw row bound exceeded')
            for entity, sid, state, t in chunk[['entity_id', 'state_id', 'state', '_t']].itertuples(index=False, name=None):
                event = grouped[entity].setdefault(int(t), dict(timestamp_ns=int(t), states=set(), examples=[]))
                event['states'].add(state)
                if len(event['examples']) < 3:
                    event['examples'].append(dict(file=p.name, state_id=sid, state=state))
    qset = sorted({e['master_timestamp_ns'] for e in selected})
    actual = {}
    for chunk in pd.read_csv(master, usecols=['Timestamp', *CHANNELS], dtype=str, keep_default_na=False,
                             na_filter=False, encoding='utf-8-sig', chunksize=CHUNK):
        ts = times(chunk.Timestamp)
        mask = np.isin(ts, qset)
        for t, values in zip(ts[mask], chunk.loc[mask, list(CHANNELS)].itertuples(index=False, name=None)):
            require(int(t) not in actual, 'Duplicate master query timestamp')
            actual[int(t)] = values
    output = []
    details = {}
    for channel_index, entity in enumerate(CHANNELS):
        events = list(grouped[entity].values())
        require(len(events) <= 250_000, 'Unique two-channel event bound exceeded')
        for e in events:
            e['states'] = sorted(e['states'])
        targets = sorted([e for e in selected if e['entity'] == entity], key=lambda e: e['master_timestamp_ns'])
        rows, na = reconstruct(events, lo, hi, [e['master_timestamp_ns'] for e in targets], registry['state_codes'])
        details[entity] = dict(unique_events=len(events), conflicting_event_timestamps=sum(len(e['states']) > 1 for e in events),
                               literal_default_na_classification=na)
        for row, original in zip(rows, targets):
            q = row['timestamp_ns']
            require(q in actual, 'Master query timestamp missing')
            value = actual[q][channel_index]
            require(value == original['master_literal'], 'Master sample differs from prior audit')
            require(row['latest_raw'] is not None and row['latest_raw']['states'] == ['NULL'] and
                    row['latest_raw']['timestamp_ns'] == pd.Timestamp(original['raw_timestamp']).value,
                    'Latest raw record differs from prior evidence')
            row.update(entity=entity, master_literal=value)
            observed = number(value, {})
            predicted = row['reconstructed_value']
            row['matches_master'] = (abs(predicted - observed) <= 1e-6
                                     if predicted is not None and observed is not None else None)
            output.append(row)
    for p in paths:
        require(stamp(p) == stamps[str(p.resolve())], 'Source size/mtime changed: ' + str(p))
    require(sorted(raw.glob('CUCyPhy*.csv')) == files, 'Raw inventory changed during scan')
    write_json(out / 'lighting_sample_reconstruction.json', output)
    write_json(out / 'lighting_event_summary.json', details)
    summary = {}
    for e in CHANNELS:
        rows = [r for r in output if r['entity'] == e]
        summary[e] = dict(samples=len(rows), matches=sum(r['matches_master'] is True for r in rows),
                          mismatches=sum(r['matches_master'] is False for r in rows),
                          unresolved=sum(r['matches_master'] is None for r in rows),
                          fill_methods=dict(Counter(r['fill_method'] for r in rows)),
                          unique_donor_events=len({r['fill_donor']['timestamp_ns'] for r in rows if r['fill_donor']}))
    report.update(audit_status='COMPLETE', per_channel=summary, source_size_mtime_recheck_passed=True,
                  output_hashes={p.name: sha(p) for p in out.iterdir() if p.is_file()},
                  conclusion=('CONSISTENT_WITH_RESAMPLE_THEN_FILL_AT_ALL_SAMPLED_POINTS'
                              if all(r['matches_master'] is True for r in output) else
                              'MISMATCH_OR_UNRESOLVED_SAMPLES_REQUIRE_REVIEW'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project_root_dir')
    p.add_argument('--input_dir', required=True, type=Path)
    p.add_argument('--audit_report', required=True, type=Path)
    p.add_argument('--raw_dir', type=Path)
    p.add_argument('--master', type=Path)
    args = p.parse_args()
    with open('configs/config.yaml') as f:
        config = yaml.safe_load(f)
    if args.project_root_dir:
        config['project_root_dir'] = args.project_root_dir
    from src.utils.get_folders_utils import get_processed_folder
    out = Path(get_processed_folder(config)) / 'v2_validation' / 'lighting_missing_reconstruction' / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    report = dict(audit_status='BLOCKED', model_forwards=0, optimizer_steps=0, configuration=config,
                  limitations=[
                      'Reconstruction hypothesis, not proof of the historically executed notebook or state dictionary.',
                      'Uses literal last_updated clocks unchanged; activity-log clock equivalence remains unestablished.',
                      'Current on/off=1/0 and registry status codes are comparison assumptions, not the unavailable CUStatesMap.',
                      'CSV missing tokens use the installed pandas default parser; original runtime is unknown.',
                      'Duplicate conflicting raw states remain unresolved barriers, rather than choosing file order.',
                      'Leading backward fill is restricted to the training grid and explicitly flagged as future-dependent.',
                      'No policy change, input rewrite, training or population-wide missing-value claim.'])
    try:
        run(args, config, out, report)
    except Exception as exc:
        report.update(audit_status='BLOCKED', error=f'{type(exc).__name__}: {exc}')
    write_json(out / 'lighting_missing_reconstruction_report.json', report)
    print('Lighting reconstruction:', report['audit_status'])
    print('Report:', out / 'lighting_missing_reconstruction_report.json')
    if 'error' in report:
        print(report['error'])
    return 0 if report['audit_status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
