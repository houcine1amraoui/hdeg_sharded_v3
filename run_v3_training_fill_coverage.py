"""Read-only training coverage audit of resample-then-fill across 23 channels."""
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


CATEGORIES = ['no_prior_record', 'explicit_missing', 'unknown', 'unavailable',
              'ordinary_mapped_state', 'unmapped_state', 'duplicate_conflict']


def affected_windows(mask, width):
    if len(mask) < width:
        return 0
    c = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
    return int(np.count_nonzero(c[width:] - c[:-width]))


def coverage(mask, lo, window):
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    lengths = ends-starts
    largest = np.argsort(-lengths, kind='stable')[:10]
    return dict(grid_seconds=int(mask.sum()), fraction=float(mask.mean()),
                interval_count=len(starts), longest_interval_seconds=int(lengths.max()) if len(lengths) else 0,
                affected_windows=affected_windows(mask, window),
                affected_pairs=affected_windows(mask, window+1),
                longest_intervals=[dict(start=str(pd.Timestamp(lo+int(starts[i])*NS)),
                    end_exclusive=str(pd.Timestamp(lo+int(ends[i])*NS)), seconds=int(lengths[i])) for i in largest])


def age_summary(values):
    if not len(values):
        return dict(count=0)
    return dict(count=len(values), min_seconds=float(np.min(values)),
                median_seconds=float(np.median(values)), p95_seconds=float(np.quantile(values,.95)),
                max_seconds=float(np.max(values)),
                at_least_seconds={str(s):int(np.count_nonzero(values>=s)) for s in [30,300,3600,86400]})


def channel_grid(events, lo, hi, codes, window):
    """Events grouped by nanosecond timestamp; conflicts remain uncertain barriers."""
    require(lo % NS == hi % NS == 0 and 0 <= (hi-lo)//NS < 3_000_000, 'Grid bound exceeded')
    events = sorted(events, key=lambda e:e['t'])
    require(len({e['t'] for e in events})==len(events), 'Duplicate event groups')
    literals = sorted({s for e in events for s in e['states']})
    na = default_missing(literals) if literals else {}
    kind, values = [], []
    for e in events:
        states = e['states']
        if len(states)>1:
            k,v = 6,None
        else:
            s = states[0]; v = number(s,codes)
            k = 1 if na[s] else 2 if s=='unknown' else 3 if s=='unavailable' else 4 if v is not None else 5
        kind.append(k); values.append(np.nan if v is None else v)
    # Index 0 is a sentinel for absence of any preceding record.
    et = np.array([e['t'] for e in events], dtype=np.int64)
    kinds = np.array([0]+kind,dtype=np.uint8)
    vals = np.array([np.nan]+values,dtype=np.float64)
    grid = np.arange(lo,hi+NS,NS,dtype=np.int64)
    latest = np.searchsorted(et,grid,side='right')
    categories = kinds[latest]
    nonmissing = (categories!=0)&(categories!=1)
    donor_pos = np.maximum.accumulate(np.where(nonmissing,np.arange(len(grid)),-1))
    first = np.flatnonzero(nonmissing)
    leading = donor_pos<0
    if len(first):
        donor_pos[leading]=first[0]
    donor = np.zeros(len(grid),dtype=np.int64)
    has_donor = donor_pos>=0
    donor[has_donor]=latest[donor_pos[has_donor]]
    predicted = vals[donor]
    flags = dict(explicit_missing=categories==1, no_prior_record=categories==0,
                 forward_filled_explicit_missing=(categories==1)&~leading,
                 leading_backward_fill=leading&has_donor,
                 no_training_grid_donor=~has_donor,
                 unresolved_reconstruction=~np.isfinite(predicted),
                 explicit_missing_with_mapped_fill=(categories==1)&np.isfinite(predicted))
    flags['explicit_missing_or_no_prior_record'] = flags['explicit_missing']|flags['no_prior_record']
    stats = {name:coverage(mask,lo,window) for name,mask in flags.items()}
    donor_ages = np.zeros(len(grid),dtype=np.float64)
    if len(et):
        donor_ages[has_donor]=(grid[has_donor]-et[donor[has_donor]-1])/NS
    source_ages = (grid[latest>0]-et[latest[latest>0]-1])/NS if len(et) else np.array([])
    summary = dict(unique_events=len(events), conflicting_event_timestamps=kind.count(6),
       raw_literal_default_na_classification={s:b for s,b in na.items() if b},
       raw_state_category_counts=dict(Counter(CATEGORIES[k] for k in kind)),
       grid_category_seconds={name:int(np.count_nonzero(categories==i)) for i,name in enumerate(CATEGORIES)},
       coverage=stats, source_record_age_seconds=age_summary(source_ages),
       ordinary_mapped_state_record_age_seconds=age_summary(source_ages[categories[latest>0]==4]),
       ffill_donor_age_seconds=age_summary(donor_ages[flags['forward_filled_explicit_missing']&has_donor]),
       leading_bfill_future_donor_lead_seconds=age_summary(-donor_ages[flags['leading_backward_fill']]),
       unique_explicit_missing_fill_donor_events=int(len(np.unique(donor[(categories==1)&has_donor]))))
    return predicted,categories,flags,summary


def expected_hash(prior, name):
    h=[v for p,v in prior['source_hashes'].items() if Path(p).name==name]
    require(len(h)==1,'Missing or ambiguous prior source: '+name)
    return h[0]


def run(args, config, out, report):
    prior=read_json(args.audit_report)
    require(prior['audit_status']=='COMPLETE','Prior audit incomplete')
    manifest_path=args.input_dir/'manifest.json'
    registry_path=Path(__file__).resolve().parent/'configs/CU_input_state_registry.json'
    manifest,registry=read_json(manifest_path),read_json(registry_path)
    require(sha(manifest_path)==expected_hash(prior,'manifest.json'),'Training manifest changed')
    require(sha(registry_path)==expected_hash(prior,registry_path.name)==manifest['registry_sha256'],'Registry changed')
    require(manifest['status']=='COMPLETE' and manifest['split']=='train','Expected completed training manifest')
    devices=[c['name'] for c in registry['channels']]
    require(len(devices)==len(set(devices))==23 and devices==manifest['devices'],'23-channel order mismatch')
    lo,hi=[int(pd.Timestamp(t).value) for t in prior['training_bounds']]
    require(lo==pd.Timestamp(manifest['shards'][0]['first_timestamp']).value and
            hi==pd.Timestamp(manifest['shards'][-1]['last_timestamp']).value,'Training bounds changed')
    require(lo%NS==hi%NS==0 and 0<=(hi-lo)//NS<3_000_000,'Training grid bound exceeded')
    n=(hi-lo)//NS+1
    require(config['preprocessing']['window_size']==30,'This protocol fixes 30-second windows')
    window=30
    root=Path(config['project_root_dir']);raw=args.raw_dir or root/'data/raw';master=args.master or root/'data/CUMaster.csv'
    files=sorted(raw.glob('CUCyPhy*.csv'))
    require([p.name for p in files]==sorted(f['name'] for f in prior['raw_files']),'Raw inventory changed')
    require(1<=len(files)<=16 and sum(p.stat().st_size for p in files)<=4_000_000_000,'Raw scan bound exceeded')
    paths=files+[master,manifest_path,registry_path,args.audit_report,Path('configs/config.yaml'),Path(__file__)]
    stamps={str(p.resolve()):stamp(p) for p in paths};hashes={}
    for p in paths:
        print('Hashing:',p.name,flush=True); hashes[str(p.resolve())]=sha(p)
        if p in files or p==master:
            require(hashes[str(p.resolve())]==expected_hash(prior,p.name),'Source changed: '+p.name)
    report.update(source_hashes=hashes,source_stamps_before=stamps,training_bounds=prior['training_bounds'],
                  devices=devices,grid_seconds=n,possible_windows=max(0,n-window+1),possible_pairs=max(0,n-window),
                  pandas_version=pd.__version__,numpy_version=np.__version__)
    groups={e:{} for e in devices}; raw_count=0; unique_count=0
    for p in files:
        print('Scanning raw:',p.name,flush=True)
        for chunk in pd.read_csv(p,usecols=['entity_id','state','last_updated'],dtype=str,
                         keep_default_na=False,na_filter=False,encoding='utf-8-sig',chunksize=CHUNK):
            chunk=chunk[chunk.entity_id.isin(devices)].copy()
            if chunk.empty:continue
            chunk['_t']=times(chunk.last_updated);chunk=chunk[chunk['_t']<=hi]
            raw_count+=len(chunk);require(raw_count<=3_000_000,'Selected raw row bound exceeded')
            for e,s,t in chunk[['entity_id','state','_t']].itertuples(index=False,name=None):
                t=int(t)
                if t not in groups[e]:
                    groups[e][t]=set();unique_count+=1
                    require(unique_count<=500_000,'Unique selected event bound exceeded')
                groups[e][t].add(s)
    # Bounded RAM, no copies of source data or intermediate arrays written to disk.
    predictions=np.empty((len(devices),n),dtype=np.float64)
    categories=np.empty((len(devices),n),dtype=np.uint8)
    unions={};per_channel={}
    for i,e in enumerate(devices):
        print('Reconstructing:',e,flush=True)
        events=[dict(t=t,states=sorted(states)) for t,states in groups.pop(e).items()]
        pred,cat,flags,summary=channel_grid(events,lo,hi,registry['state_codes'],window)
        predictions[i]=pred;categories[i]=cat;per_channel[e]=summary
        for name,mask in flags.items():
            if name not in unions:unions[name]=mask.copy()
            else:unions[name]|=mask
    counters={e:{name:Counter() for name in ['all',*CATEGORIES]} for e in devices}
    mismatch_examples=[];seen=np.zeros(n,dtype=bool)
    print('Comparing training master grid',flush=True)
    for chunk in pd.read_csv(master,usecols=['Timestamp',*devices],dtype=str,keep_default_na=False,
                             na_filter=False,encoding='utf-8-sig',chunksize=CHUNK):
        ts=times(chunk.Timestamp);mask=(ts>=lo)&(ts<=hi);ts=ts[mask]
        if not len(ts):continue
        require(np.all(ts%NS==0),'Non-second master training timestamp')
        ix=(ts-lo)//NS
        require(len(np.unique(ix))==len(ix) and not seen[ix].any(),'Duplicate master training timestamp')
        seen[ix]=True
        for i,e in enumerate(devices):
            actual=pd.to_numeric(chunk.loc[mask,e],errors='coerce').to_numpy(dtype=float)
            predicted=predictions[i,ix];cat=categories[i,ix]
            comparable=np.isfinite(actual)&np.isfinite(predicted)
            match=comparable & np.isclose(actual,predicted,atol=1e-6,rtol=0)
            mismatch=comparable&~match
            for name,sel in [('all',np.ones(len(ix),dtype=bool))]+[(name,cat==k) for k,name in enumerate(CATEGORIES)]:
                c=counters[e][name];c['rows']+=int(sel.sum());c['matches']+=int((sel&match).sum())
                c['mismatches']+=int((sel&mismatch).sum());c['unresolved']+=int((sel&~comparable).sum())
            for j in np.flatnonzero(mismatch)[:max(0,100-len(mismatch_examples))]:
                mismatch_examples.append(dict(entity=e,timestamp=str(pd.Timestamp(int(ts[j]))),
                     raw_category=CATEGORIES[int(cat[j])],predicted=float(predicted[j]),master=float(actual[j])))
    require(seen.all(),'Master training grid incomplete: '+str(int((~seen).sum()))+' seconds absent')
    for e in devices:per_channel[e]['master_comparison']={k:dict(v) for k,v in counters[e].items()}
    for p in paths:require(stamp(p)==stamps[str(p.resolve())],'Source size/mtime changed: '+str(p))
    require(sorted(raw.glob('CUCyPhy*.csv'))==files,'Raw inventory changed during scan')
    write_json(out/'channel_fill_coverage.json',per_channel)
    write_json(out/'fill_mismatch_examples.json',mismatch_examples)
    totals={k:sum(counters[e]['all'][k] for e in devices) for k in ['rows','matches','mismatches','unresolved']}
    report.update(audit_status='COMPLETE',source_size_mtime_recheck_passed=True,selected_raw_rows=raw_count,
                  unique_selected_events=unique_count,any_channel_coverage={k:coverage(v,lo,window) for k,v in unions.items()},
                  master_comparison_totals=totals,
                  conclusion='FULL_TRAINING_GRID_RECONSTRUCTION_MATCH' if totals['mismatches']==totals['unresolved']==0 else
                  'RECONSTRUCTION_HAS_MISMATCHES_OR_UNRESOLVED_VALUES',
                  output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project_root_dir');p.add_argument('--input_dir',required=True,type=Path)
    p.add_argument('--audit_report',required=True,type=Path)
    p.add_argument('--raw_dir',type=Path);p.add_argument('--master',type=Path)
    args=p.parse_args()
    with open('configs/config.yaml') as f:config=yaml.safe_load(f)
    if args.project_root_dir:config['project_root_dir']=args.project_root_dir
    from src.utils.get_folders_utils import get_processed_folder
    out=Path(get_processed_folder(config))/'v2_validation'/'training_fill_coverage'/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    report=dict(audit_status='BLOCKED',configuration=config,model_forwards=0,optimizer_steps=0,
       protocol=dict(grid_seconds=1,window_size=30,pair_union_seconds=31,stride=1,
                     comparison_atol=1e-6,comparison_rtol=0),
       limitations=[
        'Reconstruction hypothesis, not certification of the historically executed pipeline or state dictionary.',
        'Explicit missing means installed pandas default-NA literals in the latest raw record; silence alone is not missing.',
        'Age thresholds are descriptive, not expiry rules or proof of device state invalidity.',
        'Ordinary mapped state means numeric/on/off and does not assert registry-specific physical validity.',
        'No prior record means no preceding record in supplied exports, not proof of device absence.',
        'Unknown and unavailable remain nonmissing states; their verified registry codes are retained.',
        'Conflicting raw states are uncertain barriers; their propagation is unresolved, not arbitrarily filled.',
        'Leading backward fill uses future donors within training only; later-period donors are not used.',
        'Affected window counts use the complete one-second grid, not an independently verified downstream sampler.',
        'Whole-file hashes and streaming timestamp parsing read full sources; only training and preceding context values enter analysis.',
        'Activity-log clock alignment remains unestablished. No input policy or model changes.'])
    try:run(args,config,out,report)
    except Exception as exc:report.update(audit_status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    write_json(out/'training_fill_coverage_report.json',report)
    print('Training fill coverage:',report['audit_status']);print('Report:',out/'training_fill_coverage_report.json')
    if 'error' in report:print(report['error'])
    return 0 if report['audit_status']=='COMPLETE' else 2


if __name__=='__main__':raise SystemExit(main())
