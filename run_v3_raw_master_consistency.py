"""Chunked read-only raw HA to CUMaster state-code audit; no model execution."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid
import numpy as np
import pandas as pd
import yaml

CHUNK=100000
NS=1_000_000_000


def require(c,m):
    if not c:raise ValueError(m)


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def stamp(p):
    s=p.stat();return [s.st_size,s.st_mtime_ns]


def read_json(p):
    with p.open(encoding='utf-8-sig') as f:return json.load(f)


def write_json(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)


def times(series):
    # Preserve literal clock time; do not introduce a timezone or an offset.
    require(not series.str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$',regex=True).any(),
            'Timezone-bearing timestamps require an explicit reviewed conversion')
    t=pd.to_datetime(series,format='mixed',errors='coerce')
    require(not t.isna().any(),'Unparseable timestamp in selected data')
    return t.astype('datetime64[ns]').astype('int64').to_numpy()


def raw_chunks(p,devices):
    columns=['state_id','entity_id','state','last_updated']
    for chunk in pd.read_csv(p,usecols=columns,dtype=str,keep_default_na=False,na_filter=False,
                             encoding='utf-8-sig',chunksize=CHUNK):
        chunk=chunk[chunk.entity_id.isin(devices)].copy()
        if len(chunk):
            chunk['_t']=times(chunk.last_updated)
            yield chunk


def update_latest(frame,queries,latest,file_name):
    """Order-invariant as-of lookup, retaining conflicting states at identical times."""
    for entity,g in frame.groupby('entity_id',sort=False):
        if entity not in queries:continue
        g=g.sort_values('_t',kind='stable');a=g['_t'].to_numpy()
        states=g['state'].to_numpy();ids=g['state_id'].to_numpy()
        for q in queries[entity]:
            right=int(np.searchsorted(a,q,side='right'))
            if not right:continue
            t=int(a[right-1]);key=(entity,q);old=latest.get(key)
            if old is not None and t<old['raw_timestamp_ns']:continue
            left=int(np.searchsorted(a,t,side='left'))
            vals=set(str(x) for x in states[left:right])
            examples=[dict(file=file_name,state_id=str(ids[j]),state=str(states[j])) for j in range(left,min(right,left+3))]
            if old is None or t>old['raw_timestamp_ns']:
                latest[key]=dict(raw_timestamp_ns=t,states=vals,examples=examples)
            else:
                old['states'].update(vals)
                old['examples']=(old['examples']+examples)[:3]


def iso(t):return str(pd.Timestamp(t))


def finite_number(s):
    try:
        n=float(s)
        return n if np.isfinite(n) else None
    except (ValueError,TypeError):return None


def run(args,config,out,r):
    root=Path(config['project_root_dir'])
    raw=args.raw_dir or root/'data/raw';master=args.master or root/'data/CUMaster.csv'
    registry_path=Path(__file__).resolve().parent/'configs/CU_input_state_registry.json'
    manifest_path=args.input_dir/'manifest.json'
    registry=read_json(registry_path);manifest=read_json(manifest_path)
    devices=[c['name'] for c in registry['channels']]
    require(len(devices)==23 and len(set(devices))==23,'Expected 23 unique channels')
    require(manifest['status']=='COMPLETE' and manifest['split']=='train','Expected completed training manifest')
    require(manifest['devices']==devices,'Manifest/registry channel order differs')
    require(manifest['registry_sha256']==sha(registry_path),'Manifest registry hash differs')
    lo=int(pd.Timestamp(manifest['shards'][0]['first_timestamp']).value)
    hi=int(pd.Timestamp(manifest['shards'][-1]['last_timestamp']).value)
    require(lo<=hi and lo%NS==hi%NS==0,'Invalid training timestamp bounds')
    files=sorted(raw.glob('CUCyPhy*.csv'))
    require(1<=len(files)<=16,'Expected 1–16 CUCyPhy*.csv files')
    require(sum(p.stat().st_size for p in files)<=4_000_000_000,'Raw scan bound exceeded: 4 GB')
    paths=files+[master,registry_path,manifest_path,Path('configs/config.yaml'),Path(__file__)]
    before={str(p.resolve()):stamp(p) for p in paths}
    hashes={}
    start=time.monotonic()
    # Full streaming hashes; this is one byte scan in addition to two CSV passes.
    for p in paths:
        print('Hashing:',p.name,flush=True);hashes[str(p.resolve())]=sha(p)
    r.update(source_hashes=hashes,raw_files=[dict(name=p.name,bytes=p.stat().st_size) for p in files],
        training_bounds=[iso(lo),iso(hi)],registry_state_codes=registry['state_codes'],
        pandas_version=pd.__version__,numpy_version=np.__version__,source_stamps_before=before)
    ticks=[lo+int((hi-lo)//NS)*NS*i//31 for i in range(32)]
    ticks=sorted({int(t//NS*NS) for t in ticks})
    queries={e:set(ticks) for e in devices}
    event_candidates={};counts=Counter()
    # Earliest unknown/unavailable event in each of 8 temporal bins per entity/state.
    for p in files:
        print('Raw pass 1/2:',p.name,flush=True)
        for chunk in raw_chunks(p,devices):
            counts[p.name]+=len(chunk)
            g=chunk[(chunk['_t']>=lo)&(chunk['_t']<=hi)].copy()
            g['_state']=g['state'].str.strip().str.lower()
            g=g[g['_state'].isin(['unknown','unavailable'])]
            for entity,state,t in g[['entity_id','_state','_t']].itertuples(index=False,name=None):
                t=int(t);b=min(7,(t-lo)*8//(hi-lo+1));key=(entity,state,b)
                if key not in event_candidates or t<event_candidates[key]:event_candidates[key]=t
    for (entity,state,b),t in event_candidates.items():
        q=((t+NS-1)//NS)*NS
        if q<=hi:queries[entity].add(q)
    queries={e:sorted(q) for e,q in queries.items()}
    latest={}
    for p in files:
        print('Raw pass 2/2:',p.name,flush=True)
        for chunk in raw_chunks(p,devices):
            chunk=chunk[chunk['_t']<=hi]
            update_latest(chunk,queries,latest,p.name)
    all_q=sorted(set(q for qq in queries.values() for q in qq));qset=set(all_q)
    master_rows={};master_duplicates=Counter();master_conflicts=set()
    print('Scanning master:',master.name,flush=True)
    for chunk in pd.read_csv(master,usecols=['Timestamp']+devices,dtype=str,keep_default_na=False,na_filter=False,
                             encoding='utf-8-sig',chunksize=CHUNK):
        ts=times(chunk['Timestamp'])
        mask=np.isin(ts,all_q)
        for t,row in zip(ts[mask],chunk.loc[mask,devices].itertuples(index=False,name=None)):
            t=int(t);row=tuple(row)
            if t in master_rows:
                master_duplicates[t]+=1
                if master_rows[t]!=row:master_conflicts.add(t)
            else:master_rows[t]=row
    index={e:i for i,e in enumerate(devices)};evidence=[];mapping=defaultdict(Counter);verdict=Counter()
    for entity,qq in queries.items():
        for q in qq:
            item=dict(entity=entity,master_timestamp=iso(q),master_timestamp_ns=q,
                      sampling='fixed_grid' if q in ticks else 'status_event_ceiling')
            if q not in master_rows:item['status']='MASTER_TIMESTAMP_MISSING'
            elif q in master_conflicts:item['status']='MASTER_DUPLICATE_CONFLICT'
            elif (entity,q) not in latest:item['status']='NO_PRIOR_RAW_OBSERVATION'
            else:
                old=latest[(entity,q)];states=sorted(old['states']);v=master_rows[q][index[entity]]
                item.update(raw_timestamp=iso(old['raw_timestamp_ns']),raw_age_seconds=(q-old['raw_timestamp_ns'])/NS,
                    raw_states=states,raw_examples=old['examples'],master_literal=v)
                if len(states)!=1:item['status']='RAW_DUPLICATE_CONFLICT'
                else:
                    raw_state=states[0];token=raw_state.strip().lower();n=finite_number(v)
                    item['status']='COMPARABLE';mapping[(entity,raw_state)][v]+=1
                    if token in ('unknown','unavailable'):
                        expected=float(registry['state_codes'][token]);ok=n is not None and abs(n-expected)<=1e-6
                        reference={'unknown':-9.43,'unavailable':-9.35}[token]
                        item.update(policy_state=token,expected_master_code=expected,policy_match=ok,
                            notebook_saved_code=reference,notebook_saved_code_match=n is not None and abs(n-reference)<=1e-6)
                        verdict[token+('_match' if ok else '_contradiction')]+=1
                    else:
                        rn=finite_number(raw_state)
                        if rn is not None:item['numeric_identity_match']=n is not None and abs(n-rn)<=1e-6
                        # No assumed on/off/open/closed map: report observed associations only.
            evidence.append(item)
    contradictions=sum(v for k,v in verdict.items() if k.endswith('_contradiction'))
    if contradictions:decision='CONTRADICTION_AT_SAMPLED_POINTS'
    elif all(verdict[s+'_match']>0 for s in ('unknown','unavailable')):decision='SUPPORTED_AT_SAMPLED_POINTS'
    else:decision='INSUFFICIENT_EVIDENCE'
    r.update(audit_status='COMPLETE',state_policy_evidence=decision,state_policy_counts=dict(verdict),
        comparison_status_counts=dict(Counter(e['status'] for e in evidence)),query_count=len(evidence),
        selected_raw_rows_per_file=dict(counts),master_duplicate_query_timestamps=len(master_duplicates),
        targeted_status_events=len(event_candidates),clock_alignment='NOT_ESTABLISHED',
        caveats=['As-of reconstruction uses the latest last_updated across supplied files, with no timezone shift.',
        'Conflicting duplicate entity/timestamp states are excluded rather than choosing an arbitrary file order.',
        'No-prior-raw samples are excluded: leading backward-fill is not treated as observation evidence.',
        'Status-event ceiling samples may be superseded by a later event before the sampled second.',
        'Uniform and status-enriched samples are descriptive, not estimates of population error rates.',
        'Constant mappings inferred here are sample associations, not proof of the complete state dictionary.',
        'A contradiction can reflect mapping, clock, export completeness or another master transformation; it does not uniquely identify the cause.',
        'Activity-log alignment is not tested. No policy or model is changed.'])
    for p in paths:require(stamp(p)==before[str(p.resolve())],'Source size/mtime changed: '+str(p))
    require(sorted(raw.glob('CUCyPhy*.csv'))==files,'Raw file inventory changed')
    r['source_size_mtime_recheck_passed']=True
    r['elapsed_seconds']=time.monotonic()-start
    write_json(out/'sample_evidence.json',evidence)
    write_json(out/'observed_state_associations.json',[dict(entity=k[0],raw_state=k[1],master_values=dict(v)) for k,v in sorted(mapping.items())])
    r['output_hashes']={p.name:sha(p) for p in out.iterdir() if p.is_file()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project_root_dir')
    p.add_argument('--input_dir',required=True,type=Path);p.add_argument('--raw_dir',type=Path);p.add_argument('--master',type=Path)
    args=p.parse_args()
    with open('configs/config.yaml') as f:config=yaml.safe_load(f)
    if args.project_root_dir:config['project_root_dir']=args.project_root_dir
    from src.utils.get_folders_utils import get_processed_folder
    out=Path(get_processed_folder(config))/'v2_validation'/'raw_master_consistency'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    r=dict(audit_status='BLOCKED',model_forwards=0,optimizer_steps=0,configuration=config)
    try:run(args,config,out,r)
    except Exception as exc:r.update(audit_status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    write_json(out/'raw_master_consistency_report.json',r)
    print('Raw/master audit:',r['audit_status']);print('State policy:',r.get('state_policy_evidence','NOT_ASSESSED'))
    print('Report:',out/'raw_master_consistency_report.json')
    if 'error' in r:print(r['error'])
    return 0 if r['audit_status']=='COMPLETE' else 2

if __name__=='__main__':raise SystemExit(main())
