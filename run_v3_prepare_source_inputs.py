"""Prepare causal source-aware training rows without modifying existing artifacts."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
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


POLICY='cu_source_input_state_v1'
STATUS={'valid':0,'unknown':1,'unavailable':2,'missing':3,'invalid':4}
FEATURES=['value',*STATUS]


def classify(token,kind):
    if token in ('','NULL'):return 0.,3
    if token=='unknown':return 0.,1
    if token=='unavailable':return 0.,2
    if token!=token.strip():return 0.,4
    if kind=='binary' and token in ('on','off'):return float(token=='on'),0
    try:v=float(token)
    except (ValueError,TypeError):return 0.,4
    if not np.isfinite(v) or (kind=='binary' and v not in (0.,1.)):return 0.,4
    return v,0


def causal_rows(events,grid,kind):
    events=sorted(events.items())
    require(all(len(states)==1 for _,states in events),'Conflicting raw entity/timestamp states')
    ts=np.array([t for t,_ in events],dtype=np.int64)
    parsed=[classify(next(iter(states)),kind) for _,states in events]
    vals=np.array([0.]+[v for v,s in parsed],dtype=np.float64)
    states=np.array([3]+[s for v,s in parsed],dtype=np.uint8)
    ix=np.searchsorted(ts,grid,side='right')
    if len(ts):require(np.all(ts[ix[ix>0]-1]<=grid[ix>0]),'Future source record used')
    return vals[ix],states[ix],ix==0


def normalize(values,status,kind):
    valid=status==0;eligible=int(valid.sum())
    result=np.zeros(len(values),dtype=np.float32)
    meta=dict(type=kind,eligible_grid_rows=eligible)
    if kind=='binary':
        result[valid]=values[valid];meta['transform']='identity'
    else:
        require(eligible>0,'Continuous channel has no valid training values')
        lo,hi=float(values[valid].min()),float(values[valid].max())
        denominator=hi-lo if hi>lo else 1.
        result[valid]=((values[valid]-lo)/denominator).astype(np.float32)
        meta.update(transform='minmax',minimum=lo,maximum=hi,denominator=denominator,constant=hi==lo,clip=False)
    require(np.isfinite(result).all(),'Nonfinite normalized output')
    return result,meta


def expected_hash(prior,name):
    values=[v for p,v in prior['source_hashes'].items() if Path(p).name==name]
    require(len(values)==1,'Missing or ambiguous source hash: '+name)
    return values[0]


def verify_shards(out,entries,values,status,grid):
    for e in entries:
        path=out/e['file'];require(sha(path)==e['sha256'],'Shard hash mismatch')
        with np.load(path,allow_pickle=False) as z:
            a,b=e['start'],e['stop']
            require(set(z.files)=={'values','status','timestamps'},'Shard keys differ')
            require(z['values'].dtype==np.float32 and z['status'].dtype==np.uint8 and
                    z['timestamps'].dtype==np.dtype('datetime64[ns]'),'Shard dtype mismatch')
            require(np.array_equal(z['values'],values[a:b]) and np.array_equal(z['status'],status[a:b]) and
                    np.array_equal(z['timestamps'].view('i8'),grid[a:b]),'Shard readback differs')
    # Bounded cross-shard pair readback using only saved artifacts.
    n=len(grid);indices=set(np.linspace(0,n-31,min(16,n-30),dtype=np.int64).tolist())
    indices.update(max(0,min(n-31,e['start']-15)) for e in entries[1:17])
    indices=sorted(indices)[:32]
    for i in indices:
        vv=[];ss=[];tt=[]
        for e in entries:
            a,b=max(i,e['start']),min(i+31,e['stop'])
            if a>=b:continue
            with np.load(out/e['file'],allow_pickle=False) as z:
                sl=slice(a-e['start'],b-e['start'])
                vv.append(z['values'][sl]);ss.append(z['status'][sl]);tt.append(z['timestamps'][sl])
        v,s,t=np.concatenate(vv),np.concatenate(ss),np.concatenate(tt)
        f=np.concatenate([v[...,None],np.eye(5,dtype=np.float32)[s]],axis=-1)
        require(f.shape==(31,23,6) and np.all(f[...,1:].sum(-1)==1),'Six-feature pair invariant failed')
        require(np.array_equal(v,values[i:i+31]) and np.array_equal(s,status[i:i+31]) and
                np.array_equal(t.view('i8'),grid[i:i+31]),'Saved pair differs from causal rows')
    return dict(all_shards_read_back=True,pairs_verified=len(indices),pair_indices=indices,
                X_shape=[len(indices),30,23,6],Y_shape=[len(indices),30,23,6])


def run(args,config,out,report):
    prior=read_json(args.coverage_report)
    require(prior['audit_status']=='COMPLETE' and prior['conclusion']=='FULL_TRAINING_GRID_RECONSTRUCTION_MATCH',
            'Expected successful full training coverage audit')
    coverage_path=args.coverage_report.parent/'channel_fill_coverage.json'
    require(sha(coverage_path)==prior['output_hashes']['channel_fill_coverage.json'],'Coverage evidence hash changed')
    coverage=read_json(coverage_path)
    registry_path=Path(__file__).resolve().parent/'configs/CU_input_state_registry.json'
    manifest_path=args.reference_input_dir/'manifest.json'
    registry,old=read_json(registry_path),read_json(manifest_path)
    require(sha(registry_path)==expected_hash(prior,registry_path.name)==old['registry_sha256'],'Registry changed')
    require(sha(manifest_path)==expected_hash(prior,'manifest.json'),'Reference training manifest changed')
    require(old['status']=='COMPLETE' and old['split']=='train','Expected complete reference train manifest')
    devices=[c['name'] for c in registry['channels']];kinds=[c['type'] for c in registry['channels']]
    require(devices==old['devices']==prior['devices'] and len(set(devices))==23,'Channel order mismatch')
    require(kinds.count('binary')==18 and kinds.count('continuous')==5,'Channel types mismatch')
    require(config['preprocessing']['window_size']==30,'Expected window size 30')
    lo,hi=[int(pd.Timestamp(t).value) for t in prior['training_bounds']]
    require(lo%NS==hi%NS==0 and 30<=(hi-lo)//NS<3_000_000,'Training bounds invalid')
    require(lo==pd.Timestamp(old['shards'][0]['first_timestamp']).value and hi==pd.Timestamp(old['shards'][-1]['last_timestamp']).value,'Reference bounds mismatch')
    grid=np.arange(lo,hi+NS,NS,dtype=np.int64)
    require(len(grid)==prior['grid_seconds'],'Coverage grid differs')
    root=Path(config['project_root_dir']);raw=args.raw_dir or root/'data/raw'
    files=sorted(raw.glob('CUCyPhy*.csv'))
    expected=sorted(Path(p).name for p in prior['source_hashes'] if Path(p).name.startswith('CUCyPhy') and p.endswith('.csv'))
    require([p.name for p in files]==expected and 1<=len(files)<=16,'Raw inventory differs')
    require(sum(p.stat().st_size for p in files)<=4_000_000_000,'Raw source exceeds 4 GB')
    paths=files+[registry_path,manifest_path,args.coverage_report,coverage_path,Path('configs/config.yaml'),Path(__file__)]
    stamps={str(p.resolve()):stamp(p) for p in paths};hashes={}
    for p in paths:
        print('Hashing:',p.name,flush=True);hashes[str(p.resolve())]=sha(p)
        if p in files:require(hashes[str(p.resolve())]==expected_hash(prior,p.name),'Raw source changed: '+p.name)
    report.update(source_hashes=hashes,source_stamps_before=stamps)
    groups={e:{} for e in devices};raw_count=0;unique=0
    for p in files:
        print('Scanning:',p.name,flush=True)
        for chunk in pd.read_csv(p,usecols=['entity_id','state','last_updated'],dtype=str,keep_default_na=False,
                                 na_filter=False,encoding='utf-8-sig',chunksize=CHUNK):
            chunk=chunk[chunk.entity_id.isin(devices)].copy()
            if chunk.empty:continue
            chunk['_t']=times(chunk.last_updated);chunk=chunk[chunk['_t']<=hi]
            raw_count+=len(chunk);require(raw_count<=3_000_000,'Raw selected row limit exceeded')
            for e,s,t in chunk[['entity_id','state','_t']].itertuples(index=False,name=None):
                t=int(t)
                if t not in groups[e]:groups[e][t]=set();unique+=1
                require(unique<=500_000,'Unique event limit exceeded')
                groups[e][t].add(s)
                require(len(groups[e][t])==1,'Conflicting raw states: '+e+' at '+str(pd.Timestamp(t)))
    values=np.zeros((len(grid),23),dtype=np.float32);status=np.empty((len(grid),23),dtype=np.uint8)
    scaling={};diagnostics={}
    for j,(e,kind) in enumerate(zip(devices,kinds)):
        print('Preparing:',e,flush=True)
        events=groups.pop(e)
        v,s,no_prior=causal_rows(events,grid,kind)
        values[:,j],scaling[e]=normalize(v,s,kind);status[:,j]=s
        counts={name:int((s==code).sum()) for name,code in STATUS.items()}
        expected_missing=coverage[e]['grid_category_seconds']['explicit_missing']+coverage[e]['grid_category_seconds']['no_prior_record']
        require(counts['missing']==expected_missing,'Source missingness does not reproduce audited coverage: '+e)
        require(int(no_prior.sum())==coverage[e]['grid_category_seconds']['no_prior_record'],'No-prior coverage differs: '+e)
        invalid=Counter(next(iter(states)) for states in events.values() if classify(next(iter(states)),kind)[1]==4)
        diagnostics[e]=dict(status_counts=counts,no_prior_record_rows=int(no_prior.sum()),
             explicit_missing_rows=counts['missing']-int(no_prior.sum()),invalid_literals_unique_event_counts=dict(invalid),
             unique_source_events=len(events))
    require(np.all(values[status!=0]==0),'Nonvalid placeholder differs')
    require(np.isin(status,list(STATUS.values())).all(),'Unknown status code')
    entries=[]
    for a in range(0,len(grid),args.shard_rows):
        b=min(a+args.shard_rows,len(grid));name=f'shard_{len(entries):06d}.npz';path=out/name
        with path.open('xb') as f:np.savez_compressed(f,values=values[a:b],status=status[a:b],timestamps=grid[a:b].view('datetime64[ns]'))
        entries.append(dict(index=len(entries),file=name,start=a,stop=b,rows=b-a,bytes=path.stat().st_size,
             sha256=sha(path),first_timestamp=str(pd.Timestamp(int(grid[a]))),last_timestamp=str(pd.Timestamp(int(grid[b-1])))))
    verification=verify_shards(out,entries,values,status,grid)
    for p in paths:require(stamp(p)==stamps[str(p.resolve())],'Source size/mtime changed: '+str(p))
    require(sorted(raw.glob('CUCyPhy*.csv'))==files,'Raw inventory changed')
    policy=dict(policy_version=POLICY,missing_tokens=['','NULL'],status_codes=STATUS,feature_order=FEATURES,
       time_field='last_updated',timezone_conversion=None,resampling='causal_backward_asof',backward_fill=False,
       skip_nonvalid_records=False,expiry_seconds=None,duplicate_conflicts='block',nonvalid_placeholder=0,
       binary_literals={'on':1,'off':0},binary_numeric_values=[0,1],continuous_values='finite numeric literals',
       whitespace_normalization=False,numeric_status_sentinel_interpretation=False,
       continuous_scaling='valid-training-grid minmax; constant denominator 1; no clipping')
    write_json(out/'source_policy.json',policy)
    metadata=dict(policy_version=POLICY,split='train',devices=devices,status_codes=STATUS,feature_order=FEATURES,
       nonvalid_placeholder=0,registry=registry,registry_role='channel order/types only; raw parsing specified in source_policy.json',
       training=dict(first=str(pd.Timestamp(lo)),last=str(pd.Timestamp(hi)),rows_matched=len(grid),rows_expected=len(grid)),
       scaling=scaling,source_hashes=hashes,channel_diagnostics=diagnostics)
    write_json(out/'transformation_metadata.json',metadata)
    report.update(status='COMPLETE',rows_written=len(grid),shards_written=len(entries),verification=verification,
                  source_size_mtime_recheck_passed=True,raw_selected_rows=raw_count,unique_source_events=unique,
                  output_bytes=sum(e['bytes'] for e in entries),training_bounds=prior['training_bounds'])
    write_json(out/'preparation_report.json',report)
    metadata_files=['source_policy.json','transformation_metadata.json','preparation_report.json']
    manifest=dict(status='COMPLETE',format='hdeg_input_state_rows_v1',policy_version=POLICY,split='train',
       windows_constructed=False,devices=devices,num_channels=23,num_rows=len(grid),status_codes=STATUS,
       feature_order=FEATURES,registry_sha256=sha(registry_path),reference_input_manifest_sha256=sha(manifest_path),
       coverage_report_sha256=sha(args.coverage_report),source_policy_sha256=sha(out/'source_policy.json'),
       metadata={n:dict(file=n,sha256=sha(out/n)) for n in metadata_files},shards=entries)
    write_json(out/'manifest.json',manifest) # Completion marker written last.


def main():
    p=argparse.ArgumentParser(description='Prepare causal source-aware training inputs, with bounded verification')
    p.add_argument('--project_root_dir');p.add_argument('--reference_input_dir',required=True,type=Path)
    p.add_argument('--coverage_report',required=True,type=Path);p.add_argument('--raw_dir',type=Path)
    p.add_argument('--shard_rows',type=int,default=8192);args=p.parse_args()
    require(31<=args.shard_rows<=100_000,'shard_rows must be 31–100000')
    with open('configs/config.yaml') as f:config=yaml.safe_load(f)
    if args.project_root_dir:config['project_root_dir']=args.project_root_dir
    from src.utils.get_folders_utils import get_processed_folder
    out=Path(get_processed_folder(config))/'input_states'/POLICY/'train'/(
          datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status='BLOCKED',policy_version=POLICY,split='train',configuration=config,model_forwards=0,optimizer_steps=0,
       limitations=['Raw clock is unchanged; activity annotation alignment remains unresolved.',
                   'Valid denotes accepted causal source state, not freshness or physical ground truth.',
                   'Current legacy loader intentionally rejects this new policy; no old loader or model was modified.',
                   'Full source bytes are read for hashing; later-period values are excluded from preparation.'])
    try:run(args,config,out,report)
    except Exception as exc:
        report.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
        write_json(out/'failure_report.json',report)
    print('Source input preparation:',report['status']);print('Output:',out)
    if 'error' in report:print(report['error'])
    return 0 if report['status']=='COMPLETE' else 2


if __name__=='__main__':raise SystemExit(main())
