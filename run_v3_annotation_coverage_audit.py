"""Read-only annotation/timestamp feasibility audit. No model imports or fitting."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
from pathlib import Path
import uuid
import numpy as np
import yaml

DAY = 86400
WINDOW = 30
ROLES = ('fit', 'selection', 'assessment')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def load_json(path):
    with path.open(encoding='utf-8-sig') as f: return json.load(f)


def save(path, obj):
    with path.open('x') as f: json.dump(obj, f, indent=2, allow_nan=False)


def stamp(t):
    return datetime.fromtimestamp(int(t), timezone.utc).replace(tzinfo=None).isoformat(sep=' ')


def seconds(s):
    d = datetime.fromisoformat(s.strip())
    if d.microsecond: raise ValueError('Subsecond annotation requires an explicit alignment policy')
    if d.tzinfo is not None: raise ValueError('Timezone-aware log needs a reviewed explicit conversion')
    # UTC here is an arithmetic convention only, not an inferred source timezone.
    return int((d-datetime(1970,1,1)).total_seconds())


def read_log(path, field):
    good=[]; invalid=[]; duplicates=[]; seen={}
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader=csv.DictReader(f)
        if not {'StartTime','EndTime',field}.issubset(reader.fieldnames or []):
            raise ValueError('Missing log columns: '+str(path))
        for rownum,r in enumerate(reader,2):
            try:
                a,b=seconds(r['StartTime']),seconds(r['EndTime']); label=r[field].strip()
                if not label or b<=a: raise ValueError('Empty label or nonpositive interval')
                key=(a,b,label)
                if key in seen:
                    duplicates.append(dict(row=rownum,duplicate_of=seen[key]));continue
                seen[key]=rownum
                good.append(dict(row=rownum,start=a,end=b,label=label))
            except Exception as exc: invalid.append(dict(row=rownum,raw=r,error=str(exc)))
    return good,invalid,duplicates


def components(rows):
    """Half-open overlap components; touching intervals do not overlap."""
    groups=[]
    for r in sorted(rows,key=lambda r:(r['start'],r['end'],r['row'])):
        if not groups or r['start']>=max(x['end'] for x in groups[-1]):groups.append([r])
        else:groups[-1].append(r)
    return groups


def partitions(first,last_exclusive):
    first_day=((first+DAY-1)//DAY)*DAY
    end_day=(last_exclusive//DAY)*DAY
    days=list(range(first_day,end_day,DAY));n=len(days)
    if n<5: raise ValueError('Fewer than five complete source days')
    c1=first_day+int(.6*n)*DAY;c2=first_day+int(.8*n)*DAY
    # A whole day is excluded immediately before each later portion.
    spans={'fit':(first_day,c1-DAY),'selection':(c1,c2-DAY),'assessment':(c2,end_day)}
    return spans,dict(complete_days=n,cutoffs=[stamp(c1),stamp(c2)],gap_days=[stamp(c1-DAY),stamp(c2-DAY)],
                     roles={k:dict(start=stamp(a),end_exclusive=stamp(b)) for k,(a,b) in spans.items()})


def select_windows(a,b,origin):
    # Each window occupies 30 one-second source intervals. No Y window is used.
    count=(b-a)//WINDOW
    if count<1:return []
    slots=np.linspace(0,count-1,min(8,count),dtype=int)
    return [int(a-origin+WINDOW*i) for i in slots]


def audit(args,config,out,report):
    source=Path(__file__).resolve().parent
    paths={'R1':args.misc_dir/'ActivityLogR1.csv','R2':args.misc_dir/'ActivityLogR2.csv',
           'labels':args.misc_dir/'ActivityLabels.json','registry':source/'configs/CU_input_state_registry.json',
           'manifest':args.input_dir/'manifest.json','config':Path('configs/config.yaml'),
           'runner':Path(__file__),'contract':source/'HDEG_behavior_discrimination_contract_v1.md'}
    if args.review: paths['review']=args.review
    hashes={k:digest(p) for k,p in paths.items()}
    report['source_hashes']=hashes
    m=load_json(paths['manifest']);registry=load_json(paths['registry']);labels=load_json(paths['labels'])
    if m['split']!='train' or m['status']!='COMPLETE' or m['num_channels']!=23:
        raise ValueError('Expected complete 23-channel training input manifest')
    if m['registry_sha256']!=hashes['registry']:raise ValueError('Registry differs from prepared inputs')
    if m['num_rows']>2_000_000 or len(m['shards'])>64:raise ValueError('Bound exceeded: two million rows / 64 shards')
    if [c['name'] for c in registry['channels']] != m['devices']: raise ValueError('Registry channel order differs')
    origin=None;previous=None;total=0;shard_hashes={}
    for s in m['shards']:
        p=(args.input_dir/s['file']).resolve()
        if not p.is_relative_to(args.input_dir.resolve()):raise ValueError('Shard escapes input directory')
        h=digest(p)
        if h!=s['sha256']:raise ValueError('Shard hash mismatch: '+s['file'])
        shard_hashes[str(p)]=h
        with np.load(p,allow_pickle=False) as f:t=f['timestamps']
        if t.ndim!=1 or str(t.dtype)!='datetime64[ns]' or len(t)!=s['rows'] or np.isnat(t).any():raise ValueError('Invalid timestamps')
        ns=t.astype(np.int64)
        if not len(ns) or (ns%1_000_000_000).any() or not np.all(np.diff(ns)==1_000_000_000):raise ValueError('Timestamp grid is not exact one-second')
        if previous is not None and ns[0]!=previous+1_000_000_000:raise ValueError('Timestamp gap at shard boundary')
        if s['start']!=total or s['stop']!=total+len(t):raise ValueError('Invalid shard row offsets')
        if origin is None:origin=int(ns[0]//1_000_000_000)
        previous=int(ns[-1]);total+=len(t)
    if total!=m['num_rows']:raise ValueError('Row count differs')
    end=int(previous//1_000_000_000)+1
    spans,plan=partitions(origin,end);plan.update(timestamp_arithmetic='naive source clock; timezone equivalence not inferred',
        input_start=stamp(origin),input_end_exclusive=stamp(end),input_manifest_sha256=hashes['manifest'],window_rows=30,max_windows_per_episode=8)
    r1,bad1,dup1=read_log(paths['R1'],'ActID');r2,bad2,dup2=read_log(paths['R2'],'ActDetails')
    if len(r1)+len(r2)>10000:raise ValueError('Annotation bound exceeded')
    unknown=sorted({r['label'] for r in r1}-set(labels))
    review=load_json(args.review) if args.review else {}
    observable=review.get('observable_activities',{})
    reviewed={k:v for k,v in observable.items() if k in labels and isinstance(v,dict) and v.get('evidence')
              and v.get('channels') and set(v['channels']).issubset(m['devices'])}
    episodes=[];excluded=Counter();support=defaultdict(lambda:{k:dict(episodes=0,days=set(),windows=0) for k in ROLES})
    overlap_components=[]
    for component in components(r1):
        if len(component)>1:
            overlap_components.append([r['row'] for r in component])
            for r in component: episodes.append(dict(**r,eligible=False,reason='overlap_component'))
            excluded['overlap_component']+=len(component)
            continue
        r=component[0];a,b=r['start'],r['end'];reason=None
        if r['label'] not in labels:reason='undefined_activity_id'
        elif a<origin or b>end:reason='outside_training_support'
        role=next((k for k,(x,y) in spans.items() if a>=x and b<=y),None)
        if reason is None and role is None:reason='gap_partial_day_or_partition_crossing'
        starts=select_windows(a,b,origin) if reason is None else []
        if reason is None and not starts:reason='shorter_than_window'
        record=dict(**r,eligible=reason is None,role=role,reason=reason,window_start_rows=starts,
                    observability_reviewed=r['label'] in reviewed)
        episodes.append(record)
        if reason:excluded[reason]+=1
        else:
            s=support[r['label']][role];s['episodes']+=1;s['days'].add(stamp(a)[:10]);s['windows']+=len(starts)
    table={};eligible=[]
    for label in sorted({r['label'] for r in r1}):
        d=support[label];counts={k:dict(episodes=v['episodes'],days=sorted(v['days']),num_days=len(v['days']),windows=v['windows']) for k,v in d.items()}
        enough=all(counts[k]['episodes']>=n and counts[k]['num_days']>=days for k,n,days in [('fit',10,3),('selection',5,2),('assessment',5,2)])
        table[label]=dict(name=labels.get(label,{}).get('Name'),roles=counts,support_pass=enough,observability_reviewed=label in reviewed)
        if enough and label in reviewed:eligible.append(label)
    # Candidate PID matches are lexical hints, never proof of observability.
    import re
    hints={}
    for label,v in labels.items():
        pids=v.get('Databases',{}).get('CU_CyPhy',[])
        channels=[c for c in m['devices'] if (match:=re.search(r'pid_(\d+)',c)) and int(match[1]) in pids]
        hints[label]=dict(name=v.get('Name'),dictionary_CU_CyPhy_ids=pids,candidate_pid_channels=channels,
                          status='REVIEW_REQUIRED; missing lexical match does not establish absence')
    clock=review.get('clock_alignment',{})
    clock_ok=clock.get('same_clock_confirmed') is True and bool(clock.get('evidence'))
    reasons=[]
    if not clock_ok:reasons.append('Source-clock equivalence needs independent evidence')
    if bad1:reasons.append('Invalid R1 annotation rows require resolution')
    if len(eligible)<3:reasons.append('Fewer than three independently reviewed observable classes meet all support floors')
    report.update(audit_status='COMPLETE',primary_gate='BLOCKED' if reasons else 'READY_FOR_PROTOCOL_REVIEW',
        primary_blockers=reasons,cross_actor_gate='BLOCKED',
        cross_actor_blockers=['R2 free-text mapping and period-specific coverage not validated by this training-period auditor'],
        raw_R1_records=len(r1)+len(bad1)+len(dup1),raw_R2_records=len(r2)+len(bad2)+len(dup2),
        unknown_R1_ids=unknown,invalid_rows={'R1':bad1,'R2':bad2},duplicates={'R1':dup1,'R2':dup2},
        R1_overlap_components=overlap_components,R2_overlap_components=[[r['row'] for r in g] for g in components(r2) if len(g)>1],
        exclusion_counts=dict(excluded),eligible_episode_count=sum(r['eligible'] for r in episodes),
        support_eligible_classes=[k for k,v in table.items() if v['support_pass']],
        eligible_classes=eligible,clock_review_accepted=bool(clock_ok),
        observed_channels=m['devices'],timestamp_rows_verified=total,split=plan,
        notes=['READY_FOR_PROTOCOL_REVIEW never authorizes training.',
               'Undefined labels and overlaps are explicitly excluded, not converted to negative labels.',
               'All current coverage numbers describe metadata eligibility, not actual activity separability.',
               'Existing prepared scaling used the broader training period; strict prefix-fit scaling must be regenerated before a future experiment.',
               'Assessment is development data, not an untouched final test.',
               'Unknown input observability cannot be resolved from PID substring matches alone.'])
    save(out/'episode_inventory.json',episodes);save(out/'class_coverage.json',table)
    save(out/'channel_observability_candidates.json',hints);save(out/'proposed_split_manifest.json',plan)
    save(out/'R2_unmapped_descriptions.json',dict(Counter(r['label'] for r in r2)))
    for k,p in paths.items():
        if digest(p)!=hashes[k]:raise ValueError('Source changed during audit: '+k)
    for p,h in shard_hashes.items():
        if digest(Path(p))!=h:raise ValueError('Prepared shard changed during audit')
    report['source_recheck_passed']=True
    report['output_hashes']={p.name:digest(p) for p in out.iterdir() if p.is_file()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project_root_dir');p.add_argument('--input_dir',type=Path,required=True)
    p.add_argument('--misc_dir',type=Path,required=True);p.add_argument('--review',type=Path)
    args=p.parse_args()
    with open('configs/config.yaml') as f:config=yaml.safe_load(f)
    if args.project_root_dir:config['project_root_dir']=args.project_root_dir
    from src.utils.get_folders_utils import get_processed_folder
    out=Path(get_processed_folder(config))/'v2_validation'/'annotation_coverage'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    report=dict(audit_status='BLOCKED',primary_gate='BLOCKED',model_forwards=0,optimizer_steps=0,configuration=config)
    try:audit(args,config,out,report)
    except Exception as exc:report.update(audit_status='BLOCKED',primary_gate='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    save(out/'annotation_coverage_report.json',report)
    print('Annotation audit:',report['audit_status'],'| Primary gate:',report['primary_gate']);print('Report:',out/'annotation_coverage_report.json')
    for reason in report.get('primary_blockers',[]):print('BLOCKER:',reason)
    if 'error' in report:print(report['error'])
    return 0 if report['audit_status']=='COMPLETE' else 2

if __name__=='__main__':raise SystemExit(main())
