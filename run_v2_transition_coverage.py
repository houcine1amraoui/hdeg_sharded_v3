"""Read-only training transition census and deterministic, disjoint diagnostic selection."""
from __future__ import annotations
import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import yaml
from src.data_preparation import input_state_windows as windows
from src.utils.get_folders_utils import get_processed_folder

require = windows.require
STRATA = ('quiet','older_value_only','older_status_only','older_mixed',
          'new_value_only','new_status_only','new_mixed')
COMPOSITION = ('none','value_only','status_only','mixed')


def classify(value_counts, status_counts, window_size=30):
    """Transition t joins source rows t,t+1; pair i contains transitions i:i+W."""
    require(len(value_counts) == len(status_counts) and len(value_counts) >= window_size,
            'Insufficient or inconsistent transitions')
    def rolling_any(a):
        prefix = np.r_[0,np.cumsum(a > 0,dtype=np.int64)]
        return prefix[window_size:] > prefix[:-window_size]
    whole_v,whole_s = rolling_any(value_counts),rolling_any(status_counts)
    newest = (value_counts[window_size-1:] > 0).astype(np.uint8)
    newest += 2*(status_counts[window_size-1:] > 0).astype(np.uint8)
    whole = whole_v.astype(np.uint8) + 2*whole_s.astype(np.uint8)
    labels = whole.copy()
    labels[newest > 0] += 3
    # An older event can coexist with any newest composition.
    flags = (value_counts > 0) | (status_counts > 0)
    prefix = np.r_[0,np.cumsum(flags,dtype=np.int64)]
    older = prefix[window_size-1:-1] > prefix[:-window_size]
    return labels,newest,older


def choose(labels, manifest_hash, seed=42, window_size=30):
    n = len(labels)
    bounds = np.array([j*n//8 for j in range(9)],dtype=np.int64)
    counts = np.bincount(labels,minlength=7)
    order = sorted(range(7),key=lambda k:(int(counts[k]),k))
    forbidden = np.zeros(n,dtype=bool)
    selected=[];allocation=[]
    for k in order:
        quota = 8 if k == 0 else 4
        for portion in range(8):
            lo,hi = map(int,bounds[portion:portion+2])
            candidates = np.flatnonzero(labels[lo:hi] == k)+lo
            # Use first 64 bits of SHA256, with index as the deterministic tie-break.
            def priority(i):
                token=f'hdeg_transition_coverage_v1|{manifest_hash}|{seed}|{int(i)}'
                return int.from_bytes(hashlib.sha256(token.encode('ascii')).digest()[:8],'big')
            priorities=np.fromiter((priority(i) for i in candidates),dtype=np.uint64,count=len(candidates))
            ranked=np.lexsort((candidates,priorities))
            accepted=[];rejected=0
            for pos in ranked:
                i=int(candidates[pos])
                if forbidden[i]:
                    rejected+=1
                    continue
                selected.append(dict(index=i,stratum=STRATA[k],portion=portion,
                                     priority_hex=f'{int(priorities[pos]):016x}'))
                accepted.append(i)
                # 31-row half-open spans [i,i+31) must not intersect.
                forbidden[max(0,i-window_size):min(n,i+window_size+1)]=True
                if len(accepted)==quota:
                    break
            allocation.append(dict(stratum=STRATA[k],portion=portion,start=lo,stop=hi,
                eligible=int(len(candidates)),quota=quota,selected=len(accepted),
                shortfall=quota-len(accepted),overlap_rejections_examined=rejected))
    indices=sorted(x['index'] for x in selected)
    require(all(b-a >= window_size+1 for a,b in zip(indices,indices[1:])), 'Selected source spans overlap')
    return selected,allocation,order,bounds


def scan(dataset, chunks, progress=True):
    """Bounded row reads; retain only two uint8 counts per adjacent-row transition."""
    total=dataset.num_rows;channels=len(dataset.devices)
    vc=np.empty(total-1,dtype=np.uint8);sc=np.empty(total-1,dtype=np.uint8)
    per_value=np.zeros(channels,dtype=np.int64)
    per_direction=np.zeros((channels,5,5),dtype=np.int64)
    previous=None
    for start in range(0,total,chunks):
        stop=min(total,start+chunks)
        rows=dataset.read_rows(start,stop)
        values,status=rows['values'],rows['status']
        if previous is not None:
            values=np.concatenate((previous[0][None,:],values))
            status=np.concatenate((previous[1][None,:],status))
        valid=(status[:-1]==0)&(status[1:]==0)
        changed_v=valid&(values[:-1]!=values[1:])
        changed_s=status[:-1]!=status[1:]
        left=max(0,start-1)
        vc[left:stop-1]=changed_v.sum(1)
        sc[left:stop-1]=changed_s.sum(1)
        per_value+=changed_v.sum(0)
        for ch in range(channels):
            codes=5*status[:-1,ch][changed_s[:,ch]]+status[1:,ch][changed_s[:,ch]]
            per_direction[ch]+=np.bincount(codes,minlength=25).reshape(5,5)
        previous=(rows['values'][-1].copy(),rows['status'][-1].copy())
        if progress:
            print(f'Census rows {stop}/{total}',flush=True)
    return vc,sc,per_value,per_direction


def run(config,processed,args,report,out):
    settings=config.get('transition_coverage_audit',{})
    require(set(settings)<= {'chunk_rows'},'Unknown transition_coverage_audit setting')
    chunks=args.chunk_rows if args.chunk_rows is not None else settings.get('chunk_rows',65536)
    require(type(chunks) is int and 1<=chunks<=262144,'chunk_rows must be 1..262144')
    require(config.get('seed',42)==42,'Protocol seed must be 42')
    require(config['preprocessing']['window_size']==30,'Protocol requires W=30')
    require(args.input_dir.strip() and (Path(args.input_dir)/'manifest.json').is_file(),
            'input_dir must contain the prepared row manifest')
    pilot_path=Path(args.pilot_report)
    pilot=json.loads(pilot_path.read_text())
    require(pilot['status']=='COMPLETED' and pilot['protocol']=='v2_input_state_optimization_pilot_v1'
            and pilot['optimizer_steps']==100,'Expected completed fixed 100-step pilot report')
    ds=windows.InputStateWindowDataset(args.input_dir,30)
    require(ds.manifest_sha256==pilot['manifest_sha256'],'Prepared manifest differs from pilot')
    require(config['preprocessing']==pilot['configuration']['preprocessing']==ds.preparation['configuration']['preprocessing'],
            'Preprocessing differs from prepared inputs/pilot')
    dependencies={'pilot_report':pilot_path,'runner':Path(__file__),'config':Path('configs/config.yaml'),
        'window_loader':Path(windows.__file__),'devices':processed/'devices.json',
        'timestamps':processed/'timestamps.npz','selected_features':Path('configs/CU_selected_features.txt'),
        'registry':Path(config['preprocessing'].get('input_state_registry','configs/CU_input_state_registry.json'))}
    hashes={k:windows.sha256(p) for k,p in dependencies.items()}
    for k in ('window_loader','devices','timestamps','selected_features','registry'):
        require(hashes[k]==pilot['source_hashes'][k],f'Pilot dependency differs: {k}')
    for k in ('devices','timestamps','selected_features','registry'):
        require(hashes[k]==ds.metadata['source_hashes'][k],f'Prepared dependency differs: {k}')
    indices=pilot['selected_window_indices']
    require(len(indices)==32 and len(set(indices))==32 and all(type(i) is int and 0<=i<len(ds) for i in indices),
            'Expected 32 distinct valid pilot indices')
    vc,sc,per_value,directions=scan(ds,chunks)
    labels,newest,older=classify(vc,sc)
    require(len(labels)==len(ds),'Window census length differs')
    selected,allocation,order,bounds=choose(labels,ds.manifest_sha256)
    counts=np.bincount(labels,minlength=7)
    coverage=[]
    for part in range(8):
        lo,hi=map(int,bounds[part:part+2]);count=np.bincount(labels[lo:hi],minlength=7)
        coverage.append(dict(portion=part,start=lo,stop=hi,counts=dict(zip(STRATA,map(int,count)))))
    # Quiet windows have identical status on all 31 rows, so one start row suffices.
    quiet_status=np.zeros((23,5),dtype=np.int64)
    quiet_contains=np.zeros(5,dtype=np.int64)
    for start in range(0,len(ds),chunks):
        stop=min(len(ds),start+chunks)
        status=ds.read_rows(start,stop)['status'][labels[start:stop]==0]
        for ch in range(23):quiet_status[ch]+=np.bincount(status[:,ch],minlength=5)
        for code in range(5):quiet_contains[code]+=np.any(status==code,axis=1).sum()
    # Maximal all-channel unchanged row runs, including one-row runs.
    edges=np.r_[0,np.flatnonzero((vc+sc)>0)+1,ds.num_rows]
    lengths=np.diff(edges)
    run_values,run_counts=np.unique(lengths,return_counts=True)
    longest=np.lexsort((edges[:-1],-lengths))[:10]
    def timestamp(row):return str(ds.first+np.timedelta64(int(row),'s'))
    def describe(i):
        return dict(index=int(i),stratum=STRATA[int(labels[i])],
            newest_composition=COMPOSITION[int(newest[i])],older_changes_present=bool(older[i]),
            source_start_row=int(i),source_stop_row_exclusive=int(i+31),
            X_start=timestamp(i),X_end=timestamp(i+29),Y_start=timestamp(i+1),Y_end=timestamp(i+30),
            valid_value_change_cells=int(vc[i:i+30].sum()),status_transition_cells=int(sc[i:i+30].sum()))
    enriched=[{**s,**describe(s['index'])} for s in sorted(selected,key=lambda x:x['index'])]
    newest_tables={STRATA[k]:{COMPOSITION[c]:int(((labels==k)&(newest==c)).sum()) for c in range(4)} for k in range(7)}
    row_transition_composition=(vc>0).astype(np.uint8)+2*(sc>0).astype(np.uint8)
    report.update(status='COMPLETE',source_hashes=hashes,manifest_sha256=ds.manifest_sha256,
        input_dir=str(ds.directory.resolve()),num_training_rows=ds.num_rows,num_pairs=len(ds),
        adjacent_row_intervals=ds.num_rows-1,chunk_rows=chunks,devices=ds.devices,
        stratum_counts=dict(zip(STRATA,map(int,counts))),
        stratum_proportions=dict(zip(STRATA,(counts/len(ds)).tolist())),
        chronological_portions=coverage,newest_composition_by_stratum=newest_tables,
        newly_arriving_with_older_changes=int(((newest>0)&older).sum()),
        actual_transition_time_counts=dict(zip(COMPOSITION,map(int,np.bincount(row_transition_composition,minlength=4)))),
        changed_channel_count_histogram=dict(enumerate(map(int,np.bincount(vc+sc,minlength=24)))),
        valid_value_change_cells=int(vc.sum()),status_transition_cells=int(sc.sum()),
        quiet_status_counts_by_channel=quiet_status.tolist(),
        quiet_status_code_order=list(windows.STATUS_CODES),
        quiet_pairs_containing_status=dict(zip(windows.STATUS_CODES,map(int,quiet_contains))),
        quiet_status_window_channel_denominator=int(counts[0])*23,
        unchanged_row_run_length_histogram={str(int(k)):int(v) for k,v in zip(run_values,run_counts)},
        longest_unchanged_runs=[dict(start_row=int(edges[j]),stop_row_exclusive=int(edges[j+1]),
            rows=int(lengths[j]),duration_seconds=int(lengths[j]-1),start=timestamp(edges[j]),end=timestamp(edges[j+1]-1)) for j in longest],
        original_pilot_classification=[describe(i) for i in indices],
        selected_count=len(enriched),selection_allocations=allocation)
    selection=dict(protocol='v2_transition_coverage_selection_v1',manifest_sha256=ds.manifest_sha256,
        pilot_report_sha256=hashes['pilot_report'],seed=42,window_size=30,target_shift=1,
        maximum_pairs=256,chronological_bounds=bounds.tolist(),
        stratum_processing_order=[STRATA[k] for k in order],
        priority='first 64 bits big-endian of SHA256(ascii hdeg_transition_coverage_v1|manifest_sha256|42|index); tie by index',
        overlap_rule='global half-open source spans [index,index+31) disjoint',
        shortage_rule='no replacement, reallocation or separation relaxation',
        rows=enriched,allocations=allocation,
        warning='Input-stratified diagnostic sample; unweighted metrics do not estimate natural prevalence-weighted performance.')
    channels=[dict(channel=ds.devices[ch],valid_value_changes=int(per_value[ch]),
                   status_direction_counts=directions[ch].tolist(),status_order=list(windows.STATUS_CODES)) for ch in range(23)]
    ds.verify_unchanged()
    require(all(windows.sha256(p)==hashes[k] for k,p in dependencies.items()),'Dependency changed during census')
    for name,obj in [('selection_manifest.json',selection),('channel_transition_summary.json',channels)]:
        with (out/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
    report['output_hashes']={name:windows.sha256(out/name) for name in ('selection_manifest.json','channel_transition_summary.json')}
    report['source_and_artifact_recheck_passed']=True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_root_dir')
    parser.add_argument('--input_dir',required=True)
    parser.add_argument('--pilot_report',required=True)
    parser.add_argument('--chunk_rows',type=int)
    args=parser.parse_args()
    with open('configs/config.yaml') as f:config=yaml.safe_load(f)
    if args.project_root_dir:config['project_root_dir']=args.project_root_dir
    processed=Path(get_processed_folder(config))
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=processed/'v2_validation'/'transition_coverage'/run_id
    out.mkdir(parents=True,exist_ok=False)
    report=dict(protocol='v2_transition_coverage_census_v1',status='BLOCKED',configuration=config,
        optimizer_steps=0,model_forward_calls=0,
        limitations=['Prepared float32 values/statuses; not raw-source freshness or physical inactivity.',
                     'Window counts overlap; adjacent-row transition counts count each interval once.',
                     'Disjoint selected spans do not establish statistical independence.',
                     'Enriched unweighted sample metrics are not population performance estimates.'])
    try:run(config,processed,args,report,out)
    except (ValueError,TypeError,KeyError,OSError,IndexError) as exc:report.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}')
    with (out/'transition_coverage_report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(f'Transition coverage: {report["status"]}\nReport: {out/"transition_coverage_report.json"}')
    if 'error' in report:print(report['error'])
    return 0 if report['status']=='COMPLETE' else 2


if __name__=='__main__':raise SystemExit(main())
