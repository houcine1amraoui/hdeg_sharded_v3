"""Read-only source-aware loader verification: all bounded shards, 32 paired windows."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import numpy as np
import yaml
from src.data_preparation import input_state_windows as loader


def verify(input_dir, expected_manifest_sha256=None):
    path=Path(input_dir)/'manifest.json'
    if expected_manifest_sha256:
        loader.require(loader.sha256(path)==expected_manifest_sha256,'Unexpected input manifest hash')
    dataset=loader.InputStateWindowDataset(input_dir,window_size=30,cache_shards=2)
    loader.require(dataset.policy_version==loader.SOURCE_POLICY,'This verifier requires source-aware inputs')
    loader.require(dataset.num_rows<=3_000_000 and len(dataset.shards)<=512,'Verification bounds exceeded')
    counts=np.zeros((23,5),dtype=np.int64)
    transitions=[];previous=None
    for i,e in enumerate(dataset.shards):
        a=dataset.load_shard(i);s=a['status']
        for k in range(5):counts[:,k]+=(s==k).sum(axis=0)
        if len(transitions)<4:
            changed=np.flatnonzero(np.any(s[1:]!=s[:-1],axis=1))+1
            if previous is not None and np.any(s[0]!=previous):changed=np.concatenate(([0],changed))
            transitions.extend(int(e['start']+j) for j in changed[:4-len(transitions)])
        previous=s[-1].copy()
    for i,name in enumerate(dataset.devices):
        expected=dataset.metadata['channel_diagnostics'][name]['status_counts']
        loader.require(counts[i].tolist()==[expected[k] for k in loader.STATUS_CODES],
                       'Saved status counts differ from metadata: '+name)
    # Fixed budget: edges, four first status transitions, then shard crossings and uniform points.
    candidates=[0,len(dataset)-1]+[max(0,min(len(dataset)-1,j-15)) for j in transitions]
    candidates += np.linspace(0,len(dataset)-1,12,dtype=np.int64).tolist()
    candidates += [max(0,min(len(dataset)-1,e['start']-15)) for e in dataset.shards[1:17]]
    candidates += np.linspace(0,len(dataset)-1,32,dtype=np.int64).tolist()
    indices=list(dict.fromkeys(candidates))[:min(32,len(dataset))]
    for start in range(0,len(indices),8):
        subset=indices[start:start+8];batch=dataset.get_batch(subset)
        loader.require(batch['X'].shape==batch['Y'].shape==(len(subset),30,23,6),'Batch shape differs')
        for b,index in enumerate(subset):
            # Direct NPZ slicing is independent of dataset.read_rows/get_batch.
            parts={k:[] for k in ['values','status','timestamps']}
            for e in dataset.shards:
                left,right=max(index,e['start']),min(index+31,e['stop'])
                if left>=right:continue
                with np.load(dataset.directory/e['file'],allow_pickle=False) as a:
                    for k in parts:parts[k].append(a[k][left-e['start']:right-e['start']])
            v,s,t=[np.concatenate(parts[k]) for k in ['values','status','timestamps']]
            features=np.concatenate([v[...,None],np.eye(5,dtype=np.float32)[s]],axis=-1)
            loader.require(np.all(features[...,1:].sum(-1)==1),'Status flags not one-hot')
            for prefix,sl in [('X',slice(None,-1)),('Y',slice(1,None))]:
                for key,expected in [(prefix,features[sl]),(prefix+'_values',v[sl]),
                                     (prefix+'_status',s[sl]),(prefix+'_timestamps',t[sl])]:
                    loader.require(np.array_equal(batch[key][b],expected),'Direct read comparison differs: '+key)
            loader.require(np.array_equal(batch['X'][b,1:],batch['Y'][b,:-1]),'Paired shift differs')
    dataset.verify_unchanged()
    return dict(status='PASS',policy_version=dataset.policy_version,input_dir=str(dataset.directory.resolve()),
                manifest_sha256=dataset.manifest_sha256,metadata_hashes=dataset.metadata_hashes,
                loader_sha256=loader.sha256(Path(loader.__file__)),runner_sha256=loader.sha256(Path(__file__)),
                num_rows=dataset.num_rows,num_pairs=len(dataset),shards_verified=len(dataset.shards),
                pair_indices=indices,pairs_verified=len(indices),window_size=30,num_channels=23,features=loader.FEATURE_ORDER,
                per_channel_status_counts={name:{k:int(counts[i,j]) for j,k in enumerate(loader.STATUS_CODES)}
                                           for i,name in enumerate(dataset.devices)},
                source_files_unchanged=True,model_forwards=0,optimizer_steps=0,
                scope='Artifact integrity and loader compatibility only; no raw replay or scientific model validation')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project_root_dir')
    p.add_argument('--input_dir',required=True,type=Path);p.add_argument('--expected_manifest_sha256')
    args=p.parse_args()
    with open('configs/config.yaml') as f:config=yaml.safe_load(f)
    if args.project_root_dir:config['project_root_dir']=args.project_root_dir
    from src.utils.get_folders_utils import get_processed_folder
    out=Path(get_processed_folder(config))/'v2_validation'/'source_input_windows'/(
       datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    try:report=verify(args.input_dir,args.expected_manifest_sha256)
    except Exception as exc:report=dict(status='BLOCKED',error=f'{type(exc).__name__}: {exc}',model_forwards=0,optimizer_steps=0)
    report['configuration']=config
    with (out/'source_window_verification_report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print('Source window verification:',report['status']);print('Report:',out/'source_window_verification_report.json')
    if 'error' in report:print(report['error'])
    return 0 if report['status']=='PASS' else 2


if __name__=='__main__':raise SystemExit(main())
