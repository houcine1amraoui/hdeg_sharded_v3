"""Read-only analysis of a saved two-arm archive; NumPy only, no model execution."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np

LEVELS=('Z','S','S_tilde','g')

def analyze(archive):
    source_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as z:
        data={'/'.join(n.split('/')[1:]):z.read(n) for n in z.namelist() if not n.endswith('/')}
    report=json.loads(data['comparison_report.json'])
    assert report['status']=='COMPLETE'
    for name,digest in report['output_hashes'].items():
        assert hashlib.sha256(data[name]).hexdigest()==digest,name
    split=json.loads(data['split_manifest.json'])
    assignments={r['index']:r for r in split['assignments']}
    result=dict(source_archive_sha256=source_hash,source_run=archive.name,
        procedure='One coordinate-wise target mean and residual mean fitted across all 128 update pairs per state/level; no stratum-specific fitting.',
        model_forwards=0,optimizer_steps=0,states={})
    fitted={}
    for state,path in [('initial','initialization_representations.npz'),
                       ('A_hbf_only','A_hbf_only/final_representations.npz'),
                       ('B_joint','B_joint/final_representations.npz')]:
        with np.load(io.BytesIO(data[path]),allow_pickle=False) as f:
            ids=f['indices'].tolist()
            assert len(ids)==len(set(ids))==256 and set(ids)==set(assignments)
            update=np.array([assignments[i]['role']=='update' for i in ids])
            assert update.sum()==128
            strata=sorted({assignments[i]['stratum'] for i in ids})
            state_result={role:{st:{} for st in strata} for role in ('update','probe')}
            for level in LEVELS:
                x,y,pred=[f[side+'_'+level].astype(np.float64).reshape(256,-1) for side in ('X','Y','prediction')]
                assert all(np.isfinite(a).all() for a in (x,y,pred))
                target_mean=y[update].mean(0)
                residual_mean=(pred[update]-y[update]).mean(0)
                fitted[state+'_target_mean_'+level]=target_mean
                fitted[state+'_residual_mean_'+level]=residual_mean
                for role in ('update','probe'):
                    for st in strata:
                        mask=np.array([assignments[i]['role']==role and assignments[i]['stratum']==st for i in ids])
                        yy,ff,xx=y[mask],pred[mask],x[mask]
                        residual=ff-yy
                        bias_sq=float((residual.mean(0)**2).mean())
                        centered=float(((residual-residual.mean(0))**2).mean())
                        mse=float((residual**2).mean())
                        assert np.isclose(mse,bias_sq+centered,rtol=1e-12,atol=1e-18)
                        pv=float(ff.var(0).mean());yv=float(yy.var(0).mean())
                        cov=float(((ff-ff.mean(0))*(yy-yy.mean(0))).mean())
                        assert np.isclose(centered,pv+yv-2*cov,rtol=1e-12,atol=1e-18)
                        errors={
                            'hbf':(residual**2).mean(1),
                            'persistence':((xx-yy)**2).mean(1),
                            'update_target_mean':((target_mean-yy)**2).mean(1),
                            'hbf_update_offset_removed':((ff-residual_mean-yy)**2).mean(1)}
                        m=dict(n=int(mask.sum()),hbf_mse=mse,residual_mean_square=bias_sq,
                            centered_residual_mse=centered,offset_fraction=bias_sq/mse if mse>0 else None,
                            prediction_variation=pv,target_variation=yv,prediction_target_covariance=cov,
                            prediction_to_target_variation=pv/yv if yv>0 else None,
                            baselines={})
                        for name,e in errors.items():
                            m['baselines'][name]=dict(mean_mse=float(e.mean()),median_mse=float(np.median(e)),
                                wins_vs_persistence=int((e<errors['persistence']).sum()),
                                ties_vs_persistence=int((e==errors['persistence']).sum()),
                                wins_vs_hbf=int((e<errors['hbf']).sum()))
                        state_result[role][st][level]=m
            result['states'][state]=state_result
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==source_hash,'Source archive changed'
    result['checks']=['All saved output hashes match','128 update and 128 probe pairs',
        'MSE equals squared coordinate-wise mean residual plus centered residual MSE',
        'Centered residual MSE equals prediction variance plus target variance minus twice covariance',
        'Source archive unchanged']
    return result,fitted

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path);parser.add_argument('output_dir',type=Path)
    args=parser.parse_args()
    result,fitted=analyze(args.archive)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with (args.output_dir/'residual_analysis.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    with (args.output_dir/'update_fitted_vectors.npz').open('xb') as f:np.savez_compressed(f,**fitted)
    print('PASS: read-only saved-output analysis complete')
