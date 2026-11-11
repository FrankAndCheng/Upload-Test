"""Independently recompute recorded metrics and reload recommended models."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/paper-matplotlib')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from run_ablation import features,neural

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def main():
    frozen=json.loads((OUT/'frozen_recommendations.json').read_text())
    completed=json.loads((OUT/'completed.json').read_text())
    for p,h in frozen['input_hashes'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    oof=pd.read_csv(OUT/'oof_predictions.csv')
    cv=pd.read_csv(OUT/'cv_summary.csv')
    fold=pd.read_csv(OUT/'fold_results.csv')
    held=pd.read_csv(OUT/'holdout_metrics.csv')
    for r in cv.itertuples():
        p=[oof[f'{r.model}__{r.mode}__{seed}'].to_numpy() for seed in [23,2026]]
        if np.isfinite(r.CV_MSE):
            actual=float(np.mean([mean_squared_error(oof.actual,q) for q in p]))
            assert np.isclose(actual,r.CV_MSE,rtol=1e-11,atol=1e-11),(r.model,r.mode,actual,r.CV_MSE)
        z=fold[(fold.model==r.model)&(fold['mode']==r.mode)]
        assert len(z)==10 and z.groupby('seed').size().to_dict()=={23:5,2026:5}
    # Scaling alone must retain dimension, including the identical missing flags.
    for name in cv.model.unique():
        a=fold[(fold.model==name)&(fold['mode']=='raw')].sort_values(['seed','fold'])
        b=fold[(fold.model==name)&(fold['mode']=='scale')].sort_values(['seed','fold'])
        assert np.array_equal(a.dimension.to_numpy(),b.dimension.to_numpy(),equal_nan=True)
        for mode in ['pca','scale_pca']:
            z=fold[(fold.model==name)&(fold['mode']==mode)].sort_values(['seed','fold'])
            good=z.dimension.notna().to_numpy()&a.dimension.notna().to_numpy()
            assert (z.dimension.to_numpy()[good]<=a.dimension.to_numpy()[good]).all()
    specs={s['name']:s for s in frozen['specifications']}
    raw=pd.concat([pd.read_csv(ROOT/'data/d_train_20180102.csv'),pd.read_csv(ROOT/'data/d_test_A_20180102.csv')],ignore_index=True)
    raw=raw.drop(columns='血糖',errors='ignore').set_index('id',drop=False)
    checks=[];max_metric_error=0.
    for dataset in ['development','B']:
        d=pd.read_csv(OUT/f'{dataset}_predictions.csv')
        frame=raw.loc[d.id.to_numpy()].copy() if dataset=='development' else pd.read_csv(ROOT/'data/d_test_B_20180128.csv')
        assert '血糖' not in frame.columns and np.array_equal(frame.id.to_numpy(),d.id.to_numpy())
        for r in held[(held.dataset==dataset)&(held.status=='ok')].itertuples():
            p=d[f'{r.model}__{r.mode}'].to_numpy()
            diffs=[abs(mean_squared_error(d.actual,p)-r.MSE),abs(mean_absolute_error(d.actual,p)-r.MAE),abs(r2_score(d.actual,p)-r.R2)]
            max_metric_error=max(max_metric_error,*diffs)
            assert max(diffs)<1e-8,(dataset,r.model,r.mode,diffs)
        loaded={}
        for r in frozen['recommendations']:
            name=r['model'];mode=r['recommended'];s=specs[name]
            tag=f'{dataset}__{name}__{mode}';dest=OUT/'models'
            x=features(frame,s['variant'])
            if s['family'].startswith('nn'):
                pre=joblib.load(dest/f'{tag}_preprocess.joblib')
                xx=pre.transform(x)
                # Legacy Dense(activation=PReLU()) cannot be deserialized via
                # load_model in this Keras version. Rebuild the EXACT saved
                # architecture and load its native archive weights instead.
                m=neural(xx.shape[1],s)
                m(np.zeros((1,xx.shape[1]),dtype=np.float32),training=False)
                m.load_weights(dest/f'{tag}.keras')
                p=m.predict(xx,verbose=0).reshape(-1).astype(float)
            else:
                bundle=joblib.load(dest/f'{tag}.joblib');pre=bundle['preprocess']
                p=bundle['estimator'].predict(pre.transform(x) if pre is not None else x)
            expected=d[f'{name}__{mode}'].to_numpy()
            gap=float(np.max(np.abs(p-expected)))
            assert np.allclose(p,expected,rtol=1e-6,atol=1e-6),(dataset,name,mode,gap)
            checks.append(dict(dataset=dataset,model=name,mode=mode,max_reload_prediction_difference=gap))
            loaded[name]=p
        fusion=np.mean([loaded[n] for n in frozen['fusion']['members']],axis=0)
        assert np.allclose(fusion,d.Fusion__recommended,rtol=1e-6,atol=1e-6)
    result=dict(CV_recomputed=True,fold_count_verified=True,PCA_dimension_verified=True,
                holdout_metrics_recomputed=True,maximum_metric_roundtrip_error=max_metric_error,
                input_hashes_unchanged=True,reference_checks=completed['reference_checks'],
                reload_checks=checks,fusion_reloaded=True)
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('PASSED: CV/holdout metrics, folds/PCA dimensions, original baselines, input hashes, recommended-model reloads and fusion.')


if __name__=='__main__':main()
