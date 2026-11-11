"""Fixed-model 2x2 scaling/PCA experiment. No hyperparameter search.

Ten frozen model versions, identical rows/folds per treatment. Every learned
transform fits training rows only. Selection uses training OOF exclusively.
"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/paper-matplotlib')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import ast
import hashlib
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold,train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
CACHE=OUT/'cache';CACHE.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'review/sequential_experiments'))
from run_experiments import features

MODES=['raw','scale','pca','scale_pca']
SEEDS=[23,2026]
SOURCE_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def dump(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    tmp.replace(path)


def definitions():
    fixed=json.loads((ROOT/'review/external_b_comparison/frozen_configurations.json').read_text())['algorithms']
    optimized=json.loads((ROOT/'review/sequential_experiments/frozen_selection.json').read_text())['specifications']
    specs=[]
    for name,family in [('Ridge','ridge'),('DecisionTree','tree'),('AdaBoost','ada'),('RandomForest','forest')]:
        params=fixed[name]['estimator']['parameters'].copy()
        if family=='forest':params['n_jobs']=2
        if family=='ada':
            params.pop('base_estimator',None)
            params['estimator']=params['estimator']['parameters']
        specs.append(dict(name=name,family=family,variant='all',params=params,
                          reference='scale' if family=='ridge' else 'raw'))
    xgb=fixed['XGBoost']['estimator']['parameters']
    # Saved defaults can contain NaN serialized as strings. Use the explicitly
    # chosen parameters and constructor flags, rather than reinterpreting them.
    original=dict(n_estimators=350,learning_rate=.02,max_depth=4,min_child_weight=30,
                  reg_lambda=10,subsample=.8,colsample_bytree=.8)
    specs.append(dict(name='XGBoost_original',family='xgb',variant='no_hepatitis',params=original,reference='raw'))
    for key,name in [('joint_xgb_089','XGBoost_optimized'),('joint_lgb_05','LightGBM'),
                     ('joint_hist_04','HistGradientBoosting')]:
        s=optimized[key]
        specs.append(dict(name=name,family=s['family'],variant=s['variant'],params=s['params'],
                          reference='raw' if s['family']=='xgb' else 'native'))
    specs.extend([
        dict(name='NeuralNetwork_original',family='nn_original',variant='all',epochs=87,batch_size=32,reference='scale_pca'),
        dict(name='NeuralNetwork_small',family='nn_small',variant='no_hepatitis',epochs=36,batch_size=64,reference='scale'),
    ])
    return specs


def preprocessing(mode):
    if mode=='native':return None
    steps=[('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True))]
    if mode in ['scale','scale_pca']:steps.append(('scale',StandardScaler()))
    if mode in ['pca','scale_pca']:steps.append(('pca',PCA(n_components=.98,svd_solver='full')))
    return Pipeline(steps)


def model(s):
    f=s['family'];p=s.get('params',{}).copy()
    if f=='ridge':return Ridge(**p)
    if f=='tree':return DecisionTreeRegressor(**p)
    if f=='forest':return RandomForestRegressor(**p)
    if f=='ada':
        p['estimator']=DecisionTreeRegressor(**p['estimator'])
        return AdaBoostRegressor(**p)
    if f=='xgb':return XGBRegressor(objective='reg:squarederror',tree_method='hist',random_state=23,n_jobs=2,**p)
    if f=='lgb':return LGBMRegressor(objective='regression',verbosity=-1,deterministic=True,force_col_wise=True,random_state=23,n_jobs=2,**p)
    if f=='hist':return HistGradientBoostingRegressor(early_stopping=False,random_state=23,**p)
    raise ValueError(f)


def neural(n,s):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(23)
    if s['family']=='nn_original':
        # Same builder as paper.ipynb cell 47; only the input representation varies.
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Input,Dense,PReLU,Dropout,BatchNormalization
        m=Sequential([Input(shape=(n,)),Dense(400,activation=PReLU()),Dropout(.4),
             Dense(160,kernel_initializer='normal',activation=PReLU()),BatchNormalization(),Dropout(.6),
             Dense(64,kernel_initializer='normal',activation=PReLU()),BatchNormalization(),Dropout(.5),
             Dense(26,kernel_initializer='normal',activation=PReLU()),BatchNormalization(),Dropout(.6),
             Dense(1,kernel_initializer='normal')])
        m.compile(loss='mse',optimizer='adamax')
    else:
        layers=[tf.keras.layers.Input(shape=(n,))]
        for width in [64,32]:
            layers.extend([tf.keras.layers.Dense(width,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(.001)),
                           tf.keras.layers.Dropout(.1)])
        layers.append(tf.keras.layers.Dense(1))
        m=tf.keras.Sequential(layers)
        m.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(.001))
    return m


def fit_predict(s,mode,train,test,save_tag=None):
    x=features(train,s['variant']);z=features(test,s['variant'])
    y=train['血糖'].to_numpy()
    pre=preprocessing(mode)
    xx=pre.fit_transform(x) if pre is not None else x
    zz=pre.transform(z) if pre is not None else z
    if s['family'].startswith('nn'):
        m=neural(xx.shape[1],s)
        m.fit(xx,y,epochs=s['epochs'],batch_size=s['batch_size'],verbose=0)
        p=m.predict(zz,verbose=0).reshape(-1).astype(float)
    else:
        m=model(s);m.fit(xx,y);p=np.asarray(m.predict(zz),dtype=float)
    if not np.isfinite(p).all():raise ValueError('nonfinite predictions')
    if save_tag:
        dest=OUT/'models';dest.mkdir(exist_ok=True)
        if s['family'].startswith('nn'):
            m.save(dest/f'{save_tag}.keras');joblib.dump(pre,dest/f'{save_tag}_preprocess.joblib')
        else:
            joblib.dump(dict(preprocess=pre,estimator=m),dest/f'{save_tag}.joblib')
    return p,xx.shape[1]


def metrics(y,p):
    mse=float(mean_squared_error(y,p))
    return dict(MSE=mse,RMSE=float(np.sqrt(mse)),MAE=float(mean_absolute_error(y,p)),R2=float(r2_score(y,p)))


def main():
    start=time.time()
    inputs=['data/d_train_20180102.csv','data/d_test_A_20180102.csv','data/d_answer_a_20180128.csv',
            'review/external_b_comparison/frozen_configurations.json','review/sequential_experiments/frozen_selection.json',
            'review/sequential_experiments/run_experiments.py']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs}
    data_sha=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    a=pd.read_csv(ROOT/inputs[1]);a['血糖']=pd.read_csv(ROOT/inputs[2]).iloc[:,0].to_numpy()
    full=pd.concat([pd.read_csv(ROOT/inputs[0]),a],ignore_index=True)
    train,valid=train_test_split(full,test_size=.3,random_state=23)
    assert (len(train),len(valid),len(full))==(4649,1993,6642)
    assert full.id.is_unique and set(train.id).isdisjoint(valid.id)
    specs=definitions();lookup={s['name']:s for s in specs}
    plan=dict(date='2026-10-20',source_sha256=SOURCE_SHA,input_hashes=hashes,
              specs=specs,modes=MODES,cv_seeds=SEEDS,n_splits=5,pca_variance=.98,
              fixed_nn_epochs=True,all_four_modes_use_same_median_imputation_and_missing_indicators=True,
              native_reference_only=['LightGBM','HistGradientBoosting'],
              selection_rule='relative mean OOF MSE improvement >=0.005 AND >=7/10 paired fold wins vs frozen current reference; choose eligible minimum; otherwise retain reference',
              scope='Conditional on frozen features/parameters/epochs; no retuning; two CV seeds not independent samples; known development/B descriptive only.')
    dump(OUT/'plan.json',plan)
    pd.DataFrame({'id':train.id.to_numpy(),'split':'train'}).to_csv(OUT/'train_ids.csv',index=False)
    rows=[];predictions={};fold_rows=[]
    neural_initialized=False
    for s in specs:
        if s['family'].startswith('nn') and not neural_initialized:
            import tensorflow as tf
            tf.config.threading.set_inter_op_parallelism_threads(2)
            tf.config.threading.set_intra_op_parallelism_threads(2)
            neural_initialized=True
        modes=MODES+(['native'] if s['reference']=='native' else [])
        for mode in modes:
            arrays=[];losses=[];dimensions=[];errors=[]
            for seed in SEEDS:
                oof=np.full(len(train),np.nan)
                for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=seed).split(train)):
                    key=hashlib.sha256(json.dumps([SOURCE_SHA,data_sha,s,mode,seed,fold],sort_keys=True).encode()).hexdigest()[:24]
                    path=CACHE/f'{key}.npz'
                    try:
                        if path.exists():
                            z=np.load(path);p=z['prediction'];dim=int(z['dimension'])
                        else:
                            p,dim=fit_predict(s,mode,train.iloc[tr],train.iloc[va])
                            np.savez_compressed(path,prediction=p,dimension=dim)
                        assert np.isfinite(p).all()
                        loss=float(mean_squared_error(train['血糖'].iloc[va],p));oof[va]=p
                    except Exception as error:
                        dim=None;loss=float('inf');errors.append(f'seed={seed} fold={fold}: {type(error).__name__}: {error}')
                    losses.append(loss);dimensions.append(dim)
                    fold_rows.append(dict(model=s['name'],mode=mode,seed=seed,fold=fold,MSE=loss,dimension=dim))
                    if s['family'].startswith('nn'):
                        print('NN fold',s['name'],mode,seed,fold+1,'MSE',round(loss,6),flush=True)
                arrays.append(oof)
            good=all(np.isfinite(a).all() for a in arrays)
            mse=float(np.mean([mean_squared_error(train['血糖'],a) for a in arrays])) if good else float('inf')
            predictions[(s['name'],mode)]=arrays
            row=dict(model=s['name'],mode=mode,CV_MSE=mse,
                     seed23_MSE=float(mean_squared_error(train['血糖'],arrays[0])) if np.isfinite(arrays[0]).all() else float('inf'),
                     seed2026_MSE=float(mean_squared_error(train['血糖'],arrays[1])) if np.isfinite(arrays[1]).all() else float('inf'),
                     fold_MSE=losses,dimensions=dimensions,errors=errors)
            rows.append(row);dump(OUT/'progress.json',rows)
            print('CV',s['name'],mode,'MSE',round(mse,6),'seconds',round(time.time()-start),flush=True)
    frame=pd.DataFrame(rows);folds=pd.DataFrame(fold_rows)
    recommendations=[];comparisons=[]
    rng=np.random.default_rng(23)
    for s in specs:
        name=s['name'];reference=s['reference'];sub=frame[frame.model==name]
        ref=sub[sub['mode']==reference].iloc[0]
        ref_fold=np.array(ref.fold_MSE)
        eligible=[]
        for r in sub.itertuples():
            relative=(ref.CV_MSE-r.CV_MSE)/ref.CV_MSE
            wins=int((np.array(r.fold_MSE)<ref_fold-1e-12).sum())
            accepted=r.mode==reference or (relative>=.005 and wins>=7)
            if accepted:eligible.append((r.CV_MSE,r.mode))
            diffs=[]
            for seed_index in range(2):
                q=predictions[(name,r.mode)][seed_index];p=predictions[(name,reference)][seed_index]
                diffs.append((train['血糖'].to_numpy()-q)**2-(train['血糖'].to_numpy()-p)**2)
            delta=np.mean(diffs,axis=0)
            lo=hi=float('nan')
            if np.isfinite(delta).all():
                boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(2000)])
                lo,hi=np.quantile(boot,[.025,.975])
            comparisons.append(dict(model=name,mode=r.mode,reference=reference,CV_MSE=r.CV_MSE,
                relative_improvement=relative,fold_wins=wins,eligible=accepted,
                MSE_delta=float(np.mean(delta)),low95=float(lo),high95=float(hi)))
        if not eligible:raise RuntimeError(f'No valid reference or replacement for {name}')
        chosen=min(eligible)[1]
        best=str(sub.sort_values('CV_MSE').iloc[0]['mode'])
        recommendations.append(dict(model=name,reference=reference,recommended=chosen,CV_best=best,
            reference_CV_MSE=float(ref.CV_MSE),recommended_CV_MSE=float(sub[sub['mode']==chosen].CV_MSE.iloc[0])))
    rec=pd.DataFrame(recommendations)
    selected={r['model']:r['recommended'] for r in recommendations}
    fusion_members=['XGBoost_optimized','LightGBM','Ridge','NeuralNetwork_small']
    fusion_cv=[];fusion_oof={}
    for version in ['reference','recommended']:
        for i,seed in enumerate(SEEDS):
            pred=np.mean([predictions[(m,lookup[m]['reference'] if version=='reference' else selected[m])][i] for m in fusion_members],axis=0)
            fusion_oof[(version,seed)]=pred
            fusion_cv.append(dict(version=version,seed=seed,**metrics(train['血糖'],pred)))
    # Explicitly freeze BEFORE reading B or evaluating any development labels.
    frozen=dict(source_sha256=SOURCE_SHA,input_hashes=hashes,recommendations=recommendations,
                selection_rule=plan['selection_rule'],specifications=specs,
                fusion=dict(members=fusion_members,weights=[.25]*4,change_weights=False),
                fusion_CV=fusion_cv,note='No development/B-based selection; all model hyperparameters fixed.')
    dump(OUT/'frozen_recommendations.json',frozen)
    frame.drop(columns=['fold_MSE','dimensions','errors']).to_csv(OUT/'cv_summary.csv',index=False)
    folds.to_csv(OUT/'fold_results.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'paired_cv_comparisons.csv',index=False)
    rec.to_csv(OUT/'recommendations.csv',index=False)
    pd.DataFrame(fusion_cv).to_csv(OUT/'fusion_cv.csv',index=False)
    saved_oof=pd.DataFrame({'id':train.id.to_numpy(),'actual':train['血糖'].to_numpy()})
    for (name,mode),arrays in predictions.items():
        for seed,p in zip(SEEDS,arrays):saved_oof[f'{name}__{mode}__{seed}']=p
    saved_oof.to_csv(OUT/'oof_predictions.csv',index=False)
    print('FROZEN RECOMMENDATIONS',rec.to_string(index=False),flush=True)
    b=pd.read_csv(ROOT/'data/d_test_B_20180128.csv')
    b['血糖']=pd.read_csv(ROOT/'data/d_answer_b_20180130.csv',header=None).iloc[:,0].to_numpy()
    assert len(b)==1000 and set(full.id).isdisjoint(b.id)
    evaluations=[];fusion_evaluations=[]
    for dataset,fit,test in [('development',train,valid),('B',full,b)]:
        table=pd.DataFrame({'id':test.id.to_numpy(),'actual':test['血糖'].to_numpy()})
        preds={}
        for s in specs:
            for mode in MODES+(['native'] if s['reference']=='native' else []):
                status='ok';error='';dim=None
                try:
                    save=f'{dataset}__{s["name"]}__{mode}' if mode in {selected[s['name']],s['reference']} else None
                    p,dim=fit_predict(s,mode,fit,test,save_tag=save)
                    result=metrics(test['血糖'],p)
                except Exception as ex:
                    status='failed';error=f'{type(ex).__name__}: {ex}'
                    p=np.full(len(test),np.nan);result={k:float('nan') for k in ['MSE','RMSE','MAE','R2']}
                preds[(s['name'],mode)]=p;table[f'{s["name"]}__{mode}']=p
                row=dict(dataset=dataset,model=s['name'],mode=mode,status=status,error=error,dimension=dim,**result)
                evaluations.append(row)
                print('EVAL',dataset,s['name'],mode,result['MSE'],flush=True)
        for version in ['reference','recommended']:
            p=np.mean([preds[(m,lookup[m]['reference'] if version=='reference' else selected[m])] for m in fusion_members],axis=0)
            table['Fusion__'+version]=p
            fusion_evaluations.append(dict(dataset=dataset,version=version,**metrics(test['血糖'],p)))
        table.to_csv(OUT/f'{dataset}_predictions.csv',index=False)
    pd.DataFrame(evaluations).to_csv(OUT/'holdout_metrics.csv',index=False)
    pd.DataFrame(fusion_evaluations).to_csv(OUT/'fusion_holdout.csv',index=False)
    # Reproduce earlier scores for unchanged references, allowing neural HW noise.
    old=pd.read_csv(ROOT/'review/external_b_comparison/comparison.csv')
    mapping={'Ridge':'Ridge','DecisionTree':'DecisionTree','AdaBoost':'AdaBoost',
             'RandomForest':'RandomForest','XGBoost_original':'XGBoost'}
    checks=[]
    for name,old_name in mapping.items():
        prev=old[old.model==old_name].iloc[0]
        for dataset,col in [('development','development_MSE'),('B','B_MSE')]:
            actual=next(r['MSE'] for r in evaluations if r['dataset']==dataset and r['model']==name and r['mode']==lookup[name]['reference'])
            assert np.isclose(actual,prev[col],rtol=1e-8,atol=1e-9),(name,dataset,actual,prev[col])
            checks.append(dict(model=name,dataset=dataset,reference_reproduced=True))
    previous_fusion=pd.read_csv(ROOT/'review/sequential_experiments/evaluation.csv')
    for dataset in ['development','B']:
        prev=float(previous_fusion[(previous_fusion.dataset==dataset)&(previous_fusion.name=='nn_blend_06')].MSE.iloc[0])
        actual=next(r['MSE'] for r in fusion_evaluations if r['dataset']==dataset and r['version']=='reference')
        assert np.isclose(actual,prev,rtol=1e-6,atol=1e-6),(dataset,actual,prev)
    for p,h in hashes.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    dump(OUT/'completed.json',dict(seconds=time.time()-start,source_sha256=SOURCE_SHA,
         reference_checks=checks,input_files_unchanged=True,selection_frozen_before_evaluation=True))
    print('COMPLETE seconds',round(time.time()-start),flush=True)


if __name__=='__main__':main()
