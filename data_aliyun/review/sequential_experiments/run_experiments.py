"""Ordered, training-only selection. Run with Conda base Python from any directory.

The known development and B labels are never used for candidate/weight selection.
CV scores are selection scores, not unbiased estimates of the entire search.
"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/paper-matplotlib')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import hashlib
import itertools
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, ParameterSampler, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CACHE = OUT / 'cache'
CACHE.mkdir(exist_ok=True)
HEP = ['乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体']
SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SEED = 23


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                               default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


def features(frame, variant):
    # Deterministic per-row transforms only; no target, id, date, or fitted statistics.
    x = frame.drop(columns=['血糖','id','体检日期'], errors='ignore').copy()
    x['性别'] = x['性别'].map({'女':0.,'男':1.})
    x['性别未知'] = x['性别'].isna().astype(float)
    base = x.copy()
    if variant in ('missing', 'combined'):
        x['缺失项目数'] = base.isna().sum(axis=1)
        for name, cols in [('肝功能',['*天门冬氨酸氨基转换酶','*丙氨酸氨基转换酶','*总蛋白']),
                           ('血脂',['甘油三酯','总胆固醇','高密度脂蛋白胆固醇','低密度脂蛋白胆固醇']),
                           ('肾功能',['尿素','肌酐','尿酸']),('乙肝',HEP)]:
            x[name+'缺失数'] = base[cols].isna().sum(axis=1)
    if variant in ('ratios', 'combined'):
        for name, a, b in [('TG_HDL','甘油三酯','高密度脂蛋白胆固醇'),
                           ('ALT_AST','*丙氨酸氨基转换酶','*天门冬氨酸氨基转换酶'),
                           ('NEUT_LYMPH','中性粒细胞%','淋巴细胞%')]:
            x[name] = base[a] / base[b].where(base[b] > 1e-6)
    if variant in ('counts', 'combined'):
        for c in ['中性粒细胞%','淋巴细胞%','单核细胞%','嗜酸细胞%','嗜碱细胞%']:
            x[c+'估算计数'] = base['白细胞计数'] * base[c] / 100.
    if variant in ('interactions', 'combined'):
        for c in ['甘油三酯','高密度脂蛋白胆固醇','尿酸','*丙氨酸氨基转换酶']:
            x['年龄x'+c] = base['年龄'] * base[c]
        x['年龄平方'] = base['年龄'] ** 2
    if variant != 'all':
        x = x.drop(columns=HEP)
    return x.replace([np.inf, -np.inf], np.nan)


def preprocess(spec):
    steps = [('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True))]
    if spec['family'] in ('ridge','nn'):
        steps.append(('scale', StandardScaler()))
    if spec.get('pca', False):
        steps.append(('pca', PCA(n_components=.98, svd_solver='full')))
    return Pipeline(steps)


def estimator(spec):
    family = spec['family']
    p = spec.get('params', {})
    if family == 'xgb':
        m = XGBRegressor(objective='reg:squarederror', tree_method='hist', n_jobs=2, random_state=23, **p)
    elif family == 'lgb':
        m = LGBMRegressor(objective='regression', verbosity=-1, deterministic=True,
                          force_col_wise=True, n_jobs=2, random_state=23, **p)
    elif family == 'hist':
        m = HistGradientBoostingRegressor(early_stopping=False, random_state=23, **p)
    elif family == 'forest':
        m = RandomForestRegressor(n_jobs=2, random_state=23, **p)
    elif family == 'ridge':
        m = Ridge(**p)
    else:
        raise ValueError(family)
    return m if spec.get('native', False) else Pipeline([('preprocess', preprocess(spec)), ('model', m)])


def build_nn(n, spec):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    layers = [tf.keras.layers.Input(shape=(n,))]
    for width in spec['widths']:
        layers.extend([tf.keras.layers.Dense(width, activation='relu',
                       kernel_regularizer=tf.keras.regularizers.l2(.001)),
                       tf.keras.layers.Dropout(spec['dropout'])])
    layers.append(tf.keras.layers.Dense(1))
    m = tf.keras.Sequential(layers)
    m.compile(optimizer=tf.keras.optimizers.Adam(.001), loss='mse')
    return m


def fit_nn_predict(spec, train, test, epochs=None):
    import tensorflow as tf
    x = features(train, spec['variant'])
    z = features(test, spec['variant'])
    y = train['血糖'].to_numpy()
    if epochs is None:
        # Early stopping uses a split WITHIN the outer training fold. All fitted
        # preprocessing statistics also come from that internal fit subset.
        ii, jj = train_test_split(np.arange(len(x)), test_size=.2, random_state=23)
        pre = preprocess(spec)
        xi = pre.fit_transform(x.iloc[ii])
        xj = pre.transform(x.iloc[jj])
        m = build_nn(xi.shape[1], spec)
        history = m.fit(xi, y[ii], validation_data=(xj, y[jj]), batch_size=64,
                        epochs=100, verbose=0,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12)])
        epochs = int(np.argmin(history.history['val_loss']) + 1)
    pre = preprocess(spec)
    xx = pre.fit_transform(x)
    zz = pre.transform(z)
    m = build_nn(xx.shape[1], spec)
    m.fit(xx, y, epochs=epochs, batch_size=64, verbose=0)
    p = m.predict(zz, verbose=0).reshape(-1)
    return p, int(epochs), (pre, m)


def score(y, p):
    mse = float(mean_squared_error(y, p))
    return dict(MSE=mse, RMSE=float(np.sqrt(mse)), MAE=float(mean_absolute_error(y,p)),
                R2=float(r2_score(y,p)))


def cv(spec, train, seed=23):
    token = hashlib.sha256(json.dumps([SOURCE_HASH, spec, seed], sort_keys=True).encode()).hexdigest()[:20]
    path = CACHE / (token+'.npz')
    if path.exists():
        c = np.load(path)
        return c['prediction'], c['fold_mse'].tolist(), c['epochs'].tolist()
    y = train['血糖'].to_numpy()
    p = np.empty(len(train)); losses = []; epochs = []
    for i, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=seed).split(train)):
        if spec['family'] == 'nn':
            pred, ep, _ = fit_nn_predict(spec, train.iloc[tr], train.iloc[va])
            epochs.append(ep)
        else:
            m = estimator(spec)
            x = features(train, spec['variant'])
            m.fit(x.iloc[tr], y[tr])
            pred = m.predict(x.iloc[va])
        p[va] = pred
        losses.append(float(mean_squared_error(y[va], pred)))
        if spec['family'] == 'nn':
            print('NN fold', spec['name'], seed, i+1, 'epochs', ep, 'MSE', losses[-1], flush=True)
    assert np.isfinite(p).all()
    np.savez_compressed(path, prediction=p, fold_mse=losses, epochs=epochs)
    return p, losses, epochs


BASE = dict(name='baseline_xgb', family='xgb', variant='no_hepatitis', native=False,
            params=dict(n_estimators=350,learning_rate=.02,max_depth=4,min_child_weight=30,
                        reg_lambda=10,subsample=.8,colsample_bytree=.8))


def main():
    start = time.time()
    raw = pd.read_csv(ROOT/'data/d_train_20180102.csv')
    a = pd.read_csv(ROOT/'data/d_test_A_20180102.csv')
    a['血糖'] = pd.read_csv(ROOT/'data/d_answer_a_20180128.csv').iloc[:,0].to_numpy()
    full = pd.concat([raw,a],ignore_index=True)
    train, valid = train_test_split(full,test_size=.3,random_state=23)
    assert (len(train),len(valid),len(full)) == (4649,1993,6642)
    assert set(train.id).isdisjoint(valid.id) and full.id.is_unique
    # Test labels remain unconsumed until ALL selection has finished.
    y = train['血糖'].to_numpy()
    oof = {}; specs = {}; rows = []; epoch_records = {}
    def evaluate(s, stage):
        p, losses, epochs = cv(s, train)
        oof[s['name']] = p; specs[s['name']] = s; epoch_records[s['name']] = epochs
        row = dict(stage=stage, name=s['name'], **score(y,p), fold_MSE=losses)
        rows.append(row)
        dump(OUT/'progress.json',rows)
        print(stage,s['name'],'CV MSE',round(row['MSE'],6),'elapsed',round(time.time()-start),flush=True)
        return row
    stage1 = [BASE]
    for variant in ['all','missing','ratios','counts','interactions','combined']:
        stage1.append({**BASE,'name':'feature_'+variant,'variant':variant})
    stage1.append({**BASE,'name':'feature_native_missing','native':True})
    plan = dict(source_sha256=SOURCE_HASH, train_rows=4649, validation_rows=1993,
                cv_seeds=[23,2026],stage1=stage1, stage2_budget=dict(xgb=100,lgb=12,hist=8),
                stage3='pair weights .25/.5/.75 and equal triples on train OOF',
                stage4='4 small MLPs; 5 outer folds; stopping on internal 20%; max100 patience12; batch64',
                assessment='Known development/B are descriptive only; no evaluation-label selection; nonnested CV selection scores.')
    dump(OUT/'plan.json',plan)
    first = [evaluate(s,1) for s in stage1]
    first.sort(key=lambda r:r['MSE'])
    feature_winner = first[0]['name']
    variants = list(dict.fromkeys(['no_hepatitis']+[specs[r['name']]['variant'] for r in first[:3]]))
    # Jointly sample representations, missing handling, structure and regularization.
    space = dict(n_estimators=randint(200,1001),learning_rate=loguniform(.01,.12),
                 max_depth=randint(2,7),min_child_weight=loguniform(5,120),
                 reg_lambda=loguniform(.1,150),reg_alpha=[0.,.1,1.,10.],
                 gamma=[0.,.05,.2,1.],subsample=[.65,.8,1.],colsample_bytree=[.65,.8,1.])
    rng = np.random.RandomState(23)
    candidates = []
    for i,p in enumerate(ParameterSampler(space,n_iter=100,random_state=23)):
        p = {k:v.item() if isinstance(v,np.generic) else v for k,v in p.items()}
        candidates.append(dict(name=f'joint_xgb_{i:03}',family='xgb',params=p,
                               variant=variants[rng.randint(len(variants))],native=bool(rng.randint(2))))
    for i,p in enumerate(ParameterSampler(dict(n_estimators=[250,500,800],learning_rate=[.015,.03,.06],
            num_leaves=[7,15,31],min_child_samples=[20,40,80],reg_lambda=[1.,10.,50.],
            colsample_bytree=[.7,1.]), n_iter=12,random_state=23)):
        candidates.append(dict(name=f'joint_lgb_{i:02}',family='lgb',params=p,
                               variant=variants[rng.randint(len(variants))],native=True))
    for i,p in enumerate(ParameterSampler(dict(max_iter=[200,400,700],learning_rate=[.02,.05],
            max_leaf_nodes=[7,15],min_samples_leaf=[20,50,80],l2_regularization=[1.,10.,50.]),
            n_iter=8,random_state=23)):
        candidates.append(dict(name=f'joint_hist_{i:02}',family='hist',params=p,
                               variant=variants[rng.randint(len(variants))],native=True))
    dump(OUT/'joint_search_plan.json',candidates)
    second = [evaluate(s,2) for s in candidates]
    # Include the feature-stage candidates, so an unsuccessful search cannot
    # silently replace an already better training-CV baseline.
    joint_winner = min(first+second,key=lambda r:r['MSE'])['name']
    third = [evaluate(dict(name='frozen_forest',family='forest',variant='all',
                          params=dict(n_estimators=200,min_samples_leaf=7,max_features=.6)),3),
             evaluate(dict(name='frozen_ridge',family='ridge',variant='all',params=dict(alpha=100)),3)]
    members = [joint_winner,'frozen_forest','frozen_ridge']
    for fam in ['xgb','lgb','hist']:
        winner = min([r for r in first+second if specs[r['name']]['family']==fam],key=lambda r:r['MSE'])['name']
        if winner not in members: members.append(winner)
    blend_rows = []
    for a,b in itertools.combinations(members,2):
        for w in [.25,.5,.75]:
            name=f'blend_{len(blend_rows):02}'
            p=w*oof[a]+(1-w)*oof[b]
            spec=dict(name=name,family='blend',members=[a,b],weights=[w,1-w])
            oof[name]=p; specs[name]=spec
            blend_rows.append(dict(stage=3,name=name,**score(y,p)))
    for group in itertools.combinations(members,3):
        name=f'blend_{len(blend_rows):02}'
        p=np.mean([oof[m] for m in group],axis=0)
        specs[name]=dict(name=name,family='blend',members=list(group),weights=[1/3]*3)
        oof[name]=p
        blend_rows.append(dict(stage=3,name=name,**score(y,p)))
    dump(OUT/'blends.json',blend_rows)
    blend_winner=min(blend_rows,key=lambda r:r['MSE'])['name']
    print('STAGE3 winner',blend_winner,specs[blend_winner],flush=True)
    # Stage 4 starts only after the blend grid is fixed and evaluated.
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    fourth=[]
    for widths,dropout in [([64,32],.1),([128,64,16],.2)]:
        for pca in [False,True]:
            s=dict(name=f'nn_{widths[0]}_'+('pca' if pca else 'raw'),family='nn',variant='all',
                   widths=widths,dropout=dropout,pca=pca)
            fourth.append(evaluate(s,4))
    nn_winner=min(fourth,key=lambda r:r['MSE'])['name']
    finalists=list(dict.fromkeys(['baseline_xgb',feature_winner,joint_winner,blend_winner,nn_winner]))
    # Confirmation is still selection, not an independent test of the search.
    confirmation={}; final_rows=[]
    def confirm(name):
        if name in confirmation:return confirmation[name]
        s=specs[name]
        if s['family']=='blend':
            p=sum(w*confirm(m) for m,w in zip(s['members'],s['weights']))
        else:
            p,loss,ep=cv(s,train,2026)
            if s['family']=='nn':epoch_records[name]+=ep
        confirmation[name]=p
        return p
    for name in finalists:
        p=confirm(name)
        m1=score(y,oof[name])['MSE'];m2=score(y,p)['MSE']
        final_rows.append(dict(name=name,CV_seed23_MSE=m1,CV_seed2026_MSE=m2,mean_CV_MSE=(m1+m2)/2))
        print('CONFIRM',final_rows[-1],flush=True)
    selected=min(final_rows,key=lambda r:r['mean_CV_MSE'])['name']
    for s in specs.values():
        if s['family']=='nn':s['frozen_epochs']=int(np.median(epoch_records[s['name']]))
    stages=dict(baseline='baseline_xgb',feature=feature_winner,joint=joint_winner,
                blend=blend_winner,small_nn=nn_winner,selected=selected)
    frozen=dict(source_sha256=SOURCE_HASH,stage_winners=stages,selected=selected,
                confirmation=final_rows,specifications=specs,
                note='All selection finished before development/B evaluation; CV is not nested.')
    dump(OUT/'frozen_selection.json',frozen)
    pd.DataFrame(rows).to_csv(OUT/'training_cv.csv',index=False)
    pd.DataFrame(final_rows).to_csv(OUT/'confirmation.csv',index=False)
    oof_frame=pd.DataFrame({'id':train.id.to_numpy(),'actual':y})
    for name in finalists:
        oof_frame[name+'_seed23']=oof[name]
        oof_frame[name+'_seed2026']=confirmation[name]
    oof_frame.to_csv(OUT/'oof_finalists.csv',index=False)
    pd.DataFrame({n:oof[n]-y for n in members}).corr().to_csv(OUT/'residual_correlations.csv')
    # Only now access holdout labels and B files; freeze all epochs and weights.
    b=pd.read_csv(ROOT/'data/d_test_B_20180128.csv')
    b['血糖']=pd.read_csv(ROOT/'data/d_answer_b_20180130.csv',header=None).iloc[:,0].to_numpy()
    assert len(b)==1000 and set(full.id).isdisjoint(b.id)
    evaluations=[]
    for label,fit_data,test in [('development',train,valid),('B',full,b)]:
        prediction_cache={}
        def predict(name):
            if name in prediction_cache:return prediction_cache[name]
            s=specs[name]
            if s['family']=='blend':
                pred=sum(w*predict(m) for m,w in zip(s['members'],s['weights']))
            elif s['family']=='nn':
                pred,_,(pre,model)=fit_nn_predict(s,fit_data,test,epochs=s['frozen_epochs'])
                joblib.dump(pre,OUT/f'{label}_{name}_preprocessor.joblib')
                model.save(OUT/f'{label}_{name}.keras')
            else:
                model=estimator(s)
                model.fit(features(fit_data,s['variant']),fit_data['血糖'].to_numpy())
                pred=model.predict(features(test,s['variant']))
                joblib.dump(model,OUT/f'{label}_{name}.joblib')
            assert np.isfinite(pred).all() and len(pred)==len(test)
            prediction_cache[name]=pred
            return pred
        predictions=pd.DataFrame({'id':test.id.to_numpy(),'actual':test['血糖'].to_numpy()})
        for name in finalists:
            p=predict(name)
            predictions[name]=p
            evaluations.append(dict(dataset=label,name=name,**score(test['血糖'],p)))
            print('EVALUATION',evaluations[-1],flush=True)
        predictions.to_csv(OUT/f'{label}_predictions.csv',index=False)
    pd.DataFrame(evaluations).to_csv(OUT/'evaluation.csv',index=False)
    expected={'development':2.228807860191054,'B':1.2527435992815599}
    for label,baseline in expected.items():
        actual=next(r['MSE'] for r in evaluations if r['dataset']==label and r['name']=='baseline_xgb')
        assert np.isclose(actual,baseline,rtol=1e-8,atol=1e-10),(label,actual,baseline)
    # Paired intervals describe these fitted models, not search/training variance.
    intervals=[]
    for label in ['development','B']:
        d=pd.read_csv(OUT/f'{label}_predictions.csv')
        baseline=(d['actual'].to_numpy()-d['baseline_xgb'].to_numpy())**2
        rng=np.random.default_rng(23)
        for name in finalists:
            if name=='baseline_xgb':continue
            delta=(d['actual'].to_numpy()-d[name].to_numpy())**2-baseline
            samples=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(3000)])
            intervals.append(dict(dataset=label,name=name,MSE_difference=float(delta.mean()),
                                  low95=float(np.quantile(samples,.025)),high95=float(np.quantile(samples,.975))))
    pd.DataFrame(intervals).to_csv(OUT/'paired_intervals.csv',index=False)
    bands=[]
    for label in ['development','B']:
        d=pd.read_csv(OUT/f'{label}_predictions.csv')
        for name in ['baseline_xgb',selected]:
            for lo,hi in [(0,6),(6,10),(10,float('inf'))]:
                z=d[(d.actual>=lo)&(d.actual<hi)]
                err=(z.actual-z[name])**2
                bands.append(dict(dataset=label,name=name,band=f'{lo}-{hi}',rows=len(z),MSE=err.mean(),
                                  squared_error_share=err.sum()/((d.actual-d[name])**2).sum(),
                                  mean_prediction_minus_actual=(z[name]-z.actual).mean()))
    pd.DataFrame(bands).to_csv(OUT/'target_bands.csv',index=False)
    dump(OUT/'completed.json',dict(seconds=time.time()-start,source_sha256=SOURCE_HASH,
          baseline_reproduced=True,selected=selected,stage_winners=stages))
    print('COMPLETE',selected,'seconds',round(time.time()-start),flush=True)


if __name__=='__main__':
    main()
