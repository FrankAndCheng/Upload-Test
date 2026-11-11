"""Leakage-controlled blood-glucose regression experiments.
Run with Conda base Python: python paper_optimized.py --protocol all
The original paper.ipynb and data are never modified.
"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/paper-matplotlib')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'review/optimization'
HEPATITIS = ['乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体']


def load_data():
    train = pd.read_csv(ROOT/'data/d_train_20180102.csv')
    a = pd.read_csv(ROOT/'data/d_test_A_20180102.csv')
    a['血糖'] = pd.read_csv(ROOT/'data/d_answer_a_20180128.csv').iloc[:,0].to_numpy()
    b = pd.read_csv(ROOT/'data/d_test_B_20180128.csv')
    b['血糖'] = pd.read_csv(ROOT/'data/d_answer_b_20180130.csv',header=None).iloc[:,0].to_numpy()
    assert len(a)==1000 and len(b)==1000
    for d in (train,a,b):
        assert d.id.is_unique and d['血糖'].notna().all()
    return train,a,b


def make_features(d, variant='raw'):
    """Only row-wise deterministic transforms; no fitted statistics or target usage."""
    x = d.drop(columns=['id','体检日期','血糖'],errors='ignore').copy()
    x['性别'] = x['性别'].map({'女':0.,'男':1.})
    if variant=='no_hepatitis':
        x = x.drop(columns=HEPATITIS)
    if variant=='engineered':
        x['缺失项目数'] = x.isna().sum(axis=1)
        for name,cols in [('肝功能', ['*天门冬氨酸氨基转换酶','*丙氨酸氨基转换酶','*总蛋白']),
                          ('血脂',['甘油三酯','总胆固醇','高密度脂蛋白胆固醇','低密度脂蛋白胆固醇']),
                          ('肾功能',['尿素','肌酐','尿酸']),('乙肝',HEPATITIS)]:
            x[name+'缺失数'] = x[cols].isna().sum(axis=1)
        for name,num,den in [('TG_HDL','甘油三酯','高密度脂蛋白胆固醇'),
                             ('ALT_AST','*丙氨酸氨基转换酶','*天门冬氨酸氨基转换酶'),
                             ('NEUT_LYMPH','中性粒细胞%','淋巴细胞%')]:
            x[name] = x[num]/x[den].where(x[den]>1e-6)
        for percent in ['中性粒细胞%','淋巴细胞%','单核细胞%','嗜酸细胞%','嗜碱细胞%']:
            x[percent+'估算计数'] = x['白细胞计数']*x[percent]/100.
    return x.replace([np.inf,-np.inf],np.nan)


def candidate_models():
    rows=[]
    def add(name,model,variant='raw'):rows.append(dict(name=name,model=model,variant=variant))
    for alpha in [100.,1000.]:
        add(f'ridge_{int(alpha)}',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),Ridge(alpha=alpha)))
    for leaves,reg,it in [(7,10.,200),(7,30.,350),(15,30.,200)]:
        add(f'hist_l{leaves}_r{int(reg)}_n{it}',HistGradientBoostingRegressor(max_iter=it,learning_rate=.03,max_leaf_nodes=leaves,min_samples_leaf=40,l2_regularization=reg,early_stopping=False,random_state=23))
    for leaves,leaf,reg,it in [(7,40,10.,200),(7,60,30.,350),(15,40,20.,200),(15,60,30.,350)]:
        model=LGBMRegressor(objective='regression',n_estimators=it,learning_rate=.03,num_leaves=leaves,min_child_samples=leaf,reg_lambda=reg,verbosity=-1,n_jobs=2,random_state=23,deterministic=True,force_col_wise=True)
        add(f'lgb_l{leaves}_m{leaf}_r{int(reg)}_n{it}',model)
    add('xgb_original_parameters_raw',XGBRegressor(objective='reg:squarederror',n_estimators=40,learning_rate=.1,max_depth=3,min_child_weight=3,subsample=.9,colsample_bytree=.7,n_jobs=2,random_state=23,tree_method='hist'))
    for depth,child,reg,it in [(2,20,20.,250),(3,30,30.,250),(2,40,50.,400)]:
        add(f'xgb_d{depth}_m{child}_r{int(reg)}_n{it}',XGBRegressor(objective='reg:squarederror',n_estimators=it,learning_rate=.025,max_depth=depth,min_child_weight=child,reg_lambda=reg,subsample=.85,colsample_bytree=.85,n_jobs=2,random_state=23,tree_method='hist'))
    add('extra_trees',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),ExtraTreesRegressor(n_estimators=250,min_samples_leaf=12,max_features=.8,n_jobs=2,random_state=23)))
    lgb=next(c['model'] for c in rows if c['name']=='lgb_l7_m40_r10_n200')
    add('lgb_no_hepatitis',clone(lgb),'no_hepatitis')
    add('lgb_engineered',clone(lgb),'engineered')
    return rows


def metrics(y,p):
    return {'MSE':float(mean_squared_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),
            'MAE':float(mean_absolute_error(y,p)),'R2':float(r2_score(y,p))}


def cv_predictions(candidate,d,seed):
    x=make_features(d,candidate['variant']);y=d['血糖'].to_numpy();oof=np.empty(len(d));scores=[]
    for tr,va in KFold(5,shuffle=True,random_state=seed).split(x):
        m=clone(candidate['model']);m.fit(x.iloc[tr],y[tr]);oof[va]=m.predict(x.iloc[va]);scores.append(float(mean_squared_error(y[va],oof[va])))
    return oof,scores


def fit_predict(candidate,train,validation):
    m=clone(candidate['model']);m.fit(make_features(train,candidate['variant']),train['血糖'].to_numpy())
    return m.predict(make_features(validation,candidate['variant'])),m


def experiment(name,train,evaluation):
    """All selection uses training data. Evaluation labels are read after selection."""
    dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
    candidates=candidate_models();lookup={c['name']:c for c in candidates};y=train['血糖'].to_numpy()
    rows=[];oofs={}
    for c in candidates:
        p,folds=cv_predictions(c,train,23);oofs[c['name']]=p
        row={'name':c['name'],'variant':c['variant'],**metrics(y,p),'fold_MSE':folds}
        rows.append(row);print(name,c['name'],'CV MSE',round(row['MSE'],6),flush=True)
        pd.DataFrame(rows).to_json(dest/'search_progress.json',orient='records',indent=2,force_ascii=False)
    rows.sort(key=lambda r:r['MSE'])
    # Confirm the top 3 on an additional predetermined split, using train only.
    finalists=[]
    for row in rows[:3]:
        key=row['name'];p2,folds=cv_predictions(lookup[key],train,2026)
        # Averaging the two OOF losses measures split stability, not a doubled-size sample.
        score2=float(mean_squared_error(y,p2));score=(row['MSE']+score2)/2
        finalists.append({'name':key,'seed23_MSE':row['MSE'],'seed2026_MSE':score2,'mean_MSE':score,'seed2026_folds':folds})
    finalists.sort(key=lambda r:r['mean_MSE']);winner=finalists[0]['name']
    # A fixed equal-weight blend of the two best finalists is a single extra candidate.
    pair=[r['name'] for r in finalists[:2]]
    pair_second=[cv_predictions(lookup[k],train,2026)[0] for k in pair]
    blend_first=np.mean([oofs[k] for k in pair],axis=0)
    blend_second=np.mean(pair_second,axis=0)
    blend_cv=(mean_squared_error(y,blend_first)+mean_squared_error(y,blend_second))/2
    use_blend=bool(blend_cv<finalists[0]['mean_MSE'])
    selected=pair if use_blend else [winner]
    # Freeze decisions before calculating any holdout score.
    selection={'members':selected,'weights':[1/len(selected)]*len(selected),'finalists':finalists,'blend_CV_MSE':float(blend_cv),'training_rows':len(train),'search_candidates':len(rows),'protocol':'5-fold seed23 screening; top3 confirmed seed2026; equal-weight top2 blend compared on train OOF. CV is used for selection, not an unbiased final score.'}
    (dest/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
    pd.DataFrame(rows).to_csv(dest/'cv_results.csv',index=False)
    pd.DataFrame({'id':train.id,'target':y,**oofs}).to_csv(dest/'oof_predictions.csv',index=False)
    outputs={};fitted=[]
    for k in selected:
        c=lookup[k];m=clone(c['model']);m.fit(make_features(train,c['variant']),y)
        fitted.append({'name':k,'variant':c['variant'],'model':m,'weight':1/len(selected)})
    joblib.dump(fitted,dest/'selected_model.joblib')
    for label,holdout in evaluation.items():
        p=np.mean([q['model'].predict(make_features(holdout,q['variant'])) for q in fitted],axis=0)
        # Same source/rows as the selected model: a simple leakage-safe reference.
        ridgep,_=fit_predict(lookup['ridge_100'],train,holdout)
        outputs[label]={'rows':len(holdout),'selected':metrics(holdout['血糖'],p),'ridge_reference':metrics(holdout['血糖'],ridgep)}
        pd.DataFrame({'id':holdout.id,'actual':holdout['血糖'],'prediction':p,'residual':holdout['血糖']-p}).to_csv(dest/f'predictions_{label}.csv',index=False)
    result={'selection':selection,'evaluations':outputs}
    (dest/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def same_validation():
    tr,a,_=load_data();all_data=pd.concat([tr,a],ignore_index=True)
    saved_y=pd.read_csv(ROOT/'target.csv',index_col=0).iloc[:,0]
    assert np.array_equal(all_data.loc[saved_y.index,'血糖'].to_numpy(),saved_y.to_numpy())
    positions=np.arange(len(saved_y));fit,valid=train_test_split(positions,test_size=.3,random_state=23)
    subset=all_data.loc[saved_y.index].copy()
    train=subset.iloc[fit].copy();validation=subset.iloc[valid].copy()
    # Reproduce original deterministic linear / XGBoost baseline from cached PCA.
    pca=pd.read_csv(ROOT/'z_score_pca.csv',index_col=0)
    assert len(pca)==len(saved_y)
    from sklearn.linear_model import LinearRegression
    legacy={}
    for key,model in [('legacy_linear',LinearRegression()),('legacy_xgb_40',XGBRegressor(objective='reg:squarederror',n_estimators=40,learning_rate=.1,max_depth=3,min_child_weight=3,subsample=.9,colsample_bytree=.7,random_state=0,n_jobs=2))]:
        model.fit(pca.iloc[fit],saved_y.iloc[fit]);legacy[key]=metrics(saved_y.iloc[valid],model.predict(pca.iloc[valid]))
    result=experiment('same_validation',train,{'validation':validation});result['legacy_baselines']=legacy
    (OUT/'same_validation/results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def external():
    tr,a,b=load_data()
    result=experiment('external',tr,{'A':a,'B':b})
    selection=result['selection'];lookup={c['name']:c for c in candidate_models()}
    # Chronological stress test: a different sample population, not directly comparable to random validation.
    dates=pd.to_datetime(tr['体检日期'],dayfirst=True)
    threshold=dates.sort_values().iloc[int(.8*len(tr))]
    early=tr.loc[dates<threshold];late=tr.loc[dates>=threshold]
    predictions=[fit_predict(lookup[k],early,late)[0] for k in selection['members']]
    ridgep,_=fit_predict(lookup['ridge_100'],early,late)
    result['date_stress']={'cutoff':str(threshold.date()),'training_rows':len(early),'validation_rows':len(late),'selected':metrics(late['血糖'],np.mean(predictions,axis=0)),'ridge_reference':metrics(late['血糖'],ridgep)}
    (OUT/'external/results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',choices=['same','external','all'],default='all');args=parser.parse_args()
    if args.protocol in ['same','all']:same_validation()
    if args.protocol in ['external','all']:external()
