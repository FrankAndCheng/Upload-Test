"""Fixed post-selection diagnostics; does not tune models on holdout labels."""
import json,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/paper-matplotlib')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from paper_optimized import load_data,make_features,candidate_models,fit_predict
OUT=ROOT/'review/optimization'
def paired_interval(y,new,old,seed=23):
    # Conditional row-level bootstrap, not uncertainty over training/search or temporal dependence.
    delta=(y-old)**2-(y-new)**2;rng=np.random.default_rng(seed)
    means=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(3000)])
    return {'MSE_reduction':float(delta.mean()),'paired_row_bootstrap_95pct_CI':np.quantile(means,[.025,.975]).tolist(),'replicates':3000,'limitation':'Conditional on fitted models and these rows; excludes training/selection variation and date clustering.'}
tr,a,b=load_data();all_data=pd.concat([tr,a],ignore_index=True)
saved=pd.read_csv(ROOT/'target.csv',index_col=0).iloc[:,0]
fit,valid=train_test_split(np.arange(len(saved)),test_size=.3,random_state=23)
raw_train=all_data.loc[saved.index].iloc[fit];raw_valid=all_data.loc[saved.index].iloc[valid]
assert set(raw_train.id).isdisjoint(set(raw_valid.id))
pca=pd.read_csv(ROOT/'z_score_pca.csv',index_col=0)
legacy=XGBRegressor(objective='reg:squarederror',n_estimators=40,learning_rate=.1,max_depth=3,min_child_weight=3,subsample=.9,colsample_bytree=.7,random_state=0,n_jobs=2)
legacy.fit(pca.iloc[fit],saved.iloc[fit]);old=legacy.predict(pca.iloc[valid])
same=pd.read_csv(OUT/'same_validation/predictions_validation.csv');assert np.array_equal(same.id,raw_valid.id)
same['legacy_xgb_prediction']=old;same.to_csv(OUT/'same_validation/predictions_comparison.csv',index=False)
ci={'same_vs_legacy_xgb':paired_interval(same.actual.to_numpy(),same.prediction.to_numpy(),old)}
lookup={c['name']:c for c in candidate_models()}
for name,d in [('A',a),('B',b)]:
    p=pd.read_csv(OUT/f'external/predictions_{name}.csv');ref,_=fit_predict(lookup['ridge_100'],tr,d)
    ci[name+'_vs_ridge']=paired_interval(p.actual.to_numpy(),p.prediction.to_numpy(),ref)
(OUT/'uncertainty.json').write_text(json.dumps(ci,indent=2))
subgroups=[]
for name,path in [('same_validation',OUT/'same_validation/predictions_validation.csv'),('A',OUT/'external/predictions_A.csv'),('B',OUT/'external/predictions_B.csv')]:
    p=pd.read_csv(path);sse=((p.actual-p.prediction)**2).sum();p['target_band']=pd.cut(p.actual,[-np.inf,6,10,np.inf],right=False,labels=['<6','6-10','>=10'])
    for band,g in p.groupby('target_band',observed=True):
        err=g.prediction-g.actual
        subgroups.append({'dataset':name,'target_band':str(band),'n':len(g),'MSE':float((err**2).mean()),'MAE':float(err.abs().mean()),'mean_prediction_minus_actual':float(err.mean()),'fraction_total_SSE':float((err**2).sum()/sse)})
pd.DataFrame(subgroups).to_csv(OUT/'target_band_errors.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4.2),dpi=160)
labels=['Linear + PCA','XGB + PCA (40)','Original NN','Raw boosted blend'];vals=[1.7140556898185924,1.6592348445327099,1.65340,float(((same.actual-same.prediction)**2).mean())]
axes[0].bar(labels,vals,color=['#a6b6c4']*3+['#159d87']);axes[0].set_ylabel('MSE (lower is better)');axes[0].set_title('Same 1,561 validation rows');axes[0].tick_params(axis='x',labelrotation=20);axes[0].set_ylim(0,1.95)
for i,v in enumerate(vals):axes[0].text(i,v+.025,f'{v:.4f}',ha='center',fontsize=9)
axes[1].scatter(same.actual,same.prediction,s=10,alpha=.35,color='#159d87');maxv=max(same.actual.max(),same.prediction.max());axes[1].plot([3,maxv],[3,maxv],'--',color='gray');axes[1].set(xlabel='Actual glucose',ylabel='Predicted glucose',title='Remaining error concentrates in the tail')
fig.tight_layout();fig.savefig(OUT/'validation_comparison.png');plt.close(fig)
print(json.dumps(ci,indent=2));print(pd.DataFrame(subgroups).to_string(index=False))
