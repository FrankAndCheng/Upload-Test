"""第五阶段只读复算：不训练模型、不执行 Notebook，所有输出限于本目录。
重用第一阶段指标核对口径，并从原始 CSV、保存预测和配置重新计算。
运行：使用含 NumPy/pandas 的 Python 执行本文件。
"""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
hashes=json.loads((ROOT/'research_analysis/stage1/source_manifest.json').read_text())
catalog=[]
for p in ROOT.glob('*.ipynb'):
    for cell in json.loads(p.read_text())['cells']:
        catalog.append({'error_count':sum(o.get('output_type')=='error' for o in cell.get('outputs',[]))})
def save_csv(name,rows):
    pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def write_json(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))
train=pd.read_csv(ROOT/'data/d_train_20180102.csv');a=pd.read_csv(ROOT/'data/d_test_A_20180102.csv');b=pd.read_csv(ROOT/'data/d_test_B_20180128.csv')
a['血糖']=pd.read_csv(ROOT/'data/d_answer_a_20180128.csv').iloc[:,0].to_numpy();b['血糖']=pd.read_csv(ROOT/'data/d_answer_b_20180130.csv',header=None).iloc[:,0].to_numpy()
full=pd.concat([train,a],ignore_index=True); all_data=pd.concat([full,b],ignore_index=True)
permutation=np.random.RandomState(23).permutation(len(full));nv=int(np.ceil(.3*len(full)))
valid=full.iloc[permutation[:nv]];fit=full.iloc[permutation[nv:]]
ids=pd.read_csv(ROOT/'review/sequential_experiments/split_ids.csv')
print('split labels',ids.groupby('split').size().to_dict())
checks=[];metric_rows=[]
def check(name,condition,detail=''):
    checks.append(dict(check=name,passed=bool(condition),detail=str(detail)))
check('all_raw_ids_unique',all_data.id.is_unique)
check('split_training_ids',set(ids[ids.split=='train'].id)==set(fit.id))
check('split_development_ids',set(ids[ids.split!='train'].id)==set(valid.id))
check('preprocessing_training_ids',set(pd.read_csv(ROOT/'review/preprocessing_ablation/train_ids.csv').id)==set(fit.id))
truth=all_data.set_index('id')['血糖']
def metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);sse=np.sum((y-p)**2)
    return dict(MSE=float(sse/len(y)),RMSE=float(np.sqrt(sse/len(y))),MAE=float(np.mean(abs(y-p))),R2=float(1-sse/np.sum((y-y.mean())**2)))
def cmp_metrics(group,key,y,p,expected,tol=5e-7):
    m=metrics(y,p)
    for metric,value in m.items():
        if metric not in expected:continue
        error=abs(value-float(expected[metric]));metric_rows.append(dict(group=group,key=key,metric=metric,recomputed=value,saved=float(expected[metric]),absolute_difference=error,passed=error<=tol))
    return m
def read_predictions(relative,target='actual'):
    d=pd.read_csv(ROOT/relative)
    check(relative+' unique ids',d.id.is_unique)
    check(relative+' targets match raw',np.array_equal(d[target].to_numpy(),truth.loc[d.id].to_numpy()))
    check(relative+' finite',np.isfinite(d.select_dtypes('number').to_numpy()).all())
    return d
ext=read_predictions('review/external_b_comparison/predictions.csv'); ext_m=pd.read_csv(ROOT/'review/external_b_comparison/comparison.csv')
for r in ext_m.to_dict('records'):cmp_metrics('six_models_B',r['model'],ext.actual,ext[r['model']],{m:r['B_'+m] for m in ['MSE','RMSE','MAE','R2']})
seq={};pre={}
seq_scores=pd.read_csv(ROOT/'review/sequential_experiments/evaluation.csv');ps=pd.read_csv(ROOT/'review/preprocessing_ablation/holdout_metrics.csv');fs=pd.read_csv(ROOT/'review/preprocessing_ablation/fusion_holdout.csv')
for ds in ['development','B']:
    seq[ds]=d=read_predictions(f'review/sequential_experiments/{ds}_predictions.csv')
    for r in seq_scores[seq_scores.dataset==ds].to_dict('records'):cmp_metrics('sequential_'+ds,r['name'],d.actual,d[r['name']],r)
    check('sequential blend arithmetic '+ds,np.allclose(d.nn_blend_06,.25*d.nn_64_nohep_raw+.75*d.blend_33,rtol=0,atol=1e-12))
    pre[ds]=d=read_predictions(f'review/preprocessing_ablation/{ds}_predictions.csv')
    for r in ps[ps.dataset==ds].to_dict('records'):cmp_metrics('preprocessing_'+ds,r['model']+'__'+r['mode'],d.actual,d[r['model']+'__'+r['mode']],r)
    for r in fs[fs.dataset==ds].to_dict('records'):cmp_metrics('fusion_'+ds,r['version'],d.actual,d['Fusion__'+r['version']],r)
    for version,lgb in [('reference','native'),('recommended','raw')]:
        p=(d.XGBoost_optimized__raw+d['LightGBM__'+lgb]+d.Ridge__scale+d.NeuralNetwork_small__scale)/4
        check('preprocessing fusion arithmetic '+ds+' '+version,np.allclose(p,d['Fusion__'+version],rtol=0,atol=1e-12))
oof=read_predictions('review/preprocessing_ablation/oof_predictions.csv')
cv=pd.read_csv(ROOT/'review/preprocessing_ablation/cv_summary.csv');fold=pd.read_csv(ROOT/'review/preprocessing_ablation/fold_results.csv')
for r in cv.to_dict('records'):
    values=[]
    for seed in [23,2026]:
        p=oof[f'{r["model"]}__{r["mode"]}__{seed}'];m=metrics(oof.actual,p)['MSE'];values.append(m)
        cmp_metrics('preprocessing_cv',r['model']+'__'+r['mode']+'__'+str(seed),oof.actual,p,{'MSE':r[f'seed{seed}_MSE']})
        perm=np.random.RandomState(seed).permutation(len(oof))
        for k,inds in enumerate(np.array_split(perm,5)):
            saved=fold[(fold.model==r['model'])&(fold['mode']==r['mode'])&(fold.seed==seed)&(fold['fold']==k)].MSE.iloc[0]
            value=metrics(oof.actual.iloc[inds],p.iloc[inds])['MSE']
            metric_rows.append(dict(group='paired_fold_MSE',key=f'{r["model"]}__{r["mode"]}__{seed}__{k}',metric='MSE',recomputed=value,saved=saved,absolute_difference=abs(value-saved),passed=abs(value-saved)<5e-7))
    check('CV mean '+r['model']+' '+r['mode'],abs(np.mean(values)-r['CV_MSE'])<1e-10)
soof=read_predictions('review/sequential_experiments/oof_finalists.csv')
confirmation=pd.read_csv(ROOT/'review/sequential_experiments/confirmation.csv')
for r in confirmation.to_dict('records'):
    for seed in [23,2026]:cmp_metrics('sequential_cv',r['name']+'_'+str(seed),soof.actual,soof[f'{r["name"]}_seed{seed}'],{'MSE':r[f'CV_seed{seed}_MSE']})
    check('confirmation mean '+r['name'],abs((r['CV_seed23_MSE']+r['CV_seed2026_MSE'])/2-r['mean_CV_MSE'])<1e-12)
for proto in ['same_validation','external']:
    result=json.loads((ROOT/f'review/optimization/{proto}/results.json').read_text())
    for ds,r in result['evaluations'].items():
        d=read_predictions(f'review/optimization/{proto}/predictions_{ds}.csv')
        cmp_metrics('historical_'+proto,ds,d.actual,d.prediction,r['selected'])
    d=read_predictions(f'review/optimization/{proto}/oof_predictions.csv',target='target')
    for r in pd.read_csv(ROOT/f'review/optimization/{proto}/cv_results.csv').to_dict('records'):
        cmp_metrics('historical_cv_'+proto,r['name'],d.target,d[r['name']],r)

seq_pair=pd.read_csv(ROOT/'review/sequential_experiments/paired_intervals.csv')
for r in seq_pair.to_dict('records'):
    d=seq[r['dataset']];delta=(d.actual-d[r['name']])**2-(d.actual-d.baseline_xgb)**2
    check('paired MSE mean '+r['dataset']+' '+r['name'],abs(delta.mean()-r['MSE_difference'])<5e-7, f'abs difference={abs(delta.mean()-r["MSE_difference"]):.3g}; tolerance=5e-7 for CSV float precision')
check('no_notebook_error_outputs',all(r['error_count']==0 for r in catalog))
frozen=json.loads((ROOT/'review/sequential_experiments/frozen_selection.json').read_text())
check('training_only_final_choice',confirmation.sort_values('mean_CV_MSE').iloc[0]['name']==frozen['selected']=='nn_blend_06')
input_audit={}
for p in [ROOT/'review/sequential_experiments/input_manifest.json',ROOT/'review/preprocessing_ablation/plan.json',ROOT/'glucose_project/source_catalog.json']:
    value=json.loads(p.read_text());input_audit[str(p.relative_to(ROOT))]=value
    def scan(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) and (ROOT/k).is_file():
                    current=hashlib.sha256((ROOT/k).read_bytes()).hexdigest()
                    if k=='paper.ipynb' and current!=v:
                        nb=json.loads((ROOT/k).read_text());nb['cells']=nb['cells'][:71]
                        prefix=hashlib.sha256((json.dumps(nb,ensure_ascii=False,indent=1)+'\n').encode()).hexdigest()
                        check('historical paper.ipynb first71 exact snapshot',prefix==v,'Full current notebook differs; removing the two appended Markdown cells reproduces historical file SHA256 exactly.')
                    else:check(str(p.relative_to(ROOT))+' hash '+k,current==v)
                else:scan(v)
        elif isinstance(obj,list):
            for v in obj:scan(v)
    scan(value)
for name,source in [('review/sequential_experiments/frozen_selection.json','review/sequential_experiments/run_experiments.py'),('review/preprocessing_ablation/plan.json','review/preprocessing_ablation/run_ablation.py')]:
    check('source snapshot '+source,json.loads((ROOT/name).read_text())['source_sha256']==hashes[source])

manifest_changed=[k for k,v in hashes.items() if not (ROOT/k).is_file() or hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=v]
metadata_changed=[k for k in manifest_changed if Path(k).name=='.DS_Store']
changed=[k for k in manifest_changed if Path(k).name!='.DS_Store']
check('original_research_unchanged_excluding_OS_metadata',not changed,str(changed))
save_csv('metric_recalculation.csv',metric_rows)
write_json('consistency_checks.json',checks)
summary=dict(metric_comparisons=len(metric_rows),metrics_passed=sum(x['passed'] for x in metric_rows),max_metric_abs_difference=max(x['absolute_difference'] for x in metric_rows),checks=len(checks),checks_passed=sum(x['passed'] for x in checks),retrained=False,executed_notebooks=False,research_files_unchanged=not changed,manifest_files=len(hashes),research_files=sum(Path(k).name!='.DS_Store' for k in hashes),changed_OS_metadata=metadata_changed,numpy_version=np.__version__,pandas_version=pd.__version__)
write_json('recomputation_summary.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))
assert all(x['passed'] for x in checks),[x for x in checks if not x['passed']]
assert all(x['passed'] for x in metric_rows),[x for x in metric_rows if not x['passed']]
