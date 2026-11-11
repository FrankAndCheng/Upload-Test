"""Targeted TRAIN-OOF-driven NN feature ablation; never select with holdouts."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import run_experiments as ex

OUT=Path(__file__).resolve().parent


def main():
    plan=json.loads((OUT/'nn_refinement_plan.json').read_text())
    assert plan['source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # Read only training records/labels and previously frozen train-CV choices.
    a=pd.read_csv(ex.ROOT/'data/d_test_A_20180102.csv')
    a['血糖']=pd.read_csv(ex.ROOT/'data/d_answer_a_20180128.csv').iloc[:,0].to_numpy()
    full=pd.concat([pd.read_csv(ex.ROOT/'data/d_train_20180102.csv'),a],ignore_index=True)
    train,valid=train_test_split(full,test_size=.3,random_state=23)
    f=json.loads((OUT/'frozen_selection.json').read_text())
    original=OUT/'frozen_selection_before_nn_refinement.json'
    if not original.exists():original.write_text(json.dumps(f,ensure_ascii=False,indent=2))
    f=json.loads(original.read_text())
    oof=pd.read_csv(OUT/'oof_finalists.csv');cvrows=pd.read_csv(OUT/'training_cv.csv')
    choices=[];predictions={};epochs={}
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    for pca in [False,True]:
        s=dict(name='nn_64_nohep_'+('pca' if pca else 'raw'),family='nn',variant='no_hepatitis',
               widths=[64,32],dropout=.1,pca=pca)
        p,loss,ep=ex.cv(s,train,23)
        row=dict(stage=4,name=s['name'],**ex.score(train['血糖'],p),fold_MSE=str(loss))
        choices.append(row);predictions[s['name']]=p;epochs[s['name']]=ep
        f['specifications'][s['name']]=s
        print('NN REFINEMENT',row,flush=True)
    old_nn=f['stage_winners']['small_nn']
    old_row=cvrows[cvrows.name==old_nn].iloc[0].to_dict()
    best=min(choices+[old_row],key=lambda r:r['MSE'])['name']
    cvrows=pd.concat([cvrows[~cvrows.name.str.startswith('nn_64_nohep_')],pd.DataFrame(choices)],ignore_index=True)
    cvrows.to_csv(OUT/'training_cv.csv',index=False)
    confirm=pd.DataFrame(f['confirmation'])
    if best!=old_nn:
        s=f['specifications'][best]
        p2,loss,ep=ex.cv(s,train,2026)
        s['frozen_epochs']=int(np.median(epochs[best]+ep))
        row=dict(name=best,CV_seed23_MSE=ex.score(train['血糖'],predictions[best])['MSE'],
                 CV_seed2026_MSE=ex.score(train['血糖'],p2)['MSE'])
        row['mean_CV_MSE']=(row['CV_seed23_MSE']+row['CV_seed2026_MSE'])/2
        confirm=pd.concat([confirm,pd.DataFrame([row])],ignore_index=True)
        oof[best+'_seed23']=predictions[best];oof[best+'_seed2026']=p2
        f['stage_winners']['small_nn']=best
    f['confirmation']=confirm.to_dict('records')
    selected=str(confirm.sort_values('mean_CV_MSE').iloc[0]['name'])
    f['selected']=selected;f['stage_winners']['selected']=selected
    f['nn_feature_refinement_plan']=plan
    # Freeze the refinement BEFORE reading any saved holdout predictions or B.
    ex.dump(OUT/'nn_refinement_frozen_selection.json',f)
    ex.dump(OUT/'frozen_selection.json',f)
    oof.to_csv(OUT/'oof_finalists.csv',index=False)
    confirm.to_csv(OUT/'confirmation.csv',index=False)
    if best!=old_nn:
        evaluation=pd.read_csv(OUT/'evaluation.csv');extra=[]
        s=f['specifications'][best]
        b=pd.read_csv(ex.ROOT/'data/d_test_B_20180128.csv')
        b['血糖']=pd.read_csv(ex.ROOT/'data/d_answer_b_20180130.csv',header=None).iloc[:,0].to_numpy()
        for label,fit,test in [('development',train,valid),('B',full,b)]:
            p,_,(pre,model)=ex.fit_nn_predict(s,fit,test,epochs=s['frozen_epochs'])
            import joblib
            joblib.dump(pre,OUT/f'{label}_{best}_preprocessor.joblib')
            model.save(OUT/f'{label}_{best}.keras')
            d=pd.read_csv(OUT/f'{label}_predictions.csv');assert np.array_equal(d.id,test.id.to_numpy())
            d[best]=p;d.to_csv(OUT/f'{label}_predictions.csv',index=False)
            extra.append(dict(dataset=label,name=best,**ex.score(test['血糖'],p)))
            delta=(d.actual.to_numpy()-p)**2-(d.actual.to_numpy()-d.baseline_xgb.to_numpy())**2
            rng=np.random.default_rng(23)
            samples=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(3000)])
            ci=pd.read_csv(OUT/'paired_intervals.csv')
            ci=pd.concat([ci,pd.DataFrame([dict(dataset=label,name=best,MSE_difference=float(delta.mean()),
                         low95=float(np.quantile(samples,.025)),high95=float(np.quantile(samples,.975)))])],ignore_index=True)
            ci.to_csv(OUT/'paired_intervals.csv',index=False)
        pd.concat([evaluation,pd.DataFrame(extra)],ignore_index=True).to_csv(OUT/'evaluation.csv',index=False)
    print('NN refinement complete; train-only NN choice',best,flush=True)


if __name__=='__main__':main()
