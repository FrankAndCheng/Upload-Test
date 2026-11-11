"""Predeclared small-NN blending supplement. No holdout-based decisions.

Run only after run_experiments.py completes. Reuses already fitted members.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

OUT=Path(__file__).resolve().parent


def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2))


def main():
    plan=json.loads((OUT/'nn_blend_plan.json').read_text())
    assert plan['source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source=OUT/'frozen_selection.json'
    original=OUT/'frozen_selection_before_nn_blends.json'
    if not original.exists():original.write_bytes(source.read_bytes())
    f=json.loads(original.read_text());specs=f['specifications'];stages=f['stage_winners']
    oof=pd.read_csv(OUT/'oof_finalists.csv');y=oof.actual.to_numpy()
    confirmation=pd.DataFrame(f['confirmation'])
    nn=stages['small_nn']
    references=list(dict.fromkeys([stages[k] for k in plan['partner_stages']]))
    new_rows=[];new_specs={};arrays={}
    for member in references:
        for w in plan['weights_for_nn']:
            name=f'nn_blend_{len(new_rows):02}'
            scores=[]
            for seed in [23,2026]:
                p=w*oof[f'{nn}_seed{seed}'].to_numpy()+(1-w)*oof[f'{member}_seed{seed}'].to_numpy()
                arrays[f'{name}_seed{seed}']=p
                scores.append(float(mean_squared_error(y,p)))
            new_specs[name]=dict(name=name,family='blend',members=[nn,member],weights=[w,1-w])
            new_rows.append(dict(name=name,CV_seed23_MSE=scores[0],CV_seed2026_MSE=scores[1],
                                 mean_CV_MSE=float(np.mean(scores))))
    new=pd.DataFrame(new_rows)
    winner=str(new.sort_values('mean_CV_MSE').iloc[0]['name'])
    # First-fold MSE chooses the supplement candidate; seed2026 confirms that
    # same weight rather than choosing a different weight on confirmation folds.
    winner=str(new.sort_values('CV_seed23_MSE').iloc[0]['name'])
    eligible=pd.concat([confirmation,new[new.name==winner]],ignore_index=True)
    selected=str(eligible.sort_values('mean_CV_MSE').iloc[0]['name'])
    specs.update(new_specs)
    stages['nn_blend']=winner;stages['selected']=selected
    f.update(selected=selected,confirmation=eligible.to_dict('records'),
             nn_blend_plan=plan,
             note=f['note']+' Supplement uses predeclared OOF-only weights, then reads previously generated holdout predictions.')
    # Freeze the supplementary weights and final choice BEFORE accessing any
    # development/B predictions or labels in this process.
    dump(OUT/'nn_blend_frozen_selection.json',f)
    dump(source,f)
    new.to_csv(OUT/'nn_blend_training_cv.csv',index=False)
    eligible.to_csv(OUT/'confirmation.csv',index=False)
    for k,p in arrays.items():oof[k]=p
    oof.to_csv(OUT/'oof_finalists.csv',index=False)
    evaluation=pd.read_csv(OUT/'evaluation.csv')
    evaluation=evaluation[~evaluation.name.str.startswith('nn_blend_')]
    extra=[];intervals=[];bands=[]
    s=specs[winner]
    for label in ['development','B']:
        d=pd.read_csv(OUT/f'{label}_predictions.csv')
        p=sum(w*d[n].to_numpy() for n,w in zip(s['members'],s['weights']))
        d[winner]=p
        d.to_csv(OUT/f'{label}_predictions.csv',index=False)
        yy=d.actual.to_numpy();mse=float(mean_squared_error(yy,p))
        extra.append(dict(dataset=label,name=winner,MSE=mse,RMSE=float(np.sqrt(mse)),
                          MAE=float(mean_absolute_error(yy,p)),R2=float(r2_score(yy,p))))
        delta=(yy-p)**2-(yy-d.baseline_xgb.to_numpy())**2
        rng=np.random.default_rng(23)
        sample=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(3000)])
        intervals.append(dict(dataset=label,name=winner,MSE_difference=float(delta.mean()),
                              low95=float(np.quantile(sample,.025)),high95=float(np.quantile(sample,.975))))
        for name in dict.fromkeys(['baseline_xgb',selected]):
            for lo,hi in [(0,6),(6,10),(10,float('inf'))]:
                z=d[(d.actual>=lo)&(d.actual<hi)];err=(z.actual-z[name])**2
                bands.append(dict(dataset=label,name=name,band=f'{lo}-{hi}',rows=len(z),MSE=float(err.mean()),
                                  squared_error_share=float(err.sum()/((d.actual-d[name])**2).sum()),
                                  mean_prediction_minus_actual=float((z[name]-z.actual).mean())))
    pd.concat([evaluation,pd.DataFrame(extra)],ignore_index=True).to_csv(OUT/'evaluation.csv',index=False)
    old_intervals=pd.read_csv(OUT/'paired_intervals.csv')
    old_intervals=old_intervals[~old_intervals.name.str.startswith('nn_blend_')]
    pd.concat([old_intervals,pd.DataFrame(intervals)],ignore_index=True).to_csv(OUT/'paired_intervals.csv',index=False)
    pd.DataFrame(bands).to_csv(OUT/'target_bands.csv',index=False)
    done=json.loads((OUT/'completed.json').read_text())
    done.update(selected=selected,stage_winners=stages,nn_blend_supplement_complete=True)
    dump(OUT/'completed.json',done)
    print('NN BLEND supplement',winner,'final training-only choice',selected,flush=True)
    print(pd.DataFrame(extra).to_string(index=False),flush=True)


if __name__=='__main__':main()
