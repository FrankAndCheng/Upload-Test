"""Reproducible ridge baseline; requires numpy and pandas. Does not alter paper.ipynb."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]

def features(d, nonlinear=False):
    x = d.drop(columns=['id', '体检日期', '血糖'], errors='ignore').copy()
    x['性别'] = x['性别'].map({'女': 0., '男': 1.})
    if nonlinear:
        for knot in [30, 40, 50, 60, 70]:
            x[f'age_hinge_{knot}'] = np.maximum(x['年龄'] - knot, 0)
        for c in ['甘油三酯', '*r-谷氨酰基转换酶', '*丙氨酸氨基转换酶']:
            x[f'log_{c}'] = np.log1p(x[c].clip(lower=0))
    return x.to_numpy(dtype=float)

def predict(a, y, b, alpha):
    med = np.nanmedian(a, axis=0)
    a = np.c_[np.where(np.isnan(a), med, a), np.isnan(a).astype(float)]
    b = np.c_[np.where(np.isnan(b), med, b), np.isnan(b).astype(float)]
    mu, sd = a.mean(0), a.std(0)
    sd[sd < 1e-10] = 1
    a, b = (a-mu)/sd, (b-mu)/sd
    yc = y-y.mean()
    beta = np.linalg.solve(a.T@a + alpha*np.eye(a.shape[1]), a.T@yc)
    return b@beta + y.mean()

def metrics(y, p):
    return dict(MSE=float(np.mean((y-p)**2)), RMSE=float(np.sqrt(np.mean((y-p)**2))),
                MAE=float(np.mean(abs(y-p))), R2=float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)))

if __name__ == '__main__':
    train = pd.read_csv(ROOT/'data/d_train_20180102.csv')
    test_a = pd.read_csv(ROOT/'data/d_test_A_20180102.csv')
    test_b = pd.read_csv(ROOT/'data/d_test_B_20180128.csv')
    ya = pd.read_csv(ROOT/'data/d_answer_a_20180128.csv').iloc[:, 0].to_numpy()
    yb = pd.read_csv(ROOT/'data/d_answer_b_20180130.csv', header=None).iloc[:, 0].to_numpy()
    assert len(test_a) == len(ya) and len(test_b) == len(yb)
    y = train['血糖'].to_numpy()
    folds = np.array_split(np.random.default_rng(23).permutation(len(y)), 5)
    trials = []
    for nonlinear in [False, True]:
        x = features(train, nonlinear)
        for alpha in [10., 100., 1000.]:
            oof = np.empty(len(y))
            for val in folds:
                idx = np.setdiff1d(np.arange(len(y)), val)
                oof[val] = predict(x[idx], y[idx], x[val], alpha)
            trials.append(dict(nonlinear=nonlinear, alpha=alpha, CV=metrics(y, oof)))
    best = min(trials, key=lambda r: r['CV']['MSE'])
    nl, alpha = best['nonlinear'], best['alpha']
    x = features(train, nl)
    result = dict(protocol='Select on train-only 5-fold CV; fit train only; evaluate complete A/B separately. No test tuning.',
                  trials=trials, selected=best,
                  A=metrics(ya, predict(x,y,features(test_a,nl),alpha)),
                  B=metrics(yb, predict(x,y,features(test_b,nl),alpha)),
                  mean_A=metrics(ya,np.full(len(ya),y.mean())),
                  mean_B=metrics(yb,np.full(len(yb),y.mean())))
    (ROOT/'review/baseline_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
