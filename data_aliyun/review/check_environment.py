"""Check imports and small training operations used by paper.ipynb."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/paper-matplotlib')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import importlib
import json
import sys
from pathlib import Path
results = {'python': Path(sys.executable).name, 'imports': {}, 'checks': {}}
for name in ['numpy','pandas','scipy','sklearn','statsmodels','matplotlib','seaborn','xgboost','lightgbm','tensorflow','keras','factor_analyzer']:
    try:
        module = importlib.import_module(name)
        results['imports'][name] = {'ok': True, 'version': getattr(module, '__version__', 'not exposed')}
    except Exception as exc:
        results['imports'][name] = {'ok': False, 'error': repr(exc)}

def check(name, fn):
    try:
        fn()
        results['checks'][name] = {'ok': True}
    except Exception as exc:
        results['checks'][name] = {'ok': False, 'error': repr(exc)}

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import statsmodels.api as sm
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
import tensorflow as tf
rng = np.random.default_rng(23)
x = rng.normal(size=(100, 8))
y = 5 + x[:, 0] + rng.normal(size=100)*0.1
check('pandas_csv', lambda: pd.read_csv(Path(__file__).resolve().parents[1]/'data/d_train_20180102.csv'))
check('ridge_pipeline', lambda: make_pipeline(SimpleImputer(), StandardScaler(), PCA(n_components=5), Ridge()).fit(x,y).predict(x[:3]))
check('hist_gradient_boosting', lambda: HistGradientBoostingRegressor(max_iter=3).fit(x,y).predict(x[:3]))
check('adaboost', lambda: AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=2),n_estimators=3).fit(x,y).predict(x[:3]))
check('xgboost', lambda: XGBRegressor(n_estimators=3,n_jobs=1).fit(x,y).predict(x[:3]))
check('lightgbm', lambda: LGBMRegressor(n_estimators=3,n_jobs=1,verbosity=-1).fit(x,y).predict(x[:3]))
check('statsmodels_ols', lambda: sm.OLS(y,sm.add_constant(x)).fit().predict(sm.add_constant(x)))
check('factor_analysis', lambda: (calculate_kmo(x),calculate_bartlett_sphericity(x)))
def neural():
    from keras.layers import Input,Dense,PReLU,BatchNormalization,Dropout
    m=tf.keras.Sequential([Input(shape=(8,)),Dense(8,activation=PReLU()),BatchNormalization(),Dropout(.1),Dense(1)])
    m.compile(loss='mse',optimizer='adamax')
    loss=m.train_on_batch(x.astype('float32'), y.astype('float32'))
    assert np.isfinite(loss)
    assert m(x[:3],training=False).shape == (3,1)
check('keras_train_on_batch',neural)
path=Path(__file__).with_name('environment_check.json')
path.write_text(json.dumps(results,ensure_ascii=False,indent=2))
print(json.dumps(results,ensure_ascii=False,indent=2))
sys.exit(0 if all(v['ok'] for section in ['imports','checks'] for v in results[section].values()) else 1)
