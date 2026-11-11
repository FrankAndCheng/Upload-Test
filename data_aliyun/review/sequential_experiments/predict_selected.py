"""Predict with the frozen selected model, without reading any target labels."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/paper-matplotlib')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from run_experiments import features

OUT=Path(__file__).resolve().parent


def predict(frame,scope='full'):
    frozen=json.loads((OUT/'frozen_selection.json').read_text())
    specs=frozen['specifications']
    tag={'full':'B','development':'development'}[scope]
    cache={}
    def member(name):
        if name in cache:return cache[name]
        s=specs[name]
        if s['family']=='blend':
            p=sum(w*member(n) for n,w in zip(s['members'],s['weights']))
        elif s['family']=='nn':
            import tensorflow as tf
            pre=joblib.load(OUT/f'{tag}_{name}_preprocessor.joblib')
            model=tf.keras.models.load_model(OUT/f'{tag}_{name}.keras',compile=False)
            p=model.predict(pre.transform(features(frame,s['variant'])),verbose=0).reshape(-1)
        else:
            model=joblib.load(OUT/f'{tag}_{name}.joblib')
            p=model.predict(features(frame,s['variant']))
        assert len(p)==len(frame) and np.isfinite(p).all()
        cache[name]=p
        return p
    return member(frozen['selected'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--scope',default='full',choices=['full','development'])
    args=parser.parse_args()
    frame=pd.read_csv(args.input)
    p=predict(frame,args.scope)
    pd.DataFrame({'id':frame['id'].to_numpy(),'predicted_glucose':p}).to_csv(args.output,index=False)
    print('Predicted rows:',len(p))


if __name__=='__main__':main()
