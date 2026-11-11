"""Reconstruct saved algorithm choices; rerun only frozen fits, never searches.
Run from project root with Conda base Python.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/paper-matplotlib')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('LOKY_MAX_CPU_COUNT','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import ast,copy,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent

def primitive(value):
    if hasattr(value,'get_params'):
        return {'class':type(value).__name__,'parameters':primitive(value.get_params(deep=False))}
    if isinstance(value,dict):return {k:primitive(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [primitive(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return repr(value)
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return repr(value)

def recover_state():
    notebook=json.loads((ROOT/'paper.ipynb').read_text())
    frozen=json.loads((OUT/'frozen_choices.json').read_text())
    ns={'__name__':'frozen_external_evaluation'}
    # Only data loading, deterministic row transforms and definitions; no model searches.
    for index in [1,13,16,18,22,27,33,37,47]:
        exec(compile(''.join(notebook['cells'][index]['source']),f'paper_cell_{index}','exec'),ns)
    bases={}
    for name,index,var in [('Ridge',45,None),('DecisionTree',53,'DeTree'),('AdaBoost',56,'adaboost_base'),('XGBoost',59,'xgb_base'),('RandomForest',63,'forest')]:
        tree=ast.parse(''.join(notebook['cells'][index]['source']))
        nodes=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom)) or
               (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==var for t in node.targets))]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),f'constructor_{index}','exec'),ns)
        if var is None:
            node=next(node for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='linear_pipeline' for t in node.targets))
            bases[name]=eval(compile(ast.Expression(node.value.args[0]),'ridge_constructor','eval'),ns)
        else:bases[name]=ns[var]
    registry={};choices=frozen['choices'];complete={}
    # Preserve the original model registration order; neural fit uses the original fixed builder and seed.
    for row in frozen['development_results']:
        name=row['model'];choice=choices[name]
        if choice['kind']=='sklearn':
            estimator=bases[name]
            overrides={k.removeprefix('model__'):v for k,v in choice['best_search_params'].items()}
            estimator.set_params(**overrides)
            pipeline=ns['make_model_pipeline'](estimator,choice['representation'],choice['feature_set'],choice['family'])
            registry[name]={'kind':'sklearn','pipeline':pipeline}
            complete[name]={**choice,'estimator':primitive(estimator)}
        else:
            registry[name]={'kind':'keras','representation':choice['representation'],'epochs':choice['epochs']}
            builder=next(node for node in ast.parse(''.join(notebook['cells'][47]['source'])).body if isinstance(node,ast.FunctionDef) and node.name=='build_nn')
            complete[name]={**choice,'build_nn_source':ast.get_source_segment(''.join(notebook['cells'][47]['source']),builder)}
    ns['model_registry']=registry
    ns['development_results']=copy.deepcopy(frozen['development_results'])
    # Write every frozen parameter before any B labels are loaded.
    configuration={'algorithms':complete,'development_training_rows':4649,'external_training_rows':6642,
                   'external_rows':1000,'source_code_sha256':hashlib.sha256(''.join(''.join(c['source']) for c in notebook['cells'][:65] if c['cell_type']=='code').encode()).hexdigest()}
    previous=OUT/'frozen_configurations.json'
    if previous.exists():
        assert json.loads(previous.read_text())==configuration,'Frozen algorithms or source changed; refuse to silently overwrite configurations.'
    else:previous.write_text(json.dumps(configuration,ensure_ascii=False,indent=2))
    # Check that the reconstructed sklearn choices reproduce the saved development scores.
    import numpy as np
    for row in ns['development_results']:
        entry=registry[row['model']]
        if entry['kind']=='sklearn':
            model=ns['clone'](entry['pipeline']).fit(ns['X_train_raw'],ns['train_labels'])
            mse=ns['mean_squared_error'](ns['validation_labels'],model.predict(ns['X_validation_raw']))
            assert np.isclose(mse,row['development_MSE'],rtol=1e-8,atol=1e-10),(row['model'],mse,row['development_MSE'])
            print('Frozen development-score check passed:',row['model'],mse,flush=True)
    return ns

def main():
    import nbformat
    from nbclient import NotebookClient
    notebook=nbformat.read(ROOT/'paper.ipynb',as_version=4)
    prefix=nbformat.v4.new_code_cell("import sys\nsys.path.insert(0, 'review/external_b_comparison')\nfrom evaluate_frozen import recover_state\nglobals().update(recover_state())")
    indexes=[66,67,68,70]
    run=nbformat.v4.new_notebook(cells=[prefix]+[copy.deepcopy(notebook.cells[i]) for i in indexes])
    def start(cell,cell_index,**kw):print('START external step',cell_index,flush=True)
    client=NotebookClient(run,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},on_cell_start=start)
    client.execute()
    for index,executed in zip(indexes,run.cells[1:]):
        notebook.cells[index]['outputs']=executed.outputs
        # These cells were run in a fresh kernel with reconstructed frozen state.
        notebook.cells[index]['execution_count']=executed.execution_count
        notebook.cells[index]['metadata']=executed.metadata
    nbformat.write(notebook,ROOT/'paper.ipynb')
    print('All frozen models evaluated; external notebook outputs updated.',flush=True)

if __name__=='__main__':main()
