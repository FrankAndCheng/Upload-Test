"""Render completed preprocessing factorial experiment; no model selection."""
import json
from importlib.metadata import version
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
MODE={'raw':'无标准化/无PCA','scale':'仅标准化','pca':'仅PCA','scale_pca':'标准化后PCA','native':'原生缺失，无标准化/无PCA'}
MODEL={'Ridge':'Ridge','DecisionTree':'决策树','AdaBoost':'AdaBoost','RandomForest':'随机森林',
       'XGBoost_original':'XGBoost原配置','XGBoost_optimized':'XGBoost优化配置',
       'LightGBM':'LightGBM','HistGradientBoosting':'HistGradientBoosting',
       'NeuralNetwork_original':'原大网络（87轮）','NeuralNetwork_small':'去乙肝小网络（36轮）'}


def markdown(frame):
    def cell(v):
        if isinstance(v,(float,np.floating)):
            return f'{v:.6f}' if np.isfinite(v) else '失败/不可用'
        return str(v).replace('|','\\|').replace('\n',' ')
    lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    lines+=['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def main():
    completed=json.loads((OUT/'completed.json').read_text())
    frozen=json.loads((OUT/'frozen_recommendations.json').read_text())
    cv=pd.read_csv(OUT/'cv_summary.csv')
    hold=pd.read_csv(OUT/'holdout_metrics.csv')
    rec=pd.read_csv(OUT/'recommendations.csv')
    paired=pd.read_csv(OUT/'paired_cv_comparisons.csv')
    folds=pd.read_csv(OUT/'fold_results.csv')
    factorial=cv[cv['mode']!='native'].pivot(index='model',columns='mode',values='CV_MSE').reindex(rec.model)[['raw','scale','pca','scale_pca']]
    display=factorial.rename(index=MODEL,columns=MODE).reset_index(names='模型')
    display.to_csv(OUT/'factorial_cv_table.csv',index=False)
    effects=[]
    for name,r in factorial.iterrows():
        effects.append(dict(model=name,scale_without_PCA=r['scale']-r['raw'],
              scale_with_PCA=r['scale_pca']-r['pca'],PCA_without_scale=r['pca']-r['raw'],
              PCA_with_scale=r['scale_pca']-r['scale']))
    pd.DataFrame(effects).to_csv(OUT/'factorial_effects.csv',index=False)
    evidence=[]
    for name,r in factorial.iterrows():
        f=folds[folds.model==name]
        losses={m:f[f['mode']==m].sort_values(['seed','fold']).MSE.to_numpy() for m in ['raw','scale','pca','scale_pca']}
        evidence.append(dict(模型=MODEL[name],标准化相对无处理改善百分比=100*(1-r['scale']/r['raw']),
             标准化胜出折数=int((losses['scale']<losses['raw']-1e-12).sum()),
             标准化后再加PCA改善百分比=100*(1-r['scale_pca']/r['scale']),
             加PCA胜出折数=int((losses['scale_pca']<losses['scale']-1e-12).sum())))
    evidence=pd.DataFrame(evidence)
    evidence.to_csv(OUT/'treatment_evidence.csv',index=False)
    recommendations=[]
    for r in rec.itertuples():
        pair=paired[(paired.model==r.model)&(paired['mode']==r.recommended)].iloc[0]
        a=hold[(hold.model==r.model)&(hold['mode']==r.recommended)&(hold.dataset=='development')].iloc[0]
        b=hold[(hold.model==r.model)&(hold['mode']==r.recommended)&(hold.dataset=='B')].iloc[0]
        recommendations.append(dict(模型=MODEL[r.model],现有处理=MODE[r.reference],CV最低=MODE[r.CV_best],
              本轮建议=MODE[r.recommended],平均CV_MSE=r.recommended_CV_MSE,
              相对现有CV改善百分比=100*pair.relative_improvement,胜出折数=pair.fold_wins,
              开发MSE=a.MSE,B_MSE=b.MSE))
    summary=pd.DataFrame(recommendations)
    summary.to_csv(OUT/'recommendation_table.csv',index=False)
    fusion_cv=pd.read_csv(OUT/'fusion_cv.csv')
    fusion_hold=pd.read_csv(OUT/'fusion_holdout.csv')
    original_nn_dev=float(hold[(hold.model=='NeuralNetwork_original')&(hold['mode']=='scale_pca')&(hold.dataset=='development')].MSE.iloc[0])
    historical=pd.read_csv(ROOT/'review/external_b_comparison/comparison.csv')
    historical_nn_dev=float(historical[historical.model=='NeuralNetwork'].development_MSE.iloc[0])
    selected_lookup=dict(zip(rec.model,rec.recommended))
    fusion_members=frozen['fusion']['members']
    chosen_map='、'.join(f'{MODEL[n]}：{MODE[selected_lookup[n]]}' for n in fusion_members)
    # Quantify the controlled comparison: transformed sample membership and
    # pairwise heldout predictions, not a cross-model difference.
    invariant=[]
    oof=pd.read_csv(OUT/'oof_predictions.csv')
    for name in rec.model:
        x=np.concatenate([oof[f'{name}__raw__{seed}'].to_numpy() for seed in [23,2026]])
        y=np.concatenate([oof[f'{name}__scale__{seed}'].to_numpy() for seed in [23,2026]])
        invariant.append(dict(model=name,raw_vs_scale_max_prediction_difference=float(np.nanmax(np.abs(x-y))),
                              CV_MSE_difference=float(factorial.loc[name,'scale']-factorial.loc[name,'raw'])))
    pd.DataFrame(invariant).to_csv(OUT/'scaling_prediction_differences.csv',index=False)
    # Known-holdout intervals for selected versus frozen current reference.
    intervals=[]
    rng=np.random.default_rng(23)
    for dataset in ['development','B']:
        d=pd.read_csv(OUT/f'{dataset}_predictions.csv')
        for r in rec.itertuples():
            delta=(d.actual-d[f'{r.model}__{r.recommended}'])**2-(d.actual-d[f'{r.model}__{r.reference}'])**2
            if np.isfinite(delta).all():
                boot=np.array([delta.iloc[rng.integers(0,len(delta),len(delta))].mean() for _ in range(2000)])
                lo,hi=np.quantile(boot,[.025,.975])
                intervals.append(dict(dataset=dataset,model=r.model,MSE_delta=float(delta.mean()),low95=lo,high95=hi))
        delta=(d.actual-d.Fusion__recommended)**2-(d.actual-d.Fusion__reference)**2
        boot=np.array([delta.iloc[rng.integers(0,len(delta),len(delta))].mean() for _ in range(2000)])
        lo,hi=np.quantile(boot,[.025,.975])
        intervals.append(dict(dataset=dataset,model='Fusion',MSE_delta=float(delta.mean()),low95=lo,high95=hi))
    pd.DataFrame(intervals).to_csv(OUT/'holdout_paired_intervals.csv',index=False)
    status=[]
    for r in rec.itertuples():
        p=paired[(paired.model==r.model)&(paired['mode']==r.CV_best)].iloc[0]
        if r.recommended==r.reference:
            text='没有新处理达到预设替换门槛；保留现有方案，并不表示该方案在理论上必需。'
        else:
            text=f'建议由{MODE[r.reference]}改为{MODE[r.recommended]}，达到预设CV改善与折胜门槛。'
        status.append(f'- **{MODEL[r.model]}**：{text} CV最低方案为{MODE[r.CV_best]}，相对现有平均改善{100*p.relative_improvement:.3f}%，赢{int(p.fold_wins)}/10折。')
    versions={k:version(k) for k in ['scikit-learn','xgboost','lightgbm','tensorflow']}
    report=f'''# 标准化与PCA：固定模型的2×2对照实验

日期：2026-10-20。全部结果为实际运行；不安装或修改base依赖。

## 可以确认的范围

本轮确认的是**当前数据、冻结特征、冻结参数及训练预算下的处理选择**，不能推广为某类算法在所有任务中“必须”标准化或PCA。这里没有重新搜索Ridge的alpha、树参数或神经网络训练轮数，因此改变尺度带来的正则化/优化差异就是本轮测量的一部分；要比较各处理可达到的全局最好成绩，需要另做嵌套的联合调参。

本轮直接证据：树模型不需要为了提高成绩而增加标准化，PCA没有带来收益；Ridge标准化未显示明确增益，可沿用当前标准化流程；两个网络标准化分别降低平均训练CV MSE约7.63%和20.33%，而标准化后再增加PCA均没有收益。原大网络PCA仅因预设变更门槛而保留，不能把这一保留决定解释成“它需要PCA”。

## 十个模型的四组公平对照

以下均为两套五折训练OOF MSE的平均值，越低越好。同一模型每行的四个候选使用完全相同的输入行、原始特征集合和参数。

{markdown(display)}

## “是否有必要”的直接证据

{markdown(evidence)}

这里将“仅标准化”与“无标准化/无PCA”直接配对，将“标准化后PCA”与“仅标准化”直接配对；正百分比代表改善。它与下面的保守流程替换决定不同：现有处理没有被替换，不等于该处理被证明必需。

特别是原大网络，若PCA与不做PCA成绩很接近，保留现有PCA只是执行预设的0.5%/7折变更门槛，不能据此宣称PCA提高准确度。Ridge的标准化也应结合直接对照判断，不因理论上常用就自动宣称有实测收益。

- **无标准化/无PCA**：中位数填补＋缺失指示。
- **仅标准化**：相同填补后StandardScaler。
- **仅PCA**：相同填补后直接PCA，保留98%方差；PCA本身会中心化，但不把各列缩放为单位方差。
- **标准化后PCA**：相同填补→StandardScaler→PCA98%。

所有拟合统计量都来自各训练折。PCA之后没有再次标准化。四组之间不改变删列、交互项、模型参数、种子、网络轮数或缺失处理。标准化不包含血糖目标；没有目标变换或极端样本删除。

LightGBM和HistGradientBoosting另保留一个**原生缺失处理**参考。该参考用于判断是否替换原方案，不能与四组PCA候选的差异全归因于PCA，因为缺失处理也不同。四格表本身的缺失处理是严格一致的。

## 最终建议及已知集合复核

{markdown(summary)}

预设替换规则：相对各模型现有冻结处理，两套OOF平均MSE至少降低0.5%，且10个配对折至少赢7折，才接受改变；在合格候选中取最低平均MSE。未达门槛保留原方案。该门槛是工程决策规则，不是统计显著性检验。CV最低和最终保留可能不同，避免为微小随机差异改变流程。

{chr(10).join(status)}

## 四模型融合：只换预处理，不改权重

仍使用XGBoost优化配置、LightGBM、Ridge、去乙肝小网络各25%。本轮建议处理为：{chosen_map}。

{markdown(fusion_cv)}

{markdown(fusion_hold)}

融合权重沿用上轮，未根据本轮CV或B重新优化。原融合参考的开发/B MSE已复现到容差。不能因为某个已知B成绩更好，就改用B最优处理替换训练内部冻结的推荐。

本轮建议的唯一操作变化为LightGBM缺失填补（不涉及标准化/PCA）。它在单模型训练CV达到0.5%/7折门槛，但融合只在开发略好、B略差，不能据此声称四模型融合得到进一步稳定提升。旧四模型模型文件没有覆盖；两种版本作为独立对照保留。

## 全部处理在开发/B的成绩

{markdown(hold[['dataset','model','mode','status','MSE','RMSE','MAE','R2']])}

## 实验协议

1. 原train+A共6642条，以test_size=0.3、random_state=23恢复原4649条训练/1993条开发。开发含原训练1685条和A308条，非纯A验证。
2. 每个处理使用KFold5（shuffle=True），种子23和2026；共10个配对折。两套折复用同一批记录，不是独立样本量翻倍。
3. 固定原六算法配置，以及优化后的XGBoost、LightGBM、HistGradientBoosting、小网络。原大网络固定87轮/batch32，小网络固定36轮/batch64，初始化种子23；本轮四处理均不早停，以隔离变换效应。
   大网络本轮每次构建前统一clear_session并重新设种子，原notebook构建器没有clear_session；本轮其开发标准化+PCA成绩为{original_nn_dev:.6f}，与旧存档{historical_nn_dev:.6f}并非逐值复现。不能把本轮重训模型当作旧权重完全相同的实例；四组之间仍采用相同清理/初始化协议，比较以本轮成对重训结果为准。该历史差异的具体来源未进一步分解，因此不归因于某一个数值因素。五个传统模型参考成绩及原四模型融合另有数值复现检查。
4. 神经网络无标准化时可能训练不稳定，非有限输出会标为失败，不会将失败当作零误差；详见progress.json。所有失败候选不作为推荐。
5. 推荐、融合处理、参数均保存在frozen_recommendations.json后，才开始开发/B评价。开发使用4649条拟合，B使用全部6642条拟合；B1000条，行不删除。
6. 超参数和部分原始特征此前已经选择过，CV分数仍有选择偏差；本轮并非对完整选型过程的嵌套CV。开发/B此前已被反复查看，因此属于已知集合复核，不能当作全新盲测。
7. 没有进行额外的神经网络初始化种子重复，未计入所有训练随机性。所谓“需要”应理解为本轮条件下的操作建议，不能保证未来批次排序不变。

## 配对差异的解释

`paired_cv_comparisons.csv`中差值为候选−现有处理的MSE，负值代表改善。区间为两个OOF平方误差差值先按同一行平均，再做2000次行配对bootstrap；不把两套折当作独立的2N样本。该区间不包含模型训练、历史调参、折依赖、日期聚类、多重比较，不能作为全流程无偏推断。

`holdout_paired_intervals.csv`同样只描述当前固定拟合模型下的行重采样差异。没有换处理时差值与区间为零，仅表示预测相同，不能据此证明原处理必需。

## 复现和文件

- `run_ablation.py`：主实验。使用Conda base：`python review/preprocessing_ablation/run_ablation.py`。
- `render_results.py`：生成报告及`paper_preprocessing_experiments.ipynb`，默认只读结果；展示notebook已实际执行。
- `plan.json`：评价前声明的模型、四组处理、阈值和输入摘要。`frozen_recommendations.json`：读取B前冻结的决定。
- `cv_summary.csv`、`fold_results.csv`、`oof_predictions.csv`：训练内部完整结果；cache键包含实验源码、输入文件、模型、处理、种子、折，输入变化后不复用旧键。
- `factorial_effects.csv`：标准化/PCA在另一因素固定时的MSE差。正值表示变差；不是跨模型比较。
- `scaling_prediction_differences.csv`：不做PCA时，标准化前后OOF预测的最大差，帮助判断树是否近似不变。
- `recommendation_table.csv`、`holdout_metrics.csv`、`fusion_holdout.csv`及逐条预测：推荐和全部已知集合复核。
- `models/`：本轮各模型原方案及建议方案的已拟合模型。旧的四模型融合文件不覆盖；这里是独立对照实验。
- 原大网络沿用`Dense(activation=PReLU())`的旧写法，在当前Keras下不能直接`load_model`反序列化。`verify_results.py`提供了已验证的推理加载流程：按冻结结构重建网络，先用一行零输入建立PReLU变量，再`load_weights(.keras)`，随后预测。该流程只恢复推理权重，不宣称恢复优化器状态以逐步续训；不需要修改base依赖或重新训练。
- `verification.json`：指标重算、PCA维数、基线复现、原始输入及notebook保留检查。

计算耗时{completed['seconds']/60:.2f}分钟。依赖：{json.dumps(versions,ensure_ascii=False)}。源码SHA256：`{completed['source_sha256']}`。神经网络硬件/运行库变化可能导致末位差异。
'''
    (OUT/'report.md').write_text(report)
    n=nbformat.v4.new_notebook()
    n.metadata.kernelspec=dict(display_name='Python 3 (base)',language='python',name='python3')
    n.cells=[
        nbformat.v4.new_markdown_cell('# 标准化与PCA的2×2对照\n\n固定十个模型版本的特征/参数/网络训练轮数，只改变标准化与PCA。所有选择使用4649条训练记录的两套五折；开发/B是已知集合复核。\n\n[完整报告](review/preprocessing_ablation/report.md)'),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display\nROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'glucose_project').is_dir() and (p / 'review').is_dir())\nOUT = ROOT / 'review/preprocessing_ablation'\ndisplay(pd.read_csv(OUT / 'factorial_cv_table.csv'))"),
        nbformat.v4.new_markdown_cell('## 推荐规则\n平均OOF MSE至少改善0.5%，且10个配对折至少赢7折才替换原处理。没有达到门槛时保留原方案，不等于证明原方案必需。'),
        nbformat.v4.new_code_cell("display(pd.read_csv(OUT / 'recommendation_table.csv'))"),
        nbformat.v4.new_markdown_cell('## 处理是否有必要的直接证据\n保留现有流程不等于证明其必需。正百分比表示标准化或额外PCA带来改善。'),
        nbformat.v4.new_code_cell("display(pd.read_csv(OUT / 'treatment_evidence.csv'))"),
        nbformat.v4.new_markdown_cell('## 标准化与PCA的条件效应\n正值表示增加该处理后MSE变差。'),
        nbformat.v4.new_code_cell("display(pd.read_csv(OUT / 'factorial_effects.csv'))\ndisplay(pd.read_csv(OUT / 'scaling_prediction_differences.csv'))"),
        nbformat.v4.new_markdown_cell('## 训练折稳定性与配对差异\n两个CV种子复用相同记录；区间不包括完整训练/调参随机性。'),
        nbformat.v4.new_code_cell("display(pd.read_csv(OUT / 'paired_cv_comparisons.csv'))\ndisplay(pd.read_csv(OUT / 'fold_results.csv').groupby(['model', 'mode']).agg(mean_MSE=('MSE','mean'), min_dimensions=('dimension','min'), max_dimensions=('dimension','max')))"),
        nbformat.v4.new_markdown_cell('## 冻结后的开发/B复核\n不得依据以下B成绩回选处理方法。'),
        nbformat.v4.new_code_cell("display(pd.read_csv(OUT / 'holdout_metrics.csv'))\ndisplay(pd.read_csv(OUT / 'holdout_paired_intervals.csv'))"),
        nbformat.v4.new_markdown_cell('## 等权四模型融合\n仅按各模型训练CV建议改变处理，权重仍各25%。'),
        nbformat.v4.new_code_cell("display(pd.read_csv(OUT / 'fusion_cv.csv'))\ndisplay(pd.read_csv(OUT / 'fusion_holdout.csv'))"),
        nbformat.v4.new_markdown_cell('## 重跑\n默认关闭；重新运行将更新本轮对照结果和模型，不修改旧实验。'),
        nbformat.v4.new_code_cell("RUN_ABLATION = False\nif RUN_ABLATION:\n    import subprocess, sys\n    for script in ['run_ablation.py', 'render_results.py']:\n        subprocess.run([sys.executable, str(OUT / script)], cwd=ROOT, check=True)")]
    nbformat.write(n,ROOT/'paper_preprocessing_experiments.ipynb')
    print(display.to_string(index=False))
    print(summary.to_string(index=False))


if __name__=='__main__':main()
