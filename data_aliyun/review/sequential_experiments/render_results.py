"""Render completed measurements; never chooses or trains a model."""
import json
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd
from importlib.metadata import version

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def markdown(frame):
    def cell(v):
        if isinstance(v,(float,np.floating)):return f'{v:.6f}'
        return str(v).replace('|','\\|').replace('\n',' ')
    lines=['| '+' | '.join(map(str,frame.columns))+' |',
           '| '+' | '.join(['---']*len(frame.columns))+' |']
    lines+=['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def main():
    done=json.loads((OUT/'completed.json').read_text())
    frozen=json.loads((OUT/'frozen_selection.json').read_text())
    specs=frozen['specifications'];stages=frozen['stage_winners']
    cv=pd.read_csv(OUT/'training_cv.csv')
    evaluation=pd.read_csv(OUT/'evaluation.csv')
    confirm=pd.read_csv(OUT/'confirmation.csv')
    intervals=pd.read_csv(OUT/'paired_intervals.csv')
    labels={'baseline':'当前XGBoost基线','feature':'特征消融优胜','joint':'联合搜索优胜',
            'blend':'简单融合优胜','small_nn':'小型神经网络优胜',
            'nn_blend':'补充：小网络融合','selected':'训练CV最终选择'}
    stages={k:stages[k] for k in labels}
    summary=[]
    for stage,name in stages.items():
        c=confirm[confirm.name==name].iloc[0]
        dev=evaluation[(evaluation.name==name)&(evaluation.dataset=='development')].iloc[0]
        b=evaluation[(evaluation.name==name)&(evaluation.dataset=='B')].iloc[0]
        bd=evaluation[(evaluation.name=='baseline_xgb')&(evaluation.dataset=='development')].iloc[0]
        bb=evaluation[(evaluation.name=='baseline_xgb')&(evaluation.dataset=='B')].iloc[0]
        summary.append(dict(阶段=labels[stage],模型=name,训练CV均值=c.mean_CV_MSE,
                            开发MSE=dev.MSE,开发MSE下降百分比=100*(1-dev.MSE/bd.MSE),
                            B_MSE=b.MSE,B_MSE下降百分比=100*(1-b.MSE/bb.MSE),
                            B_MAE=b.MAE,B_R2=b.R2))
    table=pd.DataFrame(summary)
    table.to_csv(OUT/'stage_comparison.csv',index=False)
    ablation=cv[cv.stage==1][['name','MSE','MAE','fold_MSE']].copy()
    base=float(ablation[ablation.name=='baseline_xgb'].MSE.iloc[0])
    ablation['CV_MSE下降百分比']=100*(1-ablation.MSE/base)
    ablation.to_csv(OUT/'feature_ablation.csv',index=False)
    sel=stages['selected']; ss=specs[sel]
    dev_sel=evaluation[(evaluation.name==sel)&(evaluation.dataset=='development')].iloc[0]
    b_sel=evaluation[(evaluation.name==sel)&(evaluation.dataset=='B')].iloc[0]
    baseline_dev=evaluation[(evaluation.name=='baseline_xgb')&(evaluation.dataset=='development')].iloc[0]
    baseline_b=evaluation[(evaluation.name=='baseline_xgb')&(evaluation.dataset=='B')].iloc[0]
    gains=dict(development=100*(1-dev_sel.MSE/baseline_dev.MSE),B=100*(1-b_sel.MSE/baseline_b.MSE))
    verdict=('两个集合的MSE点估计均改善。' if min(gains.values())>0 else
             '两个集合没有同时改善，不能据此声称稳定提高泛化能力。')
    if sel=='baseline_xgb':verdict='训练内部复核最终保留原基线，本轮没有选出更好的替代方案。'
    old=pd.read_csv(ROOT/'review/external_b_comparison/comparison.csv')
    old_nn=old[old.model=='NeuralNetwork'].iloc[0]
    new_nn_name=stages['small_nn']
    new_nn_dev=evaluation[(evaluation.name==new_nn_name)&(evaluation.dataset=='development')].iloc[0]
    new_nn_b=evaluation[(evaluation.name==new_nn_name)&(evaluation.dataset=='B')].iloc[0]
    def describe(name):
        s=specs[name]
        if s['family']=='blend':
            return ' + '.join(f'{w:.2f} × {m}' for m,w in zip(s['members'],s['weights']))
        return json.dumps(s,ensure_ascii=False)
    def flattened(name,weight=1.):
        s=specs[name]
        if s['family']=='blend':
            parts=[]
            for member,w in zip(s['members'],s['weights']):parts+=flattened(member,weight*w)
            return parts
        return [(name,weight)]
    family_names={'xgb':'XGBoost','lgb':'LightGBM','ridge':'Ridge','nn':'去乙肝小网络',
                  'hist':'HistGradientBoosting','forest':'随机森林'}
    composition=' + '.join(f"{weight:.0%} {family_names[specs[name]['family']]}" for name,weight in flattened(sel))
    details='\n'.join(f'- **{labels[k]}**：`{v}`；{describe(v)}。' for k,v in stages.items() if k!='selected')
    versions={name:version(name) for name in ['scikit-learn','xgboost','lightgbm','tensorflow']}
    report=f'''# 按顺序优化实验：实际结果

## 最终结果

训练集内部两套五折选出的方案为 **{sel}**，开发验证 MSE 为 **{dev_sel.MSE:.6f}**，B MSE 为 **{b_sel.MSE:.6f}**。以下全部为实际运行结果；没有根据开发/B的成绩重新选模型、调整权重或回调参数。

最终组成：**{composition}**。

相对当前XGBoost，开发MSE下降{gains['development']:.2f}%，B MSE下降{gains['B']:.2f}%。{verdict} 点估计的改善仍需结合下方配对区间与已知验证集的局限解释。

{markdown(table)}

“下降百分比”为相对同集合当前XGBoost基线，负数表示变差。最终选择由训练CV决定，不能改用表中B最好的方案。开发和B的绝对误差不能跨集合直接比较。

## 固定实验顺序与验证边界

1. 复用主 notebook 的 train+A 共6642条、seed23的70/30划分：4649条训练，1993条开发验证。开发验证并非纯A集，包含原训练1685条和A的308条。
2. 特征分组消融：固定原XGBoost参数，8个配置；原始全列、去乙肝、缺失分组、比值、细胞计数、年龄交互、组合及原生缺失处理。
3. 特征与参数联合随机搜索：100组XGBoost、12组LightGBM、8组HistGradientBoosting，共120个候选；每组五折。特征集合从第一阶段前三名与原基线中产生，同时搜索树复杂度、轮数、学习率、正则和缺失处理。两个新增树家族是有限预算候选，并非穷尽搜索。
4. 简单融合：训练折外预测上比较两成员的0.25/0.5/0.75权重及三成员等权。成员包括联合搜索优胜、各提升树家族优胜、冻结随机森林和Ridge。没有用训练拟合预测学习权重；并未使用B学习权重。
5. 小型神经网络：64→32（Dropout0.1）和128→64→16（Dropout0.2），各比较原始/PCA输入，先运行4个候选；Adam0.001、L2=0.001、batch64、最多100轮。每个外层训练折再切20%内部早停，预处理仅拟合内部训练子集，随后按选定轮数重拟合完整外层训练折。训练OOF诊断发现id4609实际4.58却预测66.675，极端乙肝输入值得排查；在主流程读取开发/B前记录nn_refinement_plan.json，补充64→32网络去乙肝五项的原始/PCA两种对照。因此本轮共6个小网络候选，后两个是训练诊断驱动的适应性实验，并非最初四个候选的一部分。没有沿用原400→160→64→26网络。
6. 以上选择均先用seed23五折；各阶段优胜与基线用seed2026第二套五折复核。按两套训练OOF MSE均值选择最终方案。神经网络最终轮数为两套CV内部选定轮数的中位数，开发/B拟合不再早停。
7. 另在联合搜索尚未结束时预先声明小网络补充融合计划（nn_blend_plan.json）：小网络完成后，与原基线、联合搜索优胜、树融合优胜分别比较0.25/0.5/0.75权重。按seed23选择补充方案、用seed2026复核，再与原阶段优胜比较两套CV均值；没有按B挑权重。原四阶段冻结后先生成各成员的开发/B预测，补充流程只读训练OOF完成并保存最终决定，随后才读取这些已生成的评价预测进行加权。
8. 原四阶段评价前的选择保存在frozen_selection_before_nn_refinement.json；去乙肝网络补充选择保存在nn_refinement_frozen_selection.json及frozen_selection_before_nn_blends.json；最终融合决定见nn_blend_frozen_selection.json及frozen_selection.json。B评价统一在全部6642条上重拟合，B样本1000条；神经网络结构、轮数、融合权重冻结。所有样本保留，无目标裁剪、无按真实血糖区间决定预测路径。

**局限：** 两套CV仍参与选择，且候选已利用第一套折进行筛选；这不是嵌套CV，CV分数有选择偏差。第二套折不构成新的独立样本。开发与B此前已被查看，本次作为已知集合上的复核，不能视为全新盲测。B结果不反馈到本轮模型选择；未来真实收益需要新的独立数据确认。

## 第一阶段：特征消融

{markdown(ablation.drop(columns='fold_MSE'))}

第一阶段的结果是固定模型参数下的结果，不代表同一特征组在所有模型、参数下都有效。比值仅对正且非零分母计算；其余填NaN。所有派生量仅依赖同一行输入，不使用目标、id、日期。

特征组合在训练CV和开发集有小幅收益，但B变差，不能宣称特征越多越好。联合搜索的单模型在B改善而开发稍差；最终融合才在两边同时取得较低MSE。

## 第二套训练折复核

{markdown(confirm)}

## 冻结的阶段优胜方案

{details}

完整成员参数见 frozen_selection.json 的 specifications；融合模型通过成员名称引用各自冻结的特征处理和参数。

小型神经网络与此前冻结大网络的同协议参考：原网络开发MSE={old_nn.development_MSE:.6f}、B MSE={old_nn.B_MSE:.6f}；本轮小网络开发MSE={new_nn_dev.MSE:.6f}、B MSE={new_nn_b.MSE:.6f}。原网络引用此前已保存结果，本轮没有重训它，不能把单次差异当作网络结构的确定性因果收益。

训练OOF极端记录id4609在64→32原始全特征小网络中预测66.675，去乙肝同结构后预测4.699989（真实4.58）。这支持对该输入组做针对性处理的必要性，但单条记录不能证明乙肝特征对所有模型都无用。小网络单模型仍弱于树模型融合，较小权重加入后能改善本轮融合；以训练OOF验证互补性，比只看单模型排名更合适。

## 开发/B全部指标

{markdown(evaluation)}

## 相对原XGBoost的配对区间

以下差值为“新模型MSE−基线MSE”，负数表示新模型更好。固定拟合模型下3000次行配对bootstrap的描述性95%区间，不包括训练/调参随机性、日期聚类、多重比较；不能当成整个优化流程的无偏显著性检验。

{markdown(intervals)}

## 文件与复现

- `paper_experiments.ipynb`：展示本轮阶段结果；默认只读结果，不重跑搜索。
- `run_experiments.py`：完整实验。项目根目录运行 `python review/sequential_experiments/run_experiments.py`。同源代码下训练OOF可从cache恢复；最终模型仍重新拟合。改变源码后缓存键变化，重跑搜索。
- 原四阶段完成后，依次运行 `python review/sequential_experiments/refine_nn_features.py` 和 `python review/sequential_experiments/complete_nn_blends.py`，执行训练诊断驱动的去乙肝网络对照和预先声明的小网络补充融合；最后运行 `python review/sequential_experiments/render_results.py` 生成报告和展示notebook。单独重跑这些脚本会覆盖结果；完整重跑前应移走补充流程的两个before快照，以免恢复旧选择。缓存只适用于输入文件未变化的情况，可用input_manifest.json核对；输入变化时须清理cache重新计算。
- `plan.json` / `joint_search_plan.json`：实验协议及实际抽样参数；`frozen_selection.json`：评价前保存的最终决定。
- `training_cv.csv` / `confirmation.csv` / `oof_finalists.csv`：训练折、复核折、折外预测。
- `stage_comparison.csv` / `evaluation.csv` / `development_predictions.csv` / `B_predictions.csv`：对照表及逐条预测。
- `residual_correlations.csv`：融合成员在训练OOF上的残差相关。
- `target_bands.csv`：最终方案及基线按真实目标区间的事后诊断，仅用于误差分析。
- `.joblib` / `.keras`：阶段优胜及融合成员模型；预测前必须使用脚本相同的features转换，神经网络另带preprocessor文件。
- `predict_selected.py`：只加载冻结模型预测，不读取真实血糖标签。完整开发集拟合版本使用`--scope full`（默认）；`--input`指定原格式特征CSV，`--output`指定输出CSV。已验证重新加载后的预测与本轮记录一致。

基线在开发和B上分别复现2.228807860191054和1.2527435992815599，脚本已断言验证。原始CSV、base依赖和旧实验结果未修改。主notebook原有单元格保留，仅追加本轮结果入口。

重新加载最终成员模型、仅输入无目标列的原始特征后，开发/B预测与存档的最大绝对差均小于1.2×10⁻⁷；全部指标从CSV重算通过。float32预测导出CSV会产生约10⁻⁸量级指标舍入差，验证容差为5×10⁻⁷，详见verification.json。

原四阶段计算耗时{done['seconds']/60:.2f}分钟，另有去乙肝网络补充实验。依赖版本：{json.dumps(versions,ensure_ascii=False)}。原四阶段源码SHA256：`{done['source_sha256']}`；两个补充脚本摘要见各自plan.json。神经网络在不同硬件/运行库上可能出现数值差异。
'''
    (OUT/'report.md').write_text(report)
    notebook=nbformat.v4.new_notebook()
    notebook.metadata.kernelspec=dict(display_name='Python 3 (base)',language='python',name='python3')
    notebook.cells=[
        nbformat.v4.new_markdown_cell('# 按顺序优化实验\n\n特征分组消融 → 特征与参数联合随机搜索 → 简单融合 → 小型神经网络。所有选择只使用4649条训练记录；开发1993条及B1000条只在冻结后评价。CV并非嵌套，开发/B此前已查看，均有评估局限。\n\n[完整报告](review/sequential_experiments/report.md)；默认只读取实际完成的结果。'),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display\nROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'glucose_project').is_dir() and (p / 'review').is_dir())\nOUT = ROOT / 'review/sequential_experiments'\nselection = json.loads((OUT / 'frozen_selection.json').read_text())\nprint('训练CV最终选择:', selection['selected'])\ndisplay(pd.read_csv(OUT / 'stage_comparison.csv'))"),
        nbformat.v4.new_markdown_cell('## 1. 固定参数的特征分组消融\n每个候选五折，缺失填补只在各训练折拟合。'),
        nbformat.v4.new_code_cell("display(pd.read_csv(OUT / 'feature_ablation.csv').drop(columns='fold_MSE'))"),
        nbformat.v4.new_markdown_cell('## 2. 联合搜索\n100组XGBoost、12组LightGBM、8组HistGradientBoosting。以下仅展示训练CV前十，不根据开发或B排序选型。'),
        nbformat.v4.new_code_cell("cv = pd.read_csv(OUT / 'training_cv.csv')\ndisplay(cv[cv.stage == 2].sort_values('MSE').head(10).drop(columns='fold_MSE'))"),
        nbformat.v4.new_markdown_cell('## 3. 简单融合与误差互补\n权重在训练折外预测上比较。'),
        nbformat.v4.new_code_cell("display(pd.DataFrame(json.loads((OUT / 'blends.json').read_text())).sort_values('MSE').head(10))\ndisplay(pd.read_csv(OUT / 'residual_correlations.csv', index_col=0))\nname = selection['stage_winners']['blend']\nprint(name, selection['specifications'][name])"),
        nbformat.v4.new_markdown_cell('## 4. 小型神经网络与第二套五折复核\n网络早停只使用各训练折内部的20%，评价前冻结训练轮数。先做4个结构/表示候选，再根据训练OOF极端误差补充去乙肝的2个对照。'),
        nbformat.v4.new_code_cell("display(cv[cv.stage == 4].drop(columns='fold_MSE'))\ndisplay(pd.read_csv(OUT / 'confirmation.csv'))"),
        nbformat.v4.new_markdown_cell('## 补充：小网络与树模型融合\n权重计划在联合搜索阶段预先声明；先用训练OOF冻结最终决定，再读取评价预测。'),
        nbformat.v4.new_code_cell("display(pd.read_csv(OUT / 'nn_blend_training_cv.csv'))\nname = selection['stage_winners']['nn_blend']\nprint(name, selection['specifications'][name])"),
        nbformat.v4.new_markdown_cell('## 冻结后的开发/B复核\n不能依据下表再选模型。不同集合的绝对误差不可直接比较。'),
        nbformat.v4.new_code_cell("display(pd.read_csv(OUT / 'evaluation.csv'))\ndisplay(pd.read_csv(OUT / 'paired_intervals.csv'))\ndisplay(pd.read_csv(OUT / 'target_bands.csv'))"),
        nbformat.v4.new_markdown_cell('## 重跑实验\n以下默认关闭。完整运行会重拟合模型，并覆盖本轮实验的结果文件；训练OOF可使用同源代码缓存。'),
        nbformat.v4.new_code_cell("RUN_EXPERIMENTS = False\nif RUN_EXPERIMENTS:\n    import subprocess, sys\n    for snapshot in ['frozen_selection_before_nn_refinement.json', 'frozen_selection_before_nn_blends.json']:\n        path = OUT / snapshot\n        if path.exists():\n            path.rename(path.with_suffix('.previous.json'))\n    for script in ['run_experiments.py', 'refine_nn_features.py', 'complete_nn_blends.py']:\n        subprocess.run([sys.executable, str(OUT / script)], cwd=ROOT, check=True)")]
    nbformat.write(notebook,ROOT/'paper_experiments.ipynb')
    print(table.to_string(index=False))


if __name__=='__main__':main()
