"""第五阶段逐表核对：从来源 CSV/JSON 重新格式化，与当前 TeX 的表头和各行逐格比较。
描述性字段仍须结合模型人工检查；不以本脚本替代论文语义核查。
"""
from pathlib import Path
import csv,json,re,hashlib
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
EDITOR=ROOT/'research_analysis/stage3'
entries=[]
def read_csv(path):
    with (ROOT/path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def tex(s):
    return ''.join({'&':r'\&','%':r'\%','_':r'\_','#':r'\#','$':r'\$','{':r'\{','}':r'\}'}.get(c,c) for c in str(s))
def num(v,n=6):return f'{float(v):.{n}f}'
def norm(x):return re.sub(r'\s+','',str(x))
def table(token,label,caption,headers,rows,source,locator,widths=None,long=False):
    matches=[x for x in (EDITOR/'paper/tables').glob('*.tex') if r'\label{'+label+'}' in x.read_text()]
    assert len(matches)==1,(label,matches)
    body=matches[0].read_text()
    body=body.split(r'\endhead')[-1] if long else body.split(r'\midrule',1)[1]
    body=body.split(r'\bottomrule',1)[0]
    actual=[list(map(str.strip,re.split(r'(?<!\\)&',x.strip()))) for x in body.split(r'\\') if x.strip()]
    header_ok=norm(' & '.join(headers)) in norm(matches[0].read_text())
    rows_ok=[[norm(y) for y in x] for x in actual]==[[norm(y) for y in x] for x in rows]
    entries.append(dict(kind='table',label=label,file=str(matches[0].relative_to(ROOT)),rows=len(rows),headers_match=header_ok,all_cells_match=rows_ok,sources_exist=all((ROOT/x).exists() for x in source),sources=source))
def fig(token,label,caption,filename,cell):
    original=ROOT/'research_analysis/stage1/figures'/filename
    current=EDITOR/'paper/figures'/filename
    entry=dict(kind='figure',label=label,file=str(current.relative_to(ROOT)),bytes_unchanged=original.read_bytes()==current.read_bytes(),sha256=hashlib.sha256(current.read_bytes()).hexdigest(),locator=cell)
    if label=='fig:correlation':
        record=json.loads((OUT/'correlation_reformat.json').read_text())
        rendered=ROOT/record['rendered_figure']
        assert hashlib.sha256(rendered.read_bytes()).hexdigest()==record['sha256']
        entry['displayed_reformatted_figure']=record['rendered_figure']
        entry['matrix_file']=record['matrix_file']
    entries.append(entry)
table('TABLE_SYMBOLS','tab:symbols','主要符号及含义',['符号','定义'],[
 [r'$\mathcal D_T,\mathcal D_V,\mathcal D_B$','当前训练、开发与B集合，分别4649、1993、1000行'],
 [r'$F,n_F$','某次实际拟合行集合及其样本数；包括折内、内部子集或完整拟合集'],
 [r'$x_i,y_i,\hat y_i$','第i条记录的输入、真实血糖与预测血糖'],
 [r'$n_T,p$','训练行数及当前表示的输入列数；p随特征与变换改变'],
 [r'$\Phi_F,\phi_i$','只在F拟合的编码/预处理组合及其输出向量'],
 [r'$m_{ij},q_j(F),J_F$','缺失指示、拟合列中位数及拟合时出现缺失的列集合'],
 [r'$u_i,z_i,t_i$','填补并加指示后的向量、标准化向量、PCA得分'],
 [r'$\xi_i,\zeta_i$','40维训练探索填补向量及其标准化表示，不加自动缺失指示'],
 [r'$I_{s,k},\mathcal E_s(c)$','种子s的第k验证折及候选c的合并OOF损失'],
 [r'$\alpha_R,\alpha_L$','Ridge与Lasso惩罚系数，分别100与0.05'],
 [r'$\eta,\lambda_X,a_X,\gamma_X$','提升学习率、XGB叶权L2/L1及叶数惩罚，按模型参数确定'],
 [r'$\pi_m,\Delta_E,Q_g$','外层融合权重、候选减参考的MSE差、分箱平方误差贡献']
],['research_analysis/stage1/研究内容梳理报告.md'],'第4–5节',[.27,.65])
table('TABLE_PROTOCOL','tab:protocol','当前主协议的样本与拟合范围',['用途','拟合/选择行','评价行','备注'],[
 ['训练内部选择','4649','相应训练验证折','原train3957+A692；五折内估计预处理'],
 ['开发评分','4649','1993','原train1685+A308；不是纯A验证'],
 ['B集复核','6642','1000','冻结后以train+A全部重拟合；B有历史使用']
],['review/sequential_experiments/split_ids.csv','research_analysis/stage1/data_profile.json'],'paper.ipynb C002/C004/C068/C069',[.13,.13,.12,.5])
table('TABLE_BASELINE_CONFIG','tab:baseline-config','六模型的关键冻结配置',['模型','配置与表示'],[
 ['Ridge','惩罚100；全特征、填补及标准化，无PCA'],
 ['决策树','深度6，叶最少60，节点分裂最少10，随机切分，种子30；全特征填补'],
 ['随机森林','200棵，不限深度，叶最少7，特征比例0.6，种子23；全特征填补'],
 ['AdaBoost.R2','基础树深度3、叶最少20；最多300轮，学习率0.005，指数误差，外层种子32；全特征填补'],
 ['原XGBoost','350轮，学习率0.02，深度4，子节点Hessian和30，L2=10，行/列比例各0.8；去乙肝填补'],
 ['原大网络','400--160--64--26--1，PReLU、Dropout及BatchNorm，Adamax，batch32；标准化+PCA，87轮']
],['review/external_b_comparison/frozen_configurations.json'],'paper.ipynb C046–C065',[.18,.73])
fdata={r['name']:r for r in read_csv('review/sequential_experiments/feature_ablation.csv')}
flabels=[('baseline_xgb','去乙肝参考','基础输入去乙肝五项',35),('feature_all','全特征','基础40列保留',40),('feature_missing','缺失计数','去乙肝+总/4组计数',40),('feature_ratios','比值','去乙肝+3比值',38),('feature_counts','估算计数','去乙肝+5计数',40),('feature_interactions','年龄项','去乙肝+4交互/年龄平方',40),('feature_combined','组合','去乙肝+18工程项',53),('feature_native_missing','原生缺失参考','去乙肝，不填补',35)]
table('TABLE_FEATURES','tab:features','固定原XGBoost参数的特征与缺失处理对照',['配置','输入构造','填补前列数','CV23 MSE'],[[name,desc,str(n),num(fdata[key]['MSE'])] for key,name,desc,n in flabels],['review/sequential_experiments/feature_ablation.csv','review/sequential_experiments/run_experiments.py'],'paper_experiments.ipynb C004；features函数',[.18,.38,.13,.18])
table('TABLE_MEMBERS','tab:members','最终融合成员及各自输入处理',['成员','特征与预处理','权重'],[
 ['优化XGBoost','去乙肝+年龄项；中位数/缺失指示；无缩放/PCA','25\\%'],
 ['LightGBM','去乙肝+年龄项；原生缺失；无缩放/PCA','25\\%'],
 ['Ridge','全40列；中位数/缺失指示；标准化，无PCA','25\\%'],
 ['小网络','去乙肝；中位数/缺失指示；标准化，无PCA；64--32，36轮','25\\%']
],['review/sequential_experiments/frozen_selection.json'],'specifications: nn_blend_06→blend_33及四成员',[.2,.59,.1])
b=read_csv('review/external_b_comparison/comparison.csv')
names={'XGBoost':'原XGBoost','RandomForest':'随机森林','NeuralNetwork':'原大网络','Ridge':'Ridge','DecisionTree':'决策树','AdaBoost':'AdaBoost.R2'}
table('TABLE_BASELINE_RESULTS','tab:baseline-results','六基准在当前开发/B的误差（未舍入预测）',['模型','开发MSE','B MSE','开发MAE','B MAE'],[[names[r['model']]]+[num(r[k]) for k in ['development_MSE','B_MSE','development_MAE','B_MAE']] for r in b],['review/external_b_comparison/comparison.csv'],'paper.ipynb C067/C069')
stage=read_csv('review/sequential_experiments/stage_comparison.csv')[:6]
labels=['原XGBoost','组合特征','联合XGBoost','三模型等权','去乙肝小网络','四模型各25\\%']
table('TABLE_STAGES','tab:stages','顺序优化阶段的选择分数与冻结评价',['方案','两套OOF均值','开发MSE','B MSE'],[[lab]+[num(r[k]) for k in ['训练CV均值','开发MSE','B_MSE']] for lab,r in zip(labels,stage)],['review/sequential_experiments/stage_comparison.csv','review/sequential_experiments/confirmation.csv'],'paper_experiments.ipynb C010/C014',[.31,.21,.18,.18])
evaluations=read_csv('review/sequential_experiments/evaluation.csv')
table('TABLE_FINAL_METRICS','tab:final-metrics','最终四模型的开发/B综合指标',['评价集合','MSE','RMSE','MAE',r'$R^2$'],[[('开发' if r['dataset']=='development' else 'B')]+[num(r[k]) for k in ['MSE','RMSE','MAE','R2']] for r in evaluations if r['name']=='nn_blend_06'],['review/sequential_experiments/evaluation.csv'],'name=nn_blend_06；paper_experiments.ipynb C014')
table('TABLE_TREATMENTS','tab:treatments','固定模型预处理配置定义',['记号','缺失处理','标准化','PCA'],[
 ['raw','中位数+缺失指示','否','否'],['scale','同raw','是','否'],['pca','同raw','否','98\\%方差'],['scale\\_pca','同raw','是','98\\%方差'],['native（另列）','LGB/HistGB原生路由','否','否']
],['review/preprocessing_ablation/plan.json','review/preprocessing_ablation/run_ablation.py'],'preprocessing函数及paper_preprocessing_experiments.ipynb C001',[.2,.3,.15,.2])
factor=read_csv('review/preprocessing_ablation/factorial_cv_table.csv')
factor_names={'XGBoost原配置':'原XGBoost','XGBoost优化配置':'优化XGBoost','HistGradientBoosting':'HistGB','原大网络（87轮）':'大网络（87轮）','去乙肝小网络（36轮）':'小网络（36轮）'}
table('TABLE_FACTORIAL','tab:factorial','十个固定模型版本的四处理平均OOF MSE',['模型','raw','scale','pca','scale\\_pca'],[[factor_names.get(r['模型'],r['模型'])]+[num(r[k]) for k in ['无标准化/无PCA','仅标准化','仅PCA','标准化后PCA']] for r in factor],['review/preprocessing_ablation/factorial_cv_table.csv'],'paper.ipynb C073；两套OOF损失均值')
fusion=read_csv('review/preprocessing_ablation/fusion_holdout.csv')
table('TABLE_FUSION_TREATMENT','tab:fusion-treatment','保持四模型权重的预处理版本复核',['集合','版本','MSE','RMSE','MAE',r'$R^2$'],[[('开发' if r['dataset']=='development' else 'B'),('原处理' if r['version']=='reference' else '推荐处理')]+[num(r[k]) for k in ['MSE','RMSE','MAE','R2']] for r in fusion],['review/preprocessing_ablation/fusion_holdout.csv'],'paper_preprocessing_experiments.ipynb C014')
cis=[]
for r in read_csv('review/sequential_experiments/paired_intervals.csv'):
    if r['name']=='nn_blend_06':cis.append([('开发' if r['dataset']=='development' else 'B'),'四模型−原XGB']+[num(r[k]) for k in ['MSE_difference','low95','high95']])
for r in read_csv('review/preprocessing_ablation/holdout_paired_intervals.csv'):
    if r['model']=='Fusion':cis.append([('开发' if r['dataset']=='development' else 'B'),'推荐处理−原处理']+[num(r[k]) for k in ['MSE_delta','low95','high95']])
table('TABLE_INTERVALS','tab:intervals','固定预测下MSE差与95\\%行配对区间',['集合','比较','MSE差','下界','上界'],cis,['review/sequential_experiments/paired_intervals.csv','review/preprocessing_ablation/holdout_paired_intervals.csv'],'nn_blend_06/Fusion；顺序C014、预处理C012',[.08,.28,.17,.17,.17])
bands=[]
for r in read_csv('review/sequential_experiments/target_bands.csv'):
    if r['name']=='nn_blend_06':bands.append([('开发' if r['dataset']=='development' else 'B'),{'0-6':r'$[0,6)$','6-10':r'$[6,10)$','10-inf':r'$[10,+\infty)$'}[r['band']],r['rows'],num(r['MSE']),num(100*float(r['squared_error_share']),2)+r'\%',num(r['mean_prediction_minus_actual'])])
table('TABLE_BANDS','tab:bands','最终四模型按真实目标分箱的误差诊断',['集合','目标区间','行数','MSE','SSE占比','平均偏差'],bands,['review/sequential_experiments/target_bands.csv'],'name=nn_blend_06；paper_experiments.ipynb C014')
fields=read_csv('research_analysis/stage1/data_dictionary.csv')
table('TABLE_ALL_FIELDS','tab:all-fields','原始字段与训练子集缺失统计',['字段','角色','缺失行数',r'缺失比例（\%）'],[[tex(r['column']),tex(r['role']),r['fit4649_missing'],num(r['fit4649_missing_pct'],4)] for r in fields],['research_analysis/stage1/data_dictionary.csv'],'原始42字段，fit4649列',[.4,.19,.12,.17],True)
table('TABLE_BASELINE_SEARCH','tab:baseline-search','原基准参数候选范围与抽取数量',['模型','实际候选空间','抽取数'],[
 ['决策树','深度3/4/5/6/7；叶最少10/20/40/60；节点分裂最少2/10/20','24'],
 ['AdaBoost','轮数100/200/300；学习率0.005/0.01/0.03/0.1；误差linear/square/exponential','18'],
 ['原XGBoost','轮数100/200/350；学习率0.02/0.05/0.1；深度2/3/4；子节点Hessian和10/30/60；L2=10/30/60；行和列比例各0.8/1.0','24'],
 ['随机森林','深度8/15/不限；叶最少7/15/30；特征比例0.6/0.8/1.0；树数200固定','12']
],['paper.ipynb'],'C054/C057/C060/C064',[.17,.64,.09])
table('TABLE_JOINT_SEARCH','tab:joint-search','联合搜索的预声明参数范围',['家族','范围与固定处理','抽取数'],[
 ['XGBoost','轮数整数200--1000；深度整数2--6；学习率对数均匀[0.01,0.12]；子节点Hessian和对数均匀[5,120]；L2对数均匀[0.1,150]；L1=0/0.1/1/10；gamma=0/0.05/0.2/1；行、列比例各0.65/0.8/1；填补/原生缺失二选一','100'],
 ['LightGBM','轮数250/500/800；学习率0.015/0.03/0.06；叶数7/15/31；叶最少20/40/80；L2=1/10/50；列比例0.7/1；原生缺失','12'],
 ['HistGB','轮数200/400/700；学习率0.02/0.05；叶数7/15；叶最少20/50/80；L2=1/10/50；关闭早停、原生缺失','8']
],['review/sequential_experiments/run_experiments.py','review/sequential_experiments/joint_search_plan.json'],'run_experiments.py 216–239；实际抽样参数见plan',[.15,.66,.09])
historical=[]
same=json.loads((ROOT/'review/optimization/same_validation/results.json').read_text())
for name,val in same['legacy_baselines'].items():historical.append(['H1','1561验证',tex(name),num(val['MSE'])])
historical.append(['H1','1561验证','LGB+HistGB各半',num(same['evaluations']['validation']['selected']['MSE'])])
ext=json.loads((ROOT/'review/optimization/external/results.json').read_text())
for ds in ['A','B']:
    for key,name in [('ridge_reference','Ridge'),('selected','HistGB+LGB各半')]:historical.append(['H2',ds+'1000',name,num(ext['evaluations'][ds][key]['MSE'])])
table('TABLE_HISTORICAL','tab:historical','两套历史协议的独立结果',['协议','评价行','模型','MSE'],historical,['review/optimization/same_validation/results.json','review/optimization/external/results.json'],'paper_optimized.ipynb C004/C006',[.08,.18,.45,.17])
all_holdout=read_csv('review/preprocessing_ablation/holdout_metrics.csv')
assert len(all_holdout)==84 and all(r['status']=='ok' for r in all_holdout)
pretty={'DecisionTree':'决策树','RandomForest':'随机森林','XGBoost_original':'原XGB','XGBoost_optimized':'优化XGB','NeuralNetwork_original':'原大网络','NeuralNetwork_small':'小网络','HistGradientBoosting':'HistGB'}
table('TABLE_HOLDOUT_ALL','tab:holdout-all','固定模型所有处理在开发/B的完整指标',['集合','模型','处理','MSE','RMSE','MAE',r'$R^2$'],[[('开发' if r['dataset']=='development' else 'B'),pretty.get(r['model'],r['model']),tex(r['mode'])]+[num(r[k]) for k in ['MSE','RMSE','MAE','R2']] for r in all_holdout],['review/preprocessing_ablation/holdout_metrics.csv'],'paper_preprocessing_experiments.ipynb C012；全部status=ok',None,True)
table('TABLE_PROVENANCE','tab:provenance','研究环节的Notebook与文件定位',['研究内容','Notebook定位','存档/实现定位'],[
 ['数据与训练探索','主C002/C005--C044','原始CSV、数据划分id与探索输出'],
 ['原六模型','主C046--C071','external\\_b\\_comparison冻结配置、comparison及predictions'],
 ['特征/联合搜索','顺序C004/C006','run\\_experiments及feature\\_ablation/training\\_cv'],
 ['网络补充/融合','顺序C008--C012','refine\\_nn\\_features、complete\\_nn\\_blends及冻结选择'],
 ['主结果及区间','顺序C014；主C072','evaluation、paired\\_intervals、target\\_bands及逐行预测'],
 ['预处理对照','预处理C004--C014；主C073','run\\_ablation、frozen\\_recommendations与holdout指标'],
 ['历史协议','早期C004--C010','paper\\_optimized及optimization两协议结果']
],['research_analysis/stage1/notebook_cell_index.csv','research_analysis/stage1/实验结果与证据表.md'],'主=paper.ipynb；顺序=paper_experiments.ipynb；预处理=paper_preprocessing_experiments.ipynb；早期=paper_optimized.ipynb',[.2,.24,.46])

fig('FIG_AGE','fig:age','4649条训练记录的年龄分布；原图标题中的初始数据指当前训练子集。','paper_C007_O03.png','paper.ipynb C007输出3')
fig('FIG_GLUCOSE','fig:glucose','4649条训练记录的血糖分布及拟合正态参照曲线；曲线不代表正态性检验结论。','paper_C009_O01.png','paper.ipynb C009输出1')
fig('FIG_MISSING','fig:missing','4649条训练记录各列缺失比例；原样提取自缺失处理前探索。','paper_C012_O02.png','paper.ipynb C012输出2')
fig('FIG_HISTORY','fig:history','历史H1的1561条验证记录对照；旧网络仅有历史汇总值，不能作为当前主协议成绩或配对推断依据。','paper_optimized_C004_O03.png','paper_optimized.ipynb C004输出3')
fig('FIG_BOX1','fig:box1','训练填补副本的部分指标boxen图；指标量纲不同，图中极端记录均保留。','paper_C026_O01.png','paper.ipynb C026输出1')
fig('FIG_BOX2','fig:box2','训练填补副本的其余输入boxen图，包含年龄及未知指示等不同尺度变量。','paper_C027_O01.png','paper.ipynb C027输出1')
fig('FIG_RESIDUAL','fig:ols-residual','OLS训练拟合残差图，不是最终融合开发或B集残差。','paper_C032_O02.png','paper.ipynb C032输出2')
fig('FIG_QQ','fig:ols-qq','OLS训练残差Q--Q图。','paper_C032_O03.png','paper.ipynb C032输出3')
fig('FIG_CORR','fig:correlation','40列训练探索输入相关热图；不包含血糖目标，填补及成组缺失可能影响相关结构。','paper_C037_O02.png','paper.ipynb C037输出2')
fig('FIG_PCA','fig:pca-diagnostic','训练探索矩阵的PCA累计方差；原图横轴应理解为成分序号，不是各预测折统一输入维数。','paper_C040_O02.png','paper.ipynb C040输出2')
fig('FIG_NN','fig:nn-inner','原大网络内部表示与轮数选择的损失曲线；不是开发/B训练曲线。','paper_C051_O02.png','paper.ipynb C051输出2')

(OUT/'table_figure_checks.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2))
summary=dict(tables=sum(x['kind']=='table' for x in entries),table_rows=sum(x.get('rows',0) for x in entries),figures=sum(x['kind']=='figure' for x in entries),failures=[x for x in entries if (x['kind']=='table' and not all(x[k] for k in ['headers_match','all_cells_match','sources_exist'])) or (x['kind']=='figure' and not x['bytes_unchanged'])])
print(json.dumps(summary,ensure_ascii=False,indent=2))
assert not summary['failures']
