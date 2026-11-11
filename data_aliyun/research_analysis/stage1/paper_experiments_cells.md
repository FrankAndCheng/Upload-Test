# paper_experiments.ipynb 单元格与文本输出快照

编号按 cells 数组从 1 起，In[] 仅保留原记录。嵌入图片另见 figures/；不执行 Notebook。原Markdown的相对文件链接已展开到原Notebook目录，源码摘要仍按原文计算。


## C001（0 基索引 0，In[None], markdown）

# 按顺序优化实验

特征分组消融 → 特征与参数联合随机搜索 → 简单融合 → 小型神经网络。所有选择只使用4649条训练记录；开发1993条及B1000条只在冻结后评价。CV并非嵌套，开发/B此前已查看，均有评估局限。

[完整报告](../../review/sequential_experiments/report.md)；默认只读取实际完成的结果。


## C002（0 基索引 1，In[1], code）

```python
from pathlib import Path
import json
import pandas as pd
from IPython.display import display
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'glucose_project').is_dir() and (p / 'review').is_dir())
OUT = ROOT / 'review/sequential_experiments'
selection = json.loads((OUT / 'frozen_selection.json').read_text())
print('训练CV最终选择:', selection['selected'])
display(pd.read_csv(OUT / 'stage_comparison.csv'))
```


输出 1：

```text
训练CV最终选择: nn_blend_06

```


输出 2：

```text
            阶段                模型    训练CV均值     开发MSE  开发MSE下降百分比     B_MSE  \
0  当前XGBoost基线      baseline_xgb  1.806988  2.228808    0.000000  1.252744   
1       特征消融优胜  feature_combined  1.786509  2.218877    0.445586  1.275351   
2       联合搜索优胜     joint_xgb_089  1.781861  2.236749   -0.356302  1.242601   
3       简单融合优胜          blend_33  1.762972  2.196540    1.447769  1.245558   
4     小型神经网络优胜   nn_64_nohep_raw  1.911642  2.260269   -1.411581  1.352561   
5     补充：小网络融合       nn_blend_06  1.759568  2.177243    2.313554  1.234296   
6     训练CV最终选择       nn_blend_06  1.759568  2.177243    2.313554  1.234296   

   B_MSE下降百分比     B_MAE      B_R2  
0    0.000000  0.648007  0.184240  
1   -1.804630  0.654285  0.169518  
2    0.809627  0.646290  0.190845  
3    0.573555  0.652927  0.188919  
4   -7.967894  0.668018  0.119241  
5    1.472594  0.638160  0.196253  
6    1.472594  0.638160  0.196253  
```


## C003（0 基索引 2，In[None], markdown）

## 1. 固定参数的特征分组消融
每个候选五折，缺失填补只在各训练折拟合。


## C004（0 基索引 3，In[2], code）

```python
display(pd.read_csv(OUT / 'feature_ablation.csv').drop(columns='fold_MSE'))
```


输出 1：

```text
                     name       MSE       MAE  CV_MSE下降百分比
0            baseline_xgb  1.803531  0.726559     0.000000
1             feature_all  1.797830  0.724452     0.316096
2         feature_missing  1.801047  0.725372     0.137732
3          feature_ratios  1.797452  0.724395     0.337041
4          feature_counts  1.802958  0.725548     0.031789
5    feature_interactions  1.789187  0.723245     0.795335
6        feature_combined  1.783699  0.722849     1.099621
7  feature_native_missing  1.812445  0.728703    -0.494234
```


## C005（0 基索引 4，In[None], markdown）

## 2. 联合搜索
100组XGBoost、12组LightGBM、8组HistGradientBoosting。以下仅展示训练CV前十，不根据开发或B排序选型。


## C006（0 基索引 5，In[3], code）

```python
cv = pd.read_csv(OUT / 'training_cv.csv')
display(cv[cv.stage == 2].sort_values('MSE').head(10).drop(columns='fold_MSE'))
```


输出 1：

```text
     stage           name       MSE      RMSE       MAE        R2
97       2  joint_xgb_089  1.773159  1.331600  0.718638  0.183990
63       2  joint_xgb_055  1.779823  1.334100  0.725523  0.180923
10       2  joint_xgb_002  1.783853  1.335610  0.730013  0.179068
124      2  joint_hist_04  1.784014  1.335670  0.732440  0.178994
99       2  joint_xgb_091  1.784642  1.335905  0.729328  0.178705
66       2  joint_xgb_058  1.785610  1.336267  0.719216  0.178259
127      2  joint_hist_07  1.789886  1.337866  0.726841  0.176292
90       2  joint_xgb_082  1.790216  1.337990  0.730852  0.176140
125      2  joint_hist_05  1.790586  1.338128  0.728069  0.175970
67       2  joint_xgb_059  1.791412  1.338436  0.721204  0.175590
```


## C007（0 基索引 6，In[None], markdown）

## 3. 简单融合与误差互补
权重在训练折外预测上比较。


## C008（0 基索引 7，In[4], code）

```python
display(pd.DataFrame(json.loads((OUT / 'blends.json').read_text())).sort_values('MSE').head(10))
display(pd.read_csv(OUT / 'residual_correlations.csv', index_col=0))
name = selection['stage_winners']['blend']
print(name, selection['specifications'][name])
```


输出 1：

```text
    stage      name       MSE      RMSE       MAE        R2
33      3  blend_33  1.758488  1.326080  0.718510  0.190741
35      3  blend_35  1.761223  1.327111  0.722425  0.189482
8       3  blend_08  1.762240  1.327494  0.718803  0.189015
7       3  blend_07  1.763440  1.327946  0.722399  0.188462
39      3  blend_39  1.763558  1.327990  0.724097  0.188408
5       3  blend_05  1.765691  1.328793  0.715627  0.187427
34      3  blend_34  1.767210  1.329364  0.719502  0.186727
11      3  blend_11  1.768447  1.329830  0.719147  0.186158
10      3  blend_10  1.768685  1.329919  0.721541  0.186048
31      3  blend_31  1.769587  1.330258  0.716618  0.185633
```


输出 2：

```text
               joint_xgb_089  frozen_forest  frozen_ridge  joint_lgb_05  \
joint_xgb_089       1.000000       0.972759      0.960180      0.973072   
frozen_forest       0.972759       1.000000      0.969399      0.959023   
frozen_ridge        0.960180       0.969399      1.000000      0.941305   
joint_lgb_05        0.973072       0.959023      0.941305      1.000000   
joint_hist_04       0.988966       0.971805      0.961920      0.974772   

               joint_hist_04  
joint_xgb_089       0.988966  
frozen_forest       0.971805  
frozen_ridge        0.961920  
joint_lgb_05        0.974772  
joint_hist_04       1.000000  
```


输出 3：

```text
blend_33 {'name': 'blend_33', 'family': 'blend', 'members': ['joint_xgb_089', 'frozen_ridge', 'joint_lgb_05'], 'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]}

```


## C009（0 基索引 8，In[None], markdown）

## 4. 小型神经网络与第二套五折复核
网络早停只使用各训练折内部的20%，评价前冻结训练轮数。先做4个结构/表示候选，再根据训练OOF极端误差补充去乙肝的2个对照。


## C010（0 基索引 9，In[5], code）

```python
display(cv[cv.stage == 4].drop(columns='fold_MSE'))
display(pd.read_csv(OUT / 'confirmation.csv'))
```


输出 1：

```text
     stage             name       MSE      RMSE       MAE        R2
130      4        nn_64_raw  2.770506  1.664484  0.794479 -0.274991
131      4        nn_64_pca  3.065721  1.750920  0.800976 -0.410849
132      4       nn_128_raw  4.156521  2.038755  0.810145 -0.912837
133      4       nn_128_pca  2.245105  1.498367  0.780937 -0.033201
134      4  nn_64_nohep_raw  1.897055  1.377336  0.780562  0.126972
135      4  nn_64_nohep_pca  1.932535  1.390157  0.776402  0.110644
```


输出 2：

```text
               name  CV_seed23_MSE  CV_seed2026_MSE  mean_CV_MSE
0      baseline_xgb       1.803531         1.810445     1.806988
1  feature_combined       1.783699         1.789320     1.786509
2     joint_xgb_089       1.773159         1.790562     1.781861
3          blend_33       1.758488         1.767456     1.762972
4        nn_128_pca       2.245105         3.068600     2.656852
5   nn_64_nohep_raw       1.897055         1.926229     1.911642
6       nn_blend_06       1.754295         1.764841     1.759568
```


## C011（0 基索引 10，In[None], markdown）

## 补充：小网络与树模型融合
权重计划在联合搜索阶段预先声明；先用训练OOF冻结最终决定，再读取评价预测。


## C012（0 基索引 11，In[6], code）

```python
display(pd.read_csv(OUT / 'nn_blend_training_cv.csv'))
name = selection['stage_winners']['nn_blend']
print(name, selection['specifications'][name])
```


输出 1：

```text
          name  CV_seed23_MSE  CV_seed2026_MSE  mean_CV_MSE
0  nn_blend_00       1.781322         1.789464     1.785393
1  nn_blend_01       1.789506         1.801768     1.795637
2  nn_blend_02       1.828084         1.847356     1.837720
3  nn_blend_03       1.757638         1.773118     1.765378
4  nn_blend_04       1.773114         1.789914     1.781514
5  nn_blend_05       1.819586         1.840951     1.830269
6  nn_blend_06       1.754295         1.764841     1.759568
7  nn_blend_07       1.775992         1.790432     1.783212
8  nn_blend_08       1.823578         1.844228     1.833903
```


输出 2：

```text
nn_blend_06 {'name': 'nn_blend_06', 'family': 'blend', 'members': ['nn_64_nohep_raw', 'blend_33'], 'weights': [0.25, 0.75]}

```


## C013（0 基索引 12，In[None], markdown）

## 冻结后的开发/B复核
不能依据下表再选模型。不同集合的绝对误差不可直接比较。


## C014（0 基索引 13，In[7], code）

```python
display(pd.read_csv(OUT / 'evaluation.csv'))
display(pd.read_csv(OUT / 'paired_intervals.csv'))
display(pd.read_csv(OUT / 'target_bands.csv'))
```


输出 1：

```text
        dataset              name       MSE      RMSE       MAE        R2
0   development      baseline_xgb  2.228808  1.492919  0.780951  0.186578
1   development  feature_combined  2.218877  1.489589  0.774954  0.190202
2   development     joint_xgb_089  2.236749  1.495577  0.782494  0.183680
3   development          blend_33  2.196540  1.482073  0.777495  0.198354
4   development        nn_128_pca  2.392069  1.546632  0.815605  0.126994
5             B      baseline_xgb  1.252744  1.119260  0.648007  0.184240
6             B  feature_combined  1.275351  1.129314  0.654285  0.169518
7             B     joint_xgb_089  1.242601  1.114720  0.646290  0.190845
8             B          blend_33  1.245558  1.116046  0.652927  0.188919
9             B        nn_128_pca  1.388255  1.178242  0.641939  0.095998
10  development   nn_64_nohep_raw  2.260269  1.503419  0.820975  0.175096
11            B   nn_64_nohep_raw  1.352561  1.162997  0.668018  0.119241
12  development       nn_blend_06  2.177243  1.475548  0.777364  0.205397
13            B       nn_blend_06  1.234296  1.110989  0.638160  0.196253
```


输出 2：

```text
        dataset              name  MSE_difference     low95    high95
0   development  feature_combined       -0.009931 -0.037683  0.018810
1   development     joint_xgb_089        0.007941 -0.048668  0.078473
2   development          blend_33       -0.032268 -0.068530  0.001123
3   development        nn_128_pca        0.163262  0.041124  0.284980
4             B  feature_combined        0.022607 -0.016673  0.057049
5             B     joint_xgb_089       -0.010143 -0.050152  0.028644
6             B          blend_33       -0.007185 -0.042229  0.024062
7             B        nn_128_pca        0.135511  0.036389  0.246693
8   development   nn_64_nohep_raw        0.031461 -0.117256  0.163998
9             B   nn_64_nohep_raw        0.099817  0.014593  0.185646
10  development       nn_blend_06       -0.051565 -0.101899 -0.003706
11            B       nn_blend_06       -0.018448 -0.053797  0.016102
```


输出 3：

```text
        dataset          name    band  rows        MSE  squared_error_share  \
0   development  baseline_xgb     0-6  1598   0.433667             0.156010   
1   development  baseline_xgb    6-10   336   1.994917             0.150898   
2   development  baseline_xgb  10-inf    59  52.181741             0.693092   
3   development   nn_blend_06     0-6  1598   0.449848             0.165664   
4   development   nn_blend_06    6-10   336   1.923298             0.148926   
5   development   nn_blend_06  10-inf    59  50.409513             0.685410   
6             B  baseline_xgb     0-6   868   0.450626             0.312229   
7             B  baseline_xgb    6-10   118   1.748310             0.164679   
8             B  baseline_xgb  10-inf    14  46.807141             0.523092   
9             B   nn_blend_06     0-6   868   0.437047             0.307347   
10            B   nn_blend_06    6-10   118   1.768811             0.169100   
11            B   nn_blend_06  10-inf    14  46.158499             0.523553   

    mean_prediction_minus_actual  
0                       0.343933  
1                      -0.948679  
2                      -6.476527  
3                       0.341951  
4                      -0.931246  
5                      -6.358480  
6                       0.352662  
7                      -0.892780  
8                      -6.442799  
9                       0.314046  
10                     -0.887488  
11                     -6.396633  
```


## C015（0 基索引 14，In[None], markdown）

## 重跑实验
以下默认关闭。完整运行会重拟合模型，并覆盖本轮实验的结果文件；训练OOF可使用同源代码缓存。


## C016（0 基索引 15，In[8], code）

```python
RUN_EXPERIMENTS = False
if RUN_EXPERIMENTS:
    import subprocess, sys
    for snapshot in ['frozen_selection_before_nn_refinement.json', 'frozen_selection_before_nn_blends.json']:
        path = OUT / snapshot
        if path.exists():
            path.rename(path.with_suffix('.previous.json'))
    for script in ['run_experiments.py', 'refine_nn_features.py', 'complete_nn_blends.py']:
        subprocess.run([sys.executable, str(OUT / script)], cwd=ROOT, check=True)
```
