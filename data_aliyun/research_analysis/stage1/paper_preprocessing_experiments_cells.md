# paper_preprocessing_experiments.ipynb 单元格与文本输出快照

编号按 cells 数组从 1 起，In[] 仅保留原记录。嵌入图片另见 figures/；不执行 Notebook。原Markdown的相对文件链接已展开到原Notebook目录，源码摘要仍按原文计算。


## C001（0 基索引 0，In[None], markdown）

# 标准化与PCA的2×2对照

固定十个模型版本的特征/参数/网络训练轮数，只改变标准化与PCA。所有选择使用4649条训练记录的两套五折；开发/B是已知集合复核。

[完整报告](../../review/preprocessing_ablation/report.md)


## C002（0 基索引 1，In[1], code）

```python
from pathlib import Path
import json
import pandas as pd
from IPython.display import display
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'glucose_project').is_dir() and (p / 'review').is_dir())
OUT = ROOT / 'review/preprocessing_ablation'
display(pd.read_csv(OUT / 'factorial_cv_table.csv'))
```


输出 1：

```text
                     模型  无标准化/无PCA      仅标准化      仅PCA   标准化后PCA
0                 Ridge   1.848363  1.849622  1.944901  1.858446
1                   决策树   2.033633  2.033633  2.087293  2.107930
2              AdaBoost   1.990995  1.990819  2.016526  2.051130
3                  随机森林   1.851444  1.851883  1.942392  1.898163
4            XGBoost原配置   1.806988  1.806988  1.939866  1.872072
5           XGBoost优化配置   1.781861  1.781861  2.065189  1.841900
6              LightGBM   1.796304  1.805506  2.127522  1.875117
7  HistGradientBoosting   1.785411  1.784776  2.052189  1.884365
8             原大网络（87轮）   2.089127  1.929678  1.983365  1.930533
9           去乙肝小网络（36轮）   2.379670  1.895841  3.325018  1.941302
```


## C003（0 基索引 2，In[None], markdown）

## 推荐规则
平均OOF MSE至少改善0.5%，且10个配对折至少赢7折才替换原处理。没有达到门槛时保留原方案，不等于证明原方案必需。


## C004（0 基索引 3，In[2], code）

```python
display(pd.read_csv(OUT / 'recommendation_table.csv'))
```


输出 1：

```text
                     模型            现有处理       CV最低            本轮建议  平均CV_MSE  \
0                 Ridge            仅标准化  无标准化/无PCA            仅标准化  1.849622   
1                   决策树       无标准化/无PCA  无标准化/无PCA       无标准化/无PCA  2.033633   
2              AdaBoost       无标准化/无PCA       仅标准化       无标准化/无PCA  1.990995   
3                  随机森林       无标准化/无PCA  无标准化/无PCA       无标准化/无PCA  1.851444   
4            XGBoost原配置       无标准化/无PCA  无标准化/无PCA       无标准化/无PCA  1.806988   
5           XGBoost优化配置       无标准化/无PCA  无标准化/无PCA       无标准化/无PCA  1.781861   
6              LightGBM  原生缺失，无标准化/无PCA  无标准化/无PCA       无标准化/无PCA  1.796304   
7  HistGradientBoosting  原生缺失，无标准化/无PCA       仅标准化  原生缺失，无标准化/无PCA  1.790474   
8             原大网络（87轮）         标准化后PCA       仅标准化         标准化后PCA  1.930533   
9           去乙肝小网络（36轮）            仅标准化       仅标准化            仅标准化  1.895841   

   相对现有CV改善百分比  胜出折数     开发MSE     B_MSE  
0     0.000000     0  2.289115  1.372016  
1     0.000000     0  2.595176  1.401940  
2     0.000000     0  2.455500  1.444966  
3     0.000000     0  2.276036  1.334292  
4     0.000000     0  2.228808  1.252744  
5     0.000000     0  2.236749  1.242601  
6     0.653386     7  2.191367  1.263230  
7     0.000000     0  2.218320  1.265633  
8     0.000000     0  2.395645  1.334516  
9     0.000000     0  2.260269  1.352561  
```


## C005（0 基索引 4，In[None], markdown）

## 处理是否有必要的直接证据
保留现有流程不等于证明其必需。正百分比表示标准化或额外PCA带来改善。


## C006（0 基索引 5，In[3], code）

```python
display(pd.read_csv(OUT / 'treatment_evidence.csv'))
```


输出 1：

```text
                     模型  标准化相对无处理改善百分比  标准化胜出折数  标准化后再加PCA改善百分比  加PCA胜出折数
0                 Ridge      -0.068118        4       -0.477049         3
1                   决策树       0.000000        0       -3.653411         1
2              AdaBoost       0.008852        5       -3.029440         2
3                  随机森林      -0.023678        1       -2.499091         1
4            XGBoost原配置       0.000000        0       -3.601799         1
5           XGBoost优化配置       0.000000        0       -3.369478         1
6              LightGBM      -0.512283        5       -3.855465         1
7  HistGradientBoosting       0.035514        6       -5.579878         0
8             原大网络（87轮）       7.632296       10       -0.044287         4
9           去乙肝小网络（36轮）      20.331780       10       -2.397965         2
```


## C007（0 基索引 6，In[None], markdown）

## 标准化与PCA的条件效应
正值表示增加该处理后MSE变差。


## C008（0 基索引 7，In[4], code）

```python
display(pd.read_csv(OUT / 'factorial_effects.csv'))
display(pd.read_csv(OUT / 'scaling_prediction_differences.csv'))
```


输出 1：

```text
                    model  scale_without_PCA  scale_with_PCA  \
0                   Ridge           0.001259       -0.086455   
1            DecisionTree           0.000000        0.020638   
2                AdaBoost          -0.000176        0.034603   
3            RandomForest           0.000438       -0.044229   
4        XGBoost_original           0.000000       -0.067793   
5       XGBoost_optimized           0.000000       -0.223288   
6                LightGBM           0.009202       -0.252405   
7    HistGradientBoosting          -0.000634       -0.167824   
8  NeuralNetwork_original          -0.159448       -0.052832   
9     NeuralNetwork_small          -0.483829       -1.383716   

   PCA_without_scale  PCA_with_scale  
0           0.096538        0.008824  
1           0.053659        0.074297  
2           0.025531        0.060311  
3           0.090947        0.046280  
4           0.132878        0.065084  
5           0.283328        0.060039  
6           0.331218        0.069611  
7           0.266778        0.099588  
8          -0.105761        0.000855  
9           0.945348        0.045462  
```


输出 2：

```text
                    model  raw_vs_scale_max_prediction_difference  \
0                   Ridge                                0.988403   
1            DecisionTree                                0.000000   
2                AdaBoost                                0.453917   
3            RandomForest                                0.126859   
4        XGBoost_original                                0.000000   
5       XGBoost_optimized                                0.000000   
6                LightGBM                                1.108890   
7    HistGradientBoosting                                0.199140   
8  NeuralNetwork_original                                6.848016   
9     NeuralNetwork_small                                9.408543   

   CV_MSE_difference  
0           0.001259  
1           0.000000  
2          -0.000176  
3           0.000438  
4           0.000000  
5           0.000000  
6           0.009202  
7          -0.000634  
8          -0.159448  
9          -0.483829  
```


## C009（0 基索引 8，In[None], markdown）

## 训练折稳定性与配对差异
两个CV种子复用相同记录；区间不包括完整训练/调参随机性。


## C010（0 基索引 9，In[5], code）

```python
display(pd.read_csv(OUT / 'paired_cv_comparisons.csv'))
display(pd.read_csv(OUT / 'fold_results.csv').groupby(['model', 'mode']).agg(mean_MSE=('MSE','mean'), min_dimensions=('dimension','min'), max_dimensions=('dimension','max')))
```


输出 1：

```text
                     model       mode  reference    CV_MSE  \
0                    Ridge        raw      scale  1.848363   
1                    Ridge      scale      scale  1.849622   
2                    Ridge        pca      scale  1.944901   
3                    Ridge  scale_pca      scale  1.858446   
4             DecisionTree        raw        raw  2.033633   
5             DecisionTree      scale        raw  2.033633   
6             DecisionTree        pca        raw  2.087293   
7             DecisionTree  scale_pca        raw  2.107930   
8                 AdaBoost        raw        raw  1.990995   
9                 AdaBoost      scale        raw  1.990819   
10                AdaBoost        pca        raw  2.016526   
11                AdaBoost  scale_pca        raw  2.051130   
12            RandomForest        raw        raw  1.851444   
13            RandomForest      scale        raw  1.851883   
14            RandomForest        pca        raw  1.942392   
15            RandomForest  scale_pca        raw  1.898163   
16        XGBoost_original        raw        raw  1.806988   
17        XGBoost_original      scale        raw  1.806988   
18        XGBoost_original        pca        raw  1.939866   
19        XGBoost_original  scale_pca        raw  1.872072   
20       XGBoost_optimized        raw        raw  1.781861   
21       XGBoost_optimized      scale        raw  1.781861   
22       XGBoost_optimized        pca        raw  2.065189   
23       XGBoost_optimized  scale_pca        raw  1.841900   
24                LightGBM        raw     native  1.796304   
25                LightGBM      scale     native  1.805506   
26                LightGBM        pca     native  2.127522   
27                LightGBM  scale_pca     native  1.875117   
28                LightGBM     native     native  1.808118   
29    HistGradientBoosting        raw     native  1.785411   
30    HistGradientBoosting      scale     native  1.784776   
31    HistGradientBoosting        pca     native  2.052189   
32    HistGradientBoosting  scale_pca     native  1.884365   
33    HistGradientBoosting     native     native  1.790474   
34  NeuralNetwork_original        raw  scale_pca  2.089127   
35  NeuralNetwork_original      scale  scale_pca  1.929678   
36  NeuralNetwork_original        pca  scale_pca  1.983365   
37  NeuralNetwork_original  scale_pca  scale_pca  1.930533   
38     NeuralNetwork_small        raw      scale  2.379670   
39     NeuralNetwork_small      scale      scale  1.895841   
40     NeuralNetwork_small        pca      scale  3.325018   
41     NeuralNetwork_small  scale_pca      scale  1.941302   

    relative_improvement  fold_wins  eligible  MSE_delta     low95    high95  
0               0.000681          6     False  -0.001259 -0.004834  0.001895  
1               0.000000          0      True   0.000000  0.000000  0.000000  
2              -0.051512          0     False   0.095279  0.058467  0.137020  
3              -0.004770          3     False   0.008824 -0.001576  0.021089  
4               0.000000          0      True   0.000000  0.000000  0.000000  
5               0.000000          0     False   0.000000  0.000000  0.000000  
6              -0.026386          0     False   0.053659  0.021250  0.087365  
7              -0.036534          1     False   0.074297  0.036717  0.112007  
8               0.000000          0      True   0.000000  0.000000  0.000000  
9               0.000089          5     False  -0.000176 -0.001432  0.001117  
10             -0.012823          4     False   0.025531 -0.005129  0.056084  
11             -0.030203          2     False   0.060134  0.011761  0.104087  
12              0.000000          0      True   0.000000  0.000000  0.000000  
13             -0.000237          1     False   0.000438 -0.000092  0.001011  
14             -0.049122          0     False   0.090947  0.052302  0.130338  
15             -0.025234          1     False   0.046719  0.004725  0.087560  
16              0.000000          0      True   0.000000  0.000000  0.000000  
17              0.000000          0     False   0.000000  0.000000  0.000000  
18             -0.073535          0     False   0.132878  0.083000  0.186322  
19             -0.036018          1     False   0.065084  0.027451  0.101123  
20              0.000000          0      True   0.000000  0.000000  0.000000  
21              0.000000          0     False   0.000000  0.000000  0.000000  
22             -0.159007          0     False   0.283328  0.208237  0.361394  
23             -0.033695          1     False   0.060039  0.022505  0.100430  
24              0.006534          7      True  -0.011814 -0.026512  0.002507  
25              0.001445          5     False  -0.002612 -0.018700  0.011923  
26             -0.176650          0     False   0.319404  0.223410  0.423271  
27             -0.037054          1     False   0.066999  0.017699  0.119967  
28              0.000000          0      True   0.000000  0.000000  0.000000  
29              0.002828          6     False  -0.005063 -0.013420  0.002701  
30              0.003182          6     False  -0.005697 -0.014053  0.002373  
31             -0.146171          0     False   0.261715  0.189366  0.338069  
32             -0.052439          0     False   0.093891  0.052993  0.135817  
33              0.000000          0      True   0.000000  0.000000  0.000000  
34             -0.082150          0     False   0.158594  0.106022  0.218238  
35              0.000443          6     False  -0.000855 -0.020495  0.018622  
36             -0.027367          2     False   0.052832  0.011461  0.097095  
37              0.000000          0      True   0.000000  0.000000  0.000000  
38             -0.255206          0     False   0.483829  0.377812  0.602503  
39              0.000000          0      True   0.000000  0.000000  0.000000  
40             -0.753849          0     False   1.429178  1.296499  1.565678  
41             -0.023980          2     False   0.045462 -0.003205  0.102995  
```


输出 2：

```text
                                  mean_MSE  min_dimensions  max_dimensions
model                  mode                                               
AdaBoost               pca        2.016428               8               9
                       raw        1.990905              77              78
                       scale      1.990728              77              78
                       scale_pca  2.051035              32              33
DecisionTree           pca        2.087203               8               9
                       raw        2.033547              77              78
                       scale      2.033547              77              78
                       scale_pca  2.107833              32              33
HistGradientBoosting   native     1.790390              53              53
                       pca        2.052102               2               3
                       raw        1.785326              97              98
                       scale      1.784692              97              98
                       scale_pca  1.884280              29              30
LightGBM               native     1.808030              40              40
                       pca        2.127432               2               3
                       raw        1.796216              76              77
                       scale      1.805417              76              77
                       scale_pca  1.875017              27              28
NeuralNetwork_original pca        1.983265               8               9
                       raw        2.089024              77              78
                       scale      1.929575              77              78
                       scale_pca  1.930427              32              33
NeuralNetwork_small    pca        3.324876               8               8
                       raw        2.379593              67              68
                       scale      1.895750              67              68
                       scale_pca  1.941219              27              28
RandomForest           pca        1.942299               8               9
                       raw        1.851358              77              78
                       scale      1.851797              77              78
                       scale_pca  1.898071              32              33
Ridge                  pca        1.944811               8               9
                       raw        1.848272              77              78
                       scale      1.849531              77              78
                       scale_pca  1.858356              32              33
XGBoost_optimized      pca        2.065101               2               3
                       raw        1.781777              76              77
                       scale      1.781777              76              77
                       scale_pca  1.841804              27              28
XGBoost_original       pca        1.939776               8               8
                       raw        1.806901              67              68
                       scale      1.806901              67              68
                       scale_pca  1.871981              27              28
```


## C011（0 基索引 10，In[None], markdown）

## 冻结后的开发/B复核
不得依据以下B成绩回选处理方法。


## C012（0 基索引 11，In[6], code）

```python
display(pd.read_csv(OUT / 'holdout_metrics.csv'))
display(pd.read_csv(OUT / 'holdout_paired_intervals.csv'))
```


输出 1：

```text
        dataset                   model       mode status  error  dimension  \
0   development                   Ridge        raw     ok    NaN         78   
1   development                   Ridge      scale     ok    NaN         78   
2   development                   Ridge        pca     ok    NaN          9   
3   development                   Ridge  scale_pca     ok    NaN         33   
4   development            DecisionTree        raw     ok    NaN         78   
..          ...                     ...        ...    ...    ...        ...   
79            B  NeuralNetwork_original  scale_pca     ok    NaN         33   
80            B     NeuralNetwork_small        raw     ok    NaN         68   
81            B     NeuralNetwork_small      scale     ok    NaN         68   
82            B     NeuralNetwork_small        pca     ok    NaN          8   
83            B     NeuralNetwork_small  scale_pca     ok    NaN         28   

         MSE      RMSE       MAE        R2  
0   2.287914  1.512585  0.810238  0.165007  
1   2.289115  1.512982  0.807902  0.164568  
2   2.496243  1.579950  0.825501  0.088975  
3   2.325464  1.524947  0.812558  0.151303  
4   2.595176  1.610955  0.829650  0.052869  
..       ...       ...       ...       ...  
79  1.334516  1.155213  0.593131  0.130991  
80  1.451214  1.204663  0.629196  0.055000  
81  1.352561  1.162997  0.668018  0.119241  
82  1.764104  1.328196  0.793369 -0.148747  
83  1.369101  1.170086  0.686410  0.108470  

[84 rows x 10 columns]
```


输出 2：

```text
        dataset                   model  MSE_delta     low95    high95
0   development                   Ridge   0.000000  0.000000  0.000000
1   development            DecisionTree   0.000000  0.000000  0.000000
2   development                AdaBoost   0.000000  0.000000  0.000000
3   development            RandomForest   0.000000  0.000000  0.000000
4   development        XGBoost_original   0.000000  0.000000  0.000000
5   development       XGBoost_optimized   0.000000  0.000000  0.000000
6   development                LightGBM  -0.020187 -0.057329  0.018751
7   development    HistGradientBoosting   0.000000  0.000000  0.000000
8   development  NeuralNetwork_original   0.000000  0.000000  0.000000
9   development     NeuralNetwork_small   0.000000  0.000000  0.000000
10  development                  Fusion  -0.002787 -0.011883  0.006530
11            B                   Ridge   0.000000  0.000000  0.000000
12            B            DecisionTree   0.000000  0.000000  0.000000
13            B                AdaBoost   0.000000  0.000000  0.000000
14            B            RandomForest   0.000000  0.000000  0.000000
15            B        XGBoost_original   0.000000  0.000000  0.000000
16            B       XGBoost_optimized   0.000000  0.000000  0.000000
17            B                LightGBM  -0.005658 -0.048267  0.032951
18            B    HistGradientBoosting   0.000000  0.000000  0.000000
19            B  NeuralNetwork_original   0.000000  0.000000  0.000000
20            B     NeuralNetwork_small   0.000000  0.000000  0.000000
21            B                  Fusion   0.000527 -0.008715  0.009747
```


## C013（0 基索引 12，In[None], markdown）

## 等权四模型融合
仅按各模型训练CV建议改变处理，权重仍各25%。


## C014（0 基索引 13，In[7], code）

```python
display(pd.read_csv(OUT / 'fusion_cv.csv'))
display(pd.read_csv(OUT / 'fusion_holdout.csv'))
```


输出 1：

```text
       version  seed       MSE      RMSE       MAE        R2
0    reference    23  1.754095  1.324422  0.714344  0.192763
1    reference  2026  1.757978  1.325888  0.718769  0.190976
2  recommended    23  1.750974  1.323244  0.712751  0.194199
3  recommended  2026  1.754941  1.324742  0.718107  0.192373
```


输出 2：

```text
       dataset      version       MSE      RMSE       MAE        R2
0  development    reference  2.177243  1.475548  0.777364  0.205397
1  development  recommended  2.174456  1.474604  0.776223  0.206414
2            B    reference  1.234296  1.110989  0.638160  0.196253
3            B  recommended  1.234823  1.111226  0.639440  0.195910
```


## C015（0 基索引 14，In[None], markdown）

## 重跑
默认关闭；重新运行将更新本轮对照结果和模型，不修改旧实验。


## C016（0 基索引 15，In[8], code）

```python
RUN_ABLATION = False
if RUN_ABLATION:
    import subprocess, sys
    for script in ['run_ablation.py', 'render_results.py']:
        subprocess.run([sys.executable, str(OUT / script)], cwd=ROOT, check=True)
```
