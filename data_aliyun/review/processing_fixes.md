# paper.ipynb：处理思路 2/3/4/5/6 修复记录

修改范围为主 notebook，不改原始数据或 Conda 环境。

## 修改前记录

上一次完整执行使用清洗后的3651条训练记录与1993条开发验证记录。记录的主要成绩：

| 模型 | 开发验证 MSE | R² |
|---|---:|---:|
| 神经网络 | 2.30208 | 0.15984 |
| 普通线性回归 | 2.31141690 | 0.15642895 |
| XGBoost | 2.37550137 | 0.13304079 |
| 随机森林 | 2.38456605 | 0.12973255 |
| AdaBoost | 2.41296 | 0.11937 |
| 决策树 | 2.54539 | 约0.07104 |

## 2：年龄和性别信息

年龄保留原始连续值；探索中的标准化结果保留浮点，不再转整型。未知性别不删除，使用固定性别映射，并新增未知标记。断言原始年龄逐行一致及标准化前后年龄唯一值数一致。

## 3：缺失值和极端值

训练和开发验证全部保留，分别为4649/1993条；原始train+A合计6642条。缺失较多记录和超过原阈值的记录仅计数，不删行、不裁剪血糖、不更改目标。

训练集填补副本仅用于 OLS/图表。真正模型输入保留NaN；每个训练折内用SimpleImputer估计中位数并增加缺失指示。keep_empty_features=True确保折内全空列不意外消失。

## 4：相关特征和PCA

不再强制删除乙肝五项或八个相关指标，也不把PCA统一施加于全部模型。

每个传统模型预先定义8组表示：全特征/删乙肝/删八个相关特征/原方案联合删列，各比较原始表示与PCA。使用相同五折训练划分；列筛选、填补、缩放及可选PCA都处于每折管线内。

预先固定保守采用规则：删列或PCA方案必须相对原始全特征五折均不变差、平均MSE严格更低，才有资格替换默认全特征原始表示；否则保留信息。报告包含平均MSE、同折平均差、胜出折数和采用标记。该规则不等于统计显著性检验，也不保证找到所有参数联合最优配置。

特征消融先固定一个模型配置，再对选中的表示调参。两阶段搜索的CV有选择偏差，真正的模型类别比较使用开发验证集。

普通最小二乘保留为训练诊断；预测基线改为Ridge(alpha=100)。原因是完整相关特征和缺失标记下，普通最小二乘在个别训练折会产生严重病态系数和巨大误差。正则化避免以无依据删特征来掩盖这种不稳定。

KMO/Bartlett和PCA方差图保留为训练探索，不再用于决定预测流程。

## 5：中间数据来源

主 notebook 直接使用原始CSV、本轮内存数据和拟合后的Pipeline。模型不读取旧填补/PCA/目标中间表。新增特征后也显式使用列名“血糖”，避免把最后一列误当目标。

## 6：开发、早停与外部复核

- 训练CV只选特征处理和超参数；搜索预算为决策树24组、AdaBoost18组、XGBoost24组、随机森林12组，替代原大网格。
- XGBoost轮数与树结构同时在训练CV中选择，不再扫描最终验证成绩。
- 神经网络只在训练集内部留出的930条记录上比较原始表示/PCA并早停；其余3719条内部训练数据独立拟合预处理。固定表示与轮数后，重新使用完整4649条训练记录拟合。原网络层结构保留，batch size设为32以配合BatchNormalization；EarlyStopping恢复最佳权重。
- 1993条开发验证记录用于比较六个模型类别。因此冠军的开发集成绩有选型偏差，不标为最终测试成绩。
- 冻结冠军类别、表示、参数/轮数后，在全部6642条开发数据上重拟合；最后才读取B的特征和无表头答案，只对这个已冻结方案复核一次。
- B此前已被查看，流程分离无法消除历史使用，所以只能称外部复核；继续按B调整就需要新的最终测试数据。

## 已完成的验证

缩小预算的结构测试覆盖全部模型路径、完整行数、年龄信息保留、未知性别和极端记录保留、逐折中位数、验证值扰动不改变拟合参数、8组消融的采用规则、六模型登记与冻结重拟合。最后外部读取使用合成样本测试，未在这些测试中查看真实B答案。

完整预算重跑已完成，结果如下。

## 完整预算运行结果

全部 43 个代码单元格已按顺序执行，无错误输出；所有结果保存在 paper.ipynb。

| 模型 | 开发 MSE | RMSE | MAE | R² |
|---|---:|---:|---:|---:|
| XGBoost | 2.228808 | 1.492919 | 0.780951 | 0.186578 |
| RandomForest | 2.276036 | 1.508654 | 0.786652 | 0.169341 |
| Ridge | 2.289115 | 1.512982 | 0.807902 | 0.164568 |
| NeuralNetwork | 2.375530 | 1.541275 | 0.748643 | 0.133030 |
| AdaBoost | 2.455500 | 1.567004 | 0.808051 | 0.103845 |
| DecisionTree | 2.595176 | 1.610955 | 0.829650 | 0.052869 |

开发集选出的冠军为 XGBoost。相对修改前最佳神经网络 2.30208，开发MSE变化为 3.18% 的下降。开发验证的1993行未变，但训练数据、基线和选型过程均改变，不能把收益单独归因于某个修复。

冻结后在全部6642条开发数据上重拟合，再对B的1000行作单模型外部复核：MSE=1.252744，RMSE=1.119260，MAE=0.648007，R²=0.184240。B历史使用的限制仍适用。

### 本轮训练折内特征/PCA消融记录

原始全特征为各模型的默认参考。以下是固定基线参数下的表示比较，随后才调参；不等于所有特征与参数组合的联合最优搜索。

```text
Ridge 选中预处理: ('raw', 'all')
representation   feature_set   CV_MSE  delta_MSE_vs_raw_all  fold_wins_vs_raw_all  eligible  selected
           raw           all 1.852585              0.000000                     0      True      True
           raw  no_hepatitis 1.850989             -0.001596                     3     False     False
           raw no_correlated 1.849604             -0.002982                     3     False     False
           raw   legacy_drop 1.847868             -0.004717                     4     False     False
           pca           all 1.861144              0.008559                     3     False     False
           pca  no_hepatitis 1.856754              0.004168                     3     False     False
           pca no_correlated 1.850856             -0.001729                     3     False     False
           pca   legacy_drop 1.847449             -0.005137                     3     False     False
{'model': 'Ridge', 'development_MSE': 2.2891152865485442, 'development_RMSE': 1.5129822492509766, 'development_MAE': 0.807901679692159, 'development_R2': 0.16456811419988981}
```

```text
DecisionTree 选中预处理: ('raw', 'all')
representation   feature_set   CV_MSE  delta_MSE_vs_raw_all  fold_wins_vs_raw_all  eligible  selected
           raw           all 2.096819              0.000000                     0      True      True
           raw  no_hepatitis 2.054097             -0.042721                     3     False     False
           raw no_correlated 2.049593             -0.047226                     4     False     False
           raw   legacy_drop 2.097340              0.000521                     2     False     False
           pca           all 2.143096              0.046278                     1     False     False
           pca  no_hepatitis 2.070819             -0.025999                     3     False     False
           pca no_correlated 2.075253             -0.021565                     4     False     False
           pca   legacy_drop 2.047444             -0.049375                     4     False     False
```

```text
AdaBoost 选中预处理: ('raw', 'all')
representation   feature_set   CV_MSE  delta_MSE_vs_raw_all  fold_wins_vs_raw_all  eligible  selected
           raw           all 4.586761              0.000000                     0      True      True
           raw  no_hepatitis 4.680862              0.094101                     3     False     False
           raw no_correlated 5.153151              0.566391                     1     False     False
           raw   legacy_drop 4.685833              0.099073                     3     False     False
           pca           all 4.918453              0.331692                     2     False     False
           pca  no_hepatitis 5.081084              0.494323                     1     False     False
           pca no_correlated 4.899678              0.312917                     2     False     False
           pca   legacy_drop 4.920459              0.333699                     2     False     False
```

```text
XGBoost 选中预处理: ('raw', 'no_hepatitis')
representation   feature_set   CV_MSE  delta_MSE_vs_raw_all  fold_wins_vs_raw_all  eligible  selected
           raw           all 1.857819              0.000000                     0      True     False
           raw  no_hepatitis 1.855265             -0.002554                     5      True      True
           raw no_correlated 1.851780             -0.006039                     3     False     False
           raw   legacy_drop 1.852025             -0.005794                     4     False     False
           pca           all 1.921873              0.064054                     0     False     False
           pca  no_hepatitis 1.902974              0.045155                     1     False     False
           pca no_correlated 1.889575              0.031756                     2     False     False
           pca   legacy_drop 1.853107             -0.004712                     3     False     False
```

```text
RandomForest 选中预处理: ('raw', 'all')
representation   feature_set   CV_MSE  delta_MSE_vs_raw_all  fold_wins_vs_raw_all  eligible  selected
           raw           all 1.863386              0.000000                     0      True      True
           raw  no_hepatitis 1.863036             -0.000351                     2     False     False
           raw no_correlated 1.869578              0.006191                     2     False     False
           raw   legacy_drop 1.865055              0.001669                     2     False     False
           pca           all 1.924826              0.061440                     0     False     False
           pca  no_hepatitis 1.893128              0.029741                     2     False     False
           pca no_correlated 1.877155              0.013769                     2     False     False
           pca   legacy_drop 1.861161             -0.002226                     2     False     False
```


神经网络内部比较可见主 notebook：其表示和最佳轮数均由内部验证决定，不根据开发或B误差回调。没有保证未来数据的准确度；数据或模型继续改动后应使用新的未参与开发样本评估。

## 后续补充：全部冻结模型的 B 集对照

2026-10-18 按用户要求，已将主 notebook 最后的单模型复核扩展为六个冻结算法的外部基准对照，不再据B调参。完整对照见 `external_b_comparison/report.md`。按MSE两组前两名一致，但完整排序不完全一致；按MAE均为神经网络最低。原文的单模型结果作为前一阶段记录保留。
