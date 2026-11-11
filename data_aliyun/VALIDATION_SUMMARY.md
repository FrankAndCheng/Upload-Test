# 血糖预测：验证思路、模型组合与MSE证据汇总

整理日期：2026-10-22。本文按已有真实结果整理，不重新训练、不根据B重新挑模型。主指标为 **MSE（均方误差，越低越好）**；“MES”在本文统一写作MSE。分数保留6位小数，完整精度以链接的CSV/JSON为准。

## 1. 先给结论

1. 六个原算法中，原配置XGBoost的开发/B MSE分别为 **2.228808/1.252744**，在两组上均排第一；这个结论仅针对原六模型候选。
2. 顺序优化后，通过训练OOF选择的四模型等权融合为 **XGBoost优化配置＋LightGBM＋Ridge＋去乙肝小网络，各25%**，开发/B MSE为 **2.177243/1.234296**，相对原XGBoost分别降低 **2.31%/1.47%**。它是选型流程得出的方案，不能把“在所有算法、未来数据上必然最好”当作结论。
3. 预处理对照中，树模型没有因标准化或PCA取得稳定收益；Ridge标准化差异极小；两个网络标准化分别降低平均CV MSE约7.63%、20.33%。**各模型CV最低方案均不含PCA**，但原大网络PCA差异不足变更门槛，保留旧流程不等于证明PCA必需。
4. LightGBM改用中位数填补的实验版使融合开发MSE略降、B略升，未得到进一步稳定提升的证据。原四模型文件没有因此被覆盖。
5. 开发集、A/B均已被多次查看，以上属于已知数据上的验证与外部复核。配对区间中，四模型相对基线的B收益仍跨零，不能宣称已经证明未来泛化收益。

本文阅读顺序：先看第2～5节的验证口径与主表，再看第6～8节的全部组合/预处理对照，最后按第9节定位配置和源码。未评估的得分记“—”，绝不当作0或补造外部成绩。

## 2. 什么成绩可以互相比较

MSE定义为：

\[
\operatorname{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2.
\]

MSE对大误差更敏感，适合本项目的主要比较目标，但不是“准确率百分比”。降低2%的MSE不等于准确率增加2个百分点。MAE回答另一类问题：历史原大网络的B MAE低于原XGBoost，因此不能说XGBoost在所有指标都最好。

### 2.1 三种历史协议必须分开

| 协议 | 拟合/选型数据 | 验证数据 | 用途与限制 | 来源 |
| --- | --- | --- | --- | --- |
| H1：早期清洗子集 | 清洗后5202条；3641条训练 | 1561条验证 | 用于旧方案同样本比较；不能与当前1993条的绝对MSE直接排名 | [旧协议源码](paper_optimized.py#L151) |
| H2：原train外部验证 | 仅原train 5642条 | 完整A1000、完整B1000 | A不进入拟合；与后来train+A拟合的模型不是同训练口径 | [外部协议源码](paper_optimized.py#L170) |
| C：当前主协议 | train 5642＋带标签A1000＝6642；固定70/30拆分 | 训练4649、开发1993；外部B1000 | 本文当前模型、融合和预处理主结论采用这一协议 | [源码](review/sequential_experiments/run_experiments.py#L181) |

当前拆分使用`train_test_split(test_size=0.3, random_state=23)`。1993条开发验证由 **原train 1685条＋A 308条**组成，**不是纯A验证集**。相应4649条训练为原train 3957条＋A 692条。

- 开发评分：模型与填补/缩放/PCA均只拟合4649条，预测相同1993条。
- B评分：模型类型、特征、参数、轮数、权重冻结后，在全部6642条上重拟合，再预测相同1000条B。
- 开发与B的样本分布及拟合样本数不同。因此不能把“B MSE小于开发MSE”直接解释成模型升级；模型间比较应在同一列进行。
- id只做行对齐，日期不入模；年龄保留连续值，未知性别保留并标记，缺失与极端样本不按目标删行。

来源：[划分与加载](review/sequential_experiments/run_experiments.py#L181)；[训练/开发id归属](review/sequential_experiments/split_ids.csv)；[主notebook](paper.ipynb)。

## 3. 当前验证思路：每一步解决什么问题

1. **先划分，再拟合预处理。** 所有学习到的中位数、均值/方差、PCA方向都放在训练折内。特征工程只使用同一行的输入，不接触血糖、id或体检日期。
2. **训练内部筛选候选。** 原六模型在训练五折内选表示/参数。随后做8组特征消融、100组XGBoost＋12组LightGBM＋8组HistGradientBoosting的联合随机搜索。
3. **用OOF预测学习融合权重。** OOF指每条训练记录的预测来自没有拟合该记录的那一折模型；不使用模型对自身训练行的拟合预测来挑权重。
4. **先做40组树/线性融合。** 5个成员池的两两组合分别比较25%/50%/75%权重，另比较三成员等权，选出三模型等权方案T。
5. **网络在训练内部早停，再冻结轮数。** 原小网络4个结构/表示候选后，因训练OOF极端输入异常，再补2个去乙肝对照。每个训练折内部切20%早停，选中的小网络最终固定36轮；不是用开发/B挑轮数。
6. **补充9组网络融合。** N分别与X0、X、T按25%/50%/75%网络权重组合；按第一套训练五折选择，再用第二套五折复核同一方案。最终得到N25%＋T75%，展开为四模型各25%。
7. **冻结后复核开发/B。** 两套五折仍参与选型，不能称作完整搜索流程的无偏最终测试。开发集最早用于比较模型类别，后来也多次查看；B同样有历史使用，只能视为外部复核。
8. **最后单独对照预处理。** 固定参数/特征/网络轮数，做“无标准化无PCA、仅标准化、仅PCA、标准化后PCA”的2×2对照，另为两类原生缺失模型保留参考组。

源码：[特征工程](review/sequential_experiments/run_experiments.py#L46)；[训练OOF](review/sequential_experiments/run_experiments.py#L150)；[联合搜索](review/sequential_experiments/run_experiments.py#L216)；[40组融合](review/sequential_experiments/run_experiments.py#L248)；[网络特征补充](review/sequential_experiments/refine_nn_features.py#L13)；[9组网络融合](review/sequential_experiments/complete_nn_blends.py#L30)；[2×2预处理](review/preprocessing_ablation/run_ablation.py#L80)。

**重要口径差异：** 融合选型阶段的网络在不同训练折里可以早停到不同轮数，之后以中位数冻结36轮；预处理对照则各折一律固定36轮（大网络固定87轮）。两者OOF均值不能混成同一行或把差值全部归因于预处理。

## 4. 原六算法的冻结成绩

这张表来自同一次原六算法对照，非后来小网络或新XGBoost配置。每行都在当前C协议下评价。

| 模型 | 开发MSE | 开发排名 | B MSE | B排名 | 分数来源 | 源码/参数 |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost原配置 | 2.228808 | 1 | 1.252744 | 1 | [得分记录](review/external_b_comparison/comparison.csv#L2) | [源码](paper.ipynb#L3622) · [冻结参数](review/external_b_comparison/frozen_configurations.json#L96) |
| 随机森林 | 2.276036 | 2 | 1.334292 | 2 | [得分记录](review/external_b_comparison/comparison.csv#L3) | [源码](paper.ipynb#L3732) · [冻结参数](review/external_b_comparison/frozen_configurations.json#L155) |
| 原大网络（历史存档） | 2.375530 | 4 | 1.342265 | 3 | [得分记录](review/external_b_comparison/comparison.csv#L4) | [源码](paper.ipynb#L3197) · [冻结参数](review/external_b_comparison/frozen_configurations.json#L23) |
| Ridge | 2.289115 | 3 | 1.372016 | 4 | [得分记录](review/external_b_comparison/comparison.csv#L5) | [源码](paper.ipynb#L3143) · [冻结参数](review/external_b_comparison/frozen_configurations.json#L3) |
| 决策树 | 2.595176 | 6 | 1.401940 | 5 | [得分记录](review/external_b_comparison/comparison.csv#L6) | [源码](paper.ipynb#L3427) · [冻结参数](review/external_b_comparison/frozen_configurations.json#L32) |
| AdaBoost | 2.455500 | 5 | 1.444966 | 6 | [得分记录](review/external_b_comparison/comparison.csv#L7) | [源码](paper.ipynb#L3523) · [冻结参数](review/external_b_comparison/frozen_configurations.json#L59) |

**结论：** 原六模型中XGBoost在开发/B均第一，随机森林均第二；中间排名不同，不能说完整排序完全一致。原大网络B MAE为0.596760，优于原XGBoost的0.648007，所以“最好”必须限定MSE目标。

原模型的重建及开发分数复现入口为[源码](review/external_b_comparison/evaluate_frozen.py#L24)；其中从主notebook提取已冻结构造器和参数，不重跑参数搜索。

## 5. 从单模型到四模型融合：当前主证据

“CV23/2026”为同一4649条训练记录的两套五折OOF MSE；均值是选型分数，不是独立测试分数。

| 方案 | 内部ID | CV23 | CV2026 | CV均值 | 开发MSE | B MSE | 得分与源码 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 原XGBoost基线 | baseline_xgb | 1.803531 | 1.810445 | 1.806988 | 2.228808 | 1.252744 | [CV记录](review/sequential_experiments/confirmation.csv#L2) · [开发记录](review/sequential_experiments/evaluation.csv#L2) · [B记录](review/sequential_experiments/evaluation.csv#L7) · [源码](review/sequential_experiments/run_experiments.py#L181) |
| XGBoost＋组合特征 | feature_combined | 1.783699 | 1.789320 | 1.786509 | 2.218877 | 1.275351 | [CV记录](review/sequential_experiments/confirmation.csv#L3) · [开发记录](review/sequential_experiments/evaluation.csv#L3) · [B记录](review/sequential_experiments/evaluation.csv#L8) · [源码](review/sequential_experiments/run_experiments.py#L181) |
| 联合搜索后的XGBoost | joint_xgb_089 | 1.773159 | 1.790562 | 1.781861 | 2.236749 | 1.242601 | [CV记录](review/sequential_experiments/confirmation.csv#L4) · [开发记录](review/sequential_experiments/evaluation.csv#L4) · [B记录](review/sequential_experiments/evaluation.csv#L9) · [源码](review/sequential_experiments/run_experiments.py#L181) |
| XGBoost＋Ridge＋LightGBM等权 | blend_33 | 1.758488 | 1.767456 | 1.762972 | 2.196540 | 1.245558 | [CV记录](review/sequential_experiments/confirmation.csv#L5) · [开发记录](review/sequential_experiments/evaluation.csv#L5) · [B记录](review/sequential_experiments/evaluation.csv#L10) · [源码](review/sequential_experiments/run_experiments.py#L181) |
| 128→64→16小网络＋PCA | nn_128_pca | 2.245105 | 3.068600 | 2.656852 | 2.392069 | 1.388255 | [CV记录](review/sequential_experiments/confirmation.csv#L6) · [开发记录](review/sequential_experiments/evaluation.csv#L6) · [B记录](review/sequential_experiments/evaluation.csv#L11) · [源码](review/sequential_experiments/run_experiments.py#L181) |
| 去乙肝64→32小网络 | nn_64_nohep_raw | 1.897055 | 1.926229 | 1.911642 | 2.260269 | 1.352561 | [CV记录](review/sequential_experiments/confirmation.csv#L7) · [开发记录](review/sequential_experiments/evaluation.csv#L12) · [B记录](review/sequential_experiments/evaluation.csv#L13) · [源码](review/sequential_experiments/refine_nn_features.py#L30) |
| XGBoost＋LightGBM＋Ridge＋小网络各25% | nn_blend_06 | 1.754295 | 1.764841 | 1.759568 | 2.177243 | 1.234296 | [CV记录](review/sequential_experiments/confirmation.csv#L8) · [开发记录](review/sequential_experiments/evaluation.csv#L14) · [B记录](review/sequential_experiments/evaluation.csv#L15) · [源码](review/sequential_experiments/complete_nn_blends.py#L30) |

### 5.1 从这些分数能得出什么

- 组合特征使开发略好，却使B从1.252744变成1.275351，不支持“特征越多越好”。
- 联合搜索后的XGBoost使B降到1.242601，但开发略差；不能只看B就宣布新参数全面胜出。
- 三模型融合T在开发/B均优于原XGBoost，四模型等权进一步在本轮两列都降低MSE。**融合有效的依据是OOF误差互补与冻结后的实测，而不是把排行榜前几名任意平均。**
- N单模型的MSE较差，但以25%加入T后有互补收益。单模型排名较低并不意味着它对融合没有价值。
- `nn_128_pca`属于中间小网络候选，不是最早400→160→64→26原大网络；两者不能混称“神经网络”。

### 5.2 四模型具体配置与组合公式

记X为优化XGBoost，L为LightGBM，R为Ridge，N为去乙肝小网络：

\[
T=(X+R+L)/3,\qquad F_4=0.75T+0.25N=(X+L+R+N)/4.
\]

这是预测值加权平均，**没有训练第二层回归器，不是Stacking**。权重来源：[源码](review/sequential_experiments/complete_nn_blends.py#L30)；最终决定：[冻结配置](review/sequential_experiments/frozen_selection.json#L2813)。

固定配置如下（未列参数采用对应冻结配置和已记录依赖版本的默认值）：

| 代号 | 模型/内部ID | 特征与预处理 | 固定参数 | 源码与配置 |
| --- | --- | --- | --- | --- |
| X | XGBoost优化配置 / joint_xgb_089 | 去乙肝＋年龄交互，中位数填补/缺失指示；不标准化、不PCA | 393棵；lr=0.03962175009621365；depth=5；child=22.839155331252698；λ=119.89079839483685；α=1；gamma=.05；行采样=.8；列采样=.65；hist；seed23 | [源码](review/sequential_experiments/run_experiments.py#L84) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1691) |
| L | LightGBM / joint_lgb_05 | 去乙肝＋年龄交互；原生缺失；不标准化、不PCA | 800棵；lr=.015；15叶；min_child_samples=20；λ=1；列采样=1；deterministic=True；seed23 | [源码](review/sequential_experiments/run_experiments.py#L84) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1948) |
| R | Ridge / frozen_ridge | 全原始特征；中位数填补＋缺失指示＋标准化；不PCA | alpha=100；拟合截距 | [源码](review/sequential_experiments/run_experiments.py#L75) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2160) |
| N | 小网络 / nn_64_nohep_raw | 去乙肝；中位数填补＋缺失指示＋标准化；不PCA | 64→32→1；隐藏层ReLU；Dropout=.1；L2=.001；Adam lr=.001；batch64；最终36轮；seed23 | [源码](review/sequential_experiments/run_experiments.py#L103) · [源码](review/sequential_experiments/refine_nn_features.py#L30) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2718) |

两种树模型的年龄交互为：年龄×甘油三酯、年龄×高密度脂蛋白、年龄×尿酸、年龄×丙氨酸氨基转换酶、年龄平方。R保留乙肝五项，X/L/N不保留；四个模型并不是共用完全相同的输入矩阵。特征构造见[源码](review/sequential_experiments/run_experiments.py#L46)。

### 5.3 改善幅度的不确定性

差值定义为“其他方案MSE－原XGBoost MSE”，负数代表改善。

| 数据集 | 方案 | MSE差值 | 95%下界 | 95%上界 | 来源 |
| --- | --- | --- | --- | --- | --- |
| development | XGBoost＋Ridge＋LightGBM等权 | -0.032268 | -0.068530 | 0.001123 | [得分记录](review/sequential_experiments/paired_intervals.csv#L4) |
| B | XGBoost＋Ridge＋LightGBM等权 | -0.007185 | -0.042229 | 0.024062 | [得分记录](review/sequential_experiments/paired_intervals.csv#L8) |
| development | XGBoost＋LightGBM＋Ridge＋小网络各25% | -0.051565 | -0.101899 | -0.003706 | [得分记录](review/sequential_experiments/paired_intervals.csv#L12) |
| B | XGBoost＋LightGBM＋Ridge＋小网络各25% | -0.018448 | -0.053797 | 0.016102 | [得分记录](review/sequential_experiments/paired_intervals.csv#L13) |

区间为固定模型、固定训练样本下的行配对bootstrap，未覆盖训练随机性、搜索偏差、日期聚类或多重比较。四模型B区间跨零，因此只能说本轮MSE点估计改善，不能说已证明未来稳定优于基线。

## 6. 全部组合模型的得分

### 6.1 成员代号及独立成绩

X、R、L、N定义同上；X0为原XGBoost，F为冻结随机森林，H为HistGradientBoosting，T为三模型等权。下表的单模型开发/B分数取自预处理对照的对应原处理重训，用于了解成员，不冒充融合搜索当天每个候选都单独做过B评价。

| 代号 | 模型ID | 参考处理 | 开发MSE | B MSE | 分数/配置来源 |
| --- | --- | --- | --- | --- | --- |
| X | joint_xgb_089 | 填补，无标准化/无PCA | 2.236749 | 1.242601 | [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L22) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L64) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1691) |
| R | frozen_ridge | 填补＋标准化 | 2.289115 | 1.372016 | [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L3) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L45) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2160) |
| F | frozen_forest | 填补，无标准化/无PCA | 2.276036 | 1.334292 | [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L14) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L56) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2150) |
| L | joint_lgb_05 | 原生缺失，无标准化/无PCA | 2.211554 | 1.268888 | [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L30) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L72) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1948) |
| H | joint_hist_04 | 原生缺失，无标准化/无PCA | 2.218320 | 1.265633 | [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L35) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L77) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2098) |
| N | nn_64_nohep_raw | 填补＋标准化 | 2.260269 | 1.352561 | [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L41) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L83) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2718) |

### 6.2 40组树/线性融合

五个成员X/R/F/L/H的10对×3权重＝30组，以及10组三成员等权，共40组。下表按CV23排序。三成员的33.33%仅为显示四舍五入，实际权重均为1/3。只有胜出方案T做了第二套五折及阶段开发/B评价；其他组合的“—”表示该阶段没有评估，不表示分数相同或为零。

<details>
<summary>展开40组完整MSE、组合权重与源码</summary>

| 组合ID | 组合公式 | CV23 MSE | CV2026 MSE | 开发MSE | B MSE | 分数/源码/配置 |
| --- | --- | --- | --- | --- | --- | --- |
| blend_33 | 33.33%×X＋33.33%×R＋33.33%×L | 1.758488 | 1.767456 | 2.196540 | 1.245558 | [CV23记录](review/sequential_experiments/blends.json#L268) · [CV2026记录](review/sequential_experiments/confirmation.csv#L5) · [开发记录](review/sequential_experiments/evaluation.csv#L5) · [B记录](review/sequential_experiments/evaluation.csv#L10) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2570) |
| blend_35 | 33.33%×X＋33.33%×L＋33.33%×H | 1.761223 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L284) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2598) |
| blend_08 | 75.00%×X＋25.00%×L | 1.762240 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L68) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2264) |
| blend_07 | 50.00%×X＋50.00%×L | 1.763440 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L60) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2252) |
| blend_39 | 33.33%×R＋33.33%×L＋33.33%×H | 1.763558 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L316) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2654) |
| blend_05 | 75.00%×X＋25.00%×R | 1.765691 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L44) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2228) |
| blend_34 | 33.33%×X＋33.33%×R＋33.33%×H | 1.767210 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L276) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2584) |
| blend_11 | 75.00%×X＋25.00%×H | 1.768447 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L92) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2300) |
| blend_10 | 50.00%×X＋50.00%×H | 1.768685 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L84) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2288) |
| blend_31 | 33.33%×X＋33.33%×F＋33.33%×L | 1.769587 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L252) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2542) |
| blend_28 | 50.00%×L＋50.00%×H | 1.770470 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L228) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2504) |
| blend_27 | 25.00%×L＋75.00%×H | 1.771583 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L220) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2492) |
| blend_02 | 75.00%×X＋25.00%×F | 1.773274 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L20) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2192) |
| blend_38 | 33.33%×F＋33.33%×L＋33.33%×H | 1.773658 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L308) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2640) |
| blend_22 | 50.00%×R＋50.00%×L | 1.773708 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L180) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2432) |
| blend_09 | 25.00%×X＋75.00%×H | 1.773874 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L76) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2276) |
| blend_21 | 25.00%×R＋75.00%×L | 1.774525 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L172) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2420) |
| blend_24 | 25.00%×R＋75.00%×H | 1.775081 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L196) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2456) |
| blend_32 | 33.33%×X＋33.33%×F＋33.33%×H | 1.775083 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L260) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2556) |
| blend_04 | 50.00%×X＋50.00%×R | 1.776448 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L36) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2216) |
| blend_06 | 25.00%×X＋75.00%×L | 1.776760 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L52) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2240) |
| blend_29 | 75.00%×L＋25.00%×H | 1.780675 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L236) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2516) |
| blend_18 | 25.00%×F＋75.00%×H | 1.780977 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L148) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2384) |
| blend_36 | 33.33%×F＋33.33%×R＋33.33%×L | 1.781609 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L292) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2612) |
| blend_25 | 50.00%×R＋50.00%×H | 1.783541 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L204) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2468) |
| blend_30 | 33.33%×X＋33.33%×F＋33.33%×R | 1.785046 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L244) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2528) |
| blend_15 | 25.00%×F＋75.00%×L | 1.785865 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L124) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2348) |
| blend_01 | 50.00%×X＋50.00%×F | 1.786045 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L12) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2180) |
| blend_16 | 50.00%×F＋50.00%×L | 1.788314 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L132) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2360) |
| blend_37 | 33.33%×F＋33.33%×R＋33.33%×H | 1.789145 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L300) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2626) |
| blend_19 | 50.00%×F＋50.00%×H | 1.790889 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L156) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2396) |
| blend_23 | 75.00%×R＋25.00%×L | 1.799747 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L188) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2444) |
| blend_03 | 25.00%×X＋75.00%×R | 1.805432 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L28) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2204) |
| blend_26 | 75.00%×R＋25.00%×H | 1.809395 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L212) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2480) |
| blend_17 | 75.00%×F＋25.00%×L | 1.809544 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L140) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2372) |
| blend_00 | 25.00%×X＋75.00%×F | 1.811473 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L4) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2168) |
| blend_20 | 75.00%×F＋25.00%×H | 1.813749 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L164) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2408) |
| blend_13 | 50.00%×F＋50.00%×R | 1.822699 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L108) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2324) |
| blend_14 | 75.00%×F＋25.00%×R | 1.829028 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L116) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2336) |
| blend_12 | 25.00%×F＋75.00%×R | 1.830570 | — | — | — | [CV23记录](review/sequential_experiments/blends.json#L100) · [源码](review/sequential_experiments/run_experiments.py#L248) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2312) |

</details>

### 6.3 9组小网络融合

每个方案均保存两套训练OOF得分，但权重先按CV23选择，再用CV2026复核同一方案。第二套折也参与最终方案比较，仍属于选型。阶段中仅nn_blend_06保存了冻结后的开发/B成绩。

| 组合ID | 组合公式 | CV23 MSE | CV2026 MSE | CV均值 | 开发MSE | B MSE | 分数/源码 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nn_blend_00 | 25.00%×N＋75.00%×X0 | 1.781322 | 1.789464 | 1.785393 | — | — | [CV记录](review/sequential_experiments/nn_blend_training_cv.csv#L2) · [源码](review/sequential_experiments/complete_nn_blends.py#L30) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2741) |
| nn_blend_01 | 50.00%×N＋50.00%×X0 | 1.789506 | 1.801768 | 1.795637 | — | — | [CV记录](review/sequential_experiments/nn_blend_training_cv.csv#L3) · [源码](review/sequential_experiments/complete_nn_blends.py#L30) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2753) |
| nn_blend_02 | 75.00%×N＋25.00%×X0 | 1.828084 | 1.847356 | 1.837720 | — | — | [CV记录](review/sequential_experiments/nn_blend_training_cv.csv#L4) · [源码](review/sequential_experiments/complete_nn_blends.py#L30) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2765) |
| nn_blend_03 | 25.00%×N＋75.00%×X | 1.757638 | 1.773118 | 1.765378 | — | — | [CV记录](review/sequential_experiments/nn_blend_training_cv.csv#L5) · [源码](review/sequential_experiments/complete_nn_blends.py#L30) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2777) |
| nn_blend_04 | 50.00%×N＋50.00%×X | 1.773114 | 1.789914 | 1.781514 | — | — | [CV记录](review/sequential_experiments/nn_blend_training_cv.csv#L6) · [源码](review/sequential_experiments/complete_nn_blends.py#L30) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2789) |
| nn_blend_05 | 75.00%×N＋25.00%×X | 1.819586 | 1.840951 | 1.830269 | — | — | [CV记录](review/sequential_experiments/nn_blend_training_cv.csv#L7) · [源码](review/sequential_experiments/complete_nn_blends.py#L30) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2801) |
| nn_blend_06 | 25.00%×N＋75.00%×T | 1.754295 | 1.764841 | 1.759568 | 2.177243 | 1.234296 | [CV记录](review/sequential_experiments/nn_blend_training_cv.csv#L8) · [开发记录](review/sequential_experiments/evaluation.csv#L14) · [B记录](review/sequential_experiments/evaluation.csv#L15) · [源码](review/sequential_experiments/complete_nn_blends.py#L30) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2813) |
| nn_blend_07 | 50.00%×N＋50.00%×T | 1.775992 | 1.790432 | 1.783212 | — | — | [CV记录](review/sequential_experiments/nn_blend_training_cv.csv#L9) · [源码](review/sequential_experiments/complete_nn_blends.py#L30) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2825) |
| nn_blend_08 | 75.00%×N＋25.00%×T | 1.823578 | 1.844228 | 1.833903 | — | — | [CV记录](review/sequential_experiments/nn_blend_training_cv.csv#L10) · [源码](review/sequential_experiments/complete_nn_blends.py#L30) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2837) |

N25%＋T75%就是四个基础模型各25%，不是N25%、另外三个模型各75%。

## 7. 标准化/PCA对照：处理方法的直接证据

十个模型版本的四格对照均使用相同中位数填补和缺失指示。PCA保留98%方差；“仅PCA”会中心化但不做单位方差缩放。“标准化后PCA”不再对主成分二次缩放。LightGBM/H单独保留原生缺失参考，不把缺失处理差异全算作PCA效应。

| 模型 | 无标准化无PCA | 仅标准化 | 仅PCA | 标准化后PCA | 分数/源码 |
| --- | --- | --- | --- | --- | --- |
| Ridge | 1.848363 | 1.849622 | 1.944901 | 1.858446 | [raw](review/preprocessing_ablation/cv_summary.csv#L2) · [scale](review/preprocessing_ablation/cv_summary.csv#L3) · [pca](review/preprocessing_ablation/cv_summary.csv#L4) · [scale_pca](review/preprocessing_ablation/cv_summary.csv#L5) · [源码](review/preprocessing_ablation/run_ablation.py#L80) |
| 决策树 | 2.033633 | 2.033633 | 2.087293 | 2.107930 | [raw](review/preprocessing_ablation/cv_summary.csv#L6) · [scale](review/preprocessing_ablation/cv_summary.csv#L7) · [pca](review/preprocessing_ablation/cv_summary.csv#L8) · [scale_pca](review/preprocessing_ablation/cv_summary.csv#L9) · [源码](review/preprocessing_ablation/run_ablation.py#L80) |
| AdaBoost | 1.990995 | 1.990819 | 2.016526 | 2.051130 | [raw](review/preprocessing_ablation/cv_summary.csv#L10) · [scale](review/preprocessing_ablation/cv_summary.csv#L11) · [pca](review/preprocessing_ablation/cv_summary.csv#L12) · [scale_pca](review/preprocessing_ablation/cv_summary.csv#L13) · [源码](review/preprocessing_ablation/run_ablation.py#L80) |
| 随机森林 | 1.851444 | 1.851883 | 1.942392 | 1.898163 | [raw](review/preprocessing_ablation/cv_summary.csv#L14) · [scale](review/preprocessing_ablation/cv_summary.csv#L15) · [pca](review/preprocessing_ablation/cv_summary.csv#L16) · [scale_pca](review/preprocessing_ablation/cv_summary.csv#L17) · [源码](review/preprocessing_ablation/run_ablation.py#L80) |
| XGBoost原配置 | 1.806988 | 1.806988 | 1.939866 | 1.872072 | [raw](review/preprocessing_ablation/cv_summary.csv#L18) · [scale](review/preprocessing_ablation/cv_summary.csv#L19) · [pca](review/preprocessing_ablation/cv_summary.csv#L20) · [scale_pca](review/preprocessing_ablation/cv_summary.csv#L21) · [源码](review/preprocessing_ablation/run_ablation.py#L80) |
| XGBoost优化配置 | 1.781861 | 1.781861 | 2.065189 | 1.841900 | [raw](review/preprocessing_ablation/cv_summary.csv#L22) · [scale](review/preprocessing_ablation/cv_summary.csv#L23) · [pca](review/preprocessing_ablation/cv_summary.csv#L24) · [scale_pca](review/preprocessing_ablation/cv_summary.csv#L25) · [源码](review/preprocessing_ablation/run_ablation.py#L80) |
| LightGBM | 1.796304 | 1.805506 | 2.127522 | 1.875117 | [raw](review/preprocessing_ablation/cv_summary.csv#L26) · [scale](review/preprocessing_ablation/cv_summary.csv#L27) · [pca](review/preprocessing_ablation/cv_summary.csv#L28) · [scale_pca](review/preprocessing_ablation/cv_summary.csv#L29) · [源码](review/preprocessing_ablation/run_ablation.py#L80) |
| HistGradientBoosting | 1.785411 | 1.784776 | 2.052189 | 1.884365 | [raw](review/preprocessing_ablation/cv_summary.csv#L31) · [scale](review/preprocessing_ablation/cv_summary.csv#L32) · [pca](review/preprocessing_ablation/cv_summary.csv#L33) · [scale_pca](review/preprocessing_ablation/cv_summary.csv#L34) · [源码](review/preprocessing_ablation/run_ablation.py#L80) |
| 原大网络（本轮重训） | 2.089127 | 1.929678 | 1.983365 | 1.930533 | [raw](review/preprocessing_ablation/cv_summary.csv#L36) · [scale](review/preprocessing_ablation/cv_summary.csv#L37) · [pca](review/preprocessing_ablation/cv_summary.csv#L38) · [scale_pca](review/preprocessing_ablation/cv_summary.csv#L39) · [源码](review/preprocessing_ablation/run_ablation.py#L80) |
| 去乙肝小网络 | 2.379670 | 1.895841 | 3.325018 | 1.941302 | [raw](review/preprocessing_ablation/cv_summary.csv#L40) · [scale](review/preprocessing_ablation/cv_summary.csv#L41) · [pca](review/preprocessing_ablation/cv_summary.csv#L42) · [scale_pca](review/preprocessing_ablation/cv_summary.csv#L43) · [源码](review/preprocessing_ablation/run_ablation.py#L80) |

### 7.1 证据与操作决定分开

替换门槛事先设为：相对现有方案两套OOF平均MSE至少降低0.5%，且赢得至少7/10配对折。它是工程变更规则，不是统计显著性检验。

| 模型 | CV最低方案 | 按门槛保留/建议 | 解释 | 来源 |
| --- | --- | --- | --- | --- |
| Ridge | 填补，无标准化/无PCA | 填补＋标准化 | 标准化收益未获明确支持，可沿用现有处理；PCA无收益 | [得分记录](review/preprocessing_ablation/recommendations.csv#L2) · [对照入口](review/preprocessing_ablation/run_ablation.py#L177) |
| 决策树 | 填补，无标准化/无PCA | 填补，无标准化/无PCA | 标准化前后差异很小，无需增加PCA | [得分记录](review/preprocessing_ablation/recommendations.csv#L3) · [对照入口](review/preprocessing_ablation/run_ablation.py#L177) |
| AdaBoost | 填补＋标准化 | 填补，无标准化/无PCA | 标准化前后差异很小，无需增加PCA | [得分记录](review/preprocessing_ablation/recommendations.csv#L4) · [对照入口](review/preprocessing_ablation/run_ablation.py#L177) |
| 随机森林 | 填补，无标准化/无PCA | 填补，无标准化/无PCA | 标准化前后差异很小，无需增加PCA | [得分记录](review/preprocessing_ablation/recommendations.csv#L5) · [对照入口](review/preprocessing_ablation/run_ablation.py#L177) |
| XGBoost原配置 | 填补，无标准化/无PCA | 填补，无标准化/无PCA | 标准化前后差异很小，无需增加PCA | [得分记录](review/preprocessing_ablation/recommendations.csv#L6) · [对照入口](review/preprocessing_ablation/run_ablation.py#L177) |
| XGBoost优化配置 | 填补，无标准化/无PCA | 填补，无标准化/无PCA | 标准化前后差异很小，无需增加PCA | [得分记录](review/preprocessing_ablation/recommendations.csv#L7) · [对照入口](review/preprocessing_ablation/run_ablation.py#L177) |
| LightGBM | 填补，无标准化/无PCA | 填补，无标准化/无PCA | 填补替代原生缺失改善CV约0.65%，赢7/10折；不需要标准化/PCA | [得分记录](review/preprocessing_ablation/recommendations.csv#L8) · [对照入口](review/preprocessing_ablation/run_ablation.py#L177) |
| HistGradientBoosting | 填补＋标准化 | 原生缺失，无标准化/无PCA | 标准化前后差异很小，无需增加PCA | [得分记录](review/preprocessing_ablation/recommendations.csv#L9) · [对照入口](review/preprocessing_ablation/run_ablation.py#L177) |
| 原大网络（本轮重训） | 填补＋标准化 | 填补＋标准化＋PCA | 标准化改善7.63%，赢10/10折；再加PCA无收益；保留旧PCA不代表必需 | [得分记录](review/preprocessing_ablation/recommendations.csv#L10) · [对照入口](review/preprocessing_ablation/run_ablation.py#L177) |
| 去乙肝小网络 | 填补＋标准化 | 填补＋标准化 | 标准化改善20.33%，赢10/10折；再加PCA平均变差约2.40% | [得分记录](review/preprocessing_ablation/recommendations.csv#L11) · [对照入口](review/preprocessing_ablation/run_ablation.py#L177) |

原大网络仅标准化CV为1.929678，标准化后PCA为1.930533，差异约0.044%，未达到变更门槛。因此不能把“暂保留旧PCA”写成“实验证明PCA必要”。

该轮原大网络统一clear_session后重新初始化，其开发“标准化后PCA”MSE为2.395645，不是历史存档的2.375530。旧权重与新重训实例没有逐值复现；该差异的具体数值来源未再分解。本轮四组仍遵循相同初始化协议，应使用本轮成对成绩判断，而不是拼接旧、新网络结果。

源码：[网络构造](review/preprocessing_ablation/run_ablation.py#L102)；[折内拟合](review/preprocessing_ablation/run_ablation.py#L127)；[门槛判定](review/preprocessing_ablation/run_ablation.py#L224)。

### 7.2 全部42组处理的开发/B复核

以下42行均列出两组成绩；它们在预处理推荐冻结后才评价，不用于回选处理。原网络对应本轮重训实例。

<details>
<summary>展开42组完整CV、开发/B MSE及来源</summary>

| 模型 | 处理 | 两套CV均值 | 开发MSE | B MSE | 分数/源码 |
| --- | --- | --- | --- | --- | --- |
| Ridge | 填补，无标准化/无PCA | 1.848363 | 2.287914 | 1.375474 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L2) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L2) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L44) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| Ridge | 填补＋标准化 | 1.849622 | 2.289115 | 1.372016 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L3) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L3) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L45) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| Ridge | 填补＋PCA | 1.944901 | 2.496243 | 1.456986 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L4) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L4) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L46) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| Ridge | 填补＋标准化＋PCA | 1.858446 | 2.325464 | 1.369370 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L5) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L5) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L47) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| 决策树 | 填补，无标准化/无PCA | 2.033633 | 2.595176 | 1.401940 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L6) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L6) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L48) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| 决策树 | 填补＋标准化 | 2.033633 | 2.595176 | 1.401940 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L7) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L7) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L49) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| 决策树 | 填补＋PCA | 2.087293 | 2.593023 | 1.517168 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L8) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L8) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L50) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| 决策树 | 填补＋标准化＋PCA | 2.107930 | 2.601778 | 1.576842 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L9) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L9) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L51) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| AdaBoost | 填补，无标准化/无PCA | 1.990995 | 2.455500 | 1.444966 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L10) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L10) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L52) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| AdaBoost | 填补＋标准化 | 1.990819 | 2.456175 | 1.446088 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L11) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L11) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L53) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| AdaBoost | 填补＋PCA | 2.016526 | 2.558242 | 1.508366 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L12) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L12) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L54) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| AdaBoost | 填补＋标准化＋PCA | 2.051130 | 2.576916 | 1.510745 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L13) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L13) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L55) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| 随机森林 | 填补，无标准化/无PCA | 1.851444 | 2.276036 | 1.334292 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L14) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L14) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L56) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| 随机森林 | 填补＋标准化 | 1.851883 | 2.275386 | 1.332931 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L15) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L15) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L57) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| 随机森林 | 填补＋PCA | 1.942392 | 2.478859 | 1.477213 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L16) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L16) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L58) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| 随机森林 | 填补＋标准化＋PCA | 1.898163 | 2.380177 | 1.446742 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L17) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L17) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L59) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| XGBoost原配置 | 填补，无标准化/无PCA | 1.806988 | 2.228808 | 1.252744 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L18) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L18) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L60) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| XGBoost原配置 | 填补＋标准化 | 1.806988 | 2.228808 | 1.252744 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L19) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L19) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L61) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| XGBoost原配置 | 填补＋PCA | 1.939866 | 2.461235 | 1.459217 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L20) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L20) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L62) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| XGBoost原配置 | 填补＋标准化＋PCA | 1.872072 | 2.337533 | 1.367040 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L21) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L21) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L63) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| XGBoost优化配置 | 填补，无标准化/无PCA | 1.781861 | 2.236749 | 1.242601 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L22) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L22) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L64) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| XGBoost优化配置 | 填补＋标准化 | 1.781861 | 2.236749 | 1.242601 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L23) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L23) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L65) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| XGBoost优化配置 | 填补＋PCA | 2.065189 | 2.606732 | 1.512826 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L24) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L24) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L66) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| XGBoost优化配置 | 填补＋标准化＋PCA | 1.841900 | 2.315206 | 1.305805 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L25) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L25) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L67) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| LightGBM | 填补，无标准化/无PCA | 1.796304 | 2.191367 | 1.263230 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L26) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L26) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L68) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| LightGBM | 填补＋标准化 | 1.805506 | 2.191424 | 1.253797 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L27) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L27) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L69) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| LightGBM | 填补＋PCA | 2.127522 | 2.667651 | 1.630577 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L28) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L28) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L70) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| LightGBM | 填补＋标准化＋PCA | 1.875117 | 2.365943 | 1.386715 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L29) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L29) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L71) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| LightGBM | 原生缺失，无标准化/无PCA | 1.808118 | 2.211554 | 1.268888 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L30) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L30) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L72) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| HistGradientBoosting | 填补，无标准化/无PCA | 1.785411 | 2.230219 | 1.257199 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L31) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L31) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L73) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| HistGradientBoosting | 填补＋标准化 | 1.784776 | 2.233692 | 1.257199 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L32) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L32) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L74) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| HistGradientBoosting | 填补＋PCA | 2.052189 | 2.584429 | 1.517058 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L33) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L33) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L75) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| HistGradientBoosting | 填补＋标准化＋PCA | 1.884365 | 2.285720 | 1.353061 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L34) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L34) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L76) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| HistGradientBoosting | 原生缺失，无标准化/无PCA | 1.790474 | 2.218320 | 1.265633 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L35) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L35) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L77) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| 原大网络（本轮重训） | 填补，无标准化/无PCA | 2.089127 | 2.619349 | 1.380860 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L36) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L36) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L78) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| 原大网络（本轮重训） | 填补＋标准化 | 1.929678 | 2.333565 | 1.299368 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L37) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L37) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L79) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| 原大网络（本轮重训） | 填补＋PCA | 1.983365 | 2.513425 | 1.394061 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L38) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L38) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L80) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| 原大网络（本轮重训） | 填补＋标准化＋PCA | 1.930533 | 2.395645 | 1.334516 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L39) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L39) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L81) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| 去乙肝小网络 | 填补，无标准化/无PCA | 2.379670 | 2.883043 | 1.451214 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L40) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L40) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L82) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| 去乙肝小网络 | 填补＋标准化 | 1.895841 | 2.260269 | 1.352561 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L41) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L41) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L83) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| 去乙肝小网络 | 填补＋PCA | 3.325018 | 3.608054 | 1.764104 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L42) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L42) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L84) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |
| 去乙肝小网络 | 填补＋标准化＋PCA | 1.941302 | 2.316041 | 1.369101 | [CV记录](review/preprocessing_ablation/cv_summary.csv#L43) · [开发记录](review/preprocessing_ablation/holdout_metrics.csv#L43) · [B记录](review/preprocessing_ablation/holdout_metrics.csv#L85) · [评价源码](review/preprocessing_ablation/run_ablation.py#L274) |

</details>

## 8. 四模型融合更换预处理后，是否进一步提升

四个成员和25%权重均不变；唯一达到单模型替换门槛的变化是LightGBM由原生缺失改为中位数填补＋缺失指示。下面两版必须在这一轮内部比较。

| 同轮融合版本 | 固定轮数CV均值 | 开发MSE | B MSE | 分数/源码 |
| --- | --- | --- | --- | --- |
| 原四模型参考处理 | 1.756036 | 2.177243 | 1.234296 | [得分记录](review/preprocessing_ablation/fusion_holdout.csv#L2) · [得分记录](review/preprocessing_ablation/fusion_cv.csv#L2) · [源码](review/preprocessing_ablation/run_ablation.py#L247) |
| LightGBM改填补后的实验版 | 1.752957 | 2.174456 | 1.234823 | [得分记录](review/preprocessing_ablation/fusion_holdout.csv#L3) · [得分记录](review/preprocessing_ablation/fusion_cv.csv#L4) · [源码](review/preprocessing_ablation/run_ablation.py#L247) |

原融合参考CV均值与第5节的1.759568不同，是因为本轮对小网络各折固定36轮，前轮则各折内部早停。开发/B约10⁻⁸的存档差异来自重算和预测导出精度；六位小数的主要结论不变。

| 数据集 | 新处理－参考MSE | 95%下界 | 95%上界 | 来源 |
| --- | --- | --- | --- | --- |
| development | -0.002787 | -0.011883 | 0.006530 | [得分记录](review/preprocessing_ablation/holdout_paired_intervals.csv#L12) |
| B | 0.000527 | -0.008715 | 0.009747 | [得分记录](review/preprocessing_ablation/holdout_paired_intervals.csv#L23) |

**结论：** 新处理开发略好、B略差，两个差值区间均跨零。不能把单模型CV小幅改善解释成融合一定改善；本轮没有证明四模型融合进一步稳定提升。两版模型独立保存，未覆盖原冻结融合，也没有依照B成绩再次改变权重。

## 9. 源码、结果与配置如何对应

文档中的“源码来源”指项目内实际运行代码及冻结参数，不是搜索来的同名算法示例。构造器加冻结配置共同定义一个模型，只有算法类名不够复现。

| 范围 | 说明 | 源码/输出 |
| --- | --- | --- |
| 原六模型训练与选型 | paper.ipynb；代码单元格索引45/47/53/56/59/63（从0计数） | [主notebook](paper.ipynb) · [原六参数](review/external_b_comparison/frozen_configurations.json) |
| 原六冻结复核 | 只重建已选配置，不重跑搜索 | [源码](review/external_b_comparison/evaluate_frozen.py#L24) · [分数](review/external_b_comparison/comparison.csv) |
| 特征/参数/模型 | 固定行变换、模型构造、100/12/8候选计划 | [源码](review/sequential_experiments/run_experiments.py#L46) · [源码](review/sequential_experiments/run_experiments.py#L84) · [随机候选参数](review/sequential_experiments/joint_search_plan.json) |
| 树/线性融合 | 40组组合、训练OOF加权 | [源码](review/sequential_experiments/run_experiments.py#L248) · [全部组合分数](review/sequential_experiments/blends.json) |
| 网络特征对照 | 训练OOF发现极端输入后补充去乙肝 | [源码](review/sequential_experiments/refine_nn_features.py#L13) · [事前补充计划](review/sequential_experiments/nn_refinement_plan.json) |
| 网络融合 | 9组权重、冻结后读取评价预测 | [源码](review/sequential_experiments/complete_nn_blends.py#L18) · [事前权重计划](review/sequential_experiments/nn_blend_plan.json) · [最终冻结决定](review/sequential_experiments/frozen_selection.json) |
| 预处理因子对照 | 同一参数/特征/轮数，仅改标准化/PCA | [源码](review/preprocessing_ablation/run_ablation.py#L154) · [协议](review/preprocessing_ablation/plan.json) · [冻结推荐](review/preprocessing_ablation/frozen_recommendations.json) |
| 预测复现 | 已保存最终四模型预测入口 | [源码](review/sequential_experiments/predict_selected.py#L16) |
| 本轮核验 | 指标、折、维数、输入摘要、模型重新加载 | [源码](review/preprocessing_ablation/verify_results.py#L18) · [核验记录](review/preprocessing_ablation/verification.json) |
| 展示notebook | 只读已完成结果，默认不重跑搜索 | [顺序实验](paper_experiments.ipynb) · [预处理对照](paper_preprocessing_experiments.ipynb) |

推荐复现顺序是先阅读对应`plan.json`与冻结参数，再使用Conda base运行源脚本。**本次只整理已有记录，没有为写文档重新训练。** 原脚本再次运行会重新拟合并更新各自实验目录，不能把“查看文档”与“重跑实验”混为一谈。

原大网络旧PReLU写法的`.keras`不适合直接`load_model`反序列化。已核验的做法是重建相同架构、用一行零输入建立PReLU变量后再加载权重，示例见[源码](review/preprocessing_ablation/verify_results.py#L67)。该流程用于推理，不声称恢复优化器状态逐步续训。

## 10. 附录：特征、随机候选与历史组合的完整记录

### 10.1 八组特征消融

固定原XGBoost参数，仅在训练CV23比较。各组详细定义见特征源码与阶段计划；组合特征胜出不代表每一子组都有独立收益。

| 配置ID | CV23 MSE | 分数/源码 |
| --- | --- | --- |
| baseline_xgb | 1.803531 | [得分记录](review/sequential_experiments/training_cv.csv#L2) · [源码](review/sequential_experiments/run_experiments.py#L46) |
| feature_all | 1.797830 | [得分记录](review/sequential_experiments/training_cv.csv#L3) · [源码](review/sequential_experiments/run_experiments.py#L46) |
| feature_missing | 1.801047 | [得分记录](review/sequential_experiments/training_cv.csv#L4) · [源码](review/sequential_experiments/run_experiments.py#L46) |
| feature_ratios | 1.797452 | [得分记录](review/sequential_experiments/training_cv.csv#L5) · [源码](review/sequential_experiments/run_experiments.py#L46) |
| feature_counts | 1.802958 | [得分记录](review/sequential_experiments/training_cv.csv#L6) · [源码](review/sequential_experiments/run_experiments.py#L46) |
| feature_interactions | 1.789187 | [得分记录](review/sequential_experiments/training_cv.csv#L7) · [源码](review/sequential_experiments/run_experiments.py#L46) |
| feature_combined | 1.783699 | [得分记录](review/sequential_experiments/training_cv.csv#L8) · [源码](review/sequential_experiments/run_experiments.py#L46) |
| feature_native_missing | 1.812445 | [得分记录](review/sequential_experiments/training_cv.csv#L9) · [源码](review/sequential_experiments/run_experiments.py#L46) |

### 10.2 六个小网络候选

这些训练CV使用各折内部早停；不与第7节固定36/87轮的CV混排。只有阶段胜出者进一步做过第二套折与开发/B评价。

| 候选ID | CV23 MSE | CV2026 MSE | 开发MSE | B MSE | 分数/源码/配置 |
| --- | --- | --- | --- | --- | --- |
| nn_64_raw | 2.770506 | — | — | — | [得分记录](review/sequential_experiments/training_cv.csv#L132) · [源码](review/sequential_experiments/run_experiments.py#L270) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2668) |
| nn_64_pca | 3.065721 | — | — | — | [得分记录](review/sequential_experiments/training_cv.csv#L133) · [源码](review/sequential_experiments/run_experiments.py#L270) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2680) |
| nn_128_raw | 4.156521 | — | — | — | [得分记录](review/sequential_experiments/training_cv.csv#L134) · [源码](review/sequential_experiments/run_experiments.py#L270) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2692) |
| nn_128_pca | 2.245105 | 3.068600 | 2.392069 | 1.388255 | [得分记录](review/sequential_experiments/training_cv.csv#L135) · [源码](review/sequential_experiments/run_experiments.py#L270) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2705) |
| nn_64_nohep_raw | 1.897055 | 1.926229 | 2.260269 | 1.352561 | [得分记录](review/sequential_experiments/training_cv.csv#L136) · [源码](review/sequential_experiments/refine_nn_features.py#L30) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2718) |
| nn_64_nohep_pca | 1.932535 | — | — | — | [得分记录](review/sequential_experiments/training_cv.csv#L137) · [源码](review/sequential_experiments/refine_nn_features.py#L30) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2730) |

### 10.3 120组联合随机搜索候选

100组XGBoost、12组LightGBM、8组HistGradientBoosting。以下是全部训练CV23分数，不把未作外部评价的候选补成B成绩。每行参数可点击冻结配置查看。

<details>
<summary>展开120组完整随机候选及源码</summary>

| 候选ID | 家族 | 特征 | 缺失处理 | CV23 MSE | 分数/源码/参数 |
| --- | --- | --- | --- | --- | --- |
| joint_xgb_089 | xgb | interactions | 填补 | 1.773159 | [得分记录](review/sequential_experiments/training_cv.csv#L99) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1691) |
| joint_xgb_055 | xgb | combined | 填补 | 1.779823 | [得分记录](review/sequential_experiments/training_cv.csv#L65) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1113) |
| joint_xgb_002 | xgb | interactions | 填补 | 1.783853 | [得分记录](review/sequential_experiments/training_cv.csv#L12) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L212) |
| joint_hist_04 | hist | combined | 原生缺失 | 1.784014 | [得分记录](review/sequential_experiments/training_cv.csv#L126) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2098) |
| joint_xgb_091 | xgb | combined | 填补 | 1.784642 | [得分记录](review/sequential_experiments/training_cv.csv#L101) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1725) |
| joint_xgb_058 | xgb | combined | 原生缺失 | 1.785610 | [得分记录](review/sequential_experiments/training_cv.csv#L68) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1164) |
| joint_hist_07 | hist | interactions | 原生缺失 | 1.789886 | [得分记录](review/sequential_experiments/training_cv.csv#L129) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2137) |
| joint_xgb_082 | xgb | interactions | 原生缺失 | 1.790216 | [得分记录](review/sequential_experiments/training_cv.csv#L92) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1572) |
| joint_hist_05 | hist | combined | 原生缺失 | 1.790586 | [得分记录](review/sequential_experiments/training_cv.csv#L127) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2111) |
| joint_xgb_059 | xgb | ratios | 填补 | 1.791412 | [得分记录](review/sequential_experiments/training_cv.csv#L69) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1181) |
| joint_xgb_007 | xgb | interactions | 原生缺失 | 1.794450 | [得分记录](review/sequential_experiments/training_cv.csv#L17) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L297) |
| joint_xgb_070 | xgb | interactions | 填补 | 1.796207 | [得分记录](review/sequential_experiments/training_cv.csv#L80) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1368) |
| joint_xgb_098 | xgb | interactions | 原生缺失 | 1.796386 | [得分记录](review/sequential_experiments/training_cv.csv#L108) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1844) |
| joint_xgb_071 | xgb | no_hepatitis | 填补 | 1.796920 | [得分记录](review/sequential_experiments/training_cv.csv#L81) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1385) |
| joint_xgb_096 | xgb | interactions | 原生缺失 | 1.797925 | [得分记录](review/sequential_experiments/training_cv.csv#L106) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1810) |
| joint_xgb_033 | xgb | interactions | 填补 | 1.799226 | [得分记录](review/sequential_experiments/training_cv.csv#L43) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L739) |
| joint_xgb_084 | xgb | interactions | 填补 | 1.800054 | [得分记录](review/sequential_experiments/training_cv.csv#L94) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1606) |
| joint_xgb_080 | xgb | no_hepatitis | 填补 | 1.800444 | [得分记录](review/sequential_experiments/training_cv.csv#L90) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1538) |
| joint_xgb_083 | xgb | interactions | 原生缺失 | 1.800454 | [得分记录](review/sequential_experiments/training_cv.csv#L93) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1589) |
| joint_xgb_049 | xgb | combined | 填补 | 1.801263 | [得分记录](review/sequential_experiments/training_cv.csv#L59) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1011) |
| joint_lgb_05 | lgb | interactions | 原生缺失 | 1.802199 | [得分记录](review/sequential_experiments/training_cv.csv#L115) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1948) |
| joint_hist_02 | hist | combined | 原生缺失 | 1.802324 | [得分记录](review/sequential_experiments/training_cv.csv#L124) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2072) |
| joint_xgb_017 | xgb | ratios | 原生缺失 | 1.802680 | [得分记录](review/sequential_experiments/training_cv.csv#L27) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L467) |
| joint_xgb_036 | xgb | ratios | 填补 | 1.804735 | [得分记录](review/sequential_experiments/training_cv.csv#L46) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L790) |
| joint_hist_06 | hist | combined | 原生缺失 | 1.806002 | [得分记录](review/sequential_experiments/training_cv.csv#L128) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2124) |
| joint_xgb_052 | xgb | combined | 原生缺失 | 1.806108 | [得分记录](review/sequential_experiments/training_cv.csv#L62) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1062) |
| joint_xgb_054 | xgb | combined | 原生缺失 | 1.806132 | [得分记录](review/sequential_experiments/training_cv.csv#L64) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1096) |
| joint_xgb_024 | xgb | interactions | 原生缺失 | 1.807587 | [得分记录](review/sequential_experiments/training_cv.csv#L34) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L586) |
| joint_xgb_068 | xgb | ratios | 原生缺失 | 1.807676 | [得分记录](review/sequential_experiments/training_cv.csv#L78) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1334) |
| joint_xgb_088 | xgb | interactions | 填补 | 1.807737 | [得分记录](review/sequential_experiments/training_cv.csv#L98) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1674) |
| joint_xgb_031 | xgb | interactions | 填补 | 1.807955 | [得分记录](review/sequential_experiments/training_cv.csv#L41) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L705) |
| joint_xgb_046 | xgb | combined | 填补 | 1.808552 | [得分记录](review/sequential_experiments/training_cv.csv#L56) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L960) |
| joint_xgb_069 | xgb | no_hepatitis | 原生缺失 | 1.809884 | [得分记录](review/sequential_experiments/training_cv.csv#L79) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1351) |
| joint_xgb_025 | xgb | interactions | 填补 | 1.810245 | [得分记录](review/sequential_experiments/training_cv.csv#L35) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L603) |
| joint_lgb_04 | lgb | no_hepatitis | 原生缺失 | 1.811138 | [得分记录](review/sequential_experiments/training_cv.csv#L114) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1934) |
| joint_xgb_064 | xgb | ratios | 原生缺失 | 1.812727 | [得分记录](review/sequential_experiments/training_cv.csv#L74) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1266) |
| joint_lgb_03 | lgb | ratios | 原生缺失 | 1.813562 | [得分记录](review/sequential_experiments/training_cv.csv#L113) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1920) |
| joint_xgb_009 | xgb | combined | 填补 | 1.815278 | [得分记录](review/sequential_experiments/training_cv.csv#L19) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L331) |
| joint_xgb_047 | xgb | no_hepatitis | 填补 | 1.815667 | [得分记录](review/sequential_experiments/training_cv.csv#L57) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L977) |
| joint_xgb_016 | xgb | ratios | 原生缺失 | 1.816536 | [得分记录](review/sequential_experiments/training_cv.csv#L26) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L450) |
| joint_xgb_097 | xgb | combined | 原生缺失 | 1.818494 | [得分记录](review/sequential_experiments/training_cv.csv#L107) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1827) |
| joint_xgb_034 | xgb | interactions | 填补 | 1.819210 | [得分记录](review/sequential_experiments/training_cv.csv#L44) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L756) |
| joint_xgb_027 | xgb | ratios | 填补 | 1.819234 | [得分记录](review/sequential_experiments/training_cv.csv#L37) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L637) |
| joint_xgb_087 | xgb | combined | 填补 | 1.820326 | [得分记录](review/sequential_experiments/training_cv.csv#L97) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1657) |
| joint_xgb_092 | xgb | combined | 填补 | 1.820599 | [得分记录](review/sequential_experiments/training_cv.csv#L102) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1742) |
| joint_xgb_045 | xgb | ratios | 原生缺失 | 1.822361 | [得分记录](review/sequential_experiments/training_cv.csv#L55) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L943) |
| joint_hist_01 | hist | combined | 原生缺失 | 1.823049 | [得分记录](review/sequential_experiments/training_cv.csv#L123) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2059) |
| joint_lgb_01 | lgb | ratios | 原生缺失 | 1.823161 | [得分记录](review/sequential_experiments/training_cv.csv#L111) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1892) |
| joint_xgb_090 | xgb | no_hepatitis | 原生缺失 | 1.824065 | [得分记录](review/sequential_experiments/training_cv.csv#L100) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1708) |
| joint_xgb_010 | xgb | ratios | 原生缺失 | 1.824707 | [得分记录](review/sequential_experiments/training_cv.csv#L20) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L348) |
| joint_xgb_006 | xgb | combined | 原生缺失 | 1.825156 | [得分记录](review/sequential_experiments/training_cv.csv#L16) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L280) |
| joint_xgb_008 | xgb | combined | 填补 | 1.826118 | [得分记录](review/sequential_experiments/training_cv.csv#L18) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L314) |
| joint_xgb_051 | xgb | no_hepatitis | 原生缺失 | 1.826520 | [得分记录](review/sequential_experiments/training_cv.csv#L61) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1045) |
| joint_xgb_095 | xgb | ratios | 填补 | 1.827235 | [得分记录](review/sequential_experiments/training_cv.csv#L105) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1793) |
| joint_xgb_029 | xgb | no_hepatitis | 填补 | 1.827334 | [得分记录](review/sequential_experiments/training_cv.csv#L39) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L671) |
| joint_xgb_072 | xgb | no_hepatitis | 填补 | 1.827508 | [得分记录](review/sequential_experiments/training_cv.csv#L82) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1402) |
| joint_xgb_073 | xgb | combined | 原生缺失 | 1.828585 | [得分记录](review/sequential_experiments/training_cv.csv#L83) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1419) |
| joint_lgb_00 | lgb | interactions | 原生缺失 | 1.830052 | [得分记录](review/sequential_experiments/training_cv.csv#L110) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1878) |
| joint_xgb_039 | xgb | interactions | 原生缺失 | 1.830754 | [得分记录](review/sequential_experiments/training_cv.csv#L49) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L841) |
| joint_xgb_011 | xgb | combined | 填补 | 1.831209 | [得分记录](review/sequential_experiments/training_cv.csv#L21) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L365) |
| joint_lgb_06 | lgb | interactions | 原生缺失 | 1.831329 | [得分记录](review/sequential_experiments/training_cv.csv#L116) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1962) |
| joint_xgb_000 | xgb | ratios | 填补 | 1.831463 | [得分记录](review/sequential_experiments/training_cv.csv#L10) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L178) |
| joint_xgb_099 | xgb | no_hepatitis | 原生缺失 | 1.831592 | [得分记录](review/sequential_experiments/training_cv.csv#L109) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1861) |
| joint_xgb_062 | xgb | combined | 填补 | 1.831698 | [得分记录](review/sequential_experiments/training_cv.csv#L72) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1232) |
| joint_xgb_079 | xgb | interactions | 原生缺失 | 1.831793 | [得分记录](review/sequential_experiments/training_cv.csv#L89) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1521) |
| joint_xgb_026 | xgb | no_hepatitis | 原生缺失 | 1.832829 | [得分记录](review/sequential_experiments/training_cv.csv#L36) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L620) |
| joint_xgb_081 | xgb | ratios | 原生缺失 | 1.835210 | [得分记录](review/sequential_experiments/training_cv.csv#L91) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1555) |
| joint_xgb_042 | xgb | no_hepatitis | 原生缺失 | 1.837262 | [得分记录](review/sequential_experiments/training_cv.csv#L52) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L892) |
| joint_xgb_053 | xgb | no_hepatitis | 填补 | 1.840553 | [得分记录](review/sequential_experiments/training_cv.csv#L63) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1079) |
| joint_xgb_004 | xgb | no_hepatitis | 原生缺失 | 1.841390 | [得分记录](review/sequential_experiments/training_cv.csv#L14) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L246) |
| joint_xgb_012 | xgb | no_hepatitis | 原生缺失 | 1.841530 | [得分记录](review/sequential_experiments/training_cv.csv#L22) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L382) |
| joint_xgb_037 | xgb | ratios | 填补 | 1.843597 | [得分记录](review/sequential_experiments/training_cv.csv#L47) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L807) |
| joint_xgb_074 | xgb | interactions | 原生缺失 | 1.844177 | [得分记录](review/sequential_experiments/training_cv.csv#L84) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1436) |
| joint_hist_03 | hist | no_hepatitis | 原生缺失 | 1.844867 | [得分记录](review/sequential_experiments/training_cv.csv#L125) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2085) |
| joint_xgb_005 | xgb | ratios | 填补 | 1.845161 | [得分记录](review/sequential_experiments/training_cv.csv#L15) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L263) |
| joint_xgb_023 | xgb | no_hepatitis | 填补 | 1.846565 | [得分记录](review/sequential_experiments/training_cv.csv#L33) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L569) |
| joint_xgb_085 | xgb | ratios | 填补 | 1.847380 | [得分记录](review/sequential_experiments/training_cv.csv#L95) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1623) |
| joint_lgb_10 | lgb | ratios | 原生缺失 | 1.848210 | [得分记录](review/sequential_experiments/training_cv.csv#L120) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2018) |
| joint_xgb_075 | xgb | combined | 填补 | 1.848255 | [得分记录](review/sequential_experiments/training_cv.csv#L85) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1453) |
| joint_xgb_038 | xgb | interactions | 填补 | 1.848677 | [得分记录](review/sequential_experiments/training_cv.csv#L48) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L824) |
| joint_xgb_077 | xgb | combined | 原生缺失 | 1.849793 | [得分记录](review/sequential_experiments/training_cv.csv#L87) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1487) |
| joint_xgb_043 | xgb | interactions | 原生缺失 | 1.850026 | [得分记录](review/sequential_experiments/training_cv.csv#L53) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L909) |
| joint_xgb_001 | xgb | no_hepatitis | 原生缺失 | 1.850484 | [得分记录](review/sequential_experiments/training_cv.csv#L11) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L195) |
| joint_xgb_020 | xgb | no_hepatitis | 原生缺失 | 1.850633 | [得分记录](review/sequential_experiments/training_cv.csv#L30) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L518) |
| joint_xgb_094 | xgb | no_hepatitis | 原生缺失 | 1.851721 | [得分记录](review/sequential_experiments/training_cv.csv#L104) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1776) |
| joint_xgb_078 | xgb | no_hepatitis | 填补 | 1.855437 | [得分记录](review/sequential_experiments/training_cv.csv#L88) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1504) |
| joint_xgb_030 | xgb | interactions | 原生缺失 | 1.855596 | [得分记录](review/sequential_experiments/training_cv.csv#L40) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L688) |
| joint_xgb_067 | xgb | no_hepatitis | 填补 | 1.856407 | [得分记录](review/sequential_experiments/training_cv.csv#L77) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1317) |
| joint_xgb_050 | xgb | no_hepatitis | 填补 | 1.857360 | [得分记录](review/sequential_experiments/training_cv.csv#L60) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1028) |
| joint_xgb_021 | xgb | ratios | 填补 | 1.862069 | [得分记录](review/sequential_experiments/training_cv.csv#L31) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L535) |
| joint_lgb_09 | lgb | interactions | 原生缺失 | 1.865037 | [得分记录](review/sequential_experiments/training_cv.csv#L119) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2004) |
| joint_lgb_08 | lgb | combined | 原生缺失 | 1.869734 | [得分记录](review/sequential_experiments/training_cv.csv#L118) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1990) |
| joint_hist_00 | hist | combined | 原生缺失 | 1.870530 | [得分记录](review/sequential_experiments/training_cv.csv#L122) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2046) |
| joint_xgb_015 | xgb | interactions | 原生缺失 | 1.871710 | [得分记录](review/sequential_experiments/training_cv.csv#L25) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L433) |
| joint_lgb_11 | lgb | interactions | 原生缺失 | 1.873306 | [得分记录](review/sequential_experiments/training_cv.csv#L121) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L2032) |
| joint_lgb_02 | lgb | ratios | 原生缺失 | 1.873414 | [得分记录](review/sequential_experiments/training_cv.csv#L112) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1906) |
| joint_xgb_061 | xgb | interactions | 填补 | 1.873551 | [得分记录](review/sequential_experiments/training_cv.csv#L71) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1215) |
| joint_xgb_003 | xgb | ratios | 原生缺失 | 1.873633 | [得分记录](review/sequential_experiments/training_cv.csv#L13) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L229) |
| joint_xgb_032 | xgb | no_hepatitis | 原生缺失 | 1.875590 | [得分记录](review/sequential_experiments/training_cv.csv#L42) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L722) |
| joint_xgb_048 | xgb | ratios | 填补 | 1.877433 | [得分记录](review/sequential_experiments/training_cv.csv#L58) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L994) |
| joint_xgb_066 | xgb | ratios | 填补 | 1.877710 | [得分记录](review/sequential_experiments/training_cv.csv#L76) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1300) |
| joint_xgb_057 | xgb | interactions | 填补 | 1.885212 | [得分记录](review/sequential_experiments/training_cv.csv#L67) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1147) |
| joint_xgb_086 | xgb | no_hepatitis | 原生缺失 | 1.887386 | [得分记录](review/sequential_experiments/training_cv.csv#L96) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1640) |
| joint_xgb_056 | xgb | ratios | 原生缺失 | 1.888905 | [得分记录](review/sequential_experiments/training_cv.csv#L66) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1130) |
| joint_xgb_018 | xgb | ratios | 填补 | 1.888985 | [得分记录](review/sequential_experiments/training_cv.csv#L28) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L484) |
| joint_xgb_014 | xgb | ratios | 填补 | 1.890417 | [得分记录](review/sequential_experiments/training_cv.csv#L24) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L416) |
| joint_xgb_028 | xgb | no_hepatitis | 原生缺失 | 1.894060 | [得分记录](review/sequential_experiments/training_cv.csv#L38) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L654) |
| joint_xgb_093 | xgb | ratios | 填补 | 1.894676 | [得分记录](review/sequential_experiments/training_cv.csv#L103) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1759) |
| joint_xgb_076 | xgb | combined | 原生缺失 | 1.906279 | [得分记录](review/sequential_experiments/training_cv.csv#L86) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1470) |
| joint_xgb_019 | xgb | combined | 原生缺失 | 1.908131 | [得分记录](review/sequential_experiments/training_cv.csv#L29) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L501) |
| joint_xgb_041 | xgb | interactions | 原生缺失 | 1.909032 | [得分记录](review/sequential_experiments/training_cv.csv#L51) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L875) |
| joint_xgb_044 | xgb | ratios | 填补 | 1.918702 | [得分记录](review/sequential_experiments/training_cv.csv#L54) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L926) |
| joint_lgb_07 | lgb | ratios | 原生缺失 | 1.919367 | [得分记录](review/sequential_experiments/training_cv.csv#L117) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1976) |
| joint_xgb_065 | xgb | no_hepatitis | 原生缺失 | 1.931553 | [得分记录](review/sequential_experiments/training_cv.csv#L75) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1283) |
| joint_xgb_013 | xgb | no_hepatitis | 原生缺失 | 1.935726 | [得分记录](review/sequential_experiments/training_cv.csv#L23) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L399) |
| joint_xgb_063 | xgb | interactions | 原生缺失 | 1.962643 | [得分记录](review/sequential_experiments/training_cv.csv#L73) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1249) |
| joint_xgb_060 | xgb | combined | 填补 | 1.980730 | [得分记录](review/sequential_experiments/training_cv.csv#L70) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L1198) |
| joint_xgb_022 | xgb | ratios | 原生缺失 | 1.980841 | [得分记录](review/sequential_experiments/training_cv.csv#L32) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L552) |
| joint_xgb_040 | xgb | ratios | 原生缺失 | 1.992690 | [得分记录](review/sequential_experiments/training_cv.csv#L50) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L858) |
| joint_xgb_035 | xgb | no_hepatitis | 填补 | 2.000367 | [得分记录](review/sequential_experiments/training_cv.csv#L45) · [搜索源码](review/sequential_experiments/run_experiments.py#L216) · [冻结配置](review/sequential_experiments/frozen_selection.json#L773) |

</details>

### 10.4 历史融合：保留记录，但不与当前主表混排

| 协议/评价集 | 模型 | MSE | 限制 | 分数/源码 |
| --- | --- | --- | --- | --- |
| H1，1561条 | 原线性＋PCA（历史复现） | 1.714056 | 旧预处理有全数据拟合历史，作旧方案参考 | [分数](review/optimization/same_validation/results.json) · [源码](paper_optimized.py#L151) |
| H1，1561条 | 原XGBoost40轮＋PCA | 1.659235 | 同样本旧方案 | [分数](review/optimization/same_validation/results.json) · [源码](paper_optimized.py#L151) |
| H1，1561条 | Ridge同协议参考 | 1.737611 | 本协议内比较 | [分数](review/optimization/same_validation/results.json) · [源码](paper_optimized.py#L66) |
| H1，1561条 | LightGBM＋HistGB，各50% | 1.579267 | 训练仅3641条，不与当前1993验证排名 | [分数](review/optimization/same_validation/results.json) · [源码](paper_optimized.py#L103) |
| H2，A1000条 | HistGB＋LightGBM，各50% | 1.730496 | 仅原train5642拟合，A未参加训练 | [分数](review/optimization/external/results.json) · [源码](paper_optimized.py#L170) |
| H2，B1000条 | HistGB＋LightGBM，各50% | 1.274190 | 与当前6642条拟合的四融合不构成同训练样本比较 | [分数](review/optimization/external/results.json) · [源码](paper_optimized.py#L170) |
| H2，日期留出1294条 | HistGB＋LightGBM，各50% | 1.344691 | 结构曾在完整train选过，只是时间压力检查，非严格独立时间测试 | [分数](review/optimization/external/results.json) · [源码](paper_optimized.py#L170) |

H1的LightGBM为200棵/lr=.03/15叶/min_child_samples40/λ20；HistGB为350轮/lr=.03/7叶/min_samples_leaf40/L2=30。H2的LightGBM改为350棵/7叶/min_child_samples60/λ30，HistGB同前。两种历史融合不是同一参数版本。原文档旧神经网络1.65340仅属当时报告保存输出，本次未从该旧模型的逐行预测重算，故不作为可复现主表依据。

## 11. 结论应如何用于论文或汇报

可以写：

> 在固定train+A开发协议下，训练OOF选出的XGBoost、LightGBM、Ridge与小网络等权融合，开发/B MSE分别约2.177243/1.234296，相对原XGBoost分别下降约2.31%/1.47%。预处理对照支持神经网络使用标准化，未发现增加PCA的稳定收益；树模型无需额外标准化。外部B的配对差异区间仍跨零，且开发/B已被历史使用，因此这些结果是已有数据上的复核，不能视作未来泛化优势的最终证明。

不应写：

- “四模型融合在所有指标和所有人群均最好。”
- “PCA保留98%方差，所以保留98%血糖预测信息。”
- “保留旧大网络PCA，证明它必须使用PCA。”
- “后来B最低的候选就是最终模型”，而隐去其选型来自训练CV的规则。
- 把1561、1993、完整A/B以及不同拟合样本数的MSE混排计算提升。

这次汇总的最终判断是：**四模型等权是当前有训练选型依据且开发/B点估计均改善的主要方案；进一步改动预处理没有证明融合再次稳定改善。保留冻结配置、透明报告口径和不确定性，比持续围绕已知B分数选择更重要。**

## 12. 代码分类与可运行目录结构

整理日期：2026-10-22。

### 12.1 这次整理的边界

新增`glucose_project/`作为按功能分类的**可运行入口层**，统一日常调用和帮助说明；实际算法实现仍来自原来已用于产生成绩的历史源码。没有把同一套训练逻辑复制出第二份，也没有声称旧脚本内部已完成函数级拆分。

历史源码保留原位，是因为冻结配置、OOF缓存、补充实验计划和本文源码链接记录了其路径或SHA256。每个新入口使用当前Python解释器，在项目根目录调用原脚本；原脚本的`__file__`、相对输出位置和源码摘要都保持原含义。

这一区分在下面逐项标明：**分类入口负责运行，历史实现负责算法，review目录保存已有产物。** 新入口本身不改变特征、参数、模型权重或MSE结论。

### 12.2 实际目录树

```text
data_aliyun/
├── paper.ipynb                         # 主notebook与现有结果
├── paper_experiments.ipynb             # 顺序实验结果展示
├── paper_preprocessing_experiments.ipynb # 标准化/PCA结果展示
├── paper_optimized.ipynb               # 早期协议展示
├── VALIDATION_SUMMARY.md               # 验证、MSE、源码及本目录说明
├── glucose_project/                   # 新增：日常运行入口，按功能分类
│   ├── __init__.py                    # 导入时不执行训练
│   ├── __main__.py                    # python -m glucose_project
│   ├── catalog.py                     # 入口→实际源码→输出位置
│   ├── runner.py                      # 同一解释器、原路径、隔离进程调度
│   ├── source_catalog.json            # 整理时13份历史源码的SHA256
│   ├── organization_check.json        # 入口及1000条真实预测核验记录
│   ├── README.md                      # 代码导航与快速使用
│   ├── environment/
│   │   └── check.py                   # base导入与小样本检查
│   ├── experiments/
│   │   ├── external_baselines.py      # 原六模型冻结复核
│   │   └── sequential.py              # 特征→搜索→树融合→小网络
│   ├── ensembles/
│   │   ├── refine_neural.py           # 去乙肝网络补充对照
│   │   └── blend_neural.py            # 9组网络融合
│   ├── preprocessing/
│   │   └── ablation.py                # 42组标准化/PCA配对对照
│   ├── inference/
│   │   └── predict.py                 # 冻结四模型推理
│   ├── validation/
│   │   └── preprocessing.py           # 指标/维数/模型重载核验
│   ├── reports/
│   │   ├── sequential.py              # 顺序实验报告生成
│   │   ├── preprocessing.py           # 预处理报告生成
│   │   └── historical.py              # 早期协议事后分析
│   └── legacy/
│       ├── optimization.py            # 早期1561/A/B协议实验
│       └── baseline.py                # 早期Ridge独立基线
├── tests/
│   └── test_workflow_entrypoints.py   # 入口、参数、路径与源码完整性检查
├── data/                              # 既有数据，当前流程读取原始CSV
├── review/                            # 历史执行源码＋已冻结配置与实验输出
│   ├── external_b_comparison/         # 原六模型源码、参数和B对照
│   ├── sequential_experiments/        # 顺序实验、融合、逐条预测及模型
│   ├── preprocessing_ablation/        # 预处理对照、核验、模型和报告
│   ├── optimization/                  # 早期协议结果与分析
│   ├── check_environment.py           # 历史环境检查实现
│   ├── baseline_check.py              # 历史基线实现
│   └── base_audit/report.md           # 全环境审查保留报告
└── paper_optimized.py                 # 早期协议的历史实现
```

各分类目录还包含`__init__.py`，上图省略这些空的包标记和Python缓存。根目录的主notebook保留原位置，便于继续工作。

### 12.3 13个入口与实际源码一一对应

所有分类入口均支持`--help`和`--dry-run`。后者只显示解释器、源码、工作目录和输出位置，不执行实验。

| 短名称 | 功能 | 分类入口 | 实际算法/脚本源码 | 既有输出位置 |
| --- | --- | --- | --- | --- |
| environment | 检查base导入与小样本训练 | [environment.check](glucose_project/environment/check.py#L1) | [review/check_environment.py](review/check_environment.py#L1) | review/environment_check.json |
| original-algorithms | 恢复原六模型、重拟合并评估B | [experiments.external_baselines](glucose_project/experiments/external_baselines.py#L1) | [review/external_b_comparison/evaluate_frozen.py](review/external_b_comparison/evaluate_frozen.py#L1) | review/external_b_comparison/；paper.ipynb |
| sequential | 按顺序执行特征消融、联合搜索、树融合与小网络实验 | [experiments.sequential](glucose_project/experiments/sequential.py#L1) | [review/sequential_experiments/run_experiments.py](review/sequential_experiments/run_experiments.py#L1) | review/sequential_experiments/ |
| neural-features | 训练OOF驱动的去乙肝小网络补充对照 | [ensembles.refine_neural](glucose_project/ensembles/refine_neural.py#L1) | [review/sequential_experiments/refine_nn_features.py](review/sequential_experiments/refine_nn_features.py#L1) | review/sequential_experiments/ |
| neural-blend | 按既定权重网格补充小网络融合 | [ensembles.blend_neural](glucose_project/ensembles/blend_neural.py#L1) | [review/sequential_experiments/complete_nn_blends.py](review/sequential_experiments/complete_nn_blends.py#L1) | review/sequential_experiments/ |
| preprocessing | 十模型标准化/PCA四组配对实验 | [preprocessing.ablation](glucose_project/preprocessing/ablation.py#L1) | [review/preprocessing_ablation/run_ablation.py](review/preprocessing_ablation/run_ablation.py#L1) | review/preprocessing_ablation/ |
| predict | 加载已冻结四模型，只预测不训练 | [inference.predict](glucose_project/inference/predict.py#L1) | [review/sequential_experiments/predict_selected.py](review/sequential_experiments/predict_selected.py#L1) | --output指定CSV |
| verify-preprocessing | 重算指标并重新加载建议模型核验 | [validation.preprocessing](glucose_project/validation/preprocessing.py#L1) | [review/preprocessing_ablation/verify_results.py](review/preprocessing_ablation/verify_results.py#L1) | review/preprocessing_ablation/verification.json |
| report-sequential | 读取顺序实验结果并生成报告/notebook | [reports.sequential](glucose_project/reports/sequential.py#L1) | [review/sequential_experiments/render_results.py](review/sequential_experiments/render_results.py#L1) | review/sequential_experiments/report.md；paper_experiments.ipynb |
| report-preprocessing | 读取预处理结果并生成报告/notebook | [reports.preprocessing](glucose_project/reports/preprocessing.py#L1) | [review/preprocessing_ablation/render_results.py](review/preprocessing_ablation/render_results.py#L1) | review/preprocessing_ablation/report.md；paper_preprocessing_experiments.ipynb |
| report-historical | 早期协议的事后误差分析；会重拟合部分参考模型 | [reports.historical](glucose_project/reports/historical.py#L1) | [review/optimization/analyze_results.py](review/optimization/analyze_results.py#L1) | review/optimization/ |
| legacy-optimization | 早期1561条验证或仅原train的A/B协议 | [legacy.optimization](glucose_project/legacy/optimization.py#L1) | [paper_optimized.py](paper_optimized.py#L1) | review/optimization/ |
| legacy-baseline | 早期独立Ridge基线检查，不属于当前融合选型 | [legacy.baseline](glucose_project/legacy/baseline.py#L1) | [review/baseline_check.py](review/baseline_check.py#L1) | review/baseline_results.json |

`report-historical`属于早期事后分析，会重拟合部分参考模型；另外两个report入口主要读取结果生成报告。报告生成仍会更新对应的Markdown和展示notebook；只查看已有成绩时直接打开文档即可。

### 12.4 按代码职责寻找函数

下面补充到函数级定位。它是原实现的职责导航，避免把“找模型构造器”和“重跑整套实验”混在一起。

| 职责 | 内容 | 实际实现位置 |
| --- | --- | --- |
| 数据加载与固定拆分 | 加载train+A，固定4649/1993，B在冻结后读取 | [main](review/sequential_experiments/run_experiments.py#L181) |
| 特征工程 | 性别映射、未知标记、删乙肝、比值、计数、年龄交互 | [features](review/sequential_experiments/run_experiments.py#L46) |
| 训练预处理 | 逐训练折填补、缺失指示、按家族缩放和PCA | [preprocess](review/sequential_experiments/run_experiments.py#L75) |
| 四格预处理对照 | 只改变标准化/PCA；原生缺失参考独立保留 | [preprocessing](review/preprocessing_ablation/run_ablation.py#L80) |
| 传统模型构造 | Ridge、XGBoost、LightGBM、HistGB、随机森林 | [estimator](review/sequential_experiments/run_experiments.py#L84) |
| 原六算法恢复 | 读取原notebook构造器并应用冻结参数 | [recover_state](review/external_b_comparison/evaluate_frozen.py#L24) |
| 小网络构造与训练 | 网络层、优化器、内部早停、完整训练折重拟合 | [build_nn](review/sequential_experiments/run_experiments.py#L103) · [fit_nn_predict](review/sequential_experiments/run_experiments.py#L118) |
| 固定网络对照 | 原87轮/小36轮，控制初始化与表示 | [neural](review/preprocessing_ablation/run_ablation.py#L102) |
| 训练OOF | 同一批记录的训练折外预测与缓存 | [cv](review/sequential_experiments/run_experiments.py#L150) |
| 参数与特征联合搜索 | 100/12/8组预先采样候选 | [搜索空间](review/sequential_experiments/run_experiments.py#L216) |
| 树/线性融合 | 40个两成员/三成员候选 | [blend_rows](review/sequential_experiments/run_experiments.py#L248) |
| 网络融合 | 9组N与X0/X/T组合，先冻结再读取评价预测 | [权重网格](review/sequential_experiments/complete_nn_blends.py#L30) |
| 评价指标 | MSE、RMSE、MAE、R²统一计算 | [score](review/sequential_experiments/run_experiments.py#L144) |
| 冻结模型推理 | 递归加载成员，使用冻结特征与权重求预测 | [predict](review/sequential_experiments/predict_selected.py#L16) |
| 模型重载核验 | CSV重算、维数、原输入摘要和网络权重重建 | [核验main](review/preprocessing_ablation/verify_results.py#L18) |

### 12.5 常用运行方式

以下命令均在项目根目录运行。使用当前Conda base解释器，无需安装本项目为新Python包：

```bash
cd .
python -m glucose_project list
```

查看帮助或只检查路径：

```bash
python -m glucose_project run sequential --help
python -m glucose_project.experiments.sequential --dry-run
```

两种入口形式等价：`python -m glucose_project run 短名称`与`python -m glucose_project.分类.模块`。

**只加载已有四模型预测：**

```bash
python -m glucose_project.inference.predict \
  --input data/d_test_B_20180128.csv \
  --output /private/tmp/new_glucose_predictions.csv \
  --scope full
```

`full`加载使用全部6642条开发数据拟合的版本；`development`加载使用4649条训练记录拟合的版本。该预测入口仅接收特征，不读取答案、不训练；当前默认仍是原冻结的四模型各25%版本。新传入的相对路径先按调用者目录转换成绝对路径，含空格路径可正常传递。

**需要完整复现顺序实验时，顺序为：**

```bash
python -m glucose_project run sequential
python -m glucose_project run neural-features
python -m glucose_project run neural-blend
python -m glucose_project run report-sequential
```

这些命令会按旧实现更新`review/sequential_experiments/`中的配置、模型、预测和报告。若改变输入或参数，应先为旧结果、补充流程的两份`frozen_selection_before_*.json`及plan文件保留独立实验版本，避免旧before快照参与新实验；源码变更仍受原来的冻结摘要校验约束。

**预处理对照复现顺序：**

```bash
python -m glucose_project run preprocessing
python -m glucose_project run verify-preprocessing
python -m glucose_project run report-preprocessing
```

预处理实验读取顺序实验的冻结配置，故应在所需的顺序实验版本确定后运行。生成展示notebook后，可在base内核运行其只读展示单元格保存输出。

**早期协议单独运行：**

```bash
python -m glucose_project.legacy.optimization --protocol same
python -m glucose_project.legacy.optimization --protocol external
```

`same`对应旧1561条验证协议，`external`对应仅原train训练的A/B协议。它们不替代当前4649/1993/6642协议；历史分数仍按第2节分开解释。

### 12.6 数据、代码、参数、结果各放在哪里

- `glucose_project/`：可运行分类入口和调度，不存训练数据或模型权重。
- `data/`：既有输入文件；当前主流程直接读取原始CSV，不把旧PCA/填补中间表当成新模型输入。
- `review/*/plan.json`与冻结JSON：每轮实验协议、实际参数、选定处理与组合权重。
- `review/*/*.csv`：训练OOF、开发/B指标和逐条预测；未评估项仍以“—”说明。
- `review/*/*.joblib`、`*.keras`或`models/`：已拟合模型和预处理器，名称中development/B指该次评价对应的拟合范围。
- `review/*/report.md`：单轮实验说明；`VALIDATION_SUMMARY.md`：跨轮的口径对齐与总判断。
- 根目录notebook：主研究过程与结果展示。旧脚本保留为对应实验的执行基准，原源码链接继续有效。

### 12.7 本次目录整理的核验

6项入口测试全部通过：帮助/预览、导入无训练副作用、调用者相对路径与空格路径、缺少必需参数时不执行、子进程错误码传递、13份历史源码SHA256不变。另通过新分类预测入口对B的1000条原始特征执行真实推理，与已有四模型逐条预测的最大绝对差为约1.14×10⁻⁷，符合数值误差容差；本次没有重新搜索或训练模型，也未更改已有分数表。

入口检查命令：

```bash
python -m unittest discover -s tests -v
```

[入口核验代码](tests/test_workflow_entrypoints.py#L1)；[13份历史源码整理快照](glucose_project/source_catalog.json)；[真实推理核验记录](glucose_project/organization_check.json)。
