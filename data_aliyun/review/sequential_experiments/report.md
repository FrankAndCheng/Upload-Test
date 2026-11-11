# 按顺序优化实验：实际结果

## 最终结果

训练集内部两套五折选出的方案为 **nn_blend_06**，开发验证 MSE 为 **2.177243**，B MSE 为 **1.234296**。以下全部为实际运行结果；没有根据开发/B的成绩重新选模型、调整权重或回调参数。

最终组成：**25% 去乙肝小网络 + 25% XGBoost + 25% Ridge + 25% LightGBM**。

相对当前XGBoost，开发MSE下降2.31%，B MSE下降1.47%。两个集合的MSE点估计均改善。 点估计的改善仍需结合下方配对区间与已知验证集的局限解释。

| 阶段 | 模型 | 训练CV均值 | 开发MSE | 开发MSE下降百分比 | B_MSE | B_MSE下降百分比 | B_MAE | B_R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 当前XGBoost基线 | baseline_xgb | 1.806988 | 2.228808 | 0.000000 | 1.252744 | 0.000000 | 0.648007 | 0.184240 |
| 特征消融优胜 | feature_combined | 1.786509 | 2.218877 | 0.445586 | 1.275351 | -1.804630 | 0.654285 | 0.169518 |
| 联合搜索优胜 | joint_xgb_089 | 1.781861 | 2.236749 | -0.356302 | 1.242601 | 0.809627 | 0.646290 | 0.190845 |
| 简单融合优胜 | blend_33 | 1.762972 | 2.196540 | 1.447769 | 1.245558 | 0.573555 | 0.652927 | 0.188919 |
| 小型神经网络优胜 | nn_64_nohep_raw | 1.911642 | 2.260269 | -1.411581 | 1.352561 | -7.967894 | 0.668018 | 0.119241 |
| 补充：小网络融合 | nn_blend_06 | 1.759568 | 2.177243 | 2.313554 | 1.234296 | 1.472594 | 0.638160 | 0.196253 |
| 训练CV最终选择 | nn_blend_06 | 1.759568 | 2.177243 | 2.313554 | 1.234296 | 1.472594 | 0.638160 | 0.196253 |

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

| name | MSE | MAE | CV_MSE下降百分比 |
| --- | --- | --- | --- |
| baseline_xgb | 1.803531 | 0.726559 | 0.000000 |
| feature_all | 1.797830 | 0.724452 | 0.316096 |
| feature_missing | 1.801047 | 0.725372 | 0.137732 |
| feature_ratios | 1.797452 | 0.724395 | 0.337041 |
| feature_counts | 1.802958 | 0.725548 | 0.031789 |
| feature_interactions | 1.789187 | 0.723245 | 0.795335 |
| feature_combined | 1.783699 | 0.722849 | 1.099621 |
| feature_native_missing | 1.812445 | 0.728703 | -0.494234 |

第一阶段的结果是固定模型参数下的结果，不代表同一特征组在所有模型、参数下都有效。比值仅对正且非零分母计算；其余填NaN。所有派生量仅依赖同一行输入，不使用目标、id、日期。

特征组合在训练CV和开发集有小幅收益，但B变差，不能宣称特征越多越好。联合搜索的单模型在B改善而开发稍差；最终融合才在两边同时取得较低MSE。

## 第二套训练折复核

| name | CV_seed23_MSE | CV_seed2026_MSE | mean_CV_MSE |
| --- | --- | --- | --- |
| baseline_xgb | 1.803531 | 1.810445 | 1.806988 |
| feature_combined | 1.783699 | 1.789320 | 1.786509 |
| joint_xgb_089 | 1.773159 | 1.790562 | 1.781861 |
| blend_33 | 1.758488 | 1.767456 | 1.762972 |
| nn_128_pca | 2.245105 | 3.068600 | 2.656852 |
| nn_64_nohep_raw | 1.897055 | 1.926229 | 1.911642 |
| nn_blend_06 | 1.754295 | 1.764841 | 1.759568 |

## 冻结的阶段优胜方案

- **当前XGBoost基线**：`baseline_xgb`；{"name": "baseline_xgb", "family": "xgb", "variant": "no_hepatitis", "native": false, "params": {"n_estimators": 350, "learning_rate": 0.02, "max_depth": 4, "min_child_weight": 30, "reg_lambda": 10, "subsample": 0.8, "colsample_bytree": 0.8}}。
- **特征消融优胜**：`feature_combined`；{"name": "feature_combined", "family": "xgb", "variant": "combined", "native": false, "params": {"n_estimators": 350, "learning_rate": 0.02, "max_depth": 4, "min_child_weight": 30, "reg_lambda": 10, "subsample": 0.8, "colsample_bytree": 0.8}}。
- **联合搜索优胜**：`joint_xgb_089`；{"name": "joint_xgb_089", "family": "xgb", "params": {"colsample_bytree": 0.65, "gamma": 0.05, "learning_rate": 0.03962175009621365, "max_depth": 5, "min_child_weight": 22.839155331252698, "n_estimators": 393, "reg_alpha": 1.0, "reg_lambda": 119.89079839483685, "subsample": 0.8}, "variant": "interactions", "native": false}。
- **简单融合优胜**：`blend_33`；0.33 × joint_xgb_089 + 0.33 × frozen_ridge + 0.33 × joint_lgb_05。
- **小型神经网络优胜**：`nn_64_nohep_raw`；{"name": "nn_64_nohep_raw", "family": "nn", "variant": "no_hepatitis", "widths": [64, 32], "dropout": 0.1, "pca": false, "frozen_epochs": 36}。
- **补充：小网络融合**：`nn_blend_06`；0.25 × nn_64_nohep_raw + 0.75 × blend_33。

完整成员参数见 frozen_selection.json 的 specifications；融合模型通过成员名称引用各自冻结的特征处理和参数。

小型神经网络与此前冻结大网络的同协议参考：原网络开发MSE=2.375530、B MSE=1.342265；本轮小网络开发MSE=2.260269、B MSE=1.352561。原网络引用此前已保存结果，本轮没有重训它，不能把单次差异当作网络结构的确定性因果收益。

训练OOF极端记录id4609在64→32原始全特征小网络中预测66.675，去乙肝同结构后预测4.699989（真实4.58）。这支持对该输入组做针对性处理的必要性，但单条记录不能证明乙肝特征对所有模型都无用。小网络单模型仍弱于树模型融合，较小权重加入后能改善本轮融合；以训练OOF验证互补性，比只看单模型排名更合适。

## 开发/B全部指标

| dataset | name | MSE | RMSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- |
| development | baseline_xgb | 2.228808 | 1.492919 | 0.780951 | 0.186578 |
| development | feature_combined | 2.218877 | 1.489589 | 0.774954 | 0.190202 |
| development | joint_xgb_089 | 2.236749 | 1.495577 | 0.782494 | 0.183680 |
| development | blend_33 | 2.196540 | 1.482073 | 0.777495 | 0.198354 |
| development | nn_128_pca | 2.392069 | 1.546632 | 0.815605 | 0.126994 |
| B | baseline_xgb | 1.252744 | 1.119260 | 0.648007 | 0.184240 |
| B | feature_combined | 1.275351 | 1.129314 | 0.654285 | 0.169518 |
| B | joint_xgb_089 | 1.242601 | 1.114720 | 0.646290 | 0.190845 |
| B | blend_33 | 1.245558 | 1.116046 | 0.652927 | 0.188919 |
| B | nn_128_pca | 1.388255 | 1.178242 | 0.641939 | 0.095998 |
| development | nn_64_nohep_raw | 2.260269 | 1.503419 | 0.820975 | 0.175096 |
| B | nn_64_nohep_raw | 1.352561 | 1.162997 | 0.668018 | 0.119241 |
| development | nn_blend_06 | 2.177243 | 1.475548 | 0.777364 | 0.205397 |
| B | nn_blend_06 | 1.234296 | 1.110989 | 0.638160 | 0.196253 |

## 相对原XGBoost的配对区间

以下差值为“新模型MSE−基线MSE”，负数表示新模型更好。固定拟合模型下3000次行配对bootstrap的描述性95%区间，不包括训练/调参随机性、日期聚类、多重比较；不能当成整个优化流程的无偏显著性检验。

| dataset | name | MSE_difference | low95 | high95 |
| --- | --- | --- | --- | --- |
| development | feature_combined | -0.009931 | -0.037683 | 0.018810 |
| development | joint_xgb_089 | 0.007941 | -0.048668 | 0.078473 |
| development | blend_33 | -0.032268 | -0.068530 | 0.001123 |
| development | nn_128_pca | 0.163262 | 0.041124 | 0.284980 |
| B | feature_combined | 0.022607 | -0.016673 | 0.057049 |
| B | joint_xgb_089 | -0.010143 | -0.050152 | 0.028644 |
| B | blend_33 | -0.007185 | -0.042229 | 0.024062 |
| B | nn_128_pca | 0.135511 | 0.036389 | 0.246693 |
| development | nn_64_nohep_raw | 0.031461 | -0.117256 | 0.163998 |
| B | nn_64_nohep_raw | 0.099817 | 0.014593 | 0.185646 |
| development | nn_blend_06 | -0.051565 | -0.101899 | -0.003706 |
| B | nn_blend_06 | -0.018448 | -0.053797 | 0.016102 |

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

原四阶段计算耗时11.59分钟，另有去乙肝网络补充实验。依赖版本：{"scikit-learn": "1.3.0", "xgboost": "2.1.3", "lightgbm": "4.1.0", "tensorflow": "2.16.2"}。原四阶段源码SHA256：`f301925f6849114a0dfea5c17be5fb58be5f9a90d7b22c6d9015ab6c74c28d39`；两个补充脚本摘要见各自plan.json。神经网络在不同硬件/运行库上可能出现数值差异。
