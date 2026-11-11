# NumPy 2 科学计算依赖联合升级

日期：2026-11-08。目标环境：`/Users/chengchong/anaconda3`（Conda base），Python 3.10.20、Apple Silicon。TensorFlow 保持 2.21.0。

## 结论

NumPy 及 SciPy、pandas、scikit-learn、statsmodels、ml-dtypes 已完成配套升级。先在临时兼容环境验证，再应用到 base，并使用实际安装的 Conda NumPy 二进制重跑项目测试。现有测试和本次功能检查全部通过；对同一批 1000 条 B 特征的冻结模型融合预测，升级前后的最大绝对差和平均绝对差均为 0。

## 实际版本变更

| 包 | 原版本 | 当前版本 |
|---|---|---|
| factor-analyzer | 0.4.1 | 0.5.1 |
| formulaic | 未安装 | 1.2.2 |
| imbalanced-learn | 0.11.0 | 0.14.2 |
| interface-meta | 未安装 | 2.0.1 |
| ml-dtypes | 0.5.4 | 0.6.0 |
| mlxtend | 0.22.0 | 0.23.4 |
| narwhals | 未安装 | 2.28.0 |
| numpy | 1.26.4 | 2.2.5 |
| numpy-base | 1.26.4 | 2.2.5 |
| pandas | 2.0.3 | 2.3.3 |
| pandas-stubs | 2.0.3.230814 | 2.3.3.260113 |
| patsy | 0.5.3 | 1.0.3 |
| scikit-learn | 1.3.0 | 1.4.2 |
| scikit-optimize | 0.9.0 | 0.10.2 |
| scipy | 1.11.1 | 1.15.3 |
| sklearn-compat | 未安装 | 0.1.6 |
| statsmodels | 0.14.0 | 0.15.0 |
| wordcloud | 1.9.2 | 1.9.6 |

共 18 个包记录发生变化，其中 3 个为新增依赖。NumPy 和 numpy-base 由官方 Conda 仓库同步管理，其余原 pip 包从官方 PyPI 获取。没有使用 pip 覆盖 Conda 管理的 NumPy 文件。

### 版本选择

- NumPy 2.2.5 满足现有 Numba 0.61.2 的 `numpy<2.3` 约束，也有官方 Conda 构建。
- scikit-learn 1.4.2 已支持 NumPy 2，且本次核验的旧模型和真实推理正常。这是为现有模型选定的兼容升级版本，本次未采用候选的 1.7.2。
- SciPy 1.15.3、pandas 2.3.3、statsmodels 0.15.0、ml-dtypes 0.6.0 等来自针对 Python 3.10 的官方源联合解析结果。
- imbalanced-learn、mlxtend、scikit-optimize、pandas-stubs、patsy、wordcloud 等同步升级；formulaic、interface-meta、narwhals、sklearn-compat 是新版本所需依赖。
- 实测旧 factor-analyzer 0.4.1 在新 SciPy 下调用已移除的 `scipy.sum`，导致真实拟合失败。因此额外升级至官方 0.5.1，已验证拟合通过。

## 验证结果

- `pip check`：No broken requirements found。
- Conda doctor：环境依赖一致、无缺失文件。
- 项目原有 6 个入口测试全部通过，包括 13 份历史源码摘要核对。
- 项目原有环境检查：12 个库导入和 9 项操作全部通过。
- 38 个历史 joblib 文件均能加载；对暴露输入维数的估计器/预处理器额外执行匹配形状的合成输入调用。容器类归档只做加载检查。
- 8 个历史 Keras 模型均能加载并对匹配维数的合成输入输出有限值。
- 项目原冻结参数的 XGBoost、LightGBM、HistGB、RandomForest、Ridge 在 128 条真实训练样本上拟合、对 32 条样本预测成功。
- 项目小网络及原 PReLU 大网络均完成 2 轮小样本训练、预测、保存和重载验证。
- 因子分析拟合、OLS 公式接口、过采样、GP 优化、apriori、词云生成、Torch/NumPy 互操作、Numba JIT 和 HDF5 读写全部通过。
- 正式分类预测入口对同一批 1000 条真实 B 特征执行升级前/后推理，id 一致、逐条预测完全一致。

## 已知提示及验证边界

旧 joblib 中的 sklearn 对象保存于 1.3.0，当前 1.4.2 加载时仍会发出跨版本警告。上述已有归档和预测已实际验证通过；不应据此推断任意其他旧模型也必然兼容。

本次未重跑完整超参数搜索、全量交叉验证和全量模型训练；小样本训练只验证接口和数值能运行，不证明重新训练后的收敛轨迹或指标与历史一致。未使用 B 标签训练、调参或评分。

核对 261 个原有项目文件，业务源码、notebook、训练数据、历史模型、冻结参数和成绩表均未改动。唯一更新的既有文件为 `review/environment_check.json`，用于记录当前环境；旧检查结果已保存于本目录。所有新测试模型、预测和日志位于本目录。

## 复核与恢复记录

- `test_project_runtime.py`：本次 7 组功能回归检查，可用 base Python 在项目根目录重跑。脚本会更新本目录中的测试文件。
- `runtime-results-after.json`：详细功能检查结果。
- `environment-check-before.json` / `environment-check-after.json`：项目环境检查。
- `predictions-before.csv` / `predictions-after.csv`：逐行推理对照。
- `package-changes.json`、`packages-before.json` / `packages-after.json`：包版本变更。
- `pip-before.txt` / `pip-after.txt`：完整 Python 包版本快照，后者是当前版本记录。
- `conda-plan.json` / `conda-install.json`：NumPy 的 Conda 预览与安装事务。
- `stage-packages.txt` / `pip-upgrade-packages.txt`：精确目标版本；前者含 NumPy，后者仅用于原 pip 包，NumPy 应继续由 Conda 管理。
- `project-hashes-before.json` / `project-integrity.json`：原有文件摘要保护核验。

复核命令（项目根目录）：

```sh
/Users/chengchong/anaconda3/bin/python -m unittest discover -s tests -v
/Users/chengchong/anaconda3/bin/python output/numpy2_upgrade_20261108/test_project_runtime.py
```
