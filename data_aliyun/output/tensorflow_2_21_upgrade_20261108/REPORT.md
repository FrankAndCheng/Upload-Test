# TensorFlow 2.21.0 升级及项目兼容性验证

日期：2026-11-08。环境：`/Users/chengchong/anaconda3`（Conda base），Python 3.10.20，Apple Silicon。

## 结论

TensorFlow 已从 2.16.2 升级至 2.21.0，配套依赖完成更新。当前项目通过入口测试、环境检查、小样本训练、模型保存/重载、历史模型加载和真实特征推理。对同一批 1000 条 B 特征，冻结四模型融合升级前后预测的最大绝对差与平均绝对差均为 0。

## 环境变更

| 包 | 升级前 | 升级后 |
|---|---|---|
| flatbuffers | 24.12.23 | 25.12.19 |
| grpcio | 1.69.0 | 1.85.1 |
| keras | 3.8.0 | 3.12.4 |
| ml-dtypes | 0.3.2 | 0.5.4 |
| protobuf | 4.25.5 | 7.36.2 |
| tensorboard | 2.16.2 | 2.21.0 |
| tensorflow | 2.16.2 | 2.21.0 |
| tensorflow-macos | 2.16.2 | 移除旧元包 |

NumPy 保留 1.26.4；SciPy 1.11.1、pandas 2.0.3、scikit-learn 1.3.0 和 Python 均保持原版本。TensorFlow 2.21.0 的 `numpy>=1.26.0` 约束支持这一组合，因此本次无需切换 NumPy 主版本。此决定同时保留现有 sklearn/joblib 模型的运行环境。

`tensorflow-macos 2.16.2` 是锁定 `tensorflow==2.16.2` 的旧元包，移除后由提供原生 macOS ARM64 安装包的 `tensorflow 2.21.0` 直接运行。所有新增安装包来自官方 PyPI，具体版本与来源见 pip-plan.json。

## 已执行的项目测试

1. 原有 `python -m unittest discover -s tests -v`：升级前、后均 6/6 通过，包括 13 份历史源码的 SHA256 核验。
2. 原有 `python -m glucose_project.environment.check`：12 项导入、9 项操作全部通过，包含预处理、Ridge、HistGB、AdaBoost、XGBoost、LightGBM、OLS、因子分析及 Keras 批训练。
3. 项目 `run_experiments.estimator` 的 5 个冻结构造器：XGBoost、LightGBM、HistGB、RandomForest、Ridge，使用原配置在 128 条真实训练样本上拟合、对 32 条样本预测，全部通过。
4. 项目小网络 `fit_nn_predict`：128 条样本训练 2 轮，预测、保存 `.keras`、重载后预测全部通过，保存前后最大绝对差为 0。
5. 原大网络 `run_ablation.neural`：使用原 PReLU / BatchNormalization / Dropout 结构和标准化/PCA，在 128 条样本训练 2 轮；预测、保存及重载一致性通过。
6. 全部 8 个历史 `.keras` 模型：加载成功，并对匹配输入维数的合成数据输出有限值。
7. 项目正式分类推理入口 `glucose_project.inference.predict`：升级前、后均加载原冻结模型，对 1000 条真实 B 特征执行推理；id 顺序一致，逐条预测完全一致（最大绝对差 0）。
8. 全环境 `pip check`：无损坏依赖；Conda doctor：依赖一致、无缺失包文件。

## 文件保护与验证边界

未修改业务源码、notebook、训练数据、历史模型、冻结参数和原有研究分数表。核对了 261 个原有文件的内容摘要；唯一更新的原有文件是 `review/environment_check.json`，它记录当前环境检查结果。旧环境检查文件备份在本目录。新测试模型、预测和日志全部位于本目录。

本次是兼容性验证，未重新运行完整参数搜索、多轮交叉验证或全量模型训练；小样本通过不表示重新训练后的收敛轨迹与指标必然相同。测试没有使用 B 标签训练、调参或评分。

## 复核入口与记录

- `test_project_runtime.py`：本次项目功能测试，可在项目根目录用 base Python 重跑（会更新本目录的测试模型与结果）。
- `runtime-results.json`：5 组项目功能测试的详细结果。
- `predictions-before.csv` / `predictions-after.csv`：升级前后逐行预测。
- `environment-check-before.json` / `environment-check-after.json`：环境检查记录。
- `upgrade-packages.txt`：本次升级所使用的精确版本。
- `pip-before.txt` / `pip-after.txt`：完整 Python 包版本快照。
- `packages-before.json` / `packages-after.json`：Conda 包清单。
- `project-hashes-before.json` / `project-integrity.json`：原有文件保护核验。

复核现有项目入口：

```sh
/Users/chengchong/anaconda3/bin/python -m unittest discover -s tests -v
```
