# Conda base 兼容性修复

已修复 <CONDA_ENV>（Python 3.10.20）。

## 改动

- 通过官方 Conda 源将 numpy / numpy-base 从实际加载的 2.2.5 统一为 1.26.4，满足现有 SciPy 1.11.1 与 TensorFlow 2.16.2 的要求。禁用了本次操作的 Conda 自动升级，没有升级其他包。
- LightGBM 4.1.0 的 lib_lightgbm.so 引用了不存在的 /opt/homebrew/opt/libomp/lib/libomp.dylib。已将引用改为现有的 <CONDA_ENV>/lib/libomp.dylib，并重新进行本地 ad-hoc 签名。原库备份位于 /private/tmp/paper-numpy-repair/lib_lightgbm.so.before（临时目录不适合长期备份）。以后重装 LightGBM 可能覆盖此修复，需要重新核对动态库依赖。
- 没有修改 paper.ipynb 或系统默认 Python，也未安装尚未使用的 CatBoost。

## 验证

运行 `python review/check_environment.py` 已通过：

- numpy、pandas、scipy、sklearn、statsmodels、matplotlib、seaborn、xgboost、lightgbm、tensorflow、keras、factor_analyzer 全部导入成功。
- 读取原始 CSV；填补→标准化→PCA→岭回归；HistGradientBoosting、AdaBoost、XGBoost、LightGBM；OLS；KMO/Bartlett；与 notebook 相似的 Keras 层组合单批训练与推理均通过。
- 这是依赖和小规模训练验证，不等于已经运行完整 notebook 的大型搜索。

详见 environment_check.json。修复前后包清单见 conda_before_repair.json / conda_after_repair.json，项目版本约束见 paper-constraints.txt。

Jupyter/PyCharm 中应选择 python，并重启已有内核，避免继续使用进程中已加载的 NumPy 2。

## 其他已有环境问题

全环境 pip check 仍报告 sniffio 缺失、importlib-metadata 与 zipp、virtualenv 与 platformdirs 的约束问题，以及 conda-verify / grpcio 的平台元数据提示。它们不属于本次 NumPy 冲突，未为了消除提示而批量改动 base；TensorFlow 实际导入与训练已通过。

原审查报告中“环境缺少建模依赖”描述的是当时使用的默认执行环境；找到并修复 Conda base 后，建模依赖现已可用。
