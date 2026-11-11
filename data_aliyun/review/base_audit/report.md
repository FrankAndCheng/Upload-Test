# Conda base 全环境依赖审查

环境：<CONDA_ENV>，Python 3.10.20，macOS arm64。

## 当前结果与边界

- 初始发现 292 条 Python 分发记录；去除重复版本残留并补齐依赖后为 286 个唯一分发包。
- 已核对全部 Python 分发包的依赖声明、重复版本记录和 wheel 平台标签。当前 pip check 输出 No broken requirements found；没有重复分发记录，没有不兼容 wheel 标签。
- 对 289 个声明模块入口做了独立进程导入检查：285 个直接通过；余下 4 个见下文，未发现尚未解决的实际公共接口导入故障。
- 检查了 Conda 原生包的文件完整性和依赖一致性。未发现缺失/异常文件；随后经用户明确授权，已修复两项本地元数据问题；最终检查见下文。
- 这不是对所有包所有函数的全覆盖测试，也没有访问数据库、云端服务或验证外部账户配置。

## 已完成修复

1. 安装 sniffio 1.3.1，修复 OpenAI/Trio 的缺失依赖。
2. 更新 virtualenv 至 20.26.6、distlib 至 0.3.9，使其兼容现有 platformdirs 4.12.3；创建临时虚拟环境并运行 Python 已通过。
3. 将 zipp 恢复为 Conda 管理的 4.1.0，满足 importlib-metadata 9.0.0。
4. 将 setuptools 恢复为 80.9.0，保留旧工具所需的 pkg_resources，修复 clyent、nodeenv、setuptools-rust 的实际导入失败。Setuptools 82 起已移除该接口；没有升级所有包来迁就它。
5. 统一 click、exceptiongroup、filelock、platformdirs、PyJWT、zipp 等混装记录。将 8 份过时/重复 dist-info 备份移出搜索路径，保留与 Conda 文件清单一致的记录。filelock 最终为 3.29.4，兼容 virtualenv 的 <4 要求。
6. 将 mysqlclient 从依赖已丢失 Homebrew 动态库的 pip 2.2.0 安装替换为 Conda 2.3.0，并安装 mysql-common/mysql-libs。MySQLdb 导入及客户端版本查询通过；没有连接数据库。
7. 重装 OpenSSL、libcurl、ncurses（libcurl 更新为 8.23.0），消除了文件完整性异常。没有发现缺失文件。
8. 重装 grpcio 1.69.0 官方 universal2 wheel。进一步核实，其原来的 x86_64 提示是内部 WHEEL 标签错误，不能据此断言实际二进制只有 Intel 架构。实际文件有 arm64+x86_64 两个 slice，原生导入和 channel 创建通过；已备份并修正标签及 RECORD 校验信息。
9. conda-verify 是可在 Python 3 运行的 noarch 纯 Python 包，原 WHEEL 却写 cp27。导入和 CLI 验证后，备份并修正了该标签。
10. libllvm12 12.0.0 没有任何 Conda 包反向依赖，是孤立旧运行库；已通过 Conda 移除，消除其旧 zlib 约束。Numba JIT 和 llvmlite 仍正常。
11. 前一轮的 NumPy/numpy-base 1.26.4 及 LightGBM OpenMP 路径修复仍有效，模型验证再次全部通过。

改动前后清单：conda_before.json、conda_after.json、pip_before.txt、pip_after.txt；重复记录及 WHEEL 原件备份在 metadata_backup/；修改明细为 metadata_repairs.json。

## 独立导入检查中的 4 项例外

- qtpy：受限沙箱不能正确查询 neon 特征，导致 Qt 拒绝初始化。正常权限下 from PyQt6.QtCore import QT_VERSION_STR; import qtpy 已成功，版本 6.8.1 / PyQt6，无需改包。
- scripts、testing：google-auth 的 top_level.txt 中包含这些名称，但没有相应已安装公共模块，是入口枚举噪声。google.auth、google.oauth2.service_account 和 protobuf 序列化均实测通过。
- tensorflow_io_gcs_filesystem：直接显式导入会重复注册 gs，产生 AlreadyExistsError；TensorFlow 2.16.2 自身已注册 gs。通过 tf.io.gfile.get_registered_schemes() 核实 gs 可用；没有为了满足机械导入测试而修改插件代码。

原始完整结果见 imports_before.json、imports.json、metadata_after.json。

## 功能验证

review/check_environment.py 的所有 12 项包导入、9 项数据/模型操作再次通过，包括 Ridge/PCA、HistGradientBoosting、AdaBoost、XGBoost、LightGBM、OLS、因子检验和 Keras 单批训练。

另有 9 项运行测试全部通过：PyTorch 自动微分、Numba JIT、MySQL 客户端加载、gRPC 原生接口、TensorFlow GCS 注册、Google 命名空间模块与 protobuf、virtualenv 创建、skopt 搜索、Excel/PNG/XML 读写。

Tcl 8.6.15、zlib 1.3.2、OpenSSL 3.5.9 运行查询通过。

## 两项元数据问题：用户授权后已修复

### Tk：本地 Conda 记录未同步官方依赖修订

官方查询对 tk 8.6.15 / hcd8a7d5_0 的依赖为 zlib >=1.2.13,<2.0.0a0；本地仍是 >=1.2.13,<1.3.0a0。已逐项核实 name/version/build/md5/sha256 完全相同，说明不是二进制不同，而是官方 repodata 已修订、本地记录仍旧。官方查询结果保存在 /private/tmp/base-tk-builds.json；该文件可重新生成。

普通官方重装仍复用了旧记录；全新临时缓存的重装又被已配置镜像的连接错误阻断。降级 zlib 的计划会连带降级几十个包及 Python/Conda，已排除，没有执行。

拟议的最后处理是：备份 Tk 的 conda-meta 记录，仅将 depends 同步为官方同校验构建的声明。该动作最初被自动审批拒绝；用户随后明确授权，现已完成，原记录已备份。

### PySocks：dist-info 目录大小写与 Conda 文件清单不一致

实际目录名 PySocks-1.7.1.dist-info，Conda 文件清单写 pysocks-1.7.1.dist-info。在大小写不敏感文件系统上内容可用，但包管理器按路径字符串判断时误认为被 pip 覆盖，并可能尝试移除对应 Conda 记录。Installer 已是 conda；Python 的 socks 接口正常。

拟议处理是：仅将 dist-info 目录名规范为小写（经临时名进行大小写重命名），不修改内容和版本。该动作最初被自动审批拒绝；用户随后明确授权，现已完成，原目录已备份。

两项操作已取得用户明确授权；未添加任何永久 Conda pin。变更与备份位置见 authorized_metadata_repairs.json。

## 使用建议

重启既有 Jupyter/PyCharm 内核，选择 python。

项目内 base_compatibility_constraints.txt 记录了当前核心兼容版本，仅作为后续安装时的参考，未修改全局 pip/Conda 配置。修复环境不等于必须把所有包升级到最新版。

## 用户授权后的最终验证

- Conda consistency：通过。
- Conda altered-files / missing-files：通过。
- pip check：No broken requirements found。
- 9 项扩展运行测试、12 项建模依赖导入、9 项数据与模型操作再次全部通过。
- Tk/Tcl 与 PySocks 运行查询通过；Conda list 不再报告 PySocks 目录大小写导致的覆盖警告。
- 原始备份保留在 metadata_backup/；未添加永久版本锁定。

这些验证确认已发现的版本、安装记录和动态库兼容问题得到解决，不代表对所有包的全部功能作出保证。
