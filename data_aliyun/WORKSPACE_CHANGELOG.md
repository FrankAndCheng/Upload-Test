# 工作区变更与分类提交记录

日期：2026-11-11。

## 范围

本次仅整理并提交当前 `data_aliyun/` 工作区，在原分支 `master` 上创建本地提交。没有执行push、pull、fetch或其他远端操作。

整理前当前工作区共有399个待提交文件，合计约94.89 MiB。既有原始数据没有修改；模型权重、冻结参数、评分/预测表、研究图表和论文交付包随对应主题保存。

## 分组提交

| 序号 | 本地commit | 主题 | 文件数 |
| --- | --- | --- | --- |
| 1 | `c4a7ab4f1b` | 忽略规则：Python字节码、Notebook检查点、可重算训练折缓存 | 1 |
| 2 | `5831d9f46b` | 主notebook：训练/验证分离、原六算法冻结参数与B复核 | 12 |
| 3 | `e74a2daf44` | 历史优化：1561条同样本与仅原train的A/B协议 | 23 |
| 4 | `48c6eaf676` | 顺序实验：特征工程、联合搜索、OOF融合及冻结模型/预测 | 54 |
| 5 | `456cf7fdfc` | 预处理对照：标准化/PCA配对实验与模型重载验证 | 53 |
| 6 | `5ea5eb4258` | 分类入口：8类13个工作流、帮助/预览和入口测试 | 31 |
| 7 | `33bbaacb56` | 验证文档与隐私审计：MSE总表、来源、去标识化及清理记录 | 8 |
| 8 | `5609144e18` | 研究梳理：数据/Notebook/图表证据与论文结构 | 32 |
| 9 | `ec1f9eb004` | 论文源文件：LaTeX章节、表格、参考文献和图表资源 | 65 |
| 10 | `6a96a807fa` | 质量与交付：论文核对、PDF逐页检查和最终PDF/源码包 | 39 |
| 11 | `da2f64814d` | 环境兼容：base修复、TensorFlow与NumPy升级及兼容验证 | 81 |

本文件另作为最后一笔“工作区变更汇总”提交，记录前述分类；其自身commit可通过`git log -- data_aliyun/WORKSPACE_CHANGELOG.md`查看。

## 已执行检查

- 现有6项工作流入口测试通过。
- 50份新增Python源码语法检查通过，95份JSON解析通过，4个Notebook结构校验通过。
- 新增候选及两个ZIP展开内容的常见私钥/令牌模式检查未命中；两个ZIP CRC检查通过。
- 暂存文件集逐组与审查清单一致，提交后再次核对每笔commit实际包含的路径。
- 部分历史报告/日志存在既有空白格式提示；这些文件按原字节保留，避免改变冻结摘要和审计来源。
- 本次为提交整理，没有重新训练模型、调参、重算整套实验或重新编译论文。

## 留在本地的运行文件

项目级`.gitignore`排除Python字节码、Notebook检查点及两个训练缓存目录；本次核对到590个运行/缓存文件保持本地、没有删除。已拟合模型、正式预测CSV、实验配置、论文PDF/交付ZIP与审计报告均未被泛化为缓存忽略。

## 父仓库中保留的其他变更

Git根目录位于工作区上一级；下列6个文件属于其他项目，其工作内容保持原样，没有加入本次暂存或提交：

- `algorithm/Eight Qweens.ipynb`
- `algorithm/locker and student.ipynb`
- `algorithm/prime numbers.ipynb`
- `assignment/assignment.ipynb`
- `school-canteen.ipynb`
- `test.ipynb`

## 后续查看

```bash
git log --oneline -- data_aliyun
git status --short
```

当前验证与MSE结论见`VALIDATION_SUMMARY.md`；分类入口见`glucose_project/README.md`。环境升级记录保留在`output/`对应日期目录，本次未再次修改依赖环境。
