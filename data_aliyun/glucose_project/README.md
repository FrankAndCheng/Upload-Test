# 血糖预测代码导航

本目录按功能提供13个可运行入口；实际训练实现保留在原位置，保留模型路径和成绩引用；隐私修订后的源码摘要登记在 source_catalog.json，原摘要保存在 pre_privacy_sha256 字段。

[完整目录树、源码映射、命令与依赖说明](../VALIDATION_SUMMARY.md#L540)

在项目根目录使用Conda base：

```bash
python -m glucose_project list
python -m glucose_project run predict --help
python -m glucose_project run sequential --dry-run
```

分类：environment（环境）、experiments（实验）、ensembles（融合）、preprocessing（预处理）、inference（预测）、validation（核验）、reports（报告）、legacy（历史协议）。

各模块导入不会训练。`--help`只显示帮助；`--dry-run`只显示执行计划。真正运行时使用同一Python解释器、项目根工作目录和原始源码路径，输出仍在对应review实验目录。
