# 经第五阶段审查的LaTeX工程

主入口为本目录中的 paper_body.tex，配套章节和资源位于 paper/。
以XeLaTeX编译主入口，或使用编辑器自动多遍编译。原工作目录中的主入口及编辑器未替换。

审查报告：audit/质量审查报告.md
模型与公式：audit/模型与公式核对.md
来源登记：provenance/来源核对表.md

当前源文件已通过内置编译。包内没有导出的PDF；逐页视觉验收和官方模板核对尚未完成，不能视为所有提交检查均已通过。

包含九章正文、五附录、23个编号公式、19张表和11幅显示图。全部11幅原图保留，相关热图使用同一探索矩阵的重排版本，因此PNG资源共12份。完整相关矩阵位于 paper/tables/training_correlation_matrix.csv。
文献实际使用 paper/references.tex，references.bib保留供模板迁移。

audit中的指标复算与记录基于原完整研究目录，包中不附原始训练数据、全部模型及缓存。来源登记中的研究文件使用相对路径；完整研究目录中可直接追溯，独立论文包不包含这些研究数据。
provenance/package_sha256.json登记除其本身外所有包内文件的摘要。
