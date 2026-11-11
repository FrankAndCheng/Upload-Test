"""记录已完成的视觉检查，交付同一版本PDF与源码；不再修改排版。"""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import zipfile
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
EDITOR=ROOT/'research_analysis/stage3'
DELIVER=ROOT/'output/pdf'
DELIVER.mkdir(parents=True,exist_ok=True)
check=json.loads((OUT/'pdf_checks.json').read_text())
check.update(visual_review_complete=True,visual_reviewed_pages=list(range(1,33)),
             detail_reviewed_pages=[1,6,20,22,24,25,26,27,30,32],
             visual_method='All 32 pages rendered and visually inspected; final pagination changed only pages 26/27, which were re-inspected. Other page-image hashes were unchanged.',
             source_main='research_analysis/stage3/paper_body.tex',pdf_delivery='output/pdf/glucose_paper.pdf')
(OUT/'pdf_checks.json').write_text(json.dumps(check,ensure_ascii=False,indent=2))
notes=[
 '标题均衡两行；摘要、关键词与公式1完整。','问题分析标题和段落完整。',
 '符号表与章节衔接清楚。','协议表、填补及标准化公式完整，表格字距正常。',
 'PCA公式与年龄图完整。','血糖分布和缺失图完整，另作高分辨率检查。',
 'OLS、Lasso和辅助指标公式完整。','OOF、Ridge和算法列表完整。',
 '树、森林、AdaBoost和XGB公式无截断。','基准配置表与比值公式完整。',
 '特征表、工程表达与参数段落完整。','网络目标与融合公式完整。',
 '融合成员表与六模型对照表清楚。','阶段结果、最终指标与改善公式完整。',
 '预处理定义与替换规则完整。','算法2、四处理结果和引用完整。',
 '融合预处理表和配对差异公式完整。','区间、分箱表及贡献公式完整。',
 '局限与结论完整，无孤立标题。','11条参考文献及长链接完整，无溢出。',
 '字段表第一部分完整。','字段续表带标题和列头，参数表完整。',
 '联合搜索表、版本段落与历史说明完整。','历史图表及84行长表起始完整。',
 '长表续页标题、列头和数据行完整。','长表末行与底线同页，孤行已修正。',
 '说明段落完整起页，图5完整。','图6、残差图及坐标图注完整。',
 'Q-Q图和图注完整，单图页为正常浮动排版。','相关热图与PCA完整，另作高分辨率检查。',
 '网络损失图和图注完整，单图页为正常浮动排版。','来源表与追溯说明完整。']
assert len(notes)==32
with (OUT/'page_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f);writer.writerow(['page','visual_result','notes'])
    writer.writerows((i,'通过',note) for i,note in enumerate(notes,1))
report='''# 独立PDF交付与逐页排版检查

检查日期：2026-11-03。主入口仍为 `research_analysis/stage3/paper_body.tex`，没有替换编辑器文档。

**32页A4独立PDF已生成并完成逐页视觉检查。当前本地第二遍编译无错误、无警告；内置编译器也返回成功。**

## 本次修正

- 标题均衡分为两行，避免末行仅剩少数字。
- 页码统一置于页脚中央，保留PDF书签和交叉链接。
- 表格文字列改为左对齐，列间距由3pt调整为5pt；参考文献和软件版本段落调整对齐，消除7处Underfull提示。
- 两张跨页表补充续表标题；84行指标表收紧行距，末行和底线保留在第26页。
- 图表说明从第27页完整起排，消除句尾几个字单独跨页。

上述均为排版修改。23个编号公式逐式相同，19张表的表头和每行内容相同，全部图片资源字节摘要不变。没有重训、改选模型或改变实验数值。

## 检查方法与结果

先检查用户导出的32页，再对修改后的PDF逐页渲染复核。最终分页修改只影响第26、27页，其他30页的图像摘要与已完成的视觉检查一致；第26、27页已重新逐页查看。第6、30页的密集图另做高分辨率检查。

| 项目 | 结果 |
| --- | --- |
| 页面与页码 | 32页，全部A4，页脚连续1—32，无空白页 |
| 文字与公式 | 未发现截断、重叠、缺字或未解析引用标记 |
| 图表 | 11幅图、19张表均完整，图注与正文引用对应 |
| 跨页长表 | 续表列头清楚，末行未单独占新页 |
| 字体 | 检测到的字体均已嵌入 |
| 内部链接 | 158个命名目标，未发现失效内部目标 |
| 编译 | 内置成功；本地第二遍无Warning、Underfull、Overfull或Missing character提示 |
| 内容保留 | 23个编号公式、19张表及图片资源与本次排版前相同 |

初次Poppler渲染器未找到其中文映射资源，产生检查工具自身的提示，故最终图像验收采用PDFium。PDFium显示中文正常，文本提取与字体嵌入检查通过。没有通过修改论文内容来掩盖工具问题。

逐页记录见 `page_review.csv`，结构与字体检查见 `pdf_checks.json`，排版差异见 `layout_changes.patch`。单图页留白属于正常浮动排版，不是缺页。

## 交付与边界

独立PDF为 `output/pdf/glucose_paper.pdf`；交付包包含相同PDF、完整LaTeX工程、图片、参考文献、来源登记和本次检查记录。原导出目录中的 `paper_body.pdf` 也已更新为相同版本。

独立PDF和逐页排版两项已完成。官方模板、页数/匿名规则尚未提供，仍不能声称已通过赛事官方格式验收。数据单位、采样说明及历史研究限制继续按正文保留。
'''
(OUT/'PDF逐页检查报告.md').write_text(report)
shutil.copy2(EDITOR/'paper_body.pdf',DELIVER/'glucose_paper.pdf')
assert hashlib.sha256((DELIVER/'glucose_paper.pdf').read_bytes()).hexdigest()==check['PDF_SHA256']
source={str(p.relative_to(EDITOR)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [EDITOR/'paper_body.tex',*sorted((EDITOR/'paper').rglob('*.tex')),*sorted((EDITOR/'paper/figures').glob('*.png'))]}
(OUT/'final_source_sha256.json').write_text(json.dumps(source,ensure_ascii=False,indent=2))
qpath=ROOT/'research_analysis/stage5/quality_summary.json'
quality=json.loads(qpath.read_text())
quality.update(pdf_review_date='2026-11-03',overall_status='content_compilation_and_pdf_visual_review_verified_official_format_pending',pdf_exported=True,page_visual_review=True,pdf_pages=32,pdf_delivery='output/pdf/glucose_paper.pdf',scientific_content_unchanged_in_pdf_layout_revision=True)
qpath.write_text(json.dumps(quality,ensure_ascii=False,indent=2))
update='\n\n## 2026-11-03 独立PDF与逐页检查已完成\n\n32页PDF已完成逐页视觉检查及排版修正，本地编译无警告，内置编译成功。详见[逐页检查报告](pdf_review/PDF逐页检查报告.md)。独立PDF及源码交付包位于工作目录的 `output/pdf/`。早期PDF未导出的说明是历史状态；官方模板核对仍待材料。\n'
for name in ['README.md','质量审查报告.md']:
    p=ROOT/'research_analysis/stage5'/name
    if '2026-11-03 独立PDF与逐页检查已完成' not in p.read_text():p.write_text(p.read_text()+update)
p=EDITOR/'paper/README.md'
if '2026-11-03 PDF交付更新' not in p.read_text():
    p.write_text(p.read_text()+'\n\n## 2026-11-03 PDF交付更新\n\n独立PDF和32页逐页视觉检查已完成。标题、页码、表格字距、续表标题和分页已修正，23个编号公式及19张表内容不变。当前本地编译无警告，内置编译成功。官方模板仍待提供。最新独立PDF及源码交付包位于原工作目录的 `output/pdf/`；旧说明中的导出限制属于早期状态。\n')
files={str(p.relative_to(EDITOR)):p.read_bytes() for p in [EDITOR/'paper_body.tex',*sorted((EDITOR/'paper').rglob('*'))] if p.is_file()}
files['paper_body.pdf']=(DELIVER/'glucose_paper.pdf').read_bytes()
files['README.md']='''# 论文独立PDF与LaTeX工程交付

直接阅读 paper_body.pdf；修改时，以 paper_body.tex 为主入口，在本目录使用XeLaTeX编译两遍。

已完成32页逐页排版检查，报告位于 audit/PDF逐页检查报告.md。23个编号公式、19张表及实验结论保持原研究内容。图表和文献在 paper/ 中，来源登记在 provenance/ 中。

官方模板、页数和匿名要求仍未提供，本次交付不代表官方格式验收。
'''.encode()
for name in ['PDF逐页检查报告.md','page_review.csv','pdf_checks.json','layout_changes.patch','builtin_compile_result.json','final_source_sha256.json']:
    files['audit/'+name]=(OUT/name).read_bytes()
for name in ['table_figure_sources.json','reference_sources.json']:
    files['provenance/'+name]=(EDITOR/name).read_bytes()
files['provenance/package_sha256.json']=json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in files.items()},ensure_ascii=False,indent=2).encode()
dest=DELIVER/'glucose_paper_delivery.zip'
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for name,value in sorted(files.items()):archive.writestr('glucose_paper/'+name,value)
with zipfile.ZipFile(dest) as archive:
    assert archive.testzip() is None
    for name,digest in json.loads(archive.read('glucose_paper/provenance/package_sha256.json')).items():
        assert hashlib.sha256(archive.read('glucose_paper/'+name)).hexdigest()==digest
    for name,value in files.items():
        if name.endswith('.tex'):
            for command,path in re.findall(r'\\(input|includegraphics)(?:\[[^]]*\])?\{([^}]+)\}',value.decode()):
                if command=='input' and not path.endswith('.tex'):path+='.tex'
                assert path in files,(name,path)
    assert hashlib.sha256(archive.read('glucose_paper/paper_body.pdf')).hexdigest()==check['PDF_SHA256']
valid=dict(PDF_pages=len(PdfReader(DELIVER/'glucose_paper.pdf').pages),PDF_sha256=check['PDF_SHA256'],package_files=len(files),package_bytes=dest.stat().st_size,CRC_and_checksums_passed=True,all_TeX_dependencies_present=True,visual_reviewed_pages=list(range(1,33)))
(OUT/'delivery_validation.json').write_text(json.dumps(valid,ensure_ascii=False,indent=2))
print(json.dumps(valid,ensure_ascii=False,indent=2))
