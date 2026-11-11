"""打包当前论文与审查证据；不编译、不改变论文。"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EDITOR = ROOT / 'research_analysis/stage3'
prefix = 'glucose_latex_reviewed/'
files = {}
for p in [EDITOR/'paper_body.tex', *sorted((EDITOR/'paper').rglob('*'))]:
    if p.is_file():
        files[str(p.relative_to(EDITOR))] = p.read_bytes()

readme = '''# 经第五阶段审查的LaTeX工程

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
'''
files['README.md'] = readme.encode()
(OUT/'PROJECT_README.md').write_text(readme)
for p in OUT.iterdir():
    if p.suffix in {'.json', '.csv', '.md', '.patch'} and p.name not in {'README.md', 'PROJECT_README.md', 'package_validation.json'}:
        value = p.read_text()
        if p.name == '质量审查报告.md':
            value = value.replace('(glucose_latex_reviewed.zip)', '(../README.md)')
        files['audit/'+p.name] = value.encode()
files['audit/README.md'] = '本目录保存第五阶段审查与复核记录；请先阅读质量审查报告.md。论文主入口位于上一层。完整研究脚本和数据仍在原工作目录，本包不包含它们。\n'.encode()
for name in ['来源核对表.md', 'table_figure_sources.json', 'reference_sources.json']:
    value = (EDITOR/name).read_text()
    if name == '来源核对表.md':
        value = value.replace('../stage5/', '../audit/')
    files['provenance/'+name] = value.encode()
for name in ['notebook_cell_index.csv', 'figure_index.csv', 'source_manifest.json']:
    files['provenance/stage1_'+name] = (ROOT/'research_analysis/stage1'/name).read_bytes()
files['audit/privacy_revision.json'] = (ROOT/'review/privacy_review/manifest.json').read_bytes()
files['provenance/package_sha256.json'] = json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in files.items()}, ensure_ascii=False, indent=2).encode()
dest = OUT/'glucose_latex_reviewed.zip'
with zipfile.ZipFile(dest, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for name, content in sorted(files.items()):
        archive.writestr(prefix+name, content)

with zipfile.ZipFile(dest) as archive:
    assert archive.testzip() is None
    hashes = json.loads(archive.read(prefix+'provenance/package_sha256.json'))
    assert all(hashlib.sha256(archive.read(prefix+k)).hexdigest()==h for k,h in hashes.items())
    source = json.loads((OUT/'compiled_source_sha256.json').read_text())
    assert all(hashlib.sha256(archive.read(prefix+k)).hexdigest()==h for k,h in source.items())
    # 检查全部TeX依赖和显示图片均存在于包内，路径以主入口为基准。
    deps = []
    for name, content in files.items():
        assert not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
        if not name.endswith('.tex'):
            continue
        text = content.decode()
        for command, path in re.findall(r'\\(input|includegraphics)(?:\[[^]]*\])?\{([^}]+)\}', text):
            if command == 'input' and not path.endswith('.tex'):
                path += '.tex'
            assert path in files, (name, path)
            deps.append(path)
    result = dict(archive=str(dest.relative_to(ROOT)), files=len(files), bytes=dest.stat().st_size,
                  sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), crc_check=True,
                  all_file_hashes_match=True, compiled_sources_match=True,
                  all_TeX_dependencies_resolved=True, dependencies_checked=len(deps),
                  exported_paper_PDF_included=False)
(OUT/'package_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
