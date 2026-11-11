"""第五阶段源文件结构检查，不调用编译器；读取已保存的内置编译结果。"""
from pathlib import Path
import re,json,hashlib,collections
ROOT=Path(__file__).resolve().parents[2]
EDITOR=ROOT/'research_analysis/stage3';OUT=ROOT/'research_analysis/stage5'
MAIN=EDITOR/'paper_body.tex'
visited=[]
def flatten(path):
    visited.append(path)
    s=path.read_text()
    def expand(m):
        p=EDITOR/m[1]
        if not p.suffix:p=p.with_suffix('.tex')
        assert p.is_file(),p
        return flatten(p)
    return re.sub(r'\\input\{([^}]+)\}',expand,s)
new=flatten(MAIN);old=(ROOT/'research_analysis/stage4/paper_body_before_stage4.txt').read_text()
def equations(s):
    return {re.search(r'\\label\{([^}]+)\}',m[1])[1]:m[1] for m in re.finditer(r'\\begin\{equation\}(.*?)\\end\{equation\}',s,re.S)}
def mathnorm(s):
    s=re.sub(r'\\(?:begin|end)\{aligned\}','',s)
    s=re.sub(r'\\(?:left|right|bigl|bigr|quad|qquad)\b','',s)
    s=s.replace(r'\tfrac',r'\frac').replace(r'\\','').replace('&','').replace('{}','')
    return re.sub(r'\s+','',s)
oe,ne=equations(old),equations(new)
eq_changed=[label for label in oe if label not in ne or mathnorm(oe[label])!=mathnorm(ne[label])]
def tablemap(s):
    return {re.search(r'\\label\{([^}]+)\}',m[0])[1]:m[0] for m in re.finditer(r'\\begin\{(table|longtable)\}.*?\\end\{\1\}',s,re.S)}
ot,nt=tablemap(old),tablemap(new)
def tabnorm(s):
    return re.sub(r'\s+','',s.replace(r'\footnotesize',r'\small'))
tables_changed=[k for k in ot if k not in nt or tabnorm(ot[k])!=tabnorm(nt[k])]
lab=re.findall(r'\\label\{([^}]+)\}',new);use=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',new)
cited={v for s in re.findall(r'\\cite\{([^}]+)\}',new) for v in s.split(',')}
bib=set(re.findall(r'\\bibitem\{([^}]+)\}',new));bib_keys=set(re.findall(r'@\w+\{([^,]+),',(EDITOR/'paper/references.bib').read_text()))
stack=[];errs=[]
for m in re.finditer(r'\\(begin|end)\{([^}]+)\}',new):
    if m[1]=='begin':stack.append(m[2])
    elif stack and stack[-1]==m[2]:stack.pop()
    else:errs.append(m[0])
assets=json.loads((ROOT/'research_analysis/stage4/project_structure.json').read_text())
images_ok=all(hashlib.sha256((EDITOR/'paper/figures'/r['file']).read_bytes()).hexdigest()==r['sha256'] for r in assets['figures'])
rendered_images=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',new)
assert all((EDITOR/p).is_file() for p in rendered_images)
reformat=json.loads((OUT/'correlation_reformat.json').read_text())
assert hashlib.sha256((ROOT/reformat['rendered_figure']).read_bytes()).hexdigest()==reformat['sha256']
manifest=json.loads((ROOT/'research_analysis/stage1/source_manifest.json').read_text())
manifest_changed=[k for k,h in manifest.items() if not (ROOT/k).is_file() or hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=h]
# Privacy-only changes keep the historical manifest intact and record both hashes.
privacy_path=ROOT/'review/privacy_review/manifest.json'
privacy=json.loads(privacy_path.read_text()) if privacy_path.exists() else {}
privacy_records={r['file']:r for r in privacy.get('changed_files', [])}
privacy_verified=[k for k in manifest_changed if k in privacy_records
    and privacy_records[k]['before_sha256']==manifest[k]
    and hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==privacy_records[k]['after_sha256']]
research_changed=[k for k in manifest_changed if Path(k).name!='.DS_Store' and k not in privacy_verified]
compile_result=json.loads((OUT/'compile_result.json').read_text())
compile_diagnostic=json.loads(compile_result['content'][0]['text'])
check=dict(changed_OS_metadata=[k for k in manifest_changed if Path(k).name=='.DS_Store'], main_file=str(MAIN.relative_to(ROOT)),included_tex_files=len(set(visited))-1,main_sections=len(re.findall(r'\\section\{',new.split(r'\appendix')[0])),appendices=len(re.findall(r'\\section\{',new.split(r'\appendix')[1])),numbered_equations=len(ne),tables=len(nt),images=len(rendered_images),original_image_assets=len(assets['figures']),total_image_assets=len(list((EDITOR/'paper/figures').glob('*.png'))),reformatted_images=1,references=len(bib),equation_content_changed=eq_changed,table_content_changed=tables_changed,duplicate_labels=[k for k,n in collections.Counter(lab).items() if n>1],undefined_references=sorted(set(use)-set(lab)),undefined_citations=sorted(cited-bib),bib_keys_match=bib==bib_keys,environment_errors=errs,unclosed_environments=stack,original_images_byte_preserved=images_ok,original_research_changed=research_changed,privacy_only_changes_verified=privacy_verified,has_absolute_graphic_paths=bool(re.search(r'\\includegraphics(?:\[[^]]*\])?\{/',new)),desktop_compilation=compile_diagnostic['kind'],pdf_exported=False,page_visual_review=False,limitations='Built-in compile tool returns diagnostics without PDF export. Prior CUA access denial for com.openai.codex is respected; current editor kept open, no alternate PDF compiled.')
(OUT/'project_checks.json').write_text(json.dumps(check,ensure_ascii=False,indent=2))
assert not any([eq_changed,tables_changed,check['duplicate_labels'],check['undefined_references'],check['undefined_citations'],errs,stack,research_changed,check['has_absolute_graphic_paths']]),check
assert images_ok and check['bib_keys_match'] and check['desktop_compilation']=='success'
print(json.dumps(check,ensure_ascii=False,indent=2))
