"""PDF交付复核：源内容保留、页面边界、字体、链接和编译诊断。"""
from pathlib import Path
import collections
import difflib
import hashlib
import json
import re
import zipfile
import pdfplumber
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
EDITOR = ROOT / 'research_analysis/stage3'
PDF = EDITOR / 'paper_body.pdf'
before = zipfile.ZipFile(OUT / 'source_before_layout.zip')
equations = 0
table_checks = []
changed_content = []
diff = []

def normalize_layout(s):
    s = re.sub(r'\\(?:begingroup|endgroup|raggedright|par|clearpage)\b', '', s)
    return re.sub(r'\s+', '', s.replace(r'\\[0.25em]', ''))

for name in before.namelist():
    old = before.read(name).decode()
    new = (EDITOR/name).read_text()
    old_eq = re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}', old, re.S)
    new_eq = re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}', new, re.S)
    assert old_eq == new_eq, name
    equations += len(new_eq)
    if name.startswith('paper/tables/'):
        rows = lambda s: [x.strip() for x in s.splitlines() if '&' in x and x.strip().endswith(r'\\')]
        table_checks.append(dict(file=name, rows_and_headers_unchanged=rows(old) == rows(new)))
    elif name != 'paper_body.tex' and normalize_layout(old) != normalize_layout(new):
        changed_content.append(name)
    diff.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='before/'+name, tofile='after/'+name))
(OUT/'layout_changes.patch').write_text(''.join(diff))
manifest = json.loads((OUT/'before_manifest.json').read_text())
images_unchanged = all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in manifest.items() if k.endswith('.png'))

reader = PdfReader(PDF)
pages = []
fonts = {}
broken_links = []
named = reader.named_destinations
with pdfplumber.open(PDF) as document:
    for i, page in enumerate(document.pages, 1):
        text = page.extract_text() or ''
        chars = page.chars
        outside = [c for c in chars if c['x0'] < -1 or c['x1'] > page.width+1 or c['top'] < -1 or c['bottom'] > page.height+1]
        footer = ''.join(c['text'] for c in chars if c['top'] > 790).strip()
        pages.append(dict(page=i, A4=abs(page.width-595.28)<1 and abs(page.height-841.89)<1,
                          extracted_chars=len(chars), outside_page_chars=len(outside),
                          unresolved_marker='??' in text or '[?]' in text,
                          footer_page_number=footer, footer_correct=footer==str(i),
                          figures=len(page.images)))
        for font_obj in reader.pages[i-1].get('/Resources',{}).get('/Font',{}).values():
            font = font_obj.get_object()
            name = str(font.get('/BaseFont','unknown'))
            child = font.get('/DescendantFonts',[font])[0].get_object()
            desc = child.get('/FontDescriptor',{}).get_object()
            fonts[name] = dict(name=name, embedded=any(k in desc for k in ['/FontFile','/FontFile2','/FontFile3']))
        for item in reader.pages[i-1].get('/Annots',[]):
            annot = item.get_object()
            dest = annot.get('/Dest')
            action = annot.get('/A',{}).get_object()
            if action.get('/S') == '/GoTo':
                dest = action.get('/D')
            if isinstance(dest, str) and dest not in named:
                broken_links.append(dict(page=i,destination=dest))
log = (EDITOR/'paper_body.log').read_text(errors='replace')
warnings = re.findall(r'^.*(?:Warning:|Underfull|Overfull|Missing character|^!).*$', log, re.M)
summary = dict(pages=len(reader.pages), PDF_SHA256=hashlib.sha256(PDF.read_bytes()).hexdigest(),
               PDF_bytes=PDF.stat().st_size, numbered_equations_unchanged=equations,
               tables_checked=len(table_checks), table_checks=table_checks,
               changed_scientific_content=changed_content, all_figure_assets_unchanged=images_unchanged,
               fonts=list(fonts.values()), all_fonts_embedded=all(x['embedded'] for x in fonts.values()),
               broken_internal_links=broken_links, named_destinations=len(named),
               compilation_warnings=warnings, compilation_success='Output written on paper_body.pdf' in log,
               page_checks=pages, renderer='PDFium; Poppler initial rendering could not resolve its CJK data path.',
               visual_review_complete=False, official_template_verified=False)
(OUT/'pdf_checks.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
assert equations==23 and len(table_checks)==19
assert not changed_content and images_unchanged and all(x['rows_and_headers_unchanged'] for x in table_checks)
assert not warnings and not broken_links and summary['all_fonts_embedded']
assert all(p['A4'] and p['outside_page_chars']==0 and not p['unresolved_marker'] and p['footer_correct'] for p in pages), pages
print(json.dumps({k:summary[k] for k in ['pages','PDF_SHA256','PDF_bytes','numbered_equations_unchanged','tables_checked','changed_scientific_content','all_figure_assets_unchanged','all_fonts_embedded','named_destinations','broken_internal_links','compilation_warnings','compilation_success']},ensure_ascii=False,indent=2))
