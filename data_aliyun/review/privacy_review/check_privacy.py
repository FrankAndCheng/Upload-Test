"""Scan local files and expanded ZIP members without printing matched secrets."""
from pathlib import Path
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PATTERNS = {
    'personal_home': re.compile(rb'/(?:Users|home)/[^/\s\x00]+|[A-Za-z]:\\(?:Users|Documents and Settings)\\', re.I),
    'escaped_home': re.compile(rb'\\/Users\\/|\\/home\\/', re.I),
    'private_temp': re.compile(rb'/private/var/fo[l]ders/', re.I),
    'private_key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}
# Also catch the local account name outside a path. Do not store it in the report.
account = Path.home().name.encode()
if len(account) >= 4:
    PATTERNS['local_account'] = re.compile(re.escape(account), re.I)


def scan():
    findings = []
    files = 0
    members = 0
    errors = []
    def inspect(name, data):
        hits = {key: len(pattern.findall(data)) for key, pattern in PATTERNS.items()
                if pattern.search(data)}
        if hits:
            findings.append({'file': name, 'categories': hits})
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or '.git' in p.parts:
            continue
        name = p.relative_to(ROOT).as_posix()
        try:
            data = p.read_bytes()
            inspect(name, data)
            files += 1
            if zipfile.is_zipfile(p):
                with zipfile.ZipFile(p) as archive:
                    for info in archive.infolist():
                        inspect(name+'!'+info.filename, archive.read(info))
                        inspect(name+'!member-name', info.filename.encode())
                        inspect(name+'!member-comment', info.comment)
                        members += 1
                    inspect(name+'!archive-comment', archive.comment)
                    assert archive.testzip() is None
        except (OSError, zipfile.BadZipFile, RuntimeError) as error:
            errors.append({'file': name, 'error_type': type(error).__name__})
    return {'files_scanned': files, 'archive_members_scanned': members,
            'findings': findings, 'errors': errors, 'passed': not findings and not errors}

if __name__ == '__main__':
    result = scan()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
