"""Read-only structural checks for the prepared research repository (stdlib only)."""
from pathlib import Path
import argparse, hashlib, json, re, sys
from urllib.parse import unquote, urlsplit
ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {'.git', '.venv', '__pycache__', 'generated'}
JP = re.compile(r'[\u3040-\u30ff\u3400-\u9fff]')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def check(release=False):
    errors = []
    files = [p for p in ROOT.rglob('*') if p.is_file() and not EXCLUDE.intersection(p.relative_to(ROOT).parts) and p.name != '.DS_Store' and p.suffix not in {'.pyc', '.pyo'}]
    required = ['README.md', 'CONTRIBUTING.md', '.gitignore', '.gitattributes', 'requirements-analysis.txt', 'docs/REVIEWER_GUIDE.md', 'docs/INDEX.md', 'docs/REPRODUCTION.md', 'docs/RELEASE.md', 'tools/reproduce.py', 'provenance/documentation_translations.json']
    for rel in required:
        if not (ROOT / rel).is_file():
            errors.append('Missing required file: ' + rel)
    exceptions = json.loads((ROOT / 'provenance/preserved_language_records.json').read_text())
    exempt = {r['path'] for r in exceptions}
    for r in exceptions:
        p = ROOT / r['path']
        if not p.is_file() or sha(p) != r['sha256']:
            errors.append('Preserved source record changed: ' + r['path'])
    links = 0
    docs = 0
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        if p.is_symlink():
            errors.append('Symlink in release: ' + rel)
        if p.stat().st_size >= 100 * 1024**2:
            errors.append('File reaches GitHub normal-Git size limit: ' + rel)
        if p.suffix not in {'.md', '.txt', '.html'} or rel in exempt or '/external/' in rel or rel.startswith('licenses/'):
            continue
        text = p.read_text(encoding='utf-8')
        docs += 1
        if JP.search(text):
            errors.append('Untranslated reading document: ' + rel)
        # Check explicit local Markdown/HTML destinations. Archived originals are exempt.
        destinations = re.findall(r'\]\(([^)]+)\)', text) if p.suffix == '.md' else []
        if p.suffix == '.html':
            destinations += re.findall(r'href=[\"\x27]([^\"\x27]+)', text)
        for dest in destinations:
            dest = dest.strip().split(' "')[0].strip('<>')
            if not dest or dest.startswith('#') or urlsplit(dest).scheme:
                continue
            target = (p.parent / unquote(dest.split('#')[0].split('?')[0])).resolve()
            links += 1
            if not target.exists():
                errors.append('Broken local link: ' + rel + ' -> ' + dest)
    translations = json.loads((ROOT / 'provenance/documentation_translations.json').read_text())
    for item in translations:
        p = ROOT / item['english']
        if not p.is_file() or sha(p) != item['english_sha256']:
            errors.append('English document hash mismatch: ' + item['english'])
        if item['original_retained']:
            p = ROOT / item['original']
            if not p.is_file() or sha(p) != item['original_sha256']:
                errors.append('Original translation source changed: ' + item['original'])
    pending = []
    citation = ROOT / 'CITATION.cff'
    if not citation.is_file():
        pending.append('Complete actual author/repository/version metadata and activate CITATION.cff.')
    elif re.search(r'REPLACE_WITH|OWNER/REPOSITORY|YOUR_ACCOUNT|ADD_ACTUAL', citation.read_text()):
        pending.append('Replace placeholders in CITATION.cff with confirmed metadata.')
    if not any((ROOT / name).is_file() for name in ['LICENSE', 'LICENSE.md', 'LICENSE.txt']):
        pending.append('Declare approved author-owned reuse terms; retain third-party restrictions.')
    if release:
        errors += pending
    result = dict(passed=not errors, english_reading_documents_checked=docs, local_links_checked=links,
                  preserved_language_records_verified=len(exceptions), files=len(files),
                  maximum_file_bytes=max(p.stat().st_size for p in files), errors=errors,
                  publication_metadata_pending=pending, release_mode=release)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', action='store_true', help='Also require final public citation/license metadata')
    args = parser.parse_args()
    sys.exit(check(args.release))
