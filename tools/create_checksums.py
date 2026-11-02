"""Record the release payload after intentional metadata edits; does not change data."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
excluded={'generated','.git','.venv','__pycache__'}
entries={}
for p in sorted(ROOT.rglob('*')):
 rel=p.relative_to(ROOT)
 if not p.is_file() or any(s in excluded for s in rel.parts) or p.name in {'checksums.json','.DS_Store'} or p.suffix in {'.pyc','.pyo'}:continue
 entries[rel.as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT/'checksums.json').write_text(json.dumps(entries,indent=2)+'\n')
print('Recorded',len(entries),'release files in checksums.json')
