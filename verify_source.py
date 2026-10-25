"""Verify the published archive and parse its Python files without executing them."""
from pathlib import Path
import ast
import hashlib
import json
import zipfile

base = Path(__file__).resolve().parent
checksum, archive_name = (base / 'SHA256SUMS.txt').read_text().strip().split(None, 1)
archive = base / archive_name.strip()
assert hashlib.sha256(archive.read_bytes()).hexdigest() == checksum, 'Archive digest mismatch'
manifest = json.loads((base / 'source_manifest.json').read_text())
with zipfile.ZipFile(archive) as source:
    assert set(source.namelist()) == {item['path'] for item in manifest['files']}
    for item in manifest['files']:
        content = source.read(item['path'])
        assert hashlib.sha256(content).hexdigest() == item['sha256'], item['path']
        ast.parse(content.decode('utf-8'), filename=item['path'])
print(f"Verified archive integrity and syntax for {manifest['file_count']} Python files.")
print('This check does not execute the collection or inference pipeline.')
