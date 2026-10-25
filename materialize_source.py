"""Restore shared source modules to their original branch locations."""
from pathlib import Path
import argparse
import json
import zipfile
import hashlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True, help='A new, empty destination directory')
args = parser.parse_args()
base = Path(__file__).resolve().parent
layout = json.loads((base / 'layout_manifest.json').read_text())
manifest = json.loads((base / 'source_manifest.json').read_text())
hashes = {x['path']: x['sha256'] for x in manifest['files']}
destination = args.output.resolve()
if destination.exists():
    raise SystemExit('Output already exists. Select a new directory to avoid overwriting files.')
with zipfile.ZipFile(base / 'CoSpell-EN_source_20261025.zip') as archive:
    for target, source in layout.items():
        path = (destination / target).resolve()
        if not path.is_relative_to(destination):
            raise SystemExit('Invalid destination in layout manifest')
        data = archive.read(source)
        if hashlib.sha256(data).hexdigest() != hashes[source]:
            raise SystemExit('Source digest mismatch: ' + source)
    for target, source in layout.items():
        path = destination / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.read(source))
print(f'Restored {len(layout)} module locations from {len(hashes)} unique source files.')
