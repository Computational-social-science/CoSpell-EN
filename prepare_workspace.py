"""Verify the release and prepare a source workspace for a chosen POS profile.

This tool copies source files. It does not run collection, inference or validation.
Python 3.10+; no third-party packages required.
"""
from pathlib import Path, PurePosixPath
import argparse
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def inside(root, relative):
    """Resolve a portable manifest path without allowing traversal or symlinks out."""
    name = PurePosixPath(relative)
    if not relative or '\\' in relative or ':' in relative or name.is_absolute() or '..' in name.parts:
        raise ValueError(f'Unsafe relative path: {relative!r}')
    result = (root / Path(*name.parts)).resolve()
    if not result.is_relative_to(root.resolve()) or result == root.resolve():
        raise ValueError(f'Path escapes workspace: {relative!r}')
    return result


def read_release(root=ROOT):
    manifest = json.loads((root/'config/source_map.json').read_text(encoding='utf-8'))
    profiles = json.loads((root/'config/profiles.json').read_text(encoding='utf-8'))
    if manifest['format_version'] != 1:
        raise ValueError('Unsupported source-map version')
    contents = {}
    for entry in manifest['files']:
        name = entry['path']
        if name in contents:
            raise ValueError(f'Duplicate source: {name}')
        data = inside(root, name).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f'Source checksum mismatch: {name}')
        ast.parse(data, filename=name)
        contents[name] = data
    all_files = profiles['all']['files']
    expected = {target:entry['path'] for entry in manifest['files']
                for target in entry['original_locations']}
    if all_files != expected:
        raise ValueError('Full profile and source provenance disagree')
    for profile, info in profiles.items():
        seen = set()
        for target, source in info['files'].items():
            path = inside(root, target)
            folded = str(path).casefold()
            if folded in seen:
                raise ValueError(f'Conflicting output paths in profile {profile}')
            seen.add(folded)
            if source not in contents or all_files.get(target) != source:
                raise ValueError(f'Unknown source mapping in profile {profile}: {target}')
    return manifest, profiles, contents


def prepare(profile, output, root=ROOT):
    manifest, profiles, contents = read_release(root)
    if profile not in profiles:
        raise ValueError(f'Unknown profile: {profile}')
    output = Path(output).resolve()
    mapping = profiles[profile]['files']
    # Validate every destination before creating anything. Refuse overwrite.
    destinations = [(inside(output,target), contents[source])
                    for target,source in mapping.items()]
    output.mkdir(parents=True, exist_ok=False)
    for path,data in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    record = {'profile':profile, 'baseline_commit':manifest['baseline_commit'],
              'source_file_count':len(mapping), 'pipeline_executed':False}
    (output/'workspace_info.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check checksums, syntax and profile mappings only')
    parser.add_argument('--list', action='store_true', help='List available part-of-speech profiles')
    parser.add_argument('--profile', choices=['all','noun','verb','adjective','adverb'], default='all')
    parser.add_argument('--output', type=Path, help='New directory for the prepared source workspace')
    args = parser.parse_args()
    try:
        manifest, profiles, _ = read_release()
        if args.list:
            for name,info in profiles.items():
                print(f"{name}: {len(info['files'])} module locations")
        if args.check:
            print(f"Verified {len(manifest['files'])} scientific source files and {len(profiles)} profiles.")
        if args.output:
            record = prepare(args.profile, args.output)
            print(f"Prepared {record['source_file_count']} source locations in {args.output.resolve()}")
            print('No pipeline was executed. Configure external inputs and runtime state as described in docs/reproduction.md.')
        elif not (args.check or args.list):
            parser.error('Specify --check, --list, or --output PATH')
    except (ValueError, OSError, KeyError, SyntaxError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
