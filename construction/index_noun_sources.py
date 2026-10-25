"""Stream the lossless noun source into a derived, relation-aware read index."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import ijson


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


class UniqueDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise ValueError('Duplicate JSON key: ' + str(key))
        super().__setitem__(key, value)


def unique_object(pairs):
    obj = UniqueDict()
    for key, value in pairs:
        obj[key] = value
    return obj


def header(path):
    builder = ijson.common.ObjectBuilder(map_type=UniqueDict)
    with Path(path).open('rb') as f:
        for prefix, event, value in ijson.parse(f, use_float=True):
            if prefix == '' and event == 'map_key' and value == 'ngrams_data':
                builder.event('end_map', None)
                return builder.value
            builder.event(event, value)
    raise ValueError('Missing ngrams_data')


def write_json(path, obj):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean', type=Path, required=True)
    ap.add_argument('--timeseries', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    stamp = {str(p): {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns, 'sha256': digest(p)} for p in (args.clean, args.timeseries)}
    report_path = out / 'source_index_report.json'
    if report_path.exists():
        old = json.loads(report_path.read_text())
        assert old['inputs'] == stamp and old['status'] == 'passed'
        print(json.dumps({'status': 'index_already_verified', 'counts': old['counts']}), flush=True)
        return
    roots = json.loads(args.clean.read_text(), object_pairs_hook=unique_object)
    assert isinstance(roots, dict) and roots
    for root, variants in roots.items():
        assert type(root) is str and root and root.strip() == root
        assert isinstance(variants, list) and all(type(w) is str and w and w == w.strip() for w in variants)
        assert len(variants) == len(set(variants)) and root not in variants, root
    meta = header(args.timeseries)
    assert meta['schema_version'] == 'lossless-ngram-records-v1'
    years = meta['years_axis']
    assert years and years == sorted(set(years)) and all(type(y) is int for y in years)
    path = out / 'source_index.sqlite'
    if path.exists():
        raise FileExistsError('Partial index requires explicit inspection: ' + str(path))
    db = sqlite3.connect(path)
    db.executescript('''PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA cache_size=-131072;
      CREATE TABLE source_keys(word TEXT PRIMARY KEY);
      CREATE TABLE records(source_key TEXT, ordinal INTEGER, word TEXT, kind TEXT, parent TEXT, series TEXT, series_sha256 TEXT, record_sha256 TEXT, metadata TEXT, PRIMARY KEY(source_key,ordinal));
      CREATE TABLE wanted(root TEXT, word TEXT, kind TEXT, PRIMARY KEY(root,word,kind));''')
    for root, variants in roots.items():
        db.execute('INSERT INTO wanted VALUES (?,?,?)', (root, root, 'SPELLING'))
        db.executemany('INSERT INTO wanted VALUES (?,?,?)', ((root, word, 'MISSPELLING') for word in variants))
    db.commit()
    counts = Counter()
    begin = time.monotonic()
    with args.timeseries.open('rb') as f:
        for key, bucket in ijson.kvitems(f, 'ngrams_data', use_float=True, map_type=UniqueDict):
            assert type(key) is str and isinstance(bucket, list) and bucket, key
            db.execute('INSERT INTO source_keys VALUES (?)', (key,))
            for ordinal, wrapper in enumerate(bucket):
                entry = wrapper['record']
                assert entry['ngram'] == key, key
                kind, parent = entry['type'], entry['parent_ngram']
                assert kind in ('SPELLING', 'MISSPELLING')
                assert parent is None if kind == 'SPELLING' else type(parent) is str and bool(parent)
                series = entry['timeseries']
                assert isinstance(series, list) and len(series) == len(years), key
                assert all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in series), key
                raw = canonical(series)
                metadata = {k: v for k, v in wrapper.items() if k != 'record'}
                metadata['record_metadata'] = {k: v for k, v in entry.items() if k != 'timeseries'}
                db.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)', (key, ordinal, key, kind, parent or '', raw, hashlib.sha256(raw.encode()).hexdigest(), hashlib.sha256(canonical(entry).encode()).hexdigest(), canonical(metadata)))
                counts['records'] += 1
                counts[kind] += 1
            counts['surfaces'] += 1
            if counts['surfaces'] % 20000 == 0:
                db.commit()
                print(json.dumps({'phase': 'indexing', **counts, 'elapsed_seconds': round(time.monotonic()-begin, 1)}), flush=True)
    db.commit()
    db.execute('CREATE INDEX records_relation ON records(word,kind,parent)')
    db.execute('CREATE INDEX records_parent ON records(parent,kind,word)')
    db.execute('CREATE INDEX wanted_surface ON wanted(word,kind,root)')
    db.commit()
    conflicts = list(db.execute('SELECT word,kind,parent,count(*),count(DISTINCT series_sha256) FROM records GROUP BY word,kind,parent HAVING count(DISTINCT series_sha256)>1'))
    repeated = db.execute('SELECT count(*) FROM (SELECT 1 FROM records GROUP BY word,kind,parent HAVING count(*)>1)').fetchone()[0]
    missing = list(db.execute("SELECT w.root,w.word,w.kind FROM wanted w WHERE NOT EXISTS (SELECT 1 FROM records r WHERE r.word=w.word AND r.kind=w.kind AND r.parent=CASE WHEN w.kind='SPELLING' THEN '' ELSE w.root END)"))
    extra = db.execute("SELECT count(*) FROM records r WHERE NOT EXISTS (SELECT 1 FROM wanted w WHERE w.word=r.word AND w.kind=r.kind AND (r.parent=w.root OR (r.kind='SPELLING' AND w.root=r.word)))").fetchone()[0]
    # Same surface under different parents can legitimately preserve different historical inputs.
    cross = db.execute('SELECT count(*) FROM (SELECT word FROM records GROUP BY word HAVING count(DISTINCT series_sha256)>1)').fetchone()[0]
    assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.close()
    for p in (args.clean, args.timeseries):
        assert stamp[str(p)]['bytes'] == p.stat().st_size and stamp[str(p)]['mtime_ns'] == p.stat().st_mtime_ns
        assert stamp[str(p)]['sha256'] == digest(p), 'Input changed: ' + str(p)
    counts.update({'roots': len(roots), 'relations': sum(map(len, roots.values())), 'repeated_relation_keys': repeated, 'conflicting_relation_keys': len(conflicts), 'missing_relation_keys': len(missing), 'unused_source_records': extra, 'surfaces_with_different_context_series': cross})
    report = {'status': 'passed', 'inputs': stamp, 'header': meta, 'counts': dict(counts), 'conflicts': conflicts, 'missing': missing, 'source_index_path': str(path), 'source_unchanged': True, 'created_at_unix': time.time(), 'script_sha256': digest(__file__)}
    write_json(report_path, report)
    print(json.dumps({'status': 'index_complete', 'counts': counts, 'conflict_examples': conflicts[:10], 'missing_examples': missing[:10]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

