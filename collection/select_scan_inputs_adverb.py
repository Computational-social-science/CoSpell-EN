"""Eight complete random C4 shards, then per-relation natural exhaustion.

The source plan is immutable. A started file, checkpoint, heartbeat, or empty
candidate queue is never evidence that a shard was scanned to EOF.
"""
from pathlib import Path
import hashlib
import json
import sqlite3

EPOCH = 'eight_complete_random_c4_shards_20261019'
ATTEMPT_POLICY = 'verified_root_direct_substitution_20261019:' + EPOCH

def read(path):
    return json.loads(Path(path).read_text())

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def state(directory):
    d = Path(directory)
    plan_path = d/'c4_scan_gate_plan.json'
    plan = read(plan_path)
    assert plan['policy'] == EPOCH and len(plan['required_files']) == 8
    assert len({f['path'] for f in plan['required_files']}) == 8
    p = d/'c4_scan_gate_complete.json'
    if not p.exists():
        return {'ready': False, 'required_shards': 8, 'natural_floor': 0}
    proof = read(p)
    assert proof['plan_sha256'] == digest(plan_path)
    assert proof['policy'] == EPOCH and proof['prefix_replay_complete'] is True
    assert len(proof['files']) == 8
    for expected, actual in zip(plan['required_files'], proof['files']):
        assert all(actual[k] == expected[k] for k in ['path', 'bytes', 'mtime_ns'])
        assert actual['complete_file'] is True and actual['scanned_to_eof'] is True
        assert actual['final_line'] > 0
        st = Path(expected['path']).stat()
        assert (st.st_size, st.st_mtime_ns) == (expected['bytes'], expected['mtime_ns'])
    return {'ready': True, 'required_shards': 8, 'natural_floor': proof['c4_high_rowid'],
            'certificate_sha256': digest(p), 'plan_sha256': proof['plan_sha256']}

def proof_for_pair(directory, owner, cursor, window):
    gate = state(directory)
    assert gate['ready'], 'eight_complete_shards_required'
    assert cursor >= window >= gate['natural_floor'], 'natural_candidates_not_exhausted'
    return {'policy': EPOCH, 'record_id': owner, 'plan_sha256': gate['plan_sha256'],
            'certificate_sha256': gate['certificate_sha256'],
            'natural_cursor': cursor, 'natural_window_end': window,
            'required_natural_floor': gate['natural_floor']}

def validate_record(record, directory, check_queue=False):
    from retained_coverage import historical_record_allowed
    if not record.get('scan_gate') and historical_record_allowed(record, directory):
        return True
    proof = record.get('scan_gate')
    assert proof, 'missing_eight_shard_and_natural_exhaustion_proof'
    if (Path(directory)/'shared_stream_source.json').exists():
        from stream_blocks import validate_exhaustion
        validate_exhaustion(record, directory)
    gate = state(directory)
    assert gate['ready'], 'eight_complete_shards_required'
    assert proof['policy'] == EPOCH and proof['record_id'] == record['record_id']
    assert proof['plan_sha256'] == gate['plan_sha256']
    assert proof['certificate_sha256'] == gate['certificate_sha256']
    assert proof['required_natural_floor'] == gate['natural_floor']
    assert proof['natural_cursor'] >= proof['natural_window_end'] >= gate['natural_floor']
    if check_queue:
        db = sqlite3.connect('file:'+str(Path(directory)/'candidate_buffer.sqlite')+'?mode=ro', uri=True)
        try:
            key = json.dumps(('natural', record['record_id'], 'c4'), separators=(',', ':'))
            row = db.execute('SELECT position FROM cursors WHERE job=?', (key,)).fetchone()
            assert row and row[0] >= proof['natural_cursor']
            assert not db.execute("SELECT 1 FROM pending WHERE owner=? AND kind='pairs' AND json_extract(record,'$.candidate_kind')='natural' LIMIT 1", (record['record_id'],)).fetchone(), 'pending_natural_candidate'
        finally:
            db.close()
    return True

def selected_baseline(directory, variant):
    """Preserve the original baseline; select one proven natural pair for export."""
    indices = baseline_indices(directory)
    index = indices.get(variant['id'])
    return [] if index is None else [variant['parallel_corpus'][index]]

_baseline_cache = {}
def baseline_indices(directory):
    from retained_coverage import baseline_selection
    retained = baseline_selection(directory)
    if retained is not None:
        return retained
    key = str(directory)
    if key not in _baseline_cache:
        _baseline_cache[key] = read(Path(directory)/'baseline_natural_selection.json')['indices']
    return _baseline_cache[key]
