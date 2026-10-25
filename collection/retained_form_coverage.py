"""Retain explicitly audited historical ADV pairs; do not reopen filled pairs."""
from pathlib import Path
import hashlib
import json

POLICY = 'adv_retain_historical_model_qualified_pairs_20261020'
_cache = {}


def record_hash(record):
    return hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def manifest(directory):
    directory = Path(directory)
    path = directory/'retained_coverage.json'
    if not path.exists():
        return None
    key = str(directory)
    if key not in _cache:
        validation = json.loads((directory/'scheduling_validation.json').read_text())
        expected = validation['existing_coverage_update']
        assert expected['policy'] == POLICY
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected['retained_coverage_sha256']
        value = json.loads(raw)
        assert value['policy'] == POLICY
        assert value['new_no_gec_or_direct_constructions_not_restored'] is True
        _cache[key] = value
    return _cache[key]


def historical_record_allowed(record, directory, round_name=None):
    value = manifest(directory)
    if value is None:
        return False
    entry = value['historical_records'].get(record['sid'])
    if entry is None:
        return False
    assert record['is_synthetic'] is True and record['candidate_kind'] == 'synthetic'
    assert not record.get('acceptance_policy') and not record.get('scan_gate')
    expected = dict(record)
    if round_name is not None:
        assert round_name == entry['round']
        expected['validated_in_round'] = round_name
    assert expected['record_id'] == entry['record_id']
    assert expected['validated_in_round'] == entry['round']
    assert record_hash(expected) == entry['record_sha256']
    # Only the exact original accepted record is exempt from a later scan gate.
    # Final validation still replays all three original model caches and source.
    return True


def baseline_selection(directory):
    value = manifest(directory)
    return None if value is None else value['baseline_indices']
