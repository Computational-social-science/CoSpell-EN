from review_policy import POLICY as MODEL_POLICY, required_stages, verify_record
"""Admit single-span substitutions of registered, independently validated seeds.

This is a provenance/structure gate, not a GEC result. Natural candidates and
new root seeds continue through the unchanged GEC pipeline.
"""
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from scan_gate import validate_record

POLICY = 'verified_root_direct_substitution_20261019'


def sid(*parts):
    return hashlib.sha256('\0'.join(map(str, parts)).encode()).hexdigest()


def surface_like(word, surface):
    if surface == surface.lower():
        return word.lower()
    if surface.isupper():
        return word.upper()
    if surface == surface.capitalize():
        return word.capitalize()
    return None


def validate_replacement(record, seed, multiple_targets=False):
    r, s = record, seed
    assert r['candidate_kind'] == 'synthetic' and r['is_synthetic'] is True
    assert s['candidate_kind'] == 'seed' and s['is_synthetic'] is False
    assert r['seed_id'] == s['sid'] == sid(s['root'], 'seed', s['root_sentence'])
    assert s['root'] == s['variant'] == r['root']
    assert s['root_sentence'] == s['variant_sentence'] == r['root_sentence']
    assert r['source'] == s['source']
    assert r['source']['source_sentence_hash'] == sid(r['root_sentence'])
    left, right = s['target_start'], s['target_end']
    assert type(left) is int and type(right) is int and 0 <= left < right <= len(r['root_sentence'])
    assert (r['source_target_start'], r['source_target_end']) == (left, right)
    source_surface = r['root_sentence'][left:right]
    assert source_surface == s['target_surface'] == s['replacement_surface'] == r['replacement_surface']
    assert source_surface.lower() == r['root'].lower()
    assert r['target_surface'] == surface_like(r['variant'], source_surface)
    assert r['target_start'] == left and r['target_end'] == left + len(r['target_surface'])
    assert r['variant_sentence'] == r['root_sentence'][:left] + r['target_surface'] + r['root_sentence'][right:]
    assert r['variant_sentence'] != r['root_sentence'], 'zero_edit'
    assert r['sid'] == sid(r['root'], r['variant'], r['variant_sentence'])
    assert r['variant_sentence'][r['target_start']:r['target_end']] == r['target_surface']
    pattern = r"(?<![\w'\u2019-])" + re.escape(r['variant']) + r"(?![\w'\u2019-])"
    hits = list(re.finditer(pattern, r['variant_sentence'], re.I))
    assert any(h.span() == (r['target_start'], r['target_end']) for h in hits)
    if not multiple_targets:
        assert len(hits) == 1, 'multiple_target_occurrences'
    assert r['language_score'] >= .9 and s['language_score'] >= .9


def partition_inputs(records, experiment, allowed_seed_bases, multiple_targets=False):
    exp = Path(experiment)
    allowed = {Path(p).resolve() for p in allowed_seed_bases}
    gec, direct, decisions = [], [], []
    synthetic = [r for r in records if r.get('candidate_kind') == 'synthetic' or r.get('is_synthetic')]
    if not synthetic:
        return list(records), direct, decisions
    db = sqlite3.connect('file:' + str(exp/'expansion_state.sqlite') + '?mode=ro', uri=True)
    seed_cache, round_cache = {}, {}
    try:
        for r in records:
            if r.get('candidate_kind') != 'synthetic' and not r.get('is_synthetic'):
                gec.append(r)
                continue
            decision = {'sid': r['sid'], 'record_id': r['record_id'], 'pass': False,
                        'gec_evaluated': False, 'rwse_evaluated': False, 'independent_review_evaluated': False,
                        'pair_origin': 'constructed_from_verified_root', 'acceptance_policy': POLICY}
            try:
                validate_record(r, exp, check_queue=True)
                key = r['seed_id']
                if key not in seed_cache:
                    row = db.execute('SELECT record FROM seeds WHERE sid=?', (key,)).fetchone()
                    assert row, 'seed_not_registered'
                    seed_cache[key] = json.loads(row[0])
                s = seed_cache[key]
                relation = db.execute('SELECT root,variant FROM pairs WHERE id=?', (r['record_id'],)).fetchone()
                assert relation == (r['root'], r['variant']), 'foreign_relation'
                validate_replacement(r, s, multiple_targets)
                base = Path(s.get('validation_experiment', str(exp))).resolve()
                assert base in allowed, 'foreign_seed_experiment'
                name = s['validated_in_round']
                assert Path(name).name == name and name not in ('.', '..')
                folder = base/'rounds'/name
                if folder not in round_cache:
                    complete = json.loads((folder/'round_complete.json').read_text())
                    assert complete['root_only'] is True
                    evidence, passed = {}, {}
                    stages = required_stages(complete)
                    for stage in stages:
                        path = folder/(stage+'_decisions_full.jsonl')
                        raw = path.read_bytes()
                        evidence[path.name] = hashlib.sha256(raw).hexdigest()
                        passed[stage] = {x['sid'] for line in raw.splitlines() if line.strip()
                                         for x in [json.loads(line)] if x['pass'] is True}
                    path = folder/('accepted_full.jsonl' if stages == ['gec', 'rwse'] else 'qwen_pass_full.jsonl')
                    raw = path.read_bytes()
                    evidence[path.name] = hashlib.sha256(raw).hexdigest()
                    output = {x['sid']: x for line in raw.splitlines() if line.strip() for x in [json.loads(line)]}
                    if stages == ['gec', 'rwse']:
                        raw = (folder/'round_complete.json').read_bytes()
                        evidence['round_complete.json'] = hashlib.sha256(raw).hexdigest()
                        rwse_output = {x['sid']: x for line in (folder/'rwse_pass_full.jsonl').read_text().splitlines()
                                       if line.strip() for x in [json.loads(line)]}
                        qwen = {x['sid']: x for line in (folder/'qwen_decisions_full.jsonl').read_text().splitlines()
                                if line.strip() for x in [json.loads(line)]}
                        for row in output.values():verify_record(row, complete, rwse_output, qwen)
                    round_cache[folder] = evidence, passed, output
                evidence, passed, output = round_cache[folder]
                assert all(key in values for values in passed.values()), 'seed_missing_model_acceptance'
                raw_seed = output[key]
                assert {k: v for k, v in s.items() if k not in ['validated_in_round', 'validation_experiment']} == raw_seed
                proof = {'seed_id': key, 'experiment': str(base), 'round': name, 'sha256': evidence}
                direct.append(dict(r, acceptance_policy=POLICY, gec_evaluated=False, rwse_evaluated=False, independent_review_evaluated=False,
                                   pair_origin='constructed_from_verified_root', root_validation=proof))
                decision.update({'pass': True, 'reason': 'validated_root_single_span_substitution', 'root_validation': proof})
            except (AssertionError, KeyError, TypeError, ValueError) as error:
                decision['reason'] = 'invalid_root_substitution_evidence:' + str(error)
            decisions.append(decision)
    finally:
        db.close()
    return gec, direct, decisions


def ensure_policy_attempts(db):
    # Preserve the old policy's attempts and decisions verbatim. A previously
    # model-rejected substitution is eligible once under the new construction policy.
    db.execute("""CREATE TABLE IF NOT EXISTS synthetic_policy_attempts(
        sid TEXT, record_id TEXT, policy TEXT, round_name TEXT, passed INTEGER,
        PRIMARY KEY(sid,policy))""")


def accepted_round_path(folder, marker):
    name = marker.get('accepted_records_file', 'qwen_pass_full.jsonl')
    assert name in ('qwen_pass_full.jsonl', 'accepted_full.jsonl')
    if name == 'accepted_full.jsonl':
        assert marker.get('synthetic_policy') == POLICY
    return Path(folder)/name
