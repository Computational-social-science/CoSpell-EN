"""User-authorized GEC/RWSE acceptance; Qwen is not evaluated for new records."""
POLICY = 'gec_rwse_without_qwen_20261022'


def tag_record(record):
    assert record['candidate_kind'] in ('natural', 'seed')
    assert record.get('is_synthetic') is False
    return dict(record, acceptance_policy=POLICY, gec_evaluated=True,
                rwse_evaluated=True, qwen_evaluated=False,
                independent_review_evaluated=False, contextual_pos_evaluated=False,
                pair_origin=('root_seed_gec_rwse' if record['candidate_kind'] == 'seed'
                             else 'natural_gec_rwse'))


def required_stages(marker):
    policy = marker.get('model_acceptance_policy')
    if policy == POLICY:
        assert marker.get('qwen_evaluated') is False
        assert marker.get('independent_review_evaluated') is False
        assert marker.get('accepted_records_file') == 'accepted_full.jsonl'
        return ['gec', 'rwse']
    assert policy is None, 'Unknown model acceptance policy'
    return ['gec', 'rwse', 'qwen']


def seed_evidence_files(seed):
    if seed.get('acceptance_policy') == POLICY:
        return {'round_complete.json', 'gec_decisions_full.jsonl',
                'rwse_decisions_full.jsonl', 'accepted_full.jsonl'}
    return {'gec_decisions_full.jsonl', 'rwse_decisions_full.jsonl',
            'qwen_decisions_full.jsonl', 'qwen_pass_full.jsonl'}


def verify_record(record, marker, rwse_output, qwen_decisions):
    assert required_stages(marker) == ['gec', 'rwse']
    assert record['sid'] not in qwen_decisions
    expected = tag_record(rwse_output[record['sid']])
    stored = {k: v for k, v in record.items()
              if k not in ('validated_in_round', 'validation_experiment')}
    assert stored == expected, 'GEC/RWSE-only provenance mismatch'
