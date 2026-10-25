from model_runtime_client import masked_scores
from rwse_preparation import prepared
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
import re
import time

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from experiment_common import BASE, CORPUS_ID, EXPERIMENT, Cache, chunks, progress, read_jsonl, write_json, write_jsonl

MU = 10
SENSITIVITY_MU = (5, 10, 20)


def single_token_id(tokenizer, word):
    """Return only whole-word, nonspecial single-token vocabulary entries."""
    tokens = tokenizer.encode(word, add_special_tokens=False)
    if (len(tokens) == 1 and tokens[0] not in tokenizer.all_special_ids
            and not tokenizer.convert_ids_to_tokens(tokens[0]).startswith('##')):
        return tokens[0]
    return None


def overlaps_target(span, target_span):
    return span[0] < target_span[1] and span[1] > target_span[0]


def flags_by_mu(findings):
    return {str(mu): any(f['log_probability_ratio'] >= math.log(mu) for f in findings)
            for mu in SENSITIVITY_MU}


def main():
    import torch
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['pilot', 'full'], required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    begin = time.monotonic()
    torch.set_num_threads(8)
    records = read_jsonl(('pre_rwse_pass_' if (BASE/('pre_rwse_pass_' + args.phase + '.jsonl')).exists() else 'gec_pass_') + args.phase + '.jsonl')
    tokenizer = AutoTokenizer.from_pretrained(BASE / 'models/bert-base-cased', local_files_only=True)
    source = EXPERIMENT.parent / 'organized_adjectives_wordnet_20261015/final_adversarial_adjectives_organized.json'
    def build_preparation():
        graph = defaultdict(set)
        groups = list(csv.reader((BASE / 'paper_confusion_sets_modified.csv').open()))
        for words in groups:
            for word in words:
                graph[word].update(set(words) - {word})
        with source.open() as stream:
            dataset = json.load(stream)
        dataset_pairs = 0
        for root in dataset:
            for variant in root['variants']:
                if variant['vocab_type'] != 'IV':
                    continue
                dataset_pairs += 1
                alternatives = {surface for word in (root['lemma'],variant['word'])
                                for surface in (word,word.lower(),word.capitalize())}
                for word in alternatives:
                    graph[word].update(alternatives - {word})
        del dataset
        ids = {}
        for word in graph:
            token_id = single_token_id(tokenizer, word)
            if token_id is not None:
                ids[word] = token_id
        return graph,groups,dataset_pairs,ids
    graph,groups,dataset_pairs,ids=prepared(EXPERIMENT,source,tokenizer,build_preparation)
    config = {'corpus_id': CORPUS_ID, 'model': 'google-bert/bert-base-cased', 'revision': 'cd5ef92a9fb2f889e972770a36d4ed042daf221e',
              'dtype': 'float32', 'mu': MU, 'sensitivity_mu': list(SENSITIVITY_MU),
              'confusion_source': 'paper_confusion_sets_modified.csv + clean IV pairs',
              'paper_sets': len(groups), 'dataset_iv_pairs': dataset_pairs,
              'target_position_excluded': True, 'single_token_only': True}
    cache = Cache('rwse_cache', config)
    pending = []
    tasks = []
    analyses = {}
    for r in records:
        key = json.dumps([r['root_sentence'], r['target_start'], len(r['replacement_surface'])])
        if cache.get(key) is not None or key in analyses:
            continue
        encoded = tokenizer(r['root_sentence'], return_offsets_mapping=True, add_special_tokens=True)
        result = {'supported_positions': 0, 'unsupported_confusion_positions': 0,
                  'non_confusion_positions': 0, 'findings': [], 'input_tokens': len(encoded['input_ids'])}
        analyses[key] = result
        pending.append(key)
        if len(encoded['input_ids']) > 512:
            result['error'] = 'bert_input_too_long'
            continue
        positions = {tuple(offset): i for i, offset in enumerate(encoded['offset_mapping']) if offset[0] != offset[1]}
        target_span = (r['target_start'], r['target_start'] + len(r['replacement_surface']))
        assert r['root_sentence'][target_span[0]:target_span[1]] == r['replacement_surface']
        for match in re.finditer(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)?", r['root_sentence']):
            if overlaps_target(match.span(), target_span):
                continue
            word = match.group()
            if word not in graph:
                result['non_confusion_positions'] += 1
                continue
            alternatives = sorted(c for c in graph[word] if c in ids)
            if word not in ids or match.span() not in positions or not alternatives:
                result['unsupported_confusion_positions'] += 1
                continue
            position = positions[match.span()]
            if encoded['input_ids'][position] != ids[word]:
                result['unsupported_confusion_positions'] += 1
                continue
            result['supported_positions'] += 1
            masked = list(encoded['input_ids'])
            masked[position] = tokenizer.mask_token_id
            tasks.append({'key': key, 'input_ids': masked, 'mask_position': position, 'word': word,
                          'start': match.start(), 'end': match.end(), 'alternatives': alternatives})
    tasks.sort(key=lambda t: len(t['input_ids']))
    progress('rwse_' + args.phase, total_masked_positions=len(tasks), processed=0)
    processed = 0
    for batch in chunks(tasks, args.batch_size):
        inputs = tokenizer.pad([{'input_ids': task['input_ids']} for task in batch], return_tensors='pt')
        logits = masked_scores(inputs,batch,ids,BASE)
        for task, scores in zip(batch, logits):
            alternatives = task['alternatives']
            chosen = max(alternatives, key=lambda w: float(scores[ids[w]]))
            difference = float(scores[ids[chosen]] - scores[ids[task['word']]])
            if difference >= math.log(5):
                analyses[task['key']]['findings'].append({'word': task['word'], 'suggestion': chosen,
                                                         'start': task['start'], 'end': task['end'],
                                                         'log_probability_ratio': difference})
        processed += len(batch)
        if processed % (args.batch_size * 10) == 0 or processed == len(tasks):
            progress('rwse_' + args.phase, total_masked_positions=len(tasks), processed=processed)
    for key in pending:
        cache.put(key, analyses[key])
    cache.flush()
    accepted, decisions = [], []
    sensitivity = Counter()
    for r in records:
        key = json.dumps([r['root_sentence'], r['target_start'], len(r['replacement_surface'])])
        result = cache.get(key)
        flags = flags_by_mu(result['findings'])
        for mu, flag in flags.items():
            sensitivity[mu] += int(flag)
        reason = result.get('error') or ('residual_rwse_flag' if flags['10'] else 'no_supported_rwse_flag')
        passed = not result.get('error') and not flags['10']
        decisions.append({'sid': r['sid'], 'record_id': r['record_id'], 'pass': passed, 'reason': reason,
                          'flags_by_mu': flags, **result})
        if passed:
            accepted.append(r)
    summary = {'phase': args.phase, 'config': config, 'input': len(records), 'accepted': len(accepted),
               'reasons': dict(Counter(d['reason'] for d in decisions)), 'flagged_records_by_mu': dict(sensitivity),
               'unsupported_confusion_positions': sum(d['unsupported_confusion_positions'] for d in decisions),
               'supported_positions': sum(d['supported_positions'] for d in decisions),
               'limitations': 'Only single-token members of the configured confusion sets are scored. Uncovered positions are not certified error-free; Qwen review is disabled by user; unsupported positions and contextual POS are not independently certified.',
               'elapsed_seconds': time.monotonic() - begin}
    write_jsonl('rwse_pass_' + args.phase + '.jsonl', accepted)
    write_jsonl('rwse_decisions_' + args.phase + '.jsonl', decisions)
    write_json('rwse_report_' + args.phase + '.json', summary)
    progress('rwse_' + args.phase, state='complete', **summary)


if __name__ == '__main__':
    main()
