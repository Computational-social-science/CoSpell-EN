from model_runtime_client import generate
import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
import time

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from experiment_common import BASE, CORPUS_ID, Cache, chunks, norm, progress, read_jsonl, write_json, write_jsonl

PREFIX = 'Fix grammatical errors in this sentence: '


def exact_correction_reason(result, expected, mismatch_reason):
    """Fail closed on missing, incomplete or nonexact whole-sentence model outputs."""
    if not isinstance(result, dict):
        return 'gec_response_missing_or_invalid'
    if result.get('error'):
        return result['error']
    if result.get('truncated') is not False:
        return 'output_truncated_or_unverified'
    if not isinstance(result.get('text'), str):
        return 'gec_response_missing_or_invalid'
    if norm(result['text']) != norm(expected):
        return mismatch_reason
    return None


class OriginalSentenceCaches:
    """Route speculative original-sentence outputs exactly as the three normal shards do."""
    def __init__(self,count,config):self.caches=[Cache(f'gec_cache_shard_{i}',config) for i in range(count)]
    def cache(self,text):return self.caches[int(hashlib.sha256(text.encode()).hexdigest(),16)%len(self.caches)]
    def get(self,text):return self.cache(text).get(text)
    def put(self,text,value):self.cache(text).put(text,value)
    def flush(self):
        for cache in self.caches:cache.flush()


def main():
    import torch
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['pilot', 'full'], required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--prefetch-input')
    parser.add_argument('--stop-file')
    args = parser.parse_args()
    assert 0 <= args.shard_index < args.num_shards
    assert args.num_shards == 1 or args.phase == 'full'
    label = args.phase if args.num_shards == 1 else f'{args.phase}_shard_{args.shard_index}'
    if args.num_shards > 1 and not args.prefetch_input:
        os.environ['EXPERIMENT_STATUS_SUFFIX'] = f'_shard_{args.shard_index}'
    begin = time.monotonic()
    torch.set_num_threads(8)
    torch.manual_seed(20261011)
    input_name = 'gec_input_full.jsonl' if args.phase == 'full' and (BASE/'gec_input_full.jsonl').exists() else 'candidates.jsonl'
    records = read_jsonl(args.prefetch_input or ('pilot_candidates.jsonl' if args.phase == 'pilot' else input_name))
    if args.num_shards > 1 and not args.prefetch_input:
        records = [r for r in records if int(hashlib.sha256(r['variant_sentence'].encode()).hexdigest(), 16) % args.num_shards == args.shard_index]
    config = {'corpus_id': CORPUS_ID, 'model': 'grammarly/coedit-large', 'revision': '5637bcdf9d8d4419f97c8cfea36f7d35c79232b6',
              'prefix': PREFIX, 'max_input_tokens': 256, 'max_new_tokens': 320, 'num_beams': 1,
              'do_sample': False, 'dtype': 'bfloat16', 'comparison': 'whitespace normalization only'}
    cache = OriginalSentenceCaches(args.num_shards,config) if args.prefetch_input else Cache('gec_cache' if args.num_shards == 1 else f'gec_cache_shard_{args.shard_index}', config)
    tokenizer=None

    def infer(texts, stage):
        nonlocal tokenizer
        todo = sorted({text for text in texts if cache.get(text) is None}, key=lambda s: (len(s), s))
        progress('gec_' + args.phase, step=stage, total=len(todo), processed=0)
        processed = 0
        for batch in chunks(todo, args.batch_size):
            if args.stop_file and Path(args.stop_file).exists():
                return {'processed':processed,'total':len(todo),'stopped':True}
            if tokenizer is None:
                tokenizer=AutoTokenizer.from_pretrained(BASE/'models/coedit-large',local_files_only=True)

            encoded = tokenizer([PREFIX + s for s in batch], padding=True, return_tensors='pt', truncation=False)
            lengths = encoded.attention_mask.sum(1).tolist()
            valid = [i for i, length in enumerate(lengths) if length <= 256]
            for i, length in enumerate(lengths):
                if length > 256:
                    cache.put(batch[i], {'error': 'input_too_long', 'input_tokens': length})
            def generate_subset(indices):
                oom=False
                inputs={key:value[indices,:max(lengths[i] for i in indices)] for key,value in encoded.items()}
                try:
                    with torch.inference_mode():
                        generated=generate(inputs,BASE)
                except torch.cuda.OutOfMemoryError:
                    if len(indices)==1:raise
                    oom=True
                del inputs
                if oom:
                    torch.cuda.empty_cache()
                    progress('gec_'+args.phase,step='retry_smaller_batch_after_cuda_oom',batch_size=len(indices))
                    middle=len(indices)//2
                    generate_subset(indices[:middle]);generate_subset(indices[middle:]);return
                ids_out=generated;texts_out=tokenizer.batch_decode(generated,skip_special_tokens=True)
                del generated
                for i, text, ids in zip(indices, texts_out, ids_out):
                    truncated = tokenizer.eos_token_id not in ids
                    cache.put(batch[i], {'text': text, 'input_tokens': lengths[i], 'truncated': truncated})
            if valid:generate_subset(valid)
            cache.flush()
            processed += len(batch)
            if processed % (args.batch_size * 10) == 0 or processed == len(todo):
                progress('gec_' + args.phase, step=stage, processed=processed, total=len(todo),
                         elapsed_seconds=round(time.monotonic() - begin, 1))
        return {'processed':processed,'total':len(todo),'stopped':False}

    original_stats=infer([r['variant_sentence'] for r in records], 'original_sentences')
    if args.prefetch_input:
        write_json('prefetch_result.json',dict(original_stats,config=config,cache_only=True,adjudications_written=False,elapsed_seconds=time.monotonic()-begin))
        return
    first_pass = []
    decisions = []
    for r in records:
        result = cache.get(r['variant_sentence'])
        reason = exact_correction_reason(result, r['root_sentence'], 'gec_not_exact_target_only_correction')
        if reason:
            decisions.append({'sid': r['sid'], 'record_id': r['record_id'], 'pass': False, 'reason': reason})
        else:
            first_pass.append(r)
    infer([r['root_sentence'] for r in first_pass], 'corrected_sentences')
    accepted = []
    for r in first_pass:
        result = cache.get(r['root_sentence'])
        reason = exact_correction_reason(result, r['root_sentence'], 'gec_additional_edits_in_corrected_sentence')
        decisions.append({'sid': r['sid'], 'record_id': r['record_id'], 'pass': reason is None,
                          'reason': reason or 'gec_two_way_exact_match'})
        if reason is None:
            accepted.append(r)
    write_jsonl('gec_pass_' + label + '.jsonl', accepted)
    write_jsonl('gec_decisions_' + label + '.jsonl', decisions)
    summary = {'phase': args.phase, 'config': config, 'input': len(records), 'original_exact_match': len(first_pass),
               'accepted': len(accepted), 'reasons': dict(Counter(d['reason'] for d in decisions)),
               'elapsed_seconds': time.monotonic() - begin,
               'shard_index': args.shard_index, 'num_shards': args.num_shards, 'batch_size': args.batch_size,
               'cuda_oom_handling':'recursively halve only the failed batch; preserve actual output and all acceptance rules'}
    write_json('gec_report_' + label + '.json', summary)
    progress('gec_' + args.phase, state='complete', **summary)


if __name__ == '__main__':
    main()
