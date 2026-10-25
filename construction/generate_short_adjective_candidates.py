"""WordNet short adjectives -> local Llama 3 8B -> exact GBNC series.

GBNC parameters and output fields follow VERB/20250517-timeseries.py.
Network failures are fatal/retryable, never classified as missing words.
"""
import ast
import datetime as dt
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

BASE = Path('/path/to/cospell/Vocabulary/ShortWord/ADJ')
MODEL = Path('/data1/css/models/Meta-Llama-3-8B-Instruct')
REFERENCE = BASE.parent / 'VERB/20250517-timeseries.py'
GENERATION_REFERENCE = BASE.parents[1] / 'DataAnalysis/20250612LLMsCommonMisspellings.py'
YEARS = list(range(1800, 2023))
PARAMS = dict(year_start=1800, year_end=2022, corpus='english', smoothing=0)


def reference_prompt(root):
    tree = ast.parse(GENERATION_REFERENCE.read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_misspellings')
    value = next(n.value for n in fn.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'prompt' for t in n.targets))
    assert isinstance(value, ast.JoinedStr)
    assert all(isinstance(n, (ast.JoinedStr, ast.Constant, ast.FormattedValue, ast.Name, ast.Load))
               for n in ast.walk(value))
    assert all(n.id in ('word', 'pos') for n in ast.walk(value) if isinstance(n, ast.Name))
    expression = ast.Expression(body=value)
    return eval(compile(expression, str(GENERATION_REFERENCE), 'eval'),
                {'__builtins__': {}}, {'word': root, 'pos': 'Adjective'})


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def read(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def extract():
    import nltk
    nltk.data.path.insert(0, str(BASE / 'nltk_data'))
    from nltk.corpus import wordnet as wn
    wn.ensure_loaded()
    details = {}
    # all_synsets('a') includes ordinary and satellite adjective synsets.
    for synset in wn.all_synsets(pos=wn.ADJ):
        assert synset.pos() in ('a', 's')
        for lemma in synset.lemmas():
            if not re.fullmatch(r'[A-Za-z]{1,3}', lemma.name()):
                continue
            root = lemma.name().lower()
            details.setdefault(root, []).append(dict(
                lemma=lemma.name(), synset=synset.name(), pos=synset.pos(),
                definition=synset.definition()))
    roots = sorted(details)
    # Verify exact lemma extraction against the separate public inventory.
    expected = {w.lower() for w in wn.all_lemma_names(pos=wn.ADJ)
                if re.fullmatch(r'[A-Za-z]{1,3}', w)}
    assert set(roots) == expected
    save(BASE / 'short_adjectives_wordnet.json', roots)
    save(BASE / 'wordnet_details.json', {w: details[w] for w in roots})
    return roots, details, dict(nltk_version=nltk.__version__, wordnet_version=wn.get_version(),
                               wordnet_path=str(wn.root))


def parse_response(text, root):
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\[', text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and all(isinstance(x, (str, dict)) for x in value):
            accepted, rejected = [], []
            for original in value:
                item = original.get('typo', '') if isinstance(original, dict) else original
                word = item.strip() if isinstance(item, str) else ''
                reason = ('non_ascii_letters' if not re.fullmatch('[A-Za-z]+', word)
                          else 'same_as_root' if word.lower() == root.lower()
                          else 'duplicate' if word in accepted else None)
                if reason:
                    rejected.append(dict(value=original, reason=reason))
                else:
                    accepted.append(word)
            tree = ast.parse(GENERATION_REFERENCE.read_text(encoding='utf-8'))
            examples = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'example_misspellings' for t in n.targets)))
            for example_word, variants in examples.items():
                if example_word != root and len(set(accepted) & set(variants)) >= 3:
                    raise ValueError('Generated answer copies the reference examples')
            return accepted, rejected
    raise ValueError('No JSON array of strings in generated continuation')


def generate(roots, details, wnmeta):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    checkpoint = BASE / 'llama_generation_checkpoint.json'
    config = dict(model_path=str(MODEL), generation_reference=str(GENERATION_REFERENCE),
                  reference_sha256=sha(GENERATION_REFERENCE.read_bytes()),
                  prompt_example=reference_prompt('bad'), seed=20261014,
                  batch_size=8, do_sample=True, temperature=0.8, top_p=0.9,
                  max_new_tokens=128, dtype='float16', roots=roots,
                  prompt_method='Exact reference f-string as user message with native Llama chat template',
                  additional_edit_distance_or_morphology_filter=False)
    state = read(checkpoint, dict(config=config, started_at_utc=stamp(), results={},
                                 attempts={}, versions=dict(torch=torch.__version__,
                                 transformers=transformers.__version__, **wnmeta)))
    assert state['config'] == config, 'Generation settings changed; use a new checkpoint'
    for root in roots:
        if root not in state['results'] and len(state['attempts'].get(root, [])) >= 3:
            state['results'][root] = []
            state.setdefault('empty_reasons', {})[root] = 'No parseable list after three model responses'
    todo = [w for w in roots if w not in state['results']]
    if not todo:
        state.setdefault('completed_at_utc', stamp())
        save(checkpoint, state)
        return state['results']
    print('Loading local Meta-Llama-3-8B-Instruct', flush=True)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL), local_files_only=True, torch_dtype=torch.float16,
        device_map={'': 'cuda:0'}, attn_implementation='sdpa')
    model.eval()
    state['parameter_count'] = sum(p.numel() for p in model.parameters())
    state['model_config_sha256'] = sha((MODEL / 'config.json').read_bytes())
    eos = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]

    def run_batch(batch, seed):
        prompts = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': reference_prompt(root)}],
            tokenize=False, add_generation_prompt=True) for root in batch]
        inputs = tokenizer(prompts, return_tensors='pt', padding=True,
                           add_special_tokens=False).to(model.device)
        set_seed(seed)
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=True, temperature=0.8, top_p=0.9,
                                    max_new_tokens=128, pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=eos)
        texts = tokenizer.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for root, text in zip(batch, texts):
            attempt = dict(seed=seed, generated_at_utc=stamp(), raw_response=text,
                           prompt=reference_prompt(root))
            try:
                variants, rejected = parse_response(text, root)
                attempt.update(accepted=variants, rejected=rejected)
                state['results'][root] = variants
            except ValueError as exc:
                attempt['error'] = str(exc)
            state['attempts'].setdefault(root, []).append(attempt)
        save(checkpoint, state)
        print(f"LLAMA {len(state['results'])}/{len(roots)} roots; "
              f"{sum(map(len, state['results'].values()))} candidate pairs", flush=True)

    for start in range(0, len(roots), 8):
        batch = [w for w in roots[start:start + 8] if w in todo]
        if batch:
            run_batch(batch, config['seed'] + start)
    for index, root in enumerate(roots):
        for attempt in range(2):
            if state['results'].get(root) or len(state['attempts'].get(root, [])) >= 3:
                break
            run_batch([root], config['seed'] + 1000 + index * 2 + attempt)
        if root not in state['results']:
            state['results'][root] = []
            state.setdefault('empty_reasons', {})[root] = 'No parseable list after three model responses'
            save(checkpoint, state)
    assert set(state['results']) == set(roots), 'Unresolved model output errors'
    state['completed_at_utc'] = stamp()
    save(checkpoint, state)
    del model
    torch.cuda.empty_cache()
    return state['results']


def fetch(candidates):
    import requests
    words = sorted(set(candidates) | {v for vs in candidates.values() for v in vs})
    rawdir = BASE / 'gbnc_raw_responses'
    rawdir.mkdir(exist_ok=True)
    checkpoint = BASE / 'gbnc_checkpoint.json'
    state = read(checkpoint, dict(parameters=PARAMS, words=words, started_at_utc=stamp(),
                                 records={}, queried=[], confirmed_missing=[], requests=[]))
    assert state['parameters'] == PARAMS and state['words'] == words
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
    last_call = 0.0

    def query(batch):
        nonlocal last_call
        for attempt in range(4):
            time.sleep(max(0, 3 - (time.monotonic() - last_call)))
            record = dict(words=batch, attempt=attempt + 1, retrieved_at_utc=stamp())
            try:
                last_call = time.monotonic()
                response = session.get('https://books.google.com/ngrams/json',
                                       params=dict(content=','.join(batch), **PARAMS), timeout=35)
                record.update(url=response.url, http_status=response.status_code)
                response.raise_for_status()
                data = response.json()
                assert isinstance(data, list), 'GBNC response must be a list'
                rawpath = rawdir / f"{len(state['requests']):04d}_{sha(response.content)[:12]}.json"
                rawpath.write_bytes(response.content)
                record.update(raw_file=str(rawpath), response_sha256=sha(response.content))
                found, invalid = {}, {}
                for item in data:
                    word, values = item.get('ngram'), item.get('timeseries')
                    if word not in batch:
                        continue
                    valid = (isinstance(values, list) and len(values) == len(YEARS) and
                             all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                                 for v in values))
                    if not valid:
                        invalid[word] = 'incomplete_or_invalid_series'
                        continue
                    assert word not in found, f'Duplicate label {word}'
                    found[word] = dict(timeseries=values, source_url=response.url,
                                       retrieved_at_utc=record['retrieved_at_utc'], raw_file=str(rawpath))
                # Every query includes a known common control word. An empty / broken
                # service response must not silently delete the entire candidate set.
                assert 'the' in found, 'Control word the is absent: retry service response'
                record.update(returned_words=sorted(found), invalid=invalid)
                state['requests'].append(record)
                save(checkpoint, state)
                return {w: d for w, d in found.items() if w in words}
            except Exception as exc:
                record['error'] = repr(exc)
                state['requests'].append(record)
                save(checkpoint, state)
                print(f'GBNC retry {attempt + 1}: {exc}', flush=True)
                if attempt == 3:
                    raise
                time.sleep(10 * (attempt + 1))

    pending = [w for w in words if w not in state['queried']]
    for start in range(0, len(pending), 9):
        batch = pending[start:start + 9]
        state['records'].update(query(list(dict.fromkeys(batch + ['the']))))
        state['queried'].extend(batch)
        save(checkpoint, state)
        print(f"GBNC {len(state['queried'])}/{len(words)} queried; "
              f"{len(state['records'])} complete series", flush=True)
    # A second successful, controlled request verifies each apparent absence.
    missing = [w for w in words if w not in state['records'] and w not in state['confirmed_missing']]
    for start in range(0, len(missing), 9):
        batch = missing[start:start + 9]
        found = query(list(dict.fromkeys(batch + ['the'])))
        state['records'].update(found)
        state['confirmed_missing'].extend(w for w in batch if w not in found)
        save(checkpoint, state)
        print(f"GBNC confirmed missing {len(state['confirmed_missing'])}", flush=True)
    assert set(state['records']) | set(state['confirmed_missing']) == set(words)
    assert not set(state['records']) & set(state['confirmed_missing'])
    state['completed_at_utc'] = stamp()
    save(checkpoint, state)
    return state


def finalize(roots, candidates, gbnc, wnmeta):
    records, cleaned, removed_roots, removed_variants = gbnc['records'], {}, {}, {}
    for root in roots:
        available = [v for v in candidates[root] if v in records]
        missing = [v for v in candidates[root] if v not in records]
        if missing:
            removed_variants[root] = missing
        if root not in records:
            removed_roots[root] = 'missing_root_timeseries'
        elif not available:
            removed_roots[root] = ('no_generated_variants' if not candidates[root]
                                   else 'no_variants_with_timeseries')
        else:
            cleaned[root] = available
    parents = {}
    for root, variants in cleaned.items():
        for variant in variants:
            parents.setdefault(variant, []).append(root)
    retained = set(cleaned) | set(parents)
    series = {}
    for word in sorted(retained):
        roles = (['SPELLING'] if word in cleaned else []) + (['MISSPELLING'] if word in parents else [])
        series[word] = dict(ngram=word, parent_ngram=None if word in cleaned else parents[word][0],
                            parent_ngrams=parents.get(word, []), type=roles[0], types=roles,
                            timeseries=records[word]['timeseries'])
    finalseries = dict(years_axis=YEARS, ngrams_data=series,
                       parameters=PARAMS, case_insensitive=False)
    assert all(re.fullmatch('[a-z]{1,3}', r) and vs and len(vs) == len(set(vs))
               and all(re.fullmatch('[A-Za-z]+', v) and v.lower() != r.lower() for v in vs)
               for r, vs in cleaned.items())
    assert set(series) == retained
    assert all(len(d['timeseries']) == 223 for d in series.values())
    assert len(cleaned) + len(removed_roots) == len(roots)
    save(BASE / 'short_adjectives_clean.json', cleaned)
    save(BASE / 'short_adjectives_timeseries.json', finalseries)
    save(BASE / 'removed_items.json', dict(roots=removed_roots, variants_missing_timeseries=removed_variants,
                                         unique_words_missing_timeseries=gbnc['confirmed_missing']))
    from collections import Counter
    summary = dict(completed_at_utc=stamp(), **wnmeta, model_path=str(MODEL),
                   generation_reference=str(GENERATION_REFERENCE),
                   generation_reference_sha256=sha(GENERATION_REFERENCE.read_bytes()),
                   reference_script=str(REFERENCE), reference_sha256=sha(REFERENCE.read_bytes()),
                   parameters=PARAMS, root_length='1 <= length < 4', root_characters='ASCII letters only',
                   case_policy='lowercase roots; model variant case preserved; case-sensitive GBNC lookup',
                   adjective_pos=['a', 's'], additional_edit_distance_or_morphology_filter=False,
                   candidate_roots=len(roots), candidate_pairs=sum(map(len, candidates.values())),
                   queried_unique_words=len(gbnc['words']), retained_roots=len(cleaned),
                   retained_pairs=sum(map(len, cleaned.values())), retained_unique_variants=len(parents),
                   retained_unique_series=len(series), removed_roots=len(removed_roots),
                   removed_root_reasons=dict(Counter(removed_roots.values())),
                   missing_unique_words=len(gbnc['confirmed_missing']),
                   shared_variants={v: ps for v, ps in parents.items() if len(ps) > 1},
                   roots_also_variants=sorted(set(cleaned) & set(parents)),
                   years_per_series=223, validation='passed',
                   limitation='Llama-generated plausible common typo candidates; GBNC occurrence is not proof of misspelling usage or commonness. Zero-valued years are observed values, not missing data.')
    save(BASE / 'summary.json', summary)
    print('COMPLETE ' + json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    roots, details, wnmeta = extract()
    candidates = generate(roots, details, wnmeta)
    save(BASE / 'short_adjectives_llama_candidates.json', candidates)
    gbnc = fetch(candidates)
    finalize(roots, candidates, gbnc, wnmeta)


if __name__ == '__main__':
    main()

