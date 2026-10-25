"""Organize long nouns with exact relation-specific source frequencies."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import nltk
from nltk.corpus import wordnet as wn
from nltk.metrics.distance import edit_distance
from index_noun_source import digest, canonical, unique_object, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--nltk-data', type=Path, required=True)
    ap.add_argument('--classify-only', action='store_true')
    ap.add_argument('--conflict-policy', choices=['error', 'retain_all'], default='error')
    args = ap.parse_args()
    folder = args.output_dir
    audit = json.loads((folder / 'source_index_report.json').read_text())
    assert audit['status'] == 'passed'
    source_paths = list(audit['inputs'])
    for p in source_paths:
        assert digest(p) == audit['inputs'][p]['sha256'], 'Input changed: ' + p
    roots = json.loads(Path(source_paths[0]).read_text(), object_pairs_hook=unique_object)
    years = audit['header']['years_axis']
    year_keys = list(map(str, years))
    output = folder / 'final_adversarial_nouns_organized.json'
    report_path = folder / 'build_report.json'
    if output.exists() or report_path.exists():
        raise FileExistsError('Refusing to overwrite an existing organized dataset')
    nltk.data.path.insert(0, str(args.nltk_data))
    wn.ensure_loaded()
    assert wn.get_version() == '3.0'
    all_words = set(roots) | {word for words in roots.values() for word in words}
    vocab = {}
    begin = time.monotonic()
    vocab_path = folder / 'wordnet_membership.json'
    cache_manifest = folder / 'classification_complete.json'
    if cache_manifest.exists():
        cached = json.loads(cache_manifest.read_text())
        assert cached['inputs'] == audit['inputs']
        assert cached['wordnet_sha256'] == digest(args.nltk_data / 'corpora/wordnet.zip')
        assert cached['membership_sha256'] == digest(vocab_path)
        vocab = json.loads(vocab_path.read_text())
        assert set(vocab) == all_words
    else:
        for i, word in enumerate(sorted(all_words), 1):
            vocab[word] = 'IV' if wn.synsets(word) else 'OOV'
            if i % 50000 == 0:
                print(json.dumps({'phase': 'wordnet_membership', 'words': i, 'elapsed_seconds': round(time.monotonic()-begin, 1)}), flush=True)
        write_json(vocab_path, vocab)
        write_json(cache_manifest, {'status': 'passed', 'inputs': audit['inputs'], 'wordnet_sha256': digest(args.nltk_data / 'corpora/wordnet.zip'),
                                   'membership_sha256': digest(vocab_path), 'distinct_words': len(vocab)})
    print(json.dumps({'phase': 'wordnet_complete', 'distinct_words': len(vocab)}), flush=True)
    if args.classify_only:
        return
    if audit['conflicts'] and args.conflict_policy == 'error':
        raise ValueError('Conflicting series within an exact relation; see source_index_report.json')
    db = sqlite3.connect('file:' + str(folder / 'source_index.sqlite') + '?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    db.execute('PRAGMA cache_size=-131072')
    db.row_factory = sqlite3.Row
    registry = {'schema_version': 1, 'roots': {}, 'variants': {}}
    counts, vocabulary, distances = Counter(), Counter(), Counter()
    missing, samples = [], []
    provisional = output.with_suffix('.json.tmp')
    prov_path = folder / 'frequency_provenance.json'
    prov_tmp = folder / 'frequency_provenance.json.tmp'
    alternatives_path = folder / 'frequency_alternatives.jsonl'
    alternatives_tmp = folder / 'frequency_alternatives.jsonl.tmp'
    first_provenance = True
    with provisional.open('x', encoding='utf-8', newline='\n') as out, prov_tmp.open('x', encoding='utf-8', newline='\n') as provenance, alternatives_tmp.open('x', encoding='utf-8', newline='\n') as alternatives:
        out.write('[\n')
        provenance.write('{\n')

        def frequencies(word, root, role, record_id, matches):
            nonlocal first_provenance
            match_method = 'exact_surface_role_parent'
            if not matches and role == 'root':
                matches = db.execute('SELECT * FROM records WHERE word=? ORDER BY source_key,ordinal', (word,)).fetchall()
                match_method = 'identical_surface_series_fallback'
            distinct = {r['series_sha256'] for r in matches}
            if not matches:
                values = None
                entry = {'source_key': None, 'match': 'missing_exact_relation', 'root': root, 'word': word, 'role': role}
                missing.append({'id': record_id, 'lemma': root, 'word': word, 'role': role, 'reason': 'missing_surface_data'})
                counts['missing_frequency_records'] += 1
                counts['null_frequency_records'] += 1
            elif len(distinct) > 1:
                assert args.conflict_policy == 'retain_all', 'Conflicting relation: ' + record_id
                values = None
                refs = [{'source_key': r['source_key'], 'source_record_index': r['ordinal'], 'record_sha256': r['record_sha256'], 'series_sha256': r['series_sha256']} for r in matches]
                entry = {'source_key': None, 'match': 'conflicting_source_series', 'root': root, 'word': word, 'role': role,
                         'alternatives_file': alternatives_path.name, 'alternatives_id': record_id, 'source_records': refs}
                alternatives.write(canonical({'id': record_id, 'root': root, 'word': word, 'role': role, 'years_axis': years,
                    'candidates': [{**ref, 'timeseries': json.loads(row['series']), 'source_metadata': json.loads(row['metadata'])} for ref,row in zip(refs,matches)]})+'\n')
                missing.append({'id': record_id, 'lemma': root, 'word': word, 'role': role, 'reason': 'conflicting_source_series', 'alternatives': len(matches)})
                counts['conflicting_frequency_records'] += 1
                counts['null_frequency_records'] += 1
            else:
                chosen = min(matches, key=lambda r: (r['source_key'], r['ordinal']))
                values = dict(zip(year_keys, json.loads(chosen['series'])))
                entry = {'source_key': chosen['source_key'], 'source_record_index': chosen['ordinal'], 'match': match_method,
                         'root': root, 'word': word, 'role': role, 'series_sha256': chosen['series_sha256'],
                         'record_sha256': chosen['record_sha256'], 'source_metadata': json.loads(chosen['metadata']),
                         'equivalent_source_records': [{'source_key': r['source_key'], 'source_record_index': r['ordinal'], 'record_sha256': r['record_sha256']} for r in matches]}
                counts['matched_frequency_records'] += 1
                counts['frequency_values'] += len(years)
                counts[match_method] += 1
            if not first_provenance:
                provenance.write(',\n')
            first_provenance = False
            provenance.write(json.dumps(record_id) + ':' + canonical(entry))
            return values

        for number, root in enumerate(sorted(roots), 1):
            rid = f'noun_{number:06d}'
            registry['roots'][root] = rid
            registry['variants'][root] = {}
            root_sources = db.execute("SELECT * FROM records WHERE word=? AND kind='SPELLING' AND parent=''", (root,)).fetchall()
            by_variant = {}
            for row in db.execute("SELECT * FROM records WHERE parent=? AND kind='MISSPELLING' ORDER BY source_key,ordinal", (root,)):
                by_variant.setdefault(row['word'], []).append(row)
            rec = {'id': rid, 'lemma': root, 'pos': 'noun', 'lemma_frequencies': frequencies(root, root, 'root', rid, root_sources), 'variants': []}
            for vi, word in enumerate(sorted(roots[root]), 1):
                vid = f'{rid}_v{vi:03d}'
                registry['variants'][root][word] = vid
                distance = edit_distance(root, word, transpositions=False)
                rec['variants'].append({'id': vid, 'word': word, 'vocab_type': vocab[word], 'edit_distance': distance,
                    'variant_frequencies': frequencies(word, root, 'variant', vid, by_variant.get(word, [])), 'parallel_corpus': []})
                vocabulary[vocab[word]] += 1
                distances[distance] += 1
                counts['variant_relations'] += 1
            if number > 1:
                out.write(',\n')
            # One complete root per line keeps multi-gigabyte output stream-readable.
            out.write(json.dumps(rec, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
            if root in {'apple', 'lowan', 'loan', 'cupflower'} or number <= 2:
                samples.append(rec)
            if number % 2000 == 0:
                print(json.dumps({'phase': 'organizing', 'roots': number, 'relations': counts['variant_relations'], 'elapsed_seconds': round(time.monotonic()-begin, 1)}), flush=True)
        out.write('\n]\n')
        provenance.write('\n}\n')
    db.close()
    assert counts['variant_relations'] == sum(map(len, roots.values()))
    assert counts['matched_frequency_records'] + counts['null_frequency_records'] == len(roots) + counts['variant_relations']
    for p in source_paths:
        assert digest(p) == audit['inputs'][p]['sha256'], 'Input changed: ' + p
    os.replace(provisional, output)
    os.replace(prov_tmp, prov_path)
    os.replace(alternatives_tmp, alternatives_path)
    write_json(folder / 'id_registry.json', registry)
    write_json(folder / 'sample_records.json', samples)
    write_json(folder / 'wordnet_membership.json', vocab)
    counts.update({'roots': len(roots), 'unique_record_ids': len(roots) + counts['variant_relations'], 'distinct_all_words': len(all_words),
                   'distinct_variant_words': len({w for vs in roots.values() for w in vs}),
                   'source_timeseries_entries': audit['counts']['records'], 'source_conflicting_exact_relations': len(audit['conflicts'])})
    count_report = dict(counts, variant_vocab_type_by_relation=dict(vocabulary))
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'schema_version': 1, 'inputs': audit['inputs'],
        'output': {'path': str(output), 'bytes': output.stat().st_size, 'sha256': digest(output)},
        'counts': count_report, 'years': {'first': years[0], 'last': years[-1], 'count': len(years), 'axis': years},
        'missing_frequency_records': missing, 'variant_edit_distances': dict(sorted(distances.items())),
        'conflict_policy': args.conflict_policy,
        'software': {'python': sys.version, 'python_executable': sys.executable, 'nltk': nltk.__version__, 'wordnet': wn.get_version(),
                     'nltk_data': str(args.nltk_data), 'wordnet_corpus_sha256': digest(args.nltk_data / 'corpora/wordnet.zip')},
        'rules': {
            'id': 'Deterministic alphabetical noun_000001 / noun_000001_v001 IDs; registry stores all exact spellings and relations.',
            'vocab_type': 'IV iff nltk.corpus.wordnet.synsets(word) is nonempty across all POS with default case/morphology lookup, as in existing ADV/VERB/ADJ organizers. OOV is dictionary-relative, not proof of misspelling.',
            'edit_distance': 'Character Levenshtein insertion/deletion/substitution cost 1; transposition costs 2.',
            'relationships': 'Preserve all clean root-variant relations, original case and spelling; no root-length, dictionary, or semantic filtering.',
            'timeseries': 'Match exact surface + SPELLING/null parent for roots; exact surface + MISSPELLING + exact parent for variants. If a root lacks its SPELLING record, use a same-surface series only when all available series are identical. Different parent-specific series are preserved separately. No aggregation, interpolation, rounding or imputed zero.',
            'conflicts': 'Under retain_all, a relation with differing candidate series has null main frequency and every original series/metadata in frequency_alternatives.jsonl. No source is arbitrarily selected or averaged. Under error, conflicts abort the build.',
            'missing': 'Missing surface data are null and reported separately from unresolved conflicting frequencies.',
            'parallel_corpus': 'Initially empty for every variant; sentence collection/model validation is a separate experiment.',
            'pos': 'noun labels the source collection; it does not assert that every source word is a noun in every context.',
            'excluded_fields': ['nltk_candidate_synsets', 'resolved_synset_id', 'wsd_confidence']},
        'artifacts': {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in [prov_path, alternatives_path, folder/'id_registry.json', folder/'wordnet_membership.json', folder/'source_index_report.json']},
        'builder_sha256': digest(__file__), 'independent_validation': 'pending'}
    write_json(report_path, report)
    print(json.dumps({'status': 'built', 'counts': count_report, 'output': report['output']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

