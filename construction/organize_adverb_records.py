#!/usr/bin/env python3
"""Organize ADV roots, variants and yearly frequencies without semantic fields.

Requires NLTK and the WordNet corpus. Source files are never modified.
Use --registry from a previous build to preserve IDs when adding words.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import nltk
from nltk.corpus import wordnet as wn
from nltk.metrics.distance import edit_distance


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        obj[key] = value
    return obj


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique_object)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_json(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def assign_ids(roots, prior):
    registry = prior or {"schema_version": 1, "roots": {}, "variants": {}}
    if registry["schema_version"] != 1:
        raise ValueError("Unsupported registry schema")
    root_ids = registry["roots"]
    variants = registry["variants"]
    all_ids = set()
    root_numbers = []
    for root, rid in root_ids.items():
        if not re.fullmatch(r"adv_\d{6,}", rid) or rid in all_ids:
            raise ValueError(f"Invalid/duplicate root ID: {root}: {rid}")
        all_ids.add(rid)
        root_numbers.append(int(rid.split("_")[1]))
    next_root = max(root_numbers, default=0) + 1
    for root in sorted(roots):
        if root not in root_ids:
            root_ids[root] = f"adv_{next_root:06d}"
            all_ids.add(root_ids[root])
            next_root += 1
    for root, mapping in variants.items():
        if root not in root_ids:
            raise ValueError(f"Variant registry has unknown root: {root}")
        prefix = root_ids[root] + "_v"
        for vid in mapping.values():
            if not re.fullmatch(re.escape(prefix) + r"\d{3,}", vid) or vid in all_ids:
                raise ValueError(f"Invalid/duplicate variant ID: {vid}")
            all_ids.add(vid)
    for root in sorted(roots):
        mapping = variants.setdefault(root, {})
        next_variant = max((int(v.rsplit("_v", 1)[1]) for v in mapping.values()), default=0) + 1
        for word in sorted(roots[root]):
            if word not in mapping:
                mapping[word] = f"{root_ids[root]}_v{next_variant:03d}"
                next_variant += 1
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--timeseries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: args.output_dir / name for name in (
        "final_adversarial_advs_organized.json", "id_registry.json",
        "build_report.json", "sample_records.json", "frequency_provenance.json",
    )}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    source_stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in (args.clean, args.timeseries)}
    nltk.data.path.insert(0, str(args.nltk_data))
    wn.ensure_loaded()
    print(f"NLTK {nltk.__version__}; WordNet {wn.get_version()}", flush=True)
    roots = read_json(args.clean)
    if not isinstance(roots, dict) or not roots:
        raise ValueError("Expected a nonempty root-to-variants object")
    for root, words in roots.items():
        if not isinstance(root, str) or not root or root != root.strip():
            raise ValueError(f"Invalid root: {root!r}")
        if not isinstance(words, list) or any(not isinstance(w, str) or not w or w != w.strip() for w in words):
            raise ValueError(f"Invalid variant list: {root}")
        if len(words) != len(set(words)) or root in words:
            raise ValueError(f"Duplicate or self variant: {root}")
    print("Loading source timeseries...", flush=True)
    source = read_json(args.timeseries)
    years = source["years_axis"]
    if not years or any(type(y) is not int for y in years) or years != sorted(set(years)):
        raise ValueError("Year axis must contain increasing, unique integers")
    year_keys = [str(y) for y in years]
    ngrams = source["ngrams_data"]
    index = defaultdict(list)
    for key, entry in ngrams.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("ngram"), str):
            raise ValueError(f"Invalid ngram entry: {key}")
        values = entry.get("timeseries")
        if not isinstance(values, list) or len(values) != len(years):
            raise ValueError(f"Year/series length mismatch: {key}")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"Nonfinite, negative, or nonnumeric frequency: {key}")
        index[entry["ngram"]].append(key)
    conflicts = [word for word, keys in index.items()
                 if any(ngrams[k]["timeseries"] != ngrams[keys[0]]["timeseries"] for k in keys[1:])]
    if conflicts:
        raise ValueError(f"Conflicting surface-form timeseries require explicit resolution: {conflicts[:30]}")
    registry = assign_ids(roots, read_json(args.registry) if args.registry else None)
    all_words = set(roots) | {w for words in roots.values() for w in words}
    vocab = {}
    exact_lemmas = set(wn.all_lemma_names())
    morphology_only = []
    for word in sorted(all_words):
        # NLTK queries all supported POS and applies its standard morphological lookup.
        # This is dictionary membership only; no synsets or definitions are saved.
        vocab[word] = "IV" if wn.synsets(word) else "OOV"
        if vocab[word] == "IV" and word.lower() not in exact_lemmas:
            morphology_only.append(word)
    print(f"Classified {len(all_words)} distinct surface forms", flush=True)
    missing = []
    provenance = {}
    match_counts = Counter()
    vocab_counts = Counter()
    distances = Counter()
    seen_ids = set()

    def frequencies(word, root, role, record_id):
        candidates = index.get(word, [])
        if not candidates:
            missing.append({"id": record_id, "lemma": root, "word": word, "role": role})
            provenance[record_id] = {"source_key": None, "match": "missing"}
            match_counts["missing"] += 1
            return None
        matching = [k for k in candidates if
                    (role == "root" and ngrams[k].get("type") == "SPELLING" and ngrams[k].get("parent_ngram") is None)
                    or (role == "variant" and ngrams[k].get("type") == "MISSPELLING" and ngrams[k].get("parent_ngram") == root)]
        pool = matching or candidates
        chosen = word if word in pool else sorted(pool)[0]
        match = "role_and_parent" if matching else "identical_surface_series_fallback"
        provenance[record_id] = {"source_key": chosen, "match": match}
        match_counts[match] += 1
        return dict(zip(year_keys, ngrams[chosen]["timeseries"]))

    def check_record_id(record_id):
        if record_id in seen_ids:
            raise ValueError(f"Duplicate record ID: {record_id}")
        seen_ids.add(record_id)

    output = paths["final_adversarial_advs_organized.json"]
    temporary = output.with_suffix(".json.tmp")
    samples = []
    variant_count = 0
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("[\n")
        for number, root in enumerate(sorted(roots)):
            rid = registry["roots"][root]
            check_record_id(rid)
            record = {"id": rid, "lemma": root, "pos": "adv",
                      "lemma_frequencies": frequencies(root, root, "root", rid), "variants": []}
            for word in sorted(roots[root]):
                vid = registry["variants"][root][word]
                check_record_id(vid)
                distance = edit_distance(root, word, transpositions=False)
                variant = {"id": vid, "word": word, "vocab_type": vocab[word],
                           "edit_distance": distance,
                           "variant_frequencies": frequencies(word, root, "variant", vid)}
                record["variants"].append(variant)
                vocab_counts[vocab[word]] += 1
                distances[distance] += 1
                variant_count += 1
            if number:
                stream.write(",\n")
            stream.write("\n".join("  " + line for line in json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False).splitlines()))
            if root in {"luckily", "here", "afoot"}:
                samples.append(record)
        stream.write("\n]\n")
    assert variant_count == sum(map(len, roots.values()))
    assert len(seen_ids) == len(roots) + variant_count
    assert len(provenance) == len(seen_ids)
    for path in (args.clean, args.timeseries):
        if source_stats[str(path)] != (path.stat().st_size, path.stat().st_mtime_ns):
            raise RuntimeError(f"Input changed during conversion: {path}")
    temporary.rename(output)
    write_new_json(paths["id_registry.json"], registry)
    write_new_json(paths["sample_records.json"], samples)
    write_new_json(paths["frequency_provenance.json"], provenance)
    variant_words = {w for words in roots.values() for w in words}
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "inputs": {str(p): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in (args.clean, args.timeseries)},
        "output": {"path": str(output), "bytes": output.stat().st_size, "sha256": sha256(output)},
        "software": {"python": sys.version, "nltk": nltk.__version__, "wordnet": wn.get_version()},
        "rules": {
            "id": "Initial IDs follow alphabetically sorted roots and variants. Roots: adv_000001; variants: adv_000001_v001. Reuse --registry for later additions without renumbering existing IDs.",
            "vocab_type": "IV iff bool(nltk.corpus.wordnet.synsets(word)) across all POS; otherwise OOV. Includes NLTK's default case normalization and morphological lookup. OOV is relative to this WordNet version, not proof a spelling is invalid.",
            "edit_distance": "Character Levenshtein distance: insertion, deletion and substitution cost 1; adjacent transposition costs 2 (transpositions=False).",
            "relationships": "Preserve every root-variant pair in the clean input, including variants shared by roots and variants that are also roots. No WordNet filtering of records.",
            "timeseries": "Join by exact entry.ngram surface form and prefer the corresponding role/parent. Aliases may be used only after confirming all series for that surface are equal. No summation, scaling, rounding, interpolation, or missing-to-zero conversion.",
            "missing": "An absent surface-form series is represented by JSON null. Existing zero observations are preserved as zero.",
            "pos": "adv describes the source root collection; variant membership is queried across all WordNet POS.",
            "excluded_fields": ["nltk_candidate_synsets", "parallel_corpus", "resolved_synset_id", "wsd_confidence"],
        },
        "counts": {
            "roots": len(roots), "variant_relations": variant_count,
            "distinct_variant_words": len(variant_words), "distinct_all_words": len(all_words),
            "unique_record_ids": len(seen_ids), "source_timeseries_entries": len(ngrams),
            "source_alias_keys": sum(k != e["ngram"] for k, e in ngrams.items()),
            "source_repeated_surface_forms": sum(len(ks) > 1 for ks in index.values()),
            "source_conflicting_surface_series": len(conflicts),
            "variant_vocab_type_by_relation": dict(vocab_counts),
            "variant_vocab_type_by_distinct_word": dict(Counter(vocab[w] for w in variant_words)),
            "variant_edit_distances": dict(sorted(distances.items())),
            "frequency_match_methods": dict(match_counts),
            "missing_frequency_relations": len(missing),
        },
        "years": {"first": years[0], "last": years[-1], "count": len(years), "axis": years},
        "missing_frequency_records": missing,
        "iv_via_morphological_lookup_only": [w for w in morphology_only if w in variant_words],
        "wordnet_oov_roots": sorted(r for r in roots if vocab[r] == "OOV"),
        "validation": {"all_input_series_numeric_finite_nonnegative": True,
                       "all_input_series_match_year_axis": True,
                       "all_clean_relationships_preserved": True,
                       "all_record_ids_unique": True,
                       "independent_output_roundtrip_check": "pending"},
    }
    write_new_json(paths["build_report.json"], report)
    print(json.dumps({"output": str(output), "counts": report["counts"], "missing": missing}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
