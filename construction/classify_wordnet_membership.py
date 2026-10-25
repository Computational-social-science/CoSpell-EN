"""Pairwise difference, then exact English WordNet vocabulary membership.

No synsets/morphy-based membership, inflection expansion, spelling correction,
compound splitting, or whitespace replacement. Source spellings and order are
preserved. Only variants are tested for IV; root keys identify their source.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--inflections", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--nltk-data", required=True, type=Path)
    args = ap.parse_args()
    import nltk
    nltk.data.path.insert(0, str(args.nltk_data))
    from nltk.corpus import wordnet as wn
    wn.ensure_loaded()

    original_bytes = args.input.read_bytes()
    inflection_bytes = args.inflections.read_bytes()
    original = json.loads(original_bytes)
    inflections = json.loads(inflection_bytes)
    assert isinstance(original, dict) and isinstance(inflections, dict)
    assert all(isinstance(r, str) and isinstance(vs, list) and
               all(isinstance(v, str) for v in vs) for r, vs in original.items())
    assert all(r in original and set(vs) <= set(original[r]) for r, vs in inflections.items())
    args.output.mkdir(parents=True, exist_ok=True)

    # Public API, exact English lemma-name inventory. No morphological lookup.
    vocabulary = set(wn.all_lemma_names(lang="eng"))
    by_pos = {pos: set(wn.all_lemma_names(pos=pos, lang="eng"))
              for pos in ("n", "v", "a", "r")}
    assert vocabulary == set().union(*by_pos.values())

    def membership(word):
        return word in vocabulary

    # Membership is case-sensitive exact surface matching, as requested. The
    # supplied data is lowercase; capitalization is counted in the summary.
    for word in ("writer", "denounce", "pass", "address"):
        assert membership(word), word
    for word in ("passs", "addresss", "studyed", "goed"):
        assert not membership(word), word

    difference, selected = {}, {}
    details, empty_roots = [], []
    input_count = removed_count = difference_count = oov_count = 0
    for root, variants in original.items():
        banned = set(inflections.get(root, []))
        remainder = [word for word in variants if word not in banned]
        difference[root] = remainder
        retained = [word for word in remainder if membership(word)]
        if retained:
            selected[root] = retained
        else:
            empty_roots.append(root)
        input_count += len(variants)
        removed_count += len(variants) - len(remainder)
        difference_count += len(remainder)
        oov_count += len(remainder) - len(retained)
        for word in retained:
            details.append({"root": root, "variant": word,
                            "wordnet_lemma": word,
                            "wordnet_pos": [pos for pos, entries in by_pos.items() if word in entries],
                            "iv_method": "exact_english_lemma_name"})

    source_pairs = {(r, w) for r, vs in original.items() for w in vs}
    removed_pairs = {(r, w) for r, vs in inflections.items() for w in vs}
    difference_pairs = {(r, w) for r, vs in difference.items() for w in vs}
    selected_pairs = {(r, w) for r, vs in selected.items() for w in vs}
    assert difference_pairs == source_pairs - removed_pairs
    assert not selected_pairs & removed_pairs
    assert selected_pairs == {(r, w) for r, w in difference_pairs if w in vocabulary}
    assert len(details) == sum(map(len, selected.values()))
    assert input_count == removed_count + difference_count
    assert difference_count == len(details) + oov_count
    assert all(selected.values())
    assert args.input.read_bytes() == original_bytes
    assert args.inflections.read_bytes() == inflection_bytes

    write_json(args.output / "merged_verb_difference.wordnet_exact_iv.json", selected)
    write_json(args.output / "verb_difference_before_iv.json", difference)
    with (args.output / "wordnet_exact_iv.details.jsonl").open("w", encoding="utf-8") as f:
        for record in details:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(args.output / "roots_removed_no_iv_variants.json", empty_roots)
    write_json(args.output / "wordnet_exact_iv.words.json", sorted({d["variant"] for d in details}))
    summary = {
        "input": str(args.input), "inflections_subtracted": str(args.inflections),
        "input_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "inflections_sha256": hashlib.sha256(inflection_bytes).hexdigest(),
        "nltk_version": nltk.__version__, "wordnet_version": wn.get_version(),
        "wordnet_english_lemma_names": len(vocabulary),
        "iv_definition": "Variant exactly equals an English WordNet lemma name, any POS; case-sensitive; no morphological normalization.",
        "source_roots": len(original), "source_pairs": input_count,
        "subtracted_inflection_pairs": removed_count,
        "pairs_after_difference": difference_count,
        "discarded_non_exact_iv_pairs": oov_count,
        "retained_roots": len(selected), "retained_pairs": len(details),
        "retained_unique_variants": len({d['variant'] for d in details}),
        "removed_empty_roots": len(empty_roots),
        "non_lowercase_source_pairs": sum(w != w.lower() for vs in original.values() for w in vs),
        "retained_pos_pair_counts_nonexclusive": dict(Counter(pos for d in details for pos in d['wordnet_pos'])),
        "source_unchanged": True, "inflection_list_unchanged": True,
        "timeseries_processed": False,
        "checks": {"pairwise_difference_exact": True, "all_retained_exact_iv": True,
                   "no_removed_pair_reintroduced": True, "root_variant_correspondence_preserved": True,
                   "no_empty_root_in_output": True, "count_conservation": True},
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("EXAMPLES", json.dumps(dict(list(selected.items())[:8]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
