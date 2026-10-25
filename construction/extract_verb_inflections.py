"""Extract attested/regular verb inflections without changing source mappings.

Uses en_core_web_lg's VERB lemmatizer and contextual annotations. LemmInflect
guards against spaCy overlemmatizing misspellings such as passs and studyed.
Only the supplied root/variant pair is classified; roots are not themselves
lemmatized, because e.g. found is also an independent verb, distinct from find.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--dependencies", type=Path)
    args = ap.parse_args()
    if args.dependencies:
        sys.path.insert(0, str(args.dependencies))
    import spacy
    import lemminflect
    from lemminflect import getAllInflections, getInflection
    from spacy.symbols import VERB
    from spacy.tokens import Doc

    original = args.input.read_bytes()
    data = json.loads(original)
    assert isinstance(data, dict)
    assert all(isinstance(r, str) and isinstance(vs, list) and all(isinstance(v, str) for v in vs)
               for r, vs in data.items())
    args.output.mkdir(parents=True, exist_ok=True)
    nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])
    lemmatizer = nlp.get_pipe("lemmatizer")
    index = lemmatizer.lookups.get_table("lemma_index")["verb"]
    exceptions = lemmatizer.lookups.get_table("lemma_exc")["verb"]
    tags = ("VBZ", "VBD", "VBN", "VBG")
    words = list(dict.fromkeys(v.lower() for vs in data.values() for v in vs))
    lemmas = {}
    for start in range(0, len(words), 1000):
        doc = Doc(nlp.vocab, words=words[start:start + 1000])
        for token in doc:
            # Supplied data specifies verb analysis; isolated-word POS tagging
            # would incorrectly discard noun/verb homographs such as shelves.
            token.pos = VERB
            lemmas[token.text] = list(lemmatizer.rule_lemmatize(token))
    print(f"VERB lemmatization complete: {len(words)} unique variants", flush=True)

    paradigms = {}
    def paradigm(root):
        if root not in paradigms:
            known = bool(getAllInflections(root, upos="VERB"))
            paradigms[root] = (known, {
                tag: tuple(getInflection(root, tag=tag, inflect_oov=False)) if known else ()
                for tag in tags
            })
        return paradigms[root]

    def classify(root, variant):
        r, w = root.lower(), variant.lower()
        if r == w:
            return None
        same_lemma = r in lemmas.get(w, [])
        known, forms = paradigm(r)
        matched = [tag for tag in tags if w in forms[tag]]
        if matched:
            return "dictionary_inflection", matched, None
        if not same_lemma:
            return None
        # Generate only for established spaCy verb lemmas absent from the
        # inflection dictionary. Never override a known irregular paradigm.
        if not known and r in index:
            matched = [tag for tag in tags if w in getInflection(r, tag=tag, inflect_oov=True)]
            if matched:
                return "spacy_lemma_and_generated_regular_form", matched, None
        # spaCy exceptions sometimes normalize alternative base spellings
        # (tranship -> transship). Require preservation of a prefix and a
        # dictionary-backed inflection of the verb stem, e.g. out+draw/drew.
        if r in exceptions.get(w, []):
            for cut in range(1, len(r) - 2):
                prefix, base = r[:cut], r[cut:]
                _, base_forms = paradigm(base)
                matched = [tag for tag in tags if any(w == prefix + f for f in base_forms[tag])]
                if matched:
                    return "spacy_exception_and_compound_paradigm", matched, base
        return None

    # Meaningful regression checks for overlemmatization and pair identity.
    probes = {
        ("study", "studied"): True, ("study", "studyed"): False,
        ("go", "went"): True, ("go", "goed"): False,
        ("write", "written"): True, ("write", "writed"): False,
        ("pass", "passs"): False, ("address", "addresss"): False,
        ("stop", "stopped"): True, ("stop", "stoped"): False,
        ("find", "found"): True, ("found", "finds"): False,
        ("found", "founds"): True, ("write", "writer"): False,
        ("spurn", "spun"): False, ("transship", "tranship"): False,
        ("felt", "felted"): True, ("outdraw", "outdrew"): True,
    }
    for (r, w), expected in probes.items():
        if w not in lemmas:
            d = Doc(nlp.vocab, words=[w]); d[0].pos = VERB
            lemmas[w] = list(lemmatizer.rule_lemmatize(d[0]))
        assert (classify(r, w) is not None) == expected, (r, w, expected)

    accepted, rejected, output = [], [], {}
    same_lemma_count = 0
    roots_without_dictionary = 0
    for root, variants in data.items():
        if not paradigm(root.lower())[0]:
            roots_without_dictionary += 1
        for variant in variants:
            same = root.lower() in lemmas[variant.lower()]
            same_lemma_count += int(same)
            result = classify(root, variant)
            if result is None:
                if same:
                    rejected.append({"root": root, "variant": variant,
                                     "spacy_verb_lemmas": lemmas[variant.lower()],
                                     "reason": "no_supported_inflection_paradigm"})
                continue
            method, matched_tags, base = result
            output.setdefault(root, []).append(variant)
            accepted.append({"root": root, "variant": variant,
                             "inflection_tags": matched_tags, "evidence": method,
                             "compound_base": base, "spacy_verb_lemmas": lemmas[variant.lower()],
                             "spacy_lemma_matches_root": same, "context_analyses": []})

    templates = {"VBZ": ("She ", " every day."), "VBD": ("Yesterday they ", "."),
                 "VBN": ("They have ", "."), "VBG": ("They are ", ".")}
    jobs = []
    for i, record in enumerate(accepted):
        for tag in record["inflection_tags"]:
            before, after = templates[tag]
            jobs.append((before + record["variant"] + after,
                         (i, tag, len(before), len(before) + len(record["variant"]))))
    for doc, (i, expected, start, end) in nlp.pipe(jobs, as_tuples=True, batch_size=128):
        span = doc.char_span(start, end)
        if span is not None and len(span) == 1:
            t = span[0]
            accepted[i]["context_analyses"].append({"template_tag": expected,
                "text": doc.text, "lemma": t.lemma_, "pos": t.pos_,
                "tag": t.tag_, "morph": str(t.morph)})
    # Templates provide reproducible annotations, not actual corpus contexts.
    # Dictionary-confirmed forms are retained even if spaCy misses their lemma.
    for record in accepted:
        record["context_lemma_matches_root"] = any(
            a["lemma"].lower() == record["root"].lower() and a["pos"] in {"VERB", "AUX"}
            for a in record["context_analyses"])

    assert all(v in data[r] for r, vs in output.items() for v in vs)
    assert len(accepted) == sum(map(len, output.values()))
    assert len({(d["root"], d["variant"]) for d in accepted}) == len(accepted)
    assert args.input.read_bytes() == original
    dump(args.output / "verb_legal_inflections.json", output)
    for filename, records in [("verb_legal_inflections.details.jsonl", accepted),
                              ("spacy_lemma_candidates_not_selected.jsonl", rejected)]:
        with (args.output / filename).open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "input_path": str(args.input), "input_sha256": hashlib.sha256(original).hexdigest(),
        "source_root_count": len(data), "source_pair_count": sum(map(len, data.values())),
        "unique_source_variants": len(words), "spacy_same_lemma_pairs": same_lemma_count,
        "selected_root_count": len(output), "selected_pair_count": len(accepted),
        "evidence_counts": dict(Counter(x["evidence"] for x in accepted)),
        "selected_without_spacy_rule_lemma_match": sum(not x["spacy_lemma_matches_root"] for x in accepted),
        "selected_without_template_lemma_match": sum(not x["context_lemma_matches_root"] for x in accepted),
        "spacy_candidates_not_selected": len(rejected),
        "roots_absent_from_inflection_dictionary": roots_without_dictionary,
        "versions": {"spacy": spacy.__version__, "model": nlp.meta["version"],
                     "model_name": "en_core_web_lg", "lemminflect": lemminflect.__version__},
        "regression_checks_passed": len(probes), "input_unchanged": True,
        "timeseries_read_or_modified": False,
        "scope": "Pair-level verb inflection inventory; lexical possibility, not corpus-context adjudication.",
        "limitations": "Not exhaustive. Unsupported or historical forms remain unselected. Generated forms are explicitly labeled. Unselected words are not thereby proven misspellings.",
    }
    dump(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
