# CoSpell-EN

Source code for constructing and analysing English spelling-variation records across **nouns, verbs, adjectives and adverbs**.

This repository contains the **25 October 2026 source snapshot**. [CoSpell-EN_source_20261025.zip](CoSpell-EN_source_20261025.zip) contains 51 distinct source modules selected for the scientific workflow. Identical modules shared by the four branches are stored once. The supplied layout manifest restores them to 86 original module locations. Historical launch scripts, experiment notebooks, duplicated plots and standalone Qwen-review runners are omitted. [source_manifest.json](source_manifest.json) records each published file's SHA-256 digest and its corresponding source digest. [SHA256SUMS.txt](SHA256SUMS.txt) identifies the archive.

## Scope and current sentence-acceptance procedure

The code covers lexical organisation, WordNet membership, candidate generation and filtering, Google Books Ngram Corpus (GBNC) retrieval and frequency processing, C4/EN sentence collection, sentence-pair validation, and plotting. The noun, verb, adjective and adverb branches share a single-target-position constraint. The directory names retain the lengths and dates used in the source project and do not restrict the dataset to adverbs.

The current sentence procedure distinguishes two routes:

1. **Natural pairs and correct-sentence seeds.** CoEdIT performs grammatical error correction (GEC). A natural pair must yield exactly the expected correction, and its corrected sentence must remain unchanged when checked again. Comparison normalises whitespace only. Correct-sentence seeds must remain unchanged. BERT then screens supported positions outside the target for residual real-word spelling errors (RWSE), rejecting a candidate when an alternative reaches at least ten times the observed word's contextual probability.
2. **Constructed pairs.** A retained variant replaces one target position in an accepted correct-sentence seed. The admission code verifies the registered seed, its acceptance evidence, the precise substitution and preserved surrounding characters. It does **not** rerun GEC or RWSE on the injected sentence.

New model-screened records use `gec_rwse_without_qwen_20261022`. Qwen independent proofreading is disabled for these records. Historical validation branches remain in the snapshot because existing records can retain older acceptance evidence. Their presence does not mean that new records undergo Qwen review. Constructed records use `verified_root_direct_substitution_20261019` and retain their seed's validation history.

The one-position constraint permits a character edit distance greater than one. Model screening has finite coverage and does not independently certify contextual part of speech or the absence of every linguistic error. The resulting materials are a silver standard, not human-confirmed gold labels.

## Code map

Paths below are relative to the extracted `Vocabulary/` directory.

| Component | Location |
|---|---|
| Noun record organisation | `LongWord/NOUN/organized_nouns_wordnet_20261018/` |
| Noun sentence pipeline | `LongWord/NOUN/parallel_corpus_streaming_20261019/` |
| Adverb record organisation | `LongWord/ADV/organized_adv_wordnet_20261011/` |
| Adverb sentence pipeline | `LongWord/ADV/parallel_corpus_streaming_20261014/` |
| Adjective candidates and GBNC retrieval | `ShortWord/ADJ/run_short_adjectives.py` |
| Adjective sentence pipeline | `ShortWord/ADJ/parallel_corpus_streaming_20261015/` |
| Verb records and sentence pipeline | `ShortWord/VERB/organized_verbs_wordnet_20261015/` and `ShortWord/VERB/parallel_corpus_streaming_20261015/` |
| Verb WordNet and inflection filtering | `LongWord/VERB/verb_difference_wordnet_iv_20261011_01/` and `LongWord/VERB/verb_inflections_spacy_20261011_01/` |
| C4 collection support | `collect_sources.py`, `c4_gate_collector.py` and shared collection modules in each sentence branch |
| Shared inference implementation | `.shared_model_runtime_20261019/` |
| Candidate-generation prompt | `DataAnalysis/20250612LLMsCommonMisspellings.py` |
| Cross-corpus frequency analysis | `LongWord/ADV/10-plot_C4_GBNC.py` |

Within each sentence pipeline, start by reading `review_policy.py`, `synthetic_admission.py`, `run_gec.py`, `run_rwse.py` and `validate_expansion.py`. Collection is implemented in `collect_sources.py` and its included support modules. Server launchers are not part of this selected release.

## Environment and inputs

These are research scripts from a Linux/CUDA workflow, supplied as a source snapshot rather than a self-contained executable dataset release. They require the corresponding lexical inputs, C4 shards, GBNC records, confusion sets, model checkpoints and processing manifests. These resources are not bundled in this code repository.

The GEC scripts specify `grammarly/coedit-large` revision `5637bcdf9d8d4419f97c8cfea36f7d35c79232b6`. The RWSE scripts specify `google-bert/bert-base-cased` revision `cd5ef92a9fb2f889e972770a36d4ed042daf221e`. GEC uses greedy decoding, a 256-token input limit and a 320-token generation limit. See the scripts for the exact prompt and rejection conditions. Candidate-generation scripts have their own model configurations and must not be confused with the sentence-screening configuration.

[requirements-source.txt](requirements-source.txt) lists imported third-party Python packages. It is a dependency inventory, not a verified version lock. GPU inference and large-scale collection have not been rerun as part of packaging this snapshot.

Before running a stage:

1. Restore the branch layout with `python materialize_source.py --output workdir`. This copies identical shared modules into their original locations without changing their contents. The output directory must not already exist.
2. Supply the source datasets and checkpoints under their respective providers' access conditions.
3. Configure the script's input/output locations. Workstation-specific home paths have been replaced by `/path/to/cospell`, `/path/to/cospell-home` or `/path/to/local` placeholders.
4. Configure experiment manifests and shared inference settings for the local environment. The original scheduler checks paths, user IDs and file fingerprints. Server-specific runtime configurations and historical state are intentionally excluded. The collection and inference stages therefore require local configuration before execution.
5. Run the applicable collection, GEC/RWSE screening or seed-based construction, and validation stages in that order using prepared inputs for the selected branch. Preserve acceptance policies and source provenance in any resulting release.

To verify the distributed archive and Python syntax without running the research pipeline:

```bash
python verify_source.py
```

## Data availability

This is a **code release**. It does not contain the full CoSpell-EN dataset, original C4 text, downloaded GBNC records, model weights, model-output caches or a data DOI. A dataset deposit must identify its own files, release version and provenance. Code availability alone should not be cited as public availability of the complete dataset.

External data, models and libraries remain subject to their respective terms. No additional software licence is asserted in this snapshot.
