# Source provenance

The scientific Python files are byte-for-byte identical to the sanitised source in [the initial release](https://github.com/Computational-social-science/CoSpell-EN/tree/2e031bdf4c874f52c9962e0fa870f5d1b3164693). This revision changes their published names and organisation and adds reader tools. It does not change the algorithms or model decisions.

SHA-256 checksums and every original location are recorded in [source_map.json](../config/source_map.json). The [profiles](../config/profiles.json) restore each selected branch. Dates and LongWord/ShortWord occur in that compatibility mapping because the original scripts use relative imports and historical state paths.

| Published source | Original module | Copies represented |
|---|---|---|
| [runtime/inference_pool.py](../runtime/inference_pool.py) | `pool_support.py` | 5 |
| [runtime/inference_worker.py](../runtime/inference_worker.py) | `service_worker.py` | 1 |
| [runtime/gpu_resources_service.py](../runtime/gpu_resources_service.py) | `shared_gpu.py` | 1 |
| [construction/generate_llm_candidates.py](../construction/generate_llm_candidates.py) | `20250612LLMsCommonMisspellings.py` | 1 |
| [analysis/plot_corpus_agreement.py](../analysis/plot_corpus_agreement.py) | `10-plot_C4_GBNC.py` | 1 |
| [analysis/compute_frequency_series.py](../analysis/compute_frequency_series.py) | `6-compute_word_series.py` | 1 |
| [construction/organize_adverb_records.py](../construction/organize_adverb_records.py) | `build_adv_dataset.py` | 1 |
| [collection/c4_collector.py](../collection/c4_collector.py) | `c4_gate_collector.py` | 4 |
| [collection/collect_contexts_adverb.py](../collection/collect_contexts_adverb.py) | `collect_sources.py` | 1 |
| [runtime/decode_service.py](../runtime/decode_service.py) | `decode_service.py` | 4 |
| [collection/collection_state_adverb.py](../collection/collection_state_adverb.py) | `expansion_common.py` | 1 |
| [runtime/cache_and_io_adverb.py](../runtime/cache_and_io_adverb.py) | `experiment_common.py` | 1 |
| [runtime/inference_client.py](../runtime/inference_client.py) | `model_runtime_client.py` | 4 |
| [collection/retained_form_coverage.py](../collection/retained_form_coverage.py) | `retained_coverage.py` | 1 |
| [validation/acceptance_policy.py](../validation/acceptance_policy.py) | `review_policy.py` | 4 |
| [screening/screen_gec_verb_adverb.py](../screening/screen_gec_verb_adverb.py) | `run_gec.py` | 2 |
| [screening/screen_rwse_adverb.py](../screening/screen_rwse_adverb.py) | `run_rwse.py` | 1 |
| [screening/prepare_confusion_sets.py](../screening/prepare_confusion_sets.py) | `rwse_preparation.py` | 4 |
| [collection/select_scan_inputs_adverb.py](../collection/select_scan_inputs_adverb.py) | `scan_gate.py` | 1 |
| [collection/collect_shared_contexts.py](../collection/collect_shared_contexts.py) | `shared_collect.py` | 4 |
| [runtime/gpu_resources_noun_verb_adjective_adverb.py](../runtime/gpu_resources_noun_verb_adjective_adverb.py) | `shared_gpu.py` | 4 |
| [collection/stream_sentence_blocks.py](../collection/stream_sentence_blocks.py) | `stream_blocks.py` | 4 |
| [validation/validate_seed_substitution.py](../validation/validate_seed_substitution.py) | `synthetic_admission.py` | 4 |
| [validation/validate_sentence_records_adverb.py](../validation/validate_sentence_records_adverb.py) | `validate_expansion.py` | 1 |
| [construction/organize_noun_records.py](../construction/organize_noun_records.py) | `build_noun_dataset.py` | 1 |
| [construction/index_noun_sources.py](../construction/index_noun_sources.py) | `index_noun_source.py` | 1 |
| [collection/collect_contexts_noun.py](../collection/collect_contexts_noun.py) | `collect_sources.py` | 1 |
| [construction/record_io.py](../construction/record_io.py) | `dataset_io.py` | 1 |
| [collection/collection_state_noun.py](../collection/collection_state_noun.py) | `expansion_common.py` | 1 |
| [runtime/cache_and_io_noun.py](../runtime/cache_and_io_noun.py) | `experiment_common.py` | 1 |
| [collection/normalize_c4_prefixes.py](../collection/normalize_c4_prefixes.py) | `repair_c4_prefixes.py` | 1 |
| [screening/screen_gec_noun_adjective.py](../screening/screen_gec_noun_adjective.py) | `run_gec.py` | 2 |
| [screening/screen_rwse_noun.py](../screening/screen_rwse_noun.py) | `run_rwse.py` | 1 |
| [collection/select_scan_inputs_noun_verb_adjective.py](../collection/select_scan_inputs_noun_verb_adjective.py) | `scan_gate.py` | 3 |
| [validation/validate_sentence_records_noun.py](../validation/validate_sentence_records_noun.py) | `validate_expansion.py` | 1 |
| [construction/classify_wordnet_membership.py](../construction/classify_wordnet_membership.py) | `filter_wordnet_iv.py` | 1 |
| [construction/extract_verb_inflections.py](../construction/extract_verb_inflections.py) | `extract_verb_inflections.py` | 1 |
| [construction/organize_adjective_records.py](../construction/organize_adjective_records.py) | `build_adj_dataset.py` | 1 |
| [collection/collect_contexts_adjective.py](../collection/collect_contexts_adjective.py) | `collect_sources.py` | 1 |
| [collection/collection_state_adjective.py](../collection/collection_state_adjective.py) | `expansion_common.py` | 1 |
| [runtime/cache_and_io_adjective.py](../runtime/cache_and_io_adjective.py) | `experiment_common.py` | 1 |
| [screening/screen_rwse_adjective.py](../screening/screen_rwse_adjective.py) | `run_rwse.py` | 1 |
| [validation/validate_sentence_records_adjective.py](../validation/validate_sentence_records_adjective.py) | `validate_expansion.py` | 1 |
| [construction/generate_short_adjective_candidates.py](../construction/generate_short_adjective_candidates.py) | `run_short_adjectives.py` | 1 |
| [construction/generate_verb_candidates_and_series.py](../construction/generate_verb_candidates_and_series.py) | `20250517-timeseries.py` | 1 |
| [construction/organize_verb_records.py](../construction/organize_verb_records.py) | `build_verb_dataset.py` | 1 |
| [collection/collect_contexts_verb.py](../collection/collect_contexts_verb.py) | `collect_sources.py` | 1 |
| [collection/collection_state_verb.py](../collection/collection_state_verb.py) | `expansion_common.py` | 1 |
| [runtime/cache_and_io_verb.py](../runtime/cache_and_io_verb.py) | `experiment_common.py` | 1 |
| [screening/screen_rwse_verb.py](../screening/screen_rwse_verb.py) | `run_rwse.py` | 1 |
| [validation/validate_sentence_records_verb.py](../validation/validate_sentence_records_verb.py) | `validate_expansion.py` | 1 |
