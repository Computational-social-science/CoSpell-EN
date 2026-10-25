# Method-to-code guide

The resource covers nouns, verbs, adjectives and adverbs. A suffix in a filename identifies a branch-specific implementation, not the scope of the paper.

| Methodological component | Where to read |
|---|---|
| WordNet membership and lexical organisation | [construction](../construction/README.md), especially `classify_wordnet_membership.py` and the record organisers |
| Language-model candidate proposals | [generate_llm_candidates.py](../construction/generate_llm_candidates.py) |
| Short adjective candidate processing | [generate_short_adjective_candidates.py](../construction/generate_short_adjective_candidates.py) |
| Candidate and historical-series processing | [construction](../construction/README.md) and [compute_frequency_series.py](../analysis/compute_frequency_series.py) |
| Sentence extraction and collection | [collection](../collection/README.md) |
| Exact GEC correction and corrected-sentence stability | The `screen_gec` modules in [screening](../screening/README.md) |
| Residual real-word error screening | The `screen_rwse` modules and `prepare_confusion_sets.py` in [screening](../screening/README.md) |
| Current and historical acceptance policies | [acceptance_policy.py](../validation/acceptance_policy.py) |
| Registered-seed evidence and exact target substitution | [validate_seed_substitution.py](../validation/validate_seed_substitution.py) |
| Final source, cache and record checks | The `validate_sentence_records` modules in [validation](../validation/README.md) |
| C4/GBNC agreement and temporal coverage analysis | [plot_corpus_agreement.py](../analysis/plot_corpus_agreement.py) |

The plotting script retains the input conventions of the original analysis. Applying it to a different selection requires supplying the corresponding counts. A plotting implementation alone does not establish that every part of speech shares an identical numerical result.

All retained scientific modules can be compared directly with their original locations using [source_map.json](../config/source_map.json). Read the [reproduction guide](reproduction.md) before attempting execution.
