# Lexical inventory and candidate construction

WordNet selection, lexical membership, candidate proposals and record organisation.

These are the original scientific source modules under descriptive names. Use the [workspace preparation tool](../prepare_workspace.py) before execution so that original imports and relative paths are restored. See [reproduction requirements](../docs/reproduction.md).

| Module | Source branches |
|---|---|
| [generate_llm_candidates.py](generate_llm_candidates.py) | shared |
| [organize_adverb_records.py](organize_adverb_records.py) | adverb |
| [organize_noun_records.py](organize_noun_records.py) | noun |
| [index_noun_sources.py](index_noun_sources.py) | noun |
| [record_io.py](record_io.py) | noun |
| [classify_wordnet_membership.py](classify_wordnet_membership.py) | verb |
| [extract_verb_inflections.py](extract_verb_inflections.py) | verb |
| [organize_adjective_records.py](organize_adjective_records.py) | adjective |
| [generate_short_adjective_candidates.py](generate_short_adjective_candidates.py) | adjective |
| [generate_verb_candidates_and_series.py](generate_verb_candidates_and_series.py) | verb |
| [organize_verb_records.py](organize_verb_records.py) | verb |

A part-of-speech suffix identifies an implementation difference in the original source. It does not restrict the scope of CoSpell-EN. Identical modules are stored only once.
