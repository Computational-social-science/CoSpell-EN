# Model screening

GEC and RWSE checks for natural pairs and correct-sentence seeds.

These are the original scientific source modules under descriptive names. Use the [workspace preparation tool](../prepare_workspace.py) before execution so that original imports and relative paths are restored. See [reproduction requirements](../docs/reproduction.md).

| Module | Source branches |
|---|---|
| [screen_gec_verb_adverb.py](screen_gec_verb_adverb.py) | verb, adverb |
| [screen_rwse_adverb.py](screen_rwse_adverb.py) | adverb |
| [prepare_confusion_sets.py](prepare_confusion_sets.py) | noun, verb, adjective, adverb |
| [screen_gec_noun_adjective.py](screen_gec_noun_adjective.py) | noun, adjective |
| [screen_rwse_noun.py](screen_rwse_noun.py) | noun |
| [screen_rwse_adjective.py](screen_rwse_adjective.py) | adjective |
| [screen_rwse_verb.py](screen_rwse_verb.py) | verb |

A part-of-speech suffix identifies an implementation difference in the original source. It does not restrict the scope of CoSpell-EN. Identical modules are stored only once.
