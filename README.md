# CoSpell-EN

Code accompanying the English spelling-variation resource covering **nouns, verbs, adjectives and adverbs**, with historical frequency trajectories spanning **1800–2022**.

## Start here

Browse the source by its role in the study. Identical modules shared by the four parts of speech are stored once. Names describe what each file does, without server experiment dates or numbered development prefixes.

| Stage | Source | Purpose |
|---|---|---|
| 1. Lexical and candidate construction | [construction/](construction/README.md) | WordNet membership, candidate proposals and record organisation |
| 2. Context collection | [collection/](collection/README.md) | C4 sentence retrieval and source tracking |
| 3. Model screening | [screening/](screening/README.md) | Grammatical error correction (GEC) and real-word spelling error (RWSE) screening |
| 4. Record validation | [validation/](validation/README.md) | Acceptance policies and single-position substitution checks |
| 5. Frequency analysis | [analysis/](analysis/README.md) | Historical trajectories and corpus-agreement plots |
| Shared execution support | [runtime/](runtime/README.md) | Inference services, caching and GPU coordination |

Part-of-speech suffixes appear only where the original implementations differ. Length-dependent candidate rules are part of the method, not the top-level repository organisation.

## Check the release

The following commands require Python 3.10 or later and no third-party packages. They do not download data or execute model inference.

```bash
python prepare_workspace.py --check
python prepare_workspace.py --list
python -m unittest discover -s tests -v
```

To prepare the source layout for one part of speech, or for all four:

```bash
python prepare_workspace.py --profile noun --output workspaces/noun
python prepare_workspace.py --profile all --output workspaces/all
```

Choose a new output directory for each command. The tool verifies source checksums and restores the original import and relative-path layout according to [config/profiles.json](config/profiles.json). **It prepares source files, not a completed dataset or a ready-to-run production environment.** Read the [reproduction guide](docs/reproduction.md) for required inputs, model versions and execution prerequisites.

## Sentence acceptance

Natural pairs and correct-sentence seeds undergo GEC and RWSE screening. GEC must produce precisely the expected correction and leave the corrected sentence unchanged. Comparison normalises whitespace only. RWSE checks supported positions outside the target and rejects a candidate when an alternative reaches at least ten times the observed word's contextual probability.

Constructed pairs replace one target position in a screened correct-sentence seed. Validation checks seed registration, acceptance evidence, exact substitution and preservation of the surrounding characters. The current code does not rerun GEC or RWSE on the injected sentence. New records do not undergo Qwen independent proofreading. Historical policy branches remain available to interpret previously accepted records.

The single-position constraint applies to all four parts of speech and permits more than one character edit within the target word. Model-screened records are a silver standard. See the [method-to-code guide](docs/method_to_code.md) for the relevant implementations.

## Release provenance

This reader-oriented layout contains **51 distinct scientific source modules**, representing **86 original module locations**. Their contents are unchanged from the [initial source snapshot](https://github.com/Computational-social-science/CoSpell-EN/tree/2e031bdf4c874f52c9962e0fa870f5d1b3164693). The original snapshot remains available in Git history. See [source provenance](docs/source_provenance.md) and the [change log](CHANGELOG.md).

Verification covers checksums, Python syntax, branch mappings and safe workspace preparation. Full GPU inference and corpus construction have not been rerun for this packaging revision.

## Data and dependencies

This repository publishes code. The full CoSpell-EN dataset, C4 text, downloaded GBNC records, checkpoints, model-output caches and production state are not included. No dataset DOI is currently provided here.

[requirements.txt](requirements.txt) lists imported third-party packages. It is a dependency inventory, not a validated environment lock. External datasets, models and libraries retain their own terms. No additional software licence is asserted in this release.
