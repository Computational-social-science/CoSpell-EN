# Reproduction guide

## What this release provides

The repository provides the scientific source used in the current four-part-of-speech workflow. The public tree is organised by processing stage. A compatibility tool restores the original source layout before execution because several modules depend on their original import names, relative paths and historical processing state.

`prepare_workspace.py` is a release utility, not an end-to-end experiment runner. It does not create missing datasets, caches, accepted-seed registries or model configurations. Full reconstruction of a production run requires those inputs.

## Prepare and inspect

1. Download or clone this repository.
2. Run `python prepare_workspace.py --check` and `python -m unittest discover -s tests -v`.
3. Run `python prepare_workspace.py --profile all --output workspaces/all`, or select `noun`, `verb`, `adjective` or `adverb`.
4. Use [source provenance](source_provenance.md) to find each module's restored location. Configure inputs and paths in the prepared workspace, keeping the distributed source unchanged for comparison.

The output directory must not already exist. Profile preparation includes shared runtime modules and required source references. It checks source hashes before writing files. A `workspace_info.json` records the selected profile and baseline commit.

## Inputs and outputs by stage

| Stage | Required inputs | Principal outputs |
|---|---|---|
| Lexical selection and record organisation | NLTK WordNet resources and branch-specific lexical/candidate records | Canonical-form inventories and organised canonical–variant relations |
| Candidate generation and GBNC processing | Canonical forms, candidate-generation checkpoint, GBNC access or retrieved records | Candidate forms, retained relations and annual frequency series |
| Context collection | Retained forms, C4/EN shards, collection configuration and processing state | Sentence candidates, target spans and source provenance |
| GEC/RWSE screening | Natural sentence pairs or correct-sentence seeds, model checkpoints and confusion sets | Model outputs, cached decisions and acceptance reports |
| Seed-based construction | Retained variants, registered accepted seeds and their validation evidence | Single-position substituted pairs with seed provenance |
| Final validation | Candidate records, source sentences, acceptance decisions and caches | Validated records and audit summaries |
| Frequency analysis | Prepared C4 counts and GBNC annual counts | Corpus-agreement figures and coverage summaries |

The original implementation uses more than one branch-specific file schema. The source modules define those schemas. The profile configuration selects source variants and destinations, not experimental hyperparameters.

## Environment and models

The production scripts were written for Linux with CUDA. Install the relevant packages from [requirements.txt](../requirements.txt) in a suitable environment. This list is not an exact lockfile and should not be treated as proof of version-identical reproduction.

| Screening model | Repository | Revision |
|---|---|---|
| GEC | `grammarly/coedit-large` | `5637bcdf9d8d4419f97c8cfea36f7d35c79232b6` |
| RWSE | `google-bert/bert-base-cased` | `cd5ef92a9fb2f889e972770a36d4ed042daf221e` |

GEC uses greedy decoding, a 256-token input limit and a 320-token output limit. Candidate-generation scripts have separate models and parameters. Their settings must not be replaced with the screening settings.

Local checkpoint locations, corpus locations, SQLite state, confusion sets and inference-service configuration must be supplied. Sanitised `/path/to/...` values and remaining machine-specific model paths in the preserved scripts are not portable defaults. The shared runtime also uses local configuration, process ownership and fingerprint checks. Configure these for the new environment before execution.

## Validation policies

Current natural-pair and seed screening uses `gec_rwse_without_qwen_20261022`. Seed-based construction uses `verified_root_direct_substitution_20261019`. Historical policy identifiers are retained because older records may carry different acceptance evidence.

Use collection, screening and validation in that order for model-screened records. Constructed pairs require already accepted seeds and their registered evidence. Do not bypass missing-state checks or replace acceptance evidence with a fabricated pass flag.

## Verification boundary

The packaging checks establish that the published scientific source is unchanged and that profiles reconstruct the selected original source files. They do not establish availability of the external inputs, successful GPU execution or reproduction of the paper's numerical results. No data, checkpoints or synthetic research results are bundled as substitutes.
