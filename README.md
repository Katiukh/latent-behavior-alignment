# Latent–Behavior Alignment

Research project on the relationship between safety-related latent representations
and observable model behavior.

The project uses CCS / PA-CCS-style probing to study whether information encoded
inside a model is consistent with its behavioral outputs, and how this relationship
changes across model layers and model variants.

## Checkpoints

### Checkpoint 1 — latent vs behavioral scores

Checkpoint 1 establishes the basic latent–behavior comparison.

For each model and layer, the experiments:

- extract hidden-state representations for polarity inputs;
- train CCS probes and obtain a latent score;
- obtain a behavioral score from model logits;
- compare latent and behavioral signals across layers and models;
- analyze cases where latent and behavioral signals disagree.

The main outputs are per-model score tables and layer-wise comparisons across
relative depth categories.

### Checkpoint 2 — PA-CCS polarity metrics

Checkpoint 2 extends the analysis from individual latent/behavioral scores to
polarity-aware evaluation with PA-CCS.

The goal is to evaluate whether the learned CCS signal behaves consistently across
predefined polarity pairs, rather than only looking at the score of each statement
independently.

The same experimental pipeline is run on two dataset variants:

- `mixed` — the original Checkpoint 2 dataset;
- `not` — a second polarity dataset variant.

For both variants, the pipeline saves reusable hidden states, behavioral logits,
train/test splits and trained CCS probes, then performs PA-CCS analysis offline.
The two datasets use the same models and experimental settings and have independent
result directories.

## Models

Experiments use five model variants:

| Model key | Hugging Face model |
|---|---|
| `gemma-2-2b` | `google/gemma-2-2b` |
| `gemma-2-2b-it` | `google/gemma-2-2b-it` |
| `gemma-2-9b` | `google/gemma-2-9b` |
| `gemma-2-9b-it` | `google/gemma-2-9b-it` |
| `deberta-hate-tuned` | `Elron/deberta-v3-large-hate` |

## Dataset variants

Checkpoint 2 uses datasets from the neighboring
[`polarity-probing`](https://github.com/SadSabrina/polarity-probing) repository.

| Variant | Raw data | Yes inputs | No inputs |
|---|---|---|---|
| `mixed` | `raw/mixed_dataset.csv` | `yes_no/mixed_dataset_yes.csv` | `yes_no/mixed_dataset_no.csv` |
| `not` | `raw/not_hate_dataset.csv` | `yes_no/not_dataset_yes.csv` | `yes_no/not_dataset_no.csv` |

Each variant has its own train/test split and its own saved artifacts and analysis
results. Model settings, hidden-state extraction, normalization, CCS training,
behavioral scoring and downstream metric computation are shared across variants.

## Repository structure

```text
latent-behavior-alignment/
├── README.md
├── notebooks/
│   ├── checkpoint1/
│   │   ├── ccs_*.ipynb
│   │   └── compare_depth_categories.ipynb
│   └── checkpoint2/
│       ├── README.md
│       ├── mixed/
│       │   ├── 01_build_artifacts.ipynb
│       │   └── 02_offline_analysis.ipynb
│       ├── not/
│       │   ├── 01_build_artifacts.ipynb
│       │   └── 02_offline_analysis.ipynb
│       ├── run_all.py
│       ├── artifacts.py
│       ├── inference.py
│       ├── probes.py
│       ├── offline.py
│       └── validation.py
└── results/
    ├── checkpoint1/
    │   ├── *_scores.csv
    │   └── depth_comparison/
    └── checkpoint2/
        ├── mixed/
        │   ├── artifacts/
        │   ├── analysis/
        │   └── logs/
        └── not/
            ├── artifacts/
            ├── analysis/
            └── logs/
```

Checkpoint 2 keeps shared Python code in `notebooks/checkpoint2/`, while the
`mixed` and `not` folders contain dataset-specific notebooks. Results are stored
separately under `results/checkpoint2/<dataset>/`.

## Reproducing Checkpoint 2

The main entry point is:

```bash
../polarity-probing/.venv/bin/python -u -B notebooks/checkpoint2/run_all.py --dataset mixed
../polarity-probing/.venv/bin/python -u -B notebooks/checkpoint2/run_all.py --dataset not
```

Inference and offline analysis can also be run separately with `--stage inference`
or `--stage analysis`.

More detailed implementation, caching, validation and reproducibility notes are in
[`notebooks/checkpoint2/README.md`](notebooks/checkpoint2/README.md).

## Results

Checkpoint 1 results are stored in `results/checkpoint1/`.

Checkpoint 2 results are stored separately for each dataset:

- `results/checkpoint2/mixed/`
- `results/checkpoint2/not/`

Each Checkpoint 2 artifact directory contains the hidden states, behavioral logits,
dataset split, trained CCS probes and metadata needed to reproduce the offline
analysis without rerunning the main model inference.

## Acknowledgements

The probing implementation is based on the
[`polarity-probing`](https://github.com/SadSabrina/polarity-probing) repository.
