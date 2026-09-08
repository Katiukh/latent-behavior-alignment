# Latent–Behavior Alignment

Research project on the relationship between latent safety-related representations
and observable model behavior.

This repository currently contains the experiments and results from **Checkpoint 1**.

## Checkpoint 1

The goal of the first checkpoint is to establish a latent probing baseline
and study how the measured signal changes across model layers and model variants.

The experiments use CCS / PA-CCS-style probing on polarity pairs and compare
layer-wise results across several models.

### Models

- Gemma 2 2B
- Gemma 2 2B Instruct
- Gemma 2 9B
- Gemma 2 9B Instruct
- hate-tuned DeBERTa

## Repository structure
.
├── notebooks/
│   └── checkpoint1/
│       ├── ccs_deberta_hate_tuned.ipynb
│       ├── ccs_gemma-2-2b_fixed.ipynb
│       ├── ccs_gemma-2-2b-it_fixed.ipynb
│       ├── ccs_gemma-2-9b_fixed.ipynb
│       ├── ccs_gemma-2-9b-it_fixed.ipynb
│       └── compare_depth_categories.ipynb
│
└── results/
    └── checkpoint1/
        ├── *_scores.csv
        └── depth_comparison/

## Notebooks

The model-specific notebooks contain CCS / PA-CCS experiments and layer-wise
evaluation for each model.

`compare_depth_categories.ipynb` aggregates the results across models and
compares metrics across relative depth regions.

## Results

`results/checkpoint1/` contains exported layer-wise scores for each model.

`results/checkpoint1/depth_comparison/` contains aggregated depth-wise metrics,
mismatch statistics, model summaries, and selected examples.

## Acknowledgements

The probing implementation used in this project is based on the
[polarity-probing](https://github.com/SadSabrina/polarity-probing) repository.