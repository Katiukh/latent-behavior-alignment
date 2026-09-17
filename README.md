# Latent–Behavior Alignment in Language Models

Research project studying whether **safety-related information encoded in a model's internal representations is consistent with its observable behavior**.

The project compares latent alignment signals obtained with **Contrast-Consistent Search (CCS)** and **Polarity-Aware CCS (PA-CCS)** against behavioral predictions produced by the same models.

The main question is:

> Can a model produce an aligned behavioral response while its internal representation points in the opposite direction — or vice versa?

This repository contains the experimental pipeline, reusable model artifacts, layer-wise analysis, and mismatch analysis used to investigate this question.

---

## Research goals

The project focuses on three related tasks:

1. **Measure latent alignment across model layers** using CCS-based probes.
2. **Compare latent and behavioral signals** on the same examples.
3. **Identify latent–behavior mismatches**, including:

   * aligned behavior + misaligned latent representation;
   * misaligned behavior + aligned latent representation.

In addition, PA-CCS is used to evaluate whether latent representations remain consistent when the semantic polarity of an input is reversed.

---

## Method

For every model and selected layer, the pipeline:

1. builds paired positive / negative polarity inputs;
2. extracts hidden-state representations;
3. applies L2 normalization and train-derived preprocessing;
4. trains a linear CCS probe;
5. obtains a latent probability / alignment score;
6. extracts behavioral logits from the model;
7. converts them into behavioral probabilities;
8. compares latent and behavioral predictions;
9. evaluates polarity-aware metrics and mismatch cases.

The implementation separates expensive model inference from offline analysis, so CCS probes, thresholds, metrics, and mismatch analyses can be recomputed without loading the original language model again.

---

## Models

Experiments currently include:

| Model                   | Hugging Face checkpoint       |
| ----------------------- | ----------------------------- |
| Gemma 2 2B              | `google/gemma-2-2b`           |
| Gemma 2 2B Instruct     | `google/gemma-2-2b-it`        |
| Gemma 2 9B              | `google/gemma-2-9b`           |
| Gemma 2 9B Instruct     | `google/gemma-2-9b-it`        |
| DeBERTa hate classifier | `Elron/deberta-v3-large-hate` |

The analysis is performed layer-wise, including the embedding output.

---

## Datasets

The experiments use polarity datasets from the
[polarity-probing](https://github.com/SadSabrina/polarity-probing) project.

Two dataset variants are used:

### `mixed`

Contains harmful / safe pairs expressed through different types of semantic opposition.

* 1,244 examples
* 622 polarity pairs

### `not`

Contains tightly matched sentence pairs where polarity is changed primarily through negation.

* 1,250 examples
* 625 polarity pairs

Each dataset uses an independent train/test split and independent cached artifacts.

---

## Latent alignment

Latent representations are evaluated with **Contrast-Consistent Search (CCS)**.

For each layer, a linear probe is trained on paired representations corresponding to opposite answers / polarities.

The resulting scalar is used as a latent alignment signal and can be compared directly with model behavior on the same examples.

The pipeline stores probe parameters and preprocessing statistics so latent scores can be reconstructed without retraining or rerunning model inference.

---

## Behavioral alignment

Behavioral scores are derived directly from model logits.

For generative Gemma models, the score is computed from the logits associated with the relevant **Yes / No** answer tokens.

For the DeBERTa classifier, the score is derived from the logits of the non-hate / hate classes.

Raw logits are stored before thresholding, which allows behavioral thresholds to be analyzed independently from CCS training.

---

## Polarity-Aware CCS

The second part of the project extends CCS using **Polarity-Aware CCS (PA-CCS)**.

PA-CCS evaluates whether the learned latent signal behaves consistently under semantic polarity inversion.

The analysis includes:

* **Empirical Separation Accuracy (ESA)**
* **Polar Consistency**
* **Contradiction Index**

These metrics provide complementary information about whether a latent direction separates concepts and whether this separation remains structurally consistent across polarity pairs.

The PA-CCS implementation builds on the original
[polarity-probing](https://github.com/SadSabrina/polarity-probing) codebase.

---

## Latent–behavior mismatch analysis

A central part of the project is object-level comparison between latent and behavioral signals.

Each example can be placed into one of four regions:

| Behavioral signal | Latent signal | Interpretation                           |
| ----------------- | ------------- | ---------------------------------------- |
| high              | high          | aligned latent + aligned behavior        |
| high              | low           | **aligned behavior + misaligned latent** |
| low               | high          | **misaligned behavior + aligned latent** |
| low               | low           | both signals indicate misalignment       |

The mismatch regions are especially interesting because they reveal cases where observable model behavior alone may not reflect what is encoded internally.

For each region, the analysis can aggregate:

* number and fraction of examples;
* behavioral probability;
* latent probability;
* absolute latent–behavior gap;
* layer and model information.

Both fixed baseline thresholds and separately calibrated behavioral thresholds can be studied.

---

## Experimental pipeline

The Checkpoint 2 pipeline is organized into two stages.

### 1. Inference

Loads the model and saves reusable artifacts:

* hidden states;
* raw behavioral logits;
* train/test split;
* metadata.

### 2. Offline analysis

Runs without loading the original LLM and performs:

* CCS training / reconstruction;
* latent scoring;
* behavioral scoring;
* PA-CCS metrics;
* layer-wise aggregation;
* mismatch analysis.

This design makes threshold analysis and probe experiments significantly cheaper to iterate on.

---

## Repository structure

```text
latent-behavior-alignment/
├── README.md
│
├── notebooks/
│   ├── checkpoint1/
│   │   ├── ccs_*.ipynb
│   │   └── compare_depth_categories.ipynb
│   │
│   └── checkpoint2/
│       ├── mixed/
│       │   ├── 01_build_artifacts.ipynb
│       │   └── 02_offline_analysis.ipynb
│       │
│       ├── not/
│       │   ├── 01_build_artifacts.ipynb
│       │   └── 02_offline_analysis.ipynb
│       │
│       ├── run_all.py
│       ├── artifacts.py
│       ├── inference.py
│       ├── probes.py
│       ├── offline.py
│       └── val
```
