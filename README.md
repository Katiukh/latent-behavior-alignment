# Latent–Behavior Alignment in Language Models

## Research question

How closely does behavioral alignment agree with the alignment signal in a
model's hidden representations? This project compares observable harmfulness
judgments with layer-wise Contrast-Consistent Search (CCS) scores and examines
cases where they disagree. Polarity-Aware CCS (PA-CCS) additionally measures how
the probe responds to safe/harmful semantic oppositions.

Here, **aligned** means that the model's harmfulness prediction agrees with the
dataset label. Behavioral scores measure forced-choice judgments, not the safety
of freely generated responses. Latent scores are probe-derived signals, not a
direct observation of a model's beliefs or intentions.

## Datasets

The datasets and original CCS/PA-CCS implementation come from
[polarity-probing](https://github.com/SadSabrina/polarity-probing).

| Dataset | Statements | Safe/harmful pairs | Train / test | Polarity construction |
| --- | ---: | ---: | ---: | --- |
| `mixed` | 1,244 | 622 | 995 / 249 | Semantic opposition through rephrasing and negation |
| `not` | 1,250 | 625 | 1,000 / 250 | Closely matched oppositions formed through negation |

Each statement has Yes- and No-suffixed versions for representation extraction.
These answer pairs are distinct from the safe/harmful semantic pairs: each
semantic pair therefore supplies safe-Yes, safe-No, harmful-Yes and harmful-No.
The first dataset half is harmful, the second safe; matching counterparts are
separated by half the dataset length. `true_label = 1 - is_harmfull_opposition`:
1 means harmful and 0 means safe.

The source tables are `raw/mixed_dataset.csv`, `raw/not_hate_dataset.csv`, and
`yes_no/{mixed,not}_dataset_{yes,no}.csv` in the reference project. Their content,
row order and provenance are recorded in artifact metadata. `sample_idx` is the
stable positional index in that original order. Each dataset has its own saved
statement-level split: 20% test, shuffle enabled, random state 71. The split is
shared across models within a dataset and is not a pair-wise split.

## Models

| Model | Checkpoint | Layers including embedding output |
| --- | --- | ---: |
| Gemma 2 2B | `google/gemma-2-2b` | 27 |
| Gemma 2 2B Instruct | `google/gemma-2-2b-it` | 27 |
| Gemma 2 9B | `google/gemma-2-9b` | 43 |
| Gemma 2 9B Instruct | `google/gemma-2-9b-it` | 43 |
| DeBERTa hate classifier | `Elron/deberta-v3-large-hate` | 25 |

Layer 0 is the embedding output. Each dataset covers 165 model/layer combinations.
Metadata records extraction details, model dtype and token/class mappings. Hidden
states are stored as float32; their widths are 2,304 for Gemma 2B, 3,584 for Gemma
9B and 1,024 for DeBERTa. Existing extraction clipping and NaN cleanup are part
of the saved representations.

## Checkpoint 1

Checkpoint 1 contains the original per-model experiments: representation
extraction, CCS probing, behavioral scoring, and comparison of the two signals.
The depth comparison notebook aggregates results by relative layer depth and
records mismatch counts, shares and examples. Its notebooks and numerical
results are preserved.

## Checkpoint 2

Checkpoint 2 separates cached model outputs from offline analyses:

- **Behavioral scores:** Gemma answers “Is this statement harmful?” through a
  two-token No/Yes softmax; P(Yes) is P(harmful). DeBERTa uses the two-class
  non-hate/hate softmax, with class 1 meaning hate. Instruct models also retain
  chat scores, but the main analyses use the original non-chat scores.
- **Latent CCS scores:** a saved linear CCS probe supplies one probability per
  statement and layer. The raw score is `(p(No) + 1 - p(Yes)) / 2`; the saved
  train-selected orientation optionally complements it to obtain an oriented
  harmfulness score. Its class decision is `latent_score > 0.5`.
- **Probe artifacts:** weights, biases, prediction offsets, internal training
  means, loss, orientation, train accuracies, split and provenance are retained.
- **PA-CCS:** reference preprocessing and CCS-consistent preprocessing are kept
  separately, using the same probes and metric definitions.
- **Behavioral calibration:** one existing train-fitted threshold per
  dataset/model is fixed for evaluation.
- **Mismatch:** the all-samples baseline and calibrated test analysis have
  separate tables and figures.

The new analyses use existing artifacts; they do not perform model inference,
feature extraction, probe training or threshold optimization.

### CCS preprocessing

Representations are L2-normalized separately for each statement/layer and answer.
**Centering statistics are estimated on the train split separately for Yes and
No representations and then reused for evaluation data.** In the current
`checkpoint1` preprocessing convention these centers are coordinate-wise medians.

`ccs_probes.npz` stores them as `prediction_offset_pos` (Yes) and
`prediction_offset_neg` (No). These are distinct from `train_mean_pos/neg`: the
reference CCS constructor additionally mean-centers its training inputs, while
prediction uses the median-centered inputs without repeating that internal mean
subtraction. The saved prediction behavior is preserved exactly.

The existing probe configuration uses a linear CCS objective without supervised
classification loss, seed 0, AdamW learning rate 0.015 and weight decay 0.01,
1,500 epochs and 10 initializations; variance normalization is disabled.
Orientation is selected on train only. ESA (`accuracy`) is sign-invariant test
accuracy, whereas `latent_vs_true` evaluates the saved train-oriented score.

### PA-CCS

The three existing metrics have complementary roles:

- **Empirical Separation Accuracy (ESA):** how well the CCS direction separates
  harmful and safe statements, allowing the original global sign ambiguity.
- **Polar Consistency (PC):** the signed agreement quantity comparing opposite
  answers across semantic oppositions. It retains the reference formula and
  mean aggregation; it is not a percentage of matching labels.
- **Contradiction Index (CI):** the reference sum of products of probabilities
  for the same answer on opposing statements, averaged over pairs.

**PA-CCS reference** uses historical preprocessing: after L2 normalization, the
four safe/harmful × Yes/No groups are independently mean-centered by the original
`get_contrastive_probas`. Its code is in `pa_ccs_reference.py`, with compatible
imports retained in `offline.py`. `plot_pa_ccs_metrics_reference.py` reproduces
the historical plots and summaries from the unchanged
`analysis/checkpoint1_compatible/*/layer_metrics.csv` tables. Saved plots and
summaries are in `results/checkpoint2/pa_ccs_analysis_reference/`.

**Corrected/main PA-CCS** uses CCS-consistent preprocessing: `analyze_pa_ccs.py`
applies the saved Yes offset to both safe-Yes and harmful-Yes, and the saved No
offset to both safe-No and harmful-No. It computes no new centers and performs
no additional group centering. Original ESA/PC/CI definitions are reused; ESA
is checked against the saved CCS results. Outputs are in `pa_ccs_analysis/`.

Both versions preserve the original pair selection: the safe second-half rows
in test are paired with their first-half harmful counterparts. A counterpart
may belong to train. This convention is separate from the strictly test-only
mismatch evaluation and does not change the saved split.

The main `pa_ccs_layer_metrics.csv` has one row per dataset/model/layer, including
`esa`, `pc`, `ci`, original auxiliary metric columns and `esa_regime`. Separate
`mixed_pc_ci.png` and `not_pc_ci.png` summary scatter plots place one point per
model/layer at (PC, CI). Colors use the existing ESA ranges from
[panel E of the reference teaser](https://github.com/SadSabrina/polarity-probing/blob/main/teaser/ccs_teaser.png):
strong ESA ≥ 0.75 and 0.5 ≤ ESA < 0.75. The teaser's other polarity cases have no
published aggregate-layer classification rule in the supplied code; they are
not inferred from layer means. Model means average layers; dataset means give
each model equal weight.

`mixed_not_group_depth_means.png` compares both datasets in a 2×3 figure:
big models (Gemma 9B/9B-it) above small models (Gemma 2B/2B-it and DeBERTa),
with ESA, CI and PC as columns. Layers are grouped into four quarters of
relative depth (`layer / last_layer`, including embedding layer 0): Early
[0, 25%], Middle-early (25%, 50%], Middle-late (50%, 75%], and Late (75%, 100%].
Metrics are averaged within each model/quarter, then equally across models.
`pa_ccs_model_depth_means.csv` records per-model means and layer ranges;
`pa_ccs_group_depth_means.csv` records the plotted means and model/layer counts.
The same figure and depth-mean tables are also available in
`pa_ccs_analysis_reference/`, using the historical PA-CCS metric tables and
identical grouping, weighting and dataset colors.

### Behavioral threshold calibration

`analyze_behavioral_thresholds.py` is the existing calibration experiment.
`behavioral_threshold_analysis/behavioral_thresholds.csv` records one threshold
per dataset/model, selected by train accuracy. Existing ties are resolved by
proximity to 0.5, then the smaller threshold. The positive-class rule is
`P(harmful) >= behavioral_threshold`. Calibration results include train/test
accuracy, balanced accuracy, precision, recall, F1, confusion counts and train
accuracy curves, comparing the fitted threshold with 0.5.

The calibrated mismatch analysis reads that saved CSV. Thresholds are never
refitted, and test labels are not used for threshold selection. The latent
threshold stays at 0.5 with the existing CCS strict `>` rule.

### Mismatch analysis

The **baseline** in each dataset's
`mismatch_analysis_all_samples_baseline_threshold/` pools train and test within
each layer. It uses label-relative alignment probabilities: P(harmful) for a
harmful statement and 1 − P(harmful) for a safe statement. Both signals use
`alignment >= 0.5`; ties are assigned to aligned. The original numerical tables
and figures are retained. The script is `analyze_mismatch_cases_baseline.py`.

The **calibrated test analysis** in `mismatch_calibrated_test/` uses the existing
behavioral threshold and oriented CCS score. A signal is aligned if its predicted
class equals `true_label`. This is label-aware: a safe statement is behaviorally
aligned when P(harmful) is below the fitted threshold, not when it exceeds it.
At the behavioral boundary, calibration predicts harmful (`>=`); exactly 0.5
latent probability predicts safe (`>`). These retain the respective existing
calibration and CCS conventions; baseline ties remain unchanged.

| Behavioral prediction | Latent prediction | Saved category |
| --- | --- | --- |
| Aligned | Aligned | `both_aligned` |
| Aligned | Misaligned | `aligned_output_misaligned_latent` |
| Misaligned | Aligned | `misaligned_output_aligned_latent` |
| Misaligned | Misaligned | `both_misaligned` |

The two middle cases distinguish correct observable judgment with an opposing
latent signal from incorrect observable judgment with a label-consistent latent
signal. They describe disagreement on this classification task.

Calibrated per-sample tables preserve `sample_idx`, statement, label and layer,
and record split, both scores, thresholds, predicted classes, alignment classes
and category. Only saved test indices are included, checked against `split.npz`
for every layer. `behavioral_score` in these new tables is the continuous
probability; the old binary column is retained as `baseline_behavioral_prediction`.

Scatter plots keep behavior on x and latent on y, use raw harmfulness
probabilities, show both thresholds, and color all four categories. Separate
safe/harmful panels clarify the opposite alignment direction of the two labels;
titles identify dataset, model, layer, calibration and test split.

Counts and fractions include all four categories, including empty ones, per
model/layer. Model summaries pool **test sample × layer** observations and state
that denominator, the number of unique test objects and the number of layers;
they do not select a layer or treat repeated layer rows as distinct statements.
The baseline additionally retains alignment gaps, descriptive statistics and
ranked mismatch examples. No new tests of statistical significance, bootstrap
or confidence intervals are introduced.

## Repository structure

```text
README.md
notebooks/
  checkpoint1/                         Original experiments and depth comparison
  checkpoint2/
    mixed/, not/                       Dataset notebooks
    artifacts.py, inference.py         Dataset/cache contracts and extraction
    probes.py, offline.py              Existing CCS training/restoration pipeline
    pa_ccs_reference.py                Historical PA-CCS functions
    plot_pa_ccs_metrics_reference.py    Historical metric plots/summaries
    analyze_pa_ccs.py                   Main CCS-consistent PA-CCS analysis
    analyze_behavioral_thresholds.py    Existing train calibration experiment
    analyze_mismatch_cases_baseline.py  Original all-samples mismatch
    analyze_mismatch_calibrated.py      Calibrated test mismatch
    validation.py, test_*.py            Artifact and analysis checks
results/
  checkpoint1/                         Preserved original results
  checkpoint2/
    mixed/, not/
      artifacts/<model>/               Cached states, logits, split, CCS probes
      analysis/checkpoint1_compatible/ Original CCS scores and reference PA metrics
      mismatch_analysis_all_samples_baseline_threshold/
      logs/                            Existing experiment status and logs
    pa_ccs_analysis_reference/          Preserved historical PA plots/tables
    pa_ccs_analysis/                    Main layer metrics and summary plots
    behavioral_threshold_analysis/     Existing fitted thresholds and evaluations
    mismatch_calibrated_test/          Test tables, summaries and plots
```

## Results and artifacts

`hidden_states.npz` stores `X_pos`/`X_neg` as `(statement, layer, hidden_dim)`;
`behavioral_logits.npz` retains raw logits and sample indices before thresholding.
`split.npz` records the original train/test positions. `ccs_probes.npz` contains
the probe parameters and exact preprocessing state. Metadata and hashes bind
these to the model, dataset order, hidden states, split and reference CCS source.
These binary artifacts are local caches excluded from Git by the existing
`*.npz` rule. The reference source and datasets are in the adjacent
`polarity-probing` checkout; provenance paths remain in artifact metadata.

The original `scores.csv` and `layer_metrics.csv` tables remain unchanged.
PA-CCS exports retain per-layer metrics, model/dataset means and plots. Behavioral
calibration retains fitted thresholds and their evaluations. Baseline combined
summaries are named `mismatch_{summary,fractions}_all_samples_baseline_threshold.csv`.
Calibrated mismatch stores per-model object tables, layer/model summaries,
thresholds and plots, with source hashes and decision rules in `manifest.json`.

Existing `validation.json` files document CPU restoration of saved probes against
the original exports; `migration_mixed.json` records the earlier artifact
migration. The current cleanup retains historical results and checks saved-center
preprocessing, reference preservation, threshold reuse and test-only evaluation.
