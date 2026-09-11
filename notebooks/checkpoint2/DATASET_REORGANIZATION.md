# Checkpoint 2 dataset reorganization audit

The pre-existing experiment is MIXED. This change only selects dataset sources,
separates paths, adds provenance, and makes the existing summary/validation
outputs reproducible. Inference and CCS/PA-CCS mathematical code are preserved.

## Data flow

`artifacts.load_dataset(reference=REFERENCE, dataset=...)` reads all three CSVs
with `index_col=0`, preserving row order and the original fingerprint algorithm.

- `data.yes.statement` → original `extract.vectorize_df` → `X_pos`.
- `data.no.statement` → original `extract.vectorize_df` → `X_neg`.
- `data.raw.statement` → model behavioral logits.
- `data.raw.is_harmfull_opposition` → `true_label = 1 - label` for CCS/scoring.
- PA-CCS pairs second-half test indices with first-half indices offset by N/2;
  counterpart membership in train is preserved, with no pair-wise split change.

MIXED sources: `raw/mixed_dataset.csv`, `yes_no/mixed_dataset_yes.csv`,
`yes_no/mixed_dataset_no.csv`. NOT sources: `raw/not_hate_dataset.csv`,
`yes_no/not_dataset_yes.csv`, `yes_no/not_dataset_no.csv`.
All paths are relative to `/home/katyukh/projects/polarity-probing/data/`.

MIXED has 1244 rows (995 train, 249 test); NOT has 1250 (1000 train, 250 test).
The halves and label order were verified for both datasets. Dataset identifier
and resolved source paths are stored in each new `metadata.json`, alongside
the unchanged content-and-order SHA-256 algorithm. NOT cannot validate a MIXED
cache, even if someone supplies an incorrect cache directory explicitly.

## Preserved settings

| Model key | Hugging Face model | Layers including embedding | Hidden size | Dtype |
|---|---|---:|---:|---|
| gemma-2-2b | google/gemma-2-2b | 27 | 2304 | float32 |
| gemma-2-2b-it | google/gemma-2-2b-it | 27 | 2304 | float32 |
| gemma-2-9b | google/gemma-2-9b | 43 | 3584 | bfloat16 |
| gemma-2-9b-it | google/gemma-2-9b-it | 43 | 3584 | float16 |
| deberta-hate-tuned | Elron/deberta-v3-large-hate | 25 | 1024 | float32 |

- Model order remains the table order; each model runs in a separate process.
- Split: sklearn `train_test_split`, test_size=0.2, random_state=71,
  shuffle=True, no stratification, using positional indices over the selected data.
- Inference: Python/NumPy/Torch seed=42; cudnn deterministic=True, benchmark=False;
  eval mode; raw hidden states float32. Decoder last-token, DeBERTa token 0,
  every layer including embedding; original extractor max_length=512 and its
  clipping/nonfinite cleanup preserved. Source: `polarity-probing/code/extract.py`.
- Gemma: device_map=auto, eager attention, runner GPU weights limit 12GiB and
  CPU weights limit 10GiB. 9B-it low_cpu_mem_usage/trust_remote_code preserved.
- Behavioral: original plain Yes/No prompt; additional chat template for instruct
  models; raw logits for every row. Batch size 1 except 9B-it size 2; its left
  padding is set only after hidden extraction. DeBERTa truncates to 512 tokens.
- Scoring: two-logit softmax; probability strictly greater than 0.5 gives class 1.
  DeBERTa maps 0=non-hate, 1=hate. True labels and all metric names preserved.
- CCS: Python/NumPy/Torch seed=0, all layers; original CCS source loaded from
  `polarity-probing/code/ccs.py`. AdamW lr=0.015, weight_decay=0.01,
  nepochs=1500, ntries=10, batch_size=-1, linear=True,
  var_normalize=False, lambda_classification=0.0, predict_normalize=False.
- Preprocessing: independent per-vector L2 for pos/neg, subtract respective
  train median from train/test; CCS constructor additionally centers train means.
  Prediction does not subtract those internal means again. PA subsets retain
  original independent mean centering. Orientation uses train labels only.

`inference.py`, `probes.py`, and the reference repository are unchanged.
The only `offline.py` extension is `allow_training=False` for validation: it
raises before training when compatible saved probes are absent. Default analysis
behavior and the optimization/scoring formulas remain unchanged.

## Moves and checks

- `notebooks/checkpoint2/{01_build_artifacts,02_offline_analysis}.ipynb`
  moved under `notebooks/checkpoint2/mixed/`; dataset paths updated.
- Equivalent NOT notebooks use the same shared modules with DATASET="not".
- `results/checkpoint2/{artifacts,analysis,logs}` moved under `mixed/`.
- 45 existing NPZ/CSV/log files have matching before/after SHA-256 hashes.
  Five metadata files received dataset provenance; all original fields remain.
- Every migrated model cache passes validation. Every original probe signature
  still matches its raw hidden states and split. No MIXED inference or probe
  training was used to migrate or check the data.
- `results/checkpoint2/migration_mixed.json` preserves the full original manifest.
- Unit/smoke checks cover original Checkpoint 1 CCS/PA-CCS parity, raw cache and
  probe reuse, real source selection, incompatible dataset rejection, CLI dry-run,
  NOT propagation into child processes and output directories, validation exports,
  and rejection of corrupted exported scores.

NOT creates 165 model/layer combinations and 206250 score rows when all models
complete. Runtime completion and failures are recorded in
`results/checkpoint2/not/logs/run_status.json`; validation is recorded only after
CPU restoration and comparison of all five model outputs succeed.
