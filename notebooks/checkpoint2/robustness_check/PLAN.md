# Pair-aware CV implementation plan

Goal: test whether the CCS latent direction generalizes to unseen polar pairs.
Spec: user-supplied protocol in this session; all additions remain in robustness_check.

Architecture: read validated raw checkpoint2 artifacts, create dataset-level folds,
fit separate train-only states, evaluate immutable states, aggregate fold metrics,
and plot shared-scale TRAIN | TEST panels. Reuse reference CCS and corrected PA
probabilities, checkpoint1 L2/median helpers, behavioral threshold helpers and
existing category names/colors. No existing probes or results are overwritten.

- [ ] Pairing and persisted folds: honor explicit pair_id when available, otherwise
  validated first/second-half reference mapping. KFold(5, shuffle=True, seed=71).
  Test pair integrity, deterministic reload, malformed persisted assignments.
- [ ] Fold fitting and evaluation: train-only median offsets, original CCS
  constructor's additional train mean normalization (preserved), repeated_train
  selection by train loss, threshold curve and orientation on train only.
  Freeze prediction state; evaluation has no fitting methods. Test held-out
  perturbations cannot change learned state or predictions on other samples.
- [ ] Outputs: per-fold metrics/counts, train audit/probes, exactly-once test OOF,
  equal-fold mean/std (ddof=1), plots for accuracy/mismatch/ESA/PC/CI.
  Test reference metric parity, missing/duplicate OOF and incomplete folds.
- [ ] End-to-end: one real dataset/model, all layers and five folds with default
  CCS hyperparameters if practical; document command and runtime. Validate
  output coverage/ranges, plots, and unchanged hashes of all existing files.

Review focus: explicit IDs with reordered samples; corrupt persisted folds;
held-out perturbations; ESA's sign invariance versus train orientation;
partial or repeated runs must not silently mix incompatible configurations.

Execution: inline in the existing checkout, confined to the requested folders.
