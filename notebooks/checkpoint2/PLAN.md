# Checkpoint 2 implementation plan

Goal: preserve expensive model outputs once and rerun all downstream work offline.
Specification: the user's Checkpoint 2 request; all new work stays in checkpoint2.

Architecture: `artifacts.py` owns dataset identity, model specifications, split,
validation and atomic cache publication. `inference.py` loads one model, calls
the original extraction code and collects raw behavioral logits before release.
`offline.py` trains the original CCS class and calls its PA-CCS methods.
`probes.py` validates and restores derived probes with exact preprocessing.
Two notebooks expose these independent stages. Reference modules are executed
from source without writing bytecode into polarity-probing.

Constraints and decisions:
- Never edit Checkpoint 1 or polarity-probing.
- Preserve float32 raw extraction outputs, including reference extractor cleanup
  and clipping, before L2/median/CCS normalization. Include embedding layer.
- Same positional split: test_size=0.2, random_state=71, shuffle=True.
- Preserve checkpoint1 model dtypes, plain prompts and instruct chat variants.
- Behavioral logits cover all 1244 statements, indexed by row position.
- Four mandatory artifacts per model; chat logits are extra columns in the same NPZ.
- Updated user requirement: derived `ccs_probes.npz` stores weights, bias, actual
  train means, prediction offsets, layer, orientation, seed/config and identity.
  Compatible probes skip CCS training; changing preprocessing/split/config can
  retrain offline. Default keeps original Checkpoint 1 semantics.
- Check dataset content/order hash in addition to size, shape and model name.
- Invalid or partial caches raise an actionable error; no silent overwrite or inference.
- Publish from a temporary sibling directory only after validation succeeds.
- PA-CCS uses original second-half test indices and their first-half counterparts,
  including counterparts belonging to train. No new pair split.
- Keep original CCS internal mean-centering and prediction behavior unchanged.

## Tasks and checks

- [x] Cache: write and run failing roundtrip, split, corruption and identity tests;
  implement save/load/validate, then rerun tests.
- [x] Inference: write failing cache-hit and one-load lifecycle tests; implement
  lazy model imports and raw output collection, then rerun tests.
- [x] Offline: write failing score/threshold and reference-metric parity tests;
  implement layerwise normalization, train orientation and original PA methods.
- [x] Entry points: create inference/offline notebooks, usage guide and model
  artifact directories. Validate notebook schema and execute offline example
  on synthetic artifacts in a fresh process with model loading forbidden.
- [x] Final verification: 15 tests pass, including separate-process analysis with
  transformers imports and CCS training forbidden. Notebook schema and code
  compile checks pass. All 24 protected Checkpoint 1/reference code files retain
  their original hashes. Independent read-only review found no correctness gaps.

Tests use unittest and the existing polarity-probing virtual environment with
`python -B -m unittest discover -s notebooks/checkpoint2 -p 'test_*.py' -v`.
Full model inference is a separate user-run step; no CUDA is available here.
