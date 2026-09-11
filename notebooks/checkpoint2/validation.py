"""Reproduce exported scores on CPU using saved probes, never model inference."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from offline import analyze


def export_validation(root, data, models, **analysis_kwargs):
    root = Path(root)
    run_dir = root / 'analysis/checkpoint1_compatible'
    rows, all_metrics = [], []
    for name, spec in models.items():
        path = root / 'artifacts' / name
        probe_path = path / 'ccs_probes.npz'
        before = hashlib.sha256(probe_path.read_bytes()).hexdigest()
        scores, metrics = analyze(path, spec, data, device='cpu', allow_training=False,
                                  **analysis_kwargs)
        exported = pd.read_csv(run_dir / name / 'scores.csv')
        exported_metrics = pd.read_csv(run_dir / name / 'layer_metrics.csv')
        pd.testing.assert_frame_equal(scores, exported, check_dtype=False, atol=2e-5, rtol=2e-5)
        pd.testing.assert_frame_equal(metrics, exported_metrics, check_dtype=False, atol=2e-5, rtol=2e-5)
        after = hashlib.sha256(probe_path.read_bytes()).hexdigest()
        if before != after:
            raise AssertionError(f'Validation changed saved probes: {path}')
        test = exported[(exported.split == 'test') & (exported.layer == spec.n_layers-1)]
        row = dict(model=name, dataset_size=len(data), layers=spec.n_layers,
                   score_rows=len(exported), test_size=len(data.test_idx),
                   behavioral_accuracy=float(np.mean(test.behavioral_score == test.true_label)),
                   latent_accuracy_last_layer=float(np.mean((test.latent_score > .5) == test.true_label)),
                   max_cpu_gpu_latent_difference=float(np.max(np.abs(scores.latent_score-exported.latent_score))),
                   probe_reuse_verified=True)
        if spec.chat:
            row['chat_behavioral_accuracy'] = float(np.mean(test.chat_behavioral_score == test.true_label))
        rows.append(row)
        all_metrics.append(exported_metrics.assign(model=name))
    result = dict(models=rows, protected_files_unchanged=None,
                  total_score_rows=sum(r['score_rows'] for r in rows),
                  all_layers=sum(r['layers'] for r in rows), validation='passed',
                  protected_tracked_content_unchanged=None,
                  dataset=data.identifier, dataset_sources=data.sources)
    pd.concat(all_metrics, ignore_index=True).to_csv(run_dir/'all_layer_metrics.csv', index=False)
    pd.DataFrame(rows).to_csv(run_dir/'model_summary.csv', index=False)
    (run_dir/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
