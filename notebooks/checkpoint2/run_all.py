"""Run all Checkpoint 2 models on CUDA, with isolated model/analysis processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from artifacts import DATASETS, MODELS, load_dataset, output_root

def preflight(dataset):
    data = load_dataset(dataset=dataset)
    labels = data.raw.is_harmfull_opposition.to_numpy()
    n = len(data)
    if n % 2 or not ((labels[:n//2] == 0).all() and (labels[n//2:] == 1).all()):
        raise ValueError('PA-CCS expects harmful first half, safe opposition second half')
    for frame in (data.yes, data.no):
        if not frame.is_harmfull_opposition.equals(data.raw.is_harmfull_opposition):
            raise ValueError('Raw/yes/no labels differ in order')
    return dict(dataset=dataset, dataset_sources=data.sources, dataset_sha256=data.fingerprint,
                dataset_size=n, train_size=len(data.train_idx), test_size=len(data.test_idx),
                output_root=str(output_root(dataset)), models=list(MODELS))


def run_stage(name, stage, dataset):
    root = output_root(dataset)
    artifacts = root / "artifacts"
    run_dir = root / "analysis/checkpoint1_compatible"
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this run; CPU fallback is disabled')
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    data = load_dataset(dataset=dataset)
    spec = MODELS[name]
    if stage == 'inference':
        from inference import ensure_cache
        ensure_cache(artifacts / name, spec, data)
    else:
        from offline import analyze
        scores, metrics = analyze(artifacts / name, spec, data, device='cuda')
        destination = run_dir / name
        destination.mkdir(parents=True, exist_ok=True)
        scores.to_csv(destination / 'scores.csv', index=False)
        metrics.to_csv(destination / 'layer_metrics.csv', index=False)
        print('Exported:', destination, 'rows:', len(scores), 'layers:', len(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, default='mixed')
    parser.add_argument('--dry-run', action='store_true', help='Validate sources and print destinations without model loading')
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--stage', choices=['inference', 'analysis'])
    parser.add_argument('--start-model', choices=MODELS, help='Resume with this model; earlier outputs stay untouched')
    parser.add_argument('--background', action='store_true', help='Detach the runner and save its output to runner.log')
    args = parser.parse_args()
    report = preflight(args.dataset)
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    root = output_root(args.dataset)
    LOG_DIR = root / 'logs'
    RUN_DIR = root / 'analysis/checkpoint1_compatible'
    if args.model and not args.stage:
        parser.error('--model requires --stage')
    if args.background:
        if args.model:
            parser.error('--background is for the multi-model runner')
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--dataset', args.dataset]
        if args.stage:
            command += ['--stage', args.stage]
        if args.start_model:
            command += ['--start-model', args.start_model]
        with (LOG_DIR / 'runner.log').open('a') as output:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        print(f'Runner started: PID={process.pid}; log={LOG_DIR / "runner.log"}', flush=True)
        return
    if args.model:
        os.environ.setdefault('CHECKPOINT2_MAX_GPU_MEMORY', '12GiB')
        run_stage(args.model, args.stage, args.dataset)
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    status = {'dataset': args.dataset, 'dataset_sources': report['dataset_sources'], 'started_at': datetime.now(timezone.utc).isoformat(), 'models': {}}
    if args.start_model and (LOG_DIR / 'run_status.json').exists():
        status = json.loads((LOG_DIR / 'run_status.json').read_text())
        status.pop('completed_at', None)
        status['resumed_at'] = datetime.now(timezone.utc).isoformat()
    def save_status():
        tmp = LOG_DIR / 'run_status.tmp'
        tmp.write_text(json.dumps(status, indent=2) + '\n')
        tmp.replace(LOG_DIR / 'run_status.json')
    env = os.environ.copy()
    # All five model snapshots are already local. Avoid network checks each load.
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1',
               PYTHONDONTWRITEBYTECODE='1', CHECKPOINT2_MAX_GPU_MEMORY='12GiB')
    names = list(MODELS)
    if args.start_model:
        names = names[names.index(args.start_model):]
    for name in names:
        status['models'][name] = {}
        for stage in ([args.stage] if args.stage else ('inference', 'analysis')):
            status['models'][name][stage] = 'running'
            save_status()
            log = LOG_DIR / f'{name}_{stage}.log'
            print(f'{name}: {stage}; log={log}', flush=True)
            with log.open('a') as output:
                output.write('\nSTART ' + datetime.now(timezone.utc).isoformat() + '\n')
                output.flush()
                result = subprocess.run([sys.executable, '-u', '-B', str(Path(__file__).resolve()),
                                         '--dataset', args.dataset, '--model', name, '--stage', stage], env=env,
                                        stdout=output, stderr=subprocess.STDOUT)
            status['models'][name][stage] = 'complete' if result.returncode == 0 else 'failed'
            save_status()
            if result.returncode:
                raise RuntimeError(f'{name} {stage} failed; see {log}')
    if args.stage != 'inference':
        from validation import export_validation
        export_validation(root, load_dataset(dataset=args.dataset), MODELS)
    status['completed_at'] = datetime.now(timezone.utc).isoformat()
    save_status()
    print('Selected stages completed:', root, flush=True)


if __name__ == '__main__':
    main()
