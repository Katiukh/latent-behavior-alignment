"""Run all Checkpoint 2 models on CUDA, with isolated model/analysis processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from artifacts import ARTIFACTS, MODELS, PROJECT, load_dataset

RUN_DIR = PROJECT / 'results/checkpoint2/analysis/checkpoint1_compatible'
LOG_DIR = PROJECT / 'results/checkpoint2/logs'


def run_stage(name, stage):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this run; CPU fallback is disabled')
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    data = load_dataset()
    spec = MODELS[name]
    if stage == 'inference':
        from inference import ensure_cache
        ensure_cache(ARTIFACTS / name, spec, data)
    else:
        from offline import analyze
        scores, metrics = analyze(ARTIFACTS / name, spec, data, device='cuda')
        destination = RUN_DIR / name
        destination.mkdir(parents=True, exist_ok=True)
        scores.to_csv(destination / 'scores.csv', index=False)
        metrics.to_csv(destination / 'layer_metrics.csv', index=False)
        print('Exported:', destination, 'rows:', len(scores), 'layers:', len(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--stage', choices=['inference', 'analysis'])
    parser.add_argument('--start-model', choices=MODELS, help='Resume with this model; earlier outputs stay untouched')
    parser.add_argument('--background', action='store_true', help='Detach the runner and save its output to runner.log')
    args = parser.parse_args()
    if args.background:
        if args.model or args.stage:
            parser.error('--background is for the multi-model runner')
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', '-B', str(Path(__file__).resolve())]
        if args.start_model:
            command += ['--start-model', args.start_model]
        with (LOG_DIR / 'runner.log').open('a') as output:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        print(f'Runner started: PID={process.pid}; log={LOG_DIR / "runner.log"}', flush=True)
        return
    if args.model or args.stage:
        if not (args.model and args.stage):
            parser.error('--model and --stage must be supplied together')
        run_stage(args.model, args.stage)
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    status = {'started_at': datetime.now(timezone.utc).isoformat(), 'models': {}}
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
        for stage in ('inference', 'analysis'):
            status['models'][name][stage] = 'running'
            save_status()
            log = LOG_DIR / f'{name}_{stage}.log'
            print(f'{name}: {stage}; log={log}', flush=True)
            with log.open('a') as output:
                output.write('\nSTART ' + datetime.now(timezone.utc).isoformat() + '\n')
                output.flush()
                result = subprocess.run([sys.executable, '-u', '-B', str(Path(__file__).resolve()),
                                         '--model', name, '--stage', stage], env=env,
                                        stdout=output, stderr=subprocess.STDOUT)
            status['models'][name][stage] = 'complete' if result.returncode == 0 else 'failed'
            save_status()
            if result.returncode:
                raise RuntimeError(f'{name} {stage} failed; see {log}')
    import pandas as pd
    all_metrics = pd.concat([pd.read_csv(RUN_DIR / name / 'layer_metrics.csv').assign(model=name)
                             for name in MODELS], ignore_index=True)
    all_metrics.to_csv(RUN_DIR / 'all_layer_metrics.csv', index=False)
    status['completed_at'] = datetime.now(timezone.utc).isoformat()
    save_status()
    print('All five models completed:', RUN_DIR, flush=True)


if __name__ == '__main__':
    main()
