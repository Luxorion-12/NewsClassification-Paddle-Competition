"""Run the three agreed learning rates sequentially; validation only."""
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import transformers
import sklearn

root = Path(__file__).resolve().parent
output = root / 'outputs' / ('lr_search_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
output.mkdir(exist_ok=False)
environment = {'python': platform.python_version(), 'executable': sys.executable,
               'torch': torch.__version__, 'transformers': transformers.__version__,
               'sklearn': sklearn.__version__, 'data_dir': str(root.parent / 'database')}
(output / 'environment.json').write_text(json.dumps(environment, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'Experiment directory: {output}', flush=True)
results = []
for lr in ['5e-5', '1e-4', '2e-4']:
    run = output / ('lr_' + lr)
    run.mkdir()
    command = [sys.executable, '-u', str(root / 'train.py'), '--data-dir', str(root.parent / 'database'),
               '--output-dir', str(run), '--epochs', '3', '--learning-rate', lr,
               '--batch-size', '64', '--max-length', '64', '--threads', '8', '--seed', '2026']
    (run / 'command.json').write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Starting learning rate {lr}', flush=True)
    start = time.perf_counter()
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    with (run / 'training.log').open('w', encoding='utf-8') as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    if process.returncode:
        raise RuntimeError(f'Training failed ({process.returncode}); see {run / "training.log"}')
    metrics = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
    results.append({'learning_rate': float(lr), 'elapsed_seconds': time.perf_counter() - start,
                    'best_epoch': metrics['best_epoch'], 'dev_accuracy': metrics['dev_accuracy'],
                    'dev_macro_f1': metrics['dev_macro_f1'], 'history': metrics['history'],
                    'output_dir': str(run)})
    (output / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(results[-1], ensure_ascii=False), flush=True)
print('All three experiments completed.', flush=True)
