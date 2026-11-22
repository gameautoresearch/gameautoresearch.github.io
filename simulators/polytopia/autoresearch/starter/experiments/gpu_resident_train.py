"""Train one run of the starter recipe on one GPU (rl.gpu_resident.GpuResidentTrainer). The harness (`ar run`) calls this
for you; you edit it only to add command-line options of your own.

    python experiments/gpu_resident_train.py --config recipe.json --arch arch.json --run-id NAME --seconds 600

Prints one JSON line per update and a final `RESULT {...}` line.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='LearnerConfig JSON (rl/gpu_resident.py)')
    ap.add_argument('--arch', default='', help='architecture JSON (network.init options: width, blocks, ...)')
    ap.add_argument('--run-id', default=None, help='run folder name under --run-root')
    ap.add_argument('--run-root', default='/results/runs')
    ap.add_argument('--seconds', type=float, default=600., help='stop after this wall time (training_seconds)')
    ap.add_argument('--updates', type=int, default=0, help='stop after this many updates')
    ap.add_argument('--seed', type=int, default=None, help="overrides the config's seed")
    ap.add_argument('--games', type=int, default=4096, help='games stepped in parallel')
    ap.add_argument('--sets', type=int, default=4, help='engine batches stepped in a round robin (games // sets each)')
    ap.add_argument('--ring', type=int, default=160, help='steps an open segment stays addressable before a forced cut')
    ap.add_argument('--batch', type=int, default=None, help="rows per update (default: the config's)")
    ap.add_argument('--minibatch', type=int, default=None)
    ap.add_argument('--grad-accum', type=int, default=None)
    ap.add_argument('--checkpoint-every', type=int, default=None)
    ap.add_argument('--snapshot-every', type=int, default=None)
    ap.add_argument('--map-table', default=None, help='precomputed map table .npz (default: found by name)')
    ap.add_argument('--fp32-trunk', action='store_true', help='fp32 convolutions instead of bf16 operands')
    ap.add_argument('--matmul-precision', default='DEFAULT', choices=('DEFAULT', 'HIGH', 'HIGHEST'),
                    help='POLYTOPIA_MATMUL_PRECISION for the conv torso (DEFAULT = TF32)')
    ap.add_argument('--head-precision', default='HIGHEST', choices=('HIGHEST', 'DEFAULT'))
    ap.add_argument('--jax-mem-fraction', default='.45', help="JAX's share of the GPU memory (torch owns the engine's)")
    return ap.parse_args(argv)


def main(argv=None, commit=None):
    args = parse_args(argv)
    # torch and JAX share the card; the network module reads the precision settings at import
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(args.jax_mem_fraction)
    os.environ.setdefault('JAX_PLATFORMS', 'cuda')
    os.environ['POLYTOPIA_MATMUL_PRECISION'] = args.matmul_precision
    os.environ['POLYTOPIA_HEAD_PRECISION'] = args.head_precision
    if not args.fp32_trunk:
        os.environ['POLYTOPIA_TRUNK_DTYPE'] = 'bfloat16'
    import torch
    import jax
    if not torch.cuda.is_available() or jax.default_backend() != 'gpu':
        print('RESULT ' + json.dumps({'ok': False, 'error': 'needs a CUDA GPU for torch and JAX'}))
        sys.exit(2)
    from polytopia_bot.rl.gpu_resident import GpuResidentTrainer
    config = json.loads(Path(args.config).read_text())
    config.pop('_doc', None)
    config.pop('trainer', None)
    if args.updates:
        config['training_updates'] = int(args.updates)
    network = json.loads(Path(args.arch).read_text()) if args.arch else {}
    network = {k: v for k, v in network.items() if not k.startswith('_')}
    run_dir = Path(args.run_root) / (args.run_id or time.strftime('run-%m%d-%H%M'))
    trainer = GpuResidentTrainer(config, run_dir, network=network, training_seconds=args.seconds, games=args.games,
                                 sets=args.sets, ring=args.ring, minibatch=args.minibatch, batch=args.batch,
                                 grad_accum=args.grad_accum, checkpoint_every=args.checkpoint_every,
                                 snapshot_every=args.snapshot_every, map_table=args.map_table, seed=args.seed,
                                 log=lambda s: print(s, flush=True), commit=commit or (lambda: None))
    try:
        trainer.run()
    finally:
        trainer.close()
    print('RESULT ' + json.dumps(trainer.result), flush=True)
    return trainer.result


if __name__ == '__main__':
    main()
