"""Start strong PPO with the recipe's library environment before NumPy/Torch.

The numerical trainer is unchanged; exact checkpoint source/config checks remain
inside strong.train. API callers must apply the same environment before import.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

THREAD_VARIABLES={'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'}


def main():
    if not any(arg in ('--help','-h') for arg in sys.argv[1:]):
        parser=argparse.ArgumentParser(add_help=False)
        parser.add_argument('--config',type=Path,required=True)
        args,_=parser.parse_known_args()
        expected=json.loads(args.config.read_text()).get('host_thread_environment',{})
        if any(key not in THREAD_VARIABLES or value not in ('1','2','3','4') for key,value in expected.items()):
            raise ValueError('Declared CPU-library threads must be strings1..4 for OMP/MKL/OpenBLAS')
        os.environ.update(expected)
    runpy.run_module('simulators.tavern.reference_training.strong.train',run_name='__main__')


if __name__=='__main__':main()
