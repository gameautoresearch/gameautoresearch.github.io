"""The repository's GPU: one NVIDIA L40S for every game's research, fresh final training and GPU-backed grading.

Backends request `GPU` from Modal and refuse a worker whose provider-reported type does not
match. CPU cores and RAM stay each game's own declared profile. See docs/compute-defaults.md.
"""

GPU = 'L40S'                 # Modal gpu= argument
DEVICE_NAME = 'NVIDIA L40S'  # torch.cuda.get_device_name() on the worker
CUDA_ARCH = 'sm_89'          # Ada Lovelace; native kernels are compiled for this


def matches(reported):
    """True when a provider-reported GPU type is the repository's GPU."""
    return GPU in str(reported or '').upper()
