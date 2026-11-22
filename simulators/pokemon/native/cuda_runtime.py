"""Explicit CUDA call-stack provisioning for the recursive native move path."""
import ctypes as C


def prepare(library,minimum_stack_bytes=65536):
    if minimum_stack_bytes<1:raise ValueError('Positive CUDA call-stack limit required')
    library.pkn_cuda_prepare.argtypes=[C.c_size_t]+[C.POINTER(C.c_size_t)]*3
    actual=C.c_size_t();free=C.c_size_t();total=C.c_size_t()
    code=library.pkn_cuda_prepare(minimum_stack_bytes,C.byref(actual),C.byref(free),C.byref(total))
    if code:raise RuntimeError(f'Native CUDA runtime preparation failed: {code}')
    if actual.value<minimum_stack_bytes:raise RuntimeError('CUDA call-stack limit was not applied')
    return dict(stack_bytes=actual.value,free_bytes=free.value,total_bytes=total.value)
