# python -m torch.distributed.launch --nproc_per_node=1 24_bit_allreduce.py

import torch
import os
import cupy
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

def torch2cupy(tensor):
    return cupy.fromDlpack(to_dlpack(tensor))


def cupy2torch(cupy_tensor):
    return from_dlpack(cupy_tensor.toDlpack())


def decompose_cupy(tensor):
    mantissa, exponent = cupy.frexp(torch2cupy(tensor.float()))
    return cupy2torch(mantissa).half(), cupy2torch(exponent).to(torch.int8)


def decompose(t):
    if TORCH_VERSION_MAJOR < 1 or (TORCH_VERSION_MAJOR >= 1 and TORCH_VERSION_MINOR < 9):
        raise Exception('Torch version >= 1.9.0 needed for 24_bit_allreduce.decompose')
    mantissa, exponent = torch.frexp(t.float())
    return mantissa.half(), exponent.to(torch.int8)


def reconstruct(mantissa, exponent, original_dtype=torch.bfloat16):
    return torch.ldexp(mantissa, exponent).to(original_dtype)


def compressed_all_reduce_torch(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
    original_dtype = tensor.dtype
    m, e = decompose(tensor)
    torch.distributed.all_reduce(m, op=op, group=group, async_op=async_op)
    torch.distributed.all_reduce(e, op=op, group=group, async_op=async_op)
    return reconstruct(m, e, original_dtype)


def compressed_all_reduce_cupy(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
    original_dtype = tensor.dtype
    m, e = decompose_cupy(tensor)
    torch.distributed.all_reduce(m, op=op, group=group, async_op=async_op)
    torch.distributed.all_reduce(e, op=op, group=group, async_op=async_op)
    return reconstruct(m, e, original_dtype)

version = torch.__version__.split('.')
TORCH_VERSION_MAJOR = int(version[0])
TORCH_VERSION_MINOR = int(version[1])
if TORCH_VERSION_MAJOR < 1 or (TORCH_VERSION_MAJOR >= 1 and TORCH_VERSION_MINOR < 9):
    compressed_all_reduce = compressed_all_reduce_cupy
else:
    compressed_all_reduce = compressed_all_reduce_torch
