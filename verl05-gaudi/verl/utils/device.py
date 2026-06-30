# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE

import logging

import torch

logger = logging.getLogger(__name__)


def is_torch_npu_available() -> bool:
    """Check the availability of NPU"""
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False


is_cuda_available = torch.cuda.is_available()
is_npu_available = is_torch_npu_available()


def is_torch_hpu_available() -> bool:
    """Check the availability of Intel Gaudi HPU"""
    try:
        import habana_frameworks.torch  # noqa: F401
        return hasattr(torch, "hpu") and torch.hpu.is_available()
    except Exception:
        return False


is_hpu_available = is_torch_hpu_available()

if is_hpu_available:
    try:
        import functools as _ft
        import transformers.file_utils as _fu
        try:
            import transformers.utils as _tu
        except Exception:
            _tu = None
        for _n in ("cached_property", "is_torch_available", "requires_backends", "torch_required", "add_start_docstrings"):
            if not hasattr(_fu, _n) and _tu is not None and hasattr(_tu, _n):
                setattr(_fu, _n, getattr(_tu, _n))
        if not hasattr(_fu, "cached_property"):
            _fu.cached_property = _ft.cached_property
        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
        adapt_transformers_to_gaudi()
        logger.warning("optimum-habana: adapted transformers generate to Gaudi")
    except Exception as _e:
        logger.warning(f"optimum-habana adapt skipped: {_e}")

# Some Habana torch builds lack torch.hpu.empty_cache; verl calls it via
# get_torch_device().empty_cache(). Provide a no-op shim when missing.
if is_hpu_available and hasattr(torch, "hpu") and not hasattr(torch.hpu, "empty_cache"):
    try:
        torch.hpu.empty_cache = lambda *a, **k: None
    except Exception:
        pass


# Habana HCCL rejects ReduceOp.AVG/PREMUL_SUM (hccl_kernels.cpp getHCCLReduceOp).
# On a world_size==1 process group every collective is the identity, so the
# ReduceOp is mathematically irrelevant -> coerce AVG/PREMUL_SUM to SUM.
if is_hpu_available:
    try:
        import torch.distributed as _dist
        from torch.distributed.distributed_c10d import ReduceOp as _RedOp

        _BAD_OPS = [getattr(_RedOp, _n) for _n in ("AVG", "PREMUL_SUM") if hasattr(_RedOp, _n)]

        def _coerce_op(op, group):
            try:
                bad = any(op == b for b in _BAD_OPS)
            except Exception:
                bad = False
            if not bad:
                return op
            try:
                ws = _dist.get_world_size(group) if _dist.is_initialized() else 1
            except Exception:
                ws = 1
            return _RedOp.SUM if ws == 1 else op

        if not getattr(_dist, "_hpu_redop_patched", False):
            _o_ar = _dist.all_reduce

            def _ar(tensor, op=_RedOp.SUM, group=None, async_op=False):
                return _o_ar(tensor, op=_coerce_op(op, group), group=group, async_op=async_op)

            _dist.all_reduce = _ar

            if hasattr(_dist, "reduce_scatter_tensor"):
                _o_rst = _dist.reduce_scatter_tensor

                def _rst(output, input, op=_RedOp.SUM, group=None, async_op=False):
                    return _o_rst(output, input, op=_coerce_op(op, group), group=group, async_op=async_op)

                _dist.reduce_scatter_tensor = _rst

            if hasattr(_dist, "_reduce_scatter_base"):
                _o_rsb = _dist._reduce_scatter_base

                def _rsb(output, input, op=_RedOp.SUM, group=None, async_op=False):
                    return _o_rsb(output, input, op=_coerce_op(op, group), group=group, async_op=async_op)

                _dist._reduce_scatter_base = _rsb

            _o_red = _dist.reduce

            def _red(tensor, dst, op=_RedOp.SUM, group=None, async_op=False):
                return _o_red(tensor, dst, op=_coerce_op(op, group), group=group, async_op=async_op)

            _dist.reduce = _red
            _dist._hpu_redop_patched = True
            logger.warning("HPU: ReduceOp AVG/PREMUL_SUM -> SUM coercion installed (world_size==1)")
    except Exception as _e:
        logger.warning(f"HPU ReduceOp coercion skipped: {_e}")


def get_visible_devices_keyword() -> str:
    """Function that gets visible devices keyword name.
    Returns:
        'CUDA_VISIBLE_DEVICES' or `ASCEND_RT_VISIBLE_DEVICES`
    """
    if is_cuda_available:
        return "CUDA_VISIBLE_DEVICES"
    if is_hpu_available:
        return "HABANA_VISIBLE_MODULES"
    return "ASCEND_RT_VISIBLE_DEVICES"


def get_device_name() -> str:
    """Function that gets the torch.device based on the current machine.
    This currently only supports CPU, CUDA, NPU.
    Returns:
        device
    """
    if is_cuda_available:
        device = "cuda"
    elif is_npu_available:
        device = "npu"
    elif is_hpu_available:
        device = "hpu"
    else:
        device = "cpu"
    return device


def get_torch_device() -> any:
    """Return the corresponding torch attribute based on the device type string.
    Returns:
        module: The corresponding torch device namespace, or torch.cuda if not found.
    """
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda


def get_device_id() -> int:
    """Return current device id based on the device type.
    Returns:
        device index
    """
    return get_torch_device().current_device()


def get_nccl_backend() -> str:
    """Return nccl backend type based on the device type.
    Returns:
        nccl backend type string.
    """
    if is_cuda_available:
        return "nccl"
    elif is_npu_available or is_hpu_available:
        return "hccl"
    else:
        raise RuntimeError(f"No available nccl backend found on device type {get_device_name()}.")
