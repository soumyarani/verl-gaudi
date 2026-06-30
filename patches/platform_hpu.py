# Intel Gaudi (HPU) platform implementation for verl.
# Mirrors platform_npu.py; added to support Intel Gaudi 2/3 accelerators
# via habana_frameworks.torch (SynapseAI).

import logging
import os
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import enum as _enum
if not hasattr(_enum, "StrEnum"):
    class _StrEnum(str, _enum.Enum):
        def __str__(self):
            return str(self.value)
    _enum.StrEnum = _StrEnum
import torch

from .platform_base import PlatformBase
from .platform_manager import PlatformRegistry

logger = logging.getLogger(__name__)


def _ensure_torch_hpu() -> bool:
    """Try to import habana_frameworks.torch so that torch.hpu becomes available.

    Returns True if torch.hpu is usable after the attempt.
    """
    if hasattr(torch, "hpu"):
        return True
    try:
        import habana_frameworks.torch  # noqa: F401

        return hasattr(torch, "hpu")
    except Exception as e:
        logger.debug("The current machine has no torch.hpu, because: %s", e)
    return False


_ensure_torch_hpu()  # Attempt habana import at module load so availability checks are fast later

# Some Habana torch builds (e.g. torch 2.7.1) lack torch.hpu.empty_cache, which verl calls
# directly via get_torch_device().empty_cache(). Provide a no-op shim when missing.
if hasattr(torch, "hpu") and not hasattr(torch.hpu, "empty_cache"):
    try:
        torch.hpu.empty_cache = lambda *a, **k: None
    except Exception:
        pass


@PlatformRegistry.register(platform="hpu")
class PlatformHPU(PlatformBase):
    """Platform backend for Intel Gaudi (HPU)."""

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        return "hpu"

    @property
    def vendor_name(self) -> str:
        return "intel"

    @property
    def device_module(self) -> ModuleType:
        return torch.hpu

    def is_available(self) -> bool:
        return _ensure_torch_hpu() and torch.hpu.is_available()

    def is_platform_available(self, use_smi_check=False) -> bool:
        if not _ensure_torch_hpu():
            return False
        if use_smi_check:
            # habana_frameworks.torch imported successfully -> HPU environment confirmed
            return True
        return torch.hpu.is_available()

    def current_device(self) -> int:
        return torch.hpu.current_device()

    def device_count(self) -> int:
        return torch.hpu.device_count()

    def set_device(self, device_index: int) -> None:
        torch.hpu.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        torch.hpu.synchronize()

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        torch.hpu.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.hpu.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def set_allocator_settings(self, settings: str) -> None:
        # Gaudi has no public allocator-tuning API equivalent to CUDA; no-op.
        logger.debug("set_allocator_settings is a no-op on HPU (requested: %s)", settings)

    def empty_cache(self) -> None:
        if hasattr(torch.hpu, "empty_cache"):
            torch.hpu.empty_cache()

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        return (None, None)

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        return "hccl"

    def visible_devices_envvar(self) -> str:
        return "HABANA_VISIBLE_MODULES"

    # ------------------------------------------------------------------
    # Ray integration
    # ------------------------------------------------------------------

    def ray_resource_name(self) -> str:
        return "HPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        return {"resources": {"HPU": num_gpus}}

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES"]

    def rollout_env_vars(self) -> dict[str, str]:
        return {}

    # ------------------------------------------------------------------
    # IPC support
    # ------------------------------------------------------------------

    def is_ipc_supported(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @contextmanager
    def nvtx_range(self, msg: str):
        logger.debug("NVTX range (no-op on HPU): %s", msg)
        yield

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Low-level runtime API
    # ------------------------------------------------------------------

    def cudart(self) -> Any:
        return None
