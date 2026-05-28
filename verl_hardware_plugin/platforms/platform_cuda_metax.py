# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""MetaX (沐曦) platform implementation.

MetaX GPUs are CUDA-compatible, so torch.cuda.is_available() returns True
even on MetaX hardware. To distinguish MetaX from NVIDIA during auto-detection,
an optional SMI command check (mx-smi) can be enabled.
"""

import logging
import os
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import PlatformRegistry

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@PlatformRegistry.register(platform="metax")
class PlatformMetaX(PlatformBase):
    """Platform backend for MetaX (沐曦) GPUs.

    MetaX GPUs expose a CUDA-compatible interface, so this platform uses
    torch.cuda underneath. The is_available() method supports an optional
    SMI-based hardware check to distinguish MetaX from NVIDIA.
    """

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        return "cuda"

    @property
    def vendor_name(self) -> str:
        return "metax"

    @property
    def device_module(self) -> ModuleType:
        return torch.cuda

    def is_available(self, use_smi_check: bool = False) -> bool:
        if not torch.cuda.is_available():
            return False
        if use_smi_check:
            return self.check_smi_command("mx-smi")
        return True

    def current_device(self) -> int:
        return torch.cuda.current_device()

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def set_device(self, device_index: int) -> None:
        torch.cuda.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        if device_index is not None:
            torch.cuda.synchronize(device_index)
        else:
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        torch.cuda.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def set_allocator_settings(self, settings: str) -> None:
        try:
            torch.cuda.memory._set_allocator_settings(settings)
        except (AttributeError, RuntimeError):
            logger.warning("Failed to set CUDA allocator settings on MetaX")

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        return torch.cuda.get_device_capability(device_index)

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        return "nccl"

    def visible_devices_envvar(self) -> str:
        return "CUDA_VISIBLE_DEVICES"

    # ------------------------------------------------------------------
    # Ray integration
    # ------------------------------------------------------------------

    def ray_resource_name(self) -> str:
        return "GPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        return {"num_gpus": num_gpus}

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"]

    # ------------------------------------------------------------------
    # IPC support
    # ------------------------------------------------------------------

    def is_ipc_supported(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @contextmanager
    def nvtx_range(self, msg: str):
        torch.cuda.nvtx.range_push(msg)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    def profiler_start(self) -> None:
        torch.cuda.cudart().cudaProfilerStart()

    def profiler_stop(self) -> None:
        torch.cuda.cudart().cudaProfilerStop()

    # ------------------------------------------------------------------
    # Model patches
    # ------------------------------------------------------------------

    def apply_model_patches(self, model_type: str) -> None:
        pass

    # ------------------------------------------------------------------
    # Rollout engine integration
    # ------------------------------------------------------------------

    def rollout_env_vars(self) -> dict[str, str]:
        return {}

    # ------------------------------------------------------------------
    # Collective communication
    # ------------------------------------------------------------------

    def get_collective_module(self) -> Any:
        try:
            import cupy.cuda.nccl

            return cupy.cuda.nccl
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # Low-level runtime API
    # ------------------------------------------------------------------

    def cudart(self) -> Any:
        return torch.cuda.cudart()
