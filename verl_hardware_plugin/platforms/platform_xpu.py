# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Intel XPU platform implementation.

Supports Intel Data Center GPU Max Series and similar devices via
torch.xpu and oneAPI/oneCCL (xccl) communication backend.
"""

from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import PlatformRegistry


def _ensure_torch_xpu() -> bool:
    """Check if torch.xpu is available."""
    if hasattr(torch, "xpu"):
        return True
    try:
        import intel_extension_for_pytorch  # noqa: F401
        return hasattr(torch, "xpu")
    except ImportError:
        return False


@PlatformRegistry.register(platform="intel")
class PlatformXPU(PlatformBase):
    """Platform backend for Intel XPU (Data Center GPU Max, Arc, etc.)."""

    @property
    def device_name(self) -> str:
        return "xpu"

    @property
    def vendor_name(self) -> str:
        return "intel"

    @property
    def device_module(self) -> ModuleType:
        if not _ensure_torch_xpu():
            raise RuntimeError("torch.xpu is not available. Install intel-extension-for-pytorch.")
        return torch.xpu

    def is_available(self, use_smi_check: bool = False) -> bool:
        if not _ensure_torch_xpu():
            return False
        return torch.xpu.is_available()

    def current_device(self) -> int:
        return torch.xpu.current_device()

    def device_count(self) -> int:
        return torch.xpu.device_count()

    def set_device(self, device_index: int) -> None:
        torch.xpu.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        if device_index is not None:
            torch.xpu.synchronize(device_index)
        else:
            torch.xpu.synchronize()

    def manual_seed(self, seed: int) -> None:
        torch.xpu.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.xpu.manual_seed_all(seed)

    def set_allocator_settings(self, settings: str) -> None:
        pass

    def empty_cache(self) -> None:
        torch.xpu.empty_cache()

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        return (None, None)

    def communication_backend_name(self) -> str:
        return "xccl"

    def visible_devices_envvar(self) -> str:
        return "ZE_AFFINITY_MASK"

    def ray_resource_name(self) -> str:
        return "GPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        return {"num_gpus": num_gpus}

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_ZE_AFFINITY_MASK"]

    def is_ipc_supported(self) -> bool:
        return False

    @contextmanager
    def nvtx_range(self, msg: str):
        yield

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    def cudart(self) -> Any:
        return None
