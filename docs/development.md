# Development Guide: Adding a New Hardware Platform

This document explains how to add support for a new hardware accelerator to verl via the plugin system.

> **verl core PR**: The platform and engine registry mechanism is implemented in
> [verl#6086](https://github.com/verl-project/verl/pull/6086). Refer to that PR
> for the base class interfaces (`PlatformBase`, `EngineRegistry`) and the
> registration/lookup logic.

## Architecture Overview

```
verl-FL (main framework)
    │
    └── entry_points: verl.plugins → verl_hardware_plugin
            │
            ├── platforms/  → @PlatformRegistry.register(platform="vendor_name")
            └── engines/    → @EngineRegistry.register(device=..., vendor=...)
```

The plugin integrates with verl through two registries:
1. **PlatformRegistry** — registers hardware platform abstractions (device management, communication, memory, etc.)
2. **EngineRegistry** — registers training engines (hardware-specific variants of FSDP/Megatron)

## Steps to Add a New Platform

### Step 1: Create the Platform Class

Create a new file under `verl_hardware_plugin/platforms/`, e.g. `platform_my_vendor.py`:

```python
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


@PlatformRegistry.register(platform="my_vendor")
class PlatformMyDevice(PlatformBase):

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        return "my_device"  # torch device type: "cuda", "xpu", "mlu", etc.

    @property
    def vendor_name(self) -> str:
        return "my_vendor"  # vendor identifier, used for engine lookup

    @property
    def device_module(self) -> ModuleType:
        return torch.my_device  # torch device namespace

    def is_available(self, use_smi_check: bool = False) -> bool:
        if not hasattr(torch, "my_device"):
            return False
        if use_smi_check:
            return self.check_smi_command("my-smi")
        return torch.my_device.is_available()

    def current_device(self) -> int:
        return torch.my_device.current_device()

    def device_count(self) -> int:
        return torch.my_device.device_count()

    def set_device(self, device_index: int) -> None:
        torch.my_device.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        if device_index is not None:
            torch.my_device.synchronize(device_index)
        else:
            torch.my_device.synchronize()

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        torch.my_device.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.my_device.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def set_allocator_settings(self, settings: str) -> None:
        pass  # leave empty if unsupported

    def empty_cache(self) -> None:
        torch.my_device.empty_cache()

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        return (None, None)

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        return "my_ccl"  # collective communication backend name

    def visible_devices_envvar(self) -> str:
        return "MY_DEVICE_VISIBLE_DEVICES"

    # ------------------------------------------------------------------
    # Ray integration
    # ------------------------------------------------------------------

    def ray_resource_name(self) -> str:
        return "MY_DEVICE"  # Ray resource name

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        return {"resources": {"MY_DEVICE": num_gpus}}

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_MY_DEVICE_VISIBLE_DEVICES"]

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
        yield  # no-op if profiling is not supported

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Model patches
    # ------------------------------------------------------------------

    def apply_model_patches(self, model_type: str) -> None:
        pass  # apply platform-specific model patches if needed

    # ------------------------------------------------------------------
    # Rollout engine integration
    # ------------------------------------------------------------------

    def rollout_env_vars(self) -> dict[str, str]:
        return {}  # platform-specific env vars for rollout engines

    # ------------------------------------------------------------------
    # Collective communication
    # ------------------------------------------------------------------

    def get_collective_module(self) -> Any:
        return None  # return collective comm module if available

    # ------------------------------------------------------------------
    # Low-level runtime API
    # ------------------------------------------------------------------

    def cudart(self) -> Any:
        return None
```

**Key points:**
- `device_name` returns the torch device type (e.g. `"cuda"`, `"xpu"`, `"mlu"`)
- `vendor_name` returns the vendor identifier (e.g. `"metax"`, `"intel"`, `"cambricon"`)
- For CUDA-compatible hardware, `device_name` should return `"cuda"` and use `check_smi_command()` to distinguish from NVIDIA

### Step 2: Register in platforms/__init__.py

Add the following to `register_all_platforms()`:

```python
try:
    from verl_hardware_plugin.platforms import platform_my_vendor  # noqa: F401
    logger.info("Registered platform: my_vendor")
except Exception as e:
    logger.debug("my_vendor platform not registered: %s", e)
```

### Step 3: Create Engine Extensions

Create `verl_hardware_plugin/engines/fsdp_my_vendor.py`:

```python
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp import FSDPEngineWithLMHead
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithValueHead


@EngineRegistry.register(
    model_type="language_model",
    backend=["fsdp", "fsdp2"],
    device="my_device",      # torch device type
    vendor="my_vendor",      # vendor name (matches platform.vendor_name)
)
class FSDPMyDeviceEngineWithLMHead(FSDPEngineWithLMHead):
    def initialize(self):
        super().initialize()
        # Add device-specific initialization logic


@EngineRegistry.register(
    model_type="value_model",
    backend=["fsdp", "fsdp2"],
    device="my_device",
    vendor="my_vendor",
)
class FSDPMyDeviceEngineWithValueHead(FSDPEngineWithValueHead):
    def initialize(self):
        super().initialize()
```

**Engine registration key**: The engine is stored under `(device, vendor)` tuple key. During lookup, verl calls `get_device_name()` and `get_vendor()` from the active platform, then looks up the engine by `(device_name, vendor_name)`.

For CUDA-compatible vendors (like MetaX), register with `device="cuda", vendor="my_vendor"`. This ensures the lookup finds your engine when the platform reports `device_name="cuda"` and `vendor_name="my_vendor"`.

### Step 4: Register in engines/__init__.py

Add the corresponding import to `register_all_engines()`.

### Step 5: Add Documentation

Create `docs/my_vendor.md` containing:
- Overview and hardware description
- Prerequisites and installation
- Environment variable configuration
- Usage examples
- FAQ

### Step 6: Add Tests

Add registration verification tests in `tests/test_plugin_registration.py`.

## PlatformBase Interface Reference

The following table lists all methods that must be implemented when subclassing `PlatformBase`:

| Method | Category | Required | Description |
|--------|----------|----------|-------------|
| `device_name` | Core | Yes (abstract) | Torch device type string |
| `vendor_name` | Core | Yes (abstract) | Hardware vendor identifier |
| `device_module` | Core | Yes (abstract) | `torch.<device>` namespace module |
| `is_available(use_smi_check)` | Core | Yes (abstract) | Check if hardware is present |
| `current_device()` | Core | Yes (abstract) | Current device index |
| `device_count()` | Core | Yes (abstract) | Number of available devices |
| `set_device(device_index)` | Core | Yes (abstract) | Select active device |
| `synchronize(device_index)` | Core | Yes (abstract) | Block until device work completes |
| `manual_seed(seed)` | RNG | Yes (abstract) | Seed current device RNG |
| `manual_seed_all(seed)` | RNG | Yes (abstract) | Seed all devices' RNG |
| `set_allocator_settings(settings)` | Memory | Yes (abstract) | Configure memory allocator |
| `empty_cache()` | Memory | Yes (abstract) | Release unused cached memory |
| `get_device_capability(device_index)` | Properties | Yes (abstract) | `(major, minor)` compute capability |
| `communication_backend_name()` | Distributed | Yes (abstract) | Default comm backend (e.g. `"nccl"`) |
| `visible_devices_envvar()` | Distributed | Yes (abstract) | Env var controlling visible devices |
| `nvtx_range(msg)` | Profiling | Yes (abstract) | Context manager for profiler range |
| `profiler_start()` | Profiling | Yes (abstract) | Start device profiler |
| `profiler_stop()` | Profiling | Yes (abstract) | Stop device profiler |
| `apply_model_patches(model_type)` | Model | Yes (abstract) | Apply platform-specific model patches |
| `ray_resource_name()` | Ray | Yes (abstract) | Ray accelerator resource name |
| `ray_noset_envvars()` | Ray | Yes (abstract) | `RAY_EXPERIMENTAL_NOSET_*` env vars |
| `ray_resource_options(num_gpus)` | Ray | No (has default) | Ray actor resource options dict |
| `is_ipc_supported()` | IPC | Yes (abstract) | Whether IPC tensor sharing works |
| `rollout_env_vars()` | Rollout | No (has default) | Env vars for rollout engines |
| `get_collective_module()` | Collective | No (has default) | Collective comm module |
| `cudart()` | Runtime | Yes (abstract) | CUDA runtime API object or None |

## Engine Lookup Logic

The `EngineRegistry.get_engine_cls(model_type, backend)` lookup follows this priority:

1. **Exact match** `(device, vendor)` — vendor-specific engine (e.g. `("cuda", "metax")`)
2. **Device-only fallback** `device` — base engine for that device type (e.g. `"npu"`)
3. **NVIDIA fallback** — for unknown CUDA vendors, try `("cuda", "nvidia")` then `"cuda"`

Environment variable overrides:
- `VERL_ENGINE_DEVICE` — override the detected device name
- `VERL_ENGINE_VENDOR` — override the detected vendor name

## Key Design Principles

1. **Conditional imports**: Platform modules are imported inside `try/except` blocks — a missing hardware SDK does not affect other platforms.
2. **Last writer wins**: A platform registered later with the same name overrides the earlier one, allowing plugins to override built-in implementations.
3. **Auto-detection**: The first platform whose `is_available(use_smi_check=True)` returns True is selected, or it can be forced via `VERL_PLATFORM`.
4. **Minimal intrusion**: Engine extensions inject logic through inheritance + `initialize()` without modifying base class behavior.
5. **Two-dimensional engine key**: `(device, vendor)` allows multiple vendors sharing the same device type (e.g. MetaX and FlagOS both use `"cuda"`) to have distinct engines.
6. **Section-based organization**: Platform implementations should group methods by category using comment section headers for readability and consistency.

## Reference Implementations

The following files in this repository serve as examples:

| Vendor | Platform File | Engine Files |
|--------|--------------|--------------|
| Intel XPU | `platforms/platform_xpu.py` | `engines/fsdp_xpu.py`, `engines/megatron_xpu.py` |
| Cambricon MLU | `platforms/platform_mlu.py` | `engines/fsdp_mlu.py`, `engines/megatron_mlu.py` |
| MetaX | `platforms/platform_cuda_metax.py` | `engines/fsdp_metax.py`, `engines/megatron_metax.py` |

## Related Resources

- **verl core PR**: [verl#6086 — Platform & Engine Registry](https://github.com/verl-project/verl/pull/6086)
- **Platform base class**: `verl/plugin/platform/platform_base.py`
- **Engine base class**: `verl/workers/engine/base.py`
- **Platform README**: `verl/plugin/platform/README.md`
