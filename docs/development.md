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
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import PlatformRegistry


@PlatformRegistry.register(platform="my_vendor")
class PlatformMyDevice(PlatformBase):

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
        torch.my_device.synchronize()

    def manual_seed(self, seed: int) -> None:
        torch.my_device.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.my_device.manual_seed_all(seed)

    def set_allocator_settings(self, settings: str) -> None:
        pass  # leave empty if unsupported

    def empty_cache(self) -> None:
        torch.my_device.empty_cache()

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        return (None, None)

    def communication_backend_name(self) -> str:
        return "my_ccl"  # collective communication backend name

    def visible_devices_envvar(self) -> str:
        return "MY_DEVICE_VISIBLE_DEVICES"

    def ray_resource_name(self) -> str:
        return "MY_DEVICE"  # Ray resource name

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        return {"num_gpus": num_gpus}

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_MY_DEVICE_VISIBLE_DEVICES"]

    def is_ipc_supported(self) -> bool:
        return False

    @contextmanager
    def nvtx_range(self, msg: str):
        yield  # no-op if profiling is not supported

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

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

## Reference Implementations

The following files in this repository serve as examples:

| Vendor | Platform File | Engine Files |
|--------|--------------|--------------|
| Intel XPU | `platforms/platform_xpu.py` | `engines/fsdp_xpu.py`, `engines/megatron_xpu.py` |
| Cambricon MLU | `platforms/platform_mlu.py` | `engines/fsdp_mlu.py`, `engines/megatron_mlu.py` |
| MetaX | `platforms/platform_cuda_metax.py` | `engines/fsdp_metax.py`, `engines/megatron_metax.py` |
| FlagOS | `platforms/flagos.py` | `engines/fsdp_flagos.py`, `engines/megatron_flagos.py` |

## Related Resources

- **verl core PR**: [verl#6086 — Platform & Engine Registry](https://github.com/verl-project/verl/pull/6086)
- **Platform base class**: `verl/plugin/platform/platform_base.py`
- **Engine base class**: `verl/workers/engine/base.py`
- **Platform README**: `verl/plugin/platform/README.md`
