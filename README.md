# verl-hardware-plugin

Multi-chip hardware platform and engine plugin **reference implementations** for [verl](https://github.com/verl-project/verl).

This package provides platform abstraction and training engine extensions for non-CUDA accelerators. It serves as a **template and example** for hardware vendors to adapt verl to their own devices through the unified plugin interface.

## Purpose

The platforms and engines in this repository are **reference implementations** — they demonstrate how vendors can integrate their hardware with verl's plugin system. Hardware vendors can use these as templates to build their own plugins.

## Supported Hardware (Reference Implementations)

> **Note**: The implementations below are **examples only**. Full production support and maintenance require collaboration with the respective hardware vendors. These serve as templates for vendors to adapt and maintain their own integrations.

| Platform | Device | Communication | Status |
|----------|--------|---------------|--------|
| FlagOS | xxx | FlagCX | ✅ Example (requires vendor support) |
| Intel XPU | Data Center GPU Max / Arc | xccl (oneCCL) | ✅ Example (requires vendor support) |
| Cambricon MLU | MLU370 / MLU590 | CNCL | ✅ Example (requires vendor support) |
| MetaX | MetaX GPUs (CUDA-compatible) | NCCL | ✅ Example (requires vendor support) |
| Huawei NPU | Ascend 910B | HCCL | Built-in (verl core) |


## Installation

```bash
pip install -e .
```

## Usage

After `pip install`, the plugin is automatically discovered by verl through the
`verl.plugins` entry_points group. No additional configuration needed.

### Platform Selection

The platform is auto-detected based on hardware availability. During auto-detection, SMI commands (e.g. `nvidia-smi`, `mx-smi`) are used to distinguish CUDA-compatible hardware. To force a specific platform:

```bash
export VERL_PLATFORM=intel     # Force Intel XPU
export VERL_PLATFORM=cambricon # Force Cambricon MLU
export VERL_PLATFORM=metax     # Force MetaX
export VERL_PLATFORM=flagos    # Force FlagOS
```

### FlagOS Configuration

```bash
# Enable FlagGems operator library
export USE_FLAGGEMS=true

# Enable FlagCX communication library
export USE_FLAGCX=1

# Optional: operator whitelist/blacklist
export TRAINING_FL_FLAGOS_WHITELIST=rmsnorm,layernorm,softmax
# OR
export TRAINING_FL_FLAGOS_BLACKLIST=flash_attention
```

### MetaX Configuration

MetaX GPUs are CUDA-compatible, so they work with `torch.cuda` directly. The plugin uses `mx-smi` to distinguish MetaX hardware from NVIDIA during auto-detection.

```bash
export VERL_PLATFORM=metax  # or let auto-detection handle it

python -m verl.trainer.main --config your_config.yaml
```

## Architecture

```
verl-FL (main framework)
    └── entry_points: verl.plugins → verl_hardware_plugin
            │
            ├── PlatformRegistry.register("intel")    → PlatformXPU
            ├── PlatformRegistry.register("cambricon")→ PlatformMLU
            ├── PlatformRegistry.register("metax")    → PlatformMetaX
            ├── PlatformRegistry.register("flagos")   → PlatformFlagOS
            │
            ├── EngineRegistry.register(device="xpu", vendor="intel")
            ├── EngineRegistry.register(device="mlu", vendor="cambricon")
            ├── EngineRegistry.register(device="cuda", vendor="metax")
            └── EngineRegistry.register(device="cuda", vendor="flagos")
```

The plugin uses verl's decorator-based registration:
- `@PlatformRegistry.register(platform="vendor_name")` for platform classes
- `@EngineRegistry.register(model_type=..., backend=..., device=..., vendor=...)` for engine classes

Registration happens at import time. Engine lookup uses a two-level key `(device, vendor)`:
1. Exact match `(device, vendor)` — vendor-specific engine
2. Fallback to device-only key — base engine for that device type
3. For CUDA-compatible devices, fallback to base CUDA engine

### SMI-based Hardware Detection

For CUDA-compatible hardware (MetaX, NVIDIA), `torch.cuda.is_available()` returns True on both. The `is_available(use_smi_check=True)` parameter enables SMI command checks to distinguish the actual hardware:

- `PlatformCUDA` checks `nvidia-smi`
- `PlatformMetaX` checks `mx-smi`

This check is only performed during first-time auto-detection.

## Documentation

- **[Development Guide](docs/development.md)** — How to add a new hardware platform and engine (start here for adaptation)
- [Intel XPU](docs/xpu.md)
- [Cambricon MLU](docs/mlu.md)
- [FlagOS](docs/flagos.md)

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

Apache License 2.0
