# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Platform registration for all supported hardware backends.

Each platform module uses @PlatformRegistry.register() at import time.
We import them conditionally to avoid hard failures when the corresponding
hardware SDK is not installed.
"""

import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def register_all_platforms():
    """Import all platform modules to trigger their @register decorators."""

    # Intel XPU
    try:
        from verl_hardware_plugin.platforms import platform_xpu  # noqa: F401

        logger.info("Registered platform: xpu")
    except Exception as e:
        logger.debug("XPU platform not registered: %s", e)

    # Cambricon MLU
    try:
        from verl_hardware_plugin.platforms import platform_mlu  # noqa: F401

        logger.info("Registered platform: mlu")
    except Exception as e:
        logger.debug("MLU platform not registered: %s", e)

    # MetaX (沐曦)
    try:
        from verl_hardware_plugin.platforms import platform_cuda_metax  # noqa: F401

        logger.info("Registered platform: metax")
    except Exception as e:
        logger.debug("MetaX platform not registered: %s", e)

