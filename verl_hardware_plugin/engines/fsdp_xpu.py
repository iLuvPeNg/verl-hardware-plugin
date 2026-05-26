# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""FSDP engine for Intel XPU devices.

Extends the base FSDP engine with XPU-specific workarounds
(e.g., force sum reduction for xccl backend).
"""

import logging
import os

from verl.trainer.config import CheckpointConfig
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp import FSDPEngineWithLMHead
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithValueHead

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device="xpu", vendor="intel")
class FSDPXPUEngineWithLMHead(FSDPEngineWithLMHead):
    """FSDP Engine for Intel XPU with xccl communication backend."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        logger.info("FSDPXPUEngineWithLMHead initialized")

    def initialize(self):
        super().initialize()
        # xccl does not support ReduceOp.AVG; force sum-based reduction
        if hasattr(self.model, "set_force_sum_reduction_for_comms"):
            self.model.set_force_sum_reduction_for_comms(True)
            logger.info("Enabled force_sum_reduction_for_comms for XPU")


@EngineRegistry.register(model_type="value_model", backend=["fsdp", "fsdp2"], device="xpu", vendor="intel")
class FSDPXPUEngineWithValueHead(FSDPEngineWithValueHead):
    """FSDP Engine for Intel XPU value model training."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        logger.info("FSDPXPUEngineWithValueHead initialized")

    def initialize(self):
        super().initialize()
        if hasattr(self.model, "set_force_sum_reduction_for_comms"):
            self.model.set_force_sum_reduction_for_comms(True)
