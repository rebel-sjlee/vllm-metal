# SPDX-License-Identifier: Apache-2.0
"""Compatibility layer for vLLM imports.

This module provides compatibility when vLLM is not installed,
allowing the core Metal operations to be tested standalone.
"""

from enum import Enum
from typing import Any, Optional

try:
    from vllm.platforms import Platform, PlatformEnum
    from vllm.logger import init_logger
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

    class PlatformEnum(Enum):
        """Platform enumeration when vLLM is not available."""
        CUDA = "cuda"
        ROCM = "rocm"
        TPU = "tpu"
        XPU = "xpu"
        CPU = "cpu"
        OOT = "oot"
        UNSPECIFIED = "unspecified"

    class Platform:
        """Base Platform class when vLLM is not available."""
        _enum = PlatformEnum.UNSPECIFIED
        device_name: str = ""
        device_type: str = ""

        @classmethod
        def get_device_name(cls, device_id: int = 0) -> str:
            return cls.device_name

        @classmethod
        def get_device_uuid(cls, device_id: int = 0) -> str:
            return ""

        @classmethod
        def get_device_total_memory(cls, device_id: int = 0) -> int:
            return 0

        @classmethod
        def get_device_capability(cls, device_id: int = 0) -> Optional[Any]:
            return None

    def init_logger(name: str):
        """Simple logger when vLLM is not available."""
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


try:
    from vllm.attention.backends.abstract import (
        AttentionBackend,
        AttentionImpl,
        AttentionMetadata,
        AttentionMetadataBuilder,
        AttentionType,
    )
except ImportError:
    from dataclasses import dataclass
    from enum import auto

    class AttentionType(Enum):
        """Attention type when vLLM is not available."""
        DECODER = auto()
        ENCODER = auto()
        ENCODER_DECODER = auto()

    @dataclass
    class AttentionMetadata:
        """Base attention metadata when vLLM is not available."""
        pass

    class AttentionMetadataBuilder:
        """Base metadata builder when vLLM is not available."""
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            pass

        def build(self, *args, **kwargs):
            pass

    class AttentionImpl:
        """Base attention implementation when vLLM is not available."""
        def forward(self, *args, **kwargs):
            raise NotImplementedError

    class AttentionBackend:
        """Base attention backend when vLLM is not available."""
        @staticmethod
        def get_name() -> str:
            return "UNKNOWN"


try:
    from vllm.sequence import ExecuteModelRequest, SequenceGroupMetadata
except ImportError:
    @dataclass
    class SequenceGroupMetadata:
        """Sequence group metadata when vLLM is not available."""
        is_prompt: bool = False
        seq_data: dict = None
        block_tables: dict = None

    @dataclass
    class ExecuteModelRequest:
        """Execute model request when vLLM is not available."""
        seq_group_metadata_list: list = None


try:
    from vllm.worker.worker_base import WorkerBase, WorkerInput
except ImportError:
    class WorkerInput:
        """Worker input when vLLM is not available."""
        pass

    class WorkerBase:
        """Worker base when vLLM is not available."""
        pass


try:
    from vllm.model_executor.model_loader.loader import BaseModelLoader
except ImportError:
    class BaseModelLoader:
        """Base model loader when vLLM is not available."""
        def __init__(self, load_config):
            self.load_config = load_config

        def load_model(self, *args, **kwargs):
            raise NotImplementedError


try:
    from vllm.config import VllmConfig
except ImportError:
    @dataclass
    class VllmConfig:
        """vLLM config when vLLM is not available."""
        model_config: Any = None
        cache_config: Any = None
        parallel_config: Any = None
        scheduler_config: Any = None
