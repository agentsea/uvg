"""UVG - Unsloth VLM GRPO

A library for training vision-language models using GRPO (Guided Reward Policy Optimization).
"""

from .config import Config
from .data import GRPODataset
from .trainer import trainer

__version__ = "0.1.0"
__all__ = ["Config", "GRPODataset", "train"]