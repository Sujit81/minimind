# Configuration module for MiniMind
from .hindi_config import (
    HindiDatasetConfig,
    HindiTokenizerConfig,
    HindiModelConfig,
    HindiTrainingConfig,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_TOKENIZER_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    get_hf_dataset_path,
    get_hf_dataset_config,
    print_config,
)

__all__ = [
    "HindiDatasetConfig",
    "HindiTokenizerConfig",
    "HindiModelConfig",
    "HindiTrainingConfig",
    "DEFAULT_DATASET_CONFIG",
    "DEFAULT_TOKENIZER_CONFIG",
    "DEFAULT_MODEL_CONFIG",
    "DEFAULT_TRAINING_CONFIG",
    "get_hf_dataset_path",
    "get_hf_dataset_config",
    "print_config",
]