# Hindi Training Configuration
# हिंदी प्रशिक्षण कॉन्फ़िगरेशन
#
# This file contains all configurable parameters for Hindi model training.
# Modify these settings to customize your training pipeline.

from dataclasses import dataclass, field
from typing import List, Optional

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

@dataclass
class HindiDatasetConfig:
    """Configuration for Hindi dataset sources."""

    # Primary dataset: AI4Bharat Sangraha
    # https://huggingface.co/datasets/ai4bharat/sangraha/viewer/synthetic/hin_Deva
    hf_dataset_name: str = "ai4bharat/sangraha"
    hf_dataset_subset: str = "synthetic"  # Options: "synthetic", "verified"
    hf_dataset_split: str = "hin_Deva"  # Hindi in Devanagari script (this is the split name)
    hf_text_column: str = "text"  # Column containing text data

    # Data limits (set to None for full dataset)
    max_samples: Optional[int] = None  # Limit samples for testing (e.g., 100000)
    # NOTE: For ai4bharat/sangraha, use streaming=True to avoid downloading ALL languages
    streaming: bool = True  # Streaming recommended for multi-language datasets

    # Local data paths
    data_dir: str = "./dataset/hindi"
    corpus_raw_file: str = "corpus_raw.txt"
    corpus_bilingual_file: str = "corpus_bilingual.txt"
    corpus_pretrain_file: str = "corpus_pretrain.jsonl"
    sft_file: str = "sft_hindi.jsonl"

    # English corpus for bilingual training (optional)
    english_corpus_path: Optional[str] = None
    hindi_english_ratio: float = 0.8  # 80% Hindi, 20% English

    # Text processing
    min_text_length: int = 50  # Minimum characters per sample
    max_text_length: int = 10000  # Maximum characters per sample
    deduplicate: bool = True  # Remove duplicate texts
    normalize_unicode: bool = True  # NFC normalization for Devanagari


@dataclass
class HindiTokenizerConfig:
    """Configuration for Hindi tokenizer training."""

    # Tokenizer parameters
    vocab_size: int = 10000  # Vocabulary size (10K for Hindi+English)
    min_frequency: int = 2  # Minimum token frequency

    # Special tokens (must match MiniMind format)
    special_tokens: List[str] = field(default_factory=lambda: [
        "<|endoftext|>",  # ID 0: PAD/UNK token
        "<|im_start|>",   # ID 1: BOS token
        "<|im_end|>",     # ID 2: EOS token
    ])

    # Output paths
    tokenizer_dir: str = "./model_hindi"

    # Training data
    max_training_samples: Optional[int] = 500000  # Samples for tokenizer training

    # Chat template settings
    default_system_message: str = "आप एक मददगार AI सहायक हैं। You are a helpful assistant."


@dataclass
class HindiModelConfig:
    """Configuration for Hindi model architecture."""

    # Model size (Base model - 104M params)
    hidden_size: int = 768
    num_hidden_layers: int = 16
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    vocab_size: int = 10000

    # Sequence length
    max_seq_len: int = 512  # For pretraining
    max_sft_seq_len: int = 1024  # For SFT

    # Architecture options
    use_moe: bool = False
    flash_attn: bool = True

    # Output directories
    save_dir: str = "./out_hindi"
    checkpoint_dir: str = "./checkpoints_hindi"


@dataclass
class HindiTrainingConfig:
    """Configuration for Hindi model training."""

    # Pretraining
    pretrain_epochs: int = 2
    pretrain_batch_size: int = 32
    pretrain_learning_rate: float = 5e-4
    pretrain_accumulation_steps: int = 8

    # SFT
    sft_epochs: int = 2
    sft_batch_size: int = 16
    sft_learning_rate: float = 1e-6
    sft_accumulation_steps: int = 1

    # Common settings
    grad_clip: float = 1.0
    num_workers: int = 8
    dtype: str = "bfloat16"  # "bfloat16" or "float16"

    # Logging
    log_interval: int = 100
    save_interval: int = 1000
    use_wandb: bool = False
    wandb_project: str = "MiniMind-Hindi"


# =============================================================================
# DEFAULT CONFIGURATIONS
# =============================================================================

# Create default instances
DEFAULT_DATASET_CONFIG = HindiDatasetConfig()
DEFAULT_TOKENIZER_CONFIG = HindiTokenizerConfig()
DEFAULT_MODEL_CONFIG = HindiModelConfig()
DEFAULT_TRAINING_CONFIG = HindiTrainingConfig()


# =============================================================================
# QUICK ACCESS FUNCTIONS
# =============================================================================

def get_hf_dataset_path() -> str:
    """Get the full HuggingFace dataset path."""
    cfg = DEFAULT_DATASET_CONFIG
    return f"{cfg.hf_dataset_name}"


def get_hf_dataset_config() -> dict:
    """Get HuggingFace dataset loading configuration."""
    cfg = DEFAULT_DATASET_CONFIG
    return {
        "path": cfg.hf_dataset_name,
        "name": cfg.hf_dataset_subset,
        "split": cfg.hf_dataset_split,
        "streaming": cfg.streaming,
    }


def print_config():
    """Print all configuration settings."""
    print("=" * 60)
    print("HINDI TRAINING CONFIGURATION")
    print("=" * 60)

    print("\n[Dataset]")
    cfg = DEFAULT_DATASET_CONFIG
    print(f"  HuggingFace Dataset: {cfg.hf_dataset_name}")
    print(f"  Subset: {cfg.hf_dataset_subset}")
    print(f"  Language: {cfg.hf_dataset_language}")
    print(f"  Max Samples: {cfg.max_samples or 'All'}")
    print(f"  Streaming: {cfg.streaming}")

    print("\n[Tokenizer]")
    cfg = DEFAULT_TOKENIZER_CONFIG
    print(f"  Vocab Size: {cfg.vocab_size}")
    print(f"  Output Dir: {cfg.tokenizer_dir}")

    print("\n[Model]")
    cfg = DEFAULT_MODEL_CONFIG
    print(f"  Hidden Size: {cfg.hidden_size}")
    print(f"  Layers: {cfg.num_hidden_layers}")
    print(f"  Vocab Size: {cfg.vocab_size}")

    print("\n[Training]")
    cfg = DEFAULT_TRAINING_CONFIG
    print(f"  Pretrain Epochs: {cfg.pretrain_epochs}")
    print(f"  Pretrain Batch Size: {cfg.pretrain_batch_size}")
    print(f"  SFT Epochs: {cfg.sft_epochs}")

    print("=" * 60)


if __name__ == "__main__":
    print_config()