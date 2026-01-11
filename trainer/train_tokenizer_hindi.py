# Hindi Tokenizer Training Script for MiniMind
# This script trains a tokenizer optimized for Hindi + English bilingual text
# यह स्क्रिप्ट हिंदी और अंग्रेजी द्विभाषी पाठ के लिए टोकननाइज़र को प्रशिक्षित करती है
#
# Supports:
# - HuggingFace datasets (ai4bharat/sangraha)
# - Local text files (.txt, .jsonl)
# - Streaming for large datasets

import os
import sys
import json
import unicodedata
from typing import Iterator, Optional

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

# Import configuration
from config.hindi_config import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_TOKENIZER_CONFIG,
    print_config
)


def normalize_text(text: str) -> str:
    """Normalize Unicode text to NFC form for consistent Devanagari handling."""
    return unicodedata.normalize('NFC', text)


def is_valid_hindi_text(text: str, min_length: int = 50) -> bool:
    """Check if text is valid and contains sufficient content."""
    if not text or len(text) < min_length:
        return False
    # Basic check - could add more Devanagari-specific validation
    return True


def get_texts_from_huggingface(
    dataset_name: str,
    subset: str,
    split: str = "train",
    text_column: str = "text",
    max_samples: Optional[int] = None,
    streaming: bool = True,
    min_length: int = 50,
    max_retries: int = 5
) -> Iterator[str]:
    """
    Stream texts from a HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "ai4bharat/sangraha")
        subset: Dataset subset/config (e.g., "synthetic")
        split: Dataset split (e.g., "hin_Deva" for Hindi)
        text_column: Column containing text data
        max_samples: Maximum samples to yield (None for all)
        streaming: Use streaming mode for large datasets
        min_length: Minimum text length to include
        max_retries: Maximum retries on network failure
    """
    from datasets import load_dataset
    import time as time_module

    print(f"Loading dataset: {dataset_name} (subset: {subset}, split: {split})")
    print(f"Streaming mode: {streaming}")

    # Retry logic for network issues
    dataset = None
    for attempt in range(max_retries):
        try:
            dataset = load_dataset(
                dataset_name,
                subset,
                split=split,
                streaming=streaming,
                trust_remote_code=True
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # 10s, 20s, 30s...
                print(f"  Connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"  Retrying in {wait_time} seconds...")
                time_module.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts. Error: {e}")
                raise

    if dataset is None:
        raise RuntimeError("Failed to load dataset")

    count = 0
    skipped = 0

    for item in dataset:
        # Get text from the specified column
        text = item.get(text_column, "")

        if not text:
            skipped += 1
            continue

        # Normalize and validate
        text = normalize_text(text.strip())

        if not is_valid_hindi_text(text, min_length):
            skipped += 1
            continue

        yield text
        count += 1

        if count % 10000 == 0:
            print(f"  Processed: {count:,} samples (skipped: {skipped:,})")

        if max_samples and count >= max_samples:
            print(f"  Reached max_samples limit: {max_samples}")
            break

    print(f"  Total yielded: {count:,} samples (skipped: {skipped:,})")


def get_texts_from_txt(data_path: str) -> Iterator[str]:
    """Read texts from a plain text file."""
    print(f"Loading from text file: {data_path}")
    count = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = normalize_text(line.strip())
            if line:
                yield line
                count += 1
    print(f"  Loaded {count:,} lines")


def get_texts_from_jsonl(data_path: str, text_column: str = "text") -> Iterator[str]:
    """Read texts from JSONL format."""
    print(f"Loading from JSONL file: {data_path}")
    count = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get(text_column, "")
            if text:
                yield normalize_text(text.strip())
                count += 1
    print(f"  Loaded {count:,} lines")


def get_texts(
    source: str,
    hf_subset: str = None,
    hf_split: str = "train",
    text_column: str = "text",
    max_samples: Optional[int] = None,
    streaming: bool = True,
    min_length: int = 50
) -> Iterator[str]:
    """
    Get texts from various sources.

    Args:
        source: HuggingFace dataset name OR local file path
        hf_subset: Subset for HuggingFace datasets
        hf_split: Split for HuggingFace datasets
        text_column: Column name containing text
        max_samples: Maximum samples to process
        streaming: Use streaming for HuggingFace datasets
        min_length: Minimum text length
    """
    # Check if it's a local file
    if os.path.exists(source):
        if source.endswith('.txt'):
            yield from get_texts_from_txt(source)
        elif source.endswith('.jsonl'):
            yield from get_texts_from_jsonl(source, text_column)
        else:
            raise ValueError(f"Unsupported file format: {source}")
    else:
        # Assume it's a HuggingFace dataset
        yield from get_texts_from_huggingface(
            dataset_name=source,
            subset=hf_subset,
            split=hf_split,
            text_column=text_column,
            max_samples=max_samples,
            streaming=streaming,
            min_length=min_length
        )


def train_tokenizer(
    source: str,
    tokenizer_dir: str,
    vocab_size: int,
    hf_subset: str = None,
    hf_split: str = "train",
    text_column: str = "text",
    max_samples: Optional[int] = None,
    streaming: bool = True,
    min_length: int = 50
):
    """
    Train a BPE tokenizer for Hindi+English bilingual text.

    Args:
        source: Data source (HuggingFace dataset name or local file path)
        tokenizer_dir: Output directory for tokenizer
        vocab_size: Vocabulary size
        hf_subset: HuggingFace dataset subset
        hf_split: HuggingFace dataset split
        text_column: Column containing text
        max_samples: Maximum samples for training
        streaming: Use streaming mode
        min_length: Minimum text length
    """
    print("=" * 60)
    print("HINDI TOKENIZER TRAINING")
    print("=" * 60)
    print(f"Source: {source}")
    print(f"Subset: {hf_subset}")
    print(f"Vocab Size: {vocab_size}")
    print(f"Max Samples: {max_samples or 'All'}")
    print(f"Output: {tokenizer_dir}")
    print("=" * 60)

    # Get text iterator
    texts = get_texts(
        source=source,
        hf_subset=hf_subset,
        hf_split=hf_split,
        text_column=text_column,
        max_samples=max_samples,
        streaming=streaming,
        min_length=min_length
    )

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Define special tokens (Must match MiniMind format)
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

    # BPE trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # Train the tokenizer
    print("\nTraining tokenizer...")
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()

    # Verify special token IDs
    print("\nVerifying special token IDs...")
    assert tokenizer.token_to_id("<|endoftext|>") == 0, "PAD token must be ID 0"
    assert tokenizer.token_to_id("<|im_start|>") == 1, "BOS token must be ID 1"
    assert tokenizer.token_to_id("<|im_end|>") == 2, "EOS token must be ID 2"
    print("✓ Special token IDs verified: PAD=0, BOS=1, EOS=2")

    # Create output directory
    os.makedirs(tokenizer_dir, exist_ok=True)

    # Save tokenizer
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)
    print(f"✓ Tokenizer saved to {tokenizer_dir}")

    # Create tokenizer config with chat template
    chat_template = """{%- if messages[0]['role'] == 'system' -%}
    {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
{%- else -%}
    {{- '<|im_start|>system\\nआप एक मददगार AI सहायक हैं। You are a helpful assistant.<|im_end|>\\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}"""

    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        "chat_template": chat_template
    }

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    print(f"✓ Config saved to {tokenizer_dir}/tokenizer_config.json")
    print("\n" + "=" * 60)
    print("HINDI TOKENIZER TRAINING COMPLETED")
    print("=" * 60)
    print(f"Vocab size: {vocab_size}")
    print(f"Output: {tokenizer_dir}")
    print("=" * 60)


def eval_tokenizer(tokenizer_dir: str):
    """Evaluate the trained tokenizer with Hindi test cases."""
    from transformers import AutoTokenizer

    print("\n" + "=" * 60)
    print("EVALUATING HINDI TOKENIZER")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # Test messages (Hindi + English + Hinglish)
    test_cases = [
        {
            "name": "Pure Hindi",
            "messages": [
                {"role": "system", "content": "आप एक मददगार सहायक हैं।"},
                {"role": "user", "content": "भारत की राजधानी क्या है?"},
                {"role": "assistant", "content": "भारत की राजधानी नई दिल्ली है।"}
            ]
        },
        {
            "name": "Hinglish (Mixed)",
            "messages": [
                {"role": "user", "content": "मुझे Python सीखना है।"},
                {"role": "assistant", "content": "Python एक बहुत अच्छी programming language है।"}
            ]
        },
        {
            "name": "English",
            "messages": [
                {"role": "user", "content": "What is the capital of India?"},
                {"role": "assistant", "content": "The capital of India is New Delhi."}
            ]
        }
    ]

    for test in test_cases:
        print(f"\n--- {test['name']} ---")
        messages = test['messages']
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        print(f"Prompt:\n{new_prompt}")

        model_inputs = tokenizer(new_prompt)
        print(f"Token count: {len(model_inputs['input_ids'])}")

        response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
        print(f"Decode match: {response == new_prompt}")

    print("\n" + "=" * 60)
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print("=" * 60)

    # Test Hindi tokenization efficiency
    print("\n--- Hindi Token Efficiency ---")
    hindi_words = [
        "है", "हैं", "था", "का", "की", "में", "भारत", "सरकार", "विकास", "शिक्षा"
    ]
    total_tokens = 0
    for word in hindi_words:
        tokens = tokenizer.tokenize(word)
        total_tokens += len(tokens)
        print(f"  '{word:12}' -> {tokens} ({len(tokens)} tokens)")

    avg_tokens = total_tokens / len(hindi_words)
    print(f"\n  Average tokens per word: {avg_tokens:.2f} (target: < 2.0)")

    # Test common conjuncts
    print("\n--- Hindi Conjuncts ---")
    conjuncts = ["क्ष", "त्र", "ज्ञ", "श्र", "द्व"]
    for conj in conjuncts:
        tokens = tokenizer.tokenize(conj)
        print(f"  '{conj}' -> {tokens} ({len(tokens)} tokens)")


if __name__ == '__main__':
    import argparse

    # Load defaults from config
    ds_cfg = DEFAULT_DATASET_CONFIG
    tk_cfg = DEFAULT_TOKENIZER_CONFIG

    parser = argparse.ArgumentParser(
        description="Train Hindi tokenizer for MiniMind",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on HuggingFace dataset (default: ai4bharat/sangraha)
  python train_tokenizer_hindi.py

  # Train on specific subset with sample limit
  python train_tokenizer_hindi.py --source ai4bharat/sangraha --subset synthetic --max_samples 100000

  # Train on local file
  python train_tokenizer_hindi.py --source ../dataset/hindi/corpus_bilingual.txt

  # Evaluate existing tokenizer only
  python train_tokenizer_hindi.py --eval_only
        """
    )

    # Data source arguments
    parser.add_argument('--source', type=str, default=ds_cfg.hf_dataset_name,
                        help='HuggingFace dataset name OR local file path')
    parser.add_argument('--subset', type=str, default=ds_cfg.hf_dataset_subset,
                        help='HuggingFace dataset subset (e.g., synthetic)')
    parser.add_argument('--split', type=str, default=ds_cfg.hf_dataset_split,
                        help='Dataset split (default: hin_Deva for Hindi)')
    parser.add_argument('--text_column', type=str, default=ds_cfg.hf_text_column,
                        help='Column containing text data')
    parser.add_argument('--max_samples', type=int, default=tk_cfg.max_training_samples,
                        help='Maximum samples for tokenizer training (default: 500000)')
    parser.add_argument('--no_streaming', action='store_true',
                        help='Disable streaming mode (downloads ALL language splits - not recommended for sangraha)')
    parser.add_argument('--min_length', type=int, default=ds_cfg.min_text_length,
                        help='Minimum text length to include')

    # Tokenizer arguments
    parser.add_argument('--tokenizer_dir', type=str, default=tk_cfg.tokenizer_dir,
                        help='Output directory for tokenizer')
    parser.add_argument('--vocab_size', type=int, default=tk_cfg.vocab_size,
                        help='Vocabulary size')

    # Actions
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate existing tokenizer')
    parser.add_argument('--show_config', action='store_true',
                        help='Show current configuration and exit')

    args = parser.parse_args()

    if args.show_config:
        print_config()
        sys.exit(0)

    if not args.eval_only:
        train_tokenizer(
            source=args.source,
            tokenizer_dir=args.tokenizer_dir,
            vocab_size=args.vocab_size,
            hf_subset=args.subset,
            hf_split=args.split,
            text_column=args.text_column,
            max_samples=args.max_samples,
            streaming=not args.no_streaming,  # Default: True (streaming to avoid downloading all languages)
            min_length=args.min_length
        )

    eval_tokenizer(args.tokenizer_dir)
