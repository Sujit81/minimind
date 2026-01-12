# Hindi Tokenizer Training Script for MiniMind
# This script trains a SentencePiece tokenizer optimized for Hindi + English bilingual text
# यह स्क्रिप्ट हिंदी और अंग्रेजी द्विभाषी पाठ के लिए टोकननाइज़र को प्रशिक्षित करती है
#
# Uses SentencePiece with Unigram model (standard for Indic languages)
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

import sentencepiece as spm

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
    min_length: int = 50,
    character_coverage: float = 0.9995,
    model_type: str = "unigram"
):
    """
    Train a SentencePiece tokenizer for Hindi+English bilingual text.

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
        character_coverage: Character coverage (0.9995 recommended for Hindi-dominant bilingual)
        model_type: SentencePiece model type ("unigram" or "bpe")
    """
    # Validate inputs
    if not 0.0 < character_coverage <= 1.0:
        raise ValueError(f"character_coverage must be in (0, 1], got {character_coverage}")

    print("=" * 60)
    print("HINDI TOKENIZER TRAINING (SentencePiece)")
    print("=" * 60)
    print(f"Source: {source}")
    print(f"Subset: {hf_subset}")
    print(f"Vocab Size: {vocab_size}")
    print(f"Max Samples: {max_samples or 'All'}")
    print(f"Character Coverage: {character_coverage}")
    print(f"Model Type: {model_type}")
    print(f"Output: {tokenizer_dir}")
    print("=" * 60)

    # Create output directory
    os.makedirs(tokenizer_dir, exist_ok=True)

    # SentencePiece requires a text file, so we need to write data to temp file
    corpus_file = os.path.join(tokenizer_dir, "corpus_temp.txt")

    print("\nPreparing training corpus...")
    texts = get_texts(
        source=source,
        hf_subset=hf_subset,
        hf_split=hf_split,
        text_column=text_column,
        max_samples=max_samples,
        streaming=streaming,
        min_length=min_length
    )

    # Initialize Indic normalizer and syllabifier for Hindi
    indic_normalizer = None
    syllabify_func = None
    use_indic_norm = False
    use_syllabifier = False

    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        normalizer_factory = IndicNormalizerFactory()
        indic_normalizer = normalizer_factory.get_normalizer('hi')
        use_indic_norm = True
        print("  Using Indic NLP normalizer for Hindi")
    except ImportError:
        print("  Note: indic-nlp-library normalizer not available")

    # Note: Syllabifier removed - letting Unigram algorithm optimize freely
    # The indic_normalizer ensures consistent preprocessing
    # Use model/HindiTokenizer.py wrapper at inference for consistency

    # Write texts to file for SentencePiece
    # Note: We apply Indic-specific normalization here. The same normalization
    # MUST be applied at inference time using model/HindiTokenizer.py wrapper.
    count = 0
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for text in texts:
            text = text.strip()
            # Apply Indic-specific normalization (handles Devanagari quirks)
            # IMPORTANT: Use HindiTokenizer wrapper at inference for consistency
            if use_indic_norm:
                text = indic_normalizer.normalize(text)
            # Write each line
            for line in text.split('\n'):
                line = line.strip()
                if line:
                    f.write(line + '\n')
                    count += 1
            if count % 100000 == 0:
                print(f"  Written {count:,} lines...")

    print(f"✓ Corpus prepared: {count:,} lines")

    # Validate corpus before training
    if count == 0:
        raise RuntimeError("No valid training data found in corpus. Check data source and filters.")
    if count < 10000:
        print(f"  ⚠ Warning: Only {count:,} lines. Recommend at least 100,000 for good tokenization.")

    # ═══════════════════════════════════════════════════════════════
    # MINIMAL PROTECTED SYMBOLS
    # ═══════════════════════════════════════════════════════════════
    # Note: We intentionally use MINIMAL protected symbols to let the
    # Unigram algorithm optimize freely. Previous approach of forcing
    # 500+ syllables was overriding statistical optimization.
    #
    # Devanagari conjuncts render correctly even when split because:
    # - Conjunct = base + virama + consonant (e.g., क्ष = क + ् + ष)
    # - Decoding reconstructs the sequence, rendering engine handles display
    # - byte_fallback=True ensures no UNK tokens

    # Minimal protected symbols - only the 3 most common conjuncts
    # that have special typographic rendering and are extremely common.
    # Let Unigram algorithm handle everything else optimally.
    hindi_protected_symbols = [
        "क्ष",  # ksha - very common (क्षमा, रक्षा, शिक्षा)
        "त्र",  # tra - very common (त्रिभुज, मंत्र, सूत्र)
        "ज्ञ",  # gya - important (ज्ञान, विज्ञान, यज्ञ)
    ]

    print(f"  Protected symbols: {len(hindi_protected_symbols)} (minimal - letting Unigram optimize)")

    # Train SentencePiece with proper cleanup on failure
    print("\nTraining SentencePiece tokenizer...")
    model_prefix = os.path.join(tokenizer_dir, "tokenizer")

    try:
        spm.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            # Special tokens handling
            # SentencePiece reserves: unk=0, bos=1, eos=2
            # We use <|endoftext|> as unk (ID 0), which also serves as PAD
            pad_id=-1,  # Disable separate pad (we'll use unk as pad)
            unk_id=0,
            bos_id=1,
            eos_id=2,
            unk_piece="<|endoftext|>",
            bos_piece="<|im_start|>",
            eos_piece="<|im_end|>",
            # Minimal protected symbols (let Unigram optimize)
            user_defined_symbols=hindi_protected_symbols,
            # Training settings
            split_digits=True,
            byte_fallback=True,  # Handle unknown characters via bytes
            normalization_rule_name="nfkc",  # Unicode normalization
            num_threads=os.cpu_count() or 4,
            # For Indic languages
            split_by_unicode_script=True,
            split_by_whitespace=True,
            split_by_number=True,
            # Large corpus support
            train_extremely_large_corpus=True,
            input_sentence_size=5000000,  # Limit sentences for memory (5M)
            shuffle_input_sentence=True,
        )
    finally:
        # Clean up temp corpus file even on failure (can be GBs)
        if os.path.exists(corpus_file):
            os.remove(corpus_file)
            print(f"✓ Cleaned up temporary corpus file")

    print(f"✓ SentencePiece model saved to {model_prefix}.model")

    # Load and verify the trained model
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    print("\nVerifying special token IDs...")
    print(f"  PAD/UNK (<|endoftext|>): {sp.piece_to_id('<|endoftext|>')}")
    print(f"  BOS (<|im_start|>): {sp.piece_to_id('<|im_start|>')}")
    print(f"  EOS (<|im_end|>): {sp.piece_to_id('<|im_end|>')}")
    print(f"  Vocab size: {sp.get_piece_size()}")

    # Create tokenizer config with chat template
    # Use default system message from config
    default_sys_msg = DEFAULT_TOKENIZER_CONFIG.default_system_message
    chat_template = """{%- if messages[0]['role'] == 'system' -%}
    {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
{%- else -%}
    {{- '<|im_start|>system\\n""" + default_sys_msg + """<|im_end|>\\n' }}
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
        "tokenizer_class": "LlamaTokenizer",  # Use LlamaTokenizer for SentencePiece
        "unk_token": "<|endoftext|>",
        "chat_template": chat_template
    }

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    print(f"✓ Config saved to {tokenizer_dir}/tokenizer_config.json")
    print("\n" + "=" * 60)
    print("HINDI TOKENIZER TRAINING COMPLETED")
    print("=" * 60)
    print(f"Vocab size: {sp.get_piece_size()}")
    print(f"Output: {tokenizer_dir}")
    print("=" * 60)
    print("\nIMPORTANT: For inference, use the HindiTokenizer wrapper to ensure")
    print("consistent normalization between training and inference:")
    print("")
    print("  from model.HindiTokenizer import HindiTokenizer")
    print(f"  tokenizer = HindiTokenizer.from_pretrained('{tokenizer_dir}')")
    print("")
    print("This wrapper applies the same Indic normalization used during training.")
    print("=" * 60)


def eval_tokenizer(tokenizer_dir: str):
    """Evaluate the trained tokenizer with Hindi test cases."""
    print("\n" + "=" * 60)
    print("EVALUATING HINDI TOKENIZER (SentencePiece)")
    print("=" * 60)

    # Load SentencePiece model directly
    sp = spm.SentencePieceProcessor()
    model_path = os.path.join(tokenizer_dir, "tokenizer.model")
    sp.load(model_path)

    # Try loading with HuggingFace for chat template support
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        use_hf = True
    except Exception as e:
        print(f"Note: HuggingFace tokenizer not available ({e})")
        print("Using SentencePiece directly for evaluation")
        tokenizer = None
        use_hf = False

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

    if use_hf:
        for test in test_cases:
            print(f"\n--- {test['name']} ---")
            messages = test['messages']
            new_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            print(f"Prompt:\n{new_prompt}")

            model_inputs = tokenizer(new_prompt)
            print(f"Token count: {len(model_inputs['input_ids'])}")

            response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
            match = response == new_prompt
            print(f"Decode match: {match}")
            if not match:
                print(f"  Expected len: {len(new_prompt)}, Got len: {len(response)}")
                for i, (c1, c2) in enumerate(zip(new_prompt, response)):
                    if c1 != c2:
                        print(f"  First diff at pos {i}: expected {repr(c1)}, got {repr(c2)}")
                        print(f"  Context: ...{repr(new_prompt[max(0,i-5):i+10])}...")
                        break
                if len(new_prompt) != len(response):
                    print(f"  Length diff: expected={len(new_prompt)}, got={len(response)}")

    print("\n" + "=" * 60)
    print(f"Tokenizer vocab size: {sp.get_piece_size()}")
    print("=" * 60)

    # Test Hindi tokenization efficiency using SentencePiece directly
    print("\n--- Hindi Token Efficiency ---")
    hindi_words = [
        "है", "हैं", "था", "का", "की", "में", "भारत", "सरकार", "विकास", "शिक्षा"
    ]
    total_tokens = 0
    for word in hindi_words:
        tokens = sp.encode_as_pieces(word)
        total_tokens += len(tokens)
        print(f"  '{word:12}' -> {tokens} ({len(tokens)} tokens)")

    avg_tokens = total_tokens / len(hindi_words)
    print(f"\n  Average tokens per word: {avg_tokens:.2f} (target: ~1.4)")

    # Test common conjuncts
    print("\n--- Hindi Conjuncts ---")
    conjuncts = ["क्ष", "त्र", "ज्ञ", "श्र", "द्व"]
    for conj in conjuncts:
        tokens = sp.encode_as_pieces(conj)
        print(f"  '{conj}' -> {tokens} ({len(tokens)} tokens)")

    # Test encode/decode roundtrip
    print("\n--- Encode/Decode Test ---")
    test_texts = [
        "नमस्ते, मेरा नाम राहुल है।",
        "भारत एक महान देश है।",
        "Hello, this is a test in English.",
        "यह Hinglish में test है।"
    ]
    for text in test_texts:
        ids = sp.encode(text)
        decoded = sp.decode(ids)
        match = decoded == text
        print(f"  '{text[:30]:30}...' -> {len(ids):3} tokens, roundtrip: {'✓' if match else '✗'}")


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
    parser.add_argument('--character_coverage', type=float, default=0.9995,
                        help='Character coverage (0.9995 for Indic, 1.0 for Latin-only)')
    parser.add_argument('--model_type', type=str, default='unigram',
                        choices=['unigram', 'bpe'],
                        help='SentencePiece model type (unigram recommended for Indic)')

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
            min_length=args.min_length,
            character_coverage=args.character_coverage,
            model_type=args.model_type
        )

    eval_tokenizer(args.tokenizer_dir)
