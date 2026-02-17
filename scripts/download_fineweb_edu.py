#!/usr/bin/env python
"""
Download HuggingFaceFW/fineweb-edu and prepare it for MiniMind pretraining.

Downloads the full sub-dataset (no streaming), then uses multiprocessing for
filtering and tokenization.

Supports two output formats:
  jsonl  - Raw text saved as JSONL (tokenized on-the-fly during training)
  binary - Pre-tokenized uint16 flat binary (fastest training, no tokenizer at train time)

Examples
--------
# 1) Download 100K docs as JSONL:
python scripts/download_fineweb_edu.py --max_samples 100000

# 2) Pre-tokenize to binary (parallel, 8 workers):
python scripts/download_fineweb_edu.py --format binary --max_samples 500000 --num_workers 8

# 3) Specific Common-Crawl subset:
python scripts/download_fineweb_edu.py --dataset_subset CC-MAIN-2025-26 --max_samples 100000

# 4) Train after downloading (JSONL):
python trainer/train_pretrain.py \
    --data_path dataset/fineweb_edu_train.jsonl \
    --eval_data_path dataset/fineweb_edu_eval.jsonl \
    --tokenizer_path tokenizer --vocab_size 32000

# 5) Train after downloading (binary):
python trainer/train_pretrain.py \
    --data_path dataset/fineweb_edu/train.bin \
    --eval_data_path dataset/fineweb_edu/eval.bin \
    --tokenizer_path tokenizer --vocab_size 32000
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------

def save_jsonl(train_ds, eval_ds, output_dir, output_name):
    """Export HF Dataset objects to JSONL files (keeps only the 'text' column)."""
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, f"{output_name}_train.jsonl")
    eval_path = os.path.join(output_dir, f"{output_name}_eval.jsonl")

    # Keep only text column for smaller files
    extra_cols = [c for c in train_ds.column_names if c != "text"]
    if extra_cols:
        train_ds = train_ds.remove_columns(extra_cols)
        if eval_ds is not None:
            eval_ds = eval_ds.remove_columns(extra_cols)

    train_ds.to_json(train_path, force_ascii=False)
    print(f"Train: {len(train_ds):,} samples -> {train_path}")

    if eval_ds is not None and len(eval_ds) > 0:
        eval_ds.to_json(eval_path, force_ascii=False)
        print(f"Eval:  {len(eval_ds):,} samples -> {eval_path}")
    else:
        eval_path = None

    return train_path, eval_path


# ---------------------------------------------------------------------------
# Binary output
# ---------------------------------------------------------------------------

def save_binary(train_ds, eval_ds, output_dir, tokenizer_path, num_workers):
    """Tokenize with multiprocessing and write raw uint16/int32 binary files."""
    import numpy as np
    from tokenizers import Tokenizer

    tok_file = os.path.join(tokenizer_path, "tokenizer.json")
    if not os.path.exists(tok_file):
        print(f"Error: tokenizer.json not found at {tok_file}", file=sys.stderr)
        sys.exit(1)

    tokenizer = Tokenizer.from_file(tok_file)
    vocab_size = tokenizer.get_vocab_size()
    dtype = np.uint16 if vocab_size < 65535 else np.int32

    # Resolve EOS token id for document boundary markers
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        eos_id = tokenizer.token_to_id("</s>")
    if eos_id is None:
        eos_id = tokenizer.token_to_id("<|endoftext|>")
    print(f"Tokenizer: vocab_size={vocab_size}, dtype={np.dtype(dtype).name}, eos_id={eos_id}")

    def tokenize_batch(examples):
        """Batch tokenization — appends EOS after each document."""
        all_ids = []
        for text in examples["text"]:
            ids = tokenizer.encode(text).ids
            if eos_id is not None:
                ids.append(eos_id)
            all_ids.append(ids)
        return {"input_ids": all_ids}

    def tokenize_and_save(ds, path, label):
        """Tokenize a dataset in parallel and write flat binary."""
        tok_ds = ds.map(
            tokenize_batch,
            batched=True,
            batch_size=1000,
            num_proc=num_workers,
            remove_columns=ds.column_names,
            desc=f"Tokenizing {label}",
        )
        # Flatten variable-length lists into one token stream
        all_tokens = []
        for row in tok_ds:
            all_tokens.extend(row["input_ids"])
        tokens = np.array(all_tokens, dtype=dtype)
        tokens.tofile(path)
        print(f"{label}: {len(tokens):,} tokens ({np.dtype(dtype).name}) -> {path}")
        return len(tokens)

    bin_dir = os.path.join(output_dir, "fineweb_edu")
    os.makedirs(bin_dir, exist_ok=True)

    train_path = os.path.join(bin_dir, "train.bin")
    tokenize_and_save(train_ds, train_path, "Train")

    eval_path = None
    if eval_ds is not None and len(eval_ds) > 0:
        eval_path = os.path.join(bin_dir, "eval.bin")
        tokenize_and_save(eval_ds, eval_path, "Eval")

    return train_path, eval_path, vocab_size


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFaceFW/fineweb-edu for MiniMind pretraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset selection
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu",
                        help="HuggingFace dataset id (default: HuggingFaceFW/fineweb-edu)")
    parser.add_argument("--dataset_subset", type=str, default=None,
                        help="Subset / config name (default: None, i.e. no subset). "
                             "FineWeb-Edu values: sample-10BT, sample-100BT, sample-350BT, "
                             "or CC-MAIN-20XX-YY for raw Common Crawl dumps. "
                             "Omit for datasets with no subsets.")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split (default: train)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max documents to download. Uses HF slice notation so only "
                             "the requested portion is fetched (default: all)")

    # Output
    parser.add_argument("--format", type=str, default="jsonl", choices=["jsonl", "binary"],
                        help="Output format (default: jsonl)")
    parser.add_argument("--output_dir", type=str, default="dataset",
                        help="Output directory (default: dataset)")
    parser.add_argument("--output_name", type=str, default="fineweb_edu",
                        help="Filename prefix for JSONL output (default: fineweb_edu)")

    # Tokenizer (binary mode only)
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer",
                        help="Path to tokenizer dir (for binary mode, default: tokenizer)")

    # Processing
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel workers for filtering / tokenization (default: 8)")
    parser.add_argument("--eval_ratio", type=float, default=0.01,
                        help="Fraction held out for evaluation (default: 0.01)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/eval split (default: 42)")

    # Auth
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace access token (for gated datasets)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    print(f"Dataset:     {args.dataset_name}")
    print(f"Subset:      {args.dataset_subset or '(none)'}")
    print(f"Split:       {args.split}")
    print(f"Format:      {args.format}")
    print(f"Max docs:    {args.max_samples or 'all'}")
    print(f"Workers:     {args.num_workers}")
    print(f"Eval ratio:  {args.eval_ratio}")
    print()

    # ------------------------------------------------------------------
    # 1. Download the full sub-dataset
    # ------------------------------------------------------------------
    from datasets import load_dataset

    # Use HF split slicing to avoid downloading more data than needed
    split_str = args.split
    if args.max_samples:
        split_str = f"{args.split}[:{args.max_samples}]"

    load_kwargs = dict(
        path=args.dataset_name,
        split=split_str,
    )
    if args.dataset_subset:
        load_kwargs["name"] = args.dataset_subset
    if args.hf_token:
        load_kwargs["token"] = args.hf_token

    try:
        print(f"Downloading {args.dataset_name}/{args.dataset_subset} ({split_str}) ...")
        ds = load_dataset(**load_kwargs)
    except Exception as e:
        print(f"\nFailed to load dataset: {e}", file=sys.stderr)
        print("\nTips:", file=sys.stderr)
        print("  - Check subset name: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu", file=sys.stderr)
        print("  - For gated datasets, pass --hf_token YOUR_TOKEN", file=sys.stderr)
        sys.exit(1)

    print(f"Downloaded {len(ds):,} samples")

    # ------------------------------------------------------------------
    # 2. Filter empty texts (parallel)
    # ------------------------------------------------------------------
    before = len(ds)
    ds = ds.filter(
        lambda batch: [bool(t and t.strip()) for t in batch["text"]],
        batched=True,
        num_proc=args.num_workers,
        desc="Filtering empty texts",
    )
    after = len(ds)
    if before != after:
        print(f"Filtered: {before:,} -> {after:,} (removed {before - after:,} empty)")

    # ------------------------------------------------------------------
    # 3. Train / eval split
    # ------------------------------------------------------------------
    if args.eval_ratio > 0:
        splits = ds.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        train_ds, eval_ds = splits["train"], splits["test"]
    else:
        train_ds, eval_ds = ds, None

    n_eval = len(eval_ds) if eval_ds is not None else 0
    print(f"Split: {len(train_ds):,} train / {n_eval:,} eval")
    print()

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    if args.format == "jsonl":
        train_path, eval_path = save_jsonl(
            train_ds, eval_ds, args.output_dir, args.output_name,
        )
        print(f"\nTo train:")
        print(f"  python trainer/train_pretrain.py \\")
        print(f"    --data_path {train_path} \\")
        if eval_path:
            print(f"    --eval_data_path {eval_path} \\")
        print(f"    --tokenizer_path tokenizer --vocab_size 32000")
    else:
        train_path, eval_path, vocab_size = save_binary(
            train_ds, eval_ds, args.output_dir, args.tokenizer_path, args.num_workers,
        )
        print(f"\nTo train:")
        parts = [f"  python trainer/train_pretrain.py"]
        parts.append(f"    --data_path {train_path}")
        if eval_path:
            parts.append(f"    --eval_data_path {eval_path}")
        parts.append(f"    --tokenizer_path tokenizer --vocab_size {vocab_size}")
        print(" \\\n".join(parts))


if __name__ == "__main__":
    main()