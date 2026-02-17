#!/usr/bin/env python
"""
Download HuggingFace datasets and prepare them for MiniMind pretraining.

Processes parquet files one at a time to avoid memory issues with large datasets.
Each file is downloaded, processed, and written to output before moving to the next.

Supports two output formats:
  jsonl  - Raw text saved as JSONL (tokenized on-the-fly during training)
  binary - Pre-tokenized uint16 flat binary (fastest training, no tokenizer at train time)

Examples
--------
# 1) FineWeb-Edu CC-MAIN subset, 3M docs, binary:
python scripts/download_fineweb_edu.py \
    --dataset_subset CC-MAIN-2025-26 --format binary --max_samples 3000000 --num_workers 8

# 2) Dataset with no subset:
python scripts/download_fineweb_edu.py \
    --dataset_name blo05/cleaned_wiki_en --format binary --max_samples 100000

# 3) Download as JSONL:
python scripts/download_fineweb_edu.py \
    --dataset_subset sample-10BT --max_samples 100000

# 4) Train after downloading (binary):
python trainer/train_pretrain.py \
    --data_path dataset/fineweb_edu/train.bin \
    --eval_data_path dataset/fineweb_edu/eval.bin \
    --tokenizer_path tokenizer --vocab_size 32000

# 5) Train after downloading (JSONL):
python trainer/train_pretrain.py \
    --data_path dataset/fineweb_edu_train.jsonl \
    --eval_data_path dataset/fineweb_edu_eval.jsonl \
    --tokenizer_path tokenizer --vocab_size 32000
"""

import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Discover parquet files on HuggingFace Hub
# ---------------------------------------------------------------------------

def find_parquet_files(dataset_name, dataset_subset=None, split="train", hf_token=None):
    """List parquet files for a dataset on HuggingFace Hub."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=hf_token)
    base = f"datasets/{dataset_name}"

    # Try patterns in order of specificity
    patterns = []
    if dataset_subset:
        patterns.append(f"{base}/data/{dataset_subset}/*.parquet")
        patterns.append(f"{base}/{dataset_subset}/*.parquet")
        patterns.append(f"{base}/data/{dataset_subset}/**/*.parquet")
    patterns.append(f"{base}/data/{split}-*.parquet")
    patterns.append(f"{base}/data/*.parquet")
    patterns.append(f"{base}/*.parquet")

    prefix = base + "/"
    for pattern in patterns:
        files = sorted(fs.glob(pattern))
        if files:
            # Convert HfFileSystem paths to repo-relative paths for hf_hub_download
            return [f[len(prefix):] if f.startswith(prefix) else f for f in files]

    raise ValueError(
        f"No parquet files found for {dataset_name}"
        + (f" (subset={dataset_subset})" if dataset_subset else "")
    )


def prefetch_parquet_files(dataset_name, parquet_files, num_workers, hf_token=None):
    """Download parquet files in parallel into HF cache."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from huggingface_hub import hf_hub_download

    def _download(pf):
        return hf_hub_download(
            repo_id=dataset_name, filename=pf, repo_type="dataset", token=hf_token,
        )

    local_paths = {}
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_download, pf): pf for pf in parquet_files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading parquet files"):
            pf = futures[fut]
            local_paths[pf] = fut.result()

    # Return in original order
    return [local_paths[pf] for pf in parquet_files]


# ---------------------------------------------------------------------------
# Read one parquet file and yield texts
# ---------------------------------------------------------------------------

def read_parquet_texts(local_path):
    """Read a single parquet file and return list of text strings."""
    import pyarrow.parquet as pq

    table = pq.read_table(local_path, columns=["text"])
    texts = table.column("text").to_pylist()
    del table
    return texts


# ---------------------------------------------------------------------------
# JSONL output (batch-by-file)
# ---------------------------------------------------------------------------

def save_jsonl_batched(parquet_local_paths, output_dir, output_name, max_samples, eval_ratio):
    """Process parquet files one at a time and write JSONL incrementally."""
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, f"{output_name}_train.jsonl")
    eval_path = os.path.join(output_dir, f"{output_name}_eval.jsonl")

    eval_every = int(1 / eval_ratio) if eval_ratio > 0 else 0
    n_train = n_eval = n_total = 0

    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(eval_path, "w", encoding="utf-8") as f_eval:

        pbar = tqdm(parquet_local_paths, desc="Processing parquet files")
        for local_path in pbar:
            if max_samples and n_total >= max_samples:
                break

            texts = read_parquet_texts(local_path)

            for text in texts:
                if max_samples and n_total >= max_samples:
                    break
                if not text or not str(text).strip():
                    continue

                line = json.dumps({"text": str(text)}, ensure_ascii=False) + "\n"
                if eval_every > 0 and n_total % eval_every == 0:
                    f_eval.write(line)
                    n_eval += 1
                else:
                    f_train.write(line)
                    n_train += 1
                n_total += 1

            pbar.set_postfix(total=f"{n_total:,}")
            del texts

    print(f"\nTrain: {n_train:,} samples -> {train_path}")
    print(f"Eval:  {n_eval:,} samples -> {eval_path}")
    return train_path, eval_path if n_eval > 0 else None


# ---------------------------------------------------------------------------
# Binary output (batch-by-file)
# ---------------------------------------------------------------------------

def save_binary_batched(parquet_local_paths, output_dir, tokenizer_path, max_samples, eval_ratio):
    """Process parquet files one at a time, tokenize, write binary incrementally."""
    from tokenizers import Tokenizer

    tok_file = os.path.join(tokenizer_path, "tokenizer.json")
    if not os.path.exists(tok_file):
        print(f"Error: tokenizer.json not found at {tok_file}", file=sys.stderr)
        sys.exit(1)

    tokenizer = Tokenizer.from_file(tok_file)
    vocab_size = tokenizer.get_vocab_size()
    dtype = np.uint16 if vocab_size < 65535 else np.int32

    # EOS token for document boundaries
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        eos_id = tokenizer.token_to_id("</s>")
    if eos_id is None:
        eos_id = tokenizer.token_to_id("<|endoftext|>")
    print(f"Tokenizer: vocab_size={vocab_size}, dtype={np.dtype(dtype).name}, eos_id={eos_id}")

    bin_dir = os.path.join(output_dir, "fineweb_edu")
    os.makedirs(bin_dir, exist_ok=True)
    train_path = os.path.join(bin_dir, "train.bin")
    eval_path = os.path.join(bin_dir, "eval.bin")

    eval_every = int(1 / eval_ratio) if eval_ratio > 0 else 0
    n_total = 0
    n_train_tok = n_eval_tok = 0

    with open(train_path, "wb") as f_train, \
         open(eval_path, "wb") as f_eval:

        pbar = tqdm(parquet_local_paths, desc="Processing parquet files")
        for local_path in pbar:
            if max_samples and n_total >= max_samples:
                break

            texts = read_parquet_texts(local_path)

            # Limit to remaining budget
            if max_samples:
                texts = texts[: max_samples - n_total]

            # Filter empty
            texts = [str(t) for t in texts if t and str(t).strip()]
            if not texts:
                continue

            # Batch tokenize (tokenizers lib uses internal thread parallelism)
            encodings = tokenizer.encode_batch(texts)

            for i, enc in enumerate(encodings):
                ids = list(enc.ids)
                if eos_id is not None:
                    ids.append(eos_id)
                tokens = np.array(ids, dtype=dtype)

                if eval_every > 0 and (n_total + i) % eval_every == 0:
                    tokens.tofile(f_eval)
                    n_eval_tok += len(tokens)
                else:
                    tokens.tofile(f_train)
                    n_train_tok += len(tokens)

            n_total += len(texts)
            pbar.set_postfix(
                samples=f"{n_total:,}",
                train=f"{n_train_tok:,}",
                eval=f"{n_eval_tok:,}",
            )
            del texts, encodings

    print(f"\nTrain: {n_train_tok:,} tokens ({np.dtype(dtype).name}) -> {train_path}")
    eval_out = None
    if n_eval_tok > 0:
        print(f"Eval:  {n_eval_tok:,} tokens ({np.dtype(dtype).name}) -> {eval_path}")
        eval_out = eval_path
    else:
        os.remove(eval_path)

    return train_path, eval_out, vocab_size


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets and prepare for MiniMind pretraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset selection
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu",
                        help="HuggingFace dataset id (default: HuggingFaceFW/fineweb-edu)")
    parser.add_argument("--dataset_subset", type=str, default=None,
                        help="Subset / config name (default: None). "
                             "FineWeb-Edu values: sample-10BT, sample-100BT, "
                             "CC-MAIN-20XX-YY. Omit for datasets with no subsets.")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split (default: train)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max documents to process (default: all)")

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
                        help="Parallel workers for downloading parquet files (default: 8)")
    parser.add_argument("--eval_ratio", type=float, default=0.01,
                        help="Fraction held out for evaluation (default: 0.01)")

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
    # 1. Find parquet files on the Hub
    # ------------------------------------------------------------------
    print("Listing parquet files on Hub...")
    parquet_files = find_parquet_files(
        args.dataset_name, args.dataset_subset, args.split, args.hf_token,
    )
    print(f"Found {len(parquet_files)} parquet files")

    # Estimate how many files we actually need
    if args.max_samples:
        # Rough estimate: ~280K samples per file (typical for FineWeb-Edu)
        # Download a few extra files as buffer for empty-text filtering
        est_files = min(len(parquet_files), args.max_samples // 200_000 + 3)
        parquet_files = parquet_files[:est_files]
        print(f"Using first {len(parquet_files)} files (estimated sufficient for {args.max_samples:,} samples)")

    # ------------------------------------------------------------------
    # 2. Download parquet files in parallel into HF cache
    # ------------------------------------------------------------------
    local_paths = prefetch_parquet_files(
        args.dataset_name, parquet_files, args.num_workers, args.hf_token,
    )

    # ------------------------------------------------------------------
    # 3. Process file-by-file and write output incrementally
    # ------------------------------------------------------------------
    if args.format == "jsonl":
        train_path, eval_path = save_jsonl_batched(
            local_paths, args.output_dir, args.output_name,
            args.max_samples, args.eval_ratio,
        )
        print(f"\nTo train:")
        print(f"  python trainer/train_pretrain.py \\")
        print(f"    --data_path {train_path} \\")
        if eval_path:
            print(f"    --eval_data_path {eval_path} \\")
        print(f"    --tokenizer_path tokenizer --vocab_size 32000")
    else:
        train_path, eval_path, vocab_size = save_binary_batched(
            local_paths, args.output_dir, args.tokenizer_path,
            args.max_samples, args.eval_ratio,
        )
        print(f"\nTo train:")
        parts = ["  python trainer/train_pretrain.py"]
        parts.append(f"    --data_path {train_path}")
        if eval_path:
            parts.append(f"    --eval_data_path {eval_path}")
        parts.append(f"    --tokenizer_path tokenizer --vocab_size {vocab_size}")
        print(" \\\n".join(parts))


if __name__ == "__main__":
    main()
