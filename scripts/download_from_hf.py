"""
Download dataset from Hugging Face Hub to local dataset folder

Usage:
    python scripts/download_from_hf.py

Requirements:
    pip install huggingface_hub datasets
"""
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from huggingface_hub import hf_hub_download
from huggingface_hub import HfFileSystem

# Configuration
REPO_ID = "sujitpandey/pretrain_english_jsonl"
FILENAME = "english_pretrain.jsonl"
LOCAL_DIR = "../dataset"  # Relative to script location
REPO_TYPE = "dataset"


def download_from_huggingface():
    """Download dataset from Hugging Face Hub"""

    # Convert to absolute path
    local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), LOCAL_DIR))

    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    local_file_path = os.path.join(local_dir, FILENAME)

    print(f"📦 Repository: {REPO_ID}")
    print(f"📁 File: {FILENAME}")
    print(f"💾 Local path: {local_file_path}")
    print()

    # Check if file already exists
    if os.path.exists(local_file_path):
        file_size = os.path.getsize(local_file_path) / (1024 * 1024)  # MB
        print(f"⚠️  File already exists ({file_size:.2f} MB)")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ Download cancelled")
            return False
        print()

    try:
        print("⬇️  Downloading from Hugging Face...")
        print()

        # Method 1: Using hf_hub_download (simple, for single file)
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type=REPO_TYPE,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        file_size = os.path.getsize(downloaded_path) / (1024 * 1024)  # MB
        print()
        print(f"✅ Download successful!")
        print(f"📏 Size: {file_size:.2f} MB")
        print(f"💾 Saved to: {downloaded_path}")
        return True

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Verify the repository exists: https://huggingface.co/datasets/" + REPO_ID)
        print("  3. Make sure you have read access to the repository")
        return False


def download_with_datasets():
    """Alternative: Using datasets library (better for large files)"""

    # Convert to absolute path
    local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), LOCAL_DIR))
    os.makedirs(local_dir, exist_ok=True)

    local_file_path = os.path.join(local_dir, FILENAME)

    print(f"📦 Using datasets library")
    print(f"📦 Repository: {REPO_ID}")
    print(f"💾 Local path: {local_file_path}")
    print()

    # Check if file already exists
    if os.path.exists(local_file_path):
        file_size = os.path.getsize(local_file_path) / (1024 * 1024)
        print(f"⚠️  File already exists ({file_size:.2f} MB)")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ Download cancelled")
            return False

    try:
        from datasets import load_dataset

        print("⬇️  Loading dataset from Hugging Face...")
        print("   (This may take a while for large files)")
        print()

        # Load dataset (doesn't download to disk yet, keeps in memory)
        dataset = load_dataset(REPO_ID, split="train")

        print(f"📊 Dataset loaded: {len(dataset)} examples")
        print()
        print("💾 Saving to local file...")

        # Save to jsonl
        import json
        with open(local_file_path, 'w', encoding='utf-8') as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        file_size = os.path.getsize(local_file_path) / (1024 * 1024)
        print()
        print(f"✅ Download successful!")
        print(f"📊 Examples: {len(dataset)}")
        print(f"📏 Size: {file_size:.2f} MB")
        print(f"💾 Saved to: {local_file_path}")
        return True

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Install datasets: pip install datasets")
        print("  2. Check your internet connection")
        print("  3. Verify the repository exists")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download dataset from Hugging Face")
    parser.add_argument("--method", choices=["hub", "datasets"], default="hub",
                      help="Download method: 'hub' (faster, direct) or 'datasets' (loads into memory)")
    args = parser.parse_args()

    if args.method == "datasets":
        download_with_datasets()
    else:
        download_from_huggingface()