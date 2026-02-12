"""
Download single file from Hugging Face dataset repository

Usage:
    python scripts/download_hf_file.py

Requirements:
    pip install huggingface_hub
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from huggingface_hub import hf_hub_download

# Configuration
REPO_ID = "sujitpandey/pretrain_english_jsonl"
FILENAME = "english_pretrain.jsonl"
LOCAL_DIR = "../dataset"
REPO_TYPE = "dataset"


def download_file():
    """Download single file from Hugging Face Hub"""

    # Convert to absolute path
    local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), LOCAL_DIR))
    os.makedirs(local_dir, exist_ok=True)

    local_file_path = os.path.join(local_dir, FILENAME)

    print(f"[HF File Download]")
    print(f"   Repository: {REPO_ID}")
    print(f"   File: {FILENAME}")
    print(f"   Save to: {local_file_path}")
    print()

    # Check if file already exists
    if os.path.exists(local_file_path):
        file_size = os.path.getsize(local_file_path) / (1024 * 1024)  # MB
        print(f"[WARNING] File already exists ({file_size:.2f} MB)")
        response = input("Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("[Cancelled] Download aborted")
            return False

    try:
        print("[Status] Downloading from Hugging Face...")
        print("   (This may take a while for large files)")
        print()

        # Download single file
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
        print(f"[SUCCESS] Download complete!")
        print(f"   Size: {file_size:.2f} MB")
        print(f"   Saved to: {downloaded_path}")
        return True

    except Exception as e:
        print()
        print(f"[ERROR] Download failed: {e}")
        print()
        print("[Troubleshooting]")
        print("  1. Check your internet connection")
        print("  2. Verify repository exists: https://huggingface.co/datasets/" + REPO_ID)
        print("  3. Make sure you have read access to repository")
        return False


if __name__ == "__main__":
    download_file()
