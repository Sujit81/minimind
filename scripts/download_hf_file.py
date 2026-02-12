"""
Download files from Hugging Face dataset repository

Usage:
    python scripts/download_hf_file.py [--files all] [--files train.bin,val.bin]

Requirements:
    pip install huggingface_hub
"""
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from huggingface_hub import HfApi, hf_hub_download

# Configuration
REPO_ID = "sujitpandey/fineweb_edu"
LOCAL_DIR = "../dataset"
REPO_TYPE = "dataset"

# Available files in the repository
AVAILABLE_FILES = [
    "train.bin",         # 30.8 GB - pre-tokenized training data
    "val.bin",          # 155.1 MB - pre-tokenized validation data
    "train_meta.json",  # Metadata
    "val_meta.json",    # Metadata
]


def list_repo_files():
    """List all available files in the repository"""
    import os
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    from huggingface_hub import HfApi

    api = HfApi()
    repo = api.repo_info(REPO_ID, repo_type='dataset')
    siblings = api.list_repo_tree(REPO_ID, repo_type='dataset', token=os.environ.get('HF_TOKEN'))

    print("=" * 60)
    print("Repository: sujitpandey/fineweb_edu")
    print("=" * 60)
    print()
    print("Available files:")
    for i, f in enumerate(siblings, 1):
        size = f'{f.size / 1024 / 1024:.1f} MB' if f.size > 0 else '0 MB'
        print(f"  [{i}] {f.path}: {size}")
    print()


def download_files(files_to_download):
    """Download specified files from Hugging Face Hub"""

    # Convert to absolute path
    local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), LOCAL_DIR))
    os.makedirs(local_dir, exist_ok=True)

    if not files_to_download:
        return False

    print("=" * 60)
    print("[Download Configuration]")
    print(f"   Repository: {REPO_ID}")
    print(f"   Local dir: {local_dir}")
    print(f"   Files to download: {', '.join(files_to_download)}")
    print("=" * 60)
    print()

    # Check existing files
    existing_files = []
    for filename in files_to_download:
        local_file_path = os.path.join(local_dir, filename)
        if os.path.exists(local_file_path):
            file_size = os.path.getsize(local_file_path) / (1024 * 1024)  # MB
            existing_files.append(filename)
            print(f"[WARNING] {filename} already exists ({file_size:.1f} MB)")

    if existing_files:
        print()
        response = input("Continue with download? (Y/N): ").strip().lower()
        if response != 'y':
            print("[Cancelled] Download aborted")
            return False
        print()

    try:
        success_count = 0
        for filename in files_to_download:
            print(f"[{success_count + 1}/{len(files_to_download)}] Downloading {filename}...")

            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type=REPO_TYPE,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )

            file_size = os.path.getsize(downloaded_path) / (1024 * 1024)  # MB
            print(f"   Done: {file_size:.1f} MB")
            success_count += 1

        print()
        print("=" * 60)
        print("[SUCCESS] All files downloaded!")
        print(f"   Location: {local_dir}")
        print(f"   Downloaded: {success_count}/{len(files_to_download)} files")
        print("=" * 60)
        return True

    except Exception as e:
        print()
        print("=" * 60)
        print("[ERROR] Download failed!")
        print(f"   {e}")
        print()
        print("[Troubleshooting]")
        print("  1. Check your internet connection")
        print("  2. Verify repository: https://huggingface.co/datasets/" + REPO_ID)
        print("  3. Make sure you have read access")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download files from Hugging Face")
    parser.add_argument("--files", nargs="+", help="Files to download (e.g., train.bin val.bin) or 'all' for everything")
    parser.add_argument("--list", action="store_true", help="List all available files in repository")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode to select files")
    args = parser.parse_args()

    if args.list:
        list_repo_files()
    elif args.interactive:
        import os
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        from huggingface_hub import HfApi

        api = HfApi()
        siblings = api.list_repo_tree(REPO_ID, repo_type='dataset', token=os.environ.get('HF_TOKEN'))

        print("Available files:")
        for i, f in enumerate(siblings, 1):
            size = f'{f.size / 1024 / 1024:.1f} MB' if f.size > 0 else '0 MB'
            print(f"  [{i}] {f.path}: {size}")

        print()
        selections = input("Enter file numbers to download (comma-separated, or 'all'): ").strip()

        if selections.lower() == 'all':
            files_to_download = [f.path for f in siblings]
        else:
            indices = [int(x.strip()) for x in selections.split(',') if x.strip().isdigit()]
            files_to_download = [siblings[i].path for i in indices if 0 <= i < len(siblings)]

        if not files_to_download:
            print("[Cancelled] No files selected")
        else:
            download_files(files_to_download)
    elif args.files:
        download_files(args.files)
    else:
        # Default: download train.bin and val.bin
        print("[Default] Downloading train.bin and val.bin...")
        download_files(["train.bin", "val.bin"])
