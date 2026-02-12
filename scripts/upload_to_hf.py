"""
Upload english_pretrain.jsonl to Hugging Face Hub

Usage:
    python scripts/upload_to_hf.py

Requirements:
    pip install huggingface_hub
"""
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from huggingface_hub import HfApi, login

# Configuration
FILE_PATH = "../dataset/english_pretrain.jsonl"
REPO_ID = "sujitpandey/pretrain_english_jsonl"
REPO_TYPE = "dataset"
PRIVATE = False  # Set to False for public


def upload_to_huggingface():
    """Upload dataset to Hugging Face Hub"""

    # Convert to absolute path
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), FILE_PATH))

    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        print(f"   Current directory: {os.getcwd()}")
        return False

    # Initialize API
    api = HfApi()

    # Check if already logged in
    try:
        whoami = api.whoami()
        print(f"✅ Logged in as: {whoami['name']}")
    except Exception:
        print("⚠️  Not logged in to Hugging Face")
        print("   Please run: huggingface-cli login")
        print("   Or login will be prompted below...")
        try:
            login()
            whoami = api.whoami()
            print(f"✅ Logged in as: {whoami['name']}")
        except Exception as e:
            print(f"❌ Login failed: {e}")
            return False

    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    print(f"\n📁 File: {os.path.basename(file_path)}")
    print(f"📏 Size: {file_size:.2f} MB")
    print(f"📦 Repository: {REPO_ID}")
    print(f"🌐 Visibility: {'Public' if not PRIVATE else 'Private'}")
    print()

    try:
        # Check if repo exists and create if needed
        try:
            repo_info = api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
            print(f"✅ Repository exists: {repo_info.url}")
            existing = True
        except Exception as e:
            print(f"📝 Repository might not exist, creating...")
            print(f"   Info: {e}")
            try:
                api.create_repo(
                    repo_id=REPO_ID,
                    repo_type=REPO_TYPE,
                    private=PRIVATE,
                    exist_ok=True
                )
                print(f"✅ Repository created: {REPO_ID}")
            except Exception as create_error:
                # Repo already exists, that's fine
                if "already created" in str(create_error) or "409" in str(create_error):
                    print(f"✅ Repository already exists: {REPO_ID}")
                else:
                    raise create_error
            existing = False

        # Create README for dataset
        readme_content = r"""---
license: mit
task_names:
- text-generation
language:
- en
---

# English Pretraining Dataset

This dataset contains English text data for pretraining language models.

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("sujitpandey/pretrain_english_jsonl", split="train")
```

## Format

JSONL format with one sample per line:

```json
{"text": "Your English text here..."}
```
"""

        # Upload files
        print("⬆️  Uploading files...")

        # Upload the data file
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo="english_pretrain.jsonl",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message="Upload english_pretrain.jsonl"
        )

        # Upload README
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message="Add README"
        )

        print(f"\n✅ Upload successful!")
        print(f"🔗 View at: https://huggingface.co/datasets/{REPO_ID}")
        return True

    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you're logged in: huggingface-cli login")
        print("  2. Check if you have write access to the repository")
        print("  3. Verify the file path is correct")
        return False


if __name__ == "__main__":
    upload_to_huggingface()