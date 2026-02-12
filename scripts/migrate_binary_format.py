"""
Migrate pickled binary file to valid binary format

Current: train.bin contains pickled Python objects
Target: int32 numpy array (memory-mapped, non-pickled)

Usage:
    python scripts/migrate_binary_format.py [--no-backup]

Note: Backup is optional and skipped by default if disk space is low.
"""
import numpy as np
import os
import pickle
import argparse

# Configuration
SEQUENCE_LENGTH = 1024  # Expected sequence length
TYPE = np.int32  # int32 for compatibility


def get_input_path(filename):
    """Get correct path for input file - handles being run from different directories"""
    # Method 1: Check current directory first
    cwd = os.getcwd()
    if os.path.exists(filename):
        print(f"[INFO] Found {filename} in current directory")
        return filename

    # Method 2: Look in parent directories for dataset folder
    for _ in range(3):
        test_path = os.path.dirname(cwd) if _ == 0 else os.path.dirname(cwd)
        if os.path.exists(os.path.join(test_path, 'dataset')):
            print(f"[INFO] Found dataset directory at: {test_path}")
            return os.path.join(test_path, filename)

    # Method 3: Use project root if nothing else works
    print(f"[WARN] Could not auto-detect location, using current directory")
    return os.path.abspath(os.path.join(cwd, filename))


def migrate_pickled_to_binary(no_backup=True):
    """Load pickled binary file and save as proper binary format"""

    # Get correct input path (works from any running location)
    input_file = get_input_path("dataset/train.bin")

    if not os.path.exists(input_file):
        print(f"[ERROR] Input file not found: {input_file}")
        print(f"       Searched in: {os.getcwd()}")
        return False

    # Get file size
    file_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    print(f"[INFO] Input file: {input_file}")
    print(f"       Size: {file_size:.2f} MB")
    print()

    # Load the pickled data
    print("[STEP 1] Loading pickled data...")
    try:
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        print(f"       Loaded {len(data)} objects")
    except Exception as e:
        print(f"[ERROR] Failed to load pickle file: {e}")
        return False

    # Extract tokens from pickled objects
    print("[STEP 2] Extracting tokens...")
    all_tokens = []
    for obj in data:
        # Handle different possible formats
        if isinstance(obj, dict) and 'tokens' in obj:
            all_tokens.extend(obj['tokens'])
        elif isinstance(obj, (list, tuple)):
            all_tokens.extend(obj)
        elif isinstance(obj, int):
            all_tokens.append(obj)
        else:
            print(f"[WARNING] Unexpected object type: {type(obj)}, skipping...")

    print(f"       Total tokens: {len(all_tokens)}")

    if not all_tokens:
        print("[ERROR] No valid tokens found in data")
        return False

    # Trim to exact sequence length
    print(f"[STEP 3] Creating {SEQUENCE_LENGTH}-length sequences...")
    num_seqs = len(all_tokens) // SEQUENCE_LENGTH
    total_tokens = num_seqs * SEQUENCE_LENGTH

    # Create int32 array
    tokens_array = np.array(all_tokens[:total_tokens], dtype=TYPE)
    print(f"       Sequences: {num_seqs}")
    print(f"       Total tokens: {len(tokens_array)}")
    print(f"       Array shape: {tokens_array.shape}")
    print(f"       Dtype: {tokens_array.dtype}")

    # Backup original file (optional)
    if no_backup:
        print("[STEP 4] Skipping backup (no-backup flag)")
        backup_path = None
    else:
        print("[STEP 4] Backing up original file...")
        backup_path = input_file + ".pickle_backup"
        print(f"       Backup: {backup_path}")
        try:
            os.rename(input_file, backup_path)
            print(f"       Done: {os.path.getsize(backup_path) / (1024*1024):.2f} MB")
        except Exception as e:
            print(f"[ERROR] Failed to backup: {e}")
            return False

    # Save as binary (NOT pickled)
    print(f"[STEP 5] Saving as binary format...")
    print(f"       Output: {input_file}")
    try:
        np.save(input_file, tokens_array, allow_pickle=False)
        print(f"       Done!")
    except Exception as e:
        print(f"[ERROR] Failed to save: {e}")
        return False

    # Verify
    new_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    print()
    print(f"[SUCCESS] Migration complete!")
    print(f"   Original: {file_size:.2f} MB (pickled)")
    print(f"   New: {new_size:.2f} MB (binary)")
    if backup_path:
        print(f"   Backup: {backup_path}")
    else:
        print(f"   No backup created (no-backup)")
    print()
    print("You can now run training with:")
    print(f"   python trainer/train_pretrain.py --data_path ../dataset/train.bin ...")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate pickled binary file to valid binary format")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup to save disk space")
    args = parser.parse_args()

    migrate_pickled_to_binary(no_backup=args.no_backup)
