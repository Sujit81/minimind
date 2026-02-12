"""
Quick fix for tokenizer path resolution in train_pretrain.py

The issue: tokenizer_path resolves relative to script directory, not project root.
This fix creates symlinks from project root to expected locations.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def setup_tokenizer_symlinks():
    """Create symlinks from project root to tokenizer folders"""
    # Get project root (2 levels up from scripts/ directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    print(f"[INFO] Project root: {project_root}")

    # Define tokenizer locations
    tokenizer_locations = {
        'tokenizer': os.path.join(project_root, 'tokenizer'),
        'model_english': os.path.join(project_root, 'model_english'),
    }

    # Create symlinks if they don't exist
    for name, target_path in tokenizer_locations.items():
        if os.path.exists(target_path):
            print(f"[SKIP] {name} already exists at {target_path}")
            continue

        # Check if source exists
        source_path = os.path.join(project_root, 'tokenizer')
        if not os.path.exists(source_path):
            print(f"[WARN] Source tokenizer/ not found at {source_path}")
            print(f"       Creating directory instead...")
            os.makedirs(target_path)
            # Create empty tokenizer_config.json to prevent errors
            import json
            config_path = os.path.join(target_path, 'tokenizer_config.json')
            if not os.path.exists(config_path):
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "model_type": "minimind",
                        "vocab_size": 32000
                    }, f, indent=2)
                print(f"       Created {config_path}")
            continue

        # Create symlink
        os.symlink(source_path, target_path)
        print(f"[OK] Created symlink: {name} -> {target_path}")
        print(f"     Source: {source_path}")
        print(f"     Target: {target_path}")

if __name__ == "__main__":
    setup_tokenizer_symlinks()
    print()
    print("[SUCCESS] Tokenizer symlinks created!")
    print("         Now you can use --tokenizer_path tokenizer")
    print("         This will resolve to: /data/minimind/tokenizer (project root)")
    print()
    print("Please run your training command again.")
