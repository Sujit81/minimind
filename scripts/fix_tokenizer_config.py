"""Add model_type to tokenizer_config.json for compatibility"""
import json

# Read config
with open('model_english/tokenizer_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# Add model_type field
config['model_type'] = 'minimind'

# Write back
with open('model_english/tokenizer_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print('[OK] Added model_type to tokenizer_config.json')
print(f'   model_type: {config["model_type"]}')
print(f'   vocab_size: {config.get("vocab_size", "unknown")}')
print(f'   bos_token: {config.get("bos_token", "unknown")}')
print(f'   eos_token: {config.get("eos_token", "unknown")}')
