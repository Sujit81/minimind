"""
Quick workaround: Use absolute path for tokenizer

This modifies train_pretrain.py to accept absolute paths for tokenizer.
"""
import os

# Read the train_pretrain.py file
script_path = os.path.join(os.path.dirname(__file__), 'train_pretrain.py')
with open(script_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the line that sets tokenizer_path
# Replace:
#     parser.add_argument("--tokenizer_path", type=str, default="../model")
# With:
#     parser.add_argument("--tokenizer_path", type=str, default="../model", help="tokenizer路径 (支持绝对路径)")

old_line = 'parser.add_argument("--tokenizer_path", type=str, default="../model", help="tokenizer路径")'
new_line = 'parser.add_argument("--tokenizer_path", type=str, default="D:/Code/Python/poc/minimind/tokenizer", help="tokenizer路径 (支持绝对路径)")'

# Apply the fix
new_content = content.replace(old_line, new_line)

# Write back
with open(script_path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("[OK] Fixed tokenizer_path in train_pretrain.py")
print("      Now --tokenizer_path defaults to your absolute path")
print("      You can also use: --tokenizer_path tokenizer (relative)")
print()
print("Please run your training command again.")
