"""
English Tokenizer Training Script for MiniMind
Trains a BPE tokenizer with 16,000 vocabulary size optimized for English text.

Usage:
    python train_tokenizer_english.py --download fineweb --num_samples 100000
    python train_tokenizer_english.py --data_path /path/to/english_data.jsonl

Output will be saved to: minimind/model_english/
Data will be saved to: minimind/dataset/english_pretrain.jsonl
"""

import os
import json
import argparse
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer


DATASET_OPTIONS = {
    "fineweb": ("HuggingFaceFW/fineweb", "sample-10BT", "text"),
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "openwebtext": ("Skylion007/openwebtext", None, "text"),
    "wikitext": ("wikitext", "wikitext-103-raw-v1", "text"),
    "slimpajama": ("cerebras/SlimPajama-627B", None, "text"),
}


def download_english_data(output_path, dataset_name="openwebtext", num_samples=100000):
    """Download English text data from HuggingFace datasets.

    Available datasets:
        - fineweb: Best quality, 10B token sample (recommended)
        - fineweb-edu: Educational content subset
        - openwebtext: Reddit-filtered web text, 38GB
        - wikitext: Wikipedia text, fast to download
        - slimpajama: Diverse 627B token dataset
    """
    print(f"Downloading {dataset_name} dataset...")

    if dataset_name not in DATASET_OPTIONS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available options: {list(DATASET_OPTIONS.keys())}")
        return False

    try:
        from datasets import load_dataset

        repo, config, text_field = DATASET_OPTIONS[dataset_name]

        print(f"Loading {repo}" + (f" ({config})" if config else ""))
        if config:
            dataset = load_dataset(repo, config, split="train", streaming=True)
        else:
            dataset = load_dataset(repo, split="train", streaming=True)

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                if count >= num_samples:
                    break
                text = item.get(text_field, '')
                if text and len(text) > 100:  # Filter very short texts
                    f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                    count += 1
                    if count % 10000 == 0:
                        print(f"  Processed {count}/{num_samples} samples...")

        print(f"Downloaded {count} samples to {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Please provide your own English JSONL file with 'text' field.")
        return False


def get_texts(data_path, max_samples=None):
    """Generator that yields text from JSONL file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                data = json.loads(line)
                text = data.get('text', '')
                if text:
                    yield text
            except json.JSONDecodeError:
                continue


def train_tokenizer(data_path, tokenizer_dir, vocab_size=16000):
    """Train a BPE tokenizer on English text."""
    print(f"Training tokenizer with vocab_size={vocab_size}...")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,  # Tokens must appear at least twice
    )

    texts = get_texts(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()

    # Verify special tokens
    assert tokenizer.token_to_id("<|endoftext|>") == 0, "endoftext should be token 0"
    assert tokenizer.token_to_id("<|im_start|>") == 1, "im_start should be token 1"
    assert tokenizer.token_to_id("<|im_end|>") == 2, "im_end should be token 2"

    # Save tokenizer
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)

    # Create tokenizer config compatible with HuggingFace
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "0": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|endoftext|>",
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n {%- if messages[0]['role'] == 'system' -%}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else -%}\n        {{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}\n {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n   {{- '<|im_start|>' + message.role + '\\n' + content }}\n  {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
    }

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    print(f"Tokenizer saved to {tokenizer_dir}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")


def eval_tokenizer(tokenizer_dir):
    """Evaluate the trained tokenizer with English examples."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    print("\n" + "=" * 80)
    print("TOKENIZER EVALUATION")
    print("=" * 80)

    # Test with English conversations
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn from data and improve their performance without being explicitly programmed."}
    ]

    new_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print("\nChat template output:")
    print("-" * 40)
    print(new_prompt)

    print("\n" + "-" * 40)
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    model_inputs = tokenizer(new_prompt)
    print(f"Encoded length: {len(model_inputs['input_ids'])} tokens")

    response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
    print(f"Decode consistency: {response == new_prompt}")

    # Test compression ratio with English text
    print("\n" + "-" * 40)
    print("Compression ratio tests:")

    test_texts = [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning and artificial intelligence are transforming technology.",
        "Python is a popular programming language for data science.",
    ]

    for text in test_texts:
        tokens = tokenizer.encode(text)
        ratio = len(text) / len(tokens)
        print(f"  '{text[:50]}...' -> {len(tokens)} tokens ({ratio:.2f} chars/token)")

    # Show token-by-token decoding
    print("\n" + "-" * 40)
    print("Token-by-token decoding test:")

    test_text = "Hello world! This is a test."
    input_ids = tokenizer.encode(test_text)

    for tid in input_ids:
        token = tokenizer.decode([tid])
        raw_token = tokenizer.convert_ids_to_tokens(tid)
        print(f"  Token ID: {tid:5} -> Raw: {raw_token:20} -> Decoded: '{token}'")


def main():
    parser = argparse.ArgumentParser(description="Train English tokenizer for MiniMind")
    # Get the project root directory (minimind folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    parser.add_argument("--data_path", type=str,
                        default=os.path.join(project_root, "dataset", "english_pretrain.jsonl"),
                        help="Path to English JSONL training data")
    parser.add_argument("--tokenizer_dir", type=str,
                        default=os.path.join(project_root, "model_english"),
                        help="Output directory for tokenizer")
    parser.add_argument("--vocab_size", type=int, default=16000,
                        help="Vocabulary size (default: 16000)")
    parser.add_argument("--download", type=str, nargs="?", const="openwebtext",
                        help="Download English data. Options: fineweb, fineweb-edu, openwebtext, wikitext, slimpajama")
    parser.add_argument("--num_samples", type=int, default=100000,
                        help="Number of samples to download (default: 100000)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate existing tokenizer")
    args = parser.parse_args()

    if args.eval_only:
        eval_tokenizer(args.tokenizer_dir)
        return

    if args.download:
        download_english_data(args.data_path, args.download, args.num_samples)

    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        print("Please either:")
        print("  1. Provide an English JSONL file with 'text' field")
        print("  2. Run with --download to get sample data")
        return

    train_tokenizer(args.data_path, args.tokenizer_dir, args.vocab_size)
    eval_tokenizer(args.tokenizer_dir)


if __name__ == "__main__":
    main()
