import os
import time
import argparse
import warnings
import json
import torch
from transformers import AutoTokenizer, TextStreamer, PreTrainedTokenizerFast
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params

warnings.filterwarnings('ignore')

# Get script directory for relative path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path):
    """Convert relative path to absolute path based on script location."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(SCRIPT_DIR, path))


def infer_arch_from_state_dict(state_dict):
    embed_key = 'model.embed_tokens.weight'
    if embed_key not in state_dict:
        raise KeyError(f"Checkpoint missing key: {embed_key}")

    vocab_size, hidden_size = state_dict[embed_key].shape

    layer_indices = []
    layer_prefix = 'model.layers.'
    for key in state_dict.keys():
        if key.startswith(layer_prefix):
            parts = key.split('.')
            if len(parts) > 2 and parts[2].isdigit():
                layer_indices.append(int(parts[2]))
    num_hidden_layers = max(layer_indices) + 1 if layer_indices else 8

    return int(hidden_size), int(num_hidden_layers), int(vocab_size)


def init_model(args):
    tokenizer_path = resolve_path(args.tokenizer_path)
    tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
    if os.path.exists(tokenizer_file):
        config_file = os.path.join(tokenizer_path, "tokenizer_config.json")
        cfg = {}
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                cfg = json.load(f)

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            bos_token=cfg.get('bos_token', '<bos>'),
            eos_token=cfg.get('eos_token', '<eos>'),
            unk_token=cfg.get('unk_token', '<unk>'),
            pad_token=cfg.get('pad_token', '<pad>'),
        )
        print(f"[Tokenizer] Loaded PreTrainedTokenizerFast from {tokenizer_file}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # Use --checkpoint if provided, otherwise construct from save_dir/weight/hidden_size
    if args.checkpoint:
        ckp = resolve_path(args.checkpoint)
    else:
        ckp = f'{resolve_path(args.save_dir)}/{args.weight}_{args.hidden_size}.pth'

    print(f"Loading checkpoint: {ckp}")
    checkpoint = torch.load(ckp, map_location=args.device)

    # Handle both formats: full checkpoint (dict with 'model' key) or weights-only
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
        print(f"Full checkpoint loaded (epoch={checkpoint.get('epoch', 0)}, step={checkpoint.get('step', 0)}, eval_loss={checkpoint.get('eval_loss', 'N/A')})")
    else:
        state_dict = checkpoint

    if args.auto_config == 1:
        hidden_size, num_hidden_layers, vocab_size = infer_arch_from_state_dict(state_dict)
    else:
        hidden_size, num_hidden_layers, vocab_size = args.hidden_size, args.num_hidden_layers, args.vocab_size

    model = MiniMindForCausalLM(MiniMindConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        use_moe=False,
        vocab_size=vocab_size,
    ))
    print(f"[Config] hidden_size={hidden_size}, layers={num_hidden_layers}, vocab_size={vocab_size}, use_moe=0")

    moe_prefixes = ('.mlp.experts.', '.mlp.shared_experts.', '.mlp.gate.')
    moe_keys = [k for k in state_dict.keys() if any(prefix in k for prefix in moe_prefixes)]
    if moe_keys and args.allow_moe_checkpoint == 0:
        raise ValueError(
            "Checkpoint appears to be MoE, but eval_pretrain_std.py is standard (non-MoE). "
            "Use eval_pretrain.py --use_moe 1 (or --use_moe -1 with auto detection), "
            "or rerun this script with --allow_moe_checkpoint 1 to ignore MoE weights."
        )

    # Strict load by default; set --strict_load 0 only for controlled partial load experiments.
    model.load_state_dict(state_dict, strict=bool(args.strict_load))

    # Warn about MoE keys being ignored
    if moe_keys:
        print(f"⚠️  Warning: Checkpoint contains {len(moe_keys)} MoE-specific keys that will be ignored")
        print(f"   This is expected when loading MoE checkpoint into standard model")

    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer


def sanitize_inputs(input_ids, attention_mask, vocab_size):
    """Replace out-of-range token ids to avoid embedding index errors."""
    invalid = (input_ids < 0) | (input_ids >= vocab_size)
    invalid_count = int(invalid.sum().item())
    if invalid_count > 0:
        print(f"[Tokenizer] Found {invalid_count} out-of-range input token ids; replacing with 0")
        input_ids = input_ids.clone()
        input_ids[invalid] = 0
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def valid_special_token_id(token_id, vocab_size):
    if token_id is None:
        return None
    if 0 <= int(token_id) < vocab_size:
        return int(token_id)
    return None


def normalize_prompt(prompt):
    """Ensure continuation starts at a word boundary."""
    if prompt and not prompt[-1].isspace():
        return prompt + ' '
    return prompt


def main():
    parser = argparse.ArgumentParser(description="MiniMind Pretrained Model - Text Continuation (Standard/Non-MoE)")
    parser.add_argument('--checkpoint', default='', type=str, help="Direct path to checkpoint file (e.g., out/pretrain_640_best.pth)")
    parser.add_argument('--tokenizer_path', default='model_english', type=str, help="Tokenizer path")
    parser.add_argument('--save_dir', default='out', type=str, help="Model weights directory (used if --checkpoint not specified)")
    parser.add_argument('--weight', default='pretrain', type=str, help="Weight name prefix (used if --checkpoint not specified)")
    parser.add_argument('--hidden_size', default=512, type=int, help="Hidden dimension")
    parser.add_argument('--num_hidden_layers', default=14, type=int, help="Number of layers")
    parser.add_argument('--vocab_size', default=32000, type=int, help="Vocabulary size")
    parser.add_argument('--auto_config', default=1, type=int, choices=[0, 1], help="Auto infer hidden_size/layers/vocab from checkpoint")
    parser.add_argument('--strict_load', default=1, type=int, choices=[0, 1], help="Strict checkpoint loading (recommended: 1)")
    parser.add_argument('--allow_moe_checkpoint', default=0, type=int, choices=[0, 1], help="Allow loading MoE checkpoint into dense model (ignores MoE weights)")
    parser.add_argument('--max_new_tokens', default=200, type=int, help="Maximum tokens to generate")
    parser.add_argument('--prompt_max_len', default=1024, type=int, help="Max prompt token length for truncation")
    parser.add_argument('--temperature', default=0.8, type=float, help="Sampling temperature (0.1-1.5)")
    parser.add_argument('--top_p', default=0.9, type=float, help="Nucleus sampling threshold")
    parser.add_argument('--top_k', default=50, type=int, help="Top-k sampling (0 to disable)")
    parser.add_argument('--repetition_penalty', default=1.2, type=float, help="Repetition penalty (1.0 = no penalty)")
    parser.add_argument('--show_speed', default=1, type=int, help="Show generation speed")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--seed', default=42, type=int, help="Random seed (-1 for random)")
    args = parser.parse_args()

    # Sample prompts for text continuation
    prompts = [
        "India is located in",
        "The history of artificial intelligence began",
        "In year 2050, humanity",
        "The scientific method involves",
        "Once upon a time, there was a",
        "The main difference between Python and Java is",
        "Climate change is caused by",
        "The human brain contains approximately",
        "Water is found in",
        "The capital of France is",
        "Machine learning is a type of",
        "The sun rises in the",
    ]

    model, tokenizer = init_model(args)

    print("\n" + "="*60)
    print("PRETRAINED MODEL - TEXT CONTINUATION (Standard/Non-MoE)")
    print("="*60)
    print(f"Model: Standard (non-MoE) | Temperature: {args.temperature} | Top-p: {args.top_p}")
    print(f"Repetition penalty: {args.repetition_penalty}")
    print("="*60 + "\n")

    input_mode = input("[0] Auto test with sample prompts\n[1] Manual input\nSelect: ")

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    if input_mode == '0':
        prompt_iter = prompts
    else:
        prompt_iter = iter(lambda: input('\nPrompt: '), '')

    for prompt in prompt_iter:
        if args.seed >= 0:
            setup_seed(args.seed)
        else:
            setup_seed(int(time.time()) % 10000)

        if input_mode == '0':
            print(f"\nPrompt: {prompt}")

        prompt = normalize_prompt(prompt)

        # Prepend BOS only when its id is valid for current model vocab
        bos_id = valid_special_token_id(tokenizer.bos_token_id, model.config.vocab_size)
        input_text = (tokenizer.bos_token + prompt) if (tokenizer.bos_token and bos_id is not None) else prompt
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=args.prompt_max_len,
            add_special_tokens=False,
        ).to(args.device)
        input_ids, attention_mask = sanitize_inputs(inputs["input_ids"], inputs.get("attention_mask"), model.config.vocab_size)

        pad_token_id = valid_special_token_id(tokenizer.pad_token_id, model.config.vocab_size)
        eos_token_id = valid_special_token_id(tokenizer.eos_token_id, model.config.vocab_size)

        print("Generated: ", end='')
        st = time.time()

        with torch.no_grad():
            generated_ids = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                streamer=streamer,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                top_p=args.top_p,
                top_k=args.top_k if args.top_k > 0 else None,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
            )

        full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"\nFull decoded: {full_text}")

        gen_tokens = len(generated_ids[0]) - len(input_ids[0])
        elapsed = time.time() - st

        if args.show_speed:
            print(f"\n[Tokens: {gen_tokens} | Speed: {gen_tokens/elapsed:.2f} tokens/s | Time: {elapsed:.2f}s]")
        print("-" * 60)


if __name__ == "__main__":
    main()