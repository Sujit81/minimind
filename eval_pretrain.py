import time
import argparse
import warnings
import torch
from transformers import AutoTokenizer, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params

warnings.filterwarnings('ignore')


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = MiniMindForCausalLM(MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        vocab_size=args.vocab_size,
    ))
    moe_suffix = '_moe' if args.use_moe else ''
    ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
    print(f"Loading checkpoint: {ckp}")
    checkpoint = torch.load(ckp, map_location=args.device)
    # Handle both formats: full checkpoint (dict with 'model' key) or weights-only
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
        print(f"Full checkpoint loaded (epoch={checkpoint.get('epoch', 0)}, step={checkpoint.get('step', 0)}, eval_loss={checkpoint.get('eval_loss', 'N/A')})")
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer


def main():
    parser = argparse.ArgumentParser(description="MiniMind Pretrained Model - Text Continuation")
    parser.add_argument('--tokenizer_path', default='model_english', type=str, help="Tokenizer path")
    parser.add_argument('--save_dir', default='out', type=str, help="Model weights directory")
    parser.add_argument('--weight', default='pretrain', type=str, help="Weight name prefix")
    parser.add_argument('--hidden_size', default=640, type=int, help="Hidden dimension")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Number of layers")
    parser.add_argument('--use_moe', default=1, type=int, choices=[0, 1], help="Use MoE architecture")
    parser.add_argument('--vocab_size', default=16000, type=int, help="Vocabulary size")
    parser.add_argument('--max_new_tokens', default=200, type=int, help="Maximum tokens to generate")
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
        "In the year 2050, humanity",
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
    print("PRETRAINED MODEL - TEXT CONTINUATION")
    print("="*60)
    print(f"Temperature: {args.temperature} | Top-p: {args.top_p} | Top-k: {args.top_k}")
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

        # For pretrained model, just use raw text (with BOS token)
        input_text = tokenizer.bos_token + prompt if tokenizer.bos_token else prompt
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)

        print("Generated: ", end='')
        st = time.time()

        with torch.no_grad():
            generated_ids = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p,
                top_k=args.top_k if args.top_k > 0 else None,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
            )

        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        elapsed = time.time() - st

        if args.show_speed:
            print(f"\n[Tokens: {gen_tokens} | Speed: {gen_tokens/elapsed:.2f} tokens/s | Time: {elapsed:.2f}s]")
        print("-" * 60)


if __name__ == "__main__":
    main()
