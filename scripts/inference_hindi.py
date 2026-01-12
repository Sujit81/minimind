# Hindi MiniMind Inference Script
# Run in Colab notebook or locally
# Usage: python scripts/inference_hindi.py --prompt "आपका नाम क्या है?"

import os
import sys
import argparse
import torch

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def load_model(
    model_path: str = "./out_hindi/full_sft_hindi_512.pth",
    tokenizer_path: str = "./model_hindi",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    hidden_size: int = 512,
    num_hidden_layers: int = 8,
    vocab_size: int = 12000
):
    """Load the Hindi MiniMind model and tokenizer."""

    # Load tokenizer
    try:
        from model.HindiTokenizer import HindiTokenizer
        tokenizer = HindiTokenizer.from_pretrained(tokenizer_path)
        print(f"Loaded HindiTokenizer from {tokenizer_path}")
    except ImportError:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"Loaded AutoTokenizer from {tokenizer_path}")

    # Create model config
    config = MiniMindConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        use_moe=False
    )

    # Load model
    model = MiniMindForCausalLM(config)

    if os.path.exists(model_path):
        weights = torch.load(model_path, map_location=device)
        model.load_state_dict(weights, strict=False)
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}")
        print("Available files in out_hindi/:")
        if os.path.exists("./out_hindi"):
            for f in os.listdir("./out_hindi"):
                print(f"  - {f}")

    model = model.to(device).eval()
    print(f"Model loaded on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return model, tokenizer, device


def generate(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    repetition_penalty: float = 1.1
):
    """Generate text from a prompt."""

    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Generate
    with torch.no_grad():
        generated_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]  # Get last token logits

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample or greedy
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode output
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return output_text


def chat(
    model,
    tokenizer,
    user_message: str,
    device: str = "cuda",
    system_prompt: str = "आप एक सहायक AI हैं। कृपया हिंदी में उत्तर दें।",
    **generate_kwargs
):
    """Chat with the model using chat template."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate response
    full_response = generate(model, tokenizer, prompt, device, **generate_kwargs)

    # Extract assistant response (after the prompt)
    if "<|im_start|>assistant" in full_response:
        response = full_response.split("<|im_start|>assistant")[-1]
        response = response.replace("<|im_end|>", "").strip()
    else:
        # Fallback: return everything after the user message
        response = full_response[len(prompt):].strip()

    return response


def interactive_chat(model, tokenizer, device):
    """Interactive chat loop."""

    print("\n" + "="*60)
    print("Hindi MiniMind Interactive Chat")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to start a new conversation")
    print("="*60 + "\n")

    system_prompt = "आप एक सहायक AI हैं। कृपया हिंदी में उत्तर दें।"

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! अलविदा!")
                break

            if user_input.lower() == 'clear':
                print("\n--- Conversation cleared ---\n")
                continue

            # Generate response
            response = chat(
                model, tokenizer, user_input, device,
                system_prompt=system_prompt,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9
            )

            print(f"Assistant: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye! अलविदा!")
            break


# ============== Notebook-friendly functions ==============

def setup_hindi_model(
    model_path: str = "./out_hindi/full_sft_hindi_512.pth",
    tokenizer_path: str = "./model_hindi"
):
    """
    One-line setup for notebook usage.

    Example:
        model, tokenizer, device = setup_hindi_model()
        response = ask("भारत की राजधानी क्या है?", model, tokenizer, device)
    """
    return load_model(model_path, tokenizer_path)


def ask(question: str, model, tokenizer, device, **kwargs):
    """
    Simple function to ask a question.

    Example:
        response = ask("भारत की राजधानी क्या है?", model, tokenizer, device)
        print(response)
    """
    return chat(model, tokenizer, question, device, **kwargs)


# ============== Main ==============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hindi MiniMind Inference")
    parser.add_argument("--model_path", type=str, default="./out_hindi/full_sft_hindi_512.pth",
                        help="Path to model weights")
    parser.add_argument("--tokenizer_path", type=str, default="./model_hindi",
                        help="Path to tokenizer")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to generate (non-interactive)")
    parser.add_argument("--interactive", action="store_true",
                        help="Start interactive chat mode")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=12000)

    args = parser.parse_args()

    # Load model
    model, tokenizer, device = load_model(
        args.model_path,
        args.tokenizer_path,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        vocab_size=args.vocab_size
    )

    if args.prompt:
        # Single prompt mode
        response = chat(
            model, tokenizer, args.prompt, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print(f"\nQuestion: {args.prompt}")
        print(f"Answer: {response}")

    elif args.interactive:
        # Interactive mode
        interactive_chat(model, tokenizer, device)

    else:
        # Default: run a few example prompts
        print("\n" + "="*60)
        print("Hindi MiniMind - Example Outputs")
        print("="*60)

        examples = [
            "भारत की राजधानी क्या है?",
            "आपका नाम क्या है?",
            "मुझे एक छोटी कहानी सुनाओ।",
        ]

        for prompt in examples:
            print(f"\nQuestion: {prompt}")
            response = chat(
                model, tokenizer, prompt, device,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9
            )
            print(f"Answer: {response}")
            print("-" * 40)
