# Hindi Model Evaluation Script for MiniMind
# This script evaluates a trained Hindi MiniMind model with various test prompts
# यह स्क्रिप्ट प्रशिक्षित हिंदी MiniMind मॉडल का मूल्यांकन करती है
import os
import sys
import time
import argparse
import random
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora
from trainer.trainer_utils import setup_seed, get_model_params

warnings.filterwarnings('ignore')


def init_model(args):
    """Initialize the Hindi MiniMind model."""
    # Use HindiTokenizer wrapper for consistent normalization
    try:
        from model.HindiTokenizer import HindiTokenizer
        tokenizer = HindiTokenizer.from_pretrained(args.load_from)
        print("Using HindiTokenizer wrapper (consistent Indic normalization)")
    except ImportError:
        tokenizer = AutoTokenizer.from_pretrained(args.load_from)

    if 'model' in args.load_from or 'hindi' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            vocab_size=args.vocab_size,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'

        if not os.path.exists(ckp):
            print(f"Warning: Model checkpoint not found: {ckp}")
            print("Using randomly initialized model for testing...")

        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=False)

        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)

    get_model_params(model, model.config)
    return model.eval().to(args.device), tokenizer


def main():
    parser = argparse.ArgumentParser(description="MiniMind Hindi Model Evaluation")
    parser.add_argument('--load_from', default='model_hindi', type=str, help="Model loading path")
    parser.add_argument('--save_dir', default='out_hindi', type=str, help="Model weights directory")
    parser.add_argument('--weight', default='full_sft_hindi', type=str, help="Weight name prefix")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA weight name")
    parser.add_argument('--hidden_size', default=768, type=int, help="Hidden dimension")
    parser.add_argument('--num_hidden_layers', default=16, type=int, help="Number of layers")
    parser.add_argument('--vocab_size', default=10000, type=int, help="Vocabulary size")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="Use MoE architecture")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="Enable RoPE scaling")
    parser.add_argument('--max_new_tokens', default=2048, type=int, help="Max generation length")
    parser.add_argument('--temperature', default=0.85, type=float, help="Generation temperature")
    parser.add_argument('--top_p', default=0.85, type=float, help="Nucleus sampling threshold")
    parser.add_argument('--historys', default=0, type=int, help="Number of conversation history turns")
    parser.add_argument('--show_speed', default=1, type=int, help="Show decode speed (tokens/s)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="Running device")

    args = parser.parse_args()

    # Hindi test prompts
    prompts_hindi = [
        'आपका नाम क्या है?',
        'भारत की राजधानी क्या है?',
        'भारत की जनसंख्या कितनी है?',
        'महात्मा गांधी कौन थे?',
        'फोटोसिंथेसिस क्या है?',
        'पाइथागोरस प्रमेय क्या है?',
        'Python में एक फाइबोनैकी फंक्शन लिखें',
        'भारत में कित-कित राज्य हैं?',
        'योग क्या है और इसके क्या फायदे हैं?',
        'ताज महल किसन बनवाया था?',
        'Hinglish में बात करो: मुझे Python सिखाओ',
        'बताओ कि climate change का क्या प्रभाव है?',
    ]

    # Mixed Hindi-English (Hinglish) test prompts
    prompts_hinglish = [
        'मुझे Python सीखना है, क्या आप मदद कर सकते हैं?',
        'What is the capital of India? भारत की राजधानी क्या है?',
        'मैं एक new laptop खरीदना चाहता हूं, कौन सी brand सबसे अच्छी है?',
        'Explain machine learning in simple Hindi words.',
    ]

    # English test prompts (for comparison)
    prompts_english = [
        'What is your name?',
        'What is the capital of India?',
        'Explain photosynthesis in simple terms.',
    ]

    print("=" * 60)
    print("MiniMind Hindi Model Evaluation")
    print("=" * 60)

    conversation = []
    model, tokenizer = init_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    input_mode = input('[0] Hindi Auto Test\n[1] Hinglish Auto Test\n[2] Manual Chat\n')

    if input_mode == '0':
        test_prompts = prompts_hindi
    elif input_mode == '1':
        test_prompts = prompts_hinglish
    else:
        test_prompts = iter(lambda: input('💬 (Hindi/English/Hinglish): '), '')

    for prompt in test_prompts:
        setup_seed(2026)

        if isinstance(test_prompts, list):
            print(f'\n💬: {prompt}')

        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        # Apply chat template
        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        inputs_text = tokenizer.apply_chat_template(**templates)

        inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(args.device)

        print('🤖: ', end='')
        st = time.time()

        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=1.0
        )

        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})

        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        if args.show_speed:
            print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s')
        print('\n')


if __name__ == "__main__":
    main()
