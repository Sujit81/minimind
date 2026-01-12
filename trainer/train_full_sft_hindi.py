# Hindi SFT (Supervised Fine-Tuning) Script for MiniMind
# This script fine-tunes a pre-trained Hindi MiniMind model on Hindi conversations
# यह स्क्रिप्ट पूर्व-प्रशिक्षित हिंदी MiniMind मॉडल को हिंदी वार्तालाप पर fine-tune करती है
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, get_lr_with_warmup, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, warmup_steps=0, min_lr_ratio=0.1):
    """Training loop for one epoch."""
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    total_steps = args.epochs * iters

    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # Learning rate scheduling with warmup
        global_step = epoch * iters + step
        lr = get_lr_with_warmup(global_step, total_steps, args.learning_rate, warmup_steps, min_lr_ratio)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            # Apply loss mask (only train on assistant responses)
            logits_loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss = logits_loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # Logging
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_logits_loss = logits_loss.item()
            current_aux_loss = res.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(f'[Hindi SFT] Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, '
                   f'aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, '
                   f'epoch_time: {eta_min:.3f}min')

            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # Checkpoint saving
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)

            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir=args.checkpoint_dir)
            model.train()
            del state_dict

        del X, Y, loss_mask, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Hindi SFT")
    parser.add_argument("--save_dir", type=str, default="./out_hindi", help="Model save directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_hindi", help="Checkpoint directory")
    parser.add_argument('--save_weight', default='full_sft_hindi', type=str, help="Weight name prefix")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Mixed precision type")
    parser.add_argument("--num_workers", type=int, default=8, help="Data loading threads")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--warmup_steps", type=int, default=0, help="LR warmup steps (0=auto: 2%% of total)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Min LR ratio for decay")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=100, help="Model save interval")
    parser.add_argument('--hidden_size', default=512, type=int, help="Hidden dimension (Base model)")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Number of layers (Base model)")
    parser.add_argument('--vocab_size', default=12000, type=int, help="Vocabulary size")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="Training sequence length")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="Use MoE architecture")
    parser.add_argument("--data_path", type=str, default="./dataset/hindi/sft_hindi.jsonl", help="SFT data path (local JSONL or HuggingFace dataset)")
    parser.add_argument("--hf_dataset", type=str, default=None, help="HuggingFace dataset name (e.g., BhabhaAI/indic-instruct-data-v0.2-filtered)")
    parser.add_argument("--hf_subset", type=str, default=None, help="HuggingFace dataset config/subset (e.g., anudesh)")
    parser.add_argument("--hf_split", type=str, default="hi", help="HuggingFace dataset split/language (e.g., hi for Hindi)")
    parser.add_argument('--from_weight', default='pretrain_hindi', type=str, help="Base checkpoint to fine-tune")
    parser.add_argument('--tokenizer_path', default='./model_hindi', type=str, help="Hindi tokenizer path")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="Auto-detect & resume training")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Hindi-SFT", help="WandB project name")

    args = parser.parse_args()

    # Auto-fix num_workers on Windows (multiprocessing issues)
    if sys.platform == 'win32' and args.num_workers > 0:
        print(f"Windows detected: Setting num_workers=0 (was {args.num_workers})")
        args.num_workers = 0

    # ========== 1. Initialize environment and seed ==========
    Logger("=" * 60)
    Logger("MiniMind Hindi Supervised Fine-Tuning")
    Logger("=" * 60)
    Logger(f"Base checkpoint: {args.from_weight}")
    Logger(f"Vocab Size: {args.vocab_size}")
    Logger(f"Hidden Size: {args.hidden_size}")
    Logger(f"Layers: {args.num_hidden_layers}")
    Logger(f"Data: {args.hf_dataset or args.data_path}")
    Logger("=" * 60)

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. Configure directories and model ==========
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        vocab_size=args.vocab_size,
        use_moe=bool(args.use_moe)
    )

    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.checkpoint_dir) if args.from_resume == 1 else None

    # ========== 3. Setup mixed precision ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. Setup wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Hindi-SFT-Epoch{args.epochs}-BS{args.batch_size}-LR{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. Initialize model, data, optimizer ==========
    Logger("Loading model and tokenizer...")

    # Use HindiTokenizer wrapper for consistent normalization (training = inference)
    try:
        from model.HindiTokenizer import HindiTokenizer
        tokenizer = HindiTokenizer.from_pretrained(args.tokenizer_path)
        Logger("Using HindiTokenizer wrapper (consistent Indic normalization)")
    except ImportError:
        Logger("Warning: HindiTokenizer not available, using AutoTokenizer (no Indic normalization)")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = MiniMindForCausalLM(lm_config).to(args.device)

    if args.from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{args.save_dir}/{args.from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        if os.path.exists(weight_path):
            weights = torch.load(weight_path, map_location=args.device)
            model.load_state_dict(weights, strict=False)
            Logger(f"Loaded weights from {weight_path}")
        else:
            Logger(f"Warning: Weight file not found: {weight_path}")
            Logger("Training from scratch (not recommended for SFT)!")

    Logger(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    Logger(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    # ========== Load SFT Dataset ==========
    if args.hf_dataset:
        # Load from HuggingFace dataset
        from dataset.lm_dataset import HuggingFaceSFTDataset
        Logger(f"Loading from HuggingFace: {args.hf_dataset} (subset: {args.hf_subset}, split: {args.hf_split})")
        train_ds = HuggingFaceSFTDataset(
            dataset_name=args.hf_dataset,
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
            split=args.hf_split,
            subset=args.hf_subset
        )
    else:
        # Load from local JSONL file
        Logger(f"Loading from local file: {args.data_path}")
        train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    Logger(f"Dataset size: {len(train_ds)}")

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. Resume from checkpoint if available ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
        Logger(f"Resumed from epoch {start_epoch}, step {start_step}")

    # ========== 7. Wrap with DDP if distributed ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. Start training ==========
    # Calculate warmup steps (auto: 2% of total steps for SFT)
    steps_per_epoch = len(train_ds) // args.batch_size
    total_training_steps = args.epochs * steps_per_epoch
    if args.warmup_steps == 0:
        warmup_steps = int(total_training_steps * 0.02)  # 2% warmup for SFT
    else:
        warmup_steps = args.warmup_steps

    Logger("Starting SFT training...")
    Logger(f"Total steps: {total_training_steps}, Warmup: {warmup_steps}, Min LR ratio: {args.min_lr_ratio}")
    model.train()

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)

        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: Skipping first {start_step} steps')
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb, warmup_steps, args.min_lr_ratio)
        else:
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                             sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), 0, wandb, warmup_steps, args.min_lr_ratio)

    # ========== 9. Cleanup ==========
    if dist.is_initialized():
        dist.destroy_process_group()

    Logger("=" * 60)
    Logger("Hindi SFT Training Completed!")
    Logger(f"Model saved to: {args.save_dir}")
    Logger("=" * 60)
