# Hindi Pretraining Script for MiniMind
# This script trains a MiniMind model from scratch on Hindi+English bilingual corpus
# यह स्क्रिप्ट हिंदी और अंग्रेजी द्विभाषी कॉर्पस पर MiniMind मॉडल को शून्य से प्रशिक्षित करती है
#
# Supports:
# - HuggingFace datasets (ai4bharat/sangraha)
# - Local JSONL files
# - Streaming for large datasets

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
from dataset.lm_dataset import PretrainDataset, HuggingFacePretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler

# Import configuration
from config.hindi_config import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    print_config
)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None, is_streaming=False):
    """Training loop for one epoch."""
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()

    step = start_step
    for batch in loader:
        step += 1
        X, Y, loss_mask = batch
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # Learning rate scheduling with cosine annealing
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            # Apply loss mask (ignore padding tokens)
            logits_loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss = logits_loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # Logging
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_logits_loss = logits_loss.item()
            current_aux_loss = res.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']

            if is_streaming:
                Logger(f'[Hindi Pretrain] Epoch:[{epoch + 1}/{args.epochs}] Step:{step}, '
                       f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, '
                       f'aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}')
            else:
                eta_min = spend_time / step * iters // 60 - spend_time // 60
                Logger(f'[Hindi Pretrain] Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                       f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, '
                       f'aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, '
                       f'eta: {eta_min:.1f}min')

            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "step": step
                })

        # Checkpoint saving
        if step % args.save_interval == 0 and is_main_process():
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
            Logger(f"Checkpoint saved at step {step}")

        # Check max steps for streaming
        if is_streaming and args.max_steps and step >= args.max_steps:
            Logger(f"Reached max_steps: {args.max_steps}")
            break

        del X, Y, loss_mask, res, loss

    return step


if __name__ == "__main__":
    # Load defaults from config
    ds_cfg = DEFAULT_DATASET_CONFIG
    model_cfg = DEFAULT_MODEL_CONFIG
    train_cfg = DEFAULT_TRAINING_CONFIG

    parser = argparse.ArgumentParser(
        description="MiniMind Hindi Pretraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on HuggingFace dataset (default: ai4bharat/sangraha)
  python train_pretrain_hindi.py --source ai4bharat/sangraha --subset synthetic

  # Train on local JSONL file
  python train_pretrain_hindi.py --source ../dataset/hindi/corpus_pretrain.jsonl

  # Train with limited samples for testing
  python train_pretrain_hindi.py --max_samples 10000 --max_steps 1000

  # Show current configuration
  python train_pretrain_hindi.py --show_config
        """
    )

    # Data source arguments
    parser.add_argument("--source", type=str, default=ds_cfg.hf_dataset_name,
                        help="HuggingFace dataset name OR local JSONL file path")
    parser.add_argument("--subset", type=str, default=ds_cfg.hf_dataset_subset,
                        help="HuggingFace dataset subset (e.g., synthetic)")
    parser.add_argument("--split", type=str, default=ds_cfg.hf_dataset_split,
                        help="Dataset split (default: hin_Deva for Hindi)")
    parser.add_argument("--text_column", type=str, default=ds_cfg.hf_text_column,
                        help="Column containing text data")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples per epoch (for streaming datasets)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum training steps per epoch (for streaming)")
    parser.add_argument("--no_streaming", action="store_true",
                        help="Disable streaming (downloads ALL language splits - not recommended for sangraha)")
    parser.add_argument("--min_length", type=int, default=ds_cfg.min_text_length,
                        help="Minimum text length to include")

    # Model arguments
    parser.add_argument("--save_dir", type=str, default=model_cfg.save_dir,
                        help="Model save directory")
    parser.add_argument("--checkpoint_dir", type=str, default=model_cfg.checkpoint_dir,
                        help="Checkpoint directory")
    parser.add_argument('--save_weight', default='pretrain_hindi', type=str,
                        help="Weight name prefix")
    parser.add_argument('--hidden_size', default=model_cfg.hidden_size, type=int,
                        help="Hidden dimension")
    parser.add_argument('--num_hidden_layers', default=model_cfg.num_hidden_layers, type=int,
                        help="Number of layers")
    parser.add_argument('--vocab_size', default=model_cfg.vocab_size, type=int,
                        help="Vocabulary size")
    parser.add_argument('--max_seq_len', default=model_cfg.max_seq_len, type=int,
                        help="Training sequence length")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1],
                        help="Use MoE architecture")
    parser.add_argument('--tokenizer_path', default='./model_hindi', type=str,
                        help="Hindi tokenizer path")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=train_cfg.pretrain_epochs,
                        help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=train_cfg.pretrain_batch_size,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=train_cfg.pretrain_learning_rate,
                        help="Initial learning rate")
    parser.add_argument("--accumulation_steps", type=int, default=train_cfg.pretrain_accumulation_steps,
                        help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=train_cfg.grad_clip,
                        help="Gradient clipping threshold")
    parser.add_argument("--dtype", type=str, default=train_cfg.dtype,
                        help="Mixed precision type (bfloat16 or float16)")
    parser.add_argument("--num_workers", type=int, default=train_cfg.num_workers,
                        help="Data loading threads")

    # Logging arguments
    parser.add_argument("--log_interval", type=int, default=train_cfg.log_interval,
                        help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=train_cfg.save_interval,
                        help="Model save interval")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use wandb logging")
    parser.add_argument("--wandb_project", type=str, default=train_cfg.wandb_project,
                        help="WandB project name")

    # Other arguments
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Training device")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1],
                        help="Auto-detect & resume training")
    parser.add_argument("--show_config", action="store_true",
                        help="Show current configuration and exit")

    args = parser.parse_args()

    if args.show_config:
        print_config()
        sys.exit(0)

    # ========== 1. Initialize environment and seed ==========
    Logger("=" * 60)
    Logger("MiniMind Hindi Pretraining")
    Logger("=" * 60)
    Logger(f"Source: {args.source}")
    Logger(f"Subset: {args.subset}")
    Logger(f"Vocab Size: {args.vocab_size}")
    Logger(f"Hidden Size: {args.hidden_size}")
    Logger(f"Layers: {args.num_hidden_layers}")
    Logger(f"Max Samples: {args.max_samples or 'All'}")
    Logger(f"Max Steps: {args.max_steps or 'All'}")
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
        wandb_run_name = f"MiniMind-Hindi-Pretrain-{args.hidden_size}-Epoch{args.epochs}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. Initialize model, data, optimizer ==========
    Logger("Initializing model...")

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

    Logger(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    Logger(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    # Determine if using HuggingFace dataset or local file
    is_local_file = os.path.exists(args.source)
    is_streaming = (not args.no_streaming) and (not is_local_file)  # Default: True (streaming for sangraha)

    if is_local_file:
        Logger(f"Loading from local file: {args.source}")
        train_ds = PretrainDataset(args.source, tokenizer, max_length=args.max_seq_len)
        Logger(f"Dataset size: {len(train_ds)}")
    else:
        Logger(f"Loading from HuggingFace: {args.source} (subset: {args.subset})")
        train_ds = HuggingFacePretrainDataset(
            dataset_name=args.source,
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
            subset=args.subset,
            split=args.split,
            text_column=args.text_column,
            max_samples=args.max_samples,
            min_length=args.min_length,
            streaming=is_streaming
        )
        if is_streaming:
            Logger("Using streaming mode (dataset size unknown)")
        else:
            Logger(f"Dataset size: {len(train_ds)}")

    # Setup data loader
    if is_streaming:
        # Streaming datasets don't support DistributedSampler
        train_sampler = None
        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            num_workers=0,  # Streaming doesn't work well with multiple workers
            pin_memory=True
        )
        iters = args.max_steps or 100000  # Estimate for streaming
    else:
        train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        iters = len(loader)

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
    Logger("Starting training...")
    model.train()

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        if not is_streaming and epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: Skipping first {start_step} steps')
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb, is_streaming)
        else:
            train_epoch(epoch, loader, iters, 0, wandb, is_streaming)

    # ========== 9. Final save and cleanup ==========
    if is_main_process():
        model.eval()
        moe_suffix = '_moe' if lm_config.use_moe else ''
        ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        torch.save(state_dict, ckp)
        Logger(f"Final model saved to: {ckp}")

    if dist.is_initialized():
        dist.destroy_process_group()

    Logger("=" * 60)
    Logger("Hindi Pretraining Completed!")
    Logger(f"Model saved to: {args.save_dir}")
    Logger("=" * 60)
