"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from loguru import logger
from model.model_minimind import MiniMindForCausalLM

# Global logger instance
_logger_initialized = False


def setup_logger(log_dir='../logs', log_name=None, level='INFO'):
    """
    Setup loguru logger with file and console output.

    Args:
        log_dir: Directory to store log files
        log_name: Custom log filename (without extension). If None, uses timestamp.
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    global _logger_initialized

    if _logger_initialized:
        return logger

    # Only main process should log to file
    if not is_main_process():
        _logger_initialized = True
        return logger

    # Convert relative path to absolute path
    if log_dir.startswith('..') or log_dir.startswith('.'):
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), log_dir))

    # Create log directory if not exists
    os.makedirs(log_dir, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if log_name:
        log_filename = f"{log_name}_{timestamp}.log"
    else:
        log_filename = f"train_{timestamp}.log"

    log_path = os.path.join(log_dir, log_filename)

    # Remove default handler
    logger.remove()

    # Add console handler with color
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
        colorize=True
    )

    # Add file handler
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level=level,
        rotation="100 MB",  # Rotate when file reaches 100MB
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress rotated logs
        encoding="utf-8"
    )

    _logger_initialized = True
    logger.info(f"Logging initialized. Log file: {log_path}")

    return logger


def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content, level='INFO'):
    """
    Log content using loguru. Only main process logs.

    Args:
        content: Message to log
        level: Log level (DEBUG, INFO, WARNING, ERROR, SUCCESS)
    """
    if is_main_process():
        if level == 'DEBUG':
            logger.debug(content)
        elif level == 'WARNING':
            logger.warning(content)
        elif level == 'ERROR':
            logger.error(content)
        elif level == 'SUCCESS':
            logger.success(content)
        else:
            logger.info(content)


def get_lr(current_step, total_steps, lr, warmup_ratio=0.01, min_lr_ratio=0.1):
    """
    Cosine decay with linear warmup (standard for LLM pretraining).

    Args:
        current_step: Current training step
        total_steps: Total training steps
        lr: Base learning rate (max LR after warmup)
        warmup_ratio: Fraction of total steps for warmup (default 1%)
        min_lr_ratio: Minimum LR as fraction of base LR (default 10%)

    Returns:
        Learning rate for current step

    Schedule:
        - Warmup (0 to warmup_steps): Linear 0 → lr
        - Decay (warmup_steps to total_steps): Cosine lr → min_lr
    """
    warmup_steps = int(total_steps * warmup_ratio)
    min_lr = lr * min_lr_ratio

    if current_step < warmup_steps:
        # Linear warmup: 0 → lr
        return lr * current_step / warmup_steps
    else:
        # Cosine decay: lr → min_lr
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def format_time(seconds):
    """Format seconds into human readable string (e.g., '1h 23m 45s' or '5m 30s')"""
    if seconds < 0:
        return "0s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def compute_eta(start_time, current_step, total_steps, current_epoch, total_epochs):
    """
    Compute elapsed time and ETA for training.

    Args:
        start_time: Training start time (from time.time())
        current_step: Current step in current epoch
        total_steps: Total steps in current epoch
        current_epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs

    Returns:
        dict with 'elapsed', 'epoch_eta', 'total_eta' as formatted strings
    """
    import time
    elapsed = time.time() - start_time

    if current_step == 0:
        return {
            'elapsed': format_time(elapsed),
            'epoch_eta': 'calculating...',
            'total_eta': 'calculating...',
            'speed': 0.0
        }

    # Time per step
    time_per_step = elapsed / current_step

    # Steps remaining in current epoch
    steps_remaining_epoch = total_steps - current_step
    epoch_eta = steps_remaining_epoch * time_per_step

    # Total remaining: current epoch remaining + full remaining epochs
    remaining_epochs = total_epochs - current_epoch - 1
    total_eta = epoch_eta + (remaining_epochs * total_steps * time_per_step)

    # Speed (steps per second)
    speed = current_step / elapsed if elapsed > 0 else 0

    return {
        'elapsed': format_time(elapsed),
        'epoch_eta': format_time(epoch_eta),
        'total_eta': format_time(total_eta),
        'speed': speed
    }


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        from torch.nn.parallel import DistributedDataParallel
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def load_checkpoint(checkpoint_path, device='cpu'):
    """
    Load checkpoint from an explicit file path.

    Args:
        checkpoint_path: Path to the checkpoint file (.pth)
        device: Device to load the checkpoint to

    Returns:
        dict containing checkpoint data (model, optimizer, epoch, step, etc.)
        or None if file doesn't exist
    """
    if not os.path.exists(checkpoint_path):
        Logger(f'Checkpoint file not found: {checkpoint_path}', level='ERROR')
        return None

    Logger(f'Loading checkpoint from: {checkpoint_path}')
    ckp_data = torch.load(checkpoint_path, map_location=device)

    # Handle both formats: full checkpoint (dict with 'model' key) or weights-only
    if isinstance(ckp_data, dict) and 'model' in ckp_data:
        # New format: full checkpoint
        pass
    else:
        # Old format: weights-only, convert to new format
        Logger(f'Converting old checkpoint format (weights-only) to new format')
        ckp_data = {
            'model': ckp_data,
            'optimizer': None,
            'scaler': None,
            'epoch': 0,
            'step': 0,
            'world_size': 1
        }

    # Handle world_size changes for DDP
    saved_ws = ckp_data.get('world_size', 1)
    current_ws = dist.get_world_size() if dist.is_initialized() else 1
    if saved_ws != current_ws and ckp_data.get('step', 0) > 0:
        ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
        Logger(f'GPU count changed ({saved_ws}→{current_ws}), step adjusted to {ckp_data["step"]}')

    Logger(f'Checkpoint loaded: epoch={ckp_data.get("epoch", 0)}, step={ckp_data.get("step", 0)}', level='SUCCESS')
    return ckp_data


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    # Convert relative path to absolute path for local tokenizer loading
    if tokenizer_path.startswith('..') or tokenizer_path.startswith('.'):
        tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), tokenizer_path))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)