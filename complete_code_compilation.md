# MiniMind Complete Code Compilation

## Project Structure

```
MiniMind/
├── model/                      # Core model architecture
│   ├── model_minimind.py      # Main model (459 lines)
│   ├── model_lora.py          # LoRA implementation (53 lines)
│   ├── tokenizer.json         # Tokenizer vocabulary
│   └── tokenizer_config.json  # Tokenizer configuration
├── dataset/                    # Data processing
│   ├── lm_dataset.py          # All dataset classes (201 lines)
│   └── dataset.md             # Dataset documentation
├── trainer/                    # Training scripts
│   ├── trainer_utils.py       # Training utilities (159 lines)
│   ├── train_full_sft.py      # Supervised fine-tuning
│   ├── train_lora.py          # LoRA fine-tuning
│   ├── train_dpo.py           # DPO alignment
│   ├── train_distillation.py   # Knowledge distillation
│   ├── train_reason.py        # Reasoning training
│   ├── train_grpo.py          # GRPO RL
│   ├── train_ppo.py           # PPO RL
│   ├── train_spo.py           # SPO RL
│   └── train_tokenizer.py     # Tokenizer training
├── scripts/                    # Serving & conversion
│   ├── eval_llm.py            # Model evaluation (92 lines)
│   ├── web_demo.py            # Web interface (329 lines)
│   ├── serve_openai_api.py    # API server (178 lines)
│   ├── convert_model.py       # Format conversion (79 lines)
│   └── chat_openai_api.py     # API client
└── requirements.txt           # Dependencies (31 packages)
```

---

## 1. Core Model Architecture (model/model_minimind.py)

### Configuration Class
```python
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = 'silu',
        hidden_size: int = 512,              # Small: 512, Base: 768, MoE: 640
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,  # Context length
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,          # Small/MoE: 8, Base: 16
        num_key_value_heads: int = 2,        # Grouped query attention
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000.0,         # RoPE base
        inference_rope_scaling: bool = False,  # YaRN scaling
        flash_attn: bool = True,
        # MoE specific
        use_moe: bool = False,
        num_experts_per_tok: int = 2,        # Top-k routing
        n_routed_experts: int = 4,           # Total experts
        n_shared_experts: int = 1,           # Always active
        scoring_func: str = 'softmax',
        aux_loss_alpha: float = 0.01,        # Load balancing
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs
    )
```

### RMS Normalization
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scale

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
```

### RoPE Position Encoding
```python
def precompute_freqs_cis(dim: int, end: int = 32768, rope_base: float = 1e6, rope_scaling: Optional[dict] = None):
    """Pre-compute rotary embeddings with YaRN scaling"""
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    if rope_scaling is not None:  # YaRN implementation
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0)
        )
        # Linear ramp for scaling
        inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
        low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
        ramp = torch.clamp((torch.arange(dim // 2) - low) / max(high - low, 0.001), 0, 1)
        freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin
```

### Grouped Query Attention
```python
class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # KV head repetition factor
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        self.flash = hasattr(F, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        if past_key_value is not None:  # KV cache
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        if self.flash and seq_len > 1:
            output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        else:
            # Manual implementation with causal mask
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(output), past_kv
```

### MoE Implementation
```python
class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.reset_parameters()

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        scores = logits.softmax(dim=-1)

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # Auxiliary loss for load balancing
        if self.training and self.alpha > 0.0:
            if self.seq_aux:  # Sequence-level
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx.view(bsz, -1), torch.ones(bsz, seq_len * self.top_k, device=hidden_states.device))
                ce = ce.div_(seq_len * self.top_k / self.n_routed_experts)
                aux_loss = (ce * scores.view(bsz, seq_len, -1).mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:  # Token-level
                mask_ce = F.one_hot(topk_idx.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()

        return topk_idx, topk_weight, aux_loss

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_routed_experts)])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_shared_experts)])

    def forward(self, x):
        identity = x
        bsz, seq_len, _ = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)

        if self.training:
            # Training: compute all selected experts
            x = x.view(-1, x.shape[-1])
            flat_topk_idx = topk_idx.view(-1)
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)

            for i, expert in enumerate(self.experts):
                expert_out = expert(x[flat_topk_idx == i])
                y[flat_topk_idx == i] = expert_out.to(y.dtype)

            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(bsz, seq_len, -1)
        else:
            # Inference: optimized path, one-hot expert selection
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(bsz, seq_len, -1)

        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)

        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """Optimized inference routing"""
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok

        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx: continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_{0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache
```

### Complete Transformer Block
```python
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present
```

### Main Model
```python
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Pre-compute RoPE frequencies
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(hidden_states, position_embeddings, past_key_value, use_cache, attention_mask)
            presents.append(present)

        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss
```

### Causal LM Head
```python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight  # Weight tying

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, **args):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **args)
        logits = self.lm_head(hidden_states[:, -logits_to_keep:, :])
        output = CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        return output
```

---

## 2. LoRA Fine-tuning (model/model_lora.py)

```python
class LoRA(nn.Module):
    """Low-Rank Adaptation"""
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)      # Low rank matrix A
        self.B = nn.Linear(rank, out_features, bias=False)     # Low rank matrix B
        self.A.weight.data.normal_(mean=0.0, std=0.02)         # Gaussian init
        self.B.weight.data.zero_()                             # Zero init

    def forward(self, x):
        return self.B(self.A(x))

def apply_lora(model, rank=8):
    """Inject LoRA into all linear layers"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward
            module.forward = lambda x, layer1=original_forward, layer2=lora: layer1(x) + layer2(x)

def save_lora(model, path):
    """Save only LoRA weights"""
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
```

---

## 3. Data Processing Pipeline (dataset/lm_dataset.py)

### SFT Dataset with Loss Masking
```python
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def generate_loss_mask(self, input_ids):
        """Only train on assistant responses"""
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:  # Found assistant start
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:  # Found end
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1  # Mask assistant tokens
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.tokenizer.apply_chat_template(sample['conversations'], tokenize=False, add_generation_prompt=False)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self.generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)      # Input tokens
        Y = torch.tensor(input_ids[1:], dtype=torch.long)       # Target tokens
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # Shift right
        return X, Y, loss_mask
```

### DPO Dataset
```python
class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_dataset('json', data_files=file_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item['chosen']      # Preferred conversation
        rejected = item['rejected']  # Rejected conversation

        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)

        chosen_encoding = self.tokenizer(chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length')
        rejected_encoding = self.tokenizer(rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length')

        chosen_loss_mask = self.generate_loss_mask(chosen_encoding['input_ids'])
        rejected_loss_mask = self.generate_loss_mask(rejected_encoding['input_ids'])

        return {
            'x_chosen': torch.tensor(chosen_encoding['input_ids'][:-1]),
            'y_chosen': torch.tensor(chosen_encoding['input_ids'][1:]),
            'mask_chosen': torch.tensor(chosen_loss_mask[1:]),
            'x_rejected': torch.tensor(rejected_encoding['input_ids'][:-1]),
            'y_rejected': torch.tensor(rejected_encoding['input_ids'][1:]),
            'mask_rejected': torch.tensor(rejected_loss_mask[1:])
        }
```

### RLAIF Dataset
```python
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt, answer = self.create_chat_prompt(sample['conversations'])
        return {'prompt': prompt, 'answer': answer}
```

---

## 4. Training Utilities (trainer/trainer_utils.py)

```python
def get_model_params(model, config):
    """Display parameter count with MoE routing"""
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', 0)
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')

def get_lr(current_step, total_steps, lr):
    """Cosine LR schedule with warm restart"""
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))

def setup_seed(seed: int):
    """Reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def init_distributed_mode():
    """Initialize DDP"""
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """Unified checkpointing with resume support"""
    os.makedirs(save_dir, exist_ok=True)
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_suffix}_resume.pth'

    if model is not None:  # Save
        from torch.nn.parallel import DistributedDataParallel
        state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        torch.save(state_dict, ckp_path)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb.id if wandb else None
        }
        torch.save(resume_data, resume_path)
        return None
    else:  # Load
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
            return ckp_data
        return None

class SkipBatchSampler(Sampler):
    """Skip batches for resuming training"""
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
```

---

## 5. Full SFT Training Loop (trainer/train_full_sft.py)

```python
import argparse
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from contextlib import nullcontext

# ==================== Configuration ====================
parser = argparse.ArgumentParser(description="MiniMind Full SFT")
parser.add_argument("--save_dir", type=str, default="../out")
parser.add_argument("--save_weight", default='full_sft', type=str)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-6)
parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dtype", type=str, default="bfloat16")
parser.add_argument("--max_seq_len", default=340, type=int)
parser.add_argument("--hidden_size", default=512, type=int)
parser.add_argument("--num_hidden_layers", default=8, type=int)
parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl")
parser.add_argument('--from_weight', default='pretrain', type=str)
parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
args = parser.parse_args()

# ==================== Initialization ====================
local_rank = init_distributed_mode()
if dist.is_initialized(): args.device = f"cuda:{local_rank}"
setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

os.makedirs(args.save_dir, exist_ok=True)
lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

# ==================== Mixed Precision ====================
device_type = "cuda" if "cuda" in args.device else "cpu"
autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16)
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

# ==================== Model & Data ====================
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

if dist.is_initialized():
    model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
    model = DistributedDataParallel(model, device_ids=[local_rank])

optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

# ==================== Resume Training ====================
start_epoch, start_step = 0, 0
if ckp_data:
    model.load_state_dict(ckp_data['model'])
    optimizer.load_state_dict(ckp_data['optimizer'])
    scaler.load_state_dict(ckp_data['scaler'])
    start_epoch = ckp_data['epoch']
    start_step = ckp_data.get('step', 0)

# ==================== Training Loop ====================
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        X, Y, loss_mask = X.to(args.device), Y.to(args.device), loss_mask.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups: param_group['lr'] = lr

        with autocast_ctx:
            res = model(X)
            loss = loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(Y.size())
            logits_loss = (loss * loss_mask).sum() / loss_mask.sum()  # Only compute loss on assistant tokens
            loss = logits_loss + res.aux_loss

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_logits_loss = logits_loss.item()
            current_aux_loss = res.aux_loss.item()
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, ' +
                   f'logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, ' +
                   f'learning_rate: {optimizer.param_groups[-1]["lr"]:.8f}, epoch_time: {eta_min:.3f}min')

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, scaler=scaler, save_dir='../checkpoints')
            model.train()

# ==================== Main Execution ====================
for epoch in range(start_epoch, args.epochs):
    train_sampler and train_sampler.set_epoch(epoch)

    if epoch == start_epoch and start_step > 0:
        batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
        train_epoch(epoch, loader, len(loader) + start_step + 1, start_step)
    else:
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                           sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
        train_epoch(epoch, loader, len(loader))

if dist.is_initialized(): dist.destroy_process_group()
```

---

## 6. DPO Training (trainer/train_dpo.py)

```python
def logits_to_log_probs(logits, labels):
    """Extract log probabilities for loss computation"""
    log_probs = F.log_softmax(logits, dim=2)
    return torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)

def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """Direct Preference Optimization loss"""
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    batch_size = ref_log_probs.shape[0]
    chosen_ref = ref_log_probs[:batch_size // 2]
    reject_ref = ref_log_probs[batch_size // 2:]
    chosen_policy = policy_log_probs[:batch_size // 2]
    reject_policy = policy_log_probs[batch_size // 2:]

    pi_logratios = chosen_policy - reject_policy
    ref_logratios = chosen_ref - reject_ref
    logits = pi_logratios - ref_logratios
    return -F.logsigmoid(beta * logits).mean()

def train_epoch(epoch, loader, iters, ref_model, beta=0.1):
    for step, batch in enumerate(loader):
        x = torch.cat([batch['x_chosen'], batch['x_rejected']], dim=0).to(args.device)
        y = torch.cat([batch['y_chosen'], batch['y_rejected']], dim=0).to(args.device)
        mask = torch.cat([batch['mask_chosen'], batch['mask_rejected']], dim=0).to(args.device)

        with torch.no_grad():
            ref_outputs = ref_model(x)
            ref_log_probs = logits_to_log_probs(ref_outputs.logits, y)

        outputs = model(x)
        policy_log_probs = logits_to_log_probs(outputs.logits, y)

        dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta)
        loss = dpo_loss_val + outputs.aux_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## 7. Knowledge Distillation (trainer/train_distillation.py)

```python
def distillation_loss(student_logits, teacher_logits, temperature=1.0):
    """KL divergence for distillation"""
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return (temperature ** 2) * kl

def train_epoch(epoch, loader, iters, teacher_model, alpha=0.5, temperature=1.5):
    teacher_model.eval()
    teacher_model.requires_grad_(False)

    for step, (X, Y, loss_mask) in enumerate(loader):
        X, Y, loss_mask = X.to(args.device), Y.to(args.device), loss_mask.to(args.device)

        res = model(X)
        student_logits = res.logits

        with torch.no_grad():
            teacher_logits = teacher_model(X).logits
            teacher_logits = teacher_logits[..., :student_logits.size(-1)]  # Align vocab sizes

        # 1) Ground truth CE loss
        ce_loss_raw = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), Y.view(-1),
                                     ignore_index=0, reduction='none')
        ce_loss_raw = torch.sum(ce_loss_raw * loss_mask.view(-1)) / loss_mask.sum()
        ce_loss = ce_loss_raw + res.aux_loss if lm_config.use_moe else ce_loss_raw

        # 2) Distillation loss
        distill_loss = distillation_loss(
            student_logits.view(-1, student_logits.size(-1))[loss_mask.view(-1) == 1],
            teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask.view(-1) == 1],
            temperature=temperature
        )

        # 3) Combined loss
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps
        loss.backward()
```

---

## 8. Reasoning Training (trainer/train_reason.py)

```python
def train_epoch(epoch, loader, iters, tokenizer, lm_config):
    # Special tokens for reasoning
    start_of_think_ids = tokenizer('<think>').input_ids
    end_of_think_ids = tokenizer('</think>').input_ids
    start_of_answer_ids = tokenizer('<answer>').input_ids
    end_of_answer_ids = tokenizer('</answer>').input_ids

    for step, (X, Y, loss_mask) in enumerate(loader):
        X, Y, loss_mask = X.to(args.device), Y.to(args.device), loss_mask.to(args.device)

        res = model(X)
        loss = F.cross_entropy(res.logits.view(-1, res.logits.size(-1)), Y.view(-1), reduction='none').view(Y.size())

        # Increase weight for special reasoning tokens
        sp_ids = torch.isin(Y.view(-1), torch.tensor(start_of_think_ids + end_of_think_ids +
                                                      start_of_answer_ids + end_of_answer_ids).to(args.device))
        loss_mask = loss_mask.view(-1)
        loss_mask_sum = loss_mask.sum()
        loss_mask[sp_ids] = 10  # 10x weight for thinking tags
        loss_mask = loss_mask.view(Y.size())

        logits_loss = (loss * loss_mask).sum() / loss_mask_sum
        loss = logits_loss + res.aux_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## 9. Model Evaluation (eval_llm.py)

```python
def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:  # PyTorch format
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:  # Transformers format
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)

    return model.eval().to(args.device), tokenizer

def main():
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
    ]

    conversation = []
    model, tokenizer = init_model(args)

    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    for prompt in prompts if input_mode == 0 else iter(lambda: input('💬: '), ''):
        setup_seed(2026)
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        if args.weight == 'reason': templates["enable_thinking"] = True
        inputs = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (tokenizer.bos_token + prompt)
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🤖: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')
```

---

## 10. Web Demo (scripts/web_demo.py)

```python
import streamlit as st
from threading import Thread
from transformers import TextIteratorStreamer

st.set_page_config(page_title="MiniMind", initial_sidebar_state="collapsed")

def process_assistant_content(content):
    """Process thinking tags for display"""
    if '<think>' in content and '</think>' in content:
        content = re.sub(r'(<think>)(.*?)(</think>)',
                         r'<details style="..."><summary>推理内容（展开）</summary>\2</details>',
                         content, flags=re.DOTALL)
    return content

def load_model_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model.eval().to(device), tokenizer

def main():
    model, tokenizer = load_model_tokenizer(model_path)

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []

    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar=image_url):
                st.markdown(process_assistant_content(message["content"]), unsafe_allow_html=True)
                if st.button("×", key=f"delete_{i}"):  # Delete message
                    st.session_state.messages.pop(i)
                    st.session_state.messages.pop(i - 1)
                    st.rerun()
        else:
            st.markdown(f'<div style="...">{message["content"]}</div>', unsafe_allow_html=True)

    if prompt := st.chat_input(key="input", placeholder="给 MiniMind 发送消息"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant", avatar=image_url):
            placeholder = st.empty()

            setup_seed(random.randint(0, 2**32 - 1))
            st.session_state.chat_messages = system_prompt + st.session_state.chat_messages[-(history_num + 1):]

            new_prompt = tokenizer.apply_chat_template(
                st.session_state.chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            Thread(target=model.generate, kwargs={
                "input_ids": inputs.input_ids,
                "max_length": inputs.input_ids.shape[1] + max_new_tokens,
                "temperature": temperature,
                "top_p": 0.85,
                "streamer": streamer
            }).start()

            answer = ""
            for new_text in streamer:
                answer += new_text
                placeholder.markdown(process_assistant_content(answer), unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
```

---

## 11. OpenAI-Compatible API Server (scripts/serve_openai_api.py)

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    stream: bool = False
    tools: list = []

class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end: self.queue.put(None)

def generate_stream_response(messages, temperature, top_p, max_tokens):
    queue = Queue()
    streamer = CustomStreamer(tokenizer, queue)

    new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)[-max_tokens:]
    inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

    def _generate():
        model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    Thread(target=_generate).start()

    while True:
        text = queue.get()
        if text is None:
            yield json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            break
        yield json.dumps({"choices": [{"delta": {"content": text}}]})

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        if request.stream:
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens
                )),
                media_type="text/event-stream"
            )
        else:
            new_prompt = tokenizer.apply_chat_template(request.messages, tokenize=False, add_generation_prompt=True)[-request.max_tokens:]
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 12. Model Conversion (scripts/convert_model.py)

```python
# Convert PyTorch weights to Transformers format
def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.float16):
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    lm_model = MiniMindForCausalLM(lm_config)
    state_dict = torch.load(torch_path, map_location='cpu')
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)

    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)

# Convert to Llama format for ecosystem compatibility
def convert_torch2transformers_llama(torch_path, transformers_path, dtype=torch.float16):
    state_dict = torch.load(torch_path, map_location='cpu')
    llama_config = LlamaConfig(
        vocab_size=lm_config.vocab_size,
        hidden_size=lm_config.hidden_size,
        intermediate_size=64 * ((int(lm_config.hidden_size * 8 / 3) + 64 - 1) // 64),
        num_hidden_layers=lm_config.num_hidden_layers,
        num_attention_heads=lm_config.num_attention_heads,
        num_key_value_heads=lm_config.num_key_value_heads,
        max_position_embeddings=lm_config.max_position_embeddings,
        rms_norm_eps=lm_config.rms_norm_eps,
        rope_theta=lm_config.rope_theta,
        tie_word_embeddings=True
    )
    llama_model = LlamaForCausalLM(llama_config)
    llama_model.load_state_dict(state_dict, strict=False)
    llama_model = llama_model.to(dtype)
    llama_model.save_pretrained(transformers_path)
```

---

## 13. Tokenizer Configuration (model/tokenizer_config.json)

```json
{
    "add_bos_token": false,
    "add_eos_token": false,
    "bos_token": "<|im_start|>",
    "eos_token": "<|im_end|>",
    "model_max_length": 32768,
    "pad_token": "<|endoftext|>",
    "chat_template": "{%- if tools %}...{% else %}...{% endif %}"
}
```

---

## 14. Dependencies (requirements.txt)

```
torch==2.6.0
transformers==4.57.1
datasets==3.6.0
tokenizers>=0.15.0
jinja2==3.1.2
streamlit==1.50.0
fastapi==0.115.0
uvicorn==0.32.0
openai==1.59.6
sentencepiece==0.2.1
numpy==1.26.4
matplotlib==3.10.0
scikit_learn==1.5.1
swanlab==0.6.8
wandb==0.18.3
einops==0.8.1
peft==0.7.1
trl==0.13.0
Flask==3.0.3
Flask_Cors==4.0.0
jieba==0.42.1
nltk==3.8
psutil==5.9.8
pydantic==2.11.5
rich==13.7.1
sentence_transformers==2.3.1
ujson==5.1.0
```

---

## Complete Architecture Summary

### Training Pipeline Flow
```
Raw Data → Tokenizer → Dataset → DataLoader → Model → Optimizer → Checkpoint
     ↓                                                        ↓
     └───────────────────── W&B/SwanLab ─────────────────────┘
```

### Model Architecture Flow
```
Input Tokens → Embedding → Dropout → [Block × N] → RMSNorm → LM Head
                                       ↓
                                    Block:
                                    Input → RMSNorm → Attention → + → RMSNorm → MLP → +
                                              ↑                   ↓              ↑
                                              └───── Residual ───┘              └─ Residual
```

### MoE Block Flow
```
MLP Input → Gate → Top-k Routing → Expert 1 └───┐
                                   Expert 2 ─┐   │
                                   Expert 3 ─┼───┼──→ Weighted Sum → + Shared Expert → Output
                                   Expert 4 ─┘   │
                                      ↑          │
                                      └─ Top-k Weights
```

### Key Innovation Points
1. **Minimal dependencies**: Pure PyTorch implementation
2. **Educational clarity**: Every algorithm explicit and commented
3. **Scalable architecture**: Works from 26M to 334M parameters
4. **Complete pipeline**: Pretrain → SFT → DPO → RLHF → Distillation
5. **Efficient training**: Flash attention, MoE routing, gradient accumulation
6. **Production ready**: OpenAI-compatible API, web UI, model conversion
7. **Hardware accessible**: Trains on single consumer GPU in 2 hours
8. **Modern techniques**: RoPE with YaRN, grouped query attention, LoRA

This complete compilation shows MiniMind as a fully self-contained, production-ready LLM training system that bridges the gap between educational simplicity and practical utility.