# Hindi Tokenizer Implementation Plan for MiniMind

## Overview
Transform MiniMind to support Hindi + English bilingual text by training a new tokenizer and retraining the Base model (104M params) from scratch with increased vocabulary size (8000-10000).

---

## Phase 1: Data Collection & Preparation

### 1.1 Hindi Corpus Collection
**Location**: Create `D:\Code\Python\vedyon\minimind\dataset\hindi\`

**Required Data Sources**:
| Source | Type | Target Size |
|--------|------|-------------|
| Hindi Wikipedia | Articles | ~100MB |
| Hindi News (BBC, Navbharat Times) | News articles | ~50MB |
| Hindi Books (open source) | Literature | ~50MB |
| Hindi Web Content (Common Crawl) | Mixed web text | ~100MB |
| English Corpus | Existing | ~200MB |

**Tasks**:
1. Create `dataset/hindi/corpus_raw.txt` - Combined Hindi text files
2. Create `dataset/hindi/corpus_bilingual.txt` - Hindi + English mixed (70:30 ratio)
3. Create `dataset/hindi/corpus_pretrain.jsonl` - Pretraining data format
4. Create `dataset/hindi/sft_hindi.jsonl` - SFT conversation data
5. Create `dataset/hindi/dpo_hindi.jsonl` - DPO preference pairs

**Tools Needed**:
- `scripts/prepare_hindi_corpus.py` - Scraping/cleaning script
- Unicode normalization (NFC format for Devanagari)
- Deduplication script

### 1.2 Devanagari Character Coverage Verification
**File**: `dataset/hindi/verify_coverage.py`

**Must Include**:
- Basic vowels: अ आ इ ई उ ऊ ऋ ए ऐ ओ औ (11 characters)
- Consonants: क ख ग घ ङ ... (~35 consonants)
- Matras (vowel signs): ा ि ी ु ू ृ े ै ो ौ (~10 marks)
- Conjuncts: क्ष त्र ज्ञ श्र द्व (~5+ common conjuncts)
- Hindi punctuation: । ॥ ऽ
- Hindi numerals: ० १ २ ३ ४ ५ ६ ७ ८ ९
- Anusvara, Visarga: ं ः
- Chandrabindu: ँ

---

## Phase 2: Tokenizer Training

### 2.1 Modify Tokenizer Training Script
**File to Edit**: `D:\Code\Python\vedyon\minimind\trainer\train_tokenizer.py`

**Changes Required**:
```python
# Current (Line 9):
VOCAB_SIZE = 6400

# Change to:
VOCAB_SIZE = 10000  # Increased for Hindi+English bilingual

# Training data path:
TRAINING_DATA = "../dataset/hindi/corpus_bilingual.txt"

# Special tokens (preserve existing):
special_tokens = ["", "<|im_start|>", "<|im_end|>"]
```

### 2.2 Train New Hindi Tokenizer
**Command**:
```bash
cd D:\Code\Python\vedyon\minimind\trainer
python train_tokenizer.py
```

**Output Files** (in `model_hindi/`):
- `tokenizer.json` - New vocabulary with Hindi coverage
- `tokenizer_config.json` - Configuration
- `vocab.json` - Vocabulary list

### 2.3 Verify Tokenizer Quality
**Create**: `scripts/evaluate_hindi_tokenizer.py`

**Verification Metrics**:
1. Average tokens per Hindi word (target: <2.0)
2. Coverage of common Devanagari characters
3. Byte fallback rate (<1%)
4. Special token IDs preserved (0, 1, 2)

---

## Phase 3: Model Configuration Updates

### 3.1 Update Model Config
**File**: `D:\Code\Python\vedyon\minimind\model\model_minimind.py`

**Line 23 - Change default vocab_size**:
```python
# Current:
vocab_size: int = 6400

# Change to:
vocab_size: int = 10000  # For Hindi bilingual support
```

### 3.2 Create Hindi-Specific Model Config
**File**: `D:\Code\Python\vedyon\minimind\model\config_hindi.json`

```json
{
  "model_type": "minimind",
  "vocab_size": 10000,
  "hidden_size": 768,
  "num_hidden_layers": 16,
  "num_attention_heads": 8,
  "num_key_value_heads": 2,
  "max_position_embeddings": 32768,
  "rms_norm_eps": 1e-05,
  "rope_theta": 1000000.0,
  "hidden_act": "silu",
  "use_moe": false
}
```

### 3.3 Embedding Layer Sizes
**No code changes needed** - sizes are config-driven:
- `embed_tokens`: `(10000, 768)` instead of `(6400, 768)`
- `lm_head`: `(768, 10000)` instead of `(768, 6400)`

---

## Phase 4: Model Pretraining (Hindi)

### 4.1 Create Hindi Pretraining Script
**File**: `D:\Code\Python\vedyon\minimind\trainer\train_pretrain_hindi.py`

**Key Parameters**:
```python
# Model config (Base model):
hidden_size = 768
num_hidden_layers = 16
vocab_size = 10000

# Training params:
epochs = 2-3
batch_size = 32
learning_rate = 5e-4
max_seq_len = 512-1024
accumulation_steps = 8

# Paths:
tokenizer_path = "../model_hindi/"
data_path = "../dataset/hindi/corpus_pretrain.jsonl"
save_dir = "../out_hindi/"
save_weight = "pretrain_hindi"
```

### 4.2 Training Execution
**Command**:
```bash
cd D:\Code\Python\vedyon\minimind\trainer
torchrun --nproc_per_node=1 train_pretrain_hindi.py \
    --hidden_size 768 \
    --num_hidden_layers 16 \
    --vocab_size 10000 \
    --epochs 2 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --max_seq_len 512 \
    --data_path ../dataset/hindi/corpus_pretrain.jsonl \
    --save_dir ../out_hindi \
    --save_weight pretrain_hindi
```

**Estimated Training Time**:
- ~100GB of Hindi+English text
- Base model (104M params)
- Single GPU: ~2-3 weeks
- Multi-GPU (4x): ~5-7 days

---

## Phase 5: Supervised Fine-Tuning (Hindi)

### 5.1 Prepare Hindi SFT Data
**File**: `dataset/hindi/sft_hindi.jsonl`

**Format** (per line):
```json
{
  "conversations": [
    {"role": "user", "content": "भारत की राजधानी क्या है?"},
    {"role": "assistant", "content": "भारत की राजधानी नई दिल्ली है।"}
  ]
}
```

### 5.2 Run SFT Training
**File**: `D:\Code\Python\vedyon\minimind\trainer\train_full_sft_hindi.py`

**Command**:
```bash
python train_full_sft_hindi.py \
    --from_weight pretrain_hindi \
    --hidden_size 768 \
    --num_hidden_layers 16 \
    --data_path ../dataset/hindi/sft_hindi.jsonl \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 1e-6 \
    --max_seq_len 1024
```

---

## Phase 6: Evaluation & Testing

### 6.1 Create Hindi Evaluation Script
**File**: `scripts/eval_hindi.py`

**Test Prompts**:
```python
prompts_hindi = [
    "आपका नाम क्या है?",
    "भारत की जनसंख्या कितनी है?",
    "पांच अंकों की गुणा कीजिए: 234 × 567",
    "महात्मा गांधी कौन थे?",
    "फोटोसिंथेसिस क्या है?",
    "Python में एक फाइबोनैकी फंक्शन लिखें",
    # Mixed Hindi-English (Hinglish)
    "मुझे Python सिखाना है",
    "What is the capital of India? भारत की राजधानी क्या है?"
]
```

### 6.2 Evaluation Metrics
1. **Perplexity** on Hindi test set
2. **Token efficiency**: Average tokens per Hindi word
3. **Generation quality**: Human evaluation of responses
4. **Code switching**: Handling Hinglish (mixed language)

---

## Phase 7: File Structure Summary

### New Files to Create
```
minimind/
├── dataset/
│   └── hindi/
│       ├── corpus_raw.txt              # Raw Hindi text
│       ├── corpus_bilingual.txt        # Hindi+English mixed
│       ├── corpus_pretrain.jsonl       # Pretraining format
│       ├── sft_hindi.jsonl             # SFT conversations
│       ├── dpo_hindi.jsonl             # DPO pairs
│       └── verify_coverage.py          # Character coverage checker
│
├── model_hindi/                        # New tokenizer directory
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.json
│
├── out_hindi/                          # Model checkpoints
│   ├── pretrain_hindi_768.pth
│   ├── full_sft_hindi_768.pth
│   └── dpo_hindi_768.pth
│
├── trainer/
│   ├── train_tokenizer.py              # MODIFY: vocab_size=10000
│   ├── train_pretrain_hindi.py         # CREATE
│   └── train_full_sft_hindi.py         # CREATE
│
└── scripts/
    ├── prepare_hindi_corpus.py         # CREATE: data collection
    ├── evaluate_hindi_tokenizer.py     # CREATE: tokenizer validation
    └── eval_hindi.py                   # CREATE: Hindi evaluation
```

### Files to Modify
| File | Line | Change |
|------|------|--------|
| `model/model_minimind.py` | 23 | `vocab_size: int = 10000` |

---

## Phase 8: Implementation Order

### Step-by-Step Execution
1. **Week 1-2**: Data collection & corpus preparation
2. **Day 1**: Train tokenizer on prepared corpus
3. **Week 3-4**: Model pretraining (or longer depending on compute)
4. **Week 5**: SFT training on Hindi conversations
5. **Week 6**: Evaluation and iteration

### Critical Path
```
Data Collection → Tokenizer Training → Pretraining → SFT → Evaluation
                                    ↓
                            (Longest phase - most compute intensive)
```

---

## Verification Checklist

Before proceeding, verify:
- [ ] Hindi corpus collected (>200MB text)
- [ ] Devanagari character coverage verified
- [ ] Tokenizer trained with VOCAB_SIZE=10000
- [ ] Special tokens preserved (IDs 0, 1, 2)
- [ ] Tokenizer produces <2 tokens/word for Hindi
- [ ] Model config updated with vocab_size=10000
- [ ] Pretraining completed on Hindi corpus
- [ ] SFT data prepared in correct format
- [ ] Evaluation shows acceptable Hindi generation
- [ ] Hinglish (mixed) text handled correctly

---

## Hardware Requirements

| Phase | GPU Memory | Estimated Time |
|-------|-----------|----------------|
| Tokenizer Training | CPU only | 1-2 hours |
| Pretraining (104M) | 16GB+ | 2-3 weeks (1 GPU) |
| SFT Training | 12GB+ | 2-3 days |
| Evaluation | 8GB+ | Hours |

**Recommended**: Single A100 (40GB) or equivalent for reasonable training times.
