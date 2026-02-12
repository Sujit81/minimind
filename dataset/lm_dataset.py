from torch.utils.data import Dataset
import torch
import os
import numpy as np
import pickle
from datasets import load_dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BinaryPretrainDataset(Dataset):
    """
    Memory-efficient dataset for pre-tokenized binary files.

    Binary format:
    - int32 tokens, fixed 1024 sequence length
    - Memory mapped (no loading into RAM)
    - No tokenizer needed during training
    """
    def __init__(self, data_path, max_length=1024):
        super().__init__()
        self.max_length = max_length

        # Load binary file
        if os.path.isdir(data_path):
            # Directory mode: look for train.bin and eval.bin
            train_file = os.path.join(data_path, "train.bin")
            eval_file = os.path.join(data_path, "eval.bin")
            self.train_data = self._load_binary(train_file) if os.path.exists(train_file) else None
            self.eval_data = self._load_binary(eval_file) if os.path.exists(eval_file) else None
            self.mode = 'binary_dir'
            self.samples = self._create_samples_from_binary()
        elif data_path.endswith('.bin'):
            # Single file mode
            self.train_data = self._load_binary(data_path)
            self.eval_data = None
            self.mode = 'binary_file'
            self.samples = self._create_samples_from_binary()
        else:
            raise ValueError(f"Unsupported data format: {data_path}. Expected .bin or directory with train.bin/eval.bin")

    def _ensure_sequence_array(self, data, filepath):
        """Normalize loaded data to shape [num_samples, seq_len]."""
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.int32)

        if data.dtype != np.int32:
            data = data.astype(np.int32)

        if data.ndim == 1:
            total = (len(data) // self.max_length) * self.max_length
            if total == 0:
                raise ValueError(f"No enough tokens in {filepath} to form one sequence of length {self.max_length}")
            if total < len(data):
                print(f"[BinaryPretrainDataset] Trimmed {len(data) - total} tail tokens from {filepath}")
            data = data[:total].reshape(-1, self.max_length)
        elif data.ndim > 2:
            data = data.reshape(data.shape[0], -1)

        return data

    def _extract_tokens_from_pickle(self, obj, out):
        """Recursively extract integer tokens from common pickled structures."""
        if isinstance(obj, (int, np.integer)):
            out.append(int(obj))
            return

        if isinstance(obj, dict):
            if 'tokens' in obj:
                self._extract_tokens_from_pickle(obj['tokens'], out)
            else:
                for value in obj.values():
                    self._extract_tokens_from_pickle(value, out)
            return

        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                for item in obj.tolist():
                    self._extract_tokens_from_pickle(item, out)
            else:
                out.extend(obj.astype(np.int64).reshape(-1).tolist())
            return

        if isinstance(obj, (list, tuple)):
            for item in obj:
                self._extract_tokens_from_pickle(item, out)

    def _load_pickled_binary(self, filepath):
        """Load legacy pickled .bin format and convert to fixed-length int32 sequences."""
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)

        if isinstance(obj, np.ndarray) and obj.dtype != object:
            return self._ensure_sequence_array(obj, filepath)

        tokens = []
        self._extract_tokens_from_pickle(obj, tokens)
        if len(tokens) == 0:
            raise ValueError(f"No integer tokens extracted from pickled file: {filepath}")

        return self._ensure_sequence_array(np.asarray(tokens, dtype=np.int32), filepath)

    def _load_raw_binary(self, filepath):
        """Load raw token stream binary via memmap (e.g., int32/uint16 flat files)."""
        file_size = os.path.getsize(filepath)
        if file_size <= 0:
            raise ValueError(f"Empty binary file: {filepath}")

        candidates = []
        if file_size % np.dtype(np.int32).itemsize == 0:
            candidates.append(np.int32)
        if file_size % np.dtype(np.uint16).itemsize == 0:
            candidates.append(np.uint16)

        if not candidates:
            raise ValueError(f"Unsupported raw binary size for {filepath}: {file_size} bytes")

        last_error = None
        for dtype in candidates:
            try:
                data = np.memmap(filepath, dtype=dtype, mode='r')
                data = self._ensure_sequence_array(data, filepath)
                print(f"[BinaryPretrainDataset] Loaded raw binary with dtype={np.dtype(dtype).name}: {filepath}")
                return data
            except Exception as err:
                last_error = err

        raise ValueError(f"Failed to parse raw binary file {filepath}: {last_error}")

    def _load_binary(self, filepath):
        """Load .bin file as numpy format; fallback to pickle, then raw memmap."""
        try:
            data = np.load(filepath, mmap_mode='r', allow_pickle=False)
            print(f"[BinaryPretrainDataset] Loaded NumPy binary: {filepath}")
            return self._ensure_sequence_array(data, filepath)
        except ValueError as err:
            err_text = str(err).lower()
            if 'pickled' not in err_text and 'allow_pickle' not in err_text:
                raise
            print(f"[BinaryPretrainDataset] NumPy load rejected as pickled: {filepath}, trying pickle fallback")

        try:
            data = self._load_pickled_binary(filepath)
            print(f"[BinaryPretrainDataset] Loaded legacy pickled binary: {filepath}")
            return data
        except Exception as err:
            print(f"[BinaryPretrainDataset] Pickle fallback failed ({err}), trying raw binary fallback: {filepath}")

        return self._load_raw_binary(filepath)

    def _create_samples_from_binary(self):
        """Create dict-based samples for compatibility (only used for __len__)"""
        if self.train_data is not None:
            num_samples = self.train_data.shape[0] if self.mode.startswith('binary') else 0
            return [{'text': f'sample_{i}'} for i in range(num_samples)]
        return []

    def __len__(self):
        if self.mode.startswith('binary'):
            return self.train_data.shape[0] if self.train_data is not None else 0
        return len(self.samples)

    def __getitem__(self, index):
        if self.mode.startswith('binary'):
            # Binary mode: direct token access
            tokens = self.train_data[index]

            # Pad/truncate to max_length
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            elif len(tokens) < self.max_length:
                # Pad with pad token (0)
                tokens = np.concatenate([tokens, np.zeros(self.max_length - len(tokens), dtype=np.int32)])

            # X and Y for next token prediction
            X = torch.tensor(tokens[:-1], dtype=torch.long)
            Y = torch.tensor(tokens[1:], dtype=torch.long)

            # Loss mask: 1 for all tokens (binary mode has all valid tokens)
            loss_mask = torch.ones(self.max_length - 1, dtype=torch.long)

            return X, Y, loss_mask
        else:
            # Fallback to samples (should not reach here in binary mode)
            sample = self.samples[index]
            raise ValueError(f"Invalid mode: {self.mode}")


class PretrainDataset(Dataset):
    """
    Unified pretraining dataset supporting both JSONL and binary formats.
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Auto-detect format
        if os.path.isdir(data_path):
            # Directory mode: look for train.bin and eval.bin
            train_path = os.path.join(data_path, "train.bin")
            eval_path = os.path.join(data_path, "eval.bin")

            if os.path.exists(train_path):
                print(f"[PretrainDataset] Using binary format: {train_path}")
                self.train_data = BinaryPretrainDataset(train_path, max_length=max_length)
                self.eval_data = BinaryPretrainDataset(eval_path, max_length=max_length) if os.path.exists(eval_path) else None
                self.mode = 'binary_dir'
                self.samples = self.train_data
            elif os.path.exists(data_path + ".jsonl") or os.path.exists(data_path + ".json"):
                # Fall back to JSONL
                jsonl_path = data_path if data_path.endswith('.jsonl') else data_path + ".jsonl"
                print(f"[PretrainDataset] Using JSONL format: {jsonl_path}")
                self.samples = load_dataset('json', data_files=jsonl_path, split='train')
                self.train_data = None
                self.eval_data = None
                self.mode = 'jsonl'
        elif data_path.endswith('.bin'):
            # Single binary file
            print(f"[PretrainDataset] Using binary file: {data_path}")
            self.train_data = BinaryPretrainDataset(data_path, max_length=max_length)
            self.eval_data = None
            self.mode = 'binary_file'
            self.samples = self.train_data
        elif data_path.endswith('.jsonl') or data_path.endswith('.json'):
            # Single JSONL file (original behavior)
            jsonl_path = data_path
            print(f"[PretrainDataset] Using JSONL format: {jsonl_path}")
            self.samples = load_dataset('json', data_files=jsonl_path, split='train')
            self.train_data = None
            self.eval_data = None
            self.mode = 'jsonl'
        else:
            raise FileNotFoundError(f"No valid data found in {data_path} (expected train.bin, train.jsonl, or directory)")

    def __len__(self):
        if self.mode.startswith('binary'):
            return len(self.train_data) if self.train_data is not None else 0
        return len(self.samples)

    def __getitem__(self, index):
        if self.mode.startswith('binary'):
            # Binary mode: delegate to BinaryPretrainDataset
            return self.train_data[index]
        else:
            # Original JSONL mode
            sample = self.samples[index]

            # Build input text
            encoding = self.tokenizer(
                str(sample['text']),
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding.input_ids.squeeze()
            loss_mask = (input_ids != self.tokenizer.pad_token_id)

            X = torch.tensor(input_ids[:-1], dtype=torch.long)
            Y = torch.tensor(input_ids[1:], dtype=torch.long)
            loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
            return X, Y, loss_mask


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, cs):
        messages = cs.copy()
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self.generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids
        self.data = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item['chosen']
        rejected = item['rejected']

        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )

        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']
        return self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        ), answer

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt, answer = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': answer
        }


if __name__ == "__main__":
    pass
