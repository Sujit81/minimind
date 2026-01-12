# Hindi Tokenizer Wrapper
# Ensures consistent normalization between training and inference
#
# This wrapper applies the same Indic-specific normalization that was used
# during tokenizer training, solving the training/inference mismatch problem.

import os
from typing import List, Optional, Union, Dict, Any


class TokenizerOutput(dict):
    """
    Simple dict-like class that also supports attribute access.
    Mimics HuggingFace's BatchEncoding for compatibility with existing code.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'TokenizerOutput' has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def squeeze(self):
        """Return self for compatibility (single-item batch doesn't need squeeze)."""
        return self


class HindiTokenizer:
    """
    Custom Hindi tokenizer wrapper that ensures consistent normalization
    between training and inference.

    This wrapper:
    1. Applies Indic NLP normalization (same as training)
    2. Optionally applies pre-tokenization at akshar boundaries
    3. Delegates to the underlying SentencePiece/HuggingFace tokenizer

    Usage:
        tokenizer = HindiTokenizer.from_pretrained("./model_hindi")
        tokens = tokenizer.encode("नमस्ते")
        text = tokenizer.decode(tokens)
    """

    def __init__(
        self,
        tokenizer_path: str,
        use_indic_normalizer: bool = True,
        use_akshar_pretokenizer: bool = False
    ):
        """
        Initialize the Hindi tokenizer.

        Args:
            tokenizer_path: Path to the tokenizer directory
            use_indic_normalizer: Apply Indic NLP normalization (recommended)
            use_akshar_pretokenizer: Apply akshar-boundary pre-tokenization
        """
        self.tokenizer_path = tokenizer_path
        self.use_indic_normalizer = use_indic_normalizer
        self.use_akshar_pretokenizer = use_akshar_pretokenizer

        # Load underlying tokenizer
        self._load_tokenizer()

        # Initialize normalizer
        self.indic_normalizer = None
        if use_indic_normalizer:
            self._init_normalizer()

        # Initialize pre-tokenizer
        self.syllabify_func = None
        if use_akshar_pretokenizer:
            self._init_pretokenizer()

    def _load_tokenizer(self):
        """Load the underlying SentencePiece/HuggingFace tokenizer."""
        # Try HuggingFace first
        try:
            from transformers import AutoTokenizer
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True
            )
            self.use_hf = True
        except Exception:
            self.hf_tokenizer = None
            self.use_hf = False

        # Always load SentencePiece for direct access
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        model_path = os.path.join(self.tokenizer_path, "tokenizer.model")
        if os.path.exists(model_path):
            self.sp.load(model_path)
        else:
            raise FileNotFoundError(f"tokenizer.model not found at {model_path}")

    def _init_normalizer(self):
        """Initialize the Indic NLP normalizer."""
        try:
            from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
            normalizer_factory = IndicNormalizerFactory()
            self.indic_normalizer = normalizer_factory.get_normalizer('hi')
        except ImportError:
            print("Warning: indic-nlp-library not available. Normalization disabled.")
            print("Install with: pip install indic-nlp-library")
            self.indic_normalizer = None
            self.use_indic_normalizer = False

    def _init_pretokenizer(self):
        """Initialize the akshar-boundary pre-tokenizer."""
        try:
            from indicnlp.syllable import syllabifier
            self.syllabify_func = syllabifier.orthographic_syllabify
        except ImportError:
            print("Warning: indic-nlp-library syllabifier not available.")
            self.syllabify_func = None
            self.use_akshar_pretokenizer = False

    def normalize(self, text: str) -> str:
        """
        Apply Indic-specific normalization to text.

        This MUST match the normalization applied during training.
        """
        if self.indic_normalizer is not None:
            return self.indic_normalizer.normalize(text)
        return text

    def pretokenize(self, text: str) -> str:
        """
        Apply akshar-boundary pre-tokenization.

        Inserts markers at valid syllable boundaries to guide tokenization.
        This is optional and mainly useful for debugging.
        """
        if not self.syllabify_func:
            return text

        import re
        # Find Devanagari words and syllabify them
        devanagari_pattern = re.compile(r'[\u0900-\u097F]+')

        result = []
        last_end = 0

        for match in devanagari_pattern.finditer(text):
            # Add non-Devanagari text as-is
            result.append(text[last_end:match.start()])

            # Syllabify the Devanagari word
            word = match.group()
            try:
                syllables = self.syllabify_func(word, 'hi')
                if syllables:
                    # Join syllables with a thin space (U+2009) as boundary marker
                    result.append('\u2009'.join(syllables))
                else:
                    result.append(word)
            except Exception:
                result.append(word)

            last_end = match.end()

        # Add remaining text
        result.append(text[last_end:])

        return ''.join(result)

    def preprocess(self, text: str) -> str:
        """
        Apply all preprocessing steps (normalization + optional pre-tokenization).

        This ensures inference matches training preprocessing.
        """
        if self.use_indic_normalizer:
            text = self.normalize(text)
        if self.use_akshar_pretokenizer:
            text = self.pretokenize(text)
        return text

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_tensors: Optional[str] = None
    ) -> Union[List[int], Any]:
        """
        Encode text to token IDs.

        Args:
            text: Input text (will be normalized automatically)
            add_special_tokens: Add BOS/EOS tokens
            return_tensors: Return as tensors ("pt" for PyTorch)

        Returns:
            List of token IDs or tensor
        """
        # Preprocess text (normalization + optional pre-tokenization)
        text = self.preprocess(text)

        if self.use_hf and return_tensors:
            return self.hf_tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                return_tensors=return_tensors
            )['input_ids']

        # Use SentencePiece directly
        if add_special_tokens:
            # Add BOS (1) and EOS (2)
            ids = [1] + self.sp.encode(text) + [2]
        else:
            ids = self.sp.encode(text)

        if return_tensors == "pt":
            import torch
            return torch.tensor([ids])

        return ids

    # Special token mappings (must match tokenizer training config)
    SPECIAL_TOKEN_MAP = {
        0: "<|endoftext|>",  # PAD/UNK
        1: "<|im_start|>",   # BOS
        2: "<|im_end|>",     # EOS
    }

    def decode(
        self,
        token_ids: Union[List[int], Any],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Skip special tokens (BOS/EOS/PAD)

        Returns:
            Decoded text
        """
        # Convert tensor to list if needed
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
            token_ids = token_ids[0]  # Handle batch dimension

        if skip_special_tokens:
            # Remove special tokens (0=PAD, 1=BOS, 2=EOS)
            token_ids = [t for t in token_ids if t not in self.SPECIAL_TOKEN_MAP]
            return self.sp.decode(token_ids)

        # When not skipping special tokens, we need to decode in segments
        # because SentencePiece doesn't know our custom special token strings
        result = []
        regular_ids = []

        for tid in token_ids:
            if tid in self.SPECIAL_TOKEN_MAP:
                # Decode accumulated regular tokens first
                if regular_ids:
                    result.append(self.sp.decode(regular_ids))
                    regular_ids = []
                # Add the special token string
                result.append(self.SPECIAL_TOKEN_MAP[tid])
            else:
                regular_ids.append(tid)

        # Decode any remaining regular tokens
        if regular_ids:
            result.append(self.sp.decode(regular_ids))

        return ''.join(result)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into subword pieces.

        Args:
            text: Input text (will be normalized automatically)

        Returns:
            List of subword tokens
        """
        text = self.preprocess(text)
        return self.sp.encode_as_pieces(text)

    # Reverse mapping for encoding special token strings
    SPECIAL_TOKEN_STR_TO_ID = {
        "<|endoftext|>": 0,
        "<|im_start|>": 1,
        "<|im_end|>": 2,
    }

    def _encode_with_special_tokens(self, text: str, preprocess: bool = False) -> List[int]:
        """
        Encode text that may contain special token strings.

        This handles chat templates where <|im_start|> and <|im_end|>
        appear as literal strings in the text.

        Args:
            text: Input text with special token strings
            preprocess: Whether to apply preprocessing (set False if already preprocessed)
        """
        import re

        # Pattern to match special tokens
        special_pattern = re.compile(r'(<\|(?:endoftext|im_start|im_end)\|>)')

        # Split text by special tokens, keeping the delimiters
        parts = special_pattern.split(text)

        ids = []
        for part in parts:
            if not part:
                continue
            if part in self.SPECIAL_TOKEN_STR_TO_ID:
                ids.append(self.SPECIAL_TOKEN_STR_TO_ID[part])
            else:
                # Regular text - optionally preprocess and encode
                if preprocess:
                    part = self.preprocess(part)
                ids.extend(self.sp.encode(part))

        return ids

    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        HuggingFace-compatible tokenization interface.

        Args:
            text: Input text or list of texts
            add_special_tokens: Add BOS/EOS tokens
            return_tensors: Return as tensors ("pt" for PyTorch)
            padding: Pad sequences to same length
            truncation: Truncate to max_length
            max_length: Maximum sequence length

        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        if isinstance(text, str):
            text = [text]

        # Preprocess and encode all texts
        all_ids = []
        for t in text:
            # Check if text contains special token strings (e.g., from chat template)
            if '<|' in t and '|>' in t:
                # Text with special tokens is likely from apply_chat_template,
                # which already preprocessed the content. Don't preprocess again.
                ids = self._encode_with_special_tokens(t, preprocess=False)
                # Don't add BOS/EOS if text already contains special tokens
            else:
                t = self.preprocess(t)
                ids = self.sp.encode(t)
                if add_special_tokens:
                    ids = [1] + ids + [2]  # BOS + text + EOS
            if truncation and max_length:
                ids = ids[:max_length]
            all_ids.append(ids)

        # Padding
        # Handle padding='max_length' (string) or padding=True
        pad_to_max = (padding == 'max_length') or (isinstance(padding, str) and 'max' in padding.lower())
        pad_to_longest = padding is True or padding == 'longest'

        if pad_to_max and max_length:
            # Pad all sequences to max_length
            target_len = max_length
            attention_mask = []
            for i, ids in enumerate(all_ids):
                mask = [1] * len(ids)
                if len(ids) < target_len:
                    pad_len = target_len - len(ids)
                    ids.extend([0] * pad_len)  # PAD token = 0
                    mask.extend([0] * pad_len)
                all_ids[i] = ids
                attention_mask.append(mask)
        elif pad_to_longest and len(all_ids) > 1:
            # Pad to longest in batch
            max_len = max(len(ids) for ids in all_ids)
            if max_length:
                max_len = min(max_len, max_length)
            attention_mask = []
            for i, ids in enumerate(all_ids):
                mask = [1] * len(ids)
                if len(ids) < max_len:
                    pad_len = max_len - len(ids)
                    ids.extend([0] * pad_len)  # PAD token = 0
                    mask.extend([0] * pad_len)
                all_ids[i] = ids
                attention_mask.append(mask)
        else:
            attention_mask = [[1] * len(ids) for ids in all_ids]

        # For single string input, return flat list (not nested) for compatibility
        # with code like: tokenizer(text).input_ids[:max_length]
        if len(all_ids) == 1:
            input_ids = all_ids[0]
            attn_mask = attention_mask[0]
        else:
            input_ids = all_ids
            attn_mask = attention_mask

        if return_tensors == "pt":
            import torch
            if isinstance(input_ids[0], list):
                input_ids = torch.tensor(input_ids)
                attn_mask = torch.tensor(attn_mask)
            else:
                input_ids = torch.tensor([input_ids])
                attn_mask = torch.tensor([attn_mask])

        return TokenizerOutput({
            'input_ids': input_ids,
            'attention_mask': attn_mask
        })

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs
    ) -> Union[str, List[int]]:
        """
        Apply chat template to messages.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            tokenize: Return token IDs instead of string
            add_generation_prompt: Add assistant prompt for generation

        Returns:
            Formatted string or token IDs
        """
        if self.use_hf and self.hf_tokenizer is not None:
            return self.hf_tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **kwargs
            )

        # Manual chat template (matches training config)
        default_system = "आप एक सहायक AI हैं।"

        # Make a copy to avoid modifying the original
        messages = list(messages)

        parts = []

        # System message
        if messages and messages[0]['role'] == 'system':
            # Preprocess content for consistent roundtrip
            content = self.preprocess(messages[0]['content'])
            parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
            messages = messages[1:]
        else:
            content = self.preprocess(default_system)
            parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")

        # Conversation
        for msg in messages:
            role = msg['role']
            # Preprocess content for consistent roundtrip
            content = self.preprocess(msg['content'])
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        # Generation prompt
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")

        text = ''.join(parts)

        if tokenize:
            # Already preprocessed, don't preprocess again
            return self._encode_with_special_tokens(text, preprocess=False)
        return text

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.sp.get_piece_size()

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def bos_token_id(self) -> int:
        return 1

    @property
    def eos_token_id(self) -> int:
        return 2

    @property
    def pad_token(self) -> str:
        return "<|endoftext|>"

    @property
    def bos_token(self) -> str:
        return "<|im_start|>"

    @property
    def eos_token(self) -> str:
        return "<|im_end|>"

    @property
    def unk_token(self) -> str:
        return "<|endoftext|>"

    def convert_tokens_to_ids(self, token: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Convert token(s) to ID(s).

        Args:
            token: Single token string or list of tokens

        Returns:
            Token ID or list of IDs
        """
        if isinstance(token, list):
            return [self.convert_tokens_to_ids(t) for t in token]

        # Check special tokens first
        if token in self.SPECIAL_TOKEN_STR_TO_ID:
            return self.SPECIAL_TOKEN_STR_TO_ID[token]

        # Use SentencePiece for regular tokens
        return self.sp.piece_to_id(token)

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """
        Convert ID(s) to token(s).

        Args:
            ids: Single ID or list of IDs

        Returns:
            Token string or list of tokens
        """
        if isinstance(ids, list):
            return [self.convert_ids_to_tokens(i) for i in ids]

        # Check special tokens first
        if ids in self.SPECIAL_TOKEN_MAP:
            return self.SPECIAL_TOKEN_MAP[ids]

        # Use SentencePiece for regular tokens
        return self.sp.id_to_piece(ids)

    @classmethod
    def from_pretrained(
        cls,
        tokenizer_path: str,
        use_indic_normalizer: bool = True,
        use_akshar_pretokenizer: bool = False,
        **kwargs
    ) -> 'HindiTokenizer':
        """
        Load a pre-trained Hindi tokenizer.

        Args:
            tokenizer_path: Path to tokenizer directory
            use_indic_normalizer: Apply Indic normalization (recommended)
            use_akshar_pretokenizer: Apply akshar pre-tokenization

        Returns:
            HindiTokenizer instance
        """
        return cls(
            tokenizer_path=tokenizer_path,
            use_indic_normalizer=use_indic_normalizer,
            use_akshar_pretokenizer=use_akshar_pretokenizer
        )
