"""
Proper Tokenizer for HMST using HuggingFace Transformers

Uses pre-trained GPT-2 BPE tokenizer with 50K vocabulary.
"""

from transformers import GPT2Tokenizer
from typing import List, Union
import torch


class HMSTTokenizer:
    """
    Wrapper around HuggingFace GPT-2 tokenizer for HMST.

    Uses BPE (Byte-Pair Encoding) with 50,257 token vocabulary.
    Includes proper subword decomposition for better generalization.
    """

    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize tokenizer.

        Args:
            model_name: HuggingFace model name (default: gpt2)
                       Options: gpt2, gpt2-medium, gpt2-large, gpt2-xl
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Add a distinct pad token so it doesn't conflict with EOS during training
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})

        self.vocab_size = len(self.tokenizer)

        # Special token IDs
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: int = None,
        truncation: bool = True,
        padding: bool = False
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode text to token IDs.

        Args:
            text: Text string or list of strings
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum sequence length
            truncation: Truncate if exceeds max_length
            padding: Pad to max_length

        Returns:
            Token IDs (list or list of lists)
        """
        if isinstance(text, str):
            encoding = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=truncation,
                padding='max_length' if padding else False
            )
            return encoding
        else:
            # Batch encoding
            encodings = self.tokenizer.batch_encode_plus(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=truncation,
                padding='max_length' if padding else False
            )
            return encodings['input_ids']

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs (list or tensor)
            skip_special_tokens: Skip PAD/EOS/BOS tokens

        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )

    def batch_decode(
        self,
        token_ids: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode batch of token IDs to texts.

        Args:
            token_ids: Batch of token IDs
            skip_special_tokens: Skip PAD/EOS/BOS tokens

        Returns:
            List of decoded text strings
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def save(self, path: str):
        """Save tokenizer to disk."""
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str):
        """Load tokenizer from disk."""
        instance = cls.__new__(cls)
        instance.tokenizer = GPT2Tokenizer.from_pretrained(path)
        instance.vocab_size = len(instance.tokenizer)
        instance.pad_token_id = instance.tokenizer.pad_token_id
        instance.eos_token_id = instance.tokenizer.eos_token_id
        instance.bos_token_id = instance.tokenizer.bos_token_id
        return instance
