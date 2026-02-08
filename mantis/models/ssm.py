"""
State Space Model (SSM) implementation for Episodic Memory

Uses mamba-ssm for efficient selective state space computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    """
    Single Mamba block with SSM and normalization.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 256,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.ssm = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        # Pre-norm
        normed = self.norm(x)
        ssm_out = self.ssm(normed)
        output = x + self.dropout(ssm_out)

        return output


class EpisodicMemorySSM(nn.Module):
    """
    Episodic memory module using stacked Mamba blocks.

    Compresses recent interactions into a continuous state vector.
    """

    def __init__(
        self,
        d_model: int = 2048,
        d_state: int = 256,
        n_blocks: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        max_seq_len: int = 8192,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_proj = nn.Linear(d_model, d_model)

        # Stacked Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_blocks)
        ])

        # Output projection to state
        self.state_proj = nn.Linear(d_model, d_state)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        return_state: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process sequence and optionally return compressed state.

        Args:
            x: (batch, seq_len, d_model)
            return_state: Return compressed state vector

        Returns:
            output: (batch, seq_len, d_model)
            state: Optional (batch, d_state) compressed representation
        """
        # Input projection
        h = self.input_proj(x)

        # Process through Mamba blocks
        for block in self.blocks:
            h = block(h)

        # Final normalization
        h = self.final_norm(h)

        # Compress to state vector
        if return_state:
            # Mean pooling + projection
            state = h.mean(dim=1)  # (batch, d_model)
            state = self.state_proj(state)  # (batch, d_state)
            return h, state
        else:
            return h, None

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence into state vector only.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            state: (batch, d_state)
        """
        _, state = self.forward(x, return_state=True)
        return state
