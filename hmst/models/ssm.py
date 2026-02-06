"""
State Space Model (SSM) implementation for Episodic Memory

Based on Mamba architecture with selective mechanisms.

⚠️ PERFORMANCE WARNING:
This implementation uses a Python loop for the sequential scan operation,
which is 100-1000x slower than optimized CUDA kernels.

For production use, consider:
1. Using the official mamba-ssm package (pip install mamba-ssm)
2. Implementing a parallel scan algorithm
3. Using a custom CUDA kernel
4. Reducing sequence length during testing (e.g., 512 instead of 8192)

Current implementation is functional but slow - suitable for:
- Architecture validation with short sequences
- Testing and debugging
- Understanding the algorithm

NOT suitable for:
- Full-scale training with long sequences (>1024)
- Production inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (Mamba-style).

    Key innovation: Parameters A, B, C, Δ are input-dependent.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 256,
        d_conv: int = 4,
        expand: int = 2
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )

        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + self.d_inner)

        # State space matrices (fixed initialization)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, -1)
        A = A.repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # Log for numerical stability

        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through selective SSM.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Warn about performance with long sequences
        if seq_len > 1024 and not hasattr(self, '_warned_about_performance'):
            import warnings
            warnings.warn(
                f"SelectiveSSM: Processing sequence length {seq_len} with unoptimized Python loop. "
                f"Expect 100-1000x slowdown. Consider: (1) reducing sequence length, "
                f"(2) using mamba-ssm package, or (3) implementing parallel scan.",
                RuntimeWarning,
                stacklevel=2
            )
            self._warned_about_performance = True

        # Input projection
        x_and_res = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x_inner, res = x_and_res.split(self.d_inner, dim=-1)

        # Convolution (local context)
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Generate input-dependent parameters
        ssm_params = self.x_proj(x_conv)  # (batch, seq_len, d_state + d_state + d_inner)
        B, C, delta = torch.split(
            ssm_params,
            [self.d_state, self.d_state, self.d_inner],
            dim=-1
        )

        # Discretization parameter (always positive)
        delta = F.softplus(delta)

        # SSM computation
        y = self._selective_scan(x_conv, delta, B, C)

        # Gating with residual
        y = y * F.silu(res)

        # Output projection
        output = self.out_proj(y)

        return output

    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        Core selective scan operation.

        ⚠️ PERFORMANCE BOTTLENECK: This uses a Python loop over sequence length.
        For seq_len=8192, this loop runs 8192 times per forward pass, causing
        100-1000x slowdown compared to parallel scan or CUDA implementations.

        Production alternatives:
        - Parallel scan algorithm (associative scan)
        - CUDA kernel (see mamba-ssm package)
        - Reduce sequence length for testing

        Args:
            x: (batch, seq_len, d_inner)
            delta: (batch, seq_len, d_inner)
            B: (batch, seq_len, d_state)
            C: (batch, seq_len, d_state)

        Returns:
            y: (batch, seq_len, d_inner)
        """
        batch_size, seq_len, d_inner = x.shape

        # Recover A from log
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Discretize: A_bar = exp(Δ * A), B_bar = Δ * B
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (batch, seq_len, d_inner, d_state)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (batch, seq_len, d_inner, d_state)

        # Sequential scan (SLOW - Python loop over sequence length)
        # TODO: Replace with parallel scan or CUDA kernel for production
        h = torch.zeros(batch_size, d_inner, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):  # ⚠️ BOTTLENECK: O(seq_len) sequential iterations
            h = deltaA[:, t] * h + deltaB[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1) + self.D * x[:, t]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)

        return y


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
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
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
