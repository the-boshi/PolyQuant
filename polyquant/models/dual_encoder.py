"""
Dual-Encoder Transformer for edge prediction with user history context.

Architecture:
1. Market Encoder: Processes market trade sequence (causal attention)
2. User Encoder: Processes user's historical trades (bidirectional)
3. Fusion Head: Combines market context, user context, and current trade features

The model predicts edge for each trade position, using:
- Market context from previous trades in this market
- User context from the user's historical trades across all markets
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    """
    Shared transformer encoder block.

    Can be configured for causal (market) or bidirectional (user) attention.
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.causal = causal

        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)

        # Learned positional embeddings
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Dropout
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final norm
        self.final_norm = nn.LayerNorm(d_model)

        # Causal mask buffer (upper triangular = True means blocked)
        if causal:
            self.register_buffer(
                "causal_mask",
                torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1),
            )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,      # (B, L, D_input)
        mask: torch.Tensor,   # (B, L) bool, True = valid
    ) -> torch.Tensor:
        """
        Returns:
            h: (B, L, d_model) encoded sequence
        """
        B, L, _ = x.shape
        device = x.device

        # Project input
        h = self.input_proj(x)  # (B, L, d_model)

        # Add positional embeddings
        positions = torch.arange(L, device=device)
        h = h + self.pos_embed(positions).unsqueeze(0)

        # Dropout
        h = self.embed_dropout(h)

        if self.causal:
            # For causal attention with left-padded sequences:
            # We need to combine causal mask (block future) with padding mask (block padding keys)
            # AND ensure each position can attend to at least itself to avoid NaN from softmax

            # Create combined mask in float format: 0 = attend, -inf = block
            # Shape: (B, L, L) which will be broadcast over heads

            # Causal mask: upper triangle is blocked (True)
            causal = self.causal_mask[:L, :L]  # (L, L) bool

            # Padding mask: positions where key is padding (mask is False for those)
            padding_keys = ~mask  # (B, L) - True where padding

            # Combined attention mask as floats
            # Start with causal mask expanded to (1, L, L)
            attn_mask = torch.zeros(B, L, L, device=device, dtype=h.dtype)

            # Block future positions (causal)
            attn_mask.masked_fill_(causal.unsqueeze(0), float('-inf'))

            # Block padding keys (expand padding_keys to (B, 1, L) and broadcast)
            attn_mask.masked_fill_(padding_keys.unsqueeze(1), float('-inf'))

            # CRITICAL: Ensure diagonal is never -inf to avoid NaN when a token
            # would otherwise attend to nothing (softmax of all -inf = NaN)
            # This allows each position to attend to itself
            diag_indices = torch.arange(L, device=device)
            attn_mask[:, diag_indices, diag_indices] = 0.0

            # Expand to (B * n_heads, L, L) for multi-head attention
            attn_mask = attn_mask.unsqueeze(1).expand(B, self.n_heads, L, L)
            attn_mask = attn_mask.reshape(B * self.n_heads, L, L)

            h = self.encoder(h, mask=attn_mask)
        else:
            # Bidirectional: only padding mask needed
            # Handle edge case: if ALL positions are masked, we'd get NaN from softmax
            # Solution: ensure at least position 0 can be attended to (it will be zeros anyway)
            src_key_padding_mask = ~mask  # True = ignore

            # Check for samples where all positions are masked
            all_masked = ~mask.any(dim=1)  # (B,) - True if sample has no valid positions
            if all_masked.any():
                # Allow position 0 to be attended (prevents NaN, output will be zero anyway)
                src_key_padding_mask = src_key_padding_mask.clone()
                src_key_padding_mask[all_masked, 0] = False

            h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        h = self.final_norm(h)

        return h

    def pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling over valid tokens."""
        mask_expanded = mask.unsqueeze(-1).float()  # (B, L, 1)
        h_masked = h * mask_expanded
        h_sum = h_masked.sum(dim=1)  # (B, d_model)
        mask_sum = mask_expanded.sum(dim=1).clamp(min=1.0)  # (B, 1)
        return h_sum / mask_sum


class DualEncoderTransformer(nn.Module):
    """
    Dual-encoder transformer for edge prediction.

    Takes both:
    - Market sequence: trades in the current market (causal attention)
    - User sequence: user's historical trades across markets (bidirectional)

    Outputs per-position edge predictions.

    Args:
        d_market_input: Features per market trade (default 10)
        d_user_input: Features per user historical trade (default 4)
        d_model: Hidden dimension (default 128)
        n_heads: Attention heads (default 4)
        n_market_layers: Market encoder depth (default 4)
        n_user_layers: User encoder depth (default 2)
        d_ff: Feed-forward dimension (default 512)
        max_market_len: Max market sequence length (default 1024)
        max_user_len: Max user history length (default 64)
        dropout: Dropout probability (default 0.1)
    """

    def __init__(
        self,
        d_market_input: int = 10,
        d_user_input: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_market_layers: int = 4,
        n_user_layers: int = 2,
        d_ff: int = 512,
        max_market_len: int = 1024,
        max_user_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_market_len = max_market_len
        self.max_user_len = max_user_len

        # Market encoder (causal - can only see past trades in this market)
        self.market_encoder = TransformerEncoder(
            d_input=d_market_input,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_market_layers,
            d_ff=d_ff,
            max_seq_len=max_market_len,
            dropout=dropout,
            causal=True,
        )

        # User encoder (bidirectional - user's full history is known)
        self.user_encoder = TransformerEncoder(
            d_input=d_user_input,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_user_layers,
            d_ff=d_ff,
            max_seq_len=max_user_len,
            dropout=dropout,
            causal=False,  # Bidirectional for user history
        )

        # Fusion: combine market context (per-position) + user context (pooled)
        # Market encoder outputs (B, L_market, d_model) per position
        # User encoder outputs (B, d_model) after pooling
        # We concatenate user context to each market position

        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        # Output head (per-position prediction)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_fusion_weights()

    def _init_fusion_weights(self):
        for m in [self.fusion_proj, self.output_head]:
            for name, p in m.named_parameters():
                if "weight" in name and p.dim() >= 2:
                    nn.init.xavier_uniform_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)

    def forward(
        self,
        market_x: torch.Tensor,     # (B, L_market, D_market)
        market_mask: torch.Tensor,  # (B, L_market) bool
        user_x: torch.Tensor,       # (B, L_user, D_user)
        user_mask: torch.Tensor,    # (B, L_user) bool
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            market_x: Market trade features
            market_mask: Valid market positions
            user_x: User historical trade features
            user_mask: Valid user history positions

        Returns:
            logits: (B, L_market) per-position edge predictions
        """
        B, L_market, _ = market_x.shape

        # Encode market sequence (causal, per-position outputs)
        market_h = self.market_encoder(market_x, market_mask)  # (B, L_market, d_model)

        # Encode user history and pool to single vector
        user_h = self.user_encoder(user_x, user_mask)  # (B, L_user, d_model)
        user_ctx = self.user_encoder.pool(user_h, user_mask)  # (B, d_model)

        # Expand user context to match market sequence length
        user_ctx_expanded = user_ctx.unsqueeze(1).expand(B, L_market, -1)  # (B, L_market, d_model)

        # Fuse: concatenate market (per-position) with user (broadcasted)
        fused = torch.cat([market_h, user_ctx_expanded], dim=-1)  # (B, L_market, d_model*2)
        fused = self.fusion_proj(fused)  # (B, L_market, d_model)
        fused = F.gelu(fused)

        # Output predictions
        logits = self.output_head(fused).squeeze(-1)  # (B, L_market)

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DualEncoderWithCrossAttention(nn.Module):
    """
    Alternative: Use cross-attention instead of simple concatenation.

    The market encoder output queries attend to user history context.
    This allows more dynamic interaction between market state and user profile.
    """

    def __init__(
        self,
        d_market_input: int = 10,
        d_user_input: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_market_layers: int = 4,
        n_user_layers: int = 2,
        n_cross_layers: int = 2,
        d_ff: int = 512,
        max_market_len: int = 1024,
        max_user_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Encoders (same as DualEncoderTransformer)
        self.market_encoder = TransformerEncoder(
            d_input=d_market_input,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_market_layers,
            d_ff=d_ff,
            max_seq_len=max_market_len,
            dropout=dropout,
            causal=True,
        )

        self.user_encoder = TransformerEncoder(
            d_input=d_user_input,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_user_layers,
            d_ff=d_ff,
            max_seq_len=max_user_len,
            dropout=dropout,
            causal=False,
        )

        # Cross-attention layers: market queries attend to user keys/values
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_cross_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        market_x: torch.Tensor,
        market_mask: torch.Tensor,
        user_x: torch.Tensor,
        user_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L_market, _ = market_x.shape

        # Encode both sequences
        market_h = self.market_encoder(market_x, market_mask)  # (B, L_market, d_model)
        user_h = self.user_encoder(user_x, user_mask)  # (B, L_user, d_model)

        # Cross attention: market attends to user
        # tgt = market_h (queries), memory = user_h (keys/values)
        memory_key_padding_mask = ~user_mask  # True = ignore

        h = market_h
        for layer in self.cross_attn_layers:
            h = layer(
                tgt=h,
                memory=user_h,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        h = self.final_norm(h)

        logits = self.output_head(h).squeeze(-1)  # (B, L_market)

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================
# Factory functions
# ===========================

def create_small_dual_encoder(**kwargs) -> DualEncoderTransformer:
    """Small dual encoder (~2.5M parameters)."""
    defaults = dict(
        d_market_input=10,
        d_user_input=4,
        d_model=128,
        n_heads=4,
        n_market_layers=4,
        n_user_layers=2,
        d_ff=512,
        max_market_len=1024,
        max_user_len=64,
        dropout=0.1,
    )
    defaults.update(kwargs)
    return DualEncoderTransformer(**defaults)


def create_medium_dual_encoder(**kwargs) -> DualEncoderTransformer:
    """Medium dual encoder (~5M parameters) - more expressive than small."""
    defaults = dict(
        d_market_input=10,
        d_user_input=4,
        d_model=192,
        n_heads=6,
        n_market_layers=6,
        n_user_layers=3,
        d_ff=768,
        max_market_len=1024,
        max_user_len=64,
        dropout=0.1,
    )
    defaults.update(kwargs)
    return DualEncoderTransformer(**defaults)


def create_base_dual_encoder(**kwargs) -> DualEncoderTransformer:
    """Base dual encoder (~8M parameters)."""
    defaults = dict(
        d_market_input=10,
        d_user_input=4,
        d_model=256,
        n_heads=8,
        n_market_layers=6,
        n_user_layers=3,
        d_ff=1024,
        max_market_len=1024,
        max_user_len=64,
        dropout=0.1,
    )
    defaults.update(kwargs)
    return DualEncoderTransformer(**defaults)


def create_small_dual_encoder_cross_attn(**kwargs) -> DualEncoderWithCrossAttention:
    """Small dual encoder with cross-attention (~3M parameters)."""
    defaults = dict(
        d_market_input=10,
        d_user_input=4,
        d_model=128,
        n_heads=4,
        n_market_layers=4,
        n_user_layers=2,
        n_cross_layers=2,
        d_ff=512,
        max_market_len=1024,
        max_user_len=64,
        dropout=0.1,
    )
    defaults.update(kwargs)
    return DualEncoderWithCrossAttention(**defaults)


def create_medium_dual_encoder_cross_attn(**kwargs) -> DualEncoderWithCrossAttention:
    """Medium dual encoder with cross-attention (~6M parameters).

    More expressive with:
    - Wider model (d_model=192)
    - More market layers (6)
    - More cross-attention layers (4)
    """
    defaults = dict(
        d_market_input=10,
        d_user_input=4,
        d_model=192,
        n_heads=6,
        n_market_layers=6,
        n_user_layers=3,
        n_cross_layers=4,
        d_ff=768,
        max_market_len=1024,
        max_user_len=64,
        dropout=0.1,
    )
    defaults.update(kwargs)
    return DualEncoderWithCrossAttention(**defaults)


if __name__ == "__main__":
    print("=== DualEncoderTransformer (concat fusion) ===")
    model = create_small_dual_encoder()
    print(f"Parameters: {model.count_parameters():,}")

    B, L_market, L_user = 4, 512, 64
    market_x = torch.randn(B, L_market, 10)
    market_mask = torch.ones(B, L_market, dtype=torch.bool)
    market_mask[:, :100] = False  # Left-pad: first 100 are padding

    user_x = torch.randn(B, L_user, 4)
    user_mask = torch.ones(B, L_user, dtype=torch.bool)
    user_mask[:, :10] = False  # Left-pad: first 10 are padding

    logits = model(market_x, market_mask, user_x, user_mask)
    print(f"Output shape: {logits.shape}")  # (B, L_market)
    print(f"NaN in output: {torch.isnan(logits).sum().item()}")

    print("\n=== DualEncoderWithCrossAttention ===")
    model_cross = create_small_dual_encoder_cross_attn()
    print(f"Parameters: {model_cross.count_parameters():,}")

    logits_cross = model_cross(market_x, market_mask, user_x, user_mask)
    print(f"Output shape: {logits_cross.shape}")
    print(f"NaN in output: {torch.isnan(logits_cross).sum().item()}")
