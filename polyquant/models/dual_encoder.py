"""
Dual Encoder Transformer for trade outcome prediction.

Encodes both market sequence (recent trades in the market) and user sequence
(historical trades by the user) with cross-attention between them.

Architecture:
- Market Encoder: Transformer with causal attention
- User Encoder: Transformer with bidirectional attention
- Cross-Attention: Market <-> User interaction
- Fusion: Concatenate + project
- Classification: Binary outcome prediction
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention module for market-user interaction.

    Performs:
    1. Market -> User: Query from market (last token), K/V from user
    2. User -> Market: Query from user (pooled), K/V from market
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Market attending to User
        self.market_to_user = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_m2u = nn.LayerNorm(d_model)

        # User attending to Market
        self.user_to_market = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_u2m = nn.LayerNorm(d_model)

        # Combine the two cross-attention outputs
        self.combine = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h_market: torch.Tensor,      # (B, L_market, d_model)
        h_user: torch.Tensor,        # (B, L_user, d_model)
        market_mask: torch.Tensor,   # (B, L_market) - True for valid
        user_mask: torch.Tensor,     # (B, L_user) - True for valid
    ) -> torch.Tensor:
        """
        Returns:
            h_cross: (B, d_model) - fused cross-attention output
        """
        B = h_market.shape[0]
        device = h_market.device

        # Get last valid token index for market
        # market_mask: (B, L_market)
        last_idx = market_mask.sum(dim=1) - 1  # (B,)
        last_idx = last_idx.clamp(min=0)

        # Extract last market token: (B, 1, d_model)
        batch_idx = torch.arange(B, device=device)
        h_market_last = h_market[batch_idx, last_idx].unsqueeze(1)  # (B, 1, d_model)

        # Mean pool user sequence: (B, d_model)
        user_mask_exp = user_mask.unsqueeze(-1).float()  # (B, L_user, 1)
        h_user_masked = h_user * user_mask_exp
        h_user_pooled = h_user_masked.sum(dim=1) / user_mask_exp.sum(dim=1).clamp(min=1.0)
        h_user_pooled = h_user_pooled.unsqueeze(1)  # (B, 1, d_model)

        # Cross-attention masks (True = ignore for MultiheadAttention)
        user_key_mask = ~user_mask     # (B, L_user)
        market_key_mask = ~market_mask  # (B, L_market)

        # Market -> User cross-attention
        # Query: last market token, Key/Value: all user tokens
        h_m2u, _ = self.market_to_user(
            query=self.norm_m2u(h_market_last),
            key=h_user,
            value=h_user,
            key_padding_mask=user_key_mask,
            need_weights=False,
        )  # (B, 1, d_model)
        h_m2u = h_market_last + h_m2u  # Residual

        # User -> Market cross-attention
        # Query: pooled user, Key/Value: all market tokens
        h_u2m, _ = self.user_to_market(
            query=self.norm_u2m(h_user_pooled),
            key=h_market,
            value=h_market,
            key_padding_mask=market_key_mask,
            need_weights=False,
        )  # (B, 1, d_model)
        h_u2m = h_user_pooled + h_u2m  # Residual

        # Squeeze and combine
        h_m2u = h_m2u.squeeze(1)  # (B, d_model)
        h_u2m = h_u2m.squeeze(1)  # (B, d_model)

        h_cross = self.combine(torch.cat([h_m2u, h_u2m], dim=-1))  # (B, d_model)

        return h_cross


class DualEncoderTransformer(nn.Module):
    """
    Dual Encoder Transformer for trade outcome prediction.

    Encodes market sequence (with causal attention) and user sequence
    (with bidirectional attention), then fuses them with cross-attention.

    Args:
        d_market_input: Market sequence feature dimension (default 12)
        d_user_input: User sequence feature dimension (default 4)
        d_model: Hidden dimension (default 128)
        n_market_layers: Number of market encoder layers (default 4)
        n_user_layers: Number of user encoder layers (default 2)
        n_heads: Number of attention heads (default 4)
        d_ff: Feed-forward dimension (default 512)
        max_market_len: Maximum market sequence length (default 1024)
        max_user_len: Maximum user sequence length (default 128)
        dropout: Dropout probability (default 0.1)
    """

    def __init__(
        self,
        d_market_input: int = 12,
        d_user_input: int = 4,
        d_model: int = 128,
        n_market_layers: int = 4,
        n_user_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 512,
        max_market_len: int = 1024,
        max_user_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_market_len = max_market_len
        self.max_user_len = max_user_len

        # === Market Encoder ===
        self.market_input_proj = nn.Linear(d_market_input, d_model)
        self.market_pos_embed = nn.Embedding(max_market_len, d_model)
        self.market_embed_dropout = nn.Dropout(dropout)

        market_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.market_encoder = nn.TransformerEncoder(
            market_encoder_layer,
            num_layers=n_market_layers
        )
        self.market_final_norm = nn.LayerNorm(d_model)

        # Register causal mask buffer for market encoder
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_market_len, max_market_len, dtype=torch.bool), diagonal=1),
        )

        # === User Encoder ===
        self.user_input_proj = nn.Linear(d_user_input, d_model)
        self.user_pos_embed = nn.Embedding(max_user_len, d_model)
        self.user_embed_dropout = nn.Dropout(dropout)

        user_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.user_encoder = nn.TransformerEncoder(
            user_encoder_layer,
            num_layers=n_user_layers
        )
        self.user_final_norm = nn.LayerNorm(d_model)

        # === Cross-Attention ===
        self.cross_attention = CrossAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # === Fusion Layer ===
        # Combines: h_market_last, h_user_pooled, h_cross
        self.fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # === Classification Head ===
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

        # Smaller init for positional embeddings
        nn.init.normal_(self.market_pos_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.user_pos_embed.weight, mean=0.0, std=0.02)

    def encode_market(
        self,
        x: torch.Tensor,      # (B, L_market, D_market)
        mask: torch.Tensor,   # (B, L_market) - True for valid
    ) -> torch.Tensor:
        """
        Encode market sequence with causal attention.

        Returns:
            h: (B, L_market, d_model)
        """
        B, L, _ = x.shape
        device = x.device

        # Project input
        h = self.market_input_proj(x)  # (B, L, d_model)

        # Add positional embeddings
        positions = torch.arange(L, device=device)
        pos_embed = self.market_pos_embed(positions)  # (L, d_model)
        h = h + pos_embed.unsqueeze(0)

        # Dropout
        h = self.market_embed_dropout(h)

        # Causal mask for transformer
        causal_mask = self.causal_mask[:L, :L]  # (L, L)

        # Padding mask (True = ignore)
        padding_mask = ~mask  # (B, L)

        # Encode
        h = self.market_encoder(h, mask=causal_mask, src_key_padding_mask=padding_mask)
        h = self.market_final_norm(h)

        return h

    def encode_user(
        self,
        x: torch.Tensor,      # (B, L_user, D_user)
        mask: torch.Tensor,   # (B, L_user) - True for valid
    ) -> torch.Tensor:
        """
        Encode user sequence with bidirectional attention.

        Returns:
            h: (B, L_user, d_model)
        """
        B, L, _ = x.shape
        device = x.device

        # Project input
        h = self.user_input_proj(x)  # (B, L, d_model)

        # Add positional embeddings
        positions = torch.arange(L, device=device)
        pos_embed = self.user_pos_embed(positions)  # (L, d_model)
        h = h + pos_embed.unsqueeze(0)

        # Dropout
        h = self.user_embed_dropout(h)

        # Padding mask (True = ignore)
        padding_mask = ~mask  # (B, L)

        # Encode (no causal mask - bidirectional)
        h = self.user_encoder(h, src_key_padding_mask=padding_mask)
        h = self.user_final_norm(h)

        return h

    def forward(
        self,
        market_x: torch.Tensor,      # (B, L_market, D_market)
        market_mask: torch.Tensor,   # (B, L_market)
        user_x: torch.Tensor,        # (B, L_user, D_user)
        user_mask: torch.Tensor,     # (B, L_user)
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            market_x: Market sequence features (B, L_market, D_market)
            market_mask: Market valid mask, True for valid (B, L_market)
            user_x: User sequence features (B, L_user, D_user)
            user_mask: User valid mask, True for valid (B, L_user)

        Returns:
            logits: Shape (B,) - raw logits for binary classification
        """
        B = market_x.shape[0]
        device = market_x.device

        # Encode both sequences
        h_market = self.encode_market(market_x, market_mask)  # (B, L_market, d_model)
        h_user = self.encode_user(user_x, user_mask)          # (B, L_user, d_model)

        # Cross-attention
        h_cross = self.cross_attention(h_market, h_user, market_mask, user_mask)  # (B, d_model)

        # Extract representations for fusion
        # Market: last valid token
        last_idx = market_mask.sum(dim=1) - 1  # (B,)
        last_idx = last_idx.clamp(min=0)
        batch_idx = torch.arange(B, device=device)
        h_market_last = h_market[batch_idx, last_idx]  # (B, d_model)

        # User: mean pool
        user_mask_exp = user_mask.unsqueeze(-1).float()  # (B, L_user, 1)
        h_user_masked = h_user * user_mask_exp
        h_user_pooled = h_user_masked.sum(dim=1) / user_mask_exp.sum(dim=1).clamp(min=1.0)  # (B, d_model)

        # Fuse all representations
        h_fused = self.fusion(torch.cat([h_market_last, h_user_pooled, h_cross], dim=-1))  # (B, d_model)

        # Classify
        logits = self.classifier(h_fused).squeeze(-1)  # (B,)

        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_small_dual_encoder(**kwargs) -> DualEncoderTransformer:
    """Create a small dual encoder (~3M parameters)."""
    defaults = dict(
        d_market_input=12,
        d_user_input=4,
        d_model=128,
        n_market_layers=4,
        n_user_layers=2,
        n_heads=4,
        d_ff=512,
        max_market_len=1024,
        max_user_len=128,
        dropout=0.1,
    )
    defaults.update(kwargs)
    return DualEncoderTransformer(**defaults)


def create_base_dual_encoder(**kwargs) -> DualEncoderTransformer:
    """Create a base dual encoder (~12M parameters)."""
    defaults = dict(
        d_market_input=12,
        d_user_input=4,
        d_model=256,
        n_market_layers=6,
        n_user_layers=4,
        n_heads=8,
        d_ff=1024,
        max_market_len=1024,
        max_user_len=128,
        dropout=0.1,
    )
    defaults.update(kwargs)
    return DualEncoderTransformer(**defaults)


if __name__ == "__main__":
    # Quick test
    print("=== DualEncoderTransformer ===")

    model = create_small_dual_encoder()
    print(f"Total parameters: {model.count_parameters():,}")

    # Test inputs
    B = 4
    L_market, D_market = 1024, 12
    L_user, D_user = 128, 4

    market_x = torch.randn(B, L_market, D_market)
    market_mask = torch.ones(B, L_market, dtype=torch.bool)
    market_mask[:, -100:] = False  # Last 100 are padding

    user_x = torch.randn(B, L_user, D_user)
    user_mask = torch.ones(B, L_user, dtype=torch.bool)
    user_mask[:, -20:] = False  # Last 20 are padding

    logits = model(market_x, market_mask, user_x, user_mask)

    print(f"Market input: {market_x.shape}")
    print(f"User input: {user_x.shape}")
    print(f"Output logits: {logits.shape}")
    print(f"Logits: {logits}")

    # Test base model
    print("\n=== Base DualEncoderTransformer ===")
    model_base = create_base_dual_encoder()
    print(f"Total parameters: {model_base.count_parameters():,}")
