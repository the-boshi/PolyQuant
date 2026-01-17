"""
Transformer model for market outcome prediction.

Two modes:
1. MarketTransformer: Sequence-to-label (mean pooling) for market outcome
2. TradeTransformer: Per-token prediction for edge estimation (causal)

Simplest architecture choices:
- Learned positional embeddings
- GELU activation, Pre-LN
- Small model size (~1.5M parameters)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarketTransformer(nn.Module):
    """
    Transformer encoder for binary market outcome prediction.

    Args:
        d_input: Number of continuous input features (default 10)
        d_model: Model hidden dimension (default 128)
        d_user: User embedding dimension (default 32)
        n_heads: Number of attention heads (default 4)
        n_layers: Number of transformer encoder layers (default 4)
        d_ff: Feed-forward hidden dimension (default 512)
        max_seq_len: Maximum sequence length (default 512)
        user_vocab: Size of user hash vocabulary (default 50_000)
        dropout: Dropout probability (default 0.1)
    """

    def __init__(
        self,
        d_input: int = 10,
        d_model: int = 128,
        d_user: int = 16,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 512,
        user_vocab: int = 50_000,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Input projection: continuous features -> d_model
        self.input_proj = nn.Linear(d_input, d_model)

        # User embedding (smaller vocab with hash collisions acceptable)
        self.user_embed = nn.Embedding(user_vocab, d_user, padding_idx=0)

        # Combine features + user embedding -> d_model
        self.combine_proj = nn.Linear(d_model + d_user, d_model)

        # Learned positional embeddings
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Dropout after embeddings
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Classification head
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

        # Smaller init for embeddings
        nn.init.normal_(self.user_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

        # Zero out padding embedding
        with torch.no_grad():
            self.user_embed.weight[0].zero_()

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Continuous features, shape (B, L, D)
            u: User hash IDs, shape (B, L)
            mask: Boolean mask, True for valid tokens, shape (B, L)

        Returns:
            logits: Shape (B,) - raw logits for binary classification
        """
        B, L, _ = x.shape
        device = x.device

        # Project continuous features
        h = self.input_proj(x)  # (B, L, d_model)

        # Get user embeddings (apply modulo to fit vocab size)
        u_clamped = u % self.user_embed.num_embeddings
        u_embed = self.user_embed(u_clamped)  # (B, L, d_user)

        # Combine and project
        h = torch.cat([h, u_embed], dim=-1)  # (B, L, d_model + d_user)
        h = self.combine_proj(h)  # (B, L, d_model)

        # Add positional embeddings
        positions = torch.arange(L, device=device)  # (L,)
        pos_embed = self.pos_embed(positions)  # (L, d_model)
        h = h + pos_embed.unsqueeze(0)  # (B, L, d_model)

        # Dropout
        h = self.embed_dropout(h)

        # Create attention mask for transformer
        # PyTorch TransformerEncoder expects src_key_padding_mask where True = ignore
        src_key_padding_mask = ~mask  # (B, L), True where padded

        # Apply transformer encoder
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)  # (B, L, d_model)

        # Final layer norm
        h = self.final_norm(h)

        # Mean pooling over non-padded tokens
        # mask: (B, L) -> (B, L, 1)
        mask_expanded = mask.unsqueeze(-1).float()  # (B, L, 1)
        h_masked = h * mask_expanded  # Zero out padded positions
        h_sum = h_masked.sum(dim=1)  # (B, d_model)
        mask_sum = mask_expanded.sum(dim=1).clamp(min=1.0)  # (B, 1), avoid div by zero
        h_pooled = h_sum / mask_sum  # (B, d_model)

        # Classification
        logits = self.classifier(h_pooled).squeeze(-1)  # (B,)

        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_no_embed(self) -> int:
        """Count trainable parameters excluding user embeddings."""
        return sum(
            p.numel() for name, p in self.named_parameters()
            if p.requires_grad and "user_embed" not in name
        )


class MarketTransformerNoUser(nn.Module):
    """
    Transformer encoder for binary market outcome prediction without user embeddings.

    Args:
        d_input: Number of continuous input features (default 10)
        d_model: Model hidden dimension (default 128)
        n_heads: Number of attention heads (default 4)
        n_layers: Number of transformer encoder layers (default 4)
        d_ff: Feed-forward hidden dimension (default 512)
        max_seq_len: Maximum sequence length (default 512)
        dropout: Dropout probability (default 0.1)
    """

    def __init__(
        self,
        d_input: int = 10,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Input projection: continuous features -> d_model
        self.input_proj = nn.Linear(d_input, d_model)

        # Learned positional embeddings
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Dropout after embeddings
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder layers
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

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Continuous features, shape (B, L, D)
            mask: Boolean mask, True for valid tokens, shape (B, L)

        Returns:
            logits: Shape (B,) - raw logits for binary classification
        """
        B, L, _ = x.shape
        device = x.device

        # Project continuous features
        h = self.input_proj(x)  # (B, L, d_model)

        # Add positional embeddings
        positions = torch.arange(L, device=device)  # (L,)
        pos_embed = self.pos_embed(positions)  # (L, d_model)
        h = h + pos_embed.unsqueeze(0)  # (B, L, d_model)

        # Dropout
        h = self.embed_dropout(h)

        # Padding mask: True = ignore
        src_key_padding_mask = ~mask  # (B, L)

        # Apply transformer encoder
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)  # (B, L, d_model)

        # Final layer norm
        h = self.final_norm(h)

        # Mean pooling over non-padded tokens
        mask_expanded = mask.unsqueeze(-1).float()  # (B, L, 1)
        h_masked = h * mask_expanded
        h_sum = h_masked.sum(dim=1)  # (B, d_model)
        mask_sum = mask_expanded.sum(dim=1).clamp(min=1.0)  # (B, 1)
        h_pooled = h_sum / mask_sum

        # Classification
        logits = self.classifier(h_pooled).squeeze(-1)  # (B,)

        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TradeTransformer(nn.Module):
    """
    Transformer for per-trade edge prediction.

    Uses causal (autoregressive) attention so each trade only sees past trades.
    Outputs a probability for each token position.

    Args:
        d_input: Number of continuous input features (default 10)
        d_model: Model hidden dimension (default 128)
        d_user: User embedding dimension (default 16)
        n_heads: Number of attention heads (default 4)
        n_layers: Number of transformer encoder layers (default 4)
        d_ff: Feed-forward hidden dimension (default 512)
        max_seq_len: Maximum sequence length (default 512)
        user_vocab: Size of user hash vocabulary (default 50_000)
        dropout: Dropout probability (default 0.1)
    """

    def __init__(
        self,
        d_input: int = 10,
        d_model: int = 128,
        d_user: int = 16,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 512,
        user_vocab: int = 50_000,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Input projection: continuous features -> d_model
        self.input_proj = nn.Linear(d_input, d_model)

        # User embedding (smaller vocab with hash collisions acceptable)
        self.user_embed = nn.Embedding(user_vocab, d_user, padding_idx=0)

        # Combine features + user embedding -> d_model
        self.combine_proj = nn.Linear(d_model + d_user, d_model)

        # Learned positional embeddings
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Dropout after embeddings
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder layers
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

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Per-token classification head (outputs logit per position)
        self.token_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Register causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1),
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

        # Smaller init for embeddings
        nn.init.normal_(self.user_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

        # Zero out padding embedding
        with torch.no_grad():
            self.user_embed.weight[0].zero_()

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with causal attention.

        Args:
            x: Continuous features, shape (B, L, D)
            u: User hash IDs, shape (B, L)
            mask: Boolean mask, True for valid tokens, shape (B, L)

        Returns:
            logits: Shape (B, L) - raw logits per token for binary classification
        """
        B, L, _ = x.shape
        device = x.device

        # Project continuous features
        h = self.input_proj(x)  # (B, L, d_model)

        # Get user embeddings (apply modulo to fit vocab size)
        u_clamped = u % self.user_embed.num_embeddings
        u_embed = self.user_embed(u_clamped)  # (B, L, d_user)

        # Combine and project
        h = torch.cat([h, u_embed], dim=-1)  # (B, L, d_model + d_user)
        h = self.combine_proj(h)  # (B, L, d_model)

        # Add positional embeddings
        positions = torch.arange(L, device=device)  # (L,)
        pos_embed = self.pos_embed(positions)  # (L, d_model)
        h = h + pos_embed.unsqueeze(0)  # (B, L, d_model)

        # Dropout
        h = self.embed_dropout(h)

        # Causal mask: prevent attending to future tokens
        # Shape (L, L), True = masked (cannot attend)
        causal_mask = self.causal_mask[:L, :L]

        # Padding mask: True = ignore this position
        src_key_padding_mask = ~mask  # (B, L)

        # Apply transformer encoder with causal attention
        h = self.encoder(h, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)

        # Final layer norm
        h = self.final_norm(h)

        # Per-token classification
        logits = self.token_classifier(h).squeeze(-1)  # (B, L)

        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_no_embed(self) -> int:
        """Count trainable parameters excluding user embeddings."""
        return sum(
            p.numel() for name, p in self.named_parameters()
            if p.requires_grad and "user_embed" not in name
        )


def create_small_transformer(**kwargs) -> MarketTransformer:
    """Create a small transformer (~1.7M parameters total)."""
    defaults = dict(
        d_input=10,
        d_model=128,
        d_user=16,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=512,
        user_vocab=50_000,
        dropout=0.1,
    )
    defaults.update(kwargs)
    return MarketTransformer(**defaults)


def create_small_transformer_no_user(**kwargs) -> MarketTransformerNoUser:
    """Create a small transformer without user embeddings."""
    defaults = dict(
        d_input=10,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=512,
        dropout=0.1,
    )
    defaults.update(kwargs)
    return MarketTransformerNoUser(**defaults)


def create_base_transformer_no_user(**kwargs) -> MarketTransformerNoUser:
    """Create a base transformer without user embeddings."""
    defaults = dict(
        d_input=10,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1,
    )
    defaults.update(kwargs)
    return MarketTransformerNoUser(**defaults)


def create_base_transformer(**kwargs) -> MarketTransformer:
    """Create a base transformer (~8M parameters total)."""
    defaults = dict(
        d_input=10,
        d_model=256,
        d_user=32,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=512,
        user_vocab=100_000,
        dropout=0.1,
    )
    defaults.update(kwargs)
    return MarketTransformer(**defaults)


def create_small_trade_transformer(**kwargs) -> TradeTransformer:
    """Create a small trade transformer for per-token prediction."""
    defaults = dict(
        d_input=10,
        d_model=128,
        d_user=16,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=512,
        user_vocab=50_000,
        dropout=0.1,
    )
    defaults.update(kwargs)
    return TradeTransformer(**defaults)


def create_base_trade_transformer(**kwargs) -> TradeTransformer:
    """Create a base trade transformer for per-token prediction."""
    defaults = dict(
        d_input=10,
        d_model=256,
        d_user=32,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        max_seq_len=512,
        user_vocab=100_000,
        dropout=0.1,
    )
    defaults.update(kwargs)
    return TradeTransformer(**defaults)


if __name__ == "__main__":
    # Quick test for MarketTransformer
    print("=== MarketTransformer ===")
    model = create_small_transformer()
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Parameters (excl. user embed): {model.count_parameters_no_embed():,}")

    B, L, D = 4, 512, 10
    x = torch.randn(B, L, D)
    u = torch.randint(0, 1000, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[:, -100:] = False  # Last 100 tokens are padding

    logits = model(x, u, mask)
    print(f"Input: x={x.shape}, u={u.shape}, mask={mask.shape}")
    print(f"Output: logits={logits.shape}")

    # Quick test for TradeTransformer
    print("\n=== TradeTransformer ===")
    trade_model = create_small_trade_transformer()
    print(f"Total parameters: {trade_model.count_parameters():,}")

    logits = trade_model(x, u, mask)
    print(f"Input: x={x.shape}, u={u.shape}, mask={mask.shape}")
    print(f"Output: logits={logits.shape}")  # Should be (B, L)
    print(f"Output: logits={logits.shape}")
    print(f"Logits: {logits}")
