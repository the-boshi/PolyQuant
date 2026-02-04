from __future__ import annotations

import torch
import torch.nn as nn


class MarketTransformerNoUser(nn.Module):
    """
    Transformer encoder for binary market outcome prediction (no user embeddings).

    Args:
        d_input: Number of continuous input features (default 10)
        d_model: Model hidden dimension (default 256)
        n_heads: Number of attention heads (default 8)
        n_layers: Number of transformer encoder layers (default 6)
        d_ff: Feed-forward hidden dimension (default 1024)
        max_seq_len: Maximum sequence length (default 512)
        dropout: Dropout probability (default 0.1)
    """

    def __init__(
        self,
        d_input: int = 10,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
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

        # Transformer encoder (Pre-LN)
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
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
        h = h + self.pos_embed(positions).unsqueeze(0)  # (B, L, d_model)

        # Dropout
        h = self.embed_dropout(h)

        # Padding mask: True = ignore
        src_key_padding_mask = ~mask  # (B, L)

        # Apply transformer encoder
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)  # (B, L, d_model)

        # Final layer norm
        h = self.final_norm(h)

        # Mean pooling over non-padded tokens
        mask_f = mask.unsqueeze(-1).float()  # (B, L, 1)
        h_sum = (h * mask_f).sum(dim=1)      # (B, d_model)
        denom = mask_f.sum(dim=1).clamp(min=1.0)  # (B, 1)
        h_pooled = h_sum / denom             # (B, d_model)

        # Classification
        logits = self.classifier(h_pooled).squeeze(-1)  # (B,)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_base_transformer_no_user(**kwargs) -> MarketTransformerNoUser:
    """Create the base transformer without user embeddings."""
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


if __name__ == "__main__":
    model = create_base_transformer_no_user()
    print(f"Total parameters: {model.count_parameters():,}")

    B, L, D = 4, 512, 10
    x = torch.randn(B, L, D)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[:, -100:] = False  # last 100 tokens are padding

    logits = model(x, mask)
    print(f"Input: x={x.shape}, mask={mask.shape}")
    print(f"Output: logits={logits.shape}")
