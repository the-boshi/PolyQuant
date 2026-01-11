from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """
    A 1D residual block for tabular data:
    Input -> Conv1D -> LayerNorm -> GELU -> Conv1D -> LayerNorm -> Add Input -> GELU
    """
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.ln1 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.ln2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        residual = x
        out = self.conv1(x)
        out = out.transpose(1, 2)  # [B, L, C] for LayerNorm
        out = self.ln1(out)
        out = out.transpose(1, 2)  # [B, C, L]
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.ln2(out)
        out = out.transpose(1, 2)

        out = out + residual  # Skip connection
        out = self.act(out)
        return out


class ResNet1D(nn.Module):
    """
    1D ResNet for tabular data.

    Treats the feature vector as a 1D sequence and applies 1D convolutions
    with residual connections. This can capture local feature interactions.

    Args:
        in_dim: Number of input features
        hidden_channels: Number of channels in hidden layers (default: 128)
        num_blocks: Number of residual blocks (default: 3)
        kernel_size: Kernel size for convolutions (default: 3)
        dropout: Dropout rate (default: 0.1)
    """
    def __init__(
        self,
        in_dim: int,
        hidden_channels: int = 128,
        num_blocks: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input embedding: project each feature to hidden_channels dimensions
        # This creates a sequence of length in_dim with hidden_channels features each
        self.input_proj = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stack of residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock1D(hidden_channels, kernel_size, dropout)
            for _ in range(num_blocks)
        ])

        # Global pooling and classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, in_dim]
        Returns:
            Logits of shape [B]
        """
        # x: [B, in_dim] -> [B, in_dim, 1]
        x = x.unsqueeze(-1)

        # Project each feature: [B, in_dim, 1] -> [B, in_dim, hidden_channels]
        x = self.input_proj(x)

        # Transpose for Conv1d: [B, in_dim, hidden_channels] -> [B, hidden_channels, in_dim]
        x = x.transpose(1, 2)

        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Global pooling: [B, hidden_channels, in_dim] -> [B, hidden_channels, 1]
        x = self.pool(x)

        # Classifier: [B, hidden_channels, 1] -> [B]
        x = self.classifier(x)

        return x.squeeze(-1)


class ResNetMLP(nn.Module):
    """
    A hybrid ResNet-style MLP that uses residual connections in a fully-connected architecture.

    This is an alternative approach that applies residual learning to MLPs,
    which can be more effective for purely tabular data.

    Args:
        in_dim: Number of input features
        hidden: Tuple of hidden layer dimensions (default: (256, 256, 128))
        dropout: Dropout rate (default: 0.1)
    """
    def __init__(
        self,
        in_dim: int,
        hidden: tuple = (256, 256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection to first hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.LayerNorm(hidden[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i, h in enumerate(hidden):
            if i == 0:
                continue
            prev_h = hidden[i - 1]
            self.res_blocks.append(
                ResidualMLPBlock(prev_h, h, dropout)
            )

        # Output head
        self.head = nn.Linear(hidden[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, in_dim]
        Returns:
            Logits of shape [B]
        """
        x = self.input_proj(x)

        for block in self.res_blocks:
            x = block(x)

        return self.head(x).squeeze(-1)


class ResidualMLPBlock(nn.Module):
    """
    A residual block for MLP with optional dimension change.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.ln1 = nn.LayerNorm(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.ln2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        # Projection for residual if dimensions differ
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.fc1(x)
        out = self.ln1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.ln2(out)

        out = out + residual
        out = self.act(out)
        return out