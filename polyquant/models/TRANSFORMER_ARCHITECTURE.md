# Transformer Architecture for Market Outcome Prediction

## Overview

A sequence-to-label transformer that processes a variable-length sequence of trades within a prediction market and outputs a binary classification (market resolves YES=1 or NO=0).

---

## Input Format

From the `MarketWindowDataset`:

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `x` | `(B, L, D)` | float32 | Trade features (D=10) |
| `u` | `(B, L)` | int64 | User hash IDs for embedding lookup |
| `mask` | `(B, L)` | bool | True for real tokens, False for padding |
| `y` | `(B,)` | int64 | Binary label (0 or 1) |

Where:
- `B` = batch size
- `L` = max sequence length (default 512)
- `D` = 10 continuous features

### Feature Columns (in order)

1. `p_yes` — current YES probability [0, 1]
2. `dp_yes_clip` — clipped price change [-0.2, 0.2]
3. `log_dt` — log(1 + time since last trade)
4. `log_usdc_size` — log trade size in USDC
5. `user_recent_pnl_asinh` — asinh-transformed recent PnL
6. `user_avg_size_log` — log average trade size
7. `user_days_active_log` — log days active
8. `user_hist_pnl_asinh` — asinh-transformed historical PnL
9. `user_hist_winrate` — historical win rate [0, 1]
10. `user_pnl_std_log` — log PnL standard deviation

### User Embeddings

- `user_hash` values are in `[0, USER_VOCAB)` where `USER_VOCAB = 2,000,000`
- Each token has an associated user who made that trade
- User embedding provides learned representation of trader behavior

---

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  x: (B, L, 10)  ──► Linear(10, d_model) ──► (B, L, d_model)     │
│  u: (B, L)      ──► Embedding(2M, d_user) ──► (B, L, d_user)    │
│                                                                  │
│  Concat + Project: (B, L, d_model + d_user) ──► (B, L, d_model) │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POSITIONAL ENCODING                           │
├─────────────────────────────────────────────────────────────────┤
│  Option A: Learned positional embeddings (L, d_model)           │
│  Option B: Sinusoidal encoding                                   │
│  Option C: RoPE (Rotary Position Embedding)                      │
│  Option D: ALiBi (Attention with Linear Biases)                  │
│                                                                  │
│  tokens = tokens + pos_encoding                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TRANSFORMER ENCODER                            │
├─────────────────────────────────────────────────────────────────┤
│  For layer in 1..N_layers:                                       │
│                                                                  │
│    ┌──────────────────────────────────────┐                     │
│    │  Multi-Head Self-Attention           │                     │
│    │  - heads: n_heads                    │                     │
│    │  - dim per head: d_model // n_heads  │                     │
│    │  - attention_mask from `mask`        │                     │
│    └──────────────────────────────────────┘                     │
│                     │                                            │
│                     ▼                                            │
│    LayerNorm + Residual                                          │
│                     │                                            │
│                     ▼                                            │
│    ┌──────────────────────────────────────┐                     │
│    │  Feed-Forward Network                │                     │
│    │  Linear(d_model, d_ff)               │                     │
│    │  Activation (GELU / SiLU)            │                     │
│    │  Dropout                             │                     │
│    │  Linear(d_ff, d_model)               │                     │
│    └──────────────────────────────────────┘                     │
│                     │                                            │
│                     ▼                                            │
│    LayerNorm + Residual                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      POOLING LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  Option A: [CLS] token (prepend learnable token)                │
│  Option B: Mean pooling over non-padded tokens                   │
│  Option C: Last non-padded token                                 │
│  Option D: Attention-weighted pooling                            │
│                                                                  │
│  Output: (B, d_model)                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CLASSIFICATION HEAD                           │
├─────────────────────────────────────────────────────────────────┤
│  Linear(d_model, d_hidden)                                       │
│  Activation (GELU)                                               │
│  Dropout                                                         │
│  Linear(d_hidden, 1)   ──►  logit for BCE loss                  │
│                                                                  │
│  OR                                                              │
│                                                                  │
│  Linear(d_model, 2)    ──►  logits for CrossEntropy             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Recommended Hyperparameters

| Parameter | Small | Base | Large |
|-----------|-------|------|-------|
| `d_model` | 128 | 256 | 512 |
| `n_heads` | 4 | 8 | 8 |
| `n_layers` | 4 | 6 | 8 |
| `d_ff` | 512 | 1024 | 2048 |
| `d_user` | 32 | 64 | 128 |
| `dropout` | 0.1 | 0.1 | 0.1 |
| `max_seq_len` | 512 | 512 | 1024 |

Approximate parameter counts:
- **Small**: ~1.5M params
- **Base**: ~6M params
- **Large**: ~25M params

---

## Attention Masking

The `mask` tensor indicates valid (non-padded) positions. Convert to attention mask:

```python
# mask: (B, L) bool, True = valid token
# For self-attention, we need (B, 1, 1, L) or (B, 1, L, L)

attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
# Apply: attn_weights = attn_weights.masked_fill(~attn_mask, float('-inf'))
```

**Causal vs Bidirectional:**
- **Bidirectional** (recommended): All tokens can attend to all other tokens. The model sees the full trading history up to a cutoff.
- **Causal**: Each token can only attend to previous tokens. Useful if we want to predict at every timestep.

For market outcome prediction, **bidirectional** is more appropriate since we observe the sequence up to a point and then predict.

---

## Training Strategy

### Loss Function
```python
loss = F.binary_cross_entropy_with_logits(logits, y.float())
```

### Optimizer
- AdamW with weight decay 0.01–0.1
- Learning rate: 1e-4 to 5e-4
- Warmup: 5–10% of total steps
- Cosine decay schedule

### Regularization
- Dropout in attention and FFN
- Label smoothing (optional, e.g., 0.05)
- Gradient clipping (max_norm=1.0)

### Data Augmentation (optional)
- Random sequence truncation (already done by `min_prefix` sampling)
- Random feature noise injection

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **AUC-ROC** | Primary metric for ranking quality |
| **Accuracy** | At threshold 0.5 |
| **Precision/Recall** | Per-class performance |
| **Brier Score** | Calibration quality |
| **ECE** | Expected Calibration Error |

---

## Inference

```python
model.eval()
with torch.no_grad():
    logits = model(x, u, mask)  # (B, 1) or (B,)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()
```

---

## Potential Enhancements

1. **Relative Position Bias**: Use time between trades (`log_dt`) as relative position signal
2. **Cross-Attention to Market Metadata**: Add market-level features (category, duration, etc.)
3. **Pre-training**: Self-supervised pre-training on masked trade prediction
4. **Ensemble**: Multiple models with different random seeds or architectures
5. **Feature Interactions**: Explicit interaction terms before transformer
6. **Temporal Encoding**: Use `log_dt` to weight positional encodings by actual time gaps

---

## File Structure

```
polyquant/models/
├── __init__.py
├── mlp.py                    # Existing MLP baseline
├── transformer.py            # Main transformer model
├── components/
│   ├── __init__.py
│   ├── attention.py          # Multi-head attention
│   ├── embeddings.py         # Token + user + position embeddings
│   ├── encoder.py            # Transformer encoder blocks
│   └── pooling.py            # Sequence pooling strategies
└── TRANSFORMER_ARCHITECTURE.md
```

---

## References

- Vaswani et al., "Attention Is All You Need" (2017)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- Press et al., "Train Short, Test Long: Attention with Linear Biases" (ALiBi, 2022)
