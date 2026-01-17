# TradeTransformer: Per-Trade Edge Prediction

This document describes the TradeTransformer architecture and its training pipeline for predicting trading edge on prediction markets.

## Overview

The TradeTransformer is designed to predict whether each individual trade in a market sequence will be profitable, given the context of all previous trades. Unlike the MarketTransformer (which predicts a single outcome per market), the TradeTransformer produces **per-token predictions** using **causal attention**, making it suitable for online edge estimation.

---

## Architecture

### Model Variants

| Variant | d_model | d_user | n_heads | n_layers | d_ff  | user_vocab | ~Parameters |
|---------|---------|--------|---------|----------|-------|------------|-------------|
| Small   | 128     | 16     | 4       | 4        | 512   | 50,000     | ~1.7M       |
| Base    | 256     | 32     | 8       | 6        | 1024  | 100,000    | ~8M         |

### Input Features

Each trade in a sequence has 10 continuous features:

| Index | Feature      | Description                          |
|-------|--------------|--------------------------------------|
| 0     | `p_yes`      | Current YES price (probability)      |
| 1-9   | Other features | Trade metadata (amount, side, etc.) |

Plus a `user_hash` (integer) for user embeddings.

### Model Components

```
Input: x (B, L, 10), u (B, L), mask (B, L)
                │
                ▼
┌─────────────────────────────────────┐
│         Input Projection            │
│    Linear(10 → d_model)             │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│         User Embedding              │
│    Embedding(user_vocab, d_user)    │
│    (modulo applied for hash)        │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│         Combine + Project           │
│    Concat → Linear(d_model+d_user → d_model)
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│    Positional Embeddings (Learned)  │
│    Embedding(max_seq_len, d_model)  │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│         Dropout                     │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│    Transformer Encoder Layers       │
│    (n_layers × TransformerEncoderLayer)
│    - Causal attention mask          │
│    - Pre-LN (norm_first=True)       │
│    - GELU activation                │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│         Final LayerNorm             │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│    Per-Token Classifier             │
│    Linear(d_model → d_model/2)      │
│    → GELU → Dropout                 │
│    → Linear(d_model/2 → 1)          │
└─────────────────────────────────────┘
                │
                ▼
Output: logits (B, L)
```

### Causal Attention

The key difference from MarketTransformer is the use of **causal (autoregressive) attention**:

```python
# Upper triangular mask: position i cannot attend to positions j > i
causal_mask = torch.triu(ones(L, L), diagonal=1)  # True = masked
```

This ensures that the prediction at position `t` only uses information from trades `0, 1, ..., t-1`, simulating an online prediction scenario.

### User Embeddings

User hashes from the dataset may exceed the vocabulary size. The model handles this with modulo:

```python
u_clamped = u % self.user_embed.num_embeddings
```

This allows hash collisions but keeps the embedding table small enough to fit in memory.

---

## Training Script

### File: `scripts/train_trade_transformer.py`

### Hyperparameters

| Parameter           | Value    | Description                              |
|---------------------|----------|------------------------------------------|
| `BATCH_SIZE`        | 64       | Sequences per batch                      |
| `SEQ_LEN`           | 512      | Max tokens per sequence                  |
| `CAP_TRADES`        | 4096     | Max trades sampled from each market      |
| `MIN_PREFIX`        | 20       | Minimum prefix trades before windowing   |
| `MAX_STEPS`         | 100,000  | Total training steps                     |
| `LR`                | 3e-4     | Peak learning rate                       |
| `LR_MIN`            | 1e-6     | Minimum learning rate after decay        |
| `WARMUP_STEPS`      | 1,000    | Linear warmup steps                      |
| `WEIGHT_DECAY`      | 0.05     | AdamW weight decay                       |
| `DROPOUT`           | 0.15     | Dropout probability                      |
| `GRAD_CLIP_NORM`    | 1.0      | Gradient clipping threshold              |
| `AMP_ENABLED`       | True     | Mixed precision training                 |

### Learning Rate Schedule

Cosine decay with linear warmup:

```
lr(t) =
  • t < warmup_steps:  lr_max × (t / warmup_steps)
  • t ≥ warmup_steps:  lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × progress))
```

### Loss Function: PnL-Weighted BCE

The loss weights each trade by its potential profit/loss magnitude:

```python
# y: market outcome (0 or 1)
# p: trade price (p_yes)
# Weight = |profit if correct| = y*(1-p) + (1-y)*p
w = y * (1.0 - p) + (1.0 - y) * p

# Weighted BCE
loss = Σ [w × BCE(logits, y)] / Σ w
```

**Intuition**: Trades at extreme prices (near 0 or 1) have low weight because they offer small edge. Trades near 0.5 are weighted higher.

### Metrics

#### Core Metrics

| Metric     | Description                                        |
|------------|----------------------------------------------------|
| `bce`      | PnL-weighted binary cross-entropy loss             |
| `misclass` | Misclassification rate (pred > 0.5 vs actual)      |
| `mae_edge` | Mean absolute error of predicted vs true edge      |

#### Profitability Metrics (per threshold τ)

The model acts as a trading policy: **take the trade if predicted_edge > τ**

| Metric           | Formula                                              |
|------------------|------------------------------------------------------|
| `pnl/tau_X`      | Total PnL for trades where pred_edge > τ             |
| `pnl_norm/tau_X` | Average PnL per taken trade                          |
| `take_rate/tau_X`| Fraction of trades taken                             |
| `hit_rate/tau_X` | Win rate among taken trades                          |

**Thresholds evaluated**: τ ∈ {0.00, 0.005, 0.01, 0.02, 0.05}

**Edge calculation**:
```python
pred_edge = sigmoid(logits) - price  # Predicted edge
true_edge = y - price                # Actual profit per share
```

### Training Loop

```
for step in 1..MAX_STEPS:
    1. Update learning rate (cosine schedule)
    2. Forward pass with AMP autocast
    3. Compute PnL-weighted BCE loss
    4. Backward pass with gradient scaling
    5. Clip gradients (norm ≤ 1.0)
    6. Optimizer step

    if step % 10 == 0:
        Log train metrics to TensorBoard

    if step % 200 == 0:
        Run validation (50 batches)
        Log val metrics including profitability

    if step % 5000 == 0:
        Save checkpoint
```

### Checkpointing

Checkpoints are saved every 5,000 steps to `checkpoints/<run_name>/step_XXXXXX.pt`

Each checkpoint contains:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `scaler_state_dict`: AMP scaler state
- `step`, `epoch`, `run_name`: Training metadata

### Resuming Training

```bash
python scripts/train_trade_transformer.py --resume checkpoints/20251228_195545/step_0010000.pt
```

---

## Usage

### Training from Scratch

```bash
# Small model (~1.7M params)
python scripts/train_trade_transformer.py

# Base model (~8M params)
python scripts/train_trade_transformer.py --model-size base
```

### Monitoring

TensorBoard logs are saved to `runs/<run_name>/`:

```bash
tensorboard --logdir runs/
```

Key metrics to watch:
- `val/pnl_norm/tau_0p010` - Average profit per trade at 1% threshold
- `val/take_rate/tau_0p010` - Fraction of trades taken at 1% threshold
- `val/misclass` - Should decrease over time
- `train/bce` vs `val/bce` - Gap indicates overfitting

---

## Comparison: MarketTransformer vs TradeTransformer

| Aspect              | MarketTransformer            | TradeTransformer              |
|---------------------|------------------------------|-------------------------------|
| **Prediction**      | One per market               | One per trade                 |
| **Attention**       | Bidirectional                | Causal (autoregressive)       |
| **Pooling**         | Mean over sequence           | None (per-token output)       |
| **Output shape**    | `(B,)`                       | `(B, L)`                      |
| **Use case**        | Market outcome prediction    | Online edge estimation        |
| **Loss**            | BCE with market outcome      | PnL-weighted BCE              |

---

## Code References

- Model: [polyquant/models/transformer.py](../polyquant/models/transformer.py)
- Training: [scripts/train_trade_transformer.py](../scripts/train_trade_transformer.py)
- Dataset: [polyquant/data/datasets/sequence_dataset.py](../polyquant/data/datasets/sequence_dataset.py)
