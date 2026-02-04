# MarketTransformerNoUser

A small PyTorch transformer encoder for **sequence-to-label** binary prediction from **continuous per-token features**. Supports variable-length sequences via a boolean padding mask and uses mean pooling over valid tokens.

---

## Summary

- **Input**: a padded sequence of feature vectors `x ∈ R^(B×L×D)` and a boolean valid-token mask `mask ∈ {0,1}^{B×L}`
- **Model**: linear projection + learned positional embeddings + TransformerEncoder (Pre-LN, GELU)
- **Output**: one **logit per sequence** `logits ∈ R^B`
- **Pooling**: masked mean pooling over non-padded tokens
- **Use case**: binary market outcome prediction from a sequence of trade features

---

## Architecture (detailed)

### 1) Inputs

- `x`: continuous features per token  
  Shape: **(B, L, D)**  
  - `B`: batch size  
  - `L`: padded sequence length in the batch  
  - `D`: `d_input` (default: 10)

- `mask`: valid-token mask  
  Shape: **(B, L)**, dtype `bool`  
  - `True` for valid tokens  
  - `False` for padding

---

### 2) Token embedding

#### 2.1 Linear feature projection
Each token feature vector is projected into the transformer hidden space:

- `h0 = Linear(d_input → d_model)(x)`  
- Shape: **(B, L, d_model)**  
Default base: `d_model = 256`

This is the only “embedding” step since inputs are already continuous.

#### 2.2 Learned positional embeddings
Add a learned positional embedding to each token:

- `pos = Embedding(max_seq_len, d_model)(0..L-1)`  
- `h = h0 + pos`  
- Shape stays: **(B, L, d_model)**

**Note:** `max_seq_len` must be ≥ the longest `L` you will ever feed.

#### 2.3 Dropout
A dropout layer is applied after adding positional embeddings:

- `h = Dropout(h)`

---

### 3) Transformer encoder stack

A standard **TransformerEncoder** is applied:

- `encoder = TransformerEncoder(TransformerEncoderLayer(...), num_layers=n_layers)`
- Uses:
  - **Pre-LN** (`norm_first=True`) for stability
  - **GELU** activation in the feedforward sublayer
  - **Multi-head self-attention** with `n_heads`

#### Padding handling
PyTorch expects `src_key_padding_mask` where **True means “ignore this token”**, so we invert:

- `src_key_padding_mask = ~mask`  
- Shape: **(B, L)**

Then:

- `h = encoder(h, src_key_padding_mask=src_key_padding_mask)`  
- Shape: **(B, L, d_model)**

---

### 4) Final normalization

A final LayerNorm is applied to per-token representations:

- `h = LayerNorm(d_model)(h)`  
- Shape: **(B, L, d_model)**

---

### 5) Masked mean pooling (sequence aggregation)

Convert token-level embeddings into a single vector per sequence via masked mean pooling:

1. Expand mask: `m = mask.unsqueeze(-1).float()` → **(B, L, 1)**
2. Zero out padding tokens: `h_masked = h * m`
3. Sum across time: `h_sum = sum(h_masked, dim=1)` → **(B, d_model)**
4. Normalize by valid-token count:  
   `denom = sum(m, dim=1).clamp(min=1.0)` → **(B, 1)**
5. Pooled embedding: `h_pooled = h_sum / denom` → **(B, d_model)**

This ensures padding does not affect the pooled representation.

---

### 6) Classification head

A small MLP produces one logit per sequence:

- `Linear(d_model → d_model)`
- `GELU`
- `Dropout`
- `Linear(d_model → 1)`
- Output squeezed to: **(B,)**

The model returns **logits** (raw scores). Use `sigmoid` to convert to probabilities.

---

## Default “Base” configuration

- `d_input = 10`
- `d_model = 256`
- `n_heads = 8`
- `n_layers = 6`
- `d_ff = 1024`
- `max_seq_len = 512`
- `dropout = 0.1`

---

## Shapes cheat-sheet

| Stage | Tensor | Shape |
|------|--------|-------|
| Input | `x` | (B, L, D) |
| Mask | `mask` | (B, L) |
| Projection | `h0` | (B, L, d_model) |
| + Positional | `h` | (B, L, d_model) |
| Transformer | `h` | (B, L, d_model) |
| Pooling | `h_pooled` | (B, d_model) |
| Output | `logits` | (B,) |

---

## Minimal usage example

```python
import torch
from model import create_base_transformer_no_user

model = create_base_transformer_no_user(d_input=10, max_seq_len=512)

B, L, D = 4, 512, 10
x = torch.randn(B, L, D)
mask = torch.ones(B, L, dtype=torch.bool)
mask[:, -100:] = False  # padding

logits = model(x, mask)           # (B,)
probs = torch.sigmoid(logits)     # (B,)
