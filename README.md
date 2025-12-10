# PolyQuant — Trade-Level Edge Prediction for Prediction Markets

## Overview
PolyQuant is a machine-learning pipeline for modeling **trade-level predictive edge** in the Polymarket prediction-market ecosystem.  
Each trade is transformed into a rich feature vector representing:

- user behavior
- market microstructure
- rolling 1h windows
- user-in-market dynamics
- time / temporal context
- calibrated price-based signals

The goal is to predict:

```
p = P(outcome_resolves_yes | trade_features)
edge = p − trade_price
```

This repository includes:

- a **feature-builder** that ingests raw SQL trades and outputs parquet chunks  
- a **train/val/test splitter** based on market closure times  
- a **scaler and normalization pipeline**  
- a **PyTorch training loop** for an MLP baseline  
- an extensible directory structure intended for future models (LSTM, Transformer, etc.)

---

## Project Structure

```
PolyQuant/
│
├── build_features/
│   ├── build_features.py         # main feature generator from SQL → parquet
│   ├── make_splits.py            # temporal market split into train/val/test
│   ├── statistics.ipynb          # EDA and feature validation
│   └── testing.ipynb             # unit checks on features
│
├── data/
│   ├── features_dataset/
│   │   ├── train/                # train parquet files
│   │   ├── val/
│   │   ├── test/
│   │   ├── market_splits.parquet
│   │   ├── market_splits_meta.json
│   │   └── train_scaler.json     # saved normalization statistics
│   │
│   └── sql/
│       └── polymarket.db         # raw Polymarket trades + markets tables
│
├── polyquant/
│   ├── data/
│   │   ├── datasets/
│   │   │   └── tabular.py        # iterable parquet loader + shuffle buffer
│   │   ├── normalize.py          # FeatureScaler / load/save stats
│   │   └── schema.py             # feature schema + column definitions
│   │
│   ├── models/
│   │   └── mlp.py                # simple MLP baseline
│   │
│   └── utils/
│       └── misc.py               # misc helpers (duration formatting, etc.)
│
├── scripts/
│   ├── compute_norm_stats.py     # compute train normalization stats
│   ├── train_mlp.py              # main training script with TensorBoard
│   └── eval_mlp.py               # optional evaluation hook (future)
│
├── checkpoints/
│   └── <run_id>/                 # per-run saved model weights
│
├── runs/
│   └── <run_id>/                 # TensorBoard logs
│
└── README.md
```

---

## Dataset & Features

Each trade is converted into ~40 features grouped into:

### 1. **Trade-local**
- price, log(USDC size), outcome index  
- seconds since market open  
- time-of-day sin/cos  

### 2. **User-global**
- lifetime trade count, volume, days active  
- Welford mean/std of trade size  
- realized PnL statistics **on closed markets only**  
- recent PnL (20 trades)  
- intertrade interval variance (burstiness)

### 3. **Market-level**
- lifetime trades and volume  
- last YES/NO price  
- rolling 1h window volatility, volume, trade size  
- time since last market trade  

### 4. **User-in-Market**
- user’s 1h volume share  
- total lifetime volume in this market  
- count of user trades in this market  
- time since user last traded in this market  

### Labels
Two labels per trade:

```
p = 1 if final_outcome == YES else 0
y = p
edge = final_outcome_price − trade_price
```

For training the MLP:

```
target = y
loss    = BCEWithLogitsLoss
metric  = edge_MAE (|pred_edge − true_edge|)
```

---

## Training

### Command

From project root:

```bash
python scripts/train_mlp.py
```

### Logging
TensorBoard:

```bash
tensorboard --logdir runs
```

### Checkpoints

Saved under:

```
checkpoints/<run_timestamp>/*.pt
```

---

## Model Architecture

Current baseline:
- Fully-connected MLP  
- Hidden dims: (512, 512, 256, 128)  
- Dropout: 0.05  
- AdamW + weight decay  
- Cosine LR schedule with optional warmup  

Future models will go under:

```
polyquant/models/
```

This includes:
- recurrent models (LSTM/GRU)
- Transformer blocks
- user-sequence encoders
- market-state encoders

---


## Split Strategy

Markets are split by **end timestamp**:

- **train:** older markets  
- **val:** middle slice  
- **test:** newest markets  

This ensures:
- no temporal leakage  
- no market-future leakage between splits  
- all trades in a market remain in the same split


---

## Requirements

```
Python 3.10+
PyTorch 2.1+
CUDA-capable GPU (recommended)
pyarrow
numpy
tensorboard
```