# PolyQuant — Market Outcome Prediction for Prediction Markets


## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Pipeline](#data-pipeline)
- [Models](#models)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Documentation](#documentation)

---

## Overview

PolyQuant is a machine-learning pipeline for modeling market outcomes in the Polymarket prediction-market ecosystem. The project aims to predict the probability of market outcomes based on trade-level features, or sequences of trades. A number of architectures and variations were used and compared. There are 4 model types: a small MLP, a deep ResNet, a transformer with market-level sequences, and a dual-encoding transformer that takes both market trades and user trade history sequences.

### Problem Formulation

Given information available at the time of a trade, the goal is to predict the **market outcome**:

```
y = 1 if market resolves YES, 0 if market resolves NO
```

The model outputs a probability estimate `p = P(outcome = YES)`.

### Tabular vs sequential data

We consider two model types - sequential transformers and MLP. For each there is a dedicated pipeline for creating and loading the data.  
The tabular data is extracted from an SQL database into parquet files, using scripts in `polyquant_features`. The sequential data is only a re-indexing of the same data, using different scripts in the same folder.  
Data loaders for each model can be found under `polyquant/data/datasets/`. 

The data itself, which is of the order of 30Gb, if not included in the git.

### What This Repository Includes

- **Feature Engineering Pipeline** (`polyquant_features/`): Transforms raw SQL trade data into rich feature vectors
- **Multiple Dataset Formats**: Tabular (per-trade), sequence (per-market), and dual-sequence (market + user history)
- **Multiple Model Architectures**: MLP, ResNet, Transformer, and Dual-Encoder Transformer
- **Training & Evaluation**: Complete training loops with W&B logging, checkpointing, and comprehensive metrics
- **Profitability Metrics**: Beyond accuracy—measures actual trading profit potential

---

## Project Structure

```
PolyQuant/
│
├── config.json                    # Global path configuration
├── pyproject.toml                 # Package metadata
├── requirements.txt               # Python dependencies
│
├── polyquant/                     # Main training/inference package
│   ├── __init__.py
│   ├── config.py                  # Path configuration loader
│   ├── utils.py                   # Checkpoint utilities
│   │
│   ├── data/                      # Data loading & preprocessing
│   │   ├── schema.py              # Feature schema definitions
│   │   ├── normalize.py           # Feature normalization (mean/std scaling)
│   │   └── datasets/
│   │       ├── tabular.py         # TabularParquetIterable (per-trade batches)
│   │       ├── sequence_dataset.py      # MarketWindowDataset (per-market sequences)
│   │       └── dual_sequence_dataset.py # DualSequenceDataset (market + user history)
│   │
│   ├── models/                    # Neural network architectures
│   │   ├── mlp.py                 # Multi-layer perceptron baseline
│   │   ├── resnet.py              # 1D ResNet & ResNetMLP (residual connections)
│   │   ├── transformer.py         # MarketTransformer & TradeTransformer
│   │   └── dual_encoder.py        # Dual-Encoder with cross-attention
│   │
│   ├── training/                  # Training scripts
│   │   ├── train_mlp.py           # MLP training
│   │   ├── train_resnet.py        # ResNet training
│   │   ├── train_dual_encoder.py  # Dual-Encoder training
│   │   └── train_transformer_no_user.py # Market level transformer
│   │
│   ├── evaluation/                # Evaluation scripts
│   │   ├── eval_resnet.py         # ResNet evaluation
│   │   ├── eval_transformer.py    # Transformer evaluation
│   │   ├── eval_dual_encoder.py   # Dual-Encoder evaluation
│   │   └── eval_utils.py          # Shared evaluation utilities
│   │
│   └── metrics/                   # Metrics computation
│       ├── evaluation.py          # MetricsAccumulator (BCE, accuracy, AUC, etc.)
│       └── profit.py              # Profitability metrics (PnL, ROI, hit rate)
│
├── polyquant_features/            # Feature engineering package
│   ├── __init__.py
│   ├── config.py                  # Feature build configuration
│   ├── db.py                      # SQLite database utilities
│   ├── states.py                  # UserState & MarketState tracking
│   ├── features.py                # Feature computation logic
│   ├── writer.py                  # Parquet chunk writer with checkpointing
│   ├── build_features.py          # Main feature builder (SQL → parquet)
│   ├── build_sequences_store.py   # Build market sequence shards
│   ├── build_user_sequences_store.py  # Build user history store
│   ├── build_user_sequences.py    # Build per-user sequence files
│   └── make_market_meta.py        # Generate market metadata
│
├── scripts/                       # Utility scripts
│   ├── compute_norm_stats.py      # Compute train set normalization statistics
│   ├── split_features_dataset.py  # Split dataset into train/val/test
│   ├── fix_index_paths.py         # Fix relative paths in index files
│   └── smoke_loader.py            # Test data loading
│
├── data/                          # Data directory (not in git)
│   ├── features_full/             # Tabular features dataset
│   │   ├── train/                 # Training parquet files
│   │   ├── val/                   # Validation parquet files
│   │   ├── test/                  # Test parquet files
│   │   ├── train_scaler.json      # Normalization statistics
│   │   └── market_meta.parquet    # Market metadata
│   │
│   ├── sequences/                 # Market sequence dataset
│   │   ├── index.parquet          # Market index (path, start, length, label)
│   │   ├── train/                 # Training shards
│   │   ├── val/                   # Validation shards
│   │   └── test/                  # Test shards
│   │
│   └── user_sequences_store/      # User history store
│       ├── index.parquet          # User index
│       ├── train/                 # Training user sequences
│       ├── val/                   # Validation user sequences
│       └── test/                  # Test user sequences
│
├── checkpoints/                   # Model checkpoints (not in git)
│   └── YYYYMMDD_HHMMSS_<model>/
│       ├── best.pt                # Best validation checkpoint
│       ├── latest.pt              # Latest checkpoint
│       └── config.json            # Training configuration
│
├── runs/                          # W&B logs (not in git)
│
└── docs/                          # Documentation
    ├── DUAL_ENCODER_TRANSFORMER.md    # Dual-encoder architecture details
    └── DIGITALOCEAN_GPU_SETUP.md      # Cloud GPU setup guide
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for training)
- ~50GB disk space for full dataset

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/PolyQuant.git
   cd PolyQuant
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
   ```

4. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `numpy`, `pandas`, `polars` | Data manipulation |
| `pyarrow` | Parquet file handling |
| `duckdb` | SQL queries and user hash computation |
| `scikit-learn` | Metrics (AUC, etc.) |
| `wandb` | Experiment tracking |
| `tqdm` | Progress bars |
| `matplotlib` | Visualization |

---

## Data Pipeline

### Step 1: Build Features from Raw Data

Convert raw trade data from SQLite to feature-enriched parquet files:

```bash
python -m polyquant_features.build_features \
    --db data/sql/polymarket.db \
    --out data/features_raw \
    --chunk-rows 2000000
```

This processes trades chronologically, maintaining causal state for users and markets.

### Step 2: Split into Train/Val/Test

Split the dataset based on market close times (temporal split):

```bash
python scripts/split_features_dataset.py \
    --input data/features_raw \
    --output data/features_full \
    --train-ratio 0.7 \
    --val-ratio 0.15
```

### Step 3: Compute Normalization Statistics

Compute mean/std from training set only:

```bash
python scripts/compute_norm_stats.py
```

This creates `data/features_full/train_scaler.json`.

### Step 4: Build Sequence Datasets (for Transformers)

For sequence-based models, build market sequences:

```bash
python -m polyquant_features.build_sequences_store \
    --input data/features_full \
    --output data/sequences
```

For dual-encoder models, also build user history:

```bash
python -m polyquant_features.build_user_sequences_store \
    --input data/features_full \
    --output data/user_sequences_store
```

---

## Models

### 1. MLP (Baseline)

Simple multi-layer perceptron for tabular features.

```python
from polyquant.models.mlp import MLP

model = MLP(in_dim=40, hidden_dims=[256, 128, 64], dropout=0.2)
```

### 2. ResNetMLP

MLP with residual connections for better gradient flow:

```python
from polyquant.models.resnet import ResNetMLP

model = ResNetMLP(
    in_dim=40,
    hidden_dims=(256, 512, 1024, 512, 256, 128, 32),
    dropout=0.2
)
```

### 3. MarketTransformer

Transformer that processes a sequence of trades in a market to predict the market outcome:

```python
from polyquant.models.transformer import MarketTransformer

model = MarketTransformer(
    d_input=10,      # Number of continuous features per trade
    d_model=128,     # Hidden dimension
    d_user=16,       # User embedding dimension
    n_heads=4,       # Attention heads
    n_layers=4,      # Transformer layers
    max_seq_len=512  # Maximum sequence length
)
```

### 4. DualEncoderTransformer

Advanced architecture that encodes both market context and user trading history with cross-attention:

```python
from polyquant.models.dual_encoder import DualEncoderTransformer

model = DualEncoderTransformer(
    d_market=12,      # Market sequence features
    d_user=4,         # User history features
    d_model=128,
    n_market_layers=4,
    n_user_layers=2,
    L_market=1024,    # Market sequence length
    L_user=128        # User history length
)
```

---

## Training

### Training ResNet (Tabular)

```bash
python -m polyquant.training.train_resnet
```

With checkpoint resumption:
```bash
python -m polyquant.training.train_resnet --resume checkpoints/20260128_145823_resnet
```

### Training Transformer

```bash
python -m polyquant.training.train_transformer
```

### Training Dual-Encoder

```bash
python -m polyquant.training.train_dual_encoder
```

### Training Configuration

Hyperparameters are defined in each training script. Key settings:

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `BATCH_SIZE` | 512-4096 | Batch size (adjust for GPU memory) |
| `LR` | 1e-6 to 1e-4 | Learning rate |
| `WARMUP_STEPS` | 1000-5000 | Linear warmup steps |
| `MAX_STEPS` | 10000-50000 | Total training steps |
| `DROPOUT` | 0.1-0.3 | Dropout rate |
| `WEIGHT_DECAY` | 1e-3 to 1e-2 | AdamW weight decay |

### Experiment Tracking

Training logs to [Weights & Biases](https://wandb.ai). Set your API key:

```bash
wandb login
```

Metrics logged include:
- BCE loss
- Accuracy (misclassification rate)
---

## Evaluation

### Evaluate a Trained Model

```bash
python -m polyquant.evaluation.eval_resnet --checkpoint checkpoints/20260128_145823_resnet/best.pt
```

### Metrics

The evaluation computes:

| Metric | Description |
|--------|-------------|
| `bce` | Binary cross-entropy loss |
| `misclass` | Misclassification rate (1 - accuracy) |

## Configuration

### Global Paths (`config.json`)

```json
{
  "root": ".",
  "dataset_root": "data/features_full",
  "scaler_path": "data/features_full/train_scaler.json",
  "runs_dir": "runs",
  "checkpoints_dir": "checkpoints"
}
```

All paths are resolved relative to the repository root. Training scripts automatically discover `config.json` by traversing up from the script location.

### Loading Configuration

```python
from polyquant.config import load_paths

paths = load_paths(__file__)
print(paths.dataset_root)      # Path to dataset
print(paths.checkpoints_dir)   # Path to checkpoints
```

## Documentation

Detailed architecture documentation is available in the [docs/](docs/) directory:

- **[DUAL_ENCODER_TRANSFORMER.md](docs/DUAL_ENCODER_TRANSFORMER.md)**: Complete description of the dual-encoder architecture with cross-attention
- **[DIGITALOCEAN_GPU_SETUP.md](docs/DIGITALOCEAN_GPU_SETUP.md)**: Guide for setting up GPU training on DigitalOcean

---

## Quick Start Example

```python
import torch
from polyquant.config import load_paths
from polyquant.data.schema import load_schema
from polyquant.data.normalize import load_feature_scaler
from polyquant.data.datasets.tabular import make_loaders
from polyquant.models.resnet import ResNetMLP

# Load configuration
paths = load_paths(__file__)
schema = load_schema(paths.dataset_root)
scaler = load_feature_scaler(paths.scaler_path, schema)

# Create data loaders
train_loader, val_loader = make_loaders(
    dataset_root=paths.dataset_root,
    schema=schema,
    scaler=scaler,
    batch_size=512
)

# Initialize model
model = ResNetMLP(
    in_dim=len(schema.feature_cols),
    hidden_dims=(256, 512, 256, 64),
    dropout=0.2
)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCEWithLogitsLoss()

for batch in train_loader:
    x = batch["x"].to(device)
    y = batch["y"].to(device)

    logits = model(x).view(-1)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
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

The prediction target is the market outcome:

```
y = 1 if market resolved YES, else 0
```

Training objective:
```
loss = BCEWithLogitsLoss(logits, y)
```

---

## Split Strategy

Markets are split by **end timestamp** (temporal split):

| Split | Purpose | Selection |
|-------|---------|-----------|
| **Train** | Model training | Oldest 70% of markets |
| **Validation** | Hyperparameter tuning | Middle 15% of markets |
| **Test** | Final evaluation | Newest 15% of markets |

This ensures:
- ✅ No temporal leakage (future information never used)
- ✅ No market leakage (all trades from a market stay in same split)
- ✅ Realistic evaluation (model tested on "future" markets)

---

## Requirements

### Minimum Requirements
- Python 3.10+
- PyTorch 2.0+
- 8GB+ GPU memory (for training)
- 16GB+ RAM

---


## Acknowledgments

- Data sourced from [Polymarket](https://polymarket.com/)
- Built with [PyTorch](https://pytorch.org/) and [Weights & Biases](https://wandb.ai/)
