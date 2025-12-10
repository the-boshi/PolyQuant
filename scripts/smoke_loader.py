from pathlib import Path
import torch

from polyquant.data.schema import load_schema
from polyquant.data.normalize import load_feature_scaler
from polyquant.data.datasets.tabular import TabularParquetIterable

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "features_dataset"
SCALER_PATH = DATASET_ROOT / "train_scaler.json"

schema = load_schema(DATASET_ROOT)
scaler = load_feature_scaler(SCALER_PATH, schema.feature_cols, schema.no_scale_cols)

ds = TabularParquetIterable(
    split_dir=DATASET_ROOT / "train",
    feature_cols=schema.feature_cols,
    scaler=scaler,
    batch_size=8192,
    shuffle_files=True,
    shuffle_rowgroup=True,
)

it = iter(ds)
for i in range(3):
    b = next(it)
    print("batch", i, "x", tuple(b["x"].shape), "y_mean", float(b["y"].mean()), "edge_mean", float(b["edge"].mean()))
