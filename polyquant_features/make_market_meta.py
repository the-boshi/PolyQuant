from pathlib import Path
import pyarrow.parquet as pq

SPLITS = Path(r"C:\Users\nimro\PolyQuant\data\features\market_splits.parquet")
LABELS = Path(r"C:\Users\nimro\PolyQuant\data\features\market_labels.parquet")
OUT    = Path(r"C:\Users\nimro\PolyQuant\data\features\market_meta.parquet")

def main():
    splits = pq.read_table(SPLITS).select(["market_id", "split"])
    labels = pq.read_table(LABELS).select(["market_id", "y"])

    meta = splits.join(labels, keys="market_id", join_type="inner")  # keeps only labeled markets
    pq.write_table(meta, OUT, compression="zstd")
    print("wrote", OUT, "rows=", meta.num_rows)

if __name__ == "__main__":
    main()
