import random
import zlib
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
import torch


def stable_user_hash(user_id: str, vocab_size: int) -> int:
    h = zlib.crc32(user_id.encode("utf-8")) & 0xFFFFFFFF
    return int(h % vocab_size)


class MarketWindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        segments_path: str,
        meta_path: str,
        split: str,
        L: int = 512,
        min_prefix: int = 20,
        user_vocab: int = 2_000_000,
        dp_clip: float = 0.2,
        pnl_asinh_scale: float = 50.0,
        cache_files: int = 32,
        cap_trades: int | None = None,
    ):
        self.L = int(L)
        self.cap_trades = max(int(cap_trades) if cap_trades is not None else max(4 * self.L, 8192), self.L)
        self.min_prefix = int(min_prefix)
        self.user_vocab = int(user_vocab)
        self.dp_clip = float(dp_clip)
        self.pnl_asinh_scale = float(pnl_asinh_scale)

        # load labels for this split
        meta = pq.read_table(meta_path).to_pandas()
        meta = meta[meta["split"] == split].copy()
        self.y_map = dict(zip(meta["market_id"].astype(str), meta["y"].astype(int)))

        # load segments for this split, keep only labeled markets
        seg = pq.read_table(segments_path).to_pandas()
        seg = seg[seg["split"] == split].copy()
        seg = seg[seg["market_id"].astype(str).isin(self.y_map.keys())].copy()

        # group segments by market_id, sort by time so concat is correct
        by_market = defaultdict(list)
        for r in seg.itertuples(index=False):
            by_market[str(r.market_id)].append((int(r.min_ts), str(r.path), int(r.start), int(r.length)))

        self.market_ids = sorted(by_market.keys())
        self.market_segs = []
        self.market_lens = []
        for mid in self.market_ids:
            parts = by_market[mid]
            parts.sort(key=lambda x: x[0])  # by min_ts

            cum = 0
            segs = []
            for (_, p, s, l) in parts:
                segs.append((p, s, l, cum))  # cum = market-global start of this segment
                cum += l

            self.market_segs.append(segs)
            self.market_lens.append(cum)


        # parquet file handle cache
        self._pf_cache = {}
        self._pf_lru = []
        self._pf_cache_cap = int(cache_files)

        self.cols = [
            "user_id",
            "p_yes",
            "dp_yes",
            "dt_trade",
            "log_usdc_size",
            "user_historical_pnl_before",
            "user_total_trades_before",
            "user_historical_winrate_before",
            "user_total_volume_before",
        ]

    def _read_market_range(self, segs, a: int, b: int):
        # read market-global rows [a, b) across segments
        dfs = []
        for (p, s, l, cum0) in segs:
            seg_a = cum0
            seg_b = cum0 + l
            if b <= seg_a or a >= seg_b:
                continue

            lo = max(a, seg_a)
            hi = min(b, seg_b)
            file_start = s + (lo - seg_a)
            length = hi - lo
            if length <= 0:
                continue

            pf = self._get_pf(p)
            dfs.append(pf.read(columns=self.cols).slice(int(file_start), int(length)).to_pandas())

        if not dfs:
            return None
        return dfs[0] if len(dfs) == 1 else np.concatenate([d.values for d in dfs], axis=0)


    def __len__(self):
        return len(self.market_ids)

    def _get_pf(self, path: str):
        pf = self._pf_cache.get(path)
        if pf is not None:
            return pf

        pf = pq.ParquetFile(path)
        self._pf_cache[path] = pf
        self._pf_lru.append(path)

        # simple LRU eviction
        if len(self._pf_lru) > self._pf_cache_cap:
            old = self._pf_lru.pop(0)
            if old in self._pf_cache:
                del self._pf_cache[old]
        return pf

    def _read_segment(self, path: str, start: int, length: int):
        pf = self._get_pf(path)
        return pf.read(columns=self.cols).slice(start, length).to_pandas()

    def __getitem__(self, idx):
        market_id = self.market_ids[idx]
        y = int(self.y_map[market_id])

        # 1) read and concat market trades across segments
        parts = self.market_segs[idx]
        N_total = self.market_lens[idx]

        lo = max(self.min_prefix - 1, 0)
        hi = N_total - 1
        n = random.randint(lo, hi) if hi >= lo else hi

        CAP = self.cap_trades
        read_start = max(0, n - CAP + 1)
        read_end = n + 1

        df = self._read_market_range(parts, read_start, read_end)

        # unpack
        if hasattr(df, "columns"):
            user_ids = df["user_id"].astype(str).to_numpy()
            p_yes = df["p_yes"].to_numpy(dtype=np.float32)
            dp_yes = df["dp_yes"].to_numpy(dtype=np.float32)
            dt_trade = df["dt_trade"].to_numpy(dtype=np.float32)
            log_usdc = df["log_usdc_size"].to_numpy(dtype=np.float32)
            pnl = df["user_historical_pnl_before"].to_numpy(dtype=np.float32)
            u_trades = df["user_total_trades_before"].to_numpy(dtype=np.float32)
            winr = df["user_historical_winrate_before"].to_numpy(dtype=np.float32)
            u_vol = df["user_total_volume_before"].to_numpy(dtype=np.float32)
        else:
            arr = df  # ndarray, same order as self.cols
            user_ids = arr[:, 0].astype(str)
            p_yes = arr[:, 1].astype(np.float32)
            dp_yes = arr[:, 2].astype(np.float32)
            dt_trade = arr[:, 3].astype(np.float32)
            log_usdc = arr[:, 4].astype(np.float32)
            pnl = arr[:, 5].astype(np.float32)
            u_trades = arr[:, 6].astype(np.float32)
            winr = arr[:, 7].astype(np.float32)
            u_vol = arr[:, 8].astype(np.float32)

        N = len(p_yes)

        D = 8
        if N == 0:
            x = torch.zeros(self.L, D)
            u = torch.zeros(self.L, dtype=torch.long)
            mask = torch.zeros(self.L, dtype=torch.bool)
            return x, u, mask, torch.tensor(y, dtype=torch.long)

        # 2) choose random cutoff n, take window ending at n of length L
        lo = max(self.min_prefix - 1, 0)
        hi = N - 1
        n = random.randint(lo, hi) if hi >= lo else hi

        start = max(0, n - self.L + 1)
        sl = slice(start, n + 1)

        # 3) transforms
        dp = np.clip(dp_yes[sl], -self.dp_clip, self.dp_clip)

        dt = np.maximum(dt_trade[sl], 0.0)
        log_dt = np.log1p(dt)

        pnl_t = np.arcsinh(pnl[sl] / self.pnl_asinh_scale)

        trades_t = np.log1p(np.maximum(u_trades[sl], 0.0))
        vol_t = np.log1p(np.maximum(u_vol[sl], 0.0))
        win_t = np.clip(winr[sl], 0.0, 1.0)

        x = np.stack(
            [p_yes[sl], dp, log_dt, log_usdc[sl], pnl_t, trades_t, win_t, vol_t],
            axis=1,
        ).astype(np.float32)

        u = np.array([stable_user_hash(u0, self.user_vocab) for u0 in user_ids[sl]], dtype=np.int64)

        # 4) left-pad to fixed length L
        pad = self.L - x.shape[0]
        if pad > 0:
            x = np.concatenate([np.zeros((pad, D), np.float32), x], axis=0)
            u = np.concatenate([np.zeros((pad,), np.int64), u], axis=0)
            mask = np.zeros((self.L,), dtype=bool)
            mask[pad:] = True
        else:
            mask = np.ones((self.L,), dtype=bool)

        return (
            torch.from_numpy(x),
            torch.from_numpy(u),
            torch.from_numpy(mask),
            torch.tensor(y, dtype=torch.long),
        )
