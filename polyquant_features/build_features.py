#!/usr/bin/env python
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from polyquant_features.config import Config
from polyquant_features.db import (
    ensure_effective_end_ts,
    ensure_indexes,
    ensure_pragmas,
    load_market_end_ts,
    load_market_start_ts,
    trades_cursor,
)
from polyquant_features.features import (
    apply_trade_updates,
    compute_row_features,
    flush_pending_pnls,
    schedule_pnl,
)
from polyquant_features.states import MarketState, UserState
from polyquant_features.writer import (
    Checkpoint,
    append_row,
    init_rows_buffer,
    load_latest_checkpoint,
    save_checkpoint,
    write_parquet_chunk,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start-ts", type=int, default=Config.start_ts)  # uses default from class attr
    p.add_argument("--chunk-rows", type=int, default=2_000_000)
    p.add_argument("--checkpoint-every", type=int, default=20_000_000)
    return p.parse_args()


def main():
    print("Starting.")
    args = parse_args()
    cfg = Config(
        db_path=args.db,
        output_dir=args.out,
        chunk_rows=args.chunk_rows,
        checkpoint_every=args.checkpoint_every,
        start_ts=args.start_ts,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(cfg.db_path)
    ensure_pragmas(conn)
    ensure_indexes(conn)
    ensure_effective_end_ts(conn)

    market_start = load_market_start_ts(conn)
    market_end = load_market_end_ts(conn)

    ckpt = load_latest_checkpoint(cfg.output_dir)
    if ckpt is None:
        user_states: dict[str, UserState] = {}
        market_states: dict[str, MarketState] = {}
        user_market_lifetime: dict[tuple[str, str], dict] = {}

        pending_pnls: list[tuple[int, int, str, float]] = []
        pending_seq = 0

        trades_processed = 0
        chunk_idx = 0
        last_ts = -1
        last_uid = ""
        resume_after = None
        next_checkpoint = cfg.checkpoint_every
    else:
        user_states = ckpt.user_states
        market_states = ckpt.market_states
        user_market_lifetime = ckpt.user_market_lifetime

        pending_pnls = ckpt.pending_pnls
        pending_seq = ckpt.pending_seq

        trades_processed = ckpt.trades_processed
        chunk_idx = ckpt.chunk_idx
        last_ts = ckpt.last_ts
        last_uid = ckpt.last_trade_uid
        resume_after = (last_ts, last_uid)
        next_checkpoint = ((trades_processed // cfg.checkpoint_every) + 1) * cfg.checkpoint_every
    print("Connecting.")
    cur = trades_cursor(conn, cfg.start_ts, resume_after=resume_after)
    print("Conneted!")
    rows = init_rows_buffer()
    rows_in_chunk = 0

    for (
        trade_uid,
        proxy_wallet,
        condition_id,
        ts,
        price,
        size,
        usdc_size,
        outcome_index,
        pnl_raw,
    ) in cur:
        # advance checkpoint cursor
        last_ts = int(ts)
        last_uid = str(trade_uid)

        trades_processed += 1

        # 1) realize PnLs for markets that have ended by now
        flush_pending_pnls(pending_pnls, int(ts), user_states)

        if size is None or size == 0 or price is None:
            continue

        user_id = str(proxy_wallet)
        market_id = str(condition_id)

        # 2) get/create states
        u = user_states.get(user_id)
        if u is None:
            u = UserState.create(int(ts))
            user_states[user_id] = u

        m = market_states.get(market_id)
        if m is None:
            m = MarketState()
            market_states[market_id] = m
        if m.first_ts is None:
            m.first_ts = int(ts)

        um_key = (user_id, market_id)
        um = user_market_lifetime.get(um_key)
        if um is None:
            um = {"total_usdc": 0.0, "trade_count": 0, "last_ts": None}
            user_market_lifetime[um_key] = um

        # 3) schedule this trade's pnl to be realized at market end
        end_ts = market_end.get(market_id)
        pending_seq = schedule_pnl(pending_pnls, pending_seq, end_ts, user_id, pnl_raw)

        # 4) compute features BEFORE state updates
        feats = compute_row_features(
            trade_uid=str(trade_uid),
            user_id=user_id,
            market_id=market_id,
            ts=int(ts),
            price=float(price),
            size=float(size),
            usdc_size=float(usdc_size) if usdc_size is not None else None,
            outcome_index=int(outcome_index),
            pnl_raw=float(pnl_raw) if pnl_raw is not None else None,
            market_start_ts=market_start.get(market_id),
            u=u,
            m=m,
            um=um,
        )
        append_row(rows, feats)
        rows_in_chunk += 1

        # 5) apply updates with this trade
        abs_usdc = abs(float(usdc_size)) if usdc_size is not None else 0.0
        abs_size = abs(float(size))

        apply_trade_updates(
            ts=int(ts),
            price=float(price),
            outcome_index=int(outcome_index),
            abs_usdc=abs_usdc,
            abs_size=abs_size,
            user_id=user_id,
            u=u,
            m=m,
            um=um,
        )

        # 6) write chunk
        if rows_in_chunk >= cfg.chunk_rows:
            write_parquet_chunk(cfg.output_dir, rows, chunk_idx)
            chunk_idx += 1
            rows = init_rows_buffer()
            rows_in_chunk = 0

        # 7) checkpoint
        if trades_processed >= next_checkpoint:
            save_checkpoint(
                cfg.output_dir,
                Checkpoint(
                    trades_processed=trades_processed,
                    chunk_idx=chunk_idx,
                    last_ts=last_ts,
                    last_trade_uid=last_uid,
                    user_states=user_states,
                    market_states=market_states,
                    user_market_lifetime=user_market_lifetime,
                    pending_pnls=pending_pnls,
                    pending_seq=pending_seq,
                ),
            )
            next_checkpoint += cfg.checkpoint_every

    conn.close()

    if rows_in_chunk > 0:
        write_parquet_chunk(cfg.output_dir, rows, chunk_idx)

    print(f"[INFO] Done. Total trades processed: {trades_processed}")


if __name__ == "__main__":
    main()
