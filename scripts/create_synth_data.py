# scripts/make_synth_tasks.py
from __future__ import annotations

import argparse
from pathlib import Path

from grables.data.synthetic import SyntheticTxConfig, make_synthetic_transactions, save_df
from grables.tasks.logical import task_unique, task_count, task_double, task_diamond


def add_all_tasks(df):
    df = task_unique(df, "merchant_id")
    df = task_count(df, "card_id", 12, greater_than=True)
    df = task_count(df, "card_id", 3, greater_than=False)
    df = task_double(df, "card_id", "merchant_city", "ONLINE")
    df = task_diamond(df, "card_id", "merchant_city", strict=False)
    df = task_diamond(df, "card_id", "merchant_city", strict=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/processed/synthetic")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": SyntheticTxConfig(n_rows=8000, n_unique_cards=2500, n_unique_merchants=3500, online_share=0.15, seed=args.seed),
        "val":   SyntheticTxConfig(n_rows=1000, n_unique_cards=350,  n_unique_merchants=300,  online_share=0.12, seed=args.seed + 1),
        "test":  SyntheticTxConfig(n_rows=1000, n_unique_cards=350,  n_unique_merchants=300,  online_share=0.12, seed=args.seed + 2),
    }

    for split, cfg in splits.items():
        df = make_synthetic_transactions(cfg)
        df = add_all_tasks(df)
        save_df(df, out_dir / f"{split}.parquet")
        print(f"Wrote {split}: {len(df):,} rows -> {(out_dir / f'{split}.parquet')}")

if __name__ == "__main__":
    main()