# src/grables/datasets/synthetic.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticTxConfig:
    n_rows: int = 10_000
    n_unique_cards: int = 8_000
    n_unique_merchants: int = 7_500
    online_share: float = 0.22
    seed: int = 42


def make_synthetic_transactions(cfg: SyntheticTxConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    if cfg.n_unique_cards > cfg.n_rows:
        raise ValueError("n_unique_cards cannot exceed n_rows.")
    if cfg.n_unique_merchants <= 0 or cfg.n_unique_cards <= 0:
        raise ValueError("n_unique_merchants and n_unique_cards must be positive.")
    if not (0.0 <= cfg.online_share <= 1.0):
        raise ValueError("online_share must be in [0, 1].")

    cities = [
        "Brussels", "Antwerp", "Ghent", "Charleroi", "Liège", "Bruges", "Leuven", "Namur",
        "Paris", "London", "Amsterdam", "Berlin", "Madrid", "Rome", "Vienna", "Zurich",
        "Barcelona", "Rotterdam", "Lille", "Toulouse",
    ]

    merchant_ids = np.arange(1, cfg.n_unique_merchants + 1, dtype=np.int64)
    merchant_is_online = rng.random(cfg.n_unique_merchants) < cfg.online_share
    merchant_city = np.where(
        merchant_is_online,
        "ONLINE",
        rng.choice(cities, size=cfg.n_unique_merchants, replace=True),
    )
    merchant_dim = pd.DataFrame({"merchant_id": merchant_ids, "merchant_city": merchant_city})

    card_pool = np.array([f"C{idx:06d}" for idx in range(1, cfg.n_unique_cards + 1)])
    base_cards = card_pool.copy()

    remaining = cfg.n_rows - cfg.n_unique_cards
    if remaining > 0:
        weights = 1 / (np.arange(1, cfg.n_unique_cards + 1) ** 0.8)
        weights = weights / weights.sum()
        extra_cards = rng.choice(card_pool, size=remaining, replace=True, p=weights)
        card_ids = np.concatenate([base_cards, extra_cards])
    else:
        card_ids = base_cards

    merchant_weights = 1 / (np.arange(1, cfg.n_unique_merchants + 1) ** 0.7)
    merchant_weights = merchant_weights / merchant_weights.sum()
    tx_merchant_ids = rng.choice(merchant_ids, size=cfg.n_rows, replace=True, p=merchant_weights)

    df = pd.DataFrame(
        {
            "id": np.arange(1, cfg.n_rows + 1, dtype=np.int64),
            "card_id": card_ids,
            "merchant_id": tx_merchant_ids,
        }
    ).merge(merchant_dim, on="merchant_id", how="left")

    df = df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    df["id"] = np.arange(1, cfg.n_rows + 1, dtype=np.int64)
    return df


def save_df(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet"}:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
