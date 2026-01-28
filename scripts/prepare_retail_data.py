from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from grables.tasks.logical import apply_task_spec


RETAIL_TASK_SPEC = [
    {"name": "unique", "args": {"column": "StockCode"}},
    {"name": "count", "args": {"column": "InvoiceNo", "k": 75}},
    {"name": "count", "args": {"column": "InvoiceNo", "k": 15, "greater_than": False}},
    {"name": "double", "args": {"col_1": "InvoiceNo", "col_2": "StockCode", "anchor": "23084"}},
    {"name": "diamond", "args": {"col_1": "CustomerID", "col_2": "StockCode"}},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--out_dir", default="data/processed/retail")
    ap.add_argument("--date_col", default="InvoiceDate")
    ap.add_argument("--q0", type=float, default=0.90)
    ap.add_argument("--q1", type=float, default=0.98)
    ap.add_argument("--q2", type=float, default=0.99)
    ap.add_argument("--country_col", default="Country")
    ap.add_argument("--country", default=None)

    ap.add_argument("--full_name", default="df_UK.csv")
    ap.add_argument("--train_name", default="retail_train.csv")
    ap.add_argument("--val_name", default="retail_val.csv")
    ap.add_argument("--test_name", default="retail_test.csv")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.xlsx)

    # Only if you used it
    if args.country is not None:
        df = df[df[args.country_col] == args.country]

    df[args.date_col] = pd.to_datetime(df[args.date_col])

    q0 = df[args.date_col].quantile(args.q0)
    q1 = df[args.date_col].quantile(args.q1)
    q2 = df[args.date_col].quantile(args.q2)

    df_train = df[(df[args.date_col] >= q0) & (df[args.date_col] < q1)]
    df_val = df[(df[args.date_col] >= q1) & (df[args.date_col] < q2)]
    df_test = df[df[args.date_col] >= q2]

    # Same task functions as synthetic—just a different spec
    df_train = apply_task_spec(df_train, RETAIL_TASK_SPEC)
    df_val = apply_task_spec(df_val, RETAIL_TASK_SPEC)
    df_test = apply_task_spec(df_test, RETAIL_TASK_SPEC)

    df.to_csv(out_dir / args.full_name, index=False)
    df_train.to_csv(out_dir / args.train_name, index=False)
    df_val.to_csv(out_dir / args.val_name, index=False)
    df_test.to_csv(out_dir / args.test_name, index=False)


if __name__ == "__main__":
    main()