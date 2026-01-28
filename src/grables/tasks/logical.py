# src/grables/tasks/logical.py
from __future__ import annotations

from typing import Hashable

from __future__ import annotations
from typing import Any, Callable

import pandas as pd


def task_unique(df: pd.DataFrame, column: str, *, out_col: str | None = None) -> pd.DataFrame:
    """
    Label: value in `column` appears exactly once in the dataframe.
    """
    out = df.copy()
    name = out_col or f"unique_{column}"
    vc = out[column].value_counts(dropna=False)
    out[name] = out[column].map(vc).eq(1).astype("int8")
    return out


def task_count(
    df: pd.DataFrame,
    column: str,
    k: int,
    *,
    greater_than: bool = True,
    out_col: str | None = None,
) -> pd.DataFrame:
    """
    Label: count(column value) > k  (or == k if greater_than=False).
    """
    out = df.copy()
    suffix = "gt" if greater_than else "eq"
    name = out_col or f"count_{suffix}_{k}_{column}"

    vc = out[column].value_counts(dropna=False)
    counts = out[column].map(vc)

    if greater_than:
        out[name] = counts.gt(k).astype("int8")
    else:
        out[name] = counts.eq(k).astype("int8")

    return out


def task_double(
    df: pd.DataFrame,
    col_1: str,
    col_2: str,
    anchor: Hashable,
    *,
    out_col: str | None = None,
) -> pd.DataFrame:
    """
    For each row r:
      label = 1 if there exists ANOTHER row r' with r'[col_1] == r[col_1] and r'[col_2] == anchor

    “Another row” means:
      - if row has col_2 != anchor, need >=1 anchor-row in its group
      - if row has col_2 == anchor, need >=2 anchor-rows (since itself doesn't count)
    """
    out = df.copy()
    name = out_col or f"double_{col_1}_{col_2}_{anchor}"

    anchor_mask = out[col_2].eq(anchor)
    anchor_counts = out.loc[anchor_mask, col_1].value_counts(dropna=False)
    counts_per_row = out[col_1].map(anchor_counts).fillna(0)

    out[name] = (
        ((~anchor_mask) & (counts_per_row >= 1))
        | (anchor_mask & (counts_per_row >= 2))
    ).astype("int8")

    return out


def task_diamond(
    df: pd.DataFrame,
    col_1: str,
    col_2: str,
    *,
    strict: bool = False,
    out_col: str | None = None,
) -> pd.DataFrame:
    """
    Label: (col_1, col_2) pair is duplicated.
      - strict=False: count(pair) > 1
      - strict=True:  count(pair) == 2  (matches your notebook's "strict")
    """
    out = df.copy()
    name = out_col or (f"duplicate_{col_1}_{col_2}_strict" if strict else f"duplicate_{col_1}_{col_2}")

    counts = out.groupby([col_1, col_2], dropna=False)[col_1].transform("size")
    out[name] = (counts.eq(2) if strict else counts.gt(1)).astype("int8")
    return out


TASK_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {
    "unique": task_unique,
    "count": task_count,
    "double": task_double,
    "diamond": task_diamond,
    }




def apply_task_spec(df: pd.DataFrame, spec: list[dict[str, Any]]) -> pd.DataFrame:
    """
    spec example:
    [{"name":"unique","args":{"column":"StockCode"}},
    {"name":"count","args":{"column":"InvoiceNo","k":75}},
    ...]
    """
    out = df
    for step in spec:
    fn = TASK_REGISTRY[step["name"]]
    out = fn(out, **step.get("args", {}))
    return out