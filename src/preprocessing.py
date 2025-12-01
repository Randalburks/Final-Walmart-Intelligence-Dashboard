# src/preprocessing.py
from __future__ import annotations
import io
import zipfile
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path("data")

# ------------ small helpers ------------
def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["int64", "int32"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes(include=["float64", "float32"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    return df

def _cat(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

def _read_csv_from_zip(zip_path: Path, inner_csv_name: str | None = None, **read_kwargs) -> pd.DataFrame:
    """Read a CSV inside a .zip without extracting to disk."""
    with zipfile.ZipFile(zip_path) as zf:
        # If inner file name not specified, take the first CSV
        name = inner_csv_name or next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as f:
            return pd.read_csv(f, **read_kwargs)

# ------------ public API ------------
@st.cache_data(show_spinner=False)
def load_merge(
    use_cache: bool = True,
    history_days: int = 120,          # keep this small for Streamlit memory
    keep_cols: Tuple[str, ...] = ("date","store_id","item_id","sales","sell_price","wm_yr_wk","event_name_1","snap"),
) -> pd.DataFrame:
    """
    Build a compact daily panel from the three CSV ZIPs while staying under Streamlit limits:
    - Only the last `history_days` of sales are loaded.
    - Only required columns are kept and downcasted aggressively.
    """

    # ---- calendar (maps d_XXXX -> date; carries SNAP/events) ----
    cal_zip = DATA_DIR / "calendar.csv.zip"
    if not cal_zip.exists():
        raise FileNotFoundError("Missing data/calendar.csv.zip")
    cal = _read_csv_from_zip(cal_zip)
    # Keep minimal calendar fields
    cal = cal[["date", "wm_yr_wk", "d", "event_name_1", "snap_CA", "snap_TX", "snap_WI"]].copy()
    cal["date"] = pd.to_datetime(cal["date"])
    # single snap flag (max of state flags); dtype int8
    cal["snap"] = cal[["snap_CA","snap_TX","snap_WI"]].max(axis=1).fillna(0).astype("int8")
    cal = cal.drop(columns=["snap_CA","snap_TX","snap_WI"])
    _cat(cal, ["event_name_1"])
    cal = _downcast_numeric(cal)

    # ---- figure out which d_ columns to use (only last N) ----
    sales_zip = DATA_DIR / "sales_train_validation.csv.zip"
    if not sales_zip.exists():
        raise FileNotFoundError("Missing data/sales_train_validation.csv.zip")
    # Read header only to discover all day columns
    hdr = _read_csv_from_zip(sales_zip, nrows=0)
    day_cols = [c for c in hdr.columns if c.startswith("d_")]
    if not day_cols:
        raise ValueError("No day columns (d_****) found in sales file.")
    day_cols = day_cols[-history_days:]  # last N days only

    base_id_cols = ["id","item_id","dept_id","cat_id","store_id","state_id"]
    usecols = base_id_cols + day_cols

    # Now read only needed columns
    sales = _read_csv_from_zip(sales_zip, usecols=usecols)
    _cat(sales, ["item_id","dept_id","cat_id","store_id","state_id"])

    # melt to long only for last N days
    long = sales.melt(
        id_vars=base_id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales"
    )
    long["sales"] = pd.to_numeric(long["sales"], downcast="integer").fillna(0)

    # merge on calendar to get date + wm_yr_wk + events + snap
    long = long.merge(cal, on="d", how="left").drop(columns=["d"])
    # keep minimal set
    keep = [c for c in keep_cols if c in long.columns]
    long = long[keep + ["cat_id","dept_id"]].copy() if "cat_id" in long.columns and "dept_id" in long.columns else long[keep].copy()

    # ---- prices (weekly level) ----
    price_zip = DATA_DIR / "sell_prices.csv.zip"
    if not price_zip.exists():
        raise FileNotFoundError("Missing data/sell_prices.csv.zip")
    prices = _read_csv_from_zip(price_zip, usecols=["store_id","item_id","wm_yr_wk","sell_price"])
    prices["sell_price"] = pd.to_numeric(prices["sell_price"], downcast="float")
    _cat(prices, ["store_id","item_id"])

    # join price by store/item/week; forward-fill within each (store,item) to daily
    panel = long.merge(prices, on=["store_id","item_id","wm_yr_wk"], how="left")
    panel = panel.sort_values(["store_id","item_id","date"])
    panel["sell_price"] = panel.groupby(["store_id","item_id"], observed=True)["sell_price"].ffill().bfill()
    panel["sell_price"] = panel["sell_price"].astype("float32")

    # downcast IDs and categories
    _cat(panel, ["store_id","item_id","cat_id","dept_id","event_name_1"])
    panel["sales"] = pd.to_numeric(panel["sales"], downcast="integer")
    panel["snap"]  = panel["snap"].astype("int8")
    panel = _downcast_numeric(panel)

    return panel.reset_index(drop=True)


def sample_panel(base: pd.DataFrame, n_stores: int = 3, n_items: int = 30) -> pd.DataFrame:
    """
    Keep a small, representative slice (top items by recent volume per store).
    """
    if base.empty:
        return base

    # choose the top n_stores by total recent units
    store_rank = base.groupby("store_id", observed=True)["sales"].sum().sort_values(ascending=False)
    top_stores = store_rank.index.tolist()[:n_stores]

    out = []
    for s in top_stores:
        g = base[base["store_id"] == s]
        # pick recent top items in this store
        top_items = (
            g.groupby("item_id", observed=True)["sales"].sum()
            .sort_values(ascending=False)
            .index.tolist()[:n_items]
        )
        out.append(g[g["item_id"].isin(top_items)])
    panel = pd.concat(out, axis=0).reset_index(drop=True)
    return panel
