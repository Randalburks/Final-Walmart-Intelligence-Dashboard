# src/utils.py
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px

def safe_num(x):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return 0.0

def kpis_for_slice(df: pd.DataFrame) -> dict:
    out = {"avg_units": 0.0, "avg_price": 0.0, "wow_pct": 0.0}
    if df.empty:
        return out
    s = df.set_index("date")["sales"].groupby(level=0).sum().asfreq("D").fillna(0.0)
    p = df.set_index("date")["sell_price"].groupby(level=0).mean().asfreq("D")
    out["avg_units"] = float(s.mean())
    out["avg_price"] = float(np.nanmean(p))
    w = s.resample("W").sum()
    if len(w) >= 2:
        prev, curr = float(w.iloc[-2]), float(w.iloc[-1])
        out["wow_pct"] = float(100*(curr - prev)/(prev + 1e-9))
    return out

def anomaly_flags(df: pd.DataFrame, window: int = 7, z: float = 3.0) -> pd.DataFrame:
    if df.empty:
        return df.assign(anomaly=0)
    s = df.set_index("date")["sales"].groupby(level=0).sum().asfreq("D").fillna(0.0)
    mu = s.rolling(window, min_periods=max(2, window//2)).mean()
    sd = s.rolling(window, min_periods=max(2, window//2)).std(ddof=0)
    zscore = (s - mu) / (sd.replace(0, np.nan))
    out = pd.DataFrame({"date": s.index, "sales": s.values})
    out["anomaly"] = (np.abs(zscore) > z).astype(int)
    return out

def seasonal_naive(series: pd.Series, horizon: int = 28, season: int = 7) -> np.ndarray:
    if len(series) < season:
        last = float(series.iloc[-1]) if len(series) > 0 else 0.0
        return np.repeat(last, horizon)
    tail = series.iloc[-season:].values
    reps = int(np.ceil(horizon / season))
    fc = np.tile(tail, reps)[:horizon]
    return fc.astype(float)

def moving_avg(series: pd.Series, horizon: int = 28, window: int = 7) -> np.ndarray:
    if len(series) < window:
        last = float(series.iloc[-1]) if len(series) > 0 else 0.0
        return np.repeat(last, horizon)
    ma = series.rolling(window, min_periods=max(2, window//2)).mean().iloc[-1]
    return np.repeat(float(ma), horizon)

# --------- IDA/EDA helpers ----------
def info_table(df_: pd.DataFrame) -> pd.DataFrame:
    non_null = df_.notnull().sum()
    out = pd.DataFrame({
        "column": df_.columns,
        "dtype": [str(t) for t in df_.dtypes],
        "non_null": [int(non_null[c]) for c in df_.columns],
        "nulls": [int(len(df_) - non_null[c]) for c in df_.columns]
    })
    out["null_pct"] = (out["nulls"] / max(1, len(df_))).round(4)
    return out

def missingness_bar(df_: pd.DataFrame):
    nn = df_.isnull().mean().sort_values(ascending=False).reset_index()
    nn.columns = ["column","null_fraction"]
    fig = px.bar(nn, x="column", y="null_fraction", title="Missingness by Column")
    fig.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".0%")
    return fig

def missingness_heatmap(df_: pd.DataFrame, sample_rows:int=1200):
    smp = df_.sample(min(sample_rows, len(df_)), random_state=42).isnull().astype(int)
    fig = px.imshow(smp.T, color_continuous_scale="Blues",
                    labels=dict(x="Sampled Row", y="Column", color="Is Null"),
                    title="Missingness Heatmap (sampled rows)")
    return fig

def corr_heatmap(df_: pd.DataFrame, cols: list[str] | None = None):
    if cols is None:
        cols = [c for c in ["sales", "sell_price", "snap"] if c in df_.columns]
        # add at most 5 more numeric columns to keep it readable
        extra = [c for c in df_.select_dtypes(include="number").columns if c not in cols]
        cols += extra[:5]
    if not cols:
        return None
    mat = df_[cols].select_dtypes(include="number").corr().round(3)
    fig = px.imshow(mat, text_auto=True, aspect="auto", title="Correlation Matrix (numeric features)")
    return fig