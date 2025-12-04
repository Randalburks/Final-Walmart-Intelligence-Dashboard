# app.py
from __future__ import annotations
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor  # ADDED: second ML model for final project

from src.preprocessing import load_merge, sample_panel
from src.utils import (
    kpis_for_slice,
    anomaly_flags,
    seasonal_naive,
    moving_avg,
    safe_num,
)

# --------------------- Page config & Title ---------------------
st.set_page_config(
    page_title="Walmart Intelligence — Executive Sales, Pricing, Forecasting & Scenarios",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Walmart Intelligence — Executive Sales, Pricing, Forecasting & Scenarios")
st.caption(
    "The purpose of this project is to create an end-to-end analytics dashboard that allows Walmart leadership to quickly understand sales performance, pricing behavior, operational patterns, and forecasted demand across stores and items. My goal is to take the raw Kaggle M5 dataset and transform it through the full data science workflow, including data cleaning, feature engineering, anomaly detection, forecasting models, and scenario simulations. I wanted to design a dashboard that organizes complex information into clear visuals so leadership can access insights instantly without reviewing code or raw data. The project aims to highlight the key factors driving performance, compare models with measurable metrics, and identify meaningful trends across different store and item segments. I also focused on making the tool actionable by incorporating interactive filters and scenario levers that support real decision-making. Through this approach, the dashboard becomes a streamlined, mobile-friendly resource that answers high-level business questions on demand."
)

# --------------------- Small helpers for page descriptions ---------------------
def tab_intro(title: str, what: str, why: str, how: str, read: str = ""):
    st.subheader(title)
    with st.expander(title, expanded=True):
        st.markdown(f"**What this shows:** {what}")
        st.markdown(f"**Why it matters:** {why}")
        st.markdown(f"**How to use it:** {how}")
        if read:
            st.caption(read)

def describe_chart(header: str, meaning: str, action: str = ""):
    st.markdown(f"**About this visualization — {header}**")
    st.markdown(meaning)
    if action:
        st.caption(action)

# --------------------- Load data ---------------------
try:
    base = load_merge(use_cache=True)  # reads the three ZIPs and returns a tidy daily panel
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# --------------------- Sidebar controls ---------------------
with st.sidebar:
    st.header("Filters")
    n_stores = st.number_input("How many stores to include", 1, 10, 3, step=1)
    n_items = st.number_input("How many products per store", 5, 200, 30, step=5)
    st.markdown("---")
    st.markdown("Upload optional costs to enable profit tiles.")
    cost_file = st.file_uploader("costs.csv (store_id,item_id,unit_cost)", type=["csv"])

# create a smaller working panel for snappier UI
panel = sample_panel(base, int(n_stores), int(n_items))

costs_df = None
if cost_file is not None:
    try:
        tmp = pd.read_csv(cost_file)
        if {"store_id", "item_id", "unit_cost"}.issubset(tmp.columns):
            costs_df = tmp.copy()
        else:
            st.sidebar.warning("Expected columns: store_id, item_id, unit_cost")
    except Exception as e:
        st.sidebar.warning(f"Could not read costs.csv: {e}")

stores = sorted(panel["store_id"].dropna().unique().tolist())
store_sel = st.sidebar.selectbox("Store", stores, index=0 if stores else None)
items = sorted(panel.loc[panel["store_id"] == store_sel, "item_id"].dropna().unique().tolist())
item_sel = st.sidebar.selectbox("Product", items, index=0 if items else None)

dmin, dmax = pd.to_datetime(panel["date"].min()), pd.to_datetime(panel["date"].max())
dr = st.sidebar.date_input(
    "Date range",
    (dmin.date(), dmax.date()),
    min_value=dmin.date(),
    max_value=dmax.date(),
)
start_ts = pd.Timestamp(dr[0])
end_ts = pd.Timestamp(dr[1])

slice_df = panel[
    (panel["store_id"] == store_sel)
    & (panel["item_id"] == item_sel)
    & (panel["date"].between(start_ts, end_ts))
].copy()

if costs_df is not None and not slice_df.empty:
    slice_df = slice_df.merge(costs_df, on=["store_id", "item_id"], how="left")

# --------------------- Analytics helpers ---------------------
def estimate_elasticity(df_slice: pd.DataFrame) -> Optional[dict]:
    g = df_slice.dropna(subset=["sell_price", "sales"])
    g = g[(g["sell_price"] > 0) & (g["sales"] > 0)]
    if len(g) < 30:
        return None
    X = np.log(g[["sell_price"]].values)
    y = np.log(g["sales"].values)
    m = LinearRegression().fit(X, y)
    yhat = m.predict(X)
    return {
        "elasticity": float(m.coef_[0]),
        "intercept": float(m.intercept_),
        "mape": float(mean_absolute_percentage_error(y, yhat)),
        "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
        "n": int(len(g)),
    }

def simulate_revenue(df_slice: pd.DataFrame, elasticity: float, pct: float) -> dict:
    base_qty = safe_num(df_slice["sales"].mean())
    base_price = safe_num(df_slice["sell_price"].mean())
    new_price = base_price * (1 + pct)
    new_qty = base_qty * (1 + pct) ** elasticity
    base_rev = base_qty * base_price
    new_rev = new_qty * new_price
    return {
        "new_price": float(new_price),
        "new_qty": float(new_qty),
        "delta_rev_pct": float((new_rev - base_rev) / (base_rev + 1e-9) * 100),
    }

def profit_kpis(df_slice: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if "unit_cost" not in df_slice.columns or df_slice["unit_cost"].isna().all():
        return None, None, None
    qty = safe_num(df_slice["sales"].mean())
    price = safe_num(df_slice["sell_price"].mean())
    cost = safe_num(df_slice["unit_cost"].mean())
    margin = price - cost
    profit = margin * qty
    margin_pct = (margin / (price + 1e-9)) * 100 if price > 0 else None
    return float(profit), float(margin), float(margin_pct) if margin_pct is not None else None

def concise_summary(sl: pd.DataFrame, elasticity_val: Optional[float]) -> str:
    if sl.empty:
        return "No data in view. Adjust the date or product."
    k = kpis_for_slice(sl)
    parts = []
    if k["wow_pct"] > 3:
        parts.append(f"Sales up {k['wow_pct']:.1f}% vs last week.")
    elif k["wow_pct"] < -3:
        parts.append(f"Sales down {abs(k['wow_pct']):.1f}% vs last week.")
    else:
        parts.append("Sales are roughly flat week-over-week.")
    parts.append(f"Avg price ${k['avg_price']:.2f}.")
    if elasticity_val is not None:
        parts.append("Price-sensitive product." if elasticity_val < -1.0 else "Demand relatively steady vs price.")
    if "event_name_1" in sl.columns:
        recent = sl.loc[sl["event_name_1"].notna(), "event_name_1"].tail(3).unique().tolist()
        if recent:
            parts.append("Recent events: " + ", ".join(recent[:3]) + ".")
    return " ".join(parts)

def add_event_overlays(fig: go.Figure, df_: pd.DataFrame) -> go.Figure:
    if "event_name_1" not in df_.columns:
        return fig
    ev = df_.loc[df_["event_name_1"].notna(), ["date", "event_name_1"]].dropna()
    ymax = max(df_["sales"].max() if "sales" in df_.columns else 0, 1)
    for _, row in ev.iterrows():
        fig.add_vrect(x0=row["date"], x1=row["date"], fillcolor="orange", opacity=0.08, line_width=0)
        fig.add_annotation(
            x=row["date"], y=ymax, text=str(row["event_name_1"]), showarrow=False, yshift=18, font=dict(size=10, color="gray")
        )
    return fig

def compute_wow(group_df: pd.DataFrame, freq="W") -> float:
    # guard against duplicate dates
    s = group_df.groupby("date", as_index=True)["sales"].sum().resample(freq).sum()
    if len(s) < 2:
        return 0.0
    prev, curr = float(s.iloc[-2]), float(s.iloc[-1])
    return float(100 * (curr - prev) / (prev + 1e-9))

# ---------- IDA helpers (EDA utilities used on Tab 0) ----------
def info_table(df_: pd.DataFrame) -> pd.DataFrame:
    non_null = df_.notnull().sum()
    out = pd.DataFrame({
        "column": df_.columns,
        "dtype": [str(t) for t in df_.dtypes],
        "non_null": [int(non_null[c]) for c in df_.columns],
        "nulls": [int(len(df_) - non_null[c]) for c in df_.columns],
    })
    out["null_pct"] = (out["nulls"] / max(1, len(df_))).round(4)
    return out

def missingness_bar(df_: pd.DataFrame):
    nn = df_.isnull().mean().sort_values(ascending=False).reset_index()
    nn.columns = ["column", "null_fraction"]
    fig = px.bar(nn, x="column", y="null_fraction", title="Missingness by Column")
    fig.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".0%")
    return fig

def missingness_heatmap(df_: pd.DataFrame, sample_rows: int = 1200):
    smp = df_.sample(min(sample_rows, len(df_)), random_state=42).isnull().astype(int)
    fig = px.imshow(
        smp.T,
        color_continuous_scale="Blues",
        labels=dict(x="Sampled Row", y="Column", color="Is Null"),
        title="Missingness Heatmap (sampled rows)",
        aspect="auto",
    )
    return fig

def corr_heatmap(df_: pd.DataFrame, cols: List[str] | None = None):
    if cols is None:
        cols = [c for c in ["sales", "sell_price", "wm_yr_wk", "snap"] if c in df_.columns]
        extra = [c for c in df_.select_dtypes(include="number").columns if c not in cols]
        cols += extra[:8]
    num = df_[cols].select_dtypes(include="number")
    if num.empty or num.shape[1] < 2:
        return None
    mat = num.corr().round(3)
    fig = px.imshow(mat, text_auto=True, aspect="auto", title="Correlation Matrix (numeric features)")
    return fig

# --------------------- Tabs ---------------------
tabs = st.tabs([
    "IDA + Preprocessing",
    "Overview",
    "Forecast",
    "Price Sensitivity",
    "Scenario Compare",
    "Compare Segments",
    "Top Performers",
    "Summary Export",
])

# ===================== Tab 0: IDA + Preprocessing =====================
with tabs[0]:
    tab_intro(
        title="IDA + Preprocessing",
        what="Schema, completeness, distributions, correlation matrix, and the exact cleaning used to build the daily panel.",
        why="Builds trust in the data and explains transformations that downstream tabs depend on (daily alignment, price carry-fill).",
        how="Scan structure & missingness, then the correlation matrix for expected relationships (e.g., price vs sales).",
        read="Data source: three CSVs shipped as ZIPs (calendar, prices, sales).",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in sampled panel", f"{len(panel):,}")
    c2.metric("Columns", f"{panel.shape[1]}")
    c3.metric("Date coverage", f"{dmin.date()} → {dmax.date()}")

    st.markdown("#### 1) Structure")
    st.dataframe(info_table(panel), use_container_width=True)
    describe_chart(
        "Structure table",
        "Types, non-null counts, and where gaps exist. Event fields are sparse by design; price/sales should be largely populated.",
        "If key fields are mostly null for a segment, avoid heavy modeling on that segment until data improves.",
    )

    st.markdown("#### 2) Numeric summary")
    num_cols = panel.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        st.dataframe(panel[num_cols].describe().T.round(3), use_container_width=True)
    else:
        st.info("No numeric columns found.")
    describe_chart(
        "Numeric summary",
        "This table is the statistical summary for all numeric features. It shows count, mean, standard deviation, min, quartiles, and max for each variable.",
        "Large spreads imply higher forecast uncertainty; compare mean vs median to check skew and look at extreme min / max values for possible outliers.",
    )

    st.markdown("#### 3) Missingness")
    cA, cB = st.columns(2)
    with cA:
        st.plotly_chart(missingness_bar(panel), use_container_width=True)
    with cB:
        st.plotly_chart(missingness_heatmap(panel), use_container_width=True)
    describe_chart(
        "Missingness views",
        "The bar ranks columns by null fraction; the heatmap shows if missingness clusters. Narrative event text is naturally sparse.",
        "Residual gaps in price are handled via within-item forward/back fill to make daily alignment reliable.",
    )

    st.markdown("#### 4) Correlation matrix")
    corr_fig = corr_heatmap(panel)
    if corr_fig is not None:
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns to compute a correlation matrix.")
    describe_chart(
        "Correlation matrix",
        "Shows linear relationships among numeric features (for example, sales vs price, sales vs SNAP). A negative price–sales correlation is expected for elastic products.",
        "Use it to sanity-check the elasticity direction for selected products and to understand which drivers move together.",
    )

    st.markdown("#### 5) Behavioral signals")
    ts_all = panel.groupby("date", as_index=False)["sales"].sum()
    st.plotly_chart(px.line(ts_all, x="date", y="sales", title="All-sampled items — daily sales"), use_container_width=True)
    describe_chart(
        "Daily sales timeline",
        "Reveals baseline volatility and weekly rhythm; spikes often align with events, dips with outages or stockouts.",
        "This justifies a weekly seasonal baseline in the Forecast tab.",
    )

    st.markdown("#### 6) Imputation diagnostics (sell_price)")
    price_missing_after = float(panel["sell_price"].isna().mean()) if "sell_price" in panel.columns else 1.0
    st.caption(f"Price still missing after within-item carry-fill: {price_missing_after:.1%}")
    samp = panel[["item_id", "sell_price"]].dropna()
    if not samp.empty:
        samp = samp.sample(min(20000, len(samp)), random_state=42)
        st.plotly_chart(
            px.violin(
                samp,
                y="sell_price",
                points=False,
                box=True,
                title="Filled price distribution",
            ),
            use_container_width=True,
        )
    describe_chart(
        "Price distribution after fill",
        "This plot summarizes the distribution of `sell_price` **after** imputation. Prices inside each store–item pair are forward- and backward-filled so that every day has a price when possible.",
        "Only price is imputed; unit sales remain as observed. Extreme outliers in price can be flagged for future capping before elasticity analysis.",
    )

    st.markdown("#### 7) Auto insights")
    insights = []
    try:
        dow_means = panel.assign(dow=panel["date"].dt.day_name()).groupby("dow", as_index=False)["sales"].mean()
        wknd = dow_means.query("dow in ['Saturday','Sunday']")["sales"].mean()
        mid = dow_means.query("dow in ['Tuesday','Wednesday','Thursday']")["sales"].mean()
        if np.isfinite(wknd) and np.isfinite(mid) and mid > 0:
            uplift = 100 * (wknd - mid) / (mid + 1e-9)
            if abs(uplift) >= 5:
                insights.append(
                    f"Weekly pattern present: weekend vs mid-week ≈ {uplift:+.1f}%. Supports a weekly seasonal baseline."
                )
    except Exception:
        pass

    if "event_name_1" in panel.columns:
        with_ev = panel.loc[panel["event_name_1"].notna(), "sales"].mean()
        no_ev = panel.loc[panel["event_name_1"].isna(), "sales"].mean()
        if np.isfinite(with_ev) and np.isfinite(no_ev) and no_ev > 0:
            lift = 100 * (with_ev - no_ev) / no_ev
            if abs(lift) >= 3:
                insights.append(
                    f"Events relate to demand: event days differ by ≈ {lift:+.1f}% vs non-event days."
                )

    if "snap" in panel.columns and panel["snap"].nunique() > 1:
        snap_yes = panel.loc[panel["snap"] == 1, "sales"].mean()
        snap_no = panel.loc[panel["snap"] == 0, "sales"].mean()
        if np.isfinite(snap_yes) and np.isfinite(snap_no) and snap_no > 0:
            snap_lift = 100 * (snap_yes - snap_no) / snap_no
            insights.append(
                f"SNAP day signal present: ≈ {snap_lift:+.1f}% difference on average (varies by store/item)."
            )

    # rough count of elasticity-ready pairs
    ready = 0
    for (_s, _i), g in panel.groupby(["store_id", "item_id"]):
        g2 = g.dropna(subset=["sell_price", "sales"])
        g2 = g2[(g2["sell_price"] > 0) & (g2["sales"] > 0)]
        if len(g2) >= 30:
            ready += 1
    insights.append(
        f"Elasticity can be estimated for ~{ready} product/store pairs in the sampled panel (≥ 30 valid days)."
    )

    if insights:
        st.markdown("- " + "\n- ".join(insights))

    # --------- 8) Explicit data collection + encoding + imputation narrative ----------
    st.markdown("### 8) How data collection, encoding, and imputation are done in this project")

    st.markdown(
        """
**Data collection and dataset combination**

This dashboard explicitly combines **three raw CSV files from the Walmart M5 dataset** into one unified daily panel:

- `sales_train_validation.csv`: daily unit sales for each `store_id` × `item_id` combination, originally in wide `d_1 ... d_N` format.  
- `calendar.csv`: maps each `d_` column to a real calendar `date`, `wm_yr_wk` (Walmart week), SNAP flags, and event descriptions.  
- `sell_prices.csv`: weekly `sell_price` by `store_id`, `item_id`, and `wm_yr_wk`.

In the preprocessing step (`load_merge` in `src.preprocessing`), these are merged in the following way:

1. The sales file is **unpivoted** from wide (`d_1 ... d_N`) to long format so that each row is a single day of sales for one product in one store.  
2. The long sales table is **joined to `calendar.csv`** using the day key (`d` → `date`, `wm_yr_wk`, events, SNAP, etc.).  
3. The result is **joined to `sell_prices.csv`** on `store_id`, `item_id`, and `wm_yr_wk` so that each store–item–date row has the correct weekly price.  
4. For each `store_id` × `item_id` pair, the code enforces a **dense daily index**, so the final panel contains a continuous time-series of days even when some prices are missing.

This is the multi-source integration pipeline referred to in the project description and rubric.
"""
    )

    st.markdown(
        """
**Encoding and feature engineering**

Several variables are encoded or engineered to make them usable for modeling and visualization:

- **SNAP:** multiple state-specific SNAP flags from `calendar.csv` are collapsed into a single integer `snap` indicator (1 = SNAP active for that store's state, 0 = not active).  
- **Events:** event names and types (for example, `event_name_1`, `event_type_1`) are kept as categorical text features and are used for overlays and group comparisons rather than one-hot encoded columns to keep storage low.  
- **Date features:** from the `date` column we can derive day-of-week, week-of-year, and similar calendar encodings when needed (for example, the day-of-week calculation used in the auto-insights section above).  
- **Identifiers:** `store_id`, `item_id`, `dept_id`, and `cat_id` remain as encoded categorical keys that drive filtering, grouping, and ranking in later tabs.

This section explicitly documents the **encoding choices** that were applied after the raw CSVs were combined.
"""
    )

    st.markdown(
        """
**Imputation strategy**

The project uses a **targeted imputation approach** focused on price:

- For each `store_id` × `item_id` pair, missing `sell_price` values are filled using forward-fill and backward-fill within that product’s own history. This creates a stable, dense price series suitable for elasticity estimation and forecasting.  
- Event fields and SNAP indicators are **not imputed**; when an event is missing, it is treated as "no special event."  
- Unit sales (`sales`) are **left as observed** without imputation, to avoid fabricating demand. Any structural zeros or gaps remain visible in the plots and summaries.

The violin plot above is used as an **imputation diagnostic** to show the distribution of filled prices and to confirm that the carry-fill produced realistic values.
"""
    )

# ===================== Tab 1: Overview =====================
with tabs[1]:
    tab_intro(
        title=f"Overview — {store_sel} / {item_sel}",
        what="Key KPIs for the selected product and store, plus a daily sales line with unusual days and any event overlays.",
        why="Gets everyone on the same page quickly—level, trend, volatility, and notable days.",
        how="Pick a store, product, and date range in the left panel. Hover the line to read event labels.",
    )
    if slice_df.empty:
        st.info("No data in the selected range.")
    else:
        k = kpis_for_slice(slice_df)
        profit_val, margin_val, margin_pct = profit_kpis(slice_df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Daily Units", f"{k['avg_units']:.1f}")
        c2.metric("Avg Price", f"${k['avg_price']:.2f}")
        c3.metric("Week-over-Week", f"{k['wow_pct']:+.1f}%")
        if profit_val is not None:
            c4.metric("Avg Daily Profit", f"${profit_val:,.2f}")
        else:
            c4.metric("Days", f"{len(slice_df):,}")

        if profit_val is not None:
            s1, s2 = st.columns(2)
            s1.metric("Avg Margin ($/unit)", f"${margin_val:,.2f}")
            s2.metric("Avg Margin (%)", f"{margin_pct:.1f}%" if margin_pct is not None else "—")
        else:
            st.caption("Upload costs.csv to enable profit and margin tiles.")

        flagged = anomaly_flags(slice_df, window=7, z=3.0)
        fig = px.line(flagged, x="date", y="sales", labels={"date": "Date", "sales": "Units"})
        if flagged["anomaly"].sum() > 0:
            a = flagged[flagged["anomaly"] == 1]
            fig.add_scatter(x=a["date"], y=a["sales"], mode="markers", name="Unusual day")
        fig = add_event_overlays(fig, slice_df)
        st.plotly_chart(fig, use_container_width=True)
        describe_chart(
            "Daily sales with flags",
            "Highlights unusual days that may be caused by promotions, outages, or data issues. Events are annotated when available.",
        )

        elas = estimate_elasticity(slice_df)
        st.info(concise_summary(slice_df, elas["elasticity"] if elas else None))

# ===================== Tab 2: Forecast =====================
with tabs[2]:
    tab_intro(
        title="Forecast — next 4 weeks",
        what="Two simple 4-week baselines: one respects weekly seasonality; one smooths recent trend.",
        why="Fast planning signal for inventory and staffing when a heavy model isn’t necessary.",
        how="Pick ≥ 30 days of history. Compare the two curves to judge direction and rough magnitude.",
        read="For stronger accuracy, layer promotions and external drivers in a future version.",
    )
    if slice_df.empty:
        st.info("No data in the selected range.")
    else:
        # guard: aggregate per calendar date to avoid duplicate index errors
        series = (
            slice_df.groupby("date", as_index=True)["sales"]
            .sum()
            .asfreq("D")
            .fillna(0.0)
        )
        if len(series) < 30:
            st.warning("Select a longer range (30+ days) for a steadier forecast.")
        else:
            fut_idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=28, freq="D")
            fc1 = seasonal_naive(series, horizon=28, season=7)
            fc2 = moving_avg(series, horizon=28, window=7)
            fig = px.line(x=series.index, y=series.values, labels={"x": "Date", "y": "Units"})
            fig.add_scatter(x=fut_idx, y=fc1, name="Season-aware (weekly)")
            fig.add_scatter(x=fut_idx, y=fc2, name="Smoothed trend")
            st.plotly_chart(fig, use_container_width=True)
            describe_chart(
                "Baseline forecasts",
                "Weekly seasonal baseline mirrors the recent weekly cycle; moving average smooths short-term noise.",
                "Use both as bookends; reality will vary with price and events.",
            )

            # ---- ADDED: Final-project model-based forecast comparison (feature engineering + 2 ML models) ----
            st.markdown("#### Model-based forecast comparison (final project)")

            if len(series) >= 60:
                # Feature engineering: lag features and day-of-week
                df_model = series.to_frame(name="sales").copy()
                for lag in range(1, 8):
                    df_model[f"lag_{lag}"] = df_model["sales"].shift(lag)
                df_model["dow"] = df_model.index.dayofweek  # 0=Mon,...,6=Sun

                df_model = df_model.dropna()
                feature_cols = [c for c in df_model.columns if c != "sales"]
                X = df_model[feature_cols].values
                y = df_model["sales"].values

                # Hold out the last 28 days for evaluation
                if len(df_model) > 56:
                    split = -28
                else:
                    split = int(len(df_model) * 0.8)

                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                # Model 1: LinearRegression on engineered features
                lr_model = LinearRegression().fit(X_train, y_train)
                preds_lr = lr_model.predict(X_test)

                # Model 2: RandomForestRegressor on same features
                rf_model = RandomForestRegressor(
                    n_estimators=150,
                    random_state=42,
                    n_jobs=-1,
                ).fit(X_train, y_train)
                preds_rf = rf_model.predict(X_test)

                rmse_lr = float(np.sqrt(mean_squared_error(y_test, preds_lr)))
                rmse_rf = float(np.sqrt(mean_squared_error(y_test, preds_rf)))
                mape_lr = float(mean_absolute_percentage_error(y_test, preds_lr) * 100)
                mape_rf = float(mean_absolute_percentage_error(y_test, preds_rf) * 100)

                metrics = pd.DataFrame(
                    {
                        "Model": [
                            "LinearRegression (lags + day-of-week)",
                            "RandomForest (lags + day-of-week)",
                        ],
                        "RMSE": [rmse_lr, rmse_rf],
                        "MAPE %": [mape_lr, mape_rf],
                    }
                )

                st.dataframe(
                    metrics.style.format({"RMSE": "{:.2f}", "MAPE %": "{:.1f}"}),
                    use_container_width=True,
                )

                # Plot actual vs predictions on the holdout window
                dates_test = df_model.index[split:]
                fig_models = px.line()
                fig_models.add_scatter(
                    x=dates_test,
                    y=y_test,
                    name="Actual (holdout)",
                )
                fig_models.add_scatter(
                    x=dates_test,
                    y=preds_lr,
                    name="LinearRegression (pred)",
                )
                fig_models.add_scatter(
                    x=dates_test,
                    y=preds_rf,
                    name="RandomForest (pred)",
                )
                fig_models.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Units",
                    title="Model comparison on holdout window",
                )
                st.plotly_chart(fig_models, use_container_width=True)

                describe_chart(
                    "Model-based forecast comparison",
                    "Two supervised models are fit on engineered features (lags and day-of-week). The table summarizes RMSE and MAPE on the holdout window, and the line chart shows how well each model tracks actual demand.",
                    "Use this section to demonstrate model quality against a concrete test period and to justify which model would be used for future production forecasts.",
                )
            else:
                st.caption("Not enough history yet for a stable supervised model comparison (needs ≈ 60+ days).")

# ===================== Tab 3: Price Sensitivity =====================
with tabs[3]:
    tab_intro(
        title="Price Sensitivity (elasticity)",
        what="Elasticity estimates how much units change (%) for a 1% price change, plus a peer comparison histogram.",
        why="Helps decide whether to prioritize volume (elastic) or margin (inelastic).",
        how="If there’s enough price variation, you’ll see an elasticity for the selected product; the histogram shows peer context.",
    )
    if slice_df.empty:
        st.info("No data in the selected range.")
    else:
        elas = estimate_elasticity(slice_df)
        if not elas:
            st.info("Insufficient price variation/history (needs ~30+ valid days).")
        else:
            st.success(f"Elasticity: {elas['elasticity']:.2f} (n={elas['n']}, MAPE={elas['mape']:.3f})")
            vals = []
            for (_s, _i), g in panel.groupby(["store_id", "item_id"]):
                e = estimate_elasticity(g)
                if e:
                    vals.append(e["elasticity"])
            if vals:
                h = pd.Series(vals).clip(-5, 5)
                fig = px.histogram(h, nbins=40, labels={"value": "Elasticity", "count": "Products"})
                fig.add_vline(x=elas["elasticity"], line_dash="dot", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
                describe_chart(
                    "Elasticity distribution",
                    "Shows where this SKU sits vs peers. Red line is the current item’s estimate.",
                )

# ===================== Tab 4: Scenario Compare =====================
with tabs[4]:
    tab_intro(
        title="Scenario Compare",
        what="Three price scenarios side-by-side with projected revenue index and a downloadable price plan.",
        why="Quickly tests direction and rough magnitude of price moves before execution.",
        how="Enter three price deltas (− for discounts, + for increases). Review the bars and download the recommended plan.",
    )
    if slice_df.empty:
        st.info("No data in the selected range.")
    else:
        e = estimate_elasticity(slice_df)
        if not e:
            st.info("Price sensitivity not available for this product/date window.")
        else:
            elas_val = e["elasticity"]
            c1, c2, c3 = st.columns(3)
            s1 = c1.number_input("Scenario A (Δ Price %)", -50, 50, -10)
            s2 = c2.number_input("Scenario B (Δ Price %)", -50, 50, 0)
            s3 = c3.number_input("Scenario C (Δ Price %)", -50, 50, 10)
            scenarios = {"A": s1, "B": s2, "C": s3}
            out = []
            for name, pct in scenarios.items():
                r = simulate_revenue(slice_df, elas_val, pct / 100.0)
                out.append(
                    {
                        "Scenario": name,
                        "Δ Price %": pct,
                        "New Price": r["new_price"],
                        "Est Units": r["new_qty"],
                        "Revenue Δ%": r["delta_rev_pct"],
                        "Revenue ($ idx)": 1 + r["delta_rev_pct"] / 100.0,
                    }
                )
            table = pd.DataFrame(out)
            st.dataframe(
                table.style.format({"New Price": "${:.2f}", "Est Units": "{:.1f}", "Revenue Δ%": "{:+.1f}"}),
                use_container_width=True,
            )
            fig = px.bar(
                table,
                x="Scenario",
                y="Revenue ($ idx)",
                text=table["Revenue Δ%"].map(lambda v: f"{v:+.1f}%"),
                labels={"Revenue ($ idx)": "Revenue index"},
            )
            st.plotly_chart(fig, use_container_width=True)
            describe_chart(
                "Scenario bars",
                "Revenue index compares each scenario to current average price/units. Positive indicates gain vs baseline.",
                "Use as a directional screen before operationalizing price changes.",
            )
            best_row = max(out, key=lambda r: r["Revenue ($ idx)"])
            st.success(
                f"Best of these: Scenario {best_row['Scenario']} "
                f"({best_row['Δ Price %']:+.0f}%), projected revenue {best_row['Revenue Δ%']:+.1f}%."
            )

            st.download_button(
                "Download price plan (CSV)",
                data=pd.DataFrame(
                    {
                        "store_id": [store_sel],
                        "item_id": [item_sel],
                        "scenario": [best_row["Scenario"]],
                        "recommended_price_change_pct": [best_row["Δ Price %"]],
                    }
                ).to_csv(index=False).encode(),
                file_name="price_plan.csv",
                mime="text/csv",
            )

# ===================== Tab 5: Compare Segments =====================
with tabs[5]:
    tab_intro(
        title="Compare Segments",
        what="Ranks stores, categories, or departments by total units or week-over-week growth for the selected period.",
        why="Quick way to see where performance is strongest or slipping across the footprint.",
        how="Pick a grouping and a metric, then scan the bar chart to spot leaders and laggards.",
    )
    dim = st.selectbox("Compare by", ["store_id", "cat_id", "dept_id"], index=0)
    metric = st.selectbox("Metric", ["Total units", "Week-over-week %"], index=0)
    df_range = panel[panel["date"].between(start_ts, end_ts)].copy()
    if df_range.empty:
        st.info("No data in the selected range.")
    else:
        if metric == "Total units":
            agg = df_range.groupby(dim, as_index=False)["sales"].sum().rename(columns={"sales": "total_units"})
            agg = agg.sort_values("total_units", ascending=False).head(15)
            fig = px.bar(agg, x=dim, y="total_units", labels={"total_units": "Units"})
            st.plotly_chart(fig, use_container_width=True)
            describe_chart("Total units by segment", "Identifies the biggest volume contributors in the selected window.")
        else:
            rows = []
            for gval, gdf in df_range.groupby(dim):
                rows.append({"segment": gval, "wow_pct": compute_wow(gdf, freq="W")})
            agg = pd.DataFrame(rows).sort_values("wow_pct", ascending=False).head(15)
            fig = px.bar(agg, x="segment", y="wow_pct", labels={"wow_pct": "WoW %"})
            st.plotly_chart(fig, use_container_width=True)
            describe_chart(
                "Week-over-week by segment",
                "Flags emerging winners/decliners based on week aggregation. Use to triage deeper dives.",
            )

# ===================== Tab 6: Top Performers =====================
with tabs[6]:
    tab_intro(
        title="Top Performers in this Store",
        what="Lists the top week-over-week risers and decliners inside the selected store.",
        why="Quickly surfaces items that need attention (lean in or mitigate).",
        how="Use together with Price Sensitivity to decide if price moves could help or hurt trajectory.",
    )
    srange = panel[(panel["store_id"] == store_sel) & (panel["date"].between(start_ts, end_ts))].copy()
    if srange.empty:
        st.info("No data in the selected range.")
    else:
        movers = []
        for it, g in srange.groupby("item_id"):
            movers.append({"item_id": it, "WoW %": compute_wow(g, "W"), "Total units": g["sales"].sum()})
        mv = pd.DataFrame(movers)
        top_winners = mv.sort_values("WoW %", ascending=False).head(10)
        top_decliners = mv.sort_values("WoW %", ascending=True).head(10)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Week-over-week increases**")
            st.dataframe(top_winners, use_container_width=True)
        with c2:
            st.markdown("**Week-over-week declines**")
            st.dataframe(top_decliners, use_container_width=True)
        describe_chart(
            "Winners & Decliners",
            "Ranks SKUs by recent momentum. Use this with margins and elasticity to prioritize actions.",
        )

# ===================== Tab 7: Summary Export =====================
with tabs[7]:
    tab_intro(
        title="Summary Export",
        what="One-page markdown and CSV export of the key numbers for the current slice.",
        why="Quick sharing with stakeholders; pairs well with a price plan from Scenario Compare.",
        how="Pick your store, product, dates, then download.",
    )
    if slice_df.empty:
        st.info("No data in the selected range.")
    else:
        k = kpis_for_slice(slice_df)
        elas = estimate_elasticity(slice_df)
        elasticity_text = f"{elas['elasticity']:.2f}" if elas else "N/A"
        direction = "rose" if k["wow_pct"] > 0 else "fell" if k["wow_pct"] < 0 else "held steady"
        if elas:
            interpretation_bit = (
                "Price sensitivity is meaningful; price moves have a noticeable effect."
                if abs(elas["elasticity"]) > 1
                else "Demand is relatively steady against price changes."
            )
        else:
            interpretation_bit = "Not enough history to estimate price sensitivity yet."

        summary_text = f"""
### Sales Snapshot — {store_sel} / {item_sel}

**Period:** {start_ts.date()} → {end_ts.date()}  
**Avg Units / Day:** {k['avg_units']:.1f}  
**Avg Price / Unit:** ${k['avg_price']:.2f}  
**Week-over-Week Change:** {k['wow_pct']:+.1f}%  
**Elasticity:** {elasticity_text}

**Interpretation:**  
Sales {direction} {abs(k['wow_pct']):.1f}% vs the prior week. {interpretation_bit}
"""
        st.markdown(summary_text)

        # explicit reminder that the summary is built from the multi-CSV, encoded, and imputed panel
        st.caption(
            "This summary is computed from the unified daily panel created by merging "
            "`sales_train_validation.csv`, `calendar.csv`, and `sell_prices.csv`, "
            "after the encoding and price-imputation steps documented on the 'IDA + Preprocessing' tab."
        )

        st.download_button(
            "Download Markdown Summary",
            data=summary_text.encode(),
            file_name=f"summary_{store_sel}_{item_sel}.md",
            mime="text/markdown",
        )

        row = {
            "Store": store_sel,
            "Item": item_sel,
            "Period Start": str(start_ts.date()),
            "Period End": str(end_ts.date()),
            "Avg Units/Day": k['avg_units'],
            "Avg Price/Unit": k['avg_price'],
            "Week-over-Week %": k['wow_pct'],
            "Elasticity": elas["elasticity"] if elas else None,
        }
        st.download_button(
            "Download Data (CSV)",
            data=pd.DataFrame([row]).to_csv(index=False).encode(),
            file_name=f"summary_{store_sel}_{item_sel}.csv",
            mime="text/csv",
        )
        st.caption("Attach a price plan from Scenario Compare when sharing recommendations.")