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

from src.preprocessing import load_merge, sample_panel
from src.utils import (
    kpis_for_slice, anomaly_flags, seasonal_naive, moving_avg, safe_num,
    info_table, missingness_bar, missingness_heatmap, corr_heatmap
)

# ---------------------------- Page setup ----------------------------
st.set_page_config(
    page_title="Walmart Intelligence — Executive Sales, Pricing, Forecasting & Scenarios",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Walmart Intelligence — Executive Sales, Pricing, Forecasting & Scenarios")
st.caption("Storage-optimized: reads only the last ~120 days and a sampled subset of stores/items to stay under Streamlit free limits.")

# ---------------------------- Data load -----------------------------
with st.sidebar:
    st.header("Data & filters")
    st.write("Place these files under `data/` in your repo root:")
    st.code("data/calendar.csv.zip\ndata/sell_prices.csv.zip\ndata/sales_train_validation.csv.zip", language="bash")
    st.markdown("---")
    # memory guardrails – you can tweak but defaults are safe
    trailing_days = st.number_input("Days of history (last N)", 60, 365, 120, step=30)
    max_rows_target = st.number_input("Target rows after sampling (approx)", 10_000, 80_000, 30_000, step=5_000)

try:
    base = load_merge(trailing_days=int(trailing_days), max_rows_target=int(max_rows_target), use_cache=True)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.exception(e)
    st.stop()

with st.sidebar:
    n_stores = st.number_input("How many stores to include", 1, 10, 3, step=1)
    n_items  = st.number_input("How many products per store", 5, 200, 30, step=5)
    st.caption("Sampling is applied on top of the trimmed base to keep memory low.")
    st.markdown("---")
    st.markdown("Upload optional unit costs to enable margin/profit tiles.")
    cost_file = st.file_uploader("costs.csv (store_id,item_id,unit_cost)", type=["csv"])

panel = sample_panel(base, n_stores=int(n_stores), n_items=int(n_items))

costs_df = None
if cost_file is not None:
    try:
        tmp = pd.read_csv(cost_file)
        if {"store_id","item_id","unit_cost"}.issubset(tmp.columns):
            costs_df = tmp.copy()
        else:
            st.sidebar.warning("Expected columns: store_id, item_id, unit_cost")
    except Exception as e:
        st.sidebar.warning(f"Could not read costs.csv: {e}")

stores = sorted(panel["store_id"].dropna().unique().tolist())
store_sel = st.sidebar.selectbox("Store", stores, index=0 if stores else None)
items = sorted(panel.loc[panel["store_id"] == store_sel, "item_id"].dropna().unique().tolist()) if store_sel else []
item_sel = st.sidebar.selectbox("Product", items, index=0 if items else None)

if panel.empty or not stores or not items:
    st.info("No data loaded after trimming/sampling. Increase `Days of history` or `target rows`, or check your ZIP paths.")
    st.stop()

dmin, dmax = pd.to_datetime(panel["date"].min()), pd.to_datetime(panel["date"].max())
dr = st.sidebar.date_input("Date range", (dmin.date(), dmax.date()),
                           min_value=dmin.date(), max_value=dmax.date())
start_ts = pd.Timestamp(dr[0])
end_ts   = pd.Timestamp(dr[1])

slice_df = panel[
    (panel["store_id"] == store_sel) &
    (panel["item_id"] == item_sel) &
    (panel["date"].between(start_ts, end_ts))
].copy()

if costs_df is not None and not slice_df.empty:
    slice_df = slice_df.merge(costs_df, on=["store_id","item_id"], how="left")

# ---------------------------- Helpers -------------------------------
def purpose_block(title: str, what: str, why: str, how: str, read: str = ""):
    st.markdown(f"**Purpose – {title}**")
    st.markdown(f"- **What this shows:** {what}")
    st.markdown(f"- **Why it matters:** {why}")
    st.markdown(f"- **How to use it:** {how}")
    if read:
        st.caption(read)
    st.markdown("---")

def estimate_elasticity(df_slice: pd.DataFrame) -> Optional[dict]:
    g = df_slice.dropna(subset=["sell_price","sales"])
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
        "n": int(len(g))
    }

def simulate_revenue(df_slice: pd.DataFrame, elasticity: float, pct: float) -> dict:
    base_qty = safe_num(df_slice["sales"].mean())
    base_price = safe_num(df_slice["sell_price"].mean())
    new_price = base_price * (1 + pct)
    new_qty = base_qty * (1 + pct) ** elasticity
    base_rev = base_qty * base_price
    new_rev  = new_qty * new_price
    return {
        "new_price": float(new_price),
        "new_qty": float(new_qty),
        "delta_rev_pct": float((new_rev - base_rev) / (base_rev + 1e-9) * 100)
    }

def profit_kpis(df_slice: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if "unit_cost" not in df_slice.columns or df_slice["unit_cost"].isna().all():
        return None, None, None
    qty = safe_num(df_slice["sales"].mean())
    price = safe_num(df_slice["sell_price"].mean())
    cost  = safe_num(df_slice["unit_cost"].mean())
    margin = price - cost
    profit = margin * qty
    margin_pct = (margin / (price + 1e-9)) * 100 if price > 0 else None
    return float(profit), float(margin), float(margin_pct) if margin_pct is not None else None

def add_event_overlays(fig: go.Figure, df_: pd.DataFrame) -> go.Figure:
    if "event_name_1" not in df_.columns:
        return fig
    ev = df_.loc[df_["event_name_1"].notna(), ["date", "event_name_1"]].dropna()
    ymax = max(df_["sales"].max() if "sales" in df_.columns else 0, 1)
    for _, row in ev.iterrows():
        fig.add_vrect(x0=row["date"], x1=row["date"], fillcolor="orange", opacity=0.08, line_width=0)
        fig.add_annotation(x=row["date"], y=ymax, text=str(row["event_name_1"]),
                           showarrow=False, yshift=18, font=dict(size=10, color="gray"))
    return fig

def compute_wow(group_df: pd.DataFrame, freq="W") -> float:
    s = group_df.set_index("date")["sales"].resample(freq).sum()
    if len(s) < 2:
        return 0.0
    prev, curr = float(s.iloc[-2]), float(s.iloc[-1])
    return float(100 * (curr - prev) / (prev + 1e-9))

# ---------------------------- Tabs ----------------------------------
tabs = st.tabs([
    "IDA + Preprocessing",
    "Overview",
    "Forecast",
    "Price Sensitivity",
    "Scenario Compare",
    "Compare Segments",
    "Top Performers",
    "Summary Export"
])

# ---------- IDA + Preprocessing ----------
with tabs[0]:
    st.subheader("IDA + Preprocessing")
    st.markdown("""
This tab validates the **three ZIP files** and documents how the clean daily panel is built.  
It explains **what each diagnostic means** so you can trust the rest of the app.
""")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in trimmed base", f"{len(base):,}")
    c2.metric("Columns", f"{len(base.columns):,}")
    c3.metric("Date coverage", f"{pd.to_datetime(base['date']).min().date()} → {pd.to_datetime(base['date']).max().date()}")

    # 1) Structure
    st.markdown("#### 1) Structure table")
    st.dataframe(info_table(base), use_container_width=True)
    st.caption("**Meaning:** Types and null counts per column. Key fields (date, store_id, item_id, sales, sell_price) should be mostly non-null.")

    # 2) Numeric summary
    st.markdown("#### 2) Numeric summary")
    num_cols = base.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        st.dataframe(base[num_cols].describe().T.round(3), use_container_width=True)
        st.caption("**Meaning:** Center/spread for numeric fields. Watch for extreme max values (outliers) and big mean–median gaps (skew).")
    else:
        st.info("No numeric columns found.")

    # 3) Missingness
    st.markdown("#### 3) Missingness")
    cA, cB = st.columns(2)
    with cA:
        st.plotly_chart(missingness_bar(base), use_container_width=True)
    with cB:
        st.plotly_chart(missingness_heatmap(base), use_container_width=True)
    st.caption("**Meaning:** Which columns are sparse and whether missingness clumps. Price is forward/back filled within item to align to daily rows.")

    # 4) Correlation matrix
    st.markdown("#### 4) Correlation matrix (numeric features)")
    corr_fig = corr_heatmap(base)
    if corr_fig is not None:
        st.plotly_chart(corr_fig, use_container_width=True)
        st.caption("**Meaning:** Expected negative correlation between **price** and **sales** signals price sensitivity.")
    else:
        st.info("Not enough numeric columns to compute a correlation matrix.")

    # 5) Behavioral signals
    st.markdown("#### 5) Behavioral signals")
    ts_all = base.groupby("date", as_index=False)["sales"].sum()
    st.plotly_chart(
        px.line(ts_all, x="date", y="sales", title="All-sampled items — daily sales"),
        use_container_width=True
    )
    st.caption("**Meaning:** Weekly rhythm justifies the seasonality in Forecast; spikes/dips may align with events or promos.")

    # 6) Price fill check
    st.markdown("#### 6) Imputation diagnostics (sell_price)")
    price_missing_after = float(base["sell_price"].isna().mean()) if "sell_price" in base.columns else 1.0
    st.caption(f"Price still missing after within-item carry-fill: {price_missing_after:.1%}")
    samp = base[["item_id","sell_price"]].dropna()
    if not samp.empty:
        samp = samp.sample(min(20000, len(samp)), random_state=42)
        st.plotly_chart(px.violin(samp, y="sell_price", points=False, box=True, title="Filled price distribution"),
                        use_container_width=True)
        st.caption("**Meaning:** Ensures daily alignment for elasticity and scenarios.")

# ---------- Overview ----------
with tabs[1]:
    st.subheader(f"Overview — {store_sel} / {item_sel}")
    purpose_block(
        title="Overview",
        what="Key KPIs for the selected product and store, plus a daily sales line with unusual days and event overlays.",
        why="Gets everyone on the same page quickly—level, trend, volatility, and notable days.",
        how="Pick a store, product, and date range in the left panel. Hover lines to read event labels."
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
        fig = px.line(flagged, x="date", y="sales", labels={"date":"Date","sales":"Units"},
                      title="Daily Sales with Unusual-Day Flags")
        if flagged["anomaly"].sum() > 0:
            a = flagged[flagged["anomaly"] == 1]
            fig.add_scatter(x=a["date"], y=a["sales"], mode="markers", name="Unusual day")
        fig = add_event_overlays(fig, slice_df)
        st.plotly_chart(fig, use_container_width=True)

        elas = estimate_elasticity(slice_df)
        quick = []
        if k["wow_pct"] > 3: quick.append(f"Sales up {k['wow_pct']:.1f}% vs last week.")
        elif k["wow_pct"] < -3: quick.append(f"Sales down {abs(k['wow_pct']):.1f}% vs last week.")
        else: quick.append("Sales roughly flat WoW.")
        quick.append(f"Avg price ${k['avg_price']:.2f}.")
        if elas:
            quick.append("Price-sensitive product." if elas["elasticity"] < -1 else "Demand relatively steady vs price.")
        st.markdown("**Quick read:** " + " ".join(quick))

# ---------- Forecast ----------
with tabs[2]:
    st.subheader("Forecast — next 4 weeks")
    purpose_block(
        title="Forecast",
        what="Two simple 4-week baselines: one weekly-seasonal; one smoothed trend.",
        why="Fast planning signal when a heavy model isn’t necessary.",
        how="Select ≥ 30 days of history for steadier outlook.",
        read="These are baselines. Promotions/events can shift realized demand."
    )
    if slice_df.empty:
        st.info("No data in the selected range.")
    else:
        series = slice_df.set_index("date")["sales"]
        # Remove duplicate dates to avoid reindex error; then daily asfreq
        series = series.groupby(level=0).sum().asfreq("D").fillna(0.0)
        if len(series) < 30:
            st.warning("Select a longer range (30+ days) for a steadier forecast.")
        else:
            fut_idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=28, freq="D")
            fc1 = seasonal_naive(series, horizon=28, season=7)
            fc2 = moving_avg(series, horizon=28, window=7)
            fig = px.line(x=series.index, y=series.values, labels={"x":"Date","y":"Units"},
                          title="Actuals vs Baseline Forecasts")
            fig.add_scatter(x=fut_idx, y=fc1, name="Season-aware (weekly)")
            fig.add_scatter(x=fut_idx, y=fc2, name="Smoothed trend")
            st.plotly_chart(fig, use_container_width=True)

# ---------- Price Sensitivity ----------
with tabs[3]:
    st.subheader("Price sensitivity (elasticity)")
    purpose_block(
        title="Price Sensitivity",
        what="Elasticity estimates % unit change for 1% price change, plus a peer histogram.",
        why="Decide whether to prioritize volume (elastic) or margin (inelastic).",
        how="Needs ~30+ valid days with price variation."
    )
    if slice_df.empty:
        st.info("No data in the selected range.")
    else:
        elas = estimate_elasticity(slice_df)
        if not elas:
            st.info("Insufficient price variation/history (needs ~30+ days).")
        else:
            st.success(f"Elasticity: {elas['elasticity']:.2f} (n={elas['n']}, MAPE={elas['mape']:.3f})")
            vals: List[float] = []
            for (s, i), g in panel.groupby(["store_id","item_id"], sort=False):
                e = estimate_elasticity(g)
                if e:
                    vals.append(e["elasticity"])
            if vals:
                h = pd.Series(vals).clip(-5, 5)
                fig = px.histogram(h, nbins=40, labels={"value":"Elasticity","count":"Products"},
                                   title="Where This Product Sits vs Peers")
                fig.add_vline(x=elas["elasticity"], line_dash="dot", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("**Meaning:** Red line = this item. Left of −1 implies strong price sensitivity.")

# ---------- Scenario Compare ----------
with tabs[4]:
    st.subheader("Scenario compare")
    purpose_block(
        title="Scenario Compare",
        what="Three price scenarios with projected revenue index and a downloadable plan.",
        why="Test direction/magnitude before asking teams to execute.",
        how="Enter three Δ price % values (− for discount, + for increase)."
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
            base_qty = safe_num(slice_df["sales"].mean())
            base_price = safe_num(slice_df["sell_price"].mean())
            base_rev = base_qty * base_price
            for name, pct in scenarios.items():
                r = simulate_revenue(slice_df, elas_val, pct/100.0)
                out.append({"Scenario": name, "Δ Price %": pct, "New Price": r["new_price"],
                            "Est Units": r["new_qty"], "Revenue Δ%": r["delta_rev_pct"],
                            "Revenue ($ idx)": 1 + r["delta_rev_pct"]/100.0})
            table = pd.DataFrame(out)
            st.dataframe(
                table.style.format({"New Price":"${:.2f}", "Est Units":"{:.1f}", "Revenue Δ%":"{:+.1f}"}),
                use_container_width=True
            )
            fig = px.bar(table, x="Scenario", y="Revenue ($ idx)",
                         text=table["Revenue Δ%"].map(lambda v: f"{v:+.1f}%"),
                         labels={"Revenue ($ idx)":"Revenue index"},
                         title="Projected Revenue by Scenario")
            st.plotly_chart(fig, use_container_width=True)
            best_row = max(out, key=lambda r: r["Revenue ($ idx)"])
            st.success(f"Best of these: Scenario {best_row['Scenario']} "
                       f"({best_row['Δ Price %']:+.0f}%), projected revenue {best_row['Revenue Δ%']:+.1f}%.")

            st.download_button(
                "Download price plan (CSV)",
                data=pd.DataFrame({
                    "store_id":[store_sel],
                    "item_id":[item_sel],
                    "scenario":[best_row["Scenario"]],
                    "recommended_price_change_pct":[best_row["Δ Price %"]]
                }).to_csv(index=False).encode(),
                file_name="price_plan.csv", mime="text/csv"
            )

# ---------- Compare Segments ----------
with tabs[5]:
    st.subheader("Compare segments")
    purpose_block(
        title="Compare Segments",
        what="Rank stores, categories, or departments by total units or week-over-week growth.",
        why="Find where performance is strongest or slipping.",
        how="Pick a grouping and a metric."
    )
    dim = st.selectbox("Compare by", ["store_id", "cat_id", "dept_id"], index=0)
    metric = st.selectbox("Metric", ["Total units", "Week-over-week %"], index=0)
    df_range = panel[panel["date"].between(start_ts, end_ts)].copy()
    if df_range.empty:
        st.info("No data in the selected range.")
    else:
        if metric == "Total units":
            agg = df_range.groupby(dim, as_index=False)["sales"].sum().rename(columns={"sales":"total_units"})
            agg = agg.sort_values("total_units", ascending=False).head(15)
            fig = px.bar(agg, x=dim, y="total_units", labels={"total_units":"Units"})
            st.plotly_chart(fig, use_container_width=True)
            st.caption("**Meaning:** Top bars = highest demand in the selected period.")
        else:
            rows = []
            for gval, gdf in df_range.groupby(dim, sort=False):
                rows.append({"segment": gval, "wow_pct": compute_wow(gdf, freq="W")})
            agg = pd.DataFrame(rows).sort_values("wow_pct", ascending=False).head(15)
            fig = px.bar(agg, x="segment", y="wow_pct", labels={"wow_pct":"WoW %"})
            st.plotly_chart(fig, use_container_width=True)
            st.caption("**Meaning:** Positive values = faster recent momentum; negatives = softening.")

# ---------- Top Performers ----------
with tabs[6]:
    st.subheader("Top Performers in this Store")
    purpose_block(
        title="Top Performers",
        what="Fast risers and steep decliners inside one store.",
        why="Spot winners/laggards quickly.",
        how="Use WoW% and total units to triage attention."
    )
    srange = panel[(panel["store_id"] == store_sel) & (panel["date"].between(start_ts, end_ts))].copy()
    if srange.empty:
        st.info("No data in the selected range.")
    else:
        movers = []
        for it, g in srange.groupby("item_id", sort=False):
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

# ---------- Summary Export ----------
with tabs[7]:
    st.subheader("Summary Export")
    purpose_block(
        title="Summary Export",
        what="One-pager and CSV with the key numbers to share.",
        why="Create executive-friendly artifacts.",
        how="Pick your slice and download."
    )
    if slice_df.empty:
        st.info("No data in the selected range.")
    else:
        k = kpis_for_slice(slice_df)
        elas = None
        try:
            elas = (lambda r=estimate_elasticity(slice_df): r)( )
        except Exception:
            pass

        elasticity_text = f"{elas['elasticity']:.2f}" if elas else "N/A"
        direction = "rose" if k["wow_pct"] > 0 else "fell" if k["wow_pct"] < 0 else "held steady"
        interpretation_bit = (
            "Price sensitivity is meaningful; price moves have a noticeable effect."
            if (elas and abs(elas["elasticity"]) > 1) else
            "Demand is relatively steady against price changes." if elas else
            "Not enough history to estimate price sensitivity yet."
        )
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

        st.download_button(
            "Download Markdown Summary",
            data=summary_text.encode(),
            file_name=f"summary_{store_sel}_{item_sel}.md",
            mime="text/markdown"
        )

        row = {
            "Store": store_sel,
            "Item": item_sel,
            "Period Start": str(start_ts.date()),
            "Period End": str(end_ts.date()),
            "Avg Units/Day": k["avg_units"],
            "Avg Price/Unit": k["avg_price"],
            "Week-over-Week %": k["wow_pct"],
            "Elasticity": elas["elasticity"] if elas else None
        }
        st.download_button(
            "Download Data (CSV)",
            data=pd.DataFrame([row]).to_csv(index=False).encode(),
            file_name=f"summary_{store_sel}_{item_sel}.csv",
            mime="text/csv"
        )
        st.caption("Attach a price plan from Scenario Compare when sharing recommendations.")