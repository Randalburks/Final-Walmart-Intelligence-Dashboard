# Final-Project-Walmart-Intelligence-Dashboard - Executive Sales, Pricing, Forecasting & Scenario Dashboard

### Michigan State University | CMSE 830 Midterm Project  
**Author:** Randal Burks  
**Purpose:** This project delivers an interactive, executive-ready dashboard that summarizes key sales insights from Walmart’s M5 dataset. It connects descriptive, diagnostic, and prescriptive analytics—showing what happened, why it happened, and what actions to take.

---

## Dataset and Source

The raw datasets come from the **Walmart M5 Forecasting Competition** on Kaggle:  
https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Download these files and place them in your local `data/` folder:

- `calendar.csv`  
- `sell_prices.csv`  
- `sales_train_validation.csv`  

These files are not included in this repository due to size.  
Optional: you may also provide `costs.csv` with columns `store_id,item_id,unit_cost` to enable profit/margin tiles.

---

## Project Overview

This dashboard is designed for **executives and decision-makers** who need quick, actionable insight without reading raw tables or code. It unifies time-series, pricing, and short-term forecasting into one Streamlit app. The views are simple to navigate and tuned to common business questions: performance trends, price sensitivity, scenario impacts, and where to focus attention across stores and categories.

---

## Features

### 1) Overview
- KPIs for average units, average price, and week-over-week change.  
- Optional profit and margin tiles if `costs.csv` is supplied.  
- Daily sales timeline with anomaly markers and event overlays.

**Purpose:** A fast read on health, trend, and unusual days.

---

### 2) Forecast (Next 4 Weeks)
- Two robust baselines for reliability:
  - **Seasonal-naive:** repeats last week’s pattern.
  - **Moving average:** smooths recent trend.

**Purpose:** Directional outlook to support inventory, staffing, and marketing decisions.

---

### 3) Price Sensitivity
- Log–log regression to estimate **price elasticity** for the selected product/store.  
- Histogram to compare this product’s elasticity to peers in view.

**Purpose:** Quantifies how price changes are likely to affect units and revenue.

---

### 4) Scenario Compare
- Enter three price changes (e.g., −10%, 0%, +10%) and compare projected **revenue index** side-by-side.  
- Download a simple **price plan CSV** for hand-offs.

**Purpose:** Test “what if” pricing ideas before acting.

---

### 5) Compare Segments
- Rank **stores, categories, or departments** by total units or week-over-week change over the selected period.

**Purpose:** Find leaders, laggards, and where to focus attention.

---

### 6) Top Performers
- Within the selected store, list SKUs with the largest week-over-week increases and declines.

**Purpose:** Quickly spot products driving growth or needing action.

---

### 7) Summary Export
- One-page Markdown summary with period KPIs and interpretation.  
- CSV export of the key metrics for reporting or email.

**Purpose:** Produce a clear, shareable summary for meetings.

---

## Repository Structure

walmart-intelligence-dashboard/  
├── app.py  
├── requirements.txt  
├── .gitignore  
├── README.md  
├── .streamlit/  
│   └── secrets.toml        # optional; not required for this version  
├── src/  
│   ├── __init__.py  
│   ├── preprocessing.py    # data loading, merging, cleaning, caching  
│   └── utils.py            # helper KPIs, forecasting baselines, anomaly flags  
├── data/  
│   ├── calendar.csv  
│   ├── sell_prices.csv  
│   ├── sales_train_validation.csv  
│   ├── _merged_m5.parquet  # auto-generated cache (do not commit large files)  
│   └── costs.csv           # optional  
└── exports/  
    ├── summaries/          # optional: saved summaries  
    └── figures/            # optional: saved figures  

---

## File Descriptions

**`app.py`**  
The Streamlit app. It loads data, applies filters, computes KPIs, renders charts, runs elasticity estimation, compares scenarios, and produces exports.

**`src/preprocessing.py`**  
Builds the tidy daily panel the app uses:
- Reads the three M5 files.  
- Reshapes sales from wide (d_1, d_2, …) to long.  
- Joins to `calendar` for dates/events and to `sell_prices` for price by week/store/item.  
- Carries price forward/back within each item to align weekly prices to daily rows.  
- Derives a row-level SNAP flag.  
- Caches a merged Parquet for speed.

**`src/utils.py`**  
Reusable analytics helpers:
- `kpis_for_slice(df)` — average units, average price, WoW%.  
- `anomaly_flags(df)` — rolling z-score detection for unusual days.  
- `seasonal_naive()` and `moving_avg()` — stable forecasting baselines.  
- `safe_num()` — guards numeric ops against NaNs/zeros.

**`requirements.txt`**  
Python package pins to run the app consistently.

**`data/`**  
Local copy of input CSVs (not committed due to size) plus the generated cache.

**`exports/`**  
Optional folder for summaries and figures you choose to save.

---

## How to Run

```bash
# 1) Clone the repo
git clone https://github.com/your-username/walmart-intelligence-dashboard.git
cd walmart-intelligence-dashboard

# 2) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4) Add data files locally
mkdir -p data
# Place calendar.csv, sell_prices.csv, sales_train_validation.csv inside data/

# (optional) add costs.csv with columns: store_id,item_id,unit_cost

# 5) Launch the app
streamlit run app.py
