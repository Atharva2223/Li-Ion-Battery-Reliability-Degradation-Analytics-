# Li-Ion Battery Reliability & Degradation Analytics

> End-to-end pipeline for extracting, modeling, and forecasting degradation on the NASA PCoE Li-ion battery cycling dataset — from raw `.mat` files to an interactive Streamlit dashboard.

Built with **Python**, **SciPy**, **lifelines**, **scikit-learn**, **XGBoost**, **Pandas**, **Seaborn**, **Matplotlib**, and **Streamlit**.

---

## Architecture

![Project architecture](docs/architecture.png)

The pipeline runs in five stages — raw `.mat` files flow through ingestion and feature engineering, then split into two parallel modeling tracks (Weibull reliability and RUL forecasting) before being surfaced in the Streamlit dashboard.

---

## Overview

Li-ion batteries degrade with every charge/discharge cycle. Predicting when a cell will fail — and how many cycles remain — is critical for electric vehicles, consumer electronics, and aerospace applications.

This project builds a complete prognostics pipeline that:

- Ingests raw NASA PCoE battery cycling data across 100+ cells
- Engineers per-cycle features (capacity fade, voltage-curve shape, impedance growth, temperature)
- Fits Weibull reliability models to quantify failure-rate distributions
- Forecasts Remaining Useful Life (RUL) using ARIMA and gradient-boosted models
- Surfaces insights through an interactive multi-page Streamlit dashboard

---

## Dataset

**Source:** [NASA Prognostics Center of Excellence — Battery Data Set](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

**Contents:** 18650 Li-ion cells cycled at 24 °C under constant-current charge (1.5 A to 4.2 V) and discharge (2 A to 2.2–2.7 V cutoffs). Each `.mat` file contains nested cycle data with voltage, current, temperature, capacity, and impedance measurements.

**End-of-life definition:** Capacity drops to ≤ 70 % of rated (~1.4 Ah from 2.0 Ah rated) — the standard IEEE degradation threshold.

Files are not committed to this repo. Download and place them in `data/raw/`:

```
data/raw/
├── B0005.mat
├── B0006.mat
└── ...
```

---

## Project Structure

```
li-ion-battery-analytics/
├── data/
│   ├── raw/                          NASA .mat files (not tracked)
│   ├── interim/                      Parsed per-cell parquets
│   └── processed/                    Feature-engineered dataset
├── docs/
│   └── architecture.png              Pipeline diagram
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_weibull_reliability.ipynb
│   └── 04_rul_forecasting.ipynb
├── src/
│   ├── ingestion/loader.py           Parse .mat → structured dicts
│   ├── features/engineer.py          Build per-cycle feature table
│   ├── models/
│   │   ├── weibull.py                Weibull fitting & B-life
│   │   ├── arima_rul.py              ARIMA RUL forecaster
│   │   └── gbm_rul.py                Gradient-boosted RUL regressor
│   └── viz/plots.py                  Reusable plot helpers
├── app/streamlit_app.py              Multi-page dashboard
├── tests/
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone & install
git clone https://github.com/<your-username>/li-ion-battery-analytics.git
cd li-ion-battery-analytics
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download the dataset → place .mat files in data/raw/

# 3. Run the pipeline
python -m src.ingestion.loader        # parse .mat → interim
python -m src.features.engineer       # build feature table
python -m src.models.weibull          # fit Weibull
python -m src.models.gbm_rul          # train RUL model

# 4. Launch the dashboard
streamlit run app/streamlit_app.py
```

---

## Pipeline Stages

### Stage 1 — Ingestion

Parses nested MATLAB structs with `scipy.io.loadmat` (or `h5py` for v7.3 files), separates cycles by type (charge / discharge / impedance), and writes one clean parquet per cell to `data/interim/`.

### Stage 2 — Feature Engineering

Builds one row per discharge cycle per cell. Engineered features:

| Group | Features |
|---|---|
| **Capacity** | `capacity_ah`, `capacity_retention`, rolling deltas, knee-point cycle |
| **Voltage** | discharge-curve slope, time-to-3.0V, voltage flatness |
| **Impedance** | electrolyte resistance Re, charge-transfer resistance Rct |
| **Thermal** | peak temperature, temperature rise rate |
| **Target** | `rul` — cycles remaining until end-of-life |

### Stage 3 — Reliability Modeling

- Failure defined as the first cycle where `capacity_retention < 0.70`
- Cells that didn't reach failure are treated as **right-censored**
- Fits `lifelines.WeibullFitter` (MLE with censoring) → shape β, scale η, 95% CIs
- Cross-validates with `scipy.stats.weibull_min`
- Compares against non-parametric Kaplan-Meier as a sanity check
- Reports **B10 life** — the cycle at which 10% of cells fail

### Stage 4 — RUL Forecasting

**ARIMA baseline** — univariate ARIMA on per-cell capacity-vs-cycle time series, order selected via `pmdarima.auto_arima`. Extrapolates to the failure threshold to compute RUL.

**Gradient-boosted regressor** — XGBoost trained on the full multivariate feature set. Critical: splits are performed **by cell** (`GroupKFold`) to prevent leakage from same-cell future cycles. A quantile GBM (P10/P90) produces prediction intervals.

### Stage 5 — Streamlit Dashboard

Three pages:

| Page | Contents |
|---|---|
| **Fleet overview** | Capacity-fade curves, failure histogram, Weibull PDF overlay, β / η / B10 summary |
| **Cell drill-down** | Per-cell capacity / voltage / impedance plots, ARIMA vs. GBM RUL with intervals |
| **RUL predictor** | Upload CSV of cycle features → predicted RUL + P10/P90 interval |

---

## Methods Notes

**Why Weibull?** Battery degradation is a wear-out failure mode — failure probability *increases* with cycles. The Weibull distribution is the canonical model for wear-out. β > 1 confirms wear-out; β > 3 indicates failures cluster tightly around η.

**Why gradient boosting over ARIMA?** ARIMA captures temporal autocorrelation in capacity alone; gradient boosting learns non-linear interactions across all features (impedance, voltage shape, thermal). It consistently outperforms univariate baselines on multi-feature prognostic benchmarks.

**Avoiding leakage.** RUL is highly auto-correlated across cycles of the same cell. Random row splits leak future cycles into training and inflate metrics. Cell-level group splitting (leave-one-cell-out) ensures the model is evaluated only on cells it has never seen.

---

## Results

Numbers populate after a full model run.

| Model | RMSE (cycles) | MAE (cycles) |
|---|---|---|
| ARIMA (per-cell) | — | — |
| GBM (all features) | — | — |
| GBM (capacity only) | — | — |

| Weibull parameter | Estimate | 95% CI |
|---|---|---|
| Shape β | — | — |
| Scale η (cycles) | — | — |
| B10 life (cycles) | — | — |

---

## Limitations

- **Small cell count.** The NASA PCoE dataset contains ~30 cells, limiting the statistical power of the Weibull fit and the diversity of the GBM training set.
- **Controlled lab conditions.** Cells were cycled at constant temperature and current. Real-world usage involves variable loads, partial cycles, and temperature swings — degradation patterns may differ.
- **Single chemistry.** All cells are 18650 NMC/LCO. Results may not generalize to LFP, NCA, or solid-state chemistries.

---

## Tech Stack

| Layer | Libraries |
|---|---|
| Data processing | `pandas`, `numpy`, `scipy`, `h5py` |
| Reliability modeling | `lifelines`, `scipy.stats` |
| ML / forecasting | `scikit-learn`, `xgboost`, `lightgbm`, `statsmodels`, `pmdarima` |
| Visualization | `matplotlib`, `seaborn` |
| Dashboard | `streamlit` |
| Testing | `pytest` |

---

## License

MIT
