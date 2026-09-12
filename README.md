# ⚡ Energy-Cast

### AI-Powered Renewable Energy Forecasting & Grid Intelligence

Energy-Cast is an offline renewable-energy forecasting prototype developed for **HackOut-2026**. It uses historical solar-generation and weather data with an **XGBoost regression model** to forecast renewable power generation for the next **24, 48, or 72 hours**.

The system also includes an interactive **What-If engine** that allows users to modify weather conditions and immediately observe their impact on predicted generation and grid stability.

---

## 🚀 Key Features

- **24/48/72-hour solar generation forecasting**
- **XGBoost-based machine learning model**
- Historical data cleaning and feature engineering
- Chronological train/test evaluation
- Model metrics: **R², MAE, RMSE**
- Prediction confidence bounds
- Interactive **What-If weather scenarios**
- Baseline vs What-If forecast comparison
- Grid stability classification:
  - 🟢 GREEN — Stable
  - 🟡 AMBER — Generation shortfall
  - 🔴 RED — Critical condition
- Plant capacity and curtailment monitoring
- Automated grid recommendations
- Interactive Plotly visualizations
- Fully offline demo

---

## 🧠 Machine Learning

The forecasting model uses **XGBoost Regressor**.

### Input Features

- Ambient Temperature
- Module Temperature
- Irradiation
- Hour
- Day of Week
- Month
- Cyclic time features

### Target

```text
AC Power (MW)