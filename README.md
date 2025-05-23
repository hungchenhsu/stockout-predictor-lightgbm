# 📦 Stock-Out Date Predictor for Retail Inventory (LightGBM)

A production-ready machine learning pipeline for forecasting the number of days until stock-out for retail inventory SKUs.  
Designed using structured daily features, tabular time-series techniques, and scalable LightGBM models.

> ⚠️ **Note:** This project was developed as a public-facing portfolio artifact.  
> The original dataset is private and under NDA, but the entire pipeline is reusable on any retail inventory dataset with daily BOH-style features.

---

## 🧠 Why this project?

Stock-outs are a persistent challenge in retail. Most forecasting models focus on *how much* to reorder. This project asks a different question:

> “Given today’s data, **when** is this SKU likely to go out of stock?”

This shift from quantity to timing allows operations teams to prioritize interventions, optimize replenishment schedules, and reduce lost sales.

---

## 📂 Table of Contents

- [🔍 Use Case](#-use-case)
- [📁 Repository Structure](#-repository-structure)
- [⚙️ How It Works](#️-how-it-works)
- [📊 Model Overview](#-model-overview)
- [🚀 Getting Started](#-getting-started)
- [📌 Notes and Assumptions](#-notes-and-assumptions)
- [📃 License](#-license)

---

## 🔍 Use Case

- **Domain:** Retail inventory forecasting  
- **Prediction target:** Days until stock-out  
- **Input:** Daily, per-SKU features (e.g., balance on hand, rolling statistics, calendar flags)  
- **Output:** Predicted days until next stock-out  
- **Applications:** Replenishment prioritization, alerting systems, shelf availability dashboards  
- **Designed for:** SKUs with short-term inventory visibility and relatively stable consumption patterns

---

## 📁 Repository Structure

```text
stockout-predictor/
├── data/
│   ├── raw/                   ← Placeholder for input CSVs
│   ├── interim/               ← Preprocessed parquet files
│   └── external/              ← Optional: label or transaction files
│
├── models/
│   └── prod_v1/
│       ├── lgbm_stockout_model.txt     ← Final trained model
│       ├── permanent_oos_list.json     ← List of always-OOS SKUs
│       ├── params.yaml                 ← Feature list + hyperparameters
│       └── model_card.md               ← Documentation of model assumptions
│
├── src/
│   ├── preprocessing/
│   │   └── stockout_preprocess.py      ← Feature engineering pipeline
│   ├── training/
│   │   └── train_lgbm_stockout.py      ← CV training + evaluation
│   ├── inference/
│   │   ├── predict_batch.py            ← Batch scoring script
│   │   └── predict_api.py              ← (Optional) FastAPI endpoint
│   └── requirements.txt
│
└── README.md
```

---

## ⚙️ How It Works

### Step 1: Data Preparation
- **Input:** Daily SKU-level inventory dataset (simulated or anonymized)
- **Feature engineering includes:**
  - Rolling 3-day and 7-day averages and standard deviations of `DailyBOH`
  - Day-of-week and U.S. holiday flags
  - Native categorical features such as `itemsku`, `storeid`, etc.

### Step 2: Labeling
- The model learns to predict `days_to_oos` — the number of days remaining until the SKU hits zero `DailyBOH` (out-of-stock).
- If a SKU is always out-of-stock during the training window, it is handled separately (excluded from training, returned as `0` in inference).

### Step 3: Modeling
- **Algorithm:** LightGBM Regressor
- **Cross-validation:** 5-fold `GroupKFold` grouped by `itemsku` (ensures SKU leakage is avoided)
- **Target:** `days_to_oos` as a regression task
- **Categorical features:** Handled natively by LightGBM (no manual encoding needed)
- **Feature importance:** Computed post-training to enhance model interpretability

### Step 4: Deployment
- **Trained model:** Exported as `.txt` file (via LightGBM's `booster.save_model()`)
- **Batch prediction:** Done via `predict_batch.py`, returning daily forecast CSVs
- **Optional REST API:** FastAPI endpoint (`predict_api.py`) enables SKU-level real-time scoring

---

## 📊 Model Overview

| Metric                | Value              |
|-----------------------|--------------------|
| CV MAE (5-fold)       | 6.76 ± 0.29 days   |
| Hold-out MAE          | 3.55 days          |
| Observation window    | 30 daily records per SKU |
| Training samples      | ~4,000 rows        |
| Inference latency     | ~50 ms for 50k rows |
| Model size            | ~400 KB            |

### Why LightGBM?

Deep learning models such as **LSTM** and **TimeGAN** were evaluated during experimentation. However:

- Most SKUs had only ~30 days of available data
- Sequence models required significantly longer time series to converge
- Deep models produced unstable or highly variable predictions
- TimeGAN failed to simulate realistic inventory dynamics under sparse conditions

**LightGBM** was chosen due to:

- Robust performance with short tabular sequences
- Efficient handling of thousands of unique SKUs via categorical splits
- Fast training (<1 min) and low inference overhead
- Easy interpretability via feature importance
- Simplified deployment (single `.txt` model file, no GPU or DL framework required)

---

## 🚀 Getting Started

This repository includes a fully functional training pipeline, from feature engineering to model training and evaluation, focused on stock-out prediction using tabular time-series data.  
**Note:** Inference scripts (e.g., for real-time or batch scoring) are not included in this version, as the focus is on modeling and experimentation.

### 🛠 Prerequisites

- Python 3.11.8
- Recommended: virtual environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r src/requirements.txt
```

---

### 📁 Project Files

| File                             | Description |
|----------------------------------|-------------|
| `stockout_preprocess.py`         | Feature engineering pipeline that processes daily SKU-level data into model-ready format, including rolling statistics, OOS labeling, and holiday flagging. |
| `train_lgbm_stockout.py`         | Trains the LightGBM model using 5-fold `GroupKFold` cross-validation and a 5-day hold-out set. Outputs CV metrics and saves the final model. |
| `lgbm_stockout_model.txt`        | Trained LightGBM model exported in native `.txt` format. Can be reloaded for reuse with `Booster().load_model()`. |
| `permanent_oos_list.json`        | Contains SKUs that were always out-of-stock during the training window. These are excluded from model training and treated as immediately OOS in evaluation. |
| `params.yaml`                    | Captures training hyperparameters, features used, evaluation results, and training metadata for full reproducibility. |
| `training_walkthrough.ipynb`     | Jupyter notebook that demonstrates the full modeling process, including data loading, feature construction, model training, CV evaluation, and interpretability analysis. |

---

### 🧪 How to Run Locally

Once dependencies are installed, you can execute the full pipeline in two main steps:

#### 1. Preprocess your dataset

```bash
python stockout_preprocess.py --input data/raw/your_dataset.csv
```

This will generate:
- train_proc.parquet and test_proc.parquet
- permanent_oos_list.json

#### 2. Train the model

```bash
python train_lgbm_stockout.py
```

This will generate:
- lgbm_stockout_model.txt
- params.yaml
- Console output with CV metrics and hold-out performance

### 💡 Optional

Use the included notebook training_walkthrough.ipynb to:
- Explore and visualize raw and engineered features
- Analyze feature importance
- Interpret hold-out performance and model behavior
- Document assumptions and modeling decisions for future deployment

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Citation

If you find this repository helpful in your research, teaching, or professional work,
please consider citing or linking back to the repository:

Hung-Chen Hsu. Phantom Inventory Classifier: Multi-Model Detection of Retail Stock Discrepancies. GitHub, 2025.
Repository: https://github.com/hungchenhsu/phantom-inventory-classifier

This helps acknowledge the original work and supports open sharing in the machine learning and retail analytics community 🙌

---

Created with 💻 and 🎯 by Hung-Chen Hsu
