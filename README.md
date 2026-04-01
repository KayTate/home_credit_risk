# Home Credit Default Risk

Binary classification pipeline predicting loan default risk using the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) dataset.

---

## Project Structure

```text
home_credit_risk/
├── data/                          # CSV files (not tracked — see Setup)
├── lib/
│   └── data_aggregation.py        # Shared aggregation module
├── eda/
│   ├── eda_pipeline.py            # Phase 1 + Phase 3 EDA
│   ├── plots/                     # Generated PNG plots
│   └── eda_report.pdf             # Full EDA report (generated)
├── initial_training/
│   ├── model_training.py          # XGBoost training pipeline + MLflow tracking
│   ├── generate_report.py         # PDF report from training outputs
│   └── reports/
│       ├── xgb_model.json         # Saved model
│       ├── roc_curve.png
│       ├── confusion_matrix.png
│       ├── shap_summary.png
│       ├── shap_bar.png
│       └── pipeline_report.pdf    # Full training report (generated)
└── mlflow.db                      # MLflow experiment tracking database (generated)
```

---

## Setup

### 1. Download the data

Download all CSV files from [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data) and place them in a `data/` folder at the project root:

```text
data/
├── application_train.csv
├── bureau.csv
├── bureau_balance.csv
├── previous_application.csv
├── installments_payments.csv
├── POS_CASH_balance.csv
└── credit_card_balance.csv
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy reportlab scikit-learn xgboost shap mlflow
```

---

## Pipeline Overview

### Step 1 — EDA (`eda/eda_pipeline.py`)

Exploratory analysis in two phases:

- **Phase 1** — `application_train.csv`: null rates, class imbalance, numeric/categorical feature signal, correlation structure, engineered feature comparison
- **Aggregation** — joins all 6 supplementary tables to the base dataset via `lib/data_aggregation.py`
- **Phase 3** — joined dataset: feature signal ranking, missingness structure, cross-table delinquency correlations, final drop list

Outputs `eda/eda_report.pdf` covering all findings and recommendations.

```bash
python eda/eda_pipeline.py
```

### Step 2 — Model Training (`initial_training/model_training.py`)

Trains an XGBoost classifier on the fully joined dataset:

- Drops columns flagged in Sections 6 and 11 of the EDA report (>60% null, high multicollinearity, range violations)
- Stratified 80/20 train/test split
- Median imputation, LabelEncoder for standard categoricals, TargetEncoder for high-cardinality columns
- `scale_pos_weight` computed from training set to handle class imbalance (~92% repay, ~8% default)
- Early stopping on AUC (50 rounds patience)
- SHAP feature importance computed on a 2,000-row test sample

All runs are tracked in MLflow. Results are saved to `initial_training/reports/`.

```bash
python initial_training/model_training.py
```

**Current result: ROC-AUC 0.7793** — good result, in the expected range for this dataset with the full supplementary feature set.

### Step 3 — Training Report (`initial_training/generate_report.py`)

Compiles all training outputs (plots, metrics, classification report) into a single PDF at `initial_training/reports/pipeline_report.pdf`. Loads the saved model — does not retrain.

```bash
python initial_training/generate_report.py
```

### Step 4 — MLflow UI

View experiment runs, compare metrics, and inspect logged artifacts:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://127.0.0.1:5000
```

---

## Next Steps

### Second training run — SHAP-guided feature pruning

The SHAP analysis from the initial training identified **37 features with mean |SHAP| < 0.001** — these contribute negligible predictive signal and are candidates for removal. The recommended next step is:

1. Copy `initial_training/model_training.py` to a new script (e.g. `pruned_training/model_training.py`)
2. Add the 37 SHAP pruning candidates to the drop list
3. Retrain and compare ROC-AUC against the baseline run in the MLflow UI

If AUC holds steady or improves, the pruned model is strictly better — fewer features with equal or higher discrimination. If AUC drops more than ~0.002, selectively restore the most impactful pruned features.

The full candidate list is printed at the end of every training run under **"SHAP second pruning candidates"**.
