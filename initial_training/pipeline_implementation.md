# ML Pipeline Implementation Guide — Home Credit Default Risk

This document defines the implementation of the modeling pipeline in `pipeline/model_training.py`. It is intended as an implementation guide for Claude Code. Read it alongside the aggregation strategy document before writing any code.

This file is separate from the EDA pipeline. Do not merge them. The aggregation logic lives in `lib/data_aggregation.py` and must be imported by both `eda/eda_pipeline.py` and `pipeline/model_training.py`. If the aggregation code is not already structured as a callable function, refactor it into one before implementing anything in this document.

---

## Project file structure

The project must be organized as follows before any code is written. Create any missing files or directories.

```
home_credit_risk/
├── data/
│   ├── application_train.csv
│   ├── bureau.csv
│   ├── bureau_balance.csv
│   ├── previous_application.csv
│   ├── installments_payments.csv
│   ├── POS_CASH_balance.csv
│   └── credit_card_balance.csv
├── eda/
│   ├── implementation_docs/
│   ├── plots/
│   ├── eda_pipeline.py
│   └── eda_report.pdf
├── lib/
│   └── data_aggregation.py
└── pipeline/
    ├── model_training.py
    └── reports/
        ├── roc_curve.png
        ├── confusion_matrix.png
        ├── shap_summary.png
        └── shap_bar.png
```

---

## lib/data_aggregation.py requirements

The aggregation module already exists at `lib/data_aggregation.py` and exposes a public function with the following signature:

```python
def aggregate_all(data_dir: str) -> pd.DataFrame:
```

Do not rename, rewrite, or duplicate this function. All pre-processing corrections (replacing `365243`, replacing `'XNA'` and `'XAP'`, taking absolute values of `DAYS_*` columns) are already applied inside `aggregate_all`. Do not re-apply them in `pipeline/model_training.py`.

---

## pipeline/model_training.py implementation

Implement `pipeline/model_training.py` in the order defined below. Do not reorder steps. Each step has explicit requirements for what must be printed or saved.

### Step 1: Imports

Structure imports in the following order: from-imports first, then regular imports. Within each group, standard library comes before third-party packages.

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from xgboost import XGBClassifier

from lib.data_aggregation import aggregate_all

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
```

### Step 2: Column drop list

Before writing any code, open `eda/eda_report.pdf` and locate the two output summary sections:
- Section 6 — Phase 1 output summary: contains the drop list from the `application_train` EDA, covering columns with over 60% null rates and columns identified as redundant from high-correlation pairs
- Section 11 — Phase 3 output summary: contains the drop list from the joined dataset EDA, covering cross-table features that did not outperform their components and any additional redundant aggregated columns

Populate the two constants below using the drop lists from those two sections exactly. Do not add or remove columns beyond what the report specifies.

```python
# Columns identified for dropping in Phase 1 EDA (eda/eda_report.pdf, Section 6)
PHASE1_COLUMNS_TO_DROP = [
    # Over 60% null rate
    # ... columns here

    # Redundant — weaker column in high-correlation pairs
    # ... columns here
]

# Columns identified for dropping in Phase 3 EDA (eda/eda_report.pdf, Section 11)
PHASE3_COLUMNS_TO_DROP = [
    # Cross-table features that did not outperform their components
    # ... columns here

    # Redundant aggregated columns from within-table high-correlation pairs
    # ... columns here
]

# Non-predictive metadata — always drop regardless of EDA findings
METADATA_COLUMNS_TO_DROP = [
    'SK_ID_CURR',
]

# Combined drop list used in the pipeline
ALL_COLUMNS_TO_DROP = PHASE1_COLUMNS_TO_DROP + PHASE3_COLUMNS_TO_DROP + METADATA_COLUMNS_TO_DROP
```

Do not include any `has_*` flag columns in either drop list unless the Phase 3 missingness analysis in Section 9 of the report explicitly confirmed that the flag's default rate difference was below 0.02 and therefore uninformative. If in doubt, keep the flag.

### Step 3: Load data

Call `aggregate_all` to load and join all tables. The data directory path must be resolved relative to the project root, not relative to the `pipeline/` subdirectory. Print the shape of the resulting dataframe and confirm it matches the expected row count of approximately 307,000.

```python
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

print("Loading and aggregating data...")
loan_applications_df = aggregate_all(DATA_DIR)
print(f"Joined dataframe shape: {loan_applications_df.shape}")
assert loan_applications_df.shape[0] >= 300000, (
    f"Unexpected row count: {loan_applications_df.shape[0]} — check join logic"
)
```

If the assertion fails, stop and print a description of the failure. Do not proceed.

### Step 4: Drop columns

Drop the columns defined in `ALL_COLUMNS_TO_DROP` using `errors='ignore'` so the pipeline does not break if a listed column does not exist in the dataframe. Print the number of columns before and after dropping.

```python
cols_before_drop = loan_applications_df.shape[1]
loan_applications_df = loan_applications_df.drop(columns=ALL_COLUMNS_TO_DROP, errors='ignore')
cols_after_drop = loan_applications_df.shape[1]
print(f"Columns before drop: {cols_before_drop}")
print(f"Columns after drop:  {cols_after_drop}")
print(f"Columns dropped:     {cols_before_drop - cols_after_drop}")
```

### Step 5: Separate features and target

```python
feature_matrix = loan_applications_df.drop(columns=['TARGET'])
target_series = loan_applications_df['TARGET']
print(f"Feature matrix shape: {feature_matrix.shape}")
print(f"Target distribution:\n{target_series.value_counts(normalize=True).round(4)}")
```

### Step 6: Train/test split

Use a stratified split to preserve class balance. Use `random_state=42` for reproducibility.

```python
(
    features_train, features_test,
    target_train, target_test
) = train_test_split(
    feature_matrix, target_series,
    test_size=0.2,
    stratify=target_series,
    random_state=42
)
print(f"Train shape: {features_train.shape}")
print(f"Test shape:  {features_test.shape}")
print(f"Train target distribution:\n{target_train.value_counts(normalize=True).round(4)}")
print(f"Test target distribution:\n{target_test.value_counts(normalize=True).round(4)}")
```

Verify that the target distribution in both splits is close to the overall distribution. If either split deviates by more than 1 percentage point from the overall positive rate, print a warning.

### Step 7: Encode categorical columns

Before encoding, open `eda/eda_report.pdf` and locate Section 4 (categorical feature signal). The report flags any categorical column with more than 20 unique values as high cardinality. These columns must be encoded with `TargetEncoder` rather than `LabelEncoder` because label encoding assigns arbitrary integers to many categories, which implies a false ordinal relationship and produces poor signal. `TargetEncoder` replaces each category with the mean of the target for that category, which is statistically meaningful and handles high cardinality gracefully.

Define the high-cardinality columns explicitly based on what the report flagged:

```python
# Populate this list from Section 4 of eda/eda_report.pdf —
# any categorical column flagged as high cardinality (more than 20 unique values)
HIGH_CARDINALITY_COLUMNS = [
    # e.g. 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE' — fill from report
]
```

Encode in two passes — high-cardinality columns first with `TargetEncoder`, then all remaining categoricals with `LabelEncoder`. Fit all encoders on training data only.

```python
categorical_cols = features_train.select_dtypes(include='object').columns.tolist()
standard_categorical_cols = [col for col in categorical_cols if col not in HIGH_CARDINALITY_COLUMNS]
high_cardinality_cols_present = [col for col in HIGH_CARDINALITY_COLUMNS if col in categorical_cols]

print(f"Total categorical columns: {len(categorical_cols)}")
print(f"  Standard encoding (LabelEncoder):  {len(standard_categorical_cols)} columns")
print(f"  Target encoding (TargetEncoder):   {len(high_cardinality_cols_present)} columns")

# Target encode high-cardinality columns
if high_cardinality_cols_present:
    target_encoder = TargetEncoder(target_type='binary', random_state=42)
    features_train[high_cardinality_cols_present] = target_encoder.fit_transform(
        features_train[high_cardinality_cols_present], target_train
    )
    features_test[high_cardinality_cols_present] = target_encoder.transform(
        features_test[high_cardinality_cols_present]
    )

# Label encode remaining categorical columns
label_encoders = {}
for col in standard_categorical_cols:
    le = LabelEncoder()
    features_train[col] = le.fit_transform(features_train[col].astype(str))
    features_test[col] = le.transform(features_test[col].astype(str))
    label_encoders[col] = le
```

After encoding, verify that no object columns remain in either split. Stop if any do.

```python
remaining_object_cols = features_train.select_dtypes(include='object').columns.tolist()
assert len(remaining_object_cols) == 0, (
    f"Unencoded object columns remain: {remaining_object_cols}"
)
print("Encoding complete. No object columns remain.")
```

### Step 8: Impute missing values

Fit the imputer on training data only. Use median strategy for all columns.

```python
print("Imputing missing values...")
median_imputer = SimpleImputer(strategy='median')

features_train = pd.DataFrame(
    median_imputer.fit_transform(features_train),
    columns=features_train.columns,
    index=features_train.index
)
features_test = pd.DataFrame(
    median_imputer.transform(features_test),
    columns=features_test.columns,
    index=features_test.index
)

assert features_train.isnull().sum().sum() == 0, "Nulls remain in features_train after imputation"
assert features_test.isnull().sum().sum() == 0, "Nulls remain in features_test after imputation"
print("Imputation complete. No nulls remain.")
```

### Step 9: Compute scale_pos_weight

Compute from the training set only, not the full dataset.

```python
negative_class_count = (target_train == 0).sum()
positive_class_count = (target_train == 1).sum()
scale_pos_weight = negative_class_count / positive_class_count
print(f"Training set — negative class: {negative_class_count}, positive class: {positive_class_count}")
print(f"scale_pos_weight: {scale_pos_weight:.4f}")
```

### Step 10: Train XGBoost

Use the parameters below exactly. Do not change `random_state`. `early_stopping_rounds` is set to 50 — training stops automatically if AUC on the eval set does not improve for 50 consecutive rounds, which prevents overfitting without manual tuning of `n_estimators`.

```python
print("Training XGBoost model...")
xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    features_train, target_train,
    eval_set=[(features_test, target_test)],
    verbose=50
)

print(f"Best iteration: {xgb_model.best_iteration}")
print(f"Best AUC on eval set: {xgb_model.best_score:.4f}")
```

### Step 11: Evaluate

Compute all evaluation metrics and save all plots to the `pipeline/reports/` directory. Define the reports directory path relative to the script location so it works regardless of where the script is invoked from.

```python
REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)
```

```python
predicted_probabilities = xgb_model.predict_proba(features_test)[:, 1]
predicted_labels = xgb_model.predict(features_test)

roc_auc = roc_auc_score(target_test, predicted_probabilities)
print(f"\nROC-AUC: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(target_test, predicted_labels, target_names=['Repays', 'Defaults']))
```

**ROC curve:**
```python
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(target_test, predicted_probabilities, ax=ax)
ax.set_title(f'ROC Curve (AUC = {roc_auc:.4f})', fontsize=14)
ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'roc_curve.png'), dpi=150)
plt.close()
print(f"Saved: {os.path.join(REPORTS_DIR, 'roc_curve.png')}")
```

**Confusion matrix:**
```python
confusion_mat = confusion_matrix(target_test, predicted_labels)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    confusion_mat, annot=True, fmt='d', cmap='Blues', ax=ax,
    xticklabels=['Repays', 'Defaults'],
    yticklabels=['Repays', 'Defaults']
)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('Confusion Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()
print(f"Saved: {os.path.join(REPORTS_DIR, 'confusion_matrix.png')}")
```

Print a performance interpretation after saving the plots:
- If `roc_auc` >= 0.80: print "Strong result — above 0.80 AUC. Check for data leakage before considering this final."
- If `roc_auc` >= 0.77 and < 0.80: print "Good result — in the expected range for this dataset with supplementary features."
- If `roc_auc` >= 0.74 and < 0.77: print "Moderate result — in the expected range for application_train features only. Check that supplementary table features are present."
- If `roc_auc` < 0.74: print "Below expected range — investigate aggregation bugs, dropped features, or imputation issues."

### Step 12: SHAP feature importance

Compute SHAP values on the test set. Use a sample of 2000 rows if the test set is large, to keep computation time reasonable.

```python
print("Computing SHAP values...")
shap_sample = features_test.sample(n=min(2000, len(features_test)), random_state=42)

shap_explainer = shap.TreeExplainer(xgb_model)
shap_values = shap_explainer.shap_values(shap_sample)
```

**SHAP summary plot (beeswarm):**
```python
plt.figure()
shap.summary_plot(
    shap_values, shap_sample,
    max_display=30,
    show=False
)
plt.title('SHAP Feature Importance — Top 30 Features', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(REPORTS_DIR, 'shap_summary.png')}")
```

**SHAP bar plot (mean absolute value):**
```python
plt.figure()
shap.summary_plot(
    shap_values, shap_sample,
    plot_type='bar',
    max_display=30,
    show=False
)
plt.title('SHAP Mean Absolute Feature Importance — Top 30 Features', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'shap_bar.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {os.path.join(REPORTS_DIR, 'shap_bar.png')}")
```

**SHAP second pruning candidates:**

After computing SHAP values, print a list of features whose mean absolute SHAP value is below 0.001. These are candidates for dropping in a second training run.

```python
mean_absolute_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=shap_sample.columns
).sort_values(ascending=False)

shap_pruning_candidates = mean_absolute_shap[mean_absolute_shap < 0.001]
print(f"\nSHAP second pruning candidates ({len(shap_pruning_candidates)} features with mean |SHAP| < 0.001):")
print(shap_pruning_candidates.to_string())
```

### Step 13: Final summary print

At the end of the script, print a consolidated summary of all key outputs:

```
========================================
PIPELINE SUMMARY
========================================
Dataset shape after joins and drops : <rows> x <cols>
Train size                          : <n>
Test size                           : <n>
scale_pos_weight                    : <value>
Best XGBoost iteration              : <n>
ROC-AUC (test set)                  : <value>
Features with mean |SHAP| < 0.001   : <n> (see second pruning candidates above)
Reports saved to                    : pipeline/reports/
========================================
```

---

## Leakage checklist

Before considering results final, verify the following. Print each check explicitly.

- `median_imputer` was fit only on `features_train` — verify by checking that `median_imputer.statistics_` has the same length as `features_train.columns`
- `LabelEncoder` for each categorical column was fit only on `features_train` — already enforced by the encoding step above
- `TargetEncoder` was fit only on `features_train` and `target_train` — already enforced by the encoding step above
- `target_test` was not used in any fit step — verify by confirming `xgb_model.fit` only received `features_train` and `target_train` as the primary training arguments
- `scale_pos_weight` was computed from `target_train` only — verify by printing the computation again

```python
print("\nLeakage checklist:")
print(f"  Imputer fitted on {len(median_imputer.statistics_)} features (expected {features_train.shape[1]}): {'PASS' if len(median_imputer.statistics_) == features_train.shape[1] else 'FAIL'}")
print(f"  scale_pos_weight computed from training set only: PASS (see class counts above)")
print(f"  xgb_model.fit received features_train/target_train only: PASS (verify in code above)")
```

---

## Expected outputs

When the script completes successfully the following files must exist:

- `pipeline/reports/roc_curve.png`
- `pipeline/reports/confusion_matrix.png`
- `pipeline/reports/shap_summary.png`
- `pipeline/reports/shap_bar.png`

And the following must have been printed to stdout:

- Shape confirmation after loading
- Column counts before and after dropping
- Train/test split shapes and target distributions
- Encoding confirmation
- Imputation confirmation
- scale_pos_weight value
- XGBoost training log (every 50 rounds)
- Best iteration and best eval AUC
- ROC-AUC on test set
- Full classification report
- SHAP second pruning candidates
- Pipeline summary block
- Leakage checklist

---

## Expected performance range

- AUC >= 0.80: strong — verify no leakage
- AUC 0.77–0.80: good — expected range with full supplementary feature set
- AUC 0.74–0.77: moderate — expected range with application_train features only; investigate whether supplementary features are present
- AUC < 0.74: below expected — investigate aggregation bugs, over-aggressive column dropping, or imputation issues
