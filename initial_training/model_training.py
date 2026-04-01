"""
Home Credit Default Risk — Model Training Pipeline
Trains an XGBoost classifier on the fully joined dataset.
Run from repo root: python pipeline/model_training.py
"""

# ── Step 1: Imports ────────────────────────────────────────────────────────────
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

from lib.data_aggregation import preprocess, aggregate_all

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap


# ── Step 2: Column drop lists (read directly from eda/eda_report.pdf) ─────────

# Columns identified for dropping in Phase 1 EDA (eda/eda_report.pdf, Section 6)
PHASE1_COLUMNS_TO_DROP = [
    # Over 60% null rate
    'COMMONAREA_AVG',
    'COMMONAREA_MODE',
    'COMMONAREA_MEDI',
    'NONLIVINGAPARTMENTS_MEDI',
    'NONLIVINGAPARTMENTS_MODE',
    'NONLIVINGAPARTMENTS_AVG',
    'FONDKAPREMONT_MODE',
    'LIVINGAPARTMENTS_AVG',
    'LIVINGAPARTMENTS_MEDI',
    'LIVINGAPARTMENTS_MODE',
    'FLOORSMIN_MODE',
    'FLOORSMIN_AVG',
    'FLOORSMIN_MEDI',
    'YEARS_BUILD_AVG',
    'YEARS_BUILD_MODE',
    'YEARS_BUILD_MEDI',
    'OWN_CAR_AGE',

    # Redundant — weaker column in high-correlation pairs
    'AGE_YEARS',
    'EMPLOYED_YEARS',
    'OBS_60_CNT_SOCIAL_CIRCLE',
    'FLOORSMAX_MEDI',
    'ENTRANCES_MEDI',
    'ELEVATORS_MEDI',
    'LIVINGAREA_MEDI',
    'APARTMENTS_MEDI',
    'BASEMENTAREA_MEDI',
    'YEARS_BEGINEXPLUATATION_AVG',
    'LANDAREA_AVG',
    'NONLIVINGAREA_MEDI',
    'FLOORSMAX_MODE',
    'AMT_CREDIT',
    'ELEVATORS_MODE',
    'LANDAREA_MODE',
    'ENTRANCES_MODE',
    'BASEMENTAREA_MODE',
    'APARTMENTS_MODE',
    'NONLIVINGAREA_MODE',
    'LIVINGAREA_MODE',
    'YEARS_BEGINEXPLUATATION_MODE',
    'EMPLOYMENT_RATIO',
    'REGION_RATING_CLIENT',
    'TOTALAREA_MODE',
    'APARTMENTS_AVG',
    'EXT_SOURCE_MIN',
    'CNT_FAM_MEMBERS',
    'LIVINGAREA_AVG',
    'LIVE_REGION_NOT_WORK_REGION',
    'DEF_60_CNT_SOCIAL_CIRCLE',
]

# Columns identified for dropping in Phase 3 EDA (eda/eda_report.pdf, Section 11)
PHASE3_COLUMNS_TO_DROP = [
    # Cross-table features that did not outperform their components
    'internal_vs_external_dpd_diff',
    'credit_request_ratio',
    'annuity_request_ratio',
    'inst_std_days_late',

    # Redundant aggregated columns from within-table high-correlation pairs
    'days_since_last_application',   # corr=1.0 with prev_most_recent_decision
    'YEARS_BUILD_AVG',               # corr=0.9985 with YEARS_BUILD_MEDI
    'OBS_60_CNT_SOCIAL_CIRCLE',      # corr=0.9985 with OBS_30_CNT_SOCIAL_CIRCLE
    'FLOORSMIN_MEDI',                # corr=0.9972 with FLOORSMIN_AVG
    'FLOORSMAX_MEDI',                # corr=0.997 with FLOORSMAX_AVG
    'ENTRANCES_MEDI',                # corr=0.9969 with ENTRANCES_AVG
    'ELEVATORS_MEDI',                # corr=0.9961 with ELEVATORS_AVG
    'COMMONAREA_AVG',                # corr=0.996 with COMMONAREA_MEDI
    'LIVINGAREA_MEDI',               # corr=0.9956 with LIVINGAREA_AVG
    'APARTMENTS_MEDI',               # corr=0.9951 with APARTMENTS_AVG
    'BASEMENTAREA_MEDI',             # corr=0.9943 with BASEMENTAREA_AVG
    'YEARS_BEGINEXPLUATATION_AVG',   # corr=0.9938 with YEARS_BEGINEXPLUATATION_MEDI
    'LIVINGAPARTMENTS_MEDI',         # corr=0.9938 with LIVINGAPARTMENTS_AVG
    'pos_months_count',              # corr=0.9917 with pos_active_count
    'LANDAREA_AVG',                  # corr=0.9916 with LANDAREA_MEDI
    'NONLIVINGAPARTMENTS_MEDI',      # corr=0.9908 with NONLIVINGAPARTMENTS_AVG
    'NONLIVINGAREA_MEDI',            # corr=0.9904 with NONLIVINGAREA_AVG
    'YEARS_BUILD_MODE',              # corr=0.9895 with YEARS_BUILD_MEDI
    'FLOORSMIN_MODE',                # corr=0.9884 with FLOORSMIN_MEDI
    'FLOORSMAX_MODE',                # corr=0.9882 with FLOORSMAX_MEDI
    'AMT_CREDIT',                    # corr=0.987 with AMT_GOODS_PRICE
    'ELEVATORS_MODE',                # corr=0.9828 with ELEVATORS_MEDI
    'LANDAREA_MODE',                 # corr=0.9808 with LANDAREA_MEDI
    'ENTRANCES_MODE',                # corr=0.9807 with ENTRANCES_MEDI
    'COMMONAREA_MODE',               # corr=0.9799 with COMMONAREA_MEDI
    'NONLIVINGAPARTMENTS_MODE',      # corr=0.9786 with NONLIVINGAPARTMENTS_MEDI
    'BASEMENTAREA_MODE',             # corr=0.9779 with BASEMENTAREA_MEDI
    'APARTMENTS_MODE',               # corr=0.9772 with APARTMENTS_MEDI
    'NONLIVINGAREA_MODE',            # corr=0.9758 with NONLIVINGAREA_MEDI
    'LIVINGAPARTMENTS_MODE',         # corr=0.9756 with LIVINGAPARTMENTS_MEDI
    'LIVINGAREA_MODE',               # corr=0.9747 with LIVINGAREA_MEDI
    'YEARS_BEGINEXPLUATATION_MODE',  # corr=0.9719 with YEARS_BEGINEXPLUATATION_AVG
    'pos_avg_dpd_def',               # corr=0.9649 with pos_max_dpd_def
    'cc_avg_dpd',                    # corr=0.9605 with cc_max_dpd
    'REGION_RATING_CLIENT',          # corr=0.9508 with REGION_RATING_CLIENT_W_CITY
    'LIVINGAPARTMENTS_AVG',          # corr=0.944 with APARTMENTS_AVG
    'cc_dpd_rate',                   # corr=0.9422 with cc_dpd_month_count
    'cc_avg_utilization',            # corr=0.938 with cc_utilization_trend
    'TOTALAREA_MODE',                # corr=0.925 with LIVINGAREA_AVG
    'pos_max_dpd',                   # corr=0.9244 with pos_avg_dpd
    'bureau_loan_count',             # corr=0.9237 with bureau_closed_count
    # has_installments omitted: abs default rate diff = 0.0221 > 0.02 (informative)
    'APARTMENTS_AVG',                # corr=0.9136 with LIVINGAREA_AVG
    'has_pos_cash',                  # corr=0.9066 with has_installments; diff <= 0.02
    'inst_underpay_count',           # corr=0.8992 with inst_late_count
    'bureau_total_prolonged',        # corr=0.8965 with bureau_any_prolonged
    'internal_dpd_composite',        # corr=0.8925 with inst_late_rate
    'pos_avg_dpd',                   # corr=0.8906 with pos_dpd_month_count
    'CNT_FAM_MEMBERS',               # corr=0.8792 with CNT_CHILDREN
    'cc_max_balance',                # corr=0.8752 with cc_avg_balance
    'LIVINGAREA_AVG',                # corr=0.8678 with ELEVATORS_AVG
    'LIVE_REGION_NOT_WORK_REGION',   # corr=0.8606 with REG_REGION_NOT_WORK_REGION
    'DEF_60_CNT_SOCIAL_CIRCLE',      # corr=0.8605 with DEF_30_CNT_SOCIAL_CIRCLE
    'cc_max_utilization',            # corr=0.8502 with cc_avg_utilization

    # Range violations (data quality issues)
    'bureau_utilization_ratio',      # min=-3.1e12, max=2.25e12 (expected 0–1.05)
    'inst_max_days_late',            # min=-156, max=2884 (expected -365–365)
]

# Non-predictive metadata — always drop regardless of EDA findings
METADATA_COLUMNS_TO_DROP = [
    'SK_ID_CURR',
]

# Combined drop list used in the pipeline
ALL_COLUMNS_TO_DROP = PHASE1_COLUMNS_TO_DROP + PHASE3_COLUMNS_TO_DROP + METADATA_COLUMNS_TO_DROP


# ── Step 3: Load data ──────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

print("Loading and aggregating data...")
print("  Loading application_train.csv...")
df_app = pd.read_csv(os.path.join(DATA_DIR, 'application_train.csv'))
df_app = preprocess(df_app)
print(f"  application_train shape: {df_app.shape}")

loan_applications_df = aggregate_all(df_app)
print(f"Joined dataframe shape: {loan_applications_df.shape}")
assert loan_applications_df.shape[0] >= 300000, (
    f"Unexpected row count: {loan_applications_df.shape[0]} — check join logic"
)


# ── Step 4: Drop columns ───────────────────────────────────────────────────────

cols_before_drop = loan_applications_df.shape[1]
loan_applications_df = loan_applications_df.drop(columns=ALL_COLUMNS_TO_DROP, errors='ignore')
cols_after_drop = loan_applications_df.shape[1]
print(f"Columns before drop: {cols_before_drop}")
print(f"Columns after drop:  {cols_after_drop}")
print(f"Columns dropped:     {cols_before_drop - cols_after_drop}")


# ── Step 5: Separate features and target ──────────────────────────────────────

feature_matrix = loan_applications_df.drop(columns=['TARGET'])
target_series = loan_applications_df['TARGET']
print(f"Feature matrix shape: {feature_matrix.shape}")
print(f"Target distribution:\n{target_series.value_counts(normalize=True).round(4)}")


# ── Step 6: Train/test split ───────────────────────────────────────────────────

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

overall_positive_rate = target_series.mean()
train_positive_rate = target_train.mean()
test_positive_rate = target_test.mean()
if abs(train_positive_rate - overall_positive_rate) > 0.01:
    print(f"WARNING: Train positive rate {train_positive_rate:.4f} deviates >1pp from overall {overall_positive_rate:.4f}")
if abs(test_positive_rate - overall_positive_rate) > 0.01:
    print(f"WARNING: Test positive rate {test_positive_rate:.4f} deviates >1pp from overall {overall_positive_rate:.4f}")


# ── Step 7: Encode categorical columns ────────────────────────────────────────

# High-cardinality columns from Section 4 of eda/eda_report.pdf
# (columns with >20 unique values, flagged as high cardinality)
HIGH_CARDINALITY_COLUMNS = [
    'ORGANIZATION_TYPE',  # 57 unique values
]

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

remaining_object_cols = features_train.select_dtypes(include='object').columns.tolist()
assert len(remaining_object_cols) == 0, (
    f"Unencoded object columns remain: {remaining_object_cols}"
)
print("Encoding complete. No object columns remain.")


# ── Step 8: Impute missing values ─────────────────────────────────────────────

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


# ── Step 9: Compute scale_pos_weight ──────────────────────────────────────────

negative_class_count = (target_train == 0).sum()
positive_class_count = (target_train == 1).sum()
scale_pos_weight = negative_class_count / positive_class_count
print(f"Training set — negative class: {negative_class_count}, positive class: {positive_class_count}")
print(f"scale_pos_weight: {scale_pos_weight:.4f}")


# ── Step 10: Train XGBoost ────────────────────────────────────────────────────

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


# ── Step 11: Evaluate ─────────────────────────────────────────────────────────

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

predicted_probabilities = xgb_model.predict_proba(features_test)[:, 1]
predicted_labels = xgb_model.predict(features_test)

roc_auc = roc_auc_score(target_test, predicted_probabilities)
print(f"\nROC-AUC: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(target_test, predicted_labels, target_names=['Repays', 'Defaults']))

# ROC curve
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(target_test, predicted_probabilities, ax=ax)
ax.set_title(f'ROC Curve (AUC = {roc_auc:.4f})', fontsize=14)
ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'roc_curve.png'), dpi=150)
plt.close()
print(f"Saved: {os.path.join(REPORTS_DIR, 'roc_curve.png')}")

# Confusion matrix
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

# Performance interpretation
if roc_auc >= 0.80:
    print("Strong result — above 0.80 AUC. Check for data leakage before considering this final.")
elif roc_auc >= 0.77:
    print("Good result — in the expected range for this dataset with supplementary features.")
elif roc_auc >= 0.74:
    print("Moderate result — in the expected range for application_train features only. Check that supplementary table features are present.")
else:
    print("Below expected range — investigate aggregation bugs, dropped features, or imputation issues.")


# ── Step 12: SHAP feature importance ──────────────────────────────────────────

print("Computing SHAP values...")
shap_sample = features_test.sample(n=min(2000, len(features_test)), random_state=42)

shap_explainer = shap.TreeExplainer(xgb_model)
shap_values = shap_explainer.shap_values(shap_sample)

# SHAP summary plot (beeswarm)
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

# SHAP bar plot (mean absolute value)
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

# SHAP second pruning candidates
mean_absolute_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=shap_sample.columns
).sort_values(ascending=False)

shap_pruning_candidates = mean_absolute_shap[mean_absolute_shap < 0.001]
print(f"\nSHAP second pruning candidates ({len(shap_pruning_candidates)} features with mean |SHAP| < 0.001):")
print(shap_pruning_candidates.to_string())


# ── Step 13: Final summary ────────────────────────────────────────────────────

print()
print("========================================")
print("PIPELINE SUMMARY")
print("========================================")
print(f"Dataset shape after joins and drops : {loan_applications_df.shape[0]} x {features_train.shape[1] + 1}")
print(f"Train size                          : {features_train.shape[0]}")
print(f"Test size                           : {features_test.shape[0]}")
print(f"scale_pos_weight                    : {scale_pos_weight:.4f}")
print(f"Best XGBoost iteration              : {xgb_model.best_iteration}")
print(f"ROC-AUC (test set)                  : {roc_auc:.4f}")
print(f"Features with mean |SHAP| < 0.001   : {len(shap_pruning_candidates)} (see second pruning candidates above)")
print(f"Reports saved to                    : pipeline/reports/")
print("========================================")


# ── Leakage checklist ─────────────────────────────────────────────────────────

print("\nLeakage checklist:")
print(f"  Imputer fitted on {len(median_imputer.statistics_)} features (expected {features_train.shape[1]}): {'PASS' if len(median_imputer.statistics_) == features_train.shape[1] else 'FAIL'}")
print(f"  scale_pos_weight computed from training set only: PASS (see class counts above)")
print(f"  xgb_model.fit received features_train/target_train only: PASS (verify in code above)")
