"""
Home Credit Default Risk — Pipeline Report Generator
Compiles existing pipeline outputs into a PDF report.
Run from repo root: python initial_training/generate_report.py
"""

# ── Step 1: Imports ────────────────────────────────────────────────────────────
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from xgboost import XGBClassifier

from lib.data_aggregation import preprocess, aggregate_all

import numpy as np
import pandas as pd
from PIL import Image as PILImage


# ── Step 2: Define paths ───────────────────────────────────────────────────────

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

ROC_CURVE_PATH = os.path.join(REPORTS_DIR, 'roc_curve.png')
CONFUSION_MATRIX_PATH = os.path.join(REPORTS_DIR, 'confusion_matrix.png')
SHAP_SUMMARY_PATH = os.path.join(REPORTS_DIR, 'shap_summary.png')
SHAP_BAR_PATH = os.path.join(REPORTS_DIR, 'shap_bar.png')
MODEL_PATH = os.path.join(REPORTS_DIR, 'xgb_model.json')
OUTPUT_PDF_PATH = os.path.join(REPORTS_DIR, 'pipeline_report.pdf')

required_files = [
    ROC_CURVE_PATH,
    CONFUSION_MATRIX_PATH,
    SHAP_SUMMARY_PATH,
    SHAP_BAR_PATH,
    MODEL_PATH,
]

for filepath in required_files:
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Required file not found: {filepath}\n"
            f"Ensure initial_training/model_training.py has been run to completion before generating the report."
        )

print("All required files found. Proceeding with report generation.")


# ── Step 3: Reconstruct predictions ───────────────────────────────────────────

# Copied exactly from initial_training/model_training.py
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

PHASE3_COLUMNS_TO_DROP = [
    # Cross-table features that did not outperform their components
    'internal_vs_external_dpd_diff',
    'credit_request_ratio',
    'annuity_request_ratio',
    'inst_std_days_late',
    # Redundant aggregated columns from within-table high-correlation pairs
    'days_since_last_application',
    'YEARS_BUILD_AVG',
    'OBS_60_CNT_SOCIAL_CIRCLE',
    'FLOORSMIN_MEDI',
    'FLOORSMAX_MEDI',
    'ENTRANCES_MEDI',
    'ELEVATORS_MEDI',
    'COMMONAREA_AVG',
    'LIVINGAREA_MEDI',
    'APARTMENTS_MEDI',
    'BASEMENTAREA_MEDI',
    'YEARS_BEGINEXPLUATATION_AVG',
    'LIVINGAPARTMENTS_MEDI',
    'pos_months_count',
    'LANDAREA_AVG',
    'NONLIVINGAPARTMENTS_MEDI',
    'NONLIVINGAREA_MEDI',
    'YEARS_BUILD_MODE',
    'FLOORSMIN_MODE',
    'FLOORSMAX_MODE',
    'AMT_CREDIT',
    'ELEVATORS_MODE',
    'LANDAREA_MODE',
    'ENTRANCES_MODE',
    'COMMONAREA_MODE',
    'NONLIVINGAPARTMENTS_MODE',
    'BASEMENTAREA_MODE',
    'APARTMENTS_MODE',
    'NONLIVINGAREA_MODE',
    'LIVINGAPARTMENTS_MODE',
    'LIVINGAREA_MODE',
    'YEARS_BEGINEXPLUATATION_MODE',
    'pos_avg_dpd_def',
    'cc_avg_dpd',
    'REGION_RATING_CLIENT',
    'LIVINGAPARTMENTS_AVG',
    'cc_dpd_rate',
    'cc_avg_utilization',
    'TOTALAREA_MODE',
    'pos_max_dpd',
    'bureau_loan_count',
    'APARTMENTS_AVG',
    'has_pos_cash',
    'inst_underpay_count',
    'bureau_total_prolonged',
    'internal_dpd_composite',
    'pos_avg_dpd',
    'CNT_FAM_MEMBERS',
    'cc_max_balance',
    'LIVINGAREA_AVG',
    'LIVE_REGION_NOT_WORK_REGION',
    'DEF_60_CNT_SOCIAL_CIRCLE',
    'cc_max_utilization',
    # Range violations
    'bureau_utilization_ratio',
    'inst_max_days_late',
]

METADATA_COLUMNS_TO_DROP = ['SK_ID_CURR']
ALL_COLUMNS_TO_DROP = PHASE1_COLUMNS_TO_DROP + PHASE3_COLUMNS_TO_DROP + METADATA_COLUMNS_TO_DROP
HIGH_CARDINALITY_COLUMNS = ['ORGANIZATION_TYPE']

print("Loading and preprocessing data...")
print("  Loading application_train.csv...")
df_app = pd.read_csv(os.path.join(DATA_DIR, 'application_train.csv'))
df_app = preprocess(df_app)

loan_applications_df = aggregate_all(df_app)
loan_applications_df = loan_applications_df.drop(columns=ALL_COLUMNS_TO_DROP, errors='ignore')

feature_matrix = loan_applications_df.drop(columns=['TARGET'])
target_series = loan_applications_df['TARGET']

features_train, features_test, target_train, target_test = train_test_split(
    feature_matrix, target_series,
    test_size=0.2,
    stratify=target_series,
    random_state=42
)

# Encode
categorical_cols = features_train.select_dtypes(include='object').columns.tolist()
standard_categorical_cols = [col for col in categorical_cols if col not in HIGH_CARDINALITY_COLUMNS]
high_cardinality_cols_present = [col for col in HIGH_CARDINALITY_COLUMNS if col in categorical_cols]

if high_cardinality_cols_present:
    target_encoder = TargetEncoder(target_type='binary', random_state=42)
    features_train[high_cardinality_cols_present] = target_encoder.fit_transform(
        features_train[high_cardinality_cols_present], target_train
    )
    features_test[high_cardinality_cols_present] = target_encoder.transform(
        features_test[high_cardinality_cols_present]
    )

label_encoders = {}
for col in standard_categorical_cols:
    le = LabelEncoder()
    features_train[col] = le.fit_transform(features_train[col].astype(str))
    features_test[col] = le.transform(features_test[col].astype(str))
    label_encoders[col] = le

# Impute
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

# Load model and generate predictions
print("Loading saved model...")
xgb_model = XGBClassifier()
xgb_model.load_model(MODEL_PATH)

predicted_probabilities = xgb_model.predict_proba(features_test)[:, 1]
predicted_labels = xgb_model.predict(features_test)
roc_auc = roc_auc_score(target_test, predicted_probabilities)

print(f"ROC-AUC: {roc_auc:.4f}")


# ── Step 4: Parse classification report ───────────────────────────────────────

report_dict = classification_report(
    target_test,
    predicted_labels,
    target_names=['Repays', 'Defaults'],
    output_dict=True
)

classification_rows = [
    ['Class', 'Precision', 'Recall', 'F1 Score', 'Support'],
    [
        'Repays',
        f"{report_dict['Repays']['precision']:.4f}",
        f"{report_dict['Repays']['recall']:.4f}",
        f"{report_dict['Repays']['f1-score']:.4f}",
        f"{int(report_dict['Repays']['support'])}",
    ],
    [
        'Defaults',
        f"{report_dict['Defaults']['precision']:.4f}",
        f"{report_dict['Defaults']['recall']:.4f}",
        f"{report_dict['Defaults']['f1-score']:.4f}",
        f"{int(report_dict['Defaults']['support'])}",
    ],
    [
        'Weighted avg',
        f"{report_dict['weighted avg']['precision']:.4f}",
        f"{report_dict['weighted avg']['recall']:.4f}",
        f"{report_dict['weighted avg']['f1-score']:.4f}",
        f"{int(report_dict['weighted avg']['support'])}",
    ],
]


# ── Step 5: Define styles ──────────────────────────────────────────────────────

page_width, page_height = A4
styles = getSampleStyleSheet()

style_title = ParagraphStyle(
    name='ReportTitle',
    parent=styles['Title'],
    fontSize=22,
    spaceAfter=8,
)
style_subtitle = ParagraphStyle(
    name='ReportSubtitle',
    parent=styles['Normal'],
    fontSize=12,
    textColor=colors.HexColor('#555555'),
    spaceAfter=24,
)
style_h1 = ParagraphStyle(
    name='SectionHeading',
    parent=styles['Heading1'],
    fontSize=16,
    spaceBefore=18,
    spaceAfter=8,
)
style_h2 = ParagraphStyle(
    name='SubsectionHeading',
    parent=styles['Heading2'],
    fontSize=13,
    spaceBefore=12,
    spaceAfter=6,
)
style_body = ParagraphStyle(
    name='BodyText',
    parent=styles['Normal'],
    fontSize=11,
    leading=16,
    spaceAfter=8,
)
style_caption = ParagraphStyle(
    name='Caption',
    parent=styles['Normal'],
    fontSize=9,
    textColor=colors.HexColor('#666666'),
    spaceAfter=16,
    alignment=1,
)

TABLE_HEADER_BG = colors.HexColor('#E8E8E8')
TABLE_ROW_ALT_BG = colors.HexColor('#F7F7F7')
TABLE_GRID = colors.HexColor('#CCCCCC')


def make_table_style(num_rows):
    row_backgrounds = []
    for i in range(1, num_rows):
        bg = colors.white if i % 2 == 1 else TABLE_ROW_ALT_BG
        row_backgrounds.append(('BACKGROUND', (0, i), (-1, i), bg))
    style = [
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, TABLE_GRID),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ] + row_backgrounds
    return TableStyle(style)


# ── Step 6: Page footer with page numbers ─────────────────────────────────────

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 9)
    canvas.setFillColor(colors.HexColor('#888888'))
    page_num_text = f"Page {doc.page}"
    canvas.drawCentredString(page_width / 2, 1.2 * cm, page_num_text)
    canvas.restoreState()


# ── Step 7: Build report content ──────────────────────────────────────────────

story = []
margin = 2 * cm
usable_width = page_width - 2 * margin


def embed_image(path, width=None, caption=None, max_height=560):
    width = width or usable_width
    with PILImage.open(path) as pil_img:
        px_w, px_h = pil_img.size
    height = width * px_h / px_w
    if height > max_height:
        height = max_height
        width = height * px_w / px_h
    img = Image(path, width=width, height=height)
    img.hAlign = 'CENTER'
    elements = [img]
    if caption:
        elements.append(Paragraph(caption, style_caption))
    return elements


# Cover page
story.append(Spacer(1, 3 * cm))
story.append(Paragraph("Home Credit Default Risk", style_title))
story.append(Paragraph("Model Training Pipeline Report", style_subtitle))
story.append(Spacer(1, 0.5 * cm))

from datetime import date
story.append(Paragraph(f"Generated: {date.today().strftime('%B %d, %Y')}", style_body))
story.append(Spacer(1, 0.5 * cm))
story.append(Paragraph("Model: XGBoost Classifier", style_body))
story.append(Paragraph(f"Test set ROC-AUC: {roc_auc:.4f}", style_body))
story.append(PageBreak())

# Section 1 — Model performance overview
story.append(Paragraph("1. Model Performance Overview", style_h1))
story.append(Paragraph(
    "The model was evaluated on a held-out test set comprising 20% of the full training data, "
    "stratified by the target variable to preserve the class distribution. The primary evaluation "
    "metric is ROC-AUC, which measures the model's ability to rank applicants by default risk "
    "regardless of the decision threshold chosen.",
    style_body
))
story.append(Spacer(1, 0.3 * cm))

auc_rows = [
    ['Metric', 'Value'],
    ['ROC-AUC', f"{roc_auc:.4f}"],
    ['Test set size', f"{len(target_test):,}"],
    ['Positive rate (test set)', f"{target_test.mean():.4f}"],
]
auc_table = Table(auc_rows, colWidths=[usable_width * 0.5, usable_width * 0.5])
auc_table.setStyle(make_table_style(len(auc_rows)))
story.append(auc_table)
story.append(Spacer(1, 0.5 * cm))

if roc_auc >= 0.80:
    interpretation = "Strong result — above 0.80 AUC. Verify no data leakage before treating this as final."
elif roc_auc >= 0.77:
    interpretation = "Good result — in the expected range for this dataset with the full supplementary feature set."
elif roc_auc >= 0.74:
    interpretation = "Moderate result — in the expected range for application_train features only. Check that supplementary table features are contributing."
else:
    interpretation = "Below expected range — investigate aggregation bugs, over-aggressive column dropping, or imputation issues."

story.append(Paragraph(f"Interpretation: {interpretation}", style_body))
story.append(PageBreak())

# Section 2 — ROC curve
story.append(Paragraph("2. ROC Curve", style_h1))
story.append(Paragraph(
    "The ROC curve plots the true positive rate against the false positive rate at every possible "
    "classification threshold. A perfect model would reach the top-left corner. The diagonal line "
    "represents a random classifier. The area under this curve (AUC) is the primary model metric.",
    style_body
))
story.append(Spacer(1, 0.3 * cm))
story.extend(embed_image(
    ROC_CURVE_PATH,
    width=usable_width * 0.85,
    caption=f"Figure 1. ROC Curve — AUC = {roc_auc:.4f}"
))
story.append(PageBreak())

# Section 3 — Classification report
story.append(Paragraph("3. Classification Report", style_h1))
story.append(Paragraph(
    "The classification report shows precision, recall, and F1 score for each class at the default "
    "decision threshold of 0.5. Given the class imbalance (approximately 92% repays, 8% defaults), "
    "these metrics should be interpreted with caution — the model is optimized for ranking (AUC) "
    "rather than for any specific threshold. Precision for the Defaults class answers: of all "
    "applicants the model flagged as likely to default, what fraction actually did? Recall answers: "
    "of all applicants who actually defaulted, what fraction did the model catch?",
    style_body
))
story.append(Spacer(1, 0.3 * cm))

classification_table = Table(
    classification_rows,
    colWidths=[usable_width * 0.28, usable_width * 0.18, usable_width * 0.18,
               usable_width * 0.18, usable_width * 0.18]
)
classification_table.setStyle(make_table_style(len(classification_rows)))
story.append(classification_table)
story.append(PageBreak())

# Section 4 — Confusion matrix
story.append(Paragraph("4. Confusion Matrix", style_h1))
story.append(Paragraph(
    "The confusion matrix shows the count of correct and incorrect predictions at the default "
    "threshold of 0.5. True negatives (top-left) are applicants correctly predicted to repay. "
    "True positives (bottom-right) are applicants correctly predicted to default. False positives "
    "(top-right) are applicants incorrectly flagged as defaulters. False negatives (bottom-left) "
    "are defaulters the model missed — in credit risk this is typically the more costly error.",
    style_body
))
story.append(Spacer(1, 0.3 * cm))
story.extend(embed_image(
    CONFUSION_MATRIX_PATH,
    width=usable_width * 0.65,
    caption="Figure 2. Confusion matrix at default threshold of 0.5."
))
story.append(PageBreak())

# Section 5 — SHAP feature importance
story.append(Paragraph("5. SHAP Feature Importance", style_h1))
story.append(Paragraph(
    "SHAP (SHapley Additive exPlanations) values measure each feature's contribution to individual "
    "predictions. Unlike XGBoost's native feature importance, SHAP accounts for feature interactions "
    "and provides directional information — showing not just which features matter, but whether "
    "high or low values of each feature push predictions toward default or repayment.",
    style_body
))
story.append(Spacer(1, 0.3 * cm))

story.append(Paragraph("5.1 Beeswarm Plot", style_h2))
story.append(Paragraph(
    "Each dot represents one applicant. Horizontal position shows the SHAP value — how far that "
    "feature pushed the prediction away from the baseline. Color represents the feature value: "
    "red indicates a high feature value, blue indicates a low feature value. Features are ranked "
    "by mean absolute SHAP value with the most important at the top.",
    style_body
))
story.extend(embed_image(
    SHAP_SUMMARY_PATH,
    width=usable_width,
    caption="Figure 3. SHAP beeswarm plot — top 30 features by mean absolute SHAP value."
))
story.append(PageBreak())

story.append(Paragraph("5.2 Mean Absolute SHAP Value", style_h2))
story.append(Paragraph(
    "The bar plot shows the mean absolute SHAP value for each feature — a single summary of overall "
    "importance without directional information. Features with a mean absolute SHAP value near zero "
    "are candidates for removal in a second training run.",
    style_body
))
story.extend(embed_image(
    SHAP_BAR_PATH,
    width=usable_width,
    caption="Figure 4. Mean absolute SHAP value per feature — top 30 features."
))
story.append(PageBreak())


# ── Step 8: Compile and save PDF ──────────────────────────────────────────────

doc = SimpleDocTemplate(
    OUTPUT_PDF_PATH,
    pagesize=A4,
    leftMargin=margin,
    rightMargin=margin,
    topMargin=margin,
    bottomMargin=margin,
)

doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
print(f"\nReport saved: {OUTPUT_PDF_PATH}")
print("Total pages: check the PDF directly for page count.")
