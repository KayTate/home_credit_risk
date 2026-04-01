"""
Home Credit Default Risk — Data Aggregation Pipeline

Aggregates all supplementary tables and joins them to the base application
dataframe. Designed to be reused by both the EDA pipeline and the ML pipeline.

Run from repo root. DATA_DIR is resolved relative to the working directory.
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ─── PREPROCESSING ────────────────────────────────────────────────────────────

def preprocess(df, compute_abs=True):
    """Replace 365243/XNA/XAP; optionally take abs of DAYS_ columns."""
    days_cols = [c for c in df.columns if c.startswith('DAYS_')]
    for col in days_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace(365243, np.nan)
    for col in df.select_dtypes('object').columns:
        df[col] = df[col].replace({'XNA': np.nan, 'XAP': np.nan})
    if compute_abs:
        for col in days_cols:
            df[col] = df[col].abs()
    return df


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _safe_mode(s):
    """Return most common non-null value, or NaN."""
    s = s.dropna()
    return s.mode().iloc[0] if len(s) > 0 else np.nan


# ─── TABLE AGGREGATIONS ───────────────────────────────────────────────────────

def agg_bureau():
    print("  [Agg] bureau.csv")
    bur = pd.read_csv(DATA_DIR / 'bureau.csv')
    bur = preprocess(bur)

    # Recent inquiry count (DAYS_CREDIT <= 365 after abs)
    recent = bur[bur['DAYS_CREDIT'] <= 365].groupby('SK_ID_CURR').size().rename('bureau_recent_inquiry_count')

    # CREDIT_TYPE flags
    bur['is_mortgage']         = (bur['CREDIT_TYPE'] == 'Mortgage').astype(int)
    bur['is_micro']            = (bur['CREDIT_TYPE'].str.contains('Microloan|microloan', na=False)).astype(int)
    bur['credit_type_encoded'] = bur['CREDIT_TYPE'].astype('category').cat.codes

    agg = bur.groupby('SK_ID_CURR').agg(
        bureau_loan_count        = ('SK_ID_BUREAU',           'count'),
        bureau_active_count      = ('CREDIT_ACTIVE',          lambda x: (x=='Active').sum()),
        bureau_closed_count      = ('CREDIT_ACTIVE',          lambda x: (x=='Closed').sum()),
        bureau_bad_debt_count    = ('CREDIT_ACTIVE',          lambda x: (x=='Bad debt').sum()),
        bureau_max_overdue       = ('AMT_CREDIT_MAX_OVERDUE', 'max'),
        bureau_total_debt        = ('AMT_CREDIT_SUM_DEBT',    'sum'),
        bureau_avg_debt          = ('AMT_CREDIT_SUM_DEBT',    'mean'),
        bureau_total_credit      = ('AMT_CREDIT_SUM',         'sum'),
        bureau_avg_credit_age    = ('DAYS_CREDIT',            'mean'),
        bureau_total_prolonged   = ('CNT_CREDIT_PROLONG',     'sum'),
        bureau_has_mortgage      = ('is_mortgage',            'max'),
        bureau_has_microfinance  = ('is_micro',               'max'),
        bureau_credit_type_count = ('CREDIT_TYPE',            'nunique'),
    ).reset_index()

    agg['bureau_active_ratio']      = agg['bureau_active_count'] / agg['bureau_loan_count'].clip(lower=1)
    agg['bureau_utilization_ratio'] = (agg['bureau_total_debt'] /
                                       agg['bureau_total_credit'].replace(0, 1e-6).clip(lower=1e-6))
    agg['bureau_any_prolonged']     = (agg['bureau_total_prolonged'] > 0).astype(int)

    agg = agg.merge(recent.reset_index(), on='SK_ID_CURR', how='left')
    agg['bureau_recent_inquiry_count'] = agg['bureau_recent_inquiry_count'].fillna(0)

    del bur
    gc.collect()
    return agg


def agg_bureau_balance():
    print("  [Agg] bureau_balance.csv (two-hop)")
    bb = pd.read_csv(DATA_DIR / 'bureau_balance.csv')
    # Keep MONTHS_BALANCE as-is (not a DAYS_ col); replace XNA/XAP
    for col in bb.select_dtypes('object').columns:
        bb[col] = bb[col].replace({'XNA': np.nan, 'XAP': np.nan})

    bb['status_num'] = pd.to_numeric(
        bb['STATUS'].replace({'C': np.nan, 'X': np.nan}), errors='coerce')
    bb['is_dpd']  = bb['STATUS'].isin(['1','2','3','4','5']).astype(int)
    bb['is_good'] = (bb['STATUS'] == '0').astype(int)

    bb_agg = bb.groupby('SK_ID_BUREAU').agg(
        bb_months_count = ('MONTHS_BALANCE', 'count'),
        bb_dpd_count    = ('is_dpd',         'sum'),
        bb_worst_status = ('status_num',     'max'),
        bb_good_months  = ('is_good',        'sum'),
    ).reset_index()
    del bb
    gc.collect()

    # Two-hop: join to bureau to get SK_ID_CURR
    bureau_ids = pd.read_csv(DATA_DIR / 'bureau.csv',
                             usecols=['SK_ID_BUREAU', 'SK_ID_CURR'])
    bb_agg = bb_agg.merge(bureau_ids, on='SK_ID_BUREAU', how='left')
    del bureau_ids
    gc.collect()

    agg = bb_agg.groupby('SK_ID_CURR').agg(
        bureau_bal_total_months     = ('bb_months_count', 'sum'),
        bureau_bal_total_dpd        = ('bb_dpd_count',    'sum'),
        bureau_bal_avg_dpd_per_loan = ('bb_dpd_count',    'mean'),
        bureau_bal_worst_ever_status= ('bb_worst_status', 'max'),
        bureau_bal_total_good_months= ('bb_good_months',  'sum'),
    ).reset_index()
    agg['bureau_bal_dpd_rate'] = (agg['bureau_bal_total_dpd'] /
                                  agg['bureau_bal_total_months'].clip(lower=1))
    del bb_agg
    gc.collect()
    return agg


def agg_previous():
    print("  [Agg] previous_application.csv")
    prev = pd.read_csv(DATA_DIR / 'previous_application.csv')
    prev = preprocess(prev)

    prev['is_approved']  = (prev['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
    prev['is_refused']   = (prev['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    prev['is_canceled']  = (prev['NAME_CONTRACT_STATUS'] == 'Cancelled').astype(int)
    prev['is_returning'] = (prev['NAME_CLIENT_TYPE'] == 'Returning customer').astype(int)

    # Days since last approval (approved rows only)
    last_approval = (prev[prev['is_approved']==1]
                     .groupby('SK_ID_CURR')['DAYS_DECISION']
                     .min().rename('days_since_last_approval'))

    agg = prev.groupby('SK_ID_CURR').agg(
        prev_app_count                 = ('SK_ID_PREV',        'count'),
        prev_approved_count            = ('is_approved',        'sum'),
        prev_refused_count             = ('is_refused',         'sum'),
        prev_canceled_count            = ('is_canceled',        'sum'),
        prev_most_recent_decision      = ('DAYS_DECISION',      'min'),
        prev_avg_decision_days         = ('DAYS_DECISION',      'mean'),
        prev_avg_credit                = ('AMT_CREDIT',         'mean'),
        prev_max_credit                = ('AMT_CREDIT',         'max'),
        prev_avg_annuity               = ('AMT_ANNUITY',        'mean'),
        prev_avg_down_payment          = ('AMT_DOWN_PAYMENT',   'mean'),
        prev_is_returning_customer     = ('is_returning',       'max'),
        prev_most_common_reject_reason = ('CODE_REJECT_REASON', _safe_mode),
        prev_most_common_yield_group   = ('NAME_YIELD_GROUP',   _safe_mode),
    ).reset_index()
    agg['prev_approval_rate'] = agg['prev_approved_count'] / agg['prev_app_count'].clip(lower=1)
    agg = agg.merge(last_approval.reset_index(), on='SK_ID_CURR', how='left')
    agg['days_since_last_application'] = agg['prev_most_recent_decision']

    del prev
    gc.collect()
    return agg


def agg_installments():
    print("  [Agg] installments_payments.csv")
    inst = pd.read_csv(DATA_DIR / 'installments_payments.csv')
    # Replace 365243 and XNA/XAP but DO NOT take abs yet (need raw values for days_late)
    inst = preprocess(inst, compute_abs=False)

    # Derived columns BEFORE taking abs
    inst['payment_diff'] = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
    inst['days_late']    = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']

    # Now take abs of DAYS_ columns
    for col in [c for c in inst.columns if c.startswith('DAYS_')]:
        inst[col] = inst[col].abs()

    inst['is_underpay'] = (inst['payment_diff'] > 0).astype(int)
    inst['is_late']     = (inst['days_late'] > 0).astype(int)

    agg = inst.groupby('SK_ID_CURR').agg(
        inst_total_count      = ('SK_ID_PREV',   'count'),
        inst_avg_payment_diff = ('payment_diff', 'mean'),
        inst_max_payment_diff = ('payment_diff', 'max'),
        inst_underpay_count   = ('is_underpay',  'sum'),
        inst_avg_days_late    = ('days_late',     'mean'),
        inst_max_days_late    = ('days_late',     'max'),
        inst_late_count       = ('is_late',       'sum'),
        inst_std_days_late    = ('days_late',     'std'),
        inst_std_payment_diff = ('payment_diff',  'std'),
    ).reset_index()
    agg['inst_underpay_rate'] = agg['inst_underpay_count'] / agg['inst_total_count'].clip(lower=1)
    agg['inst_late_rate']     = agg['inst_late_count']     / agg['inst_total_count'].clip(lower=1)
    del inst
    gc.collect()
    return agg


def agg_pos_cash():
    print("  [Agg] POS_CASH_balance.csv")
    pos = pd.read_csv(DATA_DIR / 'POS_CASH_balance.csv')
    # MONTHS_BALANCE is not a DAYS_ col — unaffected by abs; replace XNA/XAP only
    for col in pos.select_dtypes('object').columns:
        pos[col] = pos[col].replace({'XNA': np.nan, 'XAP': np.nan})

    pos['is_dpd']     = (pos['SK_DPD'] > 0).astype(int)
    pos['completed']  = (pos['NAME_CONTRACT_STATUS'] == 'Completed').astype(int)
    pos['active_cnt'] = (pos['NAME_CONTRACT_STATUS'] == 'Active').astype(int)
    pos['demand_cnt'] = (pos['NAME_CONTRACT_STATUS'] == 'Demand').astype(int)

    # Trend: recent (MONTHS_BALANCE >= -6) vs historical
    pos_recent = pos[pos['MONTHS_BALANCE'] >= -6].groupby('SK_ID_CURR')['SK_DPD'].mean().rename('pos_recent_dpd')
    pos_hist   = pos[pos['MONTHS_BALANCE'] <  -6].groupby('SK_ID_CURR')['SK_DPD'].mean().rename('pos_hist_dpd')

    agg = pos.groupby('SK_ID_CURR').agg(
        pos_months_count    = ('MONTHS_BALANCE', 'count'),
        pos_avg_dpd         = ('SK_DPD',         'mean'),
        pos_max_dpd         = ('SK_DPD',         'max'),
        pos_avg_dpd_def     = ('SK_DPD_DEF',     'mean'),
        pos_max_dpd_def     = ('SK_DPD_DEF',     'max'),
        pos_dpd_month_count = ('is_dpd',         'sum'),
        pos_completed_count = ('completed',      'sum'),
        pos_active_count    = ('active_cnt',     'sum'),
        pos_demand_count    = ('demand_cnt',     'sum'),
    ).reset_index()
    agg['pos_dpd_rate'] = agg['pos_dpd_month_count'] / agg['pos_months_count'].clip(lower=1)
    agg = agg.merge(pos_recent.reset_index(), on='SK_ID_CURR', how='left')
    agg = agg.merge(pos_hist.reset_index(),   on='SK_ID_CURR', how='left')
    agg['pos_dpd_trend'] = agg['pos_recent_dpd'] - agg['pos_hist_dpd']
    agg.drop(columns=['pos_recent_dpd', 'pos_hist_dpd'], inplace=True)

    del pos
    gc.collect()
    return agg


def agg_credit_card():
    print("  [Agg] credit_card_balance.csv")
    cc = pd.read_csv(DATA_DIR / 'credit_card_balance.csv')
    for col in cc.select_dtypes('object').columns:
        cc[col] = cc[col].replace({'XNA': np.nan, 'XAP': np.nan})

    eps = 1e-6
    cc['utilization']      = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, eps).clip(lower=eps)
    cc['paying_above_min'] = (cc['AMT_PAYMENT_CURRENT'] > cc['AMT_INST_MIN_REGULARITY']).astype(float)
    cc['has_atm_drawing']  = (cc['AMT_DRAWINGS_ATM_CURRENT'] > 0).astype(int)
    cc['is_dpd']           = (cc['SK_DPD'] > 0).astype(int)
    cc['is_active_month']  = (cc['AMT_BALANCE'] > 0).astype(int)

    # Trend windows
    cc_recent = cc[cc['MONTHS_BALANCE'] >= -6].groupby('SK_ID_CURR').agg(
        cc_recent_dpd  = ('SK_DPD',      'mean'),
        cc_recent_util = ('utilization', 'mean'),
    ).reset_index()
    cc_hist = cc[cc['MONTHS_BALANCE'] < -6].groupby('SK_ID_CURR').agg(
        cc_hist_dpd  = ('SK_DPD',      'mean'),
        cc_hist_util = ('utilization', 'mean'),
    ).reset_index()

    agg = cc.groupby('SK_ID_CURR').agg(
        cc_months_count       = ('MONTHS_BALANCE',           'count'),
        cc_avg_utilization    = ('utilization',               'mean'),
        cc_max_utilization    = ('utilization',               'max'),
        cc_avg_balance        = ('AMT_BALANCE',               'mean'),
        cc_max_balance        = ('AMT_BALANCE',               'max'),
        cc_avg_atm_drawings   = ('AMT_DRAWINGS_ATM_CURRENT', 'mean'),
        cc_total_atm_drawings = ('AMT_DRAWINGS_ATM_CURRENT', 'sum'),
        cc_any_atm_drawings   = ('has_atm_drawing',          'max'),
        cc_above_minimum_rate = ('paying_above_min',         'mean'),
        cc_avg_dpd            = ('SK_DPD',                   'mean'),
        cc_max_dpd            = ('SK_DPD',                   'max'),
        cc_dpd_month_count    = ('is_dpd',                   'sum'),
        cc_active_month_count = ('is_active_month',          'sum'),
    ).reset_index()
    agg['cc_dpd_rate'] = agg['cc_dpd_month_count'] / agg['cc_months_count'].clip(lower=1)
    agg = agg.merge(cc_recent, on='SK_ID_CURR', how='left')
    agg = agg.merge(cc_hist,   on='SK_ID_CURR', how='left')
    agg['cc_dpd_trend']         = agg['cc_recent_dpd']  - agg['cc_hist_dpd']
    agg['cc_utilization_trend'] = agg['cc_recent_util'] - agg['cc_hist_util']
    agg.drop(columns=['cc_recent_dpd','cc_hist_dpd','cc_recent_util','cc_hist_util'],
             inplace=True)

    del cc
    gc.collect()
    return agg


# ─── MAIN JOIN ────────────────────────────────────────────────────────────────

def aggregate_all(df_app):
    """Aggregate all supplementary tables and left-join them to df_app.

    Also creates missingness flags (has_*) and cross-table features.
    No imputation is applied — nulls remain for downstream handling.
    """
    print("  [Agg] Joining all supplementary tables…")

    bur  = agg_bureau()
    bb   = agg_bureau_balance()
    prev = agg_previous()
    inst = agg_installments()
    pos  = agg_pos_cash()
    ccb  = agg_credit_card()

    joined = df_app.copy()
    for agg_df, name in [(bur,'bureau'),(bb,'bureau_bal'),(prev,'prev'),
                         (inst,'inst'),(pos,'pos'),(ccb,'cc')]:
        joined = joined.merge(agg_df, on='SK_ID_CURR', how='left')
        print(f"    merged {name}: shape={joined.shape}")
    del bur, bb, prev, inst, pos, ccb
    gc.collect()

    # ── Missingness flags (before any imputation) ──────────────────────────
    joined['has_bureau_record']    = joined['bureau_loan_count'].notna().astype(int)
    joined['has_bureau_balance']   = joined['bureau_bal_total_months'].notna().astype(int)
    joined['has_prev_application'] = joined['prev_app_count'].notna().astype(int)
    joined['has_installments']     = joined['inst_total_count'].notna().astype(int)
    joined['has_pos_cash']         = joined['pos_months_count'].notna().astype(int)
    joined['has_credit_card']      = joined['cc_months_count'].notna().astype(int)

    # ── Cross-table features ───────────────────────────────────────────────
    joined['bureau_bal_any_dpd'] = (joined['bureau_bal_total_dpd'].fillna(0) > 0).astype(int)
    joined['pos_any_dpd']        = (joined['pos_dpd_month_count'].fillna(0)  > 0).astype(int)
    joined['cc_any_dpd']         = (joined['cc_dpd_month_count'].fillna(0)   > 0).astype(int)
    joined['inst_any_late']      = (joined['inst_late_count'].fillna(0)      > 0).astype(int)
    joined['systemic_delinquency_score'] = (joined['bureau_bal_any_dpd'] +
                                            joined['pos_any_dpd'] +
                                            joined['cc_any_dpd'] +
                                            joined['inst_any_late'])

    dpd_cols = ['pos_dpd_rate', 'cc_dpd_rate', 'inst_late_rate']
    joined['internal_dpd_composite']        = joined[dpd_cols].mean(axis=1)
    joined['internal_vs_external_dpd_diff'] = (joined['internal_dpd_composite'] -
                                                joined['bureau_bal_dpd_rate'])

    eps = 1e-6
    joined['credit_request_ratio']  = (joined['AMT_CREDIT'] /
                                        joined['prev_avg_credit'].replace(0, eps).clip(lower=eps))
    joined['annuity_request_ratio'] = (joined['AMT_ANNUITY'] /
                                        joined['prev_avg_annuity'].replace(0, eps).clip(lower=eps))
    # Null out ratio features where there is no prior application
    for col in ['credit_request_ratio', 'annuity_request_ratio']:
        joined.loc[joined['has_prev_application'] == 0, col] = np.nan

    joined['total_credit_sources'] = (joined['has_bureau_record'] +
                                      joined['has_bureau_balance'] +
                                      joined['has_prev_application'] +
                                      joined['has_installments'] +
                                      joined['has_pos_cash'] +
                                      joined['has_credit_card'])

    joined['days_since_last_application'] = joined['prev_most_recent_decision']

    print(f"  [Agg] Final joined shape: {joined.shape}")
    return joined
