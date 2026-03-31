#!/usr/bin/env python3
"""
Home Credit Default Risk — EDA Pipeline
Produces eda/eda_report.pdf from data/*.csv files.
Run from repo root: python eda/eda_pipeline.py
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle,
    Image as RLImage, Spacer, PageBreak, HRFlowable
)
from pathlib import Path

import matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
import math
import gc

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_aggregation import preprocess, aggregate_all

warnings.filterwarnings('ignore')
matplotlib.use('Agg')



# ─── PATHS ───────────────────────────────────────────────────────────────────
DATA_DIR  = Path("data")
PLOTS_DIR = Path("eda/plots")
OUTPUT_PDF = Path("eda/eda_report.pdf")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── PDF LAYOUT CONSTANTS ────────────────────────────────────────────────────
A4_W, A4_H = A4
MARGIN = 2 * cm
CONTENT_W = A4_W - 2 * MARGIN

GRAY_HEADER = HexColor('#D3D3D3')
GRAY_ROW    = HexColor('#F5F5F5')
BASE_RATE   = 0.08

# ─── PDF STYLES ──────────────────────────────────────────────────────────────
def _make_styles():
    getSampleStyleSheet()
    return {
        'h1':      ParagraphStyle(name='H1',      fontSize=16, fontName='Helvetica-Bold',
                                  spaceBefore=14, spaceAfter=8,  leading=20),
        'h2':      ParagraphStyle(name='H2',      fontSize=13, fontName='Helvetica-Bold',
                                  spaceBefore=10, spaceAfter=6,  leading=16),
        'body':    ParagraphStyle(name='Body',    fontSize=11, fontName='Helvetica',
                                  spaceAfter=6,  leading=14),
        'caption': ParagraphStyle(name='Caption', fontSize=10, fontName='Helvetica-Oblique',
                                  spaceAfter=10, leading=13, alignment=TA_CENTER),
        'mono':    ParagraphStyle(name='Mono',    fontSize=9,  fontName='Courier',
                                  spaceAfter=4,  leading=11),
        'small':   ParagraphStyle(name='Small',   fontSize=9,  fontName='Helvetica',
                                  spaceAfter=4,  leading=11),
    }

STYLES = _make_styles()

# ─── PDF HELPER FUNCTIONS ────────────────────────────────────────────────────
def _rl_str(value):
    """Convert value to safe PDF string."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 'N/A'
        return f'{value:.4g}'
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'N/A'
    text = str(value)
    # Escape XML special chars
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    return text[:60]  # truncate very long strings

def _cell(value, bold=False, small=False):
    """Wrap a value in a reportlab Paragraph using the appropriate table cell style."""
    style = STYLES['small'] if small else STYLES['body']
    if bold:
        style = ParagraphStyle(name='TH', fontSize=10, fontName='Helvetica-Bold',
                                leading=12, spaceAfter=2)
    else:
        style = ParagraphStyle(name='TD', fontSize=9, fontName='Helvetica',
                                leading=11, spaceAfter=2)
    return Paragraph(_rl_str(value), style)

def _df_to_table(df, max_rows=None, col_widths=None):
    """Convert DataFrame to reportlab Table."""
    display = df.head(max_rows) if max_rows else df
    headers = list(display.columns)
    n = len(headers)

    if col_widths is None:
        col_widths = [CONTENT_W / n] * n

    rows = []
    # Header
    hdr = [Paragraph(_rl_str(h)[:30], ParagraphStyle(name='TH', fontSize=9,
           fontName='Helvetica-Bold', leading=11)) for h in headers]
    rows.append(hdr)
    # Data
    for _, row in display.iterrows():
        rows.append([Paragraph(_rl_str(v), ParagraphStyle(name='TD', fontSize=9,
                    fontName='Helvetica', leading=11)) for v in row])

    ts = TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0),  GRAY_HEADER),
        ('GRID',          (0, 0), (-1, -1), 0.4, colors.grey),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING',    (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING',   (0, 0), (-1, -1), 3),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 3),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, GRAY_ROW]),
    ])
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(ts)
    return t

def _embed_img(path, width=None):
    """Embed a saved PNG into the PDF, preserving aspect ratio."""
    if not Path(path).exists():
        return Spacer(1, 0.5*cm)
    w = width or CONTENT_W
    ir = ImageReader(str(path))
    iw, ih = ir.getSize()
    scale = w / iw
    h = ih * scale
    max_h = 0.78 * (A4_H - 2 * MARGIN)
    if h > max_h:
        h = max_h
        w = iw * (h / ih)
    return RLImage(str(path), width=w, height=h)

def _sec(story, title):
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(title, STYLES['h1']))
    story.append(HRFlowable(width=CONTENT_W, thickness=1, color=colors.darkgrey))
    story.append(Spacer(1, 0.2*cm))

def _sub(story, title):
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(title, STYLES['h2']))

def _txt(story, text):
    story.append(Paragraph(text, STYLES['body']))

def _cap(story, text):
    story.append(Paragraph(text, STYLES['caption']))

def _tbl(story, df, caption='', max_rows=None, col_widths=None):
    if df is None or len(df) == 0:
        story.append(Paragraph('(No data)', STYLES['small']))
        return
    story.append(_df_to_table(df, max_rows=max_rows, col_widths=col_widths))
    if caption:
        story.append(Spacer(1, 0.1*cm))
        _cap(story, caption)

def _img(story, path, caption='', width=None):
    story.append(Spacer(1, 0.2*cm))
    img = _embed_img(path, width=width)
    story.append(img)
    if caption:
        story.append(Spacer(1, 0.1*cm))
        _cap(story, caption)
    story.append(Spacer(1, 0.2*cm))

def _page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 9)
    canvas.drawRightString(A4_W - MARGIN, 0.7 * cm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()

def save_fig(fig, name, dpi=150):
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA STRUCTURE & QUALITY
# ═══════════════════════════════════════════════════════════════════════════════
def p1_structure(df):
    print("  [1.1] Data structure and quality")
    r = {}
    r['shape'] = df.shape

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes('object').columns.tolist()
    r['num_cols'] = num_cols
    r['cat_cols'] = cat_cols
    r['dtype_counts'] = pd.DataFrame([
        {'dtype': 'numeric (int64/float64)', 'count': len(num_cols)},
        {'dtype': 'categorical (object)',    'count': len(cat_cols)},
    ])

    # Pseudo-categorical: values only in {0,1,2}
    pseudo = []
    for col in num_cols:
        u = set(df[col].dropna().unique())
        if u and u.issubset({0.0, 1.0, 2.0}):
            pseudo.append(col)
    r['pseudo_cat'] = pseudo

    # Null rates
    nct = df.isnull().sum()
    npc = nct / len(df) * 100
    null_df = pd.DataFrame({'column': nct.index, 'null_count': nct.values,
                             'null_pct': npc.round(2).values
                            }).sort_values('null_pct', ascending=False).reset_index(drop=True)
    null_df['flag'] = null_df['null_pct'].apply(
        lambda x: '>60% DROP' if x > 60 else ('0% clean' if x == 0 else '1-60% impute'))
    r['null_df'] = null_df
    r['high_null_cols'] = null_df.loc[null_df['null_pct'] > 60, 'column'].tolist()

    # Numeric min/max with plausibility flags
    records = []
    for col in num_cols:
        mn = df[col].min()
        mx = df[col].max()
        flag = ''
        if 'AMT_' in col and pd.notna(mn) and mn < 0:
            flag = 'neg amount'
        if col == 'DAYS_BIRTH' and pd.notna(mn):
            yr_min, yr_max = mn/365, mx/365
            if not (15 <= yr_min <= 80):
                flag = f'age {yr_min:.0f}-{yr_max:.0f}yr'
        records.append({'column': col,
                        'min': round(float(mn), 4) if pd.notna(mn) else float('nan'),
                        'max': round(float(mx), 4) if pd.notna(mx) else float('nan'),
                        'flag': flag})
    r['minmax_df'] = pd.DataFrame(records)

    # Categorical unique counts
    r['cat_unique_df'] = pd.DataFrame([
        {'column': c, 'n_unique': int(df[c].nunique()),
         'high_cardinality': df[c].nunique() > 20}
        for c in cat_cols
    ]).sort_values('n_unique', ascending=False).reset_index(drop=True)

    # Duplicate SK_ID_CURR
    r['n_dupes'] = int(df['SK_ID_CURR'].duplicated().sum())
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — TARGET VARIABLE
# ═══════════════════════════════════════════════════════════════════════════════
def p1_target(df):
    print("  [1.2] Target variable")
    r = {}
    vc = df['TARGET'].value_counts().sort_index()
    pct = df['TARGET'].value_counts(normalize=True).sort_index() * 100
    r['target_df'] = pd.DataFrame({
        'TARGET': ['0 (Repays)', '1 (Defaults)'],
        'count':  [int(vc.get(0, 0)), int(vc.get(1, 0))],
        'pct':    [round(float(pct.get(0, 0)), 2), round(float(pct.get(1, 0)), 2)],
    })
    n0, n1 = int(vc.get(0, 0)), int(vc.get(1, 0))
    r['positive_rate']    = n1 / (n0 + n1) if (n0 + n1) > 0 else 0
    r['scale_pos_weight'] = n0 / n1 if n1 > 0 else 1

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['0 (Repays)', '1 (Defaults)'], [n0, n1],
                  color=['#4C72B0', '#DD8452'], edgecolor='grey')
    for bar, v in zip(bars, [n0, n1]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{v:,}', ha='center', fontsize=11)
    ax.set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlabel('TARGET', fontsize=12)
    ax.tick_params(labelsize=10)
    r['target_plot'] = save_fig(fig, 'p1_target_dist')
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — NUMERIC FEATURE SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════
def p1_numeric_signal(df):
    print("  [1.3] Numeric feature signal")
    r = {}
    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in ('SK_ID_CURR', 'TARGET')]

    g0 = df[df['TARGET'] == 0][feat_cols].mean()
    g1 = df[df['TARGET'] == 1][feat_cols].mean()
    ov = df[feat_cols].mean()
    norm_diff = (g1 - g0).abs() / ov.abs().replace(0, np.nan)

    signal_df = pd.DataFrame({
        'feature':       feat_cols,
        'mean_target0':  g0.round(4).values,
        'mean_target1':  g1.round(4).values,
        'norm_diff':     norm_diff.values,
    }).dropna(subset=['norm_diff']).sort_values('norm_diff', ascending=False).reset_index(drop=True)
    signal_df['norm_diff'] = signal_df['norm_diff'].round(4)
    r['signal_df'] = signal_df
    r['top30']     = signal_df.head(30).copy()
    top20 = signal_df.head(20)['feature'].tolist()
    r['top20'] = top20

    # KDE grid — top 20 features
    n_cols_grid = 2
    n_rows_grid = math.ceil(len(top20) / n_cols_grid)
    fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(10, n_rows_grid * 2.8))
    axes = axes.flatten()
    df0, df1 = df[df['TARGET'] == 0], df[df['TARGET'] == 1]

    for i, col in enumerate(top20):
        ax = axes[i]
        v0 = df0[col].dropna()
        v1 = df1[col].dropna()
        p1v, p99v = df[col].quantile([0.01, 0.99])
        v0c = v0.clip(p1v, p99v)
        v1c = v1.clip(p1v, p99v)
        if len(v0c.unique()) > 2 and len(v0c) > 10:
            v0c.plot.kde(ax=ax, color='#4C72B0', label='TARGET=0', linewidth=1.5)
        if len(v1c.unique()) > 2 and len(v1c) > 10:
            v1c.plot.kde(ax=ax, color='#DD8452', label='TARGET=1', linewidth=1.5)
        m0, m1 = float(v0.mean()), float(v1.mean())
        ax.axvline(m0, color='#4C72B0', linestyle='--', alpha=0.6, linewidth=1)
        ax.axvline(m1, color='#DD8452', linestyle='--', alpha=0.6, linewidth=1)
        ax.set_title(f'{col}\n0:{m0:.3g}  1:{m1:.3g}', fontsize=8, fontweight='bold')
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('')
        ax.set_ylabel('')
    for j in range(len(top20), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Top 20 Numeric Features — KDE by Target Class', fontsize=12, fontweight='bold')
    plt.tight_layout()
    r['kde_grid_plot'] = save_fig(fig, 'p1_kde_grid')

    # Skewness
    skew_recs = []
    for col in feat_cols:
        sk = float(df[col].skew())
        if abs(sk) > 2:
            p99v = float(df[col].quantile(0.99))
            skew_recs.append({
                'feature': col, 'skewness': round(sk, 3),
                'min': round(float(df[col].min()), 4),
                'max': round(float(df[col].max()), 4),
                'mean': round(float(df[col].mean()), 4),
                'p99': round(p99v, 4),
            })
    r['skew_df'] = (pd.DataFrame(skew_recs)
                    .sort_values('skewness', key=abs, ascending=False)
                    .reset_index(drop=True))

    # EXT_SOURCE
    ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in df.columns]
    ext_recs = []
    for col in ext_cols:
        ext_recs.append({
            'column':       col,
            'null_rate':    round(float(df[col].isnull().mean()), 4),
            'mean_target0': round(float(df[df['TARGET']==0][col].mean()), 4),
            'mean_target1': round(float(df[df['TARGET']==1][col].mean()), 4),
        })
    r['ext_source_df']  = pd.DataFrame(ext_recs)
    r['ext_source_corr'] = df[ext_cols].corr().round(4) if ext_cols else pd.DataFrame()

    if ext_cols:
        fig, axes = plt.subplots(1, len(ext_cols), figsize=(5*len(ext_cols), 3.5))
        if len(ext_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, ext_cols):
            v0 = df[df['TARGET']==0][col].dropna()
            v1 = df[df['TARGET']==1][col].dropna()
            if len(v0) > 10:
                v0.plot.kde(ax=ax, color='#4C72B0', label='TARGET=0', lw=1.5)
            if len(v1) > 10:
                v1.plot.kde(ax=ax, color='#DD8452', label='TARGET=1', lw=1.5)
            ax.set_title(col, fontsize=10, fontweight='bold')
            ax.legend(fontsize=9)
            ax.tick_params(labelsize=9)
        fig.suptitle('EXT_SOURCE — KDE by Target Class', fontsize=11, fontweight='bold')
        plt.tight_layout()
        r['ext_kde_plot'] = save_fig(fig, 'p1_ext_kde')

    # FLAG columns
    flag_cols = [c for c in df.columns if c.startswith('FLAG_')]
    flag_recs = []
    for col in flag_cols:
        r0 = df[df[col]==0]['TARGET'].mean() if (df[col]==0).any() else np.nan
        r1 = df[df[col]==1]['TARGET'].mean() if (df[col]==1).any() else np.nan
        if pd.notna(r0) and pd.notna(r1):
            flag_recs.append({'flag': col, 'default_rate_0': round(float(r0),4),
                              'default_rate_1': round(float(r1),4),
                              'abs_diff': round(abs(float(r1)-float(r0)),4)})
    r['flag_df'] = (pd.DataFrame(flag_recs)
                    .sort_values('abs_diff', ascending=False)
                    .reset_index(drop=True))
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — CATEGORICAL FEATURE SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════
def p1_categorical_signal(df):
    print("  [1.4] Categorical feature signal")
    r = {}
    cat_cols = df.select_dtypes('object').columns.tolist()
    cat_results = {}
    for col in cat_cols:
        agg = (df.groupby(col, dropna=False)['TARGET']
               .agg(['count', 'mean']).reset_index())
        agg.columns = ['category', 'count', 'default_rate']
        agg['default_rate'] = agg['default_rate'].round(4)
        agg = agg.sort_values('default_rate', ascending=False).reset_index(drop=True)
        agg['anomaly'] = agg['default_rate'].apply(
            lambda x: 'HIGH >50%' if x > BASE_RATE*1.5 else
                      ('LOW <50%' if x < BASE_RATE*0.5 else ''))
        agg['rare'] = agg['count'] < 100
        cat_results[col] = agg
    r['cat_results'] = cat_results
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — FEATURE CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════════
def p1_correlations(df):
    print("  [1.5] Feature correlations (this may take a moment)…")
    r = {}

    # Engineered features
    df_e = df.copy()
    eps = 1e-9
    df_e['CREDIT_INCOME_RATIO']  = df_e['AMT_CREDIT']  / df_e['AMT_INCOME_TOTAL'].replace(0, eps)
    df_e['ANNUITY_INCOME_RATIO'] = df_e['AMT_ANNUITY'] / df_e['AMT_INCOME_TOTAL'].replace(0, eps)
    if 'AMT_GOODS_PRICE' in df_e.columns:
        df_e['CREDIT_GOODS_RATIO'] = df_e['AMT_CREDIT'] / df_e['AMT_GOODS_PRICE'].replace(0, eps)
    if 'DAYS_BIRTH' in df_e.columns:
        df_e['AGE_YEARS'] = df_e['DAYS_BIRTH'] / 365.0
    if 'DAYS_EMPLOYED' in df_e.columns:
        df_e['EMPLOYED_YEARS'] = df_e['DAYS_EMPLOYED'] / 365.0
    if 'DAYS_EMPLOYED' in df_e.columns and 'DAYS_BIRTH' in df_e.columns:
        df_e['EMPLOYMENT_RATIO'] = df_e['DAYS_EMPLOYED'] / df_e['DAYS_BIRTH'].replace(0, eps)
    ext_cols = [c for c in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3'] if c in df_e.columns]
    if ext_cols:
        df_e['EXT_SOURCE_MEAN'] = df_e[ext_cols].mean(axis=1)
        df_e['EXT_SOURCE_MIN']  = df_e[ext_cols].min(axis=1)

    eng_meta = {
        'CREDIT_INCOME_RATIO':  ['AMT_CREDIT', 'AMT_INCOME_TOTAL'],
        'ANNUITY_INCOME_RATIO': ['AMT_ANNUITY','AMT_INCOME_TOTAL'],
        'CREDIT_GOODS_RATIO':   ['AMT_CREDIT', 'AMT_GOODS_PRICE'],
        'AGE_YEARS':            ['DAYS_BIRTH'],
        'EMPLOYED_YEARS':       ['DAYS_EMPLOYED'],
        'EMPLOYMENT_RATIO':     ['DAYS_EMPLOYED','DAYS_BIRTH'],
        'EXT_SOURCE_MEAN':      ext_cols,
        'EXT_SOURCE_MIN':       ext_cols,
    }

    num_all = [c for c in df_e.select_dtypes(include=[np.number]).columns
               if c != 'SK_ID_CURR']

    # Correlation with TARGET
    tc = df_e[num_all].corrwith(df_e['TARGET']).abs().sort_values(ascending=False)
    r['target_corr_df'] = pd.DataFrame({'feature': tc.index,
                                        'abs_corr_target': tc.round(4).values})
    tc_dict = tc.to_dict()

    # Engineered feature table
    eng_recs = []
    for eng, comps in eng_meta.items():
        if eng not in df_e.columns:
            continue
        ec = abs(float(df_e[eng].corr(df_e['TARGET'])))
        comp_vals = {c: abs(float(df_e[c].corr(df_e['TARGET']))) if c in df_e.columns
                     else np.nan for c in comps}
        max_comp = max((v for v in comp_vals.values() if pd.notna(v)), default=np.nan)
        outperforms = bool(ec > max_comp) if pd.notna(max_comp) else False
        row = {'feature': eng, 'corr': round(ec,4),
               'outperforms_components': outperforms}
        for c, v in comp_vals.items():
            row[f'comp_{c}'] = round(v,4) if pd.notna(v) else 'N/A'
        eng_recs.append(row)
    r['eng_df'] = pd.DataFrame(eng_recs)

    # Full correlation matrix
    corr_mat = df_e[num_all].corr()
    r['corr_matrix'] = corr_mat

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(corr_mat, cmap='coolwarm', center=0, ax=ax,
                xticklabels=False, yticklabels=False, cbar=True)
    ax.set_title('Pairwise Correlation Matrix — All Numeric Features (Phase 1)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    r['corr_heatmap'] = save_fig(fig, 'p1_corr_heatmap')

    # High-corr pairs (>0.85)
    cols_list = corr_mat.columns.tolist()
    high_corr = []
    for i in range(len(cols_list)):
        for j in range(i+1, len(cols_list)):
            c1, c2 = cols_list[i], cols_list[j]
            val = abs(float(corr_mat.loc[c1, c2]))
            if val > 0.85:
                t1 = tc_dict.get(c1, 0)
                t2 = tc_dict.get(c2, 0)
                high_corr.append({'col1': c1, 'col2': c2,
                                  'abs_corr': round(val, 4),
                                  'prefer': c1 if t1 >= t2 else c2})
    r['high_corr_df'] = (pd.DataFrame(high_corr)
                         .sort_values('abs_corr', ascending=False)
                         .reset_index(drop=True))
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
def p1_summary(struct, tgt, num, cat, corr):
    print("  [1.6] Phase 1 summary")
    r = {}
    r['shape'] = struct['shape']

    nd = struct['null_df']
    r['null_buckets'] = {
        '0% null':    int((nd['null_pct']==0).sum()),
        '1–20% null': int(((nd['null_pct']>0)&(nd['null_pct']<=20)).sum()),
        '20–60%':     int(((nd['null_pct']>20)&(nd['null_pct']<=60)).sum()),
        '>60% null':  int((nd['null_pct']>60).sum()),
    }
    r['positive_rate']    = tgt['positive_rate']
    r['scale_pos_weight'] = tgt['scale_pos_weight']
    r['top10_numeric']    = num['signal_df'].head(10)[['feature','mean_target0','mean_target1','norm_diff']].copy()

    spreads = [(col, float(d['default_rate'].max() - d['default_rate'].min()))
               for col, d in cat['cat_results'].items()]
    r['top5_cat'] = pd.DataFrame(spreads, columns=['column','spread'])\
                      .sort_values('spread', ascending=False).head(5).reset_index(drop=True)

    r['high_corr_pairs'] = corr['high_corr_df']

    cols_eng = ['feature','corr','outperforms_components']
    r['eng_summary'] = corr['eng_df'][[c for c in cols_eng if c in corr['eng_df'].columns]]

    # Drop recommendations
    drop_recs = [{'column': c, 'reason': '>60% null'} for c in struct['high_null_cols']]
    for _, row in corr['high_corr_df'].iterrows():
        weaker = row['col2'] if row['prefer'] == row['col1'] else row['col1']
        drop_recs.append({'column': weaker,
                          'reason': f"Redundant with {row['prefer']} (corr={row['abs_corr']})"})
    drop_recs.append({'column': 'SK_ID_CURR', 'reason': 'ID column – non-predictive'})
    r['drop_recs'] = (pd.DataFrame(drop_recs)
                      .drop_duplicates('column').reset_index(drop=True))
    return r

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — PIPELINE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
def p3_validation(df_joined, n_app):
    print("  [3.1] Pipeline validation")
    r = {}

    # Row count
    n_joined = len(df_joined)
    row_ok = (n_joined == n_app)
    r['row_check'] = pd.DataFrame([{
        'check': 'Row count',
        'application_train': n_app,
        'joined_df':         n_joined,
        'status':            'PASS' if row_ok else f'FAIL (diff={n_joined-n_app})',
    }])
    if not row_ok:
        print(f"  WARNING: Row count mismatch! app={n_app}, joined={n_joined}")

    # Column inventory
    def _src(c):
        if c.startswith('bureau_bal'):
            return 'bureau_balance'
        if c.startswith('bureau_'):
            return 'bureau'
        if c.startswith('prev_') or c in ('days_since_last_application',
                                           'days_since_last_approval',
                                           'annuity_request_ratio',
                                           'credit_request_ratio'):
            return 'prev_application'
        if c.startswith('inst_'):
            return 'installments'
        if c.startswith('pos_'):
            return 'pos_cash'
        if c.startswith('cc_'):
            return 'credit_card'
        if c.startswith('has_'):
            return 'missingness_flag'
        cross = {'systemic_delinquency_score','internal_dpd_composite',
                 'internal_vs_external_dpd_diff','total_credit_sources',
                 'bureau_bal_any_dpd','pos_any_dpd','cc_any_dpd','inst_any_late'}
        if c in cross:
            return 'cross_table'
        return 'application_train'

    src_map = {c: _src(c) for c in df_joined.columns}
    from collections import Counter
    cnt = Counter(src_map.values())
    r['col_inventory'] = pd.DataFrame(
        [{'source': k, 'n_columns': v} for k, v in sorted(cnt.items())])

    # Missingness consistency
    rep_cols = {
        'has_bureau_record':    'bureau_loan_count',
        'has_bureau_balance':   'bureau_bal_total_months',
        'has_prev_application': 'prev_app_count',
        'has_installments':     'inst_total_count',
        'has_pos_cash':         'pos_months_count',
        'has_credit_card':      'cc_months_count',
    }
    miss_recs = []
    for flag, rep in rep_cols.items():
        flag_zero = float((df_joined[flag] == 0).mean()) if flag in df_joined.columns else np.nan
        null_rate = float(df_joined[rep].isnull().mean()) if rep in df_joined.columns else np.nan
        diff = abs(flag_zero - null_rate) if pd.notna(flag_zero) and pd.notna(null_rate) else np.nan
        status = ('PASS' if pd.notna(diff) and diff <= 0.01
                  else f'FAIL (diff={diff:.4f})' if pd.notna(diff) else 'MISSING')
        miss_recs.append({'has_flag': flag, 'flag=0 rate': round(flag_zero,4),
                          'null_rate': round(null_rate,4), 'diff': round(diff,4) if pd.notna(diff) else 'N/A',
                          'status': status})
    r['miss_check'] = pd.DataFrame(miss_recs)

    # Range validation
    range_specs = {
        'bureau_active_ratio':        (0, 1,    False),
        'bureau_utilization_ratio':   (0, 1.05, False),
        'bureau_bal_dpd_rate':        (0, 1,    False),
        'inst_late_rate':             (0, 1,    False),
        'inst_underpay_rate':         (0, 1,    False),
        'pos_dpd_rate':               (0, 1,    False),
        'cc_dpd_rate':                (0, 1,    False),
        'cc_avg_utilization':         (0, 1.05, False),
        'systemic_delinquency_score': (0, 4,    False),
        'bureau_bal_worst_ever_status':(0, 5,   False),
        'prev_approval_rate':         (0, 1,    False),
        'pos_max_dpd':                (0, None, False),
        'cc_max_dpd':                 (0, None, False),
        'inst_max_days_late':         (-365, 365, True),
    }
    range_recs = []
    for col, (lo, hi, wide_ok) in range_specs.items():
        if col not in df_joined.columns:
            range_recs.append({'column': col, 'min': 'N/A', 'max': 'N/A',
                               'expected': f'[{lo},{hi}]', 'status': 'MISSING'})
            continue
        s = df_joined[col].dropna()
        if len(s) == 0:
            range_recs.append({'column': col, 'min': 'all null', 'max': 'all null',
                               'expected': f'[{lo},{hi}]', 'status': 'PASS (all null)'})
            continue
        mn, mx = float(s.min()), float(s.max())
        ok = True
        if lo is not None and mn < lo:
            ok = False
        if hi is not None and mx > hi:
            ok = False
        if wide_ok and (mn < -365 or mx > 365):
            ok = False
        range_recs.append({'column': col,
                           'min': round(mn,4), 'max': round(mx,4),
                           'expected': f'>={lo}' + (f', <={hi}' if hi else ''),
                           'status': 'PASS' if ok else 'FAIL'})
    r['range_check'] = pd.DataFrame(range_recs)
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — SUPPLEMENTARY FEATURE SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════
def p3_feature_signal(df_joined):
    print("  [3.2] Supplementary feature signal (computing correlations)…")
    r = {}

    def _src(c):
        if c.startswith('bureau_bal'):
            return 'bureau_balance'
        if c.startswith('bureau_'):
            return 'bureau'
        if c.startswith('prev_'):
            return 'prev_application'
        if c.startswith('inst_'):
            return 'installments'
        if c.startswith('pos_'):
            return 'pos_cash'
        if c.startswith('cc_'):
            return 'credit_card'
        if c.startswith('has_'):
            return 'missingness'
        cross = {'systemic_delinquency_score','internal_dpd_composite',
                 'internal_vs_external_dpd_diff','total_credit_sources',
                 'bureau_bal_any_dpd','pos_any_dpd','cc_any_dpd','inst_any_late',
                 'credit_request_ratio','annuity_request_ratio',
                 'days_since_last_application','days_since_last_approval'}
        if c in cross:
            return 'cross_table'
        return 'application_train'

    num_cols = [c for c in df_joined.select_dtypes(include=[np.number]).columns
                if c not in ('SK_ID_CURR',)]
    tc = df_joined[num_cols].corrwith(df_joined['TARGET']).abs().sort_values(ascending=False)
    r['full_corr_df'] = pd.DataFrame({
        'rank':   range(1, len(tc)+1),
        'feature': tc.index,
        'abs_corr': tc.round(4).values,
        'source':  [_src(c) for c in tc.index],
    })

    # Key feature ranks
    key_feats = ['inst_late_rate','bureau_bal_worst_ever_status',
                 'systemic_delinquency_score','internal_vs_external_dpd_diff']
    rank_recs = []
    for feat in key_feats:
        mask = r['full_corr_df']['feature'] == feat
        if mask.any():
            row = r['full_corr_df'][mask].iloc[0]
            rank_recs.append({'feature': feat, 'rank': int(row['rank']),
                              'abs_corr': row['abs_corr']})
        else:
            rank_recs.append({'feature': feat, 'rank': 'N/A', 'abs_corr': 'N/A'})
    r['key_feature_ranks'] = pd.DataFrame(rank_recs)

    # Top 10 supplementary only
    supp_mask = r['full_corr_df']['source'] != 'application_train'
    r['top10_supp'] = r['full_corr_df'][supp_mask].head(10).copy()

    # Cross-table feature validation
    cross_meta = {
        'systemic_delinquency_score':    ['bureau_bal_total_dpd','pos_dpd_month_count',
                                          'cc_dpd_month_count','inst_late_count'],
        'internal_vs_external_dpd_diff': ['internal_dpd_composite','bureau_bal_dpd_rate'],
        'pos_dpd_trend':                 ['pos_avg_dpd'],
        'cc_dpd_trend':                  ['cc_avg_dpd'],
        'cc_utilization_trend':          ['cc_avg_utilization'],
        'credit_request_ratio':          ['AMT_CREDIT','prev_avg_credit'],
        'annuity_request_ratio':         ['AMT_ANNUITY','prev_avg_annuity'],
        'inst_std_days_late':            ['inst_avg_days_late'],
    }
    tc_dict = r['full_corr_df'].set_index('feature')['abs_corr'].to_dict()
    cross_recs = []
    for feat, comps in cross_meta.items():
        fc = tc_dict.get(feat, 'N/A')
        comp_vals = [(c, tc_dict.get(c, 'N/A')) for c in comps if c in df_joined.columns]
        max_comp = max((v for _, v in comp_vals if isinstance(v, float)), default=np.nan)
        rec = {'cross_feature': feat, 'corr': fc}
        for c, v in comp_vals:
            rec[f'comp_{c}'] = v
        rec['recommendation'] = ('KEEP' if isinstance(fc, float) and
                                  isinstance(max_comp, float) and fc >= max_comp else 'DROP candidate')
        cross_recs.append(rec)
    r['cross_val_df'] = pd.DataFrame(cross_recs)
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — MISSINGNESS STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
def p3_missingness(df_joined):
    print("  [3.3] Missingness structure")
    r = {}
    has_cols = [c for c in df_joined.columns if c.startswith('has_')]

    # Default rate by has_* flag
    miss_recs = []
    for col in has_cols:
        r0 = df_joined[df_joined[col]==0]['TARGET'].mean() if (df_joined[col]==0).any() else np.nan
        r1 = df_joined[df_joined[col]==1]['TARGET'].mean() if (df_joined[col]==1).any() else np.nan
        diff = abs(float(r0)-float(r1)) if pd.notna(r0) and pd.notna(r1) else np.nan
        miss_recs.append({'flag': col,
                          'default_rate_0': round(float(r0),4) if pd.notna(r0) else 'N/A',
                          'default_rate_1': round(float(r1),4) if pd.notna(r1) else 'N/A',
                          'abs_diff':       round(diff,4) if pd.notna(diff) else 'N/A',
                          'informative':    'YES' if pd.notna(diff) and diff > 0.02 else 'no'})
    r['has_flag_df'] = (pd.DataFrame(miss_recs)
                        .sort_values('abs_diff', ascending=False, key=lambda x: pd.to_numeric(x, errors='coerce'))
                        .reset_index(drop=True))

    # no_history_count
    df_joined['_no_hist'] = (1 - df_joined[has_cols]).sum(axis=1)
    nhc = df_joined.groupby('_no_hist')['TARGET'].agg(['count','mean']).reset_index()
    nhc.columns = ['no_history_count','n_applicants','default_rate']
    nhc['default_rate'] = nhc['default_rate'].round(4)
    r['no_history_df'] = nhc
    r['no_history_corr'] = round(float(df_joined['_no_hist'].corr(df_joined['TARGET'])), 4)
    df_joined.drop(columns=['_no_hist'], inplace=True)

    # Missingness heatmap
    hm_data = pd.DataFrame({
        col: [float((df_joined[df_joined['TARGET']==t][col]==0).mean()) for t in [0,1]]
        for col in has_cols
    }, index=['TARGET=0', 'TARGET=1'])
    fig, ax = plt.subplots(figsize=(max(6, len(has_cols)*1.2), 3))
    sns.heatmap(hm_data, cmap='Blues', ax=ax, annot=True, fmt='.3f',
                linewidths=0.5, cbar=True, vmin=0, vmax=1)
    ax.set_title('Missingness Heatmap — Fraction with has_*=0 by Target Class',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('has_* flag', fontsize=10)
    ax.set_ylabel('TARGET', fontsize=10)
    plt.tight_layout()
    r['miss_heatmap'] = save_fig(fig, 'p3_miss_heatmap')
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — CORRELATION STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
def p3_correlations(df_joined):
    print("  [3.4] Correlation structure (full matrix — may be slow)…")
    r = {}

    def _src(c):
        if c.startswith('bureau_bal'):
            return 'bureau_balance'
        if c.startswith('bureau_'):
            return 'bureau'
        if c.startswith('prev_'):
            return 'prev_application'
        if c.startswith('inst_'):
            return 'installments'
        if c.startswith('pos_'):
            return 'pos_cash'
        if c.startswith('cc_'):
            return 'credit_card'
        if c.startswith('has_'):
            return 'missingness'
        return 'application_train'

    num_cols = [c for c in df_joined.select_dtypes(include=[np.number]).columns
                if c != 'SK_ID_CURR']
    tc = df_joined[num_cols].corrwith(df_joined['TARGET']).abs()
    corr_mat = df_joined[num_cols].corr()
    r['corr_matrix'] = corr_mat

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(corr_mat, cmap='coolwarm', center=0, ax=ax,
                xticklabels=False, yticklabels=False, cbar=True)
    ax.set_title('Pairwise Correlation Matrix — Full Joined Dataset (Phase 3)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    r['corr_heatmap'] = save_fig(fig, 'p3_corr_heatmap')

    # High-corr pairs >0.85
    cols_list = corr_mat.columns.tolist()
    high_corr = []
    for i in range(len(cols_list)):
        for j in range(i+1, len(cols_list)):
            c1, c2 = cols_list[i], cols_list[j]
            val = abs(float(corr_mat.loc[c1, c2]))
            if val > 0.85:
                t1 = float(tc.get(c1, 0))
                t2 = float(tc.get(c2, 0))
                high_corr.append({
                    'col1': c1, 'source1': _src(c1),
                    'col2': c2, 'source2': _src(c2),
                    'abs_corr': round(val,4),
                    'prefer': c1 if t1 >= t2 else c2,
                })
    r['high_corr_df'] = (pd.DataFrame(high_corr)
                         .sort_values('abs_corr', ascending=False)
                         .reset_index(drop=True))

    # Cross-table delinquency matrix (9 cols)
    delinq_cols = [c for c in [
        'bureau_bal_dpd_rate','bureau_bal_worst_ever_status',
        'inst_late_rate','inst_avg_days_late',
        'pos_dpd_rate','pos_max_dpd',
        'cc_dpd_rate','cc_max_dpd',
        'systemic_delinquency_score',
    ] if c in df_joined.columns]
    r['delinq_corr'] = df_joined[delinq_cols].corr().round(4)

    # Heatmap for delinquency matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(r['delinq_corr'], cmap='coolwarm', center=0, annot=True, fmt='.2f',
                linewidths=0.5, ax=ax, cbar=True)
    ax.set_title('Cross-Table Delinquency Feature Correlation Matrix',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    r['delinq_heatmap'] = save_fig(fig, 'p3_delinq_heatmap')

    # Within-table redundancy (pairs with corr > 0.90)
    table_prefixes = {
        'bureau':          [c for c in cols_list if c.startswith('bureau_') and not c.startswith('bureau_bal')],
        'bureau_balance':  [c for c in cols_list if c.startswith('bureau_bal')],
        'prev_application':[c for c in cols_list if c.startswith('prev_')],
        'installments':    [c for c in cols_list if c.startswith('inst_')],
        'pos_cash':        [c for c in cols_list if c.startswith('pos_')],
        'credit_card':     [c for c in cols_list if c.startswith('cc_')],
    }
    within_recs = []
    for tbl, tcols in table_prefixes.items():
        if len(tcols) < 2:
            continue
        sub = corr_mat.loc[tcols, tcols]
        for i in range(len(tcols)):
            for j in range(i+1, len(tcols)):
                v = abs(float(sub.iloc[i, j]))
                if v > 0.90:
                    within_recs.append({'table': tbl,
                                        'col1': tcols[i], 'col2': tcols[j],
                                        'abs_corr': round(v, 4)})
    r['within_redund'] = (pd.DataFrame(within_recs)
                          .sort_values('abs_corr', ascending=False)
                          .reset_index(drop=True))
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
def p3_summary(valid, signal, miss, corr):
    print("  [3.5] Phase 3 summary")
    r = {}
    r['row_check']      = valid['row_check']
    r['col_inventory']  = valid['col_inventory']
    r['miss_check']     = valid['miss_check']
    r['range_check']    = valid['range_check']
    r['top10_supp']     = signal['top10_supp']
    r['cross_val_df']   = signal['cross_val_df']
    r['has_flag_df']    = miss['has_flag_df']
    r['no_history_df']  = miss['no_history_df']
    r['high_corr_pairs']= corr['high_corr_df']
    r['within_redund']  = corr['within_redund']

    # Final drop list
    drop_recs = []
    # Cross-table features that don't outperform components
    for _, row in signal['cross_val_df'].iterrows():
        if 'DROP' in str(row.get('recommendation','')):
            drop_recs.append({'column': row['cross_feature'],
                              'reason': 'Cross-table feature does not outperform components'})
    # Redundant columns from high-corr pairs
    for _, row in corr['high_corr_df'].iterrows():
        weaker = row['col2'] if row['prefer'] == row['col1'] else row['col1']
        drop_recs.append({'column': weaker,
                          'reason': f"Redundant with {row['prefer']} (corr={row['abs_corr']})"})
    # Range failures
    fails = valid['range_check'][valid['range_check']['status'].astype(str).str.startswith('FAIL')]
    for _, row in fails.iterrows():
        drop_recs.append({'column': row['column'],
                          'reason': f"Range violation: min={row['min']}, max={row['max']}"})
    r['drop_recs'] = (pd.DataFrame(drop_recs)
                      .drop_duplicates('column').reset_index(drop=True)
                      if drop_recs else pd.DataFrame(columns=['column','reason']))
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# PDF REPORT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
def build_pdf(res):
    story = []

    # ── COVER PAGE ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 6*cm))
    story.append(Paragraph(
        'Home Credit Default Risk', ParagraphStyle(
            name='CoverTitle', fontSize=28, fontName='Helvetica-Bold',
            alignment=TA_CENTER, spaceAfter=12)))
    story.append(Paragraph(
        'Exploratory Data Analysis Report',
        ParagraphStyle(name='CoverSub', fontSize=18, fontName='Helvetica',
                       alignment=TA_CENTER, spaceAfter=8, textColor=colors.darkgrey)))
    story.append(Paragraph(
        'Phase 1 (application_train) &amp; Phase 3 (Joined Dataset)',
        ParagraphStyle(name='CoverSub2', fontSize=13, fontName='Helvetica-Oblique',
                       alignment=TA_CENTER, spaceAfter=20, textColor=colors.grey)))
    story.append(Paragraph(
        f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}',
        ParagraphStyle(name='CoverDate', fontSize=11, fontName='Helvetica',
                       alignment=TA_CENTER, textColor=colors.grey)))
    story.append(PageBreak())

    # ── SECTION 1 — DATA STRUCTURE ────────────────────────────────────────────
    st  = res['p1_struct']
    _sec(story, 'Section 1 — Data Structure Summary (Phase 1)')
    _txt(story, f'Dataset shape: <b>{st["shape"][0]:,} rows × {st["shape"][1]} columns</b>. '
                f'Duplicate SK_ID_CURR: <b>{st["n_dupes"]}</b>.')

    _sub(story, '1.1 Column Counts by dtype')
    _tbl(story, st['dtype_counts'], col_widths=[CONTENT_W*0.6, CONTENT_W*0.4])

    _sub(story, '1.2 Pseudo-Categorical Numeric Columns (values only in {0,1,2})')
    pseudo_df = pd.DataFrame({'column': st['pseudo_cat']})
    _tbl(story, pseudo_df) if st['pseudo_cat'] else _txt(story, 'None identified.')

    _sub(story, '1.3 Null Rate by Column (all columns, sorted descending)')
    _tbl(story, st['null_df'],
         caption='Columns with >60% nulls are drop candidates. Null rates computed after replacing 365243/XNA/XAP.',
         col_widths=[CONTENT_W*0.45, CONTENT_W*0.18, CONTENT_W*0.15, CONTENT_W*0.22])

    _sub(story, '1.4 Numeric Column Min / Max')
    _tbl(story, st['minmax_df'],
         caption='Flag column marks implausible values (e.g. negative amounts, age outside expected range).',
         col_widths=[CONTENT_W*0.45, CONTENT_W*0.2, CONTENT_W*0.2, CONTENT_W*0.15])

    _sub(story, '1.5 Categorical Unique Value Counts')
    _tbl(story, st['cat_unique_df'],
         caption='Columns with >20 unique values flagged as high cardinality — require special encoding.',
         col_widths=[CONTENT_W*0.5, CONTENT_W*0.25, CONTENT_W*0.25])
    story.append(PageBreak())

    # ── SECTION 2 — TARGET VARIABLE ───────────────────────────────────────────
    tgt = res['p1_tgt']
    _sec(story, 'Section 2 — Target Variable (Phase 1)')
    _tbl(story, tgt['target_df'],
         col_widths=[CONTENT_W*0.4, CONTENT_W*0.3, CONTENT_W*0.3])
    _txt(story, f'Positive rate (default rate): <b>{tgt["positive_rate"]:.4f}</b> '
                f'({tgt["positive_rate"]*100:.2f}%)')
    _txt(story, f'scale_pos_weight for XGBoost: <b>{tgt["scale_pos_weight"]:.2f}</b> '
                f'(= count(0) / count(1))')
    _img(story, tgt['target_plot'],
         caption='Class distribution. ~92% applicants repay (TARGET=0), ~8% default (TARGET=1). '
                 'Strong class imbalance — use scale_pos_weight or stratified sampling.')
    story.append(PageBreak())

    # ── SECTION 3 — NUMERIC FEATURE SIGNAL ───────────────────────────────────
    num = res['p1_num']
    _sec(story, 'Section 3 — Numeric Feature Signal (Phase 1)')

    _sub(story, '3.1 Top 30 Features by Normalised Mean Difference Between Target Classes')
    _txt(story, 'Normalised difference = |mean(TARGET=1) − mean(TARGET=0)| / |overall mean|. '
                'Higher values indicate stronger separation.')
    w3 = [CONTENT_W*0.35, CONTENT_W*0.22, CONTENT_W*0.22, CONTENT_W*0.21]
    _tbl(story, num['top30'], col_widths=w3,
         caption='Top 30 numeric features by normalised class mean difference.')

    _sub(story, '3.2 KDE Plots — Top 20 Features by Target Class')
    _img(story, num['kde_grid_plot'],
         caption='Kernel density plots for the top 20 numeric features. Blue = TARGET=0 (repays), '
                 'orange = TARGET=1 (defaults). Dashed vertical lines show class means. '
                 'Wide separation indicates high predictive value.')

    _sub(story, '3.3 Highly Skewed Features (|skewness| > 2)')
    _tbl(story, num['skew_df'],
         caption='Features with absolute skewness > 2 are candidates for log transformation before modeling.',
         col_widths=[CONTENT_W*0.32, CONTENT_W*0.12, CONTENT_W*0.14, CONTENT_W*0.14, CONTENT_W*0.14, CONTENT_W*0.14])

    _sub(story, '3.4 EXT_SOURCE Columns — Priority Signal Features')
    _tbl(story, res['p1_num']['ext_source_df'],
         col_widths=[CONTENT_W*0.3, CONTENT_W*0.18, CONTENT_W*0.26, CONTENT_W*0.26])
    if res['p1_num']['ext_source_corr'] is not None and len(res['p1_num']['ext_source_corr']) > 0:
        ext_corr_display = res['p1_num']['ext_source_corr'].reset_index()
        ext_corr_display.rename(columns={'index': 'feature'}, inplace=True)
        _tbl(story, ext_corr_display,
             caption='Pairwise correlation among EXT_SOURCE columns.')
    if 'ext_kde_plot' in num:
        _img(story, num['ext_kde_plot'],
             caption='EXT_SOURCE KDE plots by target class. These external credit scores are typically '
                     'among the strongest predictors of default.')

    _sub(story, '3.5 FLAG Columns — Default Rate Analysis')
    _tbl(story, num['flag_df'],
         caption='FLAG columns sorted by absolute difference in default rate between FLAG=0 and FLAG=1. '
                 'High-ranked flags identify applicant characteristics associated with elevated default risk.',
         col_widths=[CONTENT_W*0.45, CONTENT_W*0.18, CONTENT_W*0.18, CONTENT_W*0.19])
    story.append(PageBreak())

    # ── SECTION 4 — CATEGORICAL FEATURE SIGNAL ───────────────────────────────
    _sec(story, 'Section 4 — Categorical Feature Signal (Phase 1)')
    cat = res['p1_cat']
    for col, df_cat in cat['cat_results'].items():
        _sub(story, f'4.x  {col}  ({len(df_cat)} unique values)')
        show = df_cat[['category','count','default_rate','anomaly']].copy()
        _tbl(story, show,
             caption=f'{col}: sorted by default rate descending. '
                     '"HIGH >50%" flags categories with default rate >50% above base rate (0.08). '
                     '"LOW <50%" flags those >50% below.',
             col_widths=[CONTENT_W*0.4, CONTENT_W*0.18, CONTENT_W*0.18, CONTENT_W*0.24])
    story.append(PageBreak())

    # ── SECTION 5 — FEATURE CORRELATIONS ─────────────────────────────────────
    co = res['p1_corr']
    _sec(story, 'Section 5 — Feature Correlations (Phase 1)')

    _sub(story, '5.1 Full Pairwise Correlation Heatmap')
    _img(story, co['corr_heatmap'],
         caption='Full pairwise Pearson correlation matrix for all numeric features. '
                 'Red = positive correlation, blue = negative. Blocks of strong correlation '
                 'indicate potential multicollinearity.')

    _sub(story, '5.2 High-Correlation Pairs (|corr| > 0.85)')
    if len(co['high_corr_df']) > 0:
        _tbl(story, co['high_corr_df'],
             caption='Column pairs with absolute Pearson correlation > 0.85. '
                     '"prefer" column indicates which has higher absolute correlation with TARGET — '
                     'this is the preferred column to retain if one must be dropped.',
             col_widths=[CONTENT_W*0.32, CONTENT_W*0.32, CONTENT_W*0.14, CONTENT_W*0.22])
    else:
        _txt(story, 'No pairs found with |correlation| > 0.85.')

    _sub(story, '5.3 Engineered Feature Candidates')
    _tbl(story, co['eng_df'][['feature','corr','outperforms_components']],
         caption='For each engineered feature: Pearson correlation with TARGET and whether it '
                 'outperforms all component features. Flagged as KEEP when correlation exceeds '
                 'all components.',
         col_widths=[CONTENT_W*0.45, CONTENT_W*0.22, CONTENT_W*0.33])
    story.append(PageBreak())

    # ── SECTION 6 — PHASE 1 SUMMARY ──────────────────────────────────────────
    s1 = res['p1_summ']
    _sec(story, 'Section 6 — Phase 1 Output Summary')

    _sub(story, '6.1 Dataset Shape')
    _txt(story, f'{s1["shape"][0]:,} rows × {s1["shape"][1]} columns.')

    _sub(story, '6.2 Column Count by Null Rate Bucket')
    nb_df = pd.DataFrame(list(s1['null_buckets'].items()), columns=['null_bucket','n_columns'])
    _tbl(story, nb_df, col_widths=[CONTENT_W*0.5, CONTENT_W*0.5])

    _sub(story, '6.3 Positive Rate and scale_pos_weight')
    _txt(story, f'Positive rate: <b>{s1["positive_rate"]:.4f}</b>')
    _txt(story, f'scale_pos_weight: <b>{s1["scale_pos_weight"]:.2f}</b>')

    _sub(story, '6.4 Top 10 Numeric Features by Class Separation')
    _tbl(story, s1['top10_numeric'],
         col_widths=[CONTENT_W*0.4, CONTENT_W*0.2, CONTENT_W*0.2, CONTENT_W*0.2])

    _sub(story, '6.5 Top 5 Categorical Features by Default Rate Spread')
    _tbl(story, s1['top5_cat'], col_widths=[CONTENT_W*0.6, CONTENT_W*0.4])

    _sub(story, '6.6 All High-Correlation Pairs (|corr| > 0.85)')
    if len(s1['high_corr_pairs']) > 0:
        _tbl(story, s1['high_corr_pairs'],
             col_widths=[CONTENT_W*0.32, CONTENT_W*0.32, CONTENT_W*0.14, CONTENT_W*0.22])
    else:
        _txt(story, 'None found.')

    _sub(story, '6.7 Engineered Feature Correlation vs Components')
    _tbl(story, s1['eng_summary'],
         col_widths=[CONTENT_W*0.45, CONTENT_W*0.22, CONTENT_W*0.33])

    _sub(story, '6.8 Recommended Columns for Dropping (Phase 1)')
    _tbl(story, s1['drop_recs'],
         caption='Based on >60% nulls, high multicollinearity with a stronger predictor, '
                 'or non-predictive metadata.',
         col_widths=[CONTENT_W*0.35, CONTENT_W*0.65])
    story.append(PageBreak())

    # ── SECTION 7 — PIPELINE VALIDATION ──────────────────────────────────────
    v = res['p3_valid']
    _sec(story, 'Section 7 — Pipeline Validation (Phase 3)')

    _sub(story, '7.1 Row Count Check')
    _tbl(story, v['row_check'])

    _sub(story, '7.2 Column Inventory by Source Table')
    _tbl(story, v['col_inventory'],
         caption='Column count per source table group after all joins.',
         col_widths=[CONTENT_W*0.6, CONTENT_W*0.4])

    _sub(story, '7.3 Missingness Consistency Check')
    _txt(story, 'For each table: fraction of applicants with has_*=0 should equal null rate '
                'of representative column within 1 percentage point.')
    _tbl(story, v['miss_check'],
         caption='PASS = missingness flag and null rate agree within 1pp. '
                 'FAIL indicates a pipeline join error.',
         col_widths=[CONTENT_W*0.3, CONTENT_W*0.17, CONTENT_W*0.17, CONTENT_W*0.12, CONTENT_W*0.24])

    _sub(story, '7.4 Range Validation')
    _tbl(story, v['range_check'],
         caption='All columns must fall within expected ranges. FAIL indicates an aggregation error.',
         col_widths=[CONTENT_W*0.35, CONTENT_W*0.14, CONTENT_W*0.14, CONTENT_W*0.22, CONTENT_W*0.15])
    story.append(PageBreak())

    # ── SECTION 8 — SUPPLEMENTARY FEATURE SIGNAL ──────────────────────────────
    sg = res['p3_signal']
    _sec(story, 'Section 8 — Supplementary Feature Signal (Phase 3)')

    _sub(story, '8.1 Top 50 Features by |Correlation with TARGET| — Full Joined Dataset')
    top50 = sg['full_corr_df'].head(50)
    _tbl(story, top50,
         caption='Top 50 features by absolute Pearson correlation with TARGET across all columns '
                 'in the joined dataset. Source table annotation shows which data source each '
                 'feature originates from.',
         col_widths=[CONTENT_W*0.1, CONTENT_W*0.38, CONTENT_W*0.17, CONTENT_W*0.35])

    _sub(story, '8.2 Top 10 Supplementary Features (non-application_train)')
    _tbl(story, sg['top10_supp'],
         caption='Top 10 features from supplementary tables only — demonstrates the predictive '
                 'value added by feature engineering over the base application data.',
         col_widths=[CONTENT_W*0.1, CONTENT_W*0.38, CONTENT_W*0.17, CONTENT_W*0.35])

    _sub(story, '8.3 Key Feature Rank Positions')
    _tbl(story, sg['key_feature_ranks'],
         caption='Rank positions of the highest-signal cross-table features in the full feature '
                 'ranking. Low rank numbers confirm the aggregation strategy produced valuable features.',
         col_widths=[CONTENT_W*0.55, CONTENT_W*0.22, CONTENT_W*0.23])

    _sub(story, '8.4 Cross-Table Feature Validation')
    cross_display = sg['cross_val_df'][['cross_feature','corr','recommendation']].copy()
    _tbl(story, cross_display,
         caption='Each cross-table feature compared to its component features. '
                 'KEEP = feature adds value over components. DROP candidate = components are stronger.',
         col_widths=[CONTENT_W*0.45, CONTENT_W*0.2, CONTENT_W*0.35])
    story.append(PageBreak())

    # ── SECTION 9 — MISSINGNESS STRUCTURE ────────────────────────────────────
    ms = res['p3_miss']
    _sec(story, 'Section 9 — Missingness Structure (Phase 3)')

    _sub(story, '9.1 Default Rate by has_* Flag Value')
    _tbl(story, ms['has_flag_df'],
         caption='Default rate when applicant has no record (flag=0) vs has a record (flag=1). '
                 '"informative=YES" means absence/presence predicts default by >2pp — retain as a feature.',
         col_widths=[CONTENT_W*0.35, CONTENT_W*0.18, CONTENT_W*0.18, CONTENT_W*0.14, CONTENT_W*0.15])

    _sub(story, '9.2 no_history_count Distribution and Default Rate')
    _txt(story, f'Correlation of no_history_count with TARGET: '
                f'<b>{ms["no_history_corr"]}</b>')
    _tbl(story, ms['no_history_df'],
         caption='Distribution of no_history_count (number of supplementary tables with no record). '
                 'no_history_count=max means no history in any supplementary table — first-time borrowers.',
         col_widths=[CONTENT_W*0.3, CONTENT_W*0.35, CONTENT_W*0.35])

    _sub(story, '9.3 Missingness Heatmap')
    _img(story, ms['miss_heatmap'],
         caption='Fraction of applicants with has_*=0 (no record in that table) split by target class. '
                 'Differences between TARGET=0 and TARGET=1 rows confirm missingness is informative.')
    story.append(PageBreak())

    # ── SECTION 10 — CORRELATION STRUCTURE ───────────────────────────────────
    cr = res['p3_corr']
    _sec(story, 'Section 10 — Correlation Structure (Phase 3)')

    _sub(story, '10.1 Full Pairwise Correlation Heatmap — Joined Dataset')
    _img(story, cr['corr_heatmap'],
         caption='Pairwise Pearson correlation heatmap for all numeric columns in the joined '
                 'dataset. The matrix is large (~300+ columns) — use the pair table below for '
                 'identifying specific redundancies.')

    _sub(story, '10.2 High-Correlation Pairs (|corr| > 0.85) — Joined Dataset')
    if len(cr['high_corr_df']) > 0:
        _tbl(story, cr['high_corr_df'], max_rows=60,
             caption='Column pairs with |correlation| > 0.85 in the joined dataset. '
                     '"prefer" indicates the column with higher absolute correlation with TARGET — '
                     'the one to retain if one must be dropped. Showing up to 60 pairs.',
             col_widths=[CONTENT_W*0.25, CONTENT_W*0.14, CONTENT_W*0.25, CONTENT_W*0.14, CONTENT_W*0.12, CONTENT_W*0.1])
    else:
        _txt(story, 'No pairs found with |correlation| > 0.85.')

    _sub(story, '10.3 Cross-Table Delinquency Feature Correlation Matrix')
    _img(story, cr['delinq_heatmap'],
         caption='Pairwise correlation among the 9 key delinquency features from different source '
                 'tables. High correlation (>0.85) means two features capture the same underlying '
                 'signal — the weaker predictor is a drop candidate.')
    delinq_display = cr['delinq_corr'].reset_index()
    delinq_display.rename(columns={'index': 'feature'}, inplace=True)
    n_dc = len(delinq_display.columns)
    _tbl(story, delinq_display,
         caption='Numeric cross-table delinquency correlation matrix.',
         col_widths=[CONTENT_W * 0.22] + [CONTENT_W * 0.78 / (n_dc - 1)] * (n_dc - 1))

    _sub(story, '10.4 Within-Table Redundancy (pairs with |corr| > 0.90)')
    if len(cr['within_redund']) > 0:
        _tbl(story, cr['within_redund'],
             caption='Within-table column pairs with absolute correlation > 0.90. '
                     'These are near-redundant and one column from each pair can be dropped.',
             col_widths=[CONTENT_W*0.25, CONTENT_W*0.3, CONTENT_W*0.3, CONTENT_W*0.15])
    else:
        _txt(story, 'No within-table pairs found with |correlation| > 0.90.')
    story.append(PageBreak())

    # ── SECTION 11 — PHASE 3 SUMMARY ─────────────────────────────────────────
    s3 = res['p3_summ']
    _sec(story, 'Section 11 — Phase 3 Output Summary')

    _sub(story, '11.1 Row Count Confirmation')
    _tbl(story, s3['row_check'])

    _sub(story, '11.2 Column Count by Source Table')
    _tbl(story, s3['col_inventory'],
         col_widths=[CONTENT_W*0.6, CONTENT_W*0.4])

    _sub(story, '11.3 Missingness Consistency Check Results')
    _tbl(story, s3['miss_check'],
         col_widths=[CONTENT_W*0.3, CONTENT_W*0.17, CONTENT_W*0.17, CONTENT_W*0.12, CONTENT_W*0.24])

    _sub(story, '11.4 Range Validation Results')
    _tbl(story, s3['range_check'],
         col_widths=[CONTENT_W*0.35, CONTENT_W*0.14, CONTENT_W*0.14, CONTENT_W*0.22, CONTENT_W*0.15])

    _sub(story, '11.5 Top 10 Supplementary Features')
    _tbl(story, s3['top10_supp'],
         col_widths=[CONTENT_W*0.1, CONTENT_W*0.38, CONTENT_W*0.17, CONTENT_W*0.35])

    _sub(story, '11.6 Cross-Table Feature Validation (Keep / Drop)')
    _tbl(story, s3['cross_val_df'][['cross_feature','corr','recommendation']],
         col_widths=[CONTENT_W*0.45, CONTENT_W*0.2, CONTENT_W*0.35])

    _sub(story, '11.7 Informative has_* Flags (|Δ default rate| > 0.02)')
    info_flags = s3['has_flag_df'][s3['has_flag_df']['informative']=='YES']
    _tbl(story, info_flags[['flag','default_rate_0','default_rate_1','abs_diff']],
         col_widths=[CONTENT_W*0.4, CONTENT_W*0.2, CONTENT_W*0.2, CONTENT_W*0.2]) if len(info_flags) > 0 \
        else _txt(story, 'None identified.')

    _sub(story, '11.8 no_history_count Distribution and Default Rate')
    _tbl(story, s3['no_history_df'],
         col_widths=[CONTENT_W*0.3, CONTENT_W*0.35, CONTENT_W*0.35])

    _sub(story, '11.9 All Column Pairs with |Corr| > 0.85 (Joined Dataset, top 40)')
    if len(s3['high_corr_pairs']) > 0:
        _tbl(story, s3['high_corr_pairs'], max_rows=40,
             col_widths=[CONTENT_W*0.25, CONTENT_W*0.14, CONTENT_W*0.25, CONTENT_W*0.14, CONTENT_W*0.12, CONTENT_W*0.1])
    else:
        _txt(story, 'No pairs found with |correlation| > 0.85.')

    _sub(story, '11.10 Final Drop List — Columns Recommended for Removal')
    if len(s3['drop_recs']) > 0:
        _tbl(story, s3['drop_recs'],
             caption='Columns recommended for removal before preprocessing and modeling, '
                     'with reason for each recommendation.',
             col_widths=[CONTENT_W*0.35, CONTENT_W*0.65])
    else:
        _txt(story, 'No additional columns flagged for removal beyond Phase 1 recommendations.')

    # ── BUILD DOCUMENT ────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=2.5*cm,
        title='Home Credit EDA Report',
    )
    doc.build(story, onFirstPage=_page_number, onLaterPages=_page_number)

    page_count = doc.page
    print(f"\n  PDF saved: {OUTPUT_PDF.resolve()}")
    print(f"  Total pages: {page_count}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print('=' * 60)
    print('Home Credit Default Risk — EDA Pipeline')
    print('=' * 60)
    gen_start = datetime.datetime.now()

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    print('\n[Phase 1] Loading application_train.csv…')
    df = pd.read_csv(DATA_DIR / 'application_train.csv')
    df = preprocess(df)
    n_app = len(df)
    print(f'  Loaded: {df.shape}')

    struct  = p1_structure(df)
    target  = p1_target(df)
    num_sig = p1_numeric_signal(df)
    cat_sig = p1_categorical_signal(df)
    corr    = p1_correlations(df)
    summ1   = p1_summary(struct, target, num_sig, cat_sig, corr)
    print('[Phase 1] Complete.')

    # ── Aggregation ───────────────────────────────────────────────────────────
    print('\n[Aggregation] Building joined dataset…')
    df_joined = aggregate_all(df)
    del df
    gc.collect()
    print('[Aggregation] Complete.')

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    print('\n[Phase 3] EDA on joined dataset…')
    valid   = p3_validation(df_joined, n_app)
    signal  = p3_feature_signal(df_joined)
    miss    = p3_missingness(df_joined)
    corr3   = p3_correlations(df_joined)
    summ3   = p3_summary(valid, signal, miss, corr3)
    print('[Phase 3] Complete.')

    # ── PDF ───────────────────────────────────────────────────────────────────
    print('\n[PDF] Building report…')
    results = {
        'p1_struct': struct,
        'p1_tgt':    target,
        'p1_num':    num_sig,
        'p1_cat':    cat_sig,
        'p1_corr':   corr,
        'p1_summ':   summ1,
        'p3_valid':  valid,
        'p3_signal': signal,
        'p3_miss':   miss,
        'p3_corr':   corr3,
        'p3_summ':   summ3,
    }
    build_pdf(results)

    elapsed = datetime.datetime.now() - gen_start
    print(f'\n{"=" * 60}')
    print(f'Pipeline complete in {elapsed}.')
    print(f'Report: {OUTPUT_PDF.resolve()}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
