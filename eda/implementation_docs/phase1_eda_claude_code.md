# Phase 1 EDA — application_train.csv

This document defines the EDA to perform on `application_train.csv` before any joins to supplementary tables. It is intended as an implementation guide for Claude Code. Each section specifies what to compute, what column names or values to look for, and what the output or finding should be used for downstream.

Do not join any supplementary tables during this phase. Work only with `application_train.csv`.

---

## Pre-checks before any analysis

Before running any EDA, apply the following corrections to the dataframe. These are known data quality issues specific to this dataset:

- Replace `365243` with `NaN` in all `DAYS_*` columns. This value is used as a placeholder for unemployed or inapplicable date fields and is not a real number.
- Replace the string `'XNA'` and `'XAP'` with `NaN` in all categorical (object) columns.
- Take the absolute value of all `DAYS_*` columns after replacing the placeholder. These columns are stored as negative integers representing days before the application date. Larger absolute values mean further in the past.

Apply these corrections once at the top of the notebook before any analysis begins. Do not apply them mid-analysis.

---

## 1. Data structure and quality

The goal of this section is to understand what the data physically looks like before touching the target or any features. All outputs here should be printed or displayed, not just computed silently.

### Shape
Print the number of rows and columns. Expected: approximately 307,000 rows and 122 columns.

### Data types
Print a count of columns by dtype — how many are numeric (int64, float64) and how many are categorical (object). Then print the full list of column names grouped by dtype so it is easy to see which columns fall into each group.

Flag any numeric columns that appear to be categorical in disguise. The primary candidates in this dataset are columns that contain only the values 0, 1, and possibly 2 — these may represent encoded categories rather than true numeric quantities. Print a list of any such columns.

### Null values
Compute the count and percentage of null values per column, after applying the pre-checks above. Sort descending by percentage. Display the full list, not just the top N.

Separately flag:
- Columns with more than 60% nulls — these are candidates for dropping entirely and should be noted explicitly.
- Columns with zero nulls — these are clean and require no imputation.
- Columns with between 1% and 60% nulls — these will need imputation and possibly a missingness indicator flag.

### Anomalous placeholder values
Even after replacing `365243` and `XNA`, scan for other anomalous values:
- For numeric columns: print the min and max of every numeric column. Flag any columns where the min or max looks implausible given the column name (e.g. negative values in an amount column, zero values in an age column). The `DAYS_BIRTH` column converted to years should range from roughly 20 to 70 — values outside this range are anomalous.
- For categorical columns: print the full value counts for every categorical column. Flag any values that appear to be placeholders or codes for unknown (e.g. strings like `'Unknown'`, `'not specified'`, numeric strings used as categories).

### Duplicate rows
Check whether `SK_ID_CURR` is unique. Print the number of duplicate `SK_ID_CURR` values if any exist. There should be zero — each row represents one loan application. If duplicates exist, print the duplicate rows for inspection.

### Column naming patterns
Print the column names grouped by prefix. The naming convention in this dataset is:
- `FLAG_` — binary indicator columns
- `AMT_` — monetary amount columns
- `DAYS_` — time delta columns (days before application date)
- `CNT_` — count columns
- `EXT_SOURCE_` — external credit score columns (1, 2, and 3)

This grouping is for orientation only and does not require any transformation.

---

## 2. Target variable

The goal of this section is to understand class balance and establish the base rate. This informs the evaluation strategy and the `scale_pos_weight` parameter for XGBoost.

### Class distribution
Print the count and percentage of each class:
- `0` — applicant repays the loan
- `1` — applicant defaults

Expected: approximately 92% class 0, 8% class 1.

### Base rate
Print the exact positive rate (fraction of class 1). Store this value as a variable called `positive_rate` for use later when setting `scale_pos_weight`. The formula for `scale_pos_weight` in XGBoost is `(count of class 0) / (count of class 1)` — compute and print this value as well.

### Implication for all subsequent analysis
Every distribution, mean comparison, and correlation in sections 3 and 4 must be computed split by `TARGET` class, not on the full dataset collapsed together. With only 8% positives, signal from defaulters is otherwise washed out. This is not optional — do not produce any feature-level analysis that does not disaggregate by target.

---

## 3. Feature signal analysis

The goal of this section is to identify which features are predictive of the target and understand their distributions. Work through numeric and categorical features separately.

### Numeric features

#### Mean by target
For every numeric column (excluding `SK_ID_CURR` and `TARGET`), compute the mean grouped by `TARGET`. Compute the absolute difference between the class 0 mean and class 1 mean, normalized by the overall mean of the column (this is a simple effect size). Sort descending by this normalized difference and print the top 30. This is a fast first-pass ranking of which numeric features show the most separation between classes.

#### Distribution plots split by target
For the top 20 numeric features by the ranking above, produce overlaid KDE plots with one line per target class. Label each plot with the column name and the mean for each class. The goal is to visually confirm that the separation seen in the mean comparison is real and not driven by outliers.

Do not produce histograms of the full dataset — always split by target.

#### Skewness and outliers
For every numeric column, compute skewness. Print a list of columns with absolute skewness greater than 2 — these are candidates for log transformation before modeling. For the most skewed columns (top 10 by absolute skewness), print the min, max, mean, and the 99th percentile to understand the outlier structure.

#### EXT_SOURCE columns
Treat `EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3` as a priority group. For each:
- Print the null rate.
- Print the mean by target class.
- Produce a KDE plot split by target.

These three columns are typically among the most predictive features in this dataset. Also compute the pairwise correlations among the three EXT_SOURCE columns and print the correlation matrix.

#### FLAG columns
For every column with a `FLAG_` prefix, compute the default rate (mean of `TARGET`) for `FLAG=1` and `FLAG=0` separately. Print the results sorted by the difference in default rate between the two groups. This identifies which flags are associated with elevated or reduced default risk.

### Categorical features

#### Default rate by category
For every categorical column (object dtype), compute the mean of `TARGET` grouped by category value. Print the result for each column sorted by default rate descending. The overall base rate is approximately 0.08 — flag any category with a default rate more than 50% above or below the base rate as noteworthy.

#### Category counts
For every categorical column, print the value counts including nulls. Flag any category that appears in fewer than 100 rows — these rare categories may need to be grouped into an `'Other'` bucket before encoding.

#### High cardinality
Print the number of unique values for every categorical column. Flag any column with more than 20 unique values as high cardinality. These will need special treatment during encoding.

---

## 4. Feature relationships

The goal of this section is to identify correlations among features — both signal correlations (with the target) and redundancy correlations (between features).

### Correlation with target
Compute the Pearson correlation of every numeric column with `TARGET`. Print the full list sorted by absolute correlation descending. This is a linear-only measure — a low correlation does not mean a feature is useless, but a high correlation is a reliable signal of predictive value.

### Pairwise correlation matrix
Compute the full pairwise Pearson correlation matrix for all numeric columns. Display it as a heatmap. Do not annotate individual cells — the matrix is too large. Use a diverging colormap centered at zero.

From the correlation matrix, print a list of all column pairs with absolute correlation greater than 0.85. These are multicollinearity candidates. For each flagged pair, note which column has the higher absolute correlation with `TARGET` — that is the one to prefer if one must be dropped.

### Multicollinearity clusters to note
The following column groups are known to be closely related in this dataset. Verify by checking the correlation matrix and print the pairwise correlations within each group:
- `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE` — credit amount, monthly payment, and financed goods price
- `FLAG_DOCUMENT_*` columns — document submission flags, many of which are correlated with each other
- `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` — already covered above but also relevant here

### Engineered feature candidates
Based on the correlation and domain structure, the following derived features are worth creating during this phase to test their correlation with `TARGET`. Compute each one, then print its correlation with `TARGET` alongside the correlations of its component columns for comparison:

- `CREDIT_INCOME_RATIO` — `AMT_CREDIT / AMT_INCOME_TOTAL` (debt-to-income)
- `ANNUITY_INCOME_RATIO` — `AMT_ANNUITY / AMT_INCOME_TOTAL` (payment burden)
- `CREDIT_GOODS_RATIO` — `AMT_CREDIT / AMT_GOODS_PRICE` (loan-to-value)
- `AGE_YEARS` — `DAYS_BIRTH / 365` (age in years, after taking absolute value)
- `EMPLOYED_YEARS` — `DAYS_EMPLOYED / 365` (years employed, after taking absolute value; note that `DAYS_EMPLOYED` has already had `365243` replaced with `NaN` in the pre-checks)
- `EMPLOYMENT_RATIO` — `DAYS_EMPLOYED / DAYS_BIRTH` (fraction of life employed)
- `EXT_SOURCE_MEAN` — mean of `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` across non-null values per row
- `EXT_SOURCE_MIN` — min of the three EXT_SOURCE columns per row

If the engineered feature has a higher absolute correlation with `TARGET` than any of its components individually, flag it as a confirmed useful feature to carry forward to the modeling phase.

---

## Output summary

At the end of the notebook, print a consolidated summary containing:

1. Shape of the dataframe.
2. Count of columns by null rate bucket: 0% null, 1–20% null, 20–60% null, 60%+ null.
3. The positive rate and the computed `scale_pos_weight` value.
4. Top 10 numeric features by normalized mean difference between classes.
5. Top 5 categorical features by default rate spread across categories.
6. All column pairs with pairwise correlation above 0.85.
7. The correlation with TARGET for each engineered feature candidate versus its components.
8. A list of columns recommended for dropping, with the reason for each:
   - Columns with more than 60% nulls
   - Columns from high-correlation pairs where the weaker predictor is redundant
   - Any columns identified as IDs or non-predictive metadata (e.g. `SK_ID_CURR`)

This summary will be used to inform feature selection and preprocessing decisions before the aggregation and join phase.
