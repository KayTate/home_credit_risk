# Phase 3 EDA — Joined Table (Implementation Guide)

This document defines the EDA to perform on the fully joined dataset after all supplementary tables have been aggregated and merged to `application_train`. It is intended as an implementation guide for Claude Code. Read it alongside the Phase 1 EDA guide and the aggregation strategy document.

The joined dataframe at this point should contain all columns from `application_train`, all aggregated columns from the supplementary tables, all cross-table features, and all `has_*` missingness indicator flags. No imputation should have been applied yet — nulls should still be present in the aggregated columns. Phase 3 EDA happens before imputation.

Column names referenced throughout this document match those defined in the aggregation strategy document. If a column name does not exist in the dataframe, print a warning and skip that check rather than raising an error.

---

## 1. Pipeline validation

The goal of this section is to verify that the join and aggregation pipeline produced correct output before any analysis is run. Print the result of every check explicitly. If any check fails, print a clear warning message identifying what went wrong.

### Row count check
Print the shape of the joined dataframe. The number of rows must exactly match the number of rows in `application_train` (approximately 307,000). If the row count is higher, the join created duplicates. If it is lower, the join dropped rows. Either condition is a bug — do not proceed past this check until it passes.

### Column inventory
Print the total number of columns. Then print columns grouped by source table using these prefixes as identifiers:
- Columns from `application_train` — everything not matching the prefixes below
- Bureau columns — prefix `bureau_`
- Bureau balance columns — prefix `bureau_bal_`
- Previous application columns — prefix `prev_`
- Installments columns — prefix `inst_`
- POS cash columns — prefix `pos_`
- Credit card columns — prefix `cc_`
- Cross-table features — `systemic_delinquency_score`, `internal_dpd_composite`, `internal_vs_external_dpd_diff`, `pos_dpd_trend`, `cc_dpd_trend`, `cc_utilization_trend`, `credit_request_ratio`, `annuity_request_ratio`, `inst_std_days_late`, `inst_std_payment_diff`, `total_credit_sources`, `days_since_last_application`, `days_since_last_approval`, `bureau_recent_inquiry_count`
- Missingness flags — prefix `has_`

Print the count of columns in each group.

### Missingness consistency check
For each `has_*` flag, the null rate of the corresponding aggregated columns must match the flag. Specifically:

For each supplementary table, compute the fraction of rows where `has_*` equals 0. Then compute the null rate of one representative aggregated column from that table. These two numbers should be equal or very close (within 1 percentage point). Print both numbers side by side for each table. Flag any table where the two numbers diverge by more than 1 percentage point as a pipeline bug.

Use these representative columns for the check:
- `has_bureau_record` vs null rate of `bureau_loan_count`
- `has_bureau_balance` vs null rate of `bureau_bal_total_months`
- `has_prev_application` vs null rate of `prev_app_count`
- `has_installments` vs null rate of `inst_total_count`
- `has_pos_cash` vs null rate of `pos_months_count`
- `has_credit_card` vs null rate of `cc_months_count`

### Range validation
For the columns listed below, check that all non-null values fall within the expected range. Print any column where values fall outside the range, along with the min and max of the violating values:

- `bureau_active_ratio` — must be between 0 and 1
- `bureau_utilization_ratio` — must be between 0 and 1 (values slightly above 1 are acceptable due to rounding)
- `bureau_bal_dpd_rate` — must be between 0 and 1
- `inst_late_rate` — must be between 0 and 1
- `inst_underpay_rate` — must be between 0 and 1
- `pos_dpd_rate` — must be between 0 and 1
- `cc_dpd_rate` — must be between 0 and 1
- `cc_avg_utilization` — must be between 0 and 1 (values slightly above 1 are acceptable)
- `systemic_delinquency_score` — must be between 0 and 4
- `bureau_bal_worst_ever_status` — must be between 0 and 5
- `prev_approval_rate` — must be between 0 and 1
- `pos_max_dpd` — must be non-negative
- `cc_max_dpd` — must be non-negative
- `inst_max_days_late` — can be negative (early payment) but flag any values more extreme than -365 or greater than 365 as suspicious

---

## 2. Signal analysis — supplementary features versus main table features

The goal of this section is to measure how much predictive value the supplementary table features add over the features already present in `application_train`.

### Full feature ranking by correlation with target
Compute the Pearson correlation of every numeric column with `TARGET`. Sort by absolute correlation descending and print the full ranked list. Annotate each column with its source table group using the prefixes defined in section 1.

From this list, separately print:
- Top 20 features overall
- Top 10 features sourced from supplementary tables (excluding `application_train` columns)
- Rank position of `inst_late_rate`, `bureau_bal_worst_ever_status`, `systemic_delinquency_score`, and `internal_vs_external_dpd_diff` specifically — these are the highest-signal cross-table features and their rank positions indicate whether the aggregation strategy worked

### Mean-by-target for supplementary features
For all aggregated columns from supplementary tables, compute the mean grouped by `TARGET`. Compute the normalized absolute difference (same method as Phase 1). Print the top 20 supplementary features by this measure. This should largely agree with the correlation ranking — if it does not, investigate the discrepancy.

### Cross-table feature validation
For each cross-table feature listed below, print its correlation with `TARGET` alongside the correlations of the component features it was derived from. Flag any cross-table feature whose correlation with `TARGET` is lower than all of its components — this means the combination did not add value and the feature is a candidate for dropping.

- `systemic_delinquency_score` — compare against `bureau_bal_total_dpd`, `pos_dpd_month_count`, `cc_dpd_month_count`, `inst_late_count`
- `internal_vs_external_dpd_diff` — compare against `internal_dpd_composite` and `bureau_bal_dpd_rate`
- `pos_dpd_trend` — compare against `pos_avg_dpd`
- `cc_dpd_trend` — compare against `cc_avg_dpd`
- `cc_utilization_trend` — compare against `cc_avg_utilization`
- `credit_request_ratio` — compare against `AMT_CREDIT` and `prev_avg_credit`
- `annuity_request_ratio` — compare against `AMT_ANNUITY` and `prev_avg_annuity`
- `inst_std_days_late` — compare against `inst_avg_days_late`
- `EXT_SOURCE_MEAN` (from Phase 1 engineering) — compare against `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`

---

## 3. Missingness structure analysis

The goal of this section is to determine whether the absence of a record in a supplementary table is itself predictive, and whether missingness clusters in informative ways.

### Default rate by has_* flag
For each `has_*` flag column, compute the mean of `TARGET` for `has_* = 0` and `has_* = 1` separately. Print the results as a table with columns: flag name, default rate when 0, default rate when 1, absolute difference. Sort by absolute difference descending.

Flag any `has_*` column where the absolute difference in default rate exceeds 0.02 (2 percentage points) as a confirmed informative missingness flag — it should be retained as a feature in the model regardless of whether the aggregated columns from that table are retained.

### Missingness clustering
Create a binary missingness matrix where each column is a `has_*` flag and each row is an applicant. Compute the sum of `has_* = 0` values per row — this is a count of how many supplementary tables the applicant has no record in. Call this `no_history_count`.

Print the distribution of `no_history_count` as a value counts table. Compute and print the mean of `TARGET` grouped by `no_history_count`. Applicants with `no_history_count` equal to the maximum (no record in any supplementary table) are a distinct segment — first-time borrowers with no financial history anywhere. Print how many such applicants exist and their default rate.

Create `no_history_count` as a feature in the dataframe and print its correlation with `TARGET`.

### Missingness heatmap
Produce a heatmap where rows are `has_*` flag columns and columns are `TARGET` values (0 and 1). Each cell shows the fraction of applicants in that target class who have `has_* = 0` for that flag. This reveals whether missingness patterns differ between defaulters and non-defaulters. Use a sequential colormap.

---

## 4. Correlation structure of the full feature set

The goal of this section is to identify redundancy in the full joined feature set before it goes into the model.

### Full pairwise correlation matrix
Compute the full pairwise Pearson correlation matrix for all numeric columns in the joined dataframe. Display it as a heatmap without cell annotations. Use a diverging colormap centered at zero. The matrix will be large — this is expected.

Print all column pairs with absolute correlation greater than 0.85, sorted by absolute correlation descending. Annotate each pair with the source table of each column. For each pair, note which column has the higher absolute correlation with `TARGET`.

### Cross-table redundancy check
Specifically examine correlations between delinquency features across different source tables. Print the pairwise correlation matrix for the following columns only:

- `bureau_bal_dpd_rate`
- `bureau_bal_worst_ever_status`
- `inst_late_rate`
- `inst_avg_days_late`
- `pos_dpd_rate`
- `pos_max_dpd`
- `cc_dpd_rate`
- `cc_max_dpd`
- `systemic_delinquency_score`

Pairs with correlation above 0.85 in this group are measuring the same underlying signal from different sources. Flag them and note the higher-TARGET-correlated column as the preferred one to keep.

### Within-table redundancy check
For each supplementary table, print the pairwise correlations among aggregated columns from that table only. Flag any within-table pair with absolute correlation above 0.90 — these are almost certainly redundant and one should be dropped. Use these column groups:

- Bureau columns: all columns with prefix `bureau_` excluding `bureau_bal_*`
- Bureau balance columns: all columns with prefix `bureau_bal_`
- Previous application columns: all columns with prefix `prev_`
- Installments columns: all columns with prefix `inst_`
- POS cash columns: all columns with prefix `pos_`
- Credit card columns: all columns with prefix `cc_`

### EXT_SOURCE versus behavioral feature correlation
Print the correlations between `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`, and `EXT_SOURCE_MEAN` against the top behavioral features from supplementary tables: `inst_late_rate`, `bureau_bal_worst_ever_status`, `bureau_bal_dpd_rate`, `systemic_delinquency_score`. If any EXT_SOURCE column correlates above 0.70 with a behavioral feature, note it — this means the external credit scores and internal behavioral data are capturing overlapping signal.

---

## Output summary

At the end of the notebook, print a consolidated summary containing:

1. Row count confirmation — pass or fail with the actual count.
2. Column count by source table group.
3. Missingness consistency check results — pass or fail per table.
4. Range validation results — pass or fail per column, with details on any failures.
5. Top 10 supplementary features by correlation with `TARGET`.
6. Cross-table feature validation results — for each cross-table feature, whether it outperforms its components (keep) or does not (drop candidate).
7. `has_*` flags confirmed as informative (absolute default rate difference > 0.02).
8. `no_history_count` distribution and default rate by value.
9. All column pairs with pairwise correlation above 0.85, annotated with preferred column to keep.
10. Final drop list — columns recommended for removal before preprocessing, with reason for each:
    - Cross-table features that did not outperform their components
    - Redundant columns from high-correlation pairs (keeping the higher-TARGET-correlated column)
    - Any columns with more than 80% nulls that were not already dropped in Phase 1
    - Any range validation failures that indicate a broken aggregation

This summary, combined with the Phase 1 output summary, defines the full feature set that goes into the preprocessing and modeling pipeline.
