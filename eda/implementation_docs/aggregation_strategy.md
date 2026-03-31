# Aggregation Strategy — Home Credit Supplementary Tables

This document defines the feature engineering and aggregation strategy for joining the Home Credit supplementary tables to `application_train`. It is intended as an implementation guide and should be read alongside the Phase 1 EDA plan (application_train) and the Phase 3 EDA plan (joined table).

The join key for all tables is `SK_ID_CURR`. All joins back to the main table should be `left` joins — not every applicant appears in every supplementary table, and that missingness is itself informative. After joining, add binary indicator flags for applicants who have no record in a given table (e.g. `has_bureau_record`, `has_prev_application`) rather than leaving the absence implicit in nulls.

---

## Pre-processing steps that apply to all tables

Before any aggregation, apply these corrections universally:

- Replace `365243` with `NaN` in all `DAYS_*` columns across all tables. This value is used as a placeholder for unemployed or inapplicable date fields and will corrupt any aggregation that treats it as a real number.
- Replace the string `'XNA'` and `'XAP'` with `NaN` in all categorical columns across all tables.
- All `DAYS_*` columns are stored as negative integers (days before the application date). Take the absolute value for interpretability before aggregating, so that larger values mean further in the past.

---

## bureau.csv

### Join path
Direct join to `application_train` on `SK_ID_CURR`.

### Domain context
This table represents the applicant's credit history with external financial institutions as reported to the credit bureau. A credit analyst uses this to assess the applicant's external financial reputation — how many credit obligations they have held, whether those obligations are still active or resolved, and whether there is any evidence of overdue debt.

### Columns to aggregate and how

**`SK_ID_BUREAU` (count):**
Total number of bureau loans. Baseline measure of how credit-active this person is. Store as `bureau_loan_count`.

**`CREDIT_ACTIVE` (value counts by status):**
Break down by status: Active, Closed, Bad debt, Sold. Do not collapse to a single count. Create separate columns for each:
- `bureau_active_count` — count of active loans
- `bureau_closed_count` — count of successfully closed loans
- `bureau_bad_debt_count` — count of loans in bad debt status (strong negative signal)
- `bureau_active_ratio` — active count divided by total loan count

The ratio is more meaningful than the raw active count. An applicant with 10 closed loans and 1 active is in a very different position than one with 8 active loans and 2 closed. High active ratio suggests the applicant is currently overleveraged.

**`AMT_CREDIT_MAX_OVERDUE` (max):**
Use max, not mean. A single large overdue balance is a serious red flag even if most of the history is clean. Store as `bureau_max_overdue`.

**`AMT_CREDIT_SUM_DEBT` (sum and mean):**
Sum gives total outstanding external debt. Mean gives average debt per loan. Both are useful. Store as `bureau_total_debt` and `bureau_avg_debt`.

**`AMT_CREDIT_SUM` (sum):**
Total credit extended across all bureau loans. Store as `bureau_total_credit`.

**Derived: debt-to-credit ratio:**
`bureau_total_debt / bureau_total_credit`. This is the applicant's external utilization rate — the fraction of their total available bureau credit that is currently owed. High utilization is a classic default predictor. Store as `bureau_utilization_ratio`. Clip the denominator at a small positive number to avoid division by zero.

**`DAYS_CREDIT` (mean):**
Average number of days before the application that bureau credit relationships started. Larger values mean longer credit history. Credit age is a known positive signal — longer history means more data and generally more stability. Store as `bureau_avg_credit_age`.

**`CNT_CREDIT_PROLONG` (sum):**
Number of times any bureau loan was extended or restructured. Loan extensions often indicate the borrower could not make payments on schedule. Even one or two extensions is worth flagging. Store as `bureau_total_prolonged`. Also create a binary flag `bureau_any_prolonged`.

**`CREDIT_TYPE` (value counts for key types):**
Create binary flags for whether the applicant has ever held specific high-signal credit types:
- `bureau_has_mortgage` — mortgage presence is a proxy for financial stability since mortgage underwriting is rigorous
- `bureau_has_microfinance` — microfinance is often a last-resort borrowing product and is a negative signal
- `bureau_credit_type_count` — count of distinct credit types held, as a credit diversity measure

---

## bureau_balance.csv

### Join path
Two-hop join: `bureau_balance` → `bureau` → `application_train`. Aggregate first to `SK_ID_BUREAU` level, then join to bureau to get `SK_ID_CURR`, then re-aggregate to applicant level.

### Domain context
Monthly payment history for each bureau loan. This is the most granular external behavioral data available. The `STATUS` column encodes payment timeliness: `C` = closed, `0` = no days past due, `X` = unknown/inapplicable, `1` = 1–30 DPD, `2` = 31–60 DPD, `3` = 61–90 DPD, `4` = 91–120 DPD, `5` = 120+ DPD. Higher numeric statuses represent increasingly serious delinquencies.

### Step 1: Aggregate to SK_ID_BUREAU level

**`MONTHS_BALANCE` (count):**
Total months of history observed for this bureau loan. Store as `bb_months_count`.

**`STATUS` — DPD month count:**
Count of months where STATUS is in `['1','2','3','4','5']`. These are all delinquent months regardless of severity. Store as `bb_dpd_count`.

**`STATUS` — worst status:**
Maximum numeric STATUS value, excluding `'C'` and `'X'`. This is the single worst delinquency event ever recorded for this loan. Store as `bb_worst_status`. A value of 5 means the loan went 120+ days past due at some point — a severe signal.

**`STATUS` — good month count:**
Count of months where STATUS is `'0'` (current, no DPD). This is a positive behavioral signal representing reliable payment history. Store as `bb_good_months`.

### Step 2: Join to bureau to get SK_ID_CURR, then aggregate to applicant level

**`bb_months_count` (sum):**
Total months of bureau payment history across all bureau loans. Store as `bureau_bal_total_months`.

**`bb_dpd_count` (sum and mean):**
Sum = total delinquent months across all bureau loans. Mean = average delinquent months per loan. Store as `bureau_bal_total_dpd` and `bureau_bal_avg_dpd_per_loan`.

**`bb_worst_status` (max):**
The single worst STATUS value ever recorded across the applicant's entire bureau history. This is the most important feature from this table — it answers "what is the worst thing this person has ever done on an external credit obligation?" Store as `bureau_bal_worst_ever_status`.

**`bb_good_months` (sum):**
Total months of on-time payment behavior. Store as `bureau_bal_total_good_months`.

**Derived: DPD rate:**
`bureau_bal_total_dpd / bureau_bal_total_months`. The fraction of all observed bureau months that were delinquent. Store as `bureau_bal_dpd_rate`.

---

## previous_application.csv

### Join path
Direct join to `application_train` on `SK_ID_CURR`.

### Domain context
All prior applications to Home Credit by applicants in the current sample. This is internal behavioral data, which is generally more reliable than bureau data because Home Credit collected it directly. A credit analyst uses this to understand the applicant's relationship history with Home Credit specifically — whether prior applications were approved or rejected, and the terms and nature of prior loans.

### Columns to aggregate and how

**`SK_ID_PREV` (count):**
Total number of prior applications to Home Credit. Store as `prev_app_count`.

**`NAME_CONTRACT_STATUS` (value counts):**
Create counts for each status — Approved, Refused, Canceled, Unused offer:
- `prev_approved_count`
- `prev_refused_count`
- `prev_canceled_count`

**Derived: approval rate:**
`prev_approved_count / prev_app_count`. If Home Credit's own underwriting has previously rejected this applicant multiple times, that is a meaningful internal risk signal. Store as `prev_approval_rate`.

**`CODE_REJECT_REASON` (mode of non-null values):**
The most common reason for prior refusals. Encode the mode. Refusals for high credit risk (`HC`) are qualitatively different from administrative refusals. Store as `prev_most_common_reject_reason`.

**`DAYS_DECISION` (min and mean):**
Min = most recent prior decision (smallest absolute value = most recent). Mean = average decision recency. A very recent refusal is more relevant than one from five years ago. Store as `prev_most_recent_decision` and `prev_avg_decision_days`.

**`AMT_CREDIT` (mean and max):**
Mean = typical prior loan size. Max = largest prior approved loan. Comparing these to the current application's requested amount is informative. Store as `prev_avg_credit` and `prev_max_credit`.

**`AMT_ANNUITY` (mean):**
Average monthly payment on prior loans. Another basis for comparing against the current application. Store as `prev_avg_annuity`.

**`AMT_DOWN_PAYMENT` (mean):**
Applicants who consistently make larger down payments are demonstrating both financial capacity and commitment. Store as `prev_avg_down_payment`.

**`NAME_CLIENT_TYPE` (flag):**
Whether the applicant is a returning customer. Create a binary flag `prev_is_returning_customer`. Returning customers with positive history are lower risk than first-time applicants.

**`NAME_YIELD_GROUP` (mode):**
The interest rate tier of prior loans. High-yield prior loans may indicate Home Credit already perceived this borrower as higher risk. Store as `prev_most_common_yield_group`.

---

## installments_payments.csv

### Join path
Join to `application_train` on `SK_ID_CURR` (this column is present directly in the table).

### Domain context
Actual repayment history for prior Home Credit loans. Each row is either a payment made or a missed installment. This table provides direct behavioral evidence of whether the applicant paid their obligations, by how much, and how late. It is the most behaviorally diagnostic internal table.

### Derived columns to create before aggregating

Create these two columns before any groupby:

**`payment_diff`:**
`AMT_INSTALMENT - AMT_PAYMENT`. Positive values mean the applicant paid less than required (underpayment). Negative values mean they overpaid. This captures payment completeness.

**`days_late`:**
`DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT`. Positive values mean the payment was made after the due date (late). Negative values mean early payment. This captures payment timeliness.

### Columns to aggregate and how

**`SK_ID_PREV` (count):**
Total number of installment records — a proxy for how much internal repayment history exists for this applicant. More history means more reliable signal. Store as `inst_total_count`.

**`payment_diff` (mean and max):**
Mean = chronic underpayment pattern. Max = worst single underpayment. Store as `inst_avg_payment_diff` and `inst_max_payment_diff`.

**`payment_diff` — underpayment count:**
Count of rows where `payment_diff > 0`. Store as `inst_underpay_count`.

**Derived: underpayment rate:**
`inst_underpay_count / inst_total_count`. The fraction of installments where the applicant paid less than required. Store as `inst_underpay_rate`.

**`days_late` (mean and max):**
Mean = chronic lateness pattern. Max = worst single late payment. Store as `inst_avg_days_late` and `inst_max_days_late`.

**`days_late` — late payment count:**
Count of rows where `days_late > 0`. Store as `inst_late_count`.

**Derived: late payment rate:**
`inst_late_count / inst_total_count`. The fraction of installments paid late. This is one of the strongest predictors in the entire dataset. Store as `inst_late_rate`.

---

## POS_CASH_balance.csv

### Join path
Join to `application_train` on `SK_ID_CURR`.

### Domain context
Monthly balance snapshots of prior point-of-sale and cash loans at Home Credit. Covers consumer credit and personal loans. The key behavioral signals are days past due per month and how prior contracts resolved.

### Columns to aggregate and how

**`MONTHS_BALANCE` (count):**
Total months of POS/cash history. More history = more reliable signal. Store as `pos_months_count`.

**`SK_DPD` (mean and max):**
Mean = average monthly DPD across all prior POS/cash loans. Max = single worst DPD month ever. Max is the more important of the two. Store as `pos_avg_dpd` and `pos_max_dpd`.

**`SK_DPD_DEF` (mean and max):**
The defined-default DPD measure — a more conservative version that only triggers after a threshold. Treat identically to `SK_DPD`. Store as `pos_avg_dpd_def` and `pos_max_dpd_def`.

**`SK_DPD` — delinquency month count:**
Count of months where `SK_DPD > 0`. Store as `pos_dpd_month_count`.

**Derived: DPD rate:**
`pos_dpd_month_count / pos_months_count`. Store as `pos_dpd_rate`.

**`NAME_CONTRACT_STATUS` (value counts):**
- `pos_completed_count` — count of completed contracts (positive signal)
- `pos_active_count` — count of currently active contracts (current debt load)
- `pos_demand_count` — count of contracts in demand status (negative signal, indicates the lender called the loan)

---

## credit_card_balance.csv

### Join path
Join to `application_train` on `SK_ID_CURR`.

### Domain context
Monthly snapshots of prior credit card accounts at Home Credit. Credit card behavior is particularly diagnostic because revolving credit reveals financial habits more directly than installment loans — the applicant has discretion over how much they borrow and repay each month.

### Derived columns to create before aggregating

**`utilization`:**
`AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL`. Credit utilization per month. Clip the denominator at a small positive number to avoid division by zero.

### Columns to aggregate and how

**`MONTHS_BALANCE` (count):**
Total months of credit card history. Store as `cc_months_count`.

**`utilization` (mean and max):**
Mean = sustained utilization level over time. Max = peak utilization. High sustained utilization means the applicant consistently lives near their credit limit. Store as `cc_avg_utilization` and `cc_max_utilization`.

**`AMT_BALANCE` (mean and max):**
Mean and max balance carried. Store as `cc_avg_balance` and `cc_max_balance`.

**`AMT_DRAWINGS_ATM_CURRENT` (mean and sum):**
Cash advance activity. Frequent ATM drawings from a credit card is a known financial distress signal. Store as `cc_avg_atm_drawings` and `cc_total_atm_drawings`. Also create a binary flag `cc_any_atm_drawings`.

**`AMT_PAYMENT_CURRENT` vs `AMT_INST_MIN_REGULARITY`:**
Create a derived column `paying_above_minimum` = 1 if `AMT_PAYMENT_CURRENT > AMT_INST_MIN_REGULARITY`, else 0. Then aggregate as the mean across months — this gives the fraction of months where the applicant paid more than the minimum. Consistent minimum-only payments suggest the applicant is managing debt without reducing it. Store as `cc_above_minimum_rate`.

**`SK_DPD` (mean and max):**
Store as `cc_avg_dpd` and `cc_max_dpd`.

**`SK_DPD` — delinquency month count:**
Count of months where `SK_DPD > 0`. Store as `cc_dpd_month_count`.

**Derived: DPD rate:**
`cc_dpd_month_count / cc_months_count`. Store as `cc_dpd_rate`.

**`AMT_BALANCE` — active month count:**
Count of months where `AMT_BALANCE > 0`. Someone who rarely carries a balance is a different borrower profile from someone who is always drawn up. Store as `cc_active_month_count`.

---

## Cross-table features

These features are created after all tables have been aggregated and joined to the main table. They capture patterns that span multiple data sources.

### Systemic delinquency score

Create a binary flag from each of the four tables that contain DPD information, indicating whether any delinquency was observed at all:
- `bureau_bal_any_dpd` — 1 if `bureau_bal_total_dpd > 0`
- `pos_any_dpd` — 1 if `pos_dpd_month_count > 0`
- `cc_any_dpd` — 1 if `cc_dpd_month_count > 0`
- `inst_any_late` — 1 if `inst_late_count > 0`

Sum these four flags into a single `systemic_delinquency_score` ranging from 0 to 4. A score of 0 means no delinquency in any data source. A score of 4 means delinquency across every data source — this is a behavioral signature of systemic financial difficulty, qualitatively different from a one-off event.

### Internal versus external delinquency comparison

Compute a composite internal delinquency score by averaging the DPD rates from the three Home Credit internal tables: `pos_dpd_rate`, `cc_dpd_rate`, and `inst_late_rate`. Store as `internal_dpd_composite`.

Compare this to `bureau_bal_dpd_rate` (external delinquency).

Create a difference feature: `internal_dpd_composite - bureau_bal_dpd_rate`. Store as `internal_vs_external_dpd_diff`.

A large positive value means the applicant is worse with Home Credit than with external institutions — they may be deprioritizing Home Credit repayments while maintaining their external credit reputation. A large negative value means the opposite. This asymmetry is only visible by comparing across tables.

### Behavior trend features

For `POS_CASH_balance` and `credit_card_balance`, before aggregating to a single row per applicant, split the monthly history into a recent window (most recent 6 months, i.e. `MONTHS_BALANCE >= -6`) and a historical window (older than 6 months). Compute mean DPD separately for each window.

Create trend features:
- `pos_dpd_trend` — recent mean DPD minus historical mean DPD. Positive = deteriorating. Negative = improving.
- `cc_dpd_trend` — same for credit card DPD.
- `cc_utilization_trend` — recent mean utilization minus historical mean utilization. Rising utilization suggests increasing financial pressure.

Store the trend features alongside the overall aggregated features.

### Loan amount trajectory

After joining `previous_application` aggregations, create:

**`credit_request_ratio`:**
Current application's `AMT_CREDIT` divided by `prev_avg_credit`. Values significantly above 1 mean the applicant is requesting more than they typically have in the past — a potential risk signal. Clip the denominator at a small positive value to handle applicants with no prior applications (who will have a null `prev_avg_credit` — handle via the `has_prev_application` flag).

**`annuity_request_ratio`:**
Current application's `AMT_ANNUITY` divided by `prev_avg_annuity`. Same logic — requesting a much higher monthly payment than historically suggests potential overextension.

### Repayment consistency index

Using `installments_payments`, before aggregating to applicant level, compute the standard deviation of `days_late` and `payment_diff` per applicant in addition to the mean and max. Store as `inst_std_days_late` and `inst_std_payment_diff`.

High standard deviation in payment behavior means the applicant is unpredictable — sometimes good, sometimes bad. Low standard deviation means consistent behavior in either direction. Unpredictability is itself a risk factor because it makes the applicant harder to underwrite reliably.

### Credit history breadth

Combine bureau and previous application data to create a measure of credit diversity:

**`total_credit_sources`:**
Count of distinct tables in which the applicant has any record at all — bureau, previous_application, POS_CASH, credit_card, installments. Ranges from 0 to 5. More sources means more history and generally more creditworthiness signal, though it also means more exposure.

**`bureau_has_mortgage` (from bureau aggregation) used as cross-table context:**
When this flag is present, treat any delinquency in internal tables with slightly different interpretation — a mortgage holder who is delinquent on a Home Credit consumer loan may be prioritizing the mortgage, which is a rational but still risky behavior pattern.

### Time gap features

After joining `previous_application` aggregations, create:

**`days_since_last_application`:**
Current application date (day 0 by convention) minus the most recent prior application date (`prev_most_recent_decision`). A very short gap between the last application and the current one may indicate the applicant is urgently seeking credit from multiple sources — a distress signal.

**`days_since_last_approval`:**
Same concept but using only approved prior applications. A long gap since the last approval means the applicant has not recently had access to credit, which changes the interpretation of the current application.

**`bureau_recent_inquiry_count`:**
From `bureau.csv`, count the number of bureau credit relationships that started in the most recent 12 months (i.e. `DAYS_CREDIT_ENDDATE` is recent or `DAYS_CREDIT` is small). Multiple recent credit inquiries across institutions in a short window is a classic warning sign of financial distress or credit shopping.

---

## Missingness flags to create after all joins

After all supplementary tables have been joined, create the following binary indicator columns. These capture the signal embedded in the absence of a record — for example, having no bureau history is a distinct risk profile from having a clean bureau history.

- `has_bureau_record` — 1 if applicant appears in bureau.csv
- `has_bureau_balance` — 1 if applicant has any bureau_balance history
- `has_prev_application` — 1 if applicant has any previous Home Credit application
- `has_pos_cash` — 1 if applicant has any POS/cash loan history
- `has_credit_card` — 1 if applicant has any credit card history with Home Credit
- `has_installments` — 1 if applicant has any installment payment history

These flags should be created before any imputation. After creating them, impute remaining nulls in the aggregated columns with a sensible default (typically 0 for counts and rates, median for amount columns), since null in an aggregated column after a left join simply means the applicant had no record in that table.
