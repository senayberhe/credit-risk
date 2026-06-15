"""
Run this once to trim loan.csv from 145 columns down to 74.

Usage:
    uv run python filter_loan.py

Input:  data/raw/loan.csv        (your original Kaggle download)
Output: data/raw/loan_74col.csv  (74 columns, all rows preserved)
"""

import csv
from pathlib import Path

KEEP_74 = [
    # Loan identifiers & terms
    'id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term',
    'int_rate', 'installment', 'grade', 'sub_grade', 'issue_d',
    # Loan metadata
    'loan_status', 'purpose', 'initial_list_status', 'application_type',
    # Borrower info
    'emp_title', 'emp_length', 'home_ownership', 'annual_inc',
    'verification_status', 'addr_state', 'dti', 'earliest_cr_line',
    # Delinquency & public records
    'delinq_2yrs', 'mths_since_last_delinq', 'mths_since_last_record',
    'mths_since_last_major_derog', 'acc_now_delinq', 'delinq_amnt',
    'pub_rec', 'pub_rec_bankruptcies', 'tax_liens', 'num_accts_ever_120_pd',
    'num_tl_90g_dpd_24m',
    # Inquiries
    'inq_last_6mths', 'inq_last_12m', 'mths_since_recent_inq',
    # Open / active accounts
    'open_acc', 'total_acc', 'open_acc_6m', 'open_act_il',
    'open_il_12m', 'open_il_24m', 'open_rv_12m', 'open_rv_24m',
    'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_tl', 'num_il_tl',
    'acc_open_past_24mths',
    # Balances & utilisation
    'revol_bal', 'revol_util', 'total_rev_hi_lim',
    'tot_cur_bal', 'tot_coll_amt', 'total_bal_il', 'il_util',
    'avg_cur_bal', 'bc_util', 'all_util',
    # Mortgage & credit limits
    'mort_acc', 'tot_hi_cred_lim', 'total_bal_ex_mort',
    'total_bc_limit', 'total_il_high_credit_limit',
    # Recency
    'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
    'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',
    'mths_since_recent_bc', 'mths_since_recent_revol_delinq',
    'mths_since_rcnt_il',
    # Performance ratios & misc
    'pct_tl_nvr_dlq', 'percent_bc_gt_75',
    'collections_12_mths_ex_med',
]

assert len(KEEP_74) == 74, f"Expected 74 cols, got {len(KEEP_74)}"

src = Path(__file__).parent / 'data' / 'raw' / 'loan.csv'
out = Path(__file__).parent / 'data' / 'raw' / 'loan_74col.csv'

print(f"Reading: {src}")
print(f"Writing: {out}")

with open(src, newline='', encoding='utf-8') as fin, \
     open(out, 'w', newline='', encoding='utf-8') as fout:

    reader = csv.DictReader(fin)

    # Warn about any requested columns missing from the file
    missing = [c for c in KEEP_74 if c not in reader.fieldnames]
    if missing:
        print(f"⚠️  Columns not found in source (will be empty): {missing}")

    writer = csv.DictWriter(fout, fieldnames=KEEP_74, extrasaction='ignore')
    writer.writeheader()

    rows = 0
    for row in reader:
        writer.writerow({k: row.get(k, '') for k in KEEP_74})
        rows += 1
        if rows % 100_000 == 0:
            print(f"  {rows:,} rows written...")

print(f"\nDone: {rows:,} rows × 74 columns → {out.name}")
