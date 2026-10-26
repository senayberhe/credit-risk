"""
Raw-to-model preprocessing shared by the PD, LGD/EAD and monitoring notebooks.

Mirrors the steps in 01_load_data_creditrisk_preprocessing_pd.ipynb / eda.ipynb so
new data (e.g. the 2015 loans) can be prepared exactly like the 2007-2014 training data.
"""
import numpy as np
import pandas as pd

from utils import preproc_input

# Reference date used for every "months since" variable.
REF_DATE = pd.Timestamp('2017-12-01')

# Loan statuses treated as default (good_bad = 0).
BAD_STATUSES = [
    'Charged Off', 'Default',
    'Does not meet the credit policy. Status:Charged Off',
    'Late (31-120 days)',
]

# Statuses whose losses are final; LGD and EAD are estimated on these loans only.
WRITTEN_OFF_STATUSES = [
    'Charged Off',
    'Does not meet the credit policy. Status:Charged Off',
]

DUMMY_COLUMNS = ['grade', 'sub_grade', 'home_ownership', 'verification_status',
                 'loan_status', 'purpose', 'addr_state', 'initial_list_status']

# ── PD model: dummies kept in the final model (credit_risk_model.ipynb) ──────
PD_FEATURES = [
    'grade:A', 'grade:B', 'grade:C', 'grade:D', 'grade:E', 'grade:F', 'grade:G',
    'home_ownership:RENT_OTHER_NONE_ANY', 'home_ownership:OWN', 'home_ownership:MORTGAGE',
    'addr_state:ND_NE_IA_NV_FL_HI_AL', 'addr_state:NM_VA', 'addr_state:NY',
    'addr_state:OK_TN_MO_LA_MD_NC', 'addr_state:CA', 'addr_state:UT_KY_AZ_NJ',
    'addr_state:AR_MI_PA_OH_MN', 'addr_state:RI_MA_DE_SD_IN', 'addr_state:GA_WA_OR',
    'addr_state:WI_MT', 'addr_state:TX', 'addr_state:IL_CT',
    'addr_state:KS_SC_CO_VT_AK_MS', 'addr_state:WV_NH_WY_DC_ME_ID',
    'verification_status:Not Verified', 'verification_status:Source Verified',
    'verification_status:Verified',
    'purpose:educ__sm_b__wedd__ren_en__mov__house', 'purpose:credit_card',
    'purpose:debt_consolidation', 'purpose:oth__med__vacation',
    'purpose:major_purch__car__home_impr',
    'initial_list_status:f', 'initial_list_status:w',
    'term:36', 'term:60',
    'emp_length:0', 'emp_length:1', 'emp_length:2-4', 'emp_length:5-6',
    'emp_length:7-9', 'emp_length:10',
    'mths_since_issue_d:<38', 'mths_since_issue_d:38-39', 'mths_since_issue_d:40-41',
    'mths_since_issue_d:42-48', 'mths_since_issue_d:49-52', 'mths_since_issue_d:53-64',
    'mths_since_issue_d:65-84', 'mths_since_issue_d:>84',
    'int_rate:<9.548', 'int_rate:9.548-12.025', 'int_rate:12.025-15.74',
    'int_rate:15.74-20.281', 'int_rate:>20.281',
    'mths_since_earliest_cr_line:<140', 'mths_since_earliest_cr_line:141-164',
    'mths_since_earliest_cr_line:165-247', 'mths_since_earliest_cr_line:248-270',
    'mths_since_earliest_cr_line:271-352', 'mths_since_earliest_cr_line:>352',
    'inq_last_6mths:0', 'inq_last_6mths:1-2', 'inq_last_6mths:3-6', 'inq_last_6mths:>6',
    'acc_now_delinq:0', 'acc_now_delinq:>=1',
    'annual_inc:<20K', 'annual_inc:20K-30K', 'annual_inc:30K-40K', 'annual_inc:40K-50K',
    'annual_inc:50K-60K', 'annual_inc:60K-70K', 'annual_inc:70K-80K', 'annual_inc:80K-90K',
    'annual_inc:90K-100K', 'annual_inc:100K-120K', 'annual_inc:120K-140K', 'annual_inc:>140K',
    'dti:<=1.4', 'dti:1.4-3.5', 'dti:3.5-7.7', 'dti:7.7-10.5', 'dti:10.5-16.1',
    'dti:16.1-20.3', 'dti:20.3-21.7', 'dti:21.7-22.4', 'dti:22.4-35', 'dti:>35',
    'mths_since_last_delinq:Missing', 'mths_since_last_delinq:0-3',
    'mths_since_last_delinq:4-30', 'mths_since_last_delinq:31-56',
    'mths_since_last_delinq:>=57',
    'mths_since_last_record:Missing', 'mths_since_last_record:0-2',
    'mths_since_last_record:3-20', 'mths_since_last_record:21-31',
    'mths_since_last_record:32-80', 'mths_since_last_record:81-86',
    'mths_since_last_record:>86',
]

PD_REF_CATEGORIES = [
    'grade:G', 'home_ownership:RENT_OTHER_NONE_ANY', 'addr_state:ND_NE_IA_NV_FL_HI_AL',
    'verification_status:Verified', 'purpose:educ__sm_b__wedd__ren_en__mov__house',
    'initial_list_status:f', 'term:60', 'emp_length:0', 'mths_since_issue_d:>84',
    'int_rate:>20.281', 'mths_since_earliest_cr_line:<140', 'inq_last_6mths:>6',
    'acc_now_delinq:0', 'annual_inc:<20K', 'dti:>35', 'mths_since_last_delinq:0-3',
    'mths_since_last_record:0-2',
]


def _months_since(date_strings):
    dates = pd.to_datetime(date_strings, format='%b-%y')
    return ((REF_DATE - dates).dt.days / 30.4375).round()


def general_preprocessing(loan_data):
    """Clean raw LendingClub data: numeric fields, dates, missing values, dummies, target."""
    loan_data = loan_data.copy()

    emp = (loan_data['emp_length'].astype(str)
           .str.replace('+ years', '', regex=False)
           .str.replace('< 1 year', '0', regex=False)
           .str.replace('n/a', '0', regex=False)
           .str.replace(' years', '', regex=False)
           .str.replace(' year', '', regex=False))
    loan_data['emp_length_int'] = pd.to_numeric(emp, errors='coerce').fillna(0).astype(int)
    loan_data['term_int'] = loan_data['term'].astype(str).str.extract(r'(\d+)')[0].astype(int)

    loan_data['mths_since_earliest_cr_line'] = _months_since(loan_data['earliest_cr_line'])
    # Two-digit years before 1969 parse into the future; cap them at the oldest value.
    negative = loan_data['mths_since_earliest_cr_line'] < 0
    loan_data.loc[negative, 'mths_since_earliest_cr_line'] = loan_data['mths_since_earliest_cr_line'].max()
    loan_data['mths_since_issue_d'] = _months_since(loan_data['issue_d'])

    loan_data['total_rev_hi_lim'] = loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'])
    loan_data['annual_inc'] = loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean())
    cols_fill_zero = ['mths_since_earliest_cr_line', 'mths_since_issue_d', 'acc_now_delinq',
                      'total_acc', 'pub_rec', 'open_acc', 'inq_last_6mths', 'delinq_2yrs']
    loan_data[cols_fill_zero] = loan_data[cols_fill_zero].fillna(0)

    dummies = [pd.get_dummies(loan_data[col], prefix=col, prefix_sep=':', dtype=int)
               for col in DUMMY_COLUMNS]
    loan_data = pd.concat([loan_data] + dummies, axis=1)

    loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(BAD_STATUSES), 0, 1)
    return loan_data


def add_pd_dummies(df):
    """Create the coarse-classed dummies the PD model and scorecard use."""
    df = preproc_input(df.copy())
    m = df['mths_since_issue_d']
    df['mths_since_issue_d:<38'] = np.where(m < 38, 1, 0)
    df['mths_since_issue_d:38-39'] = np.where((m >= 38) & (m <= 39), 1, 0)
    df['mths_since_issue_d:40-41'] = np.where((m >= 40) & (m <= 41), 1, 0)
    df['mths_since_issue_d:42-48'] = np.where((m >= 42) & (m <= 48), 1, 0)
    df['mths_since_issue_d:49-52'] = np.where((m >= 49) & (m <= 52), 1, 0)
    df['mths_since_issue_d:53-64'] = np.where((m >= 53) & (m <= 64), 1, 0)
    df['mths_since_issue_d:65-84'] = np.where((m >= 65) & (m <= 84), 1, 0)
    df['mths_since_issue_d:>84'] = np.where(m >= 85, 1, 0)  # FIX: range(85, max) dropped the max value
    return df


def pd_design_matrix(df, with_ref_categories=False):
    """Select PD model inputs in training column order; categories absent from df become 0."""
    X = df.reindex(columns=PD_FEATURES, fill_value=0).astype(float)
    if not with_ref_categories:
        X = X.drop(columns=PD_REF_CATEGORIES)
    return X
