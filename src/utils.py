import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn import linear_model
import scipy.stats as stat


# ── Functions used in Notebook 1 ─────────────────────────────────────────────

def woe_discrete(df, discrete_variable_name, good_bad_variable_df):
    """
    Calculate WoE and IV of categorical features.
    Args: dataframe, column name string, target dataframe.
    """
    df = pd.concat([df[discrete_variable_name], good_bad_variable_df], axis=1)
    df = pd.concat([
        df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
        df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()
    ], axis=1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    # FIX: clip to avoid log(0)
    df['prop_n_good'] = (df['n_good'] / df['n_good'].sum()).clip(1e-9)
    df['prop_n_bad'] = (df['n_bad'] / df['n_bad'].sum()).clip(1e-9)
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE']).reset_index(drop=True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df


def woe_ordered_continuous(df, discrete_variable_name, good_bad_variable_df):
    """
    Calculate WoE and IV of continuous features (preserves bin order).
    Args: dataframe, column name string, target dataframe.
    """
    df = pd.concat([df[discrete_variable_name], good_bad_variable_df], axis=1)
    df = pd.concat([
        df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
        df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()
    ], axis=1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    # FIX: clip to avoid log(0)
    df['prop_n_good'] = (df['n_good'] / df['n_good'].sum()).clip(1e-9)
    df['prop_n_bad'] = (df['n_bad'] / df['n_bad'].sum()).clip(1e-9)
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    # Note: order preserved for continuous variables (no sort)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df


def plot_by_woe(df_WoE, rotation_x_axis_labels=0):
    """Plot Weight of Evidence across categories."""
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    y = df_WoE['WoE']
    plt.figure(figsize=(12, 4))
    plt.plot(x, y, marker='o', linestyle='--', color='k')
    plt.xlabel(df_WoE.columns[0])
    plt.ylabel('Weight of Evidence')
    plt.title('Weight of Evidence by ' + df_WoE.columns[0])
    plt.xticks(rotation=rotation_x_axis_labels)


def preproc_target(loan_data):
    """Create good_bad target: 0 = bad/default, 1 = good."""
    bad_def = [
        'Charged Off', 'Default',
        'Does not meet the credit policy. Status: Charged Off.',
        'Late (31-120 days)'
    ]
    loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(bad_def), 0, 1)
    return loan_data


def _safe_sum(df, cols):
    """Sum columns, creating missing ones as zeros (avoids KeyError for rare states)."""
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return sum(df[c] for c in cols)


def preproc_input(df_inputs_prepr):
    """Preprocess input features — WoE coarse classing for the 2007-2014 dataset."""

    # 1 home_ownership
    df_inputs_prepr['home_ownership:RENT_OTHER_NONE_ANY'] = _safe_sum(df_inputs_prepr, [
        'home_ownership:RENT', 'home_ownership:OTHER',
        'home_ownership:NONE', 'home_ownership:ANY'
    ])

    # 2 addr_state: ensure ND column exists
    if 'addr_state:ND' not in df_inputs_prepr.columns:  # FIX: was checking list-in-array
        df_inputs_prepr['addr_state:ND'] = 0

    # 3 addr_state grouped bins
    df_inputs_prepr['addr_state:ND_NE_IA_NV_FL_HI_AL'] = _safe_sum(df_inputs_prepr, [
        'addr_state:ND', 'addr_state:NE', 'addr_state:IA',
        'addr_state:NV', 'addr_state:FL', 'addr_state:HI', 'addr_state:AL'
    ])
    df_inputs_prepr['addr_state:NM_VA'] = _safe_sum(df_inputs_prepr, ['addr_state:NM', 'addr_state:VA'])
    df_inputs_prepr['addr_state:OK_TN_MO_LA_MD_NC'] = _safe_sum(df_inputs_prepr, [
        'addr_state:OK', 'addr_state:TN', 'addr_state:MO',
        'addr_state:LA', 'addr_state:MD', 'addr_state:NC'
    ])
    df_inputs_prepr['addr_state:UT_KY_AZ_NJ'] = _safe_sum(df_inputs_prepr, [
        'addr_state:UT', 'addr_state:KY', 'addr_state:AZ', 'addr_state:NJ'
    ])
    df_inputs_prepr['addr_state:AR_MI_PA_OH_MN'] = _safe_sum(df_inputs_prepr, [
        'addr_state:AR', 'addr_state:MI', 'addr_state:PA',
        'addr_state:OH', 'addr_state:MN'
    ])
    df_inputs_prepr['addr_state:RI_MA_DE_SD_IN'] = _safe_sum(df_inputs_prepr, [
        'addr_state:RI', 'addr_state:MA', 'addr_state:DE',
        'addr_state:SD', 'addr_state:IN'
    ])
    df_inputs_prepr['addr_state:GA_WA_OR'] = _safe_sum(df_inputs_prepr, [
        'addr_state:GA', 'addr_state:WA', 'addr_state:OR'
    ])
    df_inputs_prepr['addr_state:WI_MT'] = _safe_sum(df_inputs_prepr, ['addr_state:WI', 'addr_state:MT'])
    df_inputs_prepr['addr_state:IL_CT'] = _safe_sum(df_inputs_prepr, ['addr_state:IL', 'addr_state:CT'])
    df_inputs_prepr['addr_state:KS_SC_CO_VT_AK_MS'] = _safe_sum(df_inputs_prepr, [
        'addr_state:KS', 'addr_state:SC', 'addr_state:CO',
        'addr_state:VT', 'addr_state:AK', 'addr_state:MS'
    ])
    df_inputs_prepr['addr_state:WV_NH_WY_DC_ME_ID'] = _safe_sum(df_inputs_prepr, [
        'addr_state:WV', 'addr_state:NH', 'addr_state:WY',
        'addr_state:DC', 'addr_state:ME', 'addr_state:ID'
    ])

    # 4 purpose
    df_inputs_prepr['purpose:educ__sm_b__wedd__ren_en__mov__house'] = _safe_sum(df_inputs_prepr, [
        'purpose:educational', 'purpose:small_business', 'purpose:wedding',
        'purpose:renewable_energy', 'purpose:moving', 'purpose:house'
    ])
    df_inputs_prepr['purpose:oth__med__vacation'] = _safe_sum(df_inputs_prepr, [
        'purpose:other', 'purpose:medical', 'purpose:vacation'
    ])
    df_inputs_prepr['purpose:major_purch__car__home_impr'] = _safe_sum(df_inputs_prepr, [
        'purpose:major_purchase', 'purpose:car', 'purpose:home_improvement'
    ])

    # 5 term
    df_inputs_prepr['term:36'] = np.where(df_inputs_prepr['term_int'] == 36, 1, 0)
    df_inputs_prepr['term:60'] = np.where(df_inputs_prepr['term_int'] == 60, 1, 0)

    # 6 emp_length
    df_inputs_prepr['emp_length:0']   = np.where(df_inputs_prepr['emp_length_int'].isin([0]), 1, 0)
    df_inputs_prepr['emp_length:1']   = np.where(df_inputs_prepr['emp_length_int'].isin([1]), 1, 0)
    df_inputs_prepr['emp_length:2-4'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(2, 5)), 1, 0)
    df_inputs_prepr['emp_length:5-6'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(5, 7)), 1, 0)
    df_inputs_prepr['emp_length:7-9'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(7, 10)), 1, 0)
    df_inputs_prepr['emp_length:10']  = np.where(df_inputs_prepr['emp_length_int'].isin([10]), 1, 0)

    # 7 int_rate
    df_inputs_prepr['int_rate:<9.548']       = np.where(df_inputs_prepr['int_rate'] <= 9.548, 1, 0)
    df_inputs_prepr['int_rate:9.548-12.025'] = np.where((df_inputs_prepr['int_rate'] > 9.548)  & (df_inputs_prepr['int_rate'] <= 12.025), 1, 0)
    df_inputs_prepr['int_rate:12.025-15.74'] = np.where((df_inputs_prepr['int_rate'] > 12.025) & (df_inputs_prepr['int_rate'] <= 15.74),  1, 0)
    df_inputs_prepr['int_rate:15.74-20.281'] = np.where((df_inputs_prepr['int_rate'] > 15.74)  & (df_inputs_prepr['int_rate'] <= 20.281), 1, 0)
    df_inputs_prepr['int_rate:>20.281']      = np.where(df_inputs_prepr['int_rate'] > 20.281, 1, 0)

    # 8 mths_since_earliest_cr_line
    # FIX: replaced range() with comparison operators (range misses floats and the max value)
    df_inputs_prepr['mths_since_earliest_cr_line:<140']    = np.where(df_inputs_prepr['mths_since_earliest_cr_line'] < 140, 1, 0)
    df_inputs_prepr['mths_since_earliest_cr_line:141-164'] = np.where((df_inputs_prepr['mths_since_earliest_cr_line'] >= 140) & (df_inputs_prepr['mths_since_earliest_cr_line'] <= 164), 1, 0)
    df_inputs_prepr['mths_since_earliest_cr_line:165-247'] = np.where((df_inputs_prepr['mths_since_earliest_cr_line'] >= 165) & (df_inputs_prepr['mths_since_earliest_cr_line'] <= 247), 1, 0)
    df_inputs_prepr['mths_since_earliest_cr_line:248-270'] = np.where((df_inputs_prepr['mths_since_earliest_cr_line'] >= 248) & (df_inputs_prepr['mths_since_earliest_cr_line'] <= 270), 1, 0)
    df_inputs_prepr['mths_since_earliest_cr_line:271-352'] = np.where((df_inputs_prepr['mths_since_earliest_cr_line'] >= 271) & (df_inputs_prepr['mths_since_earliest_cr_line'] <= 352), 1, 0)
    df_inputs_prepr['mths_since_earliest_cr_line:>352']    = np.where(df_inputs_prepr['mths_since_earliest_cr_line'] > 352, 1, 0)

    # 9 delinq_2yrs
    df_inputs_prepr['delinq_2yrs:0']   = np.where(df_inputs_prepr['delinq_2yrs'] == 0, 1, 0)
    df_inputs_prepr['delinq_2yrs:1-3'] = np.where((df_inputs_prepr['delinq_2yrs'] >= 1) & (df_inputs_prepr['delinq_2yrs'] <= 3), 1, 0)
    df_inputs_prepr['delinq_2yrs:>=4'] = np.where(df_inputs_prepr['delinq_2yrs'] >= 4, 1, 0)  # FIX: was >= 9

    # 10 inq_last_6mths
    df_inputs_prepr['inq_last_6mths:0']   = np.where(df_inputs_prepr['inq_last_6mths'] == 0, 1, 0)
    df_inputs_prepr['inq_last_6mths:1-2'] = np.where((df_inputs_prepr['inq_last_6mths'] >= 1) & (df_inputs_prepr['inq_last_6mths'] <= 2), 1, 0)
    df_inputs_prepr['inq_last_6mths:3-6'] = np.where((df_inputs_prepr['inq_last_6mths'] >= 3) & (df_inputs_prepr['inq_last_6mths'] <= 6), 1, 0)
    df_inputs_prepr['inq_last_6mths:>6']  = np.where(df_inputs_prepr['inq_last_6mths'] > 6, 1, 0)

    # 11 open_acc
    df_inputs_prepr['open_acc:0']     = np.where(df_inputs_prepr['open_acc'] == 0, 1, 0)
    df_inputs_prepr['open_acc:1-3']   = np.where((df_inputs_prepr['open_acc'] >= 1)  & (df_inputs_prepr['open_acc'] <= 3),  1, 0)
    df_inputs_prepr['open_acc:4-12']  = np.where((df_inputs_prepr['open_acc'] >= 4)  & (df_inputs_prepr['open_acc'] <= 12), 1, 0)
    df_inputs_prepr['open_acc:13-17'] = np.where((df_inputs_prepr['open_acc'] >= 13) & (df_inputs_prepr['open_acc'] <= 17), 1, 0)
    df_inputs_prepr['open_acc:18-22'] = np.where((df_inputs_prepr['open_acc'] >= 18) & (df_inputs_prepr['open_acc'] <= 22), 1, 0)
    df_inputs_prepr['open_acc:23-25'] = np.where((df_inputs_prepr['open_acc'] >= 23) & (df_inputs_prepr['open_acc'] <= 25), 1, 0)
    df_inputs_prepr['open_acc:26-30'] = np.where((df_inputs_prepr['open_acc'] >= 26) & (df_inputs_prepr['open_acc'] <= 30), 1, 0)
    df_inputs_prepr['open_acc:>=31']  = np.where(df_inputs_prepr['open_acc'] >= 31, 1, 0)

    # 12 pub_rec
    df_inputs_prepr['pub_rec:0-2'] = np.where((df_inputs_prepr['pub_rec'] >= 0) & (df_inputs_prepr['pub_rec'] <= 2), 1, 0)
    df_inputs_prepr['pub_rec:3-4'] = np.where((df_inputs_prepr['pub_rec'] >= 3) & (df_inputs_prepr['pub_rec'] <= 4), 1, 0)
    df_inputs_prepr['pub_rec:>=5'] = np.where(df_inputs_prepr['pub_rec'] >= 5, 1, 0)

    # 13 total_acc
    df_inputs_prepr['total_acc:<=27']  = np.where(df_inputs_prepr['total_acc'] <= 27, 1, 0)
    df_inputs_prepr['total_acc:28-51'] = np.where((df_inputs_prepr['total_acc'] >= 28) & (df_inputs_prepr['total_acc'] <= 51), 1, 0)
    df_inputs_prepr['total_acc:>=52']  = np.where(df_inputs_prepr['total_acc'] >= 52, 1, 0)

    # 14 acc_now_delinq
    df_inputs_prepr['acc_now_delinq:0']   = np.where(df_inputs_prepr['acc_now_delinq'] == 0, 1, 0)
    df_inputs_prepr['acc_now_delinq:>=1'] = np.where(df_inputs_prepr['acc_now_delinq'] >= 1, 1, 0)

    # 15 total_rev_hi_lim
    df_inputs_prepr['total_rev_hi_lim:<=5K']    = np.where(df_inputs_prepr['total_rev_hi_lim'] <= 5000,  1, 0)
    df_inputs_prepr['total_rev_hi_lim:5K-10K']  = np.where((df_inputs_prepr['total_rev_hi_lim'] > 5000)  & (df_inputs_prepr['total_rev_hi_lim'] <= 10000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:10K-20K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 10000) & (df_inputs_prepr['total_rev_hi_lim'] <= 20000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:20K-30K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 20000) & (df_inputs_prepr['total_rev_hi_lim'] <= 30000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:30K-40K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 30000) & (df_inputs_prepr['total_rev_hi_lim'] <= 40000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:40K-55K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 40000) & (df_inputs_prepr['total_rev_hi_lim'] <= 55000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:55K-95K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 55000) & (df_inputs_prepr['total_rev_hi_lim'] <= 95000), 1, 0)
    df_inputs_prepr['total_rev_hi_lim:>95K']    = np.where(df_inputs_prepr['total_rev_hi_lim'] > 95000, 1, 0)

    # 16 annual_inc
    df_inputs_prepr['annual_inc:<20K']      = np.where(df_inputs_prepr['annual_inc'] <= 20000,  1, 0)
    df_inputs_prepr['annual_inc:20K-30K']   = np.where((df_inputs_prepr['annual_inc'] > 20000)  & (df_inputs_prepr['annual_inc'] <= 30000),  1, 0)
    df_inputs_prepr['annual_inc:30K-40K']   = np.where((df_inputs_prepr['annual_inc'] > 30000)  & (df_inputs_prepr['annual_inc'] <= 40000),  1, 0)
    df_inputs_prepr['annual_inc:40K-50K']   = np.where((df_inputs_prepr['annual_inc'] > 40000)  & (df_inputs_prepr['annual_inc'] <= 50000),  1, 0)
    df_inputs_prepr['annual_inc:50K-60K']   = np.where((df_inputs_prepr['annual_inc'] > 50000)  & (df_inputs_prepr['annual_inc'] <= 60000),  1, 0)
    df_inputs_prepr['annual_inc:60K-70K']   = np.where((df_inputs_prepr['annual_inc'] > 60000)  & (df_inputs_prepr['annual_inc'] <= 70000),  1, 0)
    df_inputs_prepr['annual_inc:70K-80K']   = np.where((df_inputs_prepr['annual_inc'] > 70000)  & (df_inputs_prepr['annual_inc'] <= 80000),  1, 0)
    df_inputs_prepr['annual_inc:80K-90K']   = np.where((df_inputs_prepr['annual_inc'] > 80000)  & (df_inputs_prepr['annual_inc'] <= 90000),  1, 0)
    df_inputs_prepr['annual_inc:90K-100K']  = np.where((df_inputs_prepr['annual_inc'] > 90000)  & (df_inputs_prepr['annual_inc'] <= 100000), 1, 0)
    df_inputs_prepr['annual_inc:100K-120K'] = np.where((df_inputs_prepr['annual_inc'] > 100000) & (df_inputs_prepr['annual_inc'] <= 120000), 1, 0)
    df_inputs_prepr['annual_inc:120K-140K'] = np.where((df_inputs_prepr['annual_inc'] > 120000) & (df_inputs_prepr['annual_inc'] <= 140000), 1, 0)
    df_inputs_prepr['annual_inc:>140K']     = np.where(df_inputs_prepr['annual_inc'] > 140000,  1, 0)

    # 17 mths_since_last_delinq
    df_inputs_prepr['mths_since_last_delinq:Missing'] = np.where(df_inputs_prepr['mths_since_last_delinq'].isnull(), 1, 0)
    df_inputs_prepr['mths_since_last_delinq:0-3']     = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 0)  & (df_inputs_prepr['mths_since_last_delinq'] <= 3),  1, 0)
    df_inputs_prepr['mths_since_last_delinq:4-30']    = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 4)  & (df_inputs_prepr['mths_since_last_delinq'] <= 30), 1, 0)
    df_inputs_prepr['mths_since_last_delinq:31-56']   = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 31) & (df_inputs_prepr['mths_since_last_delinq'] <= 56), 1, 0)
    df_inputs_prepr['mths_since_last_delinq:>=57']    = np.where(df_inputs_prepr['mths_since_last_delinq'] >= 57, 1, 0)

    # 18 dti
    df_inputs_prepr['dti:<=1.4']     = np.where(df_inputs_prepr['dti'] <= 1.4,  1, 0)
    df_inputs_prepr['dti:1.4-3.5']   = np.where((df_inputs_prepr['dti'] > 1.4)  & (df_inputs_prepr['dti'] <= 3.5),  1, 0)
    df_inputs_prepr['dti:3.5-7.7']   = np.where((df_inputs_prepr['dti'] > 3.5)  & (df_inputs_prepr['dti'] <= 7.7),  1, 0)
    df_inputs_prepr['dti:7.7-10.5']  = np.where((df_inputs_prepr['dti'] > 7.7)  & (df_inputs_prepr['dti'] <= 10.5), 1, 0)
    df_inputs_prepr['dti:10.5-16.1'] = np.where((df_inputs_prepr['dti'] > 10.5) & (df_inputs_prepr['dti'] <= 16.1), 1, 0)
    df_inputs_prepr['dti:16.1-20.3'] = np.where((df_inputs_prepr['dti'] > 16.1) & (df_inputs_prepr['dti'] <= 20.3), 1, 0)
    df_inputs_prepr['dti:20.3-21.7'] = np.where((df_inputs_prepr['dti'] > 20.3) & (df_inputs_prepr['dti'] <= 21.7), 1, 0)
    df_inputs_prepr['dti:21.7-22.4'] = np.where((df_inputs_prepr['dti'] > 21.7) & (df_inputs_prepr['dti'] <= 22.4), 1, 0)
    df_inputs_prepr['dti:22.4-35']   = np.where((df_inputs_prepr['dti'] > 22.4) & (df_inputs_prepr['dti'] <= 35),   1, 0)
    df_inputs_prepr['dti:>35']        = np.where(df_inputs_prepr['dti'] > 35, 1, 0)

    # 19 mths_since_last_record
    df_inputs_prepr['mths_since_last_record:Missing'] = np.where(df_inputs_prepr['mths_since_last_record'].isnull(), 1, 0)
    df_inputs_prepr['mths_since_last_record:0-2']     = np.where((df_inputs_prepr['mths_since_last_record'] >= 0)  & (df_inputs_prepr['mths_since_last_record'] <= 2),  1, 0)
    df_inputs_prepr['mths_since_last_record:3-20']    = np.where((df_inputs_prepr['mths_since_last_record'] >= 3)  & (df_inputs_prepr['mths_since_last_record'] <= 20), 1, 0)
    df_inputs_prepr['mths_since_last_record:21-31']   = np.where((df_inputs_prepr['mths_since_last_record'] >= 21) & (df_inputs_prepr['mths_since_last_record'] <= 31), 1, 0)
    df_inputs_prepr['mths_since_last_record:32-80']   = np.where((df_inputs_prepr['mths_since_last_record'] >= 32) & (df_inputs_prepr['mths_since_last_record'] <= 80), 1, 0)
    df_inputs_prepr['mths_since_last_record:81-86']   = np.where((df_inputs_prepr['mths_since_last_record'] >= 81) & (df_inputs_prepr['mths_since_last_record'] <= 86), 1, 0)
    df_inputs_prepr['mths_since_last_record:>86']     = np.where(df_inputs_prepr['mths_since_last_record'] > 86, 1, 0)

    return df_inputs_prepr


# ── Classes used in Notebooks 2 & 3 ──────────────────────────────────────────

class LogisticRegression_with_p_values:
    """
    Logistic Regression wrapper that also computes p-values for each coefficient.
    After .fit(), access: .coef_, .intercept_, .p_values
    """
    def __init__(self, *args, **kwargs):
        self.model = linear_model.LogisticRegression(*args, **kwargs)

    def fit(self, X, y):
        # FIX: convert to float64 — categorical/object dtypes cause UFuncTypeError
        X = np.array(X, dtype=float)
        self.model.fit(X, y)
        denom = 2.0 * (1.0 + np.cosh(self.model.decision_function(X)))
        denom = np.tile(denom, (X.shape[1], 1)).T
        F_ij = np.dot((X / denom).T, X)
        Cramer_Rao = np.linalg.inv(F_ij)
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values


class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression with t-statistics and p-values for each coefficient.
    After .fit(), access: .t, .p
    """
    # FIX: removed deprecated 'normalize' parameter (removed in sklearn 1.2)
    def __init__(self, fit_intercept=True, copy_X=True, n_jobs=1):
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs)

    def fit(self, X, y, n_jobs=1):
        # FIX: n_jobs removed from super().fit() in sklearn 1.0+
        super(LinearRegression, self).fit(X, y)
        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
        self.t = self.coef_ / se
        self.p = np.squeeze(2 * (1 - stat.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1])))
        return self