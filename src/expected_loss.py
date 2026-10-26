"""
LGD, EAD, Expected Loss and Population Stability Index helpers.

Expected Loss = PD × LGD × EAD, where
    LGD = 1 - recovery rate      (share of the exposure lost once a loan defaults)
    EAD = CCF × funded amount    (amount still owed at default)
"""
import numpy as np
import pandas as pd

# LGD and EAD use the raw numeric variables plus a few dummy groups.
LGD_EAD_FEATURES = [
    'grade:A', 'grade:B', 'grade:C', 'grade:D', 'grade:E', 'grade:F', 'grade:G',
    'home_ownership:MORTGAGE', 'home_ownership:NONE', 'home_ownership:OTHER',
    'home_ownership:OWN', 'home_ownership:RENT',
    'verification_status:Not Verified', 'verification_status:Source Verified',
    'verification_status:Verified',
    'purpose:car', 'purpose:credit_card', 'purpose:debt_consolidation',
    'purpose:educational', 'purpose:home_improvement', 'purpose:house',
    'purpose:major_purchase', 'purpose:medical', 'purpose:moving', 'purpose:other',
    'purpose:renewable_energy', 'purpose:small_business', 'purpose:vacation',
    'purpose:wedding',
    'initial_list_status:f', 'initial_list_status:w',
    'term_int', 'emp_length_int', 'mths_since_issue_d', 'mths_since_earliest_cr_line',
    'funded_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'delinq_2yrs',
    'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc',
    'pub_rec', 'total_acc', 'acc_now_delinq', 'total_rev_hi_lim',
]

LGD_EAD_REF_CATEGORIES = [
    'grade:G', 'home_ownership:RENT', 'verification_status:Verified',
    'purpose:credit_card', 'initial_list_status:f',
]


def lgd_ead_design_matrix(df):
    """Select LGD/EAD inputs with reference categories dropped; missing delinquency months -> 0."""
    X = df.reindex(columns=LGD_EAD_FEATURES, fill_value=0).astype(float)
    X[['mths_since_last_delinq', 'mths_since_last_record']] = (
        X[['mths_since_last_delinq', 'mths_since_last_record']].fillna(0))
    return X.drop(columns=LGD_EAD_REF_CATEGORIES)


def recovery_rate(df):
    """Recoveries as a share of the funded amount, bounded to [0, 1]."""
    return (df['recoveries'] / df['funded_amnt']).clip(0, 1)


def credit_conversion_factor(df):
    """Share of the funded amount still outstanding at default."""
    return ((df['funded_amnt'] - df['total_rec_prncp']) / df['funded_amnt']).clip(0, 1)


def psi(expected, actual):
    """
    Population Stability Index between two distributions given as proportions
    over the same bins. < 0.1 stable, 0.1-0.25 some shift, > 0.25 significant shift.
    """
    e = np.clip(np.asarray(expected, dtype=float), 1e-6, None)
    a = np.clip(np.asarray(actual, dtype=float), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_by_feature(expected_dummies, actual_dummies, original_feature_names):
    """
    PSI per original variable from its dummy columns.

    expected_dummies / actual_dummies: 0/1 frames with the same columns.
    original_feature_names: maps each dummy column to its source variable.
    """
    shares = pd.DataFrame({
        'expected': expected_dummies.mean(),
        'actual': actual_dummies.mean(),
        'feature': pd.Series(original_feature_names),
    })
    return (shares.groupby('feature')
            .apply(lambda g: psi(g['expected'], g['actual']), include_groups=False)
            .rename('PSI').sort_values(ascending=False))
