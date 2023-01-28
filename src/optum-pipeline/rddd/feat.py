"""
Utilities for featurizing final merged Optum data frames for RDD analysis.

For all the dataframes, assumes that the SES table has been merged.
"""

import pandas as pd
import numpy as np


def _compute_age(df, idx_date_col='fst_dt_pre'):
    """
    Computes age (in years) from index date and drops rows with invalid ages.

    Args:
        df (pd.DataFrame): input data frame
        idx_col (str): the index date column, defaults to 'fst_dt_pre'

    Returns:
        df with age computed and invalid ages removed
    """

    # this will be according to fst_dt_pre, rounded down to the year
    df['age'] = df[idx_date_col].dt.year - df['yrdob']

    return df[(df['age'] >= 0) & (df['age'] <= 90)]

# SES categories to featurize
UNORDERED_CATS = [
    'd_race_code',
    'gdr_cd',
    'bus'
]

ORDERED_CATS = [
    'd_education_level_code',
    'd_household_income_range_code'
]

def _featurize_cols(feat_df):
    """
    Converts categorical columns into 1-hot representations for machine learning.

    Args:
        feat_df (pd.DataFrame): the input data frame with feature columns selected

    Returns:
        df with featurized columns
    """
    feat_df = pd.get_dummies(feat_df, columns=UNORDERED_CATS, dummy_na=True)


    # ordered categoricals
    edu_dict = {char: num for char, num in zip('ABCD', range(1,5))}
    feat_df['d_education_level_code'] = feat_df['d_education_level_code'].map(edu_dict)

    # TODO need to replace 0s in income codes with nan
    feat_df['d_household_income_range_code'] = feat_df['d_household_income_range_code'].replace('0', np.nan)

    for cat in ORDERED_CATS:
        feat_df[cat] = feat_df[cat].astype(float)
        feat_df[cat] = feat_df[cat].astype("Int64")

    # create nan indicator
    feat_df = pd.concat([feat_df, feat_df[ORDERED_CATS].isnull().astype(int).add_suffix("_nan")], axis=1)

    # fill missing values
    #display(feat_df[ORDERED_CATS].mode())
    feat_df[ORDERED_CATS] = feat_df[ORDERED_CATS].fillna(value=feat_df[ORDERED_CATS].mode().iloc[0])

    # format columns
    feat_df = feat_df.reset_index(drop=True)
    feat_df.columns = feat_df.columns.str.replace(' ', '_')

    return feat_df


def gen_feat_df(df, rdd_cols=['indicator'], compute_age=True):
    """
    Selects relevant columns from the data frame and generates end-to-end featurized columns.

    Args:
        df (pd.DataFrame): input, unformatted column
        rdd_cols (list): list of all columns from original df to keep for RDD analysis

    Returns:
        feat_df: featurized, cleaned dataframe
    """
    if compute_age:
        df = _compute_age(df)

    # remove patients with prior diagnosis of the treatment of interest
    df = df[df['prior_indicator'] == 0]

    # populate encounter day column
    df['encounter_day'] = (df['fst_dt_pre'] - pd.to_datetime('2004-01-01')).dt.days

    feat_df = df[rdd_cols + ['age', 'encounter_day'] +  UNORDERED_CATS + ORDERED_CATS].copy()

    feat_df = _featurize_cols(feat_df)

    # remove patients with nan demographics
    nan_cols = list(feat_df.columns[feat_df.columns.str.contains('nan')])
    if 'd_race_code_' in list(feat_df.columns):
        nan_cols = nan_cols + ['d_race_code_']
    if 'gdr_cd_U' in list(feat_df.columns):
        nan_cols = nan_cols + ['gdr_cd_U']

    feat_df = feat_df[(feat_df[nan_cols] == 0).all(axis=1)]
    feat_df = feat_df.drop(nan_cols, axis='columns')

    # select adults and patients with valid ages
    feat_df = feat_df[(feat_df['age'] >= 18) & (feat_df['age'] <= 90)]    

    return feat_df


def get_descriptives(baseline_df, col):
    count_df = baseline_df[col].value_counts().to_frame()
    pct_df = (baseline_df[col].value_counts() / baseline_df.shape[0]).to_frame()
    display(pd.concat([count_df, pct_df], axis=1))
