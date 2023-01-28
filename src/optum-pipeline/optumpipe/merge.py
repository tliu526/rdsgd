"""
Functions for merging raw distributed dataframes into a single df for analysis.
"""
import configparser
import json
import numpy as np
import os
import pandas as pd

# load default configuration
config = configparser.ConfigParser()
config.read("config.ini")
default_config = config['default']

def is_number(s):
    """Checks whether the value can be intepreted as a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False

def lr_patch_lab(row):
    """Fill the lab value for readings placed in the rslt_txt field"""
    if row.rslt_nbr < 0.0001 and is_number(row.rslt_txt):
        return float(row.rslt_txt)
    elif row.rslt_nbr < -99999:
        return np.nan
    else:
        return row.rslt_nbr


def lr_prep(lr_df):
    """
    Formats and prepares lr frames for merging.
    
    Currently consists of formatting lab numbers and dropping null rows.
    """

    lr_df['lr_fmt'] = lr_df.apply(lr_patch_lab, axis=1)

    lr_df = lr_df.dropna(subset=['lr_fmt'])
    lr_df = lr_df.reset_index(drop=True)

    return lr_df

def rx_prep(r_df):
    """
    Prepares r(x) frames for merging.

    Currently consists of:
        1. copying the fill date to the "fst_dt" index column
        2. dropping rows that are not a first fill
    """

    r_df['fst_dt'] = r_df['fill_dt']
    #r_df = r_df[r_df['fst_fill'] == 'Y']

    return r_df

def gen_diag_indicator(pre_df, post_df, window, idx='patid', 
                       save_tmp_df=False, tmp_out_path=None):
    """
    Generates a diagnosis indicator column, defined by the presence of a row
    matching a patid in post_df within the defined time window.

    As part of ths process, sorts pre_df based on fst_dt and dedups wrt to idx.
    This results in pre_df containing the "first" screen/lab test/encounter
    for the given patid.

    A left join is then performed between pre_df and post_df, and rows are 
    filtered based on the given time window.

    Pre:
        Both pre_df and post_df have fst_dt and idx columns

    Args:
        pre_df (pd.df): the pre-diag dataframe
        post_df (pd.df): the diagnosis dataframe to be matching to patids
        window (int): the time window in days that constitute a match
        idx (str): the column to index on, default 'patid' only
        save_tmp_df (bool): optionally save the intermediate df
        tmp_out_path (str): the path for the temporary dfs

    Returns:
        indicator_df: a dataframe mapping 1:1 between patids in pre_df and
            whether a matching diagnosis was found in post_df.
            Columns: patid, fst_dt_pre, fst_dt_post, indicator
    """

    # sort and format idx
    first_df = pre_df.sort_values(by=[idx, 'fst_dt']).copy()
    first_df[idx] = first_df[idx].astype(str)

    # grab first instance for each patid
    first_df = first_df.groupby("patid").head(1)
    #first_df = first_df.reset_index(drop=True)

    post_df[idx] = post_df[idx].astype(str)

    first_df = first_df.merge(post_df, on=idx, how='left', 
                              suffixes=['_pre', '_post'])

    first_df['date_diff'] = first_df['fst_dt_post'] -  first_df['fst_dt_pre'] 

    start_td = pd.to_timedelta(0, unit='d')
    end_td = pd.to_timedelta(window, unit='d')

    first_df.loc[
        (first_df['date_diff'] < start_td) | (first_df['date_diff'] > end_td), 
        ['fst_dt_post', 'date_diff']    
    ] = np.nan

    matched_patids = first_df[~first_df['fst_dt_post'].isna()][idx].unique()

    # drop any duplicates from the one to many left merge
    first_df = first_df.sort_values(by=[idx, 'fst_dt_post'])
    first_df = first_df.drop_duplicates(subset=idx)

    first_df['indicator'] = first_df[idx].isin(matched_patids).astype(int)

    first_df = first_df.reset_index(drop=True)

    return first_df

def gen_prior_diag_indicator(indicator_df, post_df, idx='patid'):
    """
    Generates an indicator column for the presence of prior diagnosis.

    Assumes gen_diag_indicator has already been run.

    Args:
        indicator_df (pd.DataFrame): the indicator df, with dedup'd pre dates
        post_df (pd.df): the diagnosis dataframe to be matching to patids
        idx (str): the column to index on, default 'patid' only
    Returns:
        indicator_df, with new column 'prior_indicator'
    """
    temp_merged = left_join_patid(indicator_df.copy(), post_df)

    # the post_df date will be 'fst_dt'
    temp_merged = temp_merged[temp_merged['fst_dt_pre'] > temp_merged['fst_dt']]

    prior_diag_patids = temp_merged['patid'].unique()

    indicator_df['prior_indicator'] = (indicator_df['patid'].astype(float).astype(int).isin(prior_diag_patids)).astype(int)

    return indicator_df

def left_join_patid(left_df, right_df, suffixes=("_x", "_y")):
    """
    Formats the given left_df and left joins it with right_df.

    Also applies speed-ups by temporarily making "patid" the index.

    Args:
        left_df (pd.df): left df to be joined
        right_df (pd.df): right df to be joined
        suffixes (list-like): pass-through for the suffixes parameter on pd.merge
    """
    
    # make patid an int for faster index merge
    left_df['patid'] = left_df['patid'].astype(float).astype(int) 
    left_df = left_df.set_index("patid")
    left_df = left_df.sort_index()

    right_df['patid'] = right_df['patid'].astype(float).astype(int) 
    right_df = right_df.set_index("patid")
    right_df = right_df.sort_index()

    merged_df = left_df.merge(
                    right_df, 
                    left_index=True, 
                    right_index=True, 
                    how='left',
                    suffixes=suffixes
    )
    
    merged_df = merged_df.reset_index()
    
    return merged_df





    
