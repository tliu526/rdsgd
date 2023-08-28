"""
Utilities for effective sample size (neff) calculations.
"""
import numpy as np
import pandas as pd

from scipy.stats import norm


def neff_influence(data_df, grp_ind_col, treat='T', instr='Z'):
    """
    Computes the influence function value for each row, given the group indicator
    column grp_ind_col. 
    
    Assumes 0 to data_df.shape[0] indexing.

    Returns as a new column.
    """
    assert data_df.index.unique().shape[0] == data_df.shape[0], \
        'data_df indices are not unique'
    
    # Obtain "G" group
    grp_df = data_df[data_df[grp_ind_col] == 1]
                             
    N_G = grp_df.shape[0]
    
    # "instrument-treated" group
    N_ZG = grp_df[grp_df[instr] == 1].shape[0]
    # "instrument-control" group
    N_CG = grp_df[grp_df[instr] == 0].shape[0]
    
    # empirical mean of treatment for ZG group
    mu_ZG = grp_df[grp_df[instr] == 1][treat].mean()
    mu_CG = grp_df[grp_df[instr] == 0][treat].mean()
    tau_G = mu_ZG - mu_CG

    try:
        Z_term = grp_df[instr] * (N_G / N_ZG) * (grp_df[treat] - mu_ZG)
        C_term = (1 - grp_df[instr]) * (N_G / N_CG) * (grp_df[treat] - mu_CG)

        neff_influences = (2 * N_G * tau_G * (Z_term - C_term)) #+ ((N_G - 1) * (tau_G**2))
        
        neff_influences = neff_influences.reindex(list(range(0, data_df.shape[0])), 
                                                fill_value=0)

        assert data_df.shape[0] == neff_influences.shape[0]
        
        return neff_influences
    except ZeroDivisionError:
        print("neff division by zero")
        return pd.Series(np.zeros(data_df.shape[0]))

def neff_covar(df, grp1_idx, grp2_idx=1, treat='T', instr='Z'):
    """Computes the var-covar matrix for both the group and whole sample neff.
    
    grp2_idx defaults to 1 to represent the "whole population" indicator.

    Returns:
        grp1, grp2 var-covar matrix
    """
        
    df['grp1_indicator'] = grp1_idx.astype(int)
    df['grp2_indicator'] = grp2_idx if isinstance(grp2_idx, int) else grp2_idx.astype(int)

    infl_df = pd.DataFrame()
    infl_df['grp1_neff_infl'] = neff_influence(df, 'grp1_indicator', treat=treat, instr=instr)
    infl_df['grp2_neff_infl'] = neff_influence(df, 'grp2_indicator', treat=treat, instr=instr)

    # divide by 2 as it is a pooled variance between two groups
    var_df = (infl_df.transpose().dot(infl_df)) / (((df['grp1_indicator'].sum() + df['grp2_indicator'].sum()) / 2)**2)

    assert var_df.shape == (2,2)

    return var_df


def neff(df, treat='T', instr='Z'):
    """computes effective sample size, n*p_comply^2"""
    comply_rate = df[(df[instr] == 1)][treat].mean() - df[(df[instr] == 0)][treat].mean()
    
    try:
        return df.shape[0] * (comply_rate**2)
    except:
        return np.nan


# TODO figure out which tstat function to use
def neff_tstat(df, grp1_idx, grp2_idx=1, treat='T', instr='Z', grp1_tau=None, grp2_tau=None):
    """
    Computes neff difference t-stat for arbitrary groups.

    grp2_idx defaults to 1 to represent the "whole population" indicator.
    
    Optionally provide grp1_tau and grp2_tau if computed outside of function.
    """
    
    grp1_df = df[grp1_idx]
    #print(type(grp2_idx))
    if isinstance(grp2_idx, int) and grp2_idx == 1:
        grp2_df = df
    else:
        grp2_df = df[grp2_idx]

    if grp1_tau is None:
        grp1_neff = neff(grp1_df, treat=treat, instr=instr)
    else:
        grp1_neff = grp1_tau**2 * grp1_df.shape[0]
    
    if grp2_tau is None:
        grp2_neff = neff(grp2_df, treat=treat, instr=instr)
    else:
        grp2_neff = grp2_tau**2 * grp2_df.shape[0]

    covar_mat = neff_covar(df, grp1_idx=grp1_idx, grp2_idx=grp2_idx, 
                            treat=treat, instr=instr).values
    
    grp1_var = covar_mat[0, 0]
    grp2_var = covar_mat[1, 1]
    covar = covar_mat[0, 1]

    t_diff = grp1_neff - grp2_neff

    # variance of difference of two random variables
    t_var = grp1_var + grp2_var - (2*covar)

    t_stat = t_diff / np.sqrt(t_var)

    return t_stat


def pval(t_stat):
    """Computes p-value for t-stat."""
    return 2 * (1 - norm.cdf(t_stat, loc=0, scale=1))


def neff_test(df, grp1_idx, grp2_idx=1, treat='T', instr='Z'):
    """Computes the t-stat and pvalue for the neff difference test."""
    t_stat = neff_tstat(df, grp1_idx, grp2_idx, treat=treat, instr=instr)
    p_val = pval(t_stat)
    
    return t_stat, p_val
