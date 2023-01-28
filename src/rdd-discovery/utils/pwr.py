"""
Utilities for closed-form power calculations.
"""
import numpy as np

from scipy.stats import norm


def rdd_power(effect, var, bias=0, alpha=0.05):
    """
    Implements the power function of the LATE as described in
    Cattaneo et al. 2019.
    Assumes a two-sided hypothesis test.
    Params:
        effect (float): desired LATE to be detected (against a no effect null)
        var (float): the estimated variance of the treatment effect estimator
        bias (float): estimated misspecification bias of the estimator,
                      defaults to 0
        alpha (float): significance level, defaults to 0.05
    Returns:
        power (float): the power of the specified setup
    """
    lower = norm.cdf((effect + bias) / np.sqrt(var) + norm.ppf(1 - (alpha/2)))
    upper = norm.cdf((effect + bias) / np.sqrt(var) - norm.ppf(1 - (alpha/2)))
    power = 1 - lower + upper

    return power


def first_stage_asymp_var(ptake_lower, ptake_upper, total_n, n_incl):
    """
    Computes the first stage asymptotic variance in closed form, from Imbens and Lee 2008.
    
    This calculation assumes a symmetric bandwidth.
    
    Args:
        ptake_lower (float): the probability of treatment, lower limit
        ptake_upper (float): the probability of treatment, upper limit
        total_n (int): the total sample size
        n_incl (int): the number of included units within the bandwidth.edge
        
    Returns:
        float, the asymptotic variance to be plugged into a power calculation.
    """
    
    sigma_low = ptake_lower * (1 - ptake_lower)
    sigma_hi = (ptake_upper) * (1 - ptake_upper)
    
    return (sigma_low + sigma_hi) * (8 / n_incl)


def diff_var(group_idx, df):
    """Computes the t-stat variance for TAU difference according to Dwivedi et al."""
    grp_df = df[group_idx]
    grp_comp_df = df[~group_idx]

    grp_comp_ctl_rate = grp_comp_df[grp_comp_df['z'] == 0].shape[0] / df[df['z'] == 0].shape[0]
    grp_comp_trt_rate = grp_comp_df[grp_comp_df['z'] == 1].shape[0] / df[df['z'] == 1].shape[0]

    grp_ctl_pr_T = grp_df[grp_df['z'] == 0]['t'].mean()
    grp_trt_pr_T = grp_df[grp_df['z'] == 1]['t'].mean()

    grp_comp_ctl_pr_T = grp_comp_df[grp_comp_df['z'] == 0]['t'].mean()
    grp_comp_trt_pr_T = grp_comp_df[grp_comp_df['z'] == 1]['t'].mean()


    var_grp_ctl = (grp_ctl_pr_T * (1 - grp_ctl_pr_T)) / grp_df[grp_df['z'] == 0].shape[0]
    var_grp_comp_ctl = (grp_comp_ctl_pr_T * (1 - grp_comp_ctl_pr_T)) / grp_comp_df[grp_comp_df['z'] == 0].shape[0]

    var_grp_trt = (grp_trt_pr_T * (1 - grp_trt_pr_T)) / grp_df[grp_df['z'] == 1].shape[0]
    var_grp_comp_trt = (grp_comp_trt_pr_T * (1 - grp_comp_trt_pr_T)) / grp_comp_df[grp_comp_df['z'] == 1].shape[0]


    return (grp_comp_ctl_rate**2 * (var_grp_ctl + var_grp_comp_ctl)) + (grp_comp_trt_rate**2 * (var_grp_trt + var_grp_comp_trt))

    
def diff_tstat(group_idx, df):
    """Computes the t-stat for TAU difference according to Dwivedi et al."""
    group_df = df[group_idx]
    group_tau = group_df[(group_df['z'] == 1)]['t'].mean() - group_df[(group_df['z'] == 0)]['t'].mean()

    avg_tau = df[(df['z'] == 1)]['t'].mean() - df[(df['z'] == 0)]['t'].mean()
    
    try:
        return (group_tau - avg_tau) / (np.sqrt(diff_var(group_idx, df))) #/ np.sqrt(group_idx.shape[0]))
    except:
        return np.nan
    

def diff_tstat_wrapper(group_tau, group_idx, df):    
    """For numerical differentiation wrt to group_tau"""
    group_df = df[group_idx]
    avg_tau = df[(df['z'] == 1)]['t'].mean() - df[(df['z'] == 0)]['t'].mean()
    
    try:
        return (group_tau - avg_tau) / (np.sqrt(diff_var(group_idx, df))) #/ np.sqrt(group_idx.shape[0]))
    except:
        return np.nan
    

def neff(df):
    comply_rate = df[(df['z'] == 1)]['t'].mean() - df[(df['z'] == 0)]['t'].mean()
    
    try:
        return df.shape[0] * (comply_rate**2)
    except:
        return np.nan
    
def n_comply(df):
    comply_rate = 1 - (1 - df[(df['z'] == 1)]['t'].mean()) - (df[(df['z'] == 0)]['t'].mean())

    try:
        return df.shape[0] * (comply_rate)
    except:
        return np.nan