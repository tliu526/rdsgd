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