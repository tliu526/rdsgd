"""
Utility functions for simulated data.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression

__all__ = [
    'gen_fuzzy_rdd_first_stage',
    'generate_cont_blended_rdd',
    'generate_blended_rdd_with_covars',
    'point_plot'
]

def gen_fuzzy_rdd_first_stage(n_samples, fuzzy_gap, take=0.1, seed=0, cutoff=0.5, 
                              use_covars=False,
                              reg_dict={},
                              covar_str=1, 
                              noise_std=0.1,
                              const=0.2):
    """
    Builds a fuzzy RDD first stage at the given cutoff.

    p_take is the "true" probability of getting treatment and is a function of x, the cutoff indicator, and other
    covariates

    Params:
        n_samples (int): number of samples to draw
        fuzzy_gap (float): the fuzzy gap in treatment probability
        take (float): the coefficient for probability of taking treatment as a function of x, defaults to 0.1
        seed (int): seed for reproducibility
        cutoff (float): the cutoff to center fuzzy RDD around
        use_covars (bool): optionally generate covariates that influence treatment uptake compliance
        reg_dict (dict): kwargs to pass to sklearn.make_regression, only used with use_covars = True
        covar_str (float): scaling factor for how much influence the covariates have on p_treat
        noise_std (float): the amount of noise to apply to probability of treatment (default 0.1)
        const (float): the baseline probability of treatment (default 0.2)
    Returns:
        df (pd.DataFrame): pandas Dataframe with x,y,t,z,u,p populated (note, do not use u, p in regression)
    """
    np.random.seed(seed)

    # observed covariates
    if use_covars:
        feat_df = _create_covariates(n_samples=n_samples, seed=seed, **reg_dict)

        # verify that comply_coeff is centered around 0
        # loc, scale = norm.fit(feat_df['comply_coeff'])
        #print(feat_df['comply_coeff'].describe())
    else:
        feat_df = pd.DataFrame()
        feat_df['comply_coeff'] = [0] * n_samples

    # running variable
    x = np.random.uniform(0, 1, n_samples)
        
    # cutoff indicator
    z = (x > cutoff).astype(int)

    # noise
    noise_p = np.random.normal(0, noise_std, n_samples)

    p_take = (take*x) + (z*fuzzy_gap) + (covar_str*feat_df['comply_coeff']) + const + noise_p
    
    # 0-1 clip p_take
    p_take = np.clip(p_take, 0,1)

    # treatment indicator
    t = np.random.binomial(1, p_take, n_samples)

    feat_df['x'] = x # running variable
    feat_df['z'] = z # indicator for above/below threshold
    feat_df['t'] = t # indicator for actual treatment assignment
    feat_df['p'] = p_take # true probability of treatment
    
    return feat_df


def _create_covariates(n_samples, seed=0, lower_scale=-1, **kwargs):
    """
    Generates a covariate matrix of the given number of samples and features.

    Args:
        n_samples (int): number of samples to generate
        seed (int): seed for reproducibility
        lower_scale (int): the lower bound for re-scaling coeffs to: [lower_scale, 1]
        **kwargs (dict): other keyword arguments passed to make_regression

    Returns:
        pd.DataFrame
    """
    #print(kwargs)

    X, y = make_regression(n_samples=n_samples, random_state=seed, **kwargs)

    feat_cols = ["feat_" + str(x) for x in range(X.shape[1])]

    df = pd.DataFrame(X, columns=feat_cols)

    # min max scale to [0, 1]
    # change 0 to -1 to scale to [-1, 1]
    rescale_y = lower_scale + ((y - np.min(y))*(1 - (lower_scale)) / (np.max(y) - np.min(y)))

    df['comply_coeff'] = rescale_y

    return df

def point_plot(x, target, df, scale, ax, errwidth=0):
    """Visualizes the simulated data with a pointplot."""

    sns.pointplot(np.floor(df[x]*scale) / scale, df[target], join=False, errwidth=errwidth, ax=ax)


"""Blended RDD simulations"""

# CONSTANTS
TAKE = 0.05
BOUNDARY1 = 0.25
BOUNDARY2 = 0.75

def generate_cont_blended_rdd(seed, n, fuzzy_gap=0.8, reg_dict=None):
    """Helper function for generating blended RDD in the continuous case"""
    # TODO could also add a categorical variable, for age
    use_covars = True if reg_dict else False

    df1 = gen_fuzzy_rdd_first_stage(n_samples=n//2,
                                    fuzzy_gap=fuzzy_gap, 
                                    take=TAKE, 
                                    seed=seed, 
                                    cutoff=BOUNDARY1, 
                                    use_covars=use_covars,
                                    reg_dict=reg_dict)

    df2 = gen_fuzzy_rdd_first_stage(n_samples=n//2,
                                    fuzzy_gap=fuzzy_gap, 
                                    take=TAKE, 
                                    seed=seed+1, 
                                    cutoff=BOUNDARY2, 
                                    use_covars=use_covars,
                                    reg_dict=reg_dict)


    df1['covar'] = np.random.uniform(0, .4999, size=n//2)
    df2['covar'] = np.random.uniform(0.5, 1, size=n//2)

    cont_blend_df = df1.append(df2)
    cont_blend_df = cont_blend_df.sample(frac=1).reset_index(drop=True)

    return cont_blend_df


def generate_blended_rdd_with_covars(seed, n_samples, fuzzy_gap, take, reg_dict, noise_std=0.1, const=0.2):
    """
    Generate blended RDD where multiple covariates determine cutoff compliance.
    Note here that we hard code the different cutoffs.
    """
    np.random.seed(seed)

    
    feat_df = _create_covariates(n_samples=n_samples, seed=seed, 
                                 lower_scale=0,
                                 **reg_dict)
    # running variable
    x = np.random.uniform(0, 1, n_samples)
    
    
    cutoffs = np.where(feat_df['comply_coeff'] > 0.5, 0.25, 0.75)
    
    z = (x > cutoffs).astype(int)

    # noise
    noise_p = np.random.normal(0, noise_std, n_samples)

    p_take = (take*x) + (z*fuzzy_gap) + const + noise_p
    
    # 0-1 clip p_take
    p_take = np.clip(p_take, 0,1)

    # treatment indicator
    t = np.random.binomial(1, p_take, n_samples)

    feat_df['x'] = x # running variable
    feat_df['z'] = z # indicator for above/below threshold
    feat_df['t'] = t # indicator for actual treatment assignment
    feat_df['p'] = p_take # true probability of treatment
    feat_df['cutoff'] = cutoffs
    
    return feat_df