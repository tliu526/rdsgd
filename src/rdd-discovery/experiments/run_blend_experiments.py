"""
Runs blended RDD experiments.
"""

import argparse
import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
import sys

sys.path.append("../")

from utils.rddd import *
from utils.sim import *


def _fs_sim_wrapper(seed, gap):
    """Wrapper function for first stage discovery to pass to multiprocessing pool"""
    n_samples = 1000
    running_cols = ['x', 'covar']
    treat = 't'    
    # we use a fixed bandwidth to simplify power analysis
    bw = 0.25

    grid_dict = {
        'x': np.arange(0.05, 0.96, 0.05),
        'covar': np.arange(0.1, 0.91, 0.1),
    }

    df = generate_cont_blended_rdd(
        n=n_samples,
        seed=seed,
        fuzzy_gap=gap
    )
    
    return first_stage_discovery(df, running_cols, grid_dict, treat='t', 
                                 kernel="triangular", alpha=0.05,
                                 bw=bw)


def _policy_tree_sim_wrapper(seed, gap):
    """Wrapper function for policy tree discovery"""
    n_samples = 1000
    running_cols = ['x', 'covar']
    treat = 't'    
    # we use a fixed bandwidth to simplify power analysis
    bw = 0.25
    alpha = 0.05

    grid_dict = {
        'x': np.arange(0.05, 0.96, 0.05),
        'covar': np.arange(0.1, 0.91, 0.1),
    }

    df = generate_cont_blended_rdd(
        n=n_samples,
        seed=seed,
        fuzzy_gap=gap
    )
    # drop ground truth compliance columns so causal forests won't use them as features
    df = df.drop(['comply_coeff', 'p'], axis='columns')
    
    return policy_tree_discovery(df, running_cols, grid_dict, treat=treat, bw=bw, alpha=0.05, omit_mask=True, random_state=seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("offset", type=int)
    parser.add_argument("out_path", type=str)
    parser.add_argument("discovery_type", choices=['baseline', 'subgroup'])
    args = parser.parse_args()
    n_trials = 100
    offset = args.offset
    fuzzy_gaps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    if args.discovery_type == 'subgroup':
        target_func = _policy_tree_sim_wrapper
    elif args.discovery_type == 'baseline':
        target_func = _fs_sim_wrapper
        
    out_path = args.out_path + "/seed{0}/blended_rdd_fixed_bw_{1}.pkl"

    for gap in fuzzy_gaps:
        args = [(seed + offset, gap) for seed in range(n_trials)]
        with multiprocessing.Pool(4) as pool:
            results = pool.starmap(target_func, args)

            pickle.dump(results, open(out_path.format(offset, gap), "wb"), -1)