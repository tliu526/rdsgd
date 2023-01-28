import zipfile
import pandas as pd
import numpy as np
import scipy as sp
import os
import math
import time
import pickle as pkl
import datetime
import argparse

#import statsmodels
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smd

from data_functions import *
from subset_functions import *
from model_functions import *
from search_functions import *
from analysis_functions import *
from parsing import *

#import matplotlib.pyplot as plt


def main(args):
    
    np.random.seed(args.seed)
    date_now = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    if not (np.any(args.ks)):
        k = [args.k]
    else:
        k = args.ks

    # experiments
    vars_run = {}
    for val_i in range(len(args.linspace)):
        vars_run[val_i] = {}

        exp_i = 0
        failures = 0
        while exp_i < args.exp_n:
            vars_run[val_i][exp_i] = {}

            print ('Experiment #', exp_i, args.variable, args.linspace[val_i])
            t0 = time.time()

            # Generate data
            if args.data_verbose: print ('Generating data...')
            if args.variable == "z_eff":
                x, y, z, T, D_z, z_eff, beta_y_T, discont = data_synthetic(args.data_type, n=args.n_size, px=args.px, pz=args.pz,
                                                                              z_eff=args.linspace[val_i],
                                                                              poly_xT=args.poly_xT,
                                                                              discont_type=args.discont_type,
                                                                              plotting=False,
                                                                              verbose=args.data_verbose)
            elif args.variable == "T_eff":
                x, y, z, T, D_z, z_eff, beta_y_T, discont = data_synthetic(args.data_type, n=args.n_size, px=args.px, pz=args.pz,
                                                                              beta_y_T=args.linspace[val_i],
                                                                              poly_xT=args.poly_xT,
                                                                              discont_type=args.discont_type,
                                                                              plotting=False,
                                                                              verbose=args.data_verbose)

            elif args.variable == "px":
                x, y, z, T, D_z, z_eff, beta_y_T, discont = data_synthetic(args.data_type, n=args.n_size,             pz=args.pz,
                                                                              px=int(args.linspace[val_i]),
                                                                              z_eff=args.z_eff,
                                                                              poly_xT=args.poly_xT,
                                                                              discont_type=args.discont_type,
                                                                              plotting=False,
                                                                              verbose=args.data_verbose)

            elif args.variable == "pz":
                x, y, z, T, D_z, z_eff, beta_y_T, discont = data_synthetic(args.data_type, n=args.n_size, px=args.px,
                                                                              pz=int(args.linspace[val_i]),
                                                                              z_eff=args.z_eff,
                                                                              poly_xT=args.poly_xT,
                                                                              discont_type=args.discont_type,
                                                                              plotting=False,
                                                                              verbose=args.data_verbose)

            elif args.variable == "fuzzy": # real data
                # data
                file_json = 'data_inst.json'
                x, y, z, T, x_cols, inst, discont = data_real(args.data_type, file_json, subsample=args.subsample, plotting=False, verbose=args.data_verbose)
                print ('y.columns',y.columns)
                print ('original x:', x.shape)
                # fuzzy
                T = fuzzy_binary_T(T, args.linspace[val_i])
                # normalize
                x, x_means, x_stds = normalize_xz(x, z)
                print ('normalized x:', x.shape)

            else:
                print (args.variable)
                assert (False, 'Not yet implemented')


            # create result structures at the first step
            if (val_i == 0) and (exp_i == 0):
                if args.data_verbose: 'Creating data structures...'
                z_effs = np.zeros((args.exp_n, len(args.linspace), 2, y.shape[1]))
                accs = np.zeros((args.exp_n, len(args.linspace), 4, y.shape[1]))
                T_effs = np.zeros((args.exp_n, len(args.linspace), 8, y.shape[1]))
                T_bses = np.zeros((args.exp_n, len(args.linspace), 8, y.shape[1]))
                T_placebos = np.zeros((args.exp_n, len(args.linspace), 2, x.shape[1], 2, y.shape[1])) # args.px, 2, y.shape[1]))


            # Search for discontinuity
            if args.search_verbose: print ('RDSS searching...')
            output = RDSS_residual_multik(args.obs_model, T, x, z, f_base=args.f_base,
                                     all_points=args.all_points, ks=k, verbose=args.search_verbose)
            if len(output) == 1:
                failures += 1
                print ('Failure #', failures, 'Trying to redraw again...')  # todo make sure this doesnt continue in an infinite loop...
                if failures > 10:
                    assert False, 'Too many failures'
                continue
            failures = 0
            llrs, neighs, subsets_best, beta_0_n, beta_1_n, T_fx, llrs_n, llrs_a, centers_n, pivots_best, subset_imax = output
            T_hat_master = get_pred_mean(T_fx, args.f_base, x, verbose=args.search_verbose)

            # un-normalize
            if args.variable == "fuzzy":
                if args.search_verbose: print ('Unnormalizing')
                x = unnormalize_xz(x, z, x_means, x_stds)

            # Validation of subsets (including potential randomization testing)
            if args.iters_rand > 0:
                #todo remove T_fx from both rand_testing
                llr_sig, llr_max_samples, llr_all_samples = rand_testing(args.obs_model, T_fx, T_hat_master, args.k_samples, args.iters_rand, args.alpha, T, x, z, args.f_base, all_points=args.all_points, k=k)
                validated_idxs = validate_subsets(llrs, llr_sig, pivots_best, centers_n, x, z, pval_sig=0.05, top_subsets=args.top_subsets, verbose=False)
                validated_idxs_all = validate_subsets(llrs, llr_sig, pivots_best, centers_n, x, z, pval_sig=0.05, top_subsets=x.shape[0], verbose=False)
            else:
                validated_idxs = validate_subsets(llrs, False, pivots_best, centers_n, x, z, pval_sig=0.05, top_subsets=args.top_subsets, verbose=False)
                validated_idxs_all = validate_subsets(llrs, False, pivots_best, centers_n, x, z, pval_sig=0.05, top_subsets=x.shape[0], verbose=False)

            # process treatment effect
            for col_idx, col in enumerate(y.columns):
                if args.data_verbose: print ('T_eff for y output:', col)
                y_col = np.array(y[col])

                if args.search_verbose: print ('Compute results for y...')
                print (T_placebos[exp_i, val_i, :, :, :, col_idx].shape)
                output = RDSS_result_stats_sig(args.obs_model, llrs, validated_idxs, beta_0_n, beta_1_n, subsets_best, neighs, discont, x, y_col, T, T_hat_master, f_base=args.f_base, f_yT=args.f_yT, verbose=args.search_verbose)
                try:
                    z_effs[exp_i, val_i, :, col_idx] = output[0]
                    accs[exp_i, val_i, :, col_idx] = output[1]
                    T_effs[exp_i, val_i, :, col_idx] = output[2]
                    T_bses[exp_i, val_i, :, col_idx] = output[3]
                    T_placebos[exp_i, val_i, :, :, :, col_idx] = output[4]
                except:
                    print ('Error: issue with results from RDSS_result_stats_sig')
            
            # store relevant data from this run
            if args.save_all_data | (args.iters_rand > 0):
                print ('storing all')
                vars_to_save = ['x', 'y', 'T', 'z', 'discont', 'T_hat_master', 'llrs',  'subsets_best', 'beta_0_n', 'beta_1_n',
                 'validated_idxs','validated_idxs_all', 'pivots_best', 'subset_imax']
                if args.iters_rand > 0:
                    vars_to_save.extend(['llr_sig', 'llr_max_samples'])
                for vv in vars_to_save:
                    vars_run[val_i][exp_i][vv] = locals()[vv]

            # Update
            exp_i += 1
            print ('updated exp_i', exp_i)

            # Timing
            t1 = time.time()
            print ('RD_script iter:', t1 - t0)


    # Save results
    if args.search_verbose: print ('Saving results...')
    dir_results = '../results'
    dict_save = {}
    
    vars_to_save = ['z_effs', 'accs', 'T_effs', 'T_placebos', 'T_bses']
    if args.save_all_data | (args.iters_rand > 0):
        vars_to_save.append('vars_run')
    
    #vars_to_save = ['z_effs', 'accs', 'T_effs', 'T_placebos', 'T_bses', 'vars_run']
    #'x', 'y', 'T', 'z', 'discont', 'T_hat_master', 'llrs',  'subsets_best', 'beta_0_n', 'beta_1_n', 'validated_idxs','validated_idxs_all']
    # REMOVED FOR SPACE: 'neighs', 'centers_n', 'llrs_n', 'llrs_a', 'pivots_best', 

    for i in vars_to_save:
        dict_save[i] = locals()[i]
    print (dict_save.keys())
    filename = os.path.join(dir_results, "exp_" + date_now)
    pkl.dump([dict_save, args], open(filename + ".p", "wb"))

def parse_args():
    parser, groups = get_default_argparser()


    # RDD testing arguments
    groups['model'].add_argument('--exp_n',
        default=50, type=int,
        help="Number of experiments per z_eff value")
    groups['model'].add_argument('--linspace',
        default=(0.1, 0.5, 5), type=float, nargs=3,
        help="Linspace for values to vary")
    groups['model'].add_argument('--variable',
         default="z_eff", type=str,
         help="Target variable to vary")
    groups['data'].add_argument('--save_all_data',
        action="store_true",
        help="Save the data for every experimental run")
    groups['model'].add_argument('--discont_type',
         default="square", type=str,
         help="Discontinuity type {square, linear, poly#}")

    #Parse args
    args = parse_input(parser)

    # Adjust variables
    args.linspace = np.linspace(args.linspace[0], args.linspace[1], int(args.linspace[2]))

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
