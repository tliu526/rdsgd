import pandas as pd
import numpy as np
import scipy as sp
import os
import math
import time
from subset_functions import *
from model_functions import *

# For McCray 2008 tests
'''
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
mccray = importr('rdd')
'''


def tau_compute(tau_method, obs_model, idxs, neighs, x, y, T, T_hat_master, beta_0_n, beta_1_n, subsets_best, llrs, estimation='Lee', f_base='RLM_poly1', f_yT='RLM_poly1', agg_type='avg', instrument=True, verbose=False):
    '''
    Estiamted the treatment effect (\tau) through a variety of different methods

    Args:
        tau_method (str): method for computing \tau {'nonpara, 2SLS, group'}
        obs_model (str): obbservation model {'normal', 'bernoulli'}
        idxs, neighs, x, y, T, T_hat_master, beta_0_n, beta_1_n, subsets_best, llrs, estimation='Lee', f_base='RLM_poly1', f_yT='RLM_poly1', agg_type='avg', instrument=True, verbose=False
    '''

    if verbose: print ('compute tau with:', tau_method)
    # Compute T_eff for each neigh
    T_effs = np.zeros(len(idxs))
    T_bses = np.zeros(len(idxs))
    
    for ii, idx in enumerate(idxs):

        # Get subset indicators
        s0 = subset_neigh(neighs[:, idx], subsets_best[idx])
        s1 = subset_neigh(neighs[:, idx], ~subsets_best[idx])

        if tau_method == 'nonpara':            
            T_effs[ii] = (np.mean(y[s0]) - np.mean(y[s1])) / (np.mean(T[s0]) - np.mean(T[s1]))
            #T_bses[ii] # TODO

        if tau_method == '2SLS':
            # todo if perfect separation, then this errors out

            T_in = pd.DataFrame(T[neighs[:, idx]])
            y_in = y[neighs[:, idx]]
            x_in = x[neighs[:, idx]]

            try:
                if estimation == 'Lee':
                    if verbose: print ('1 polynomial option via Lee and Lemiuex')
                    ss = pd.DataFrame(subsets_best[idx])
                    if verbose: print (' first regression')
                    try:
                        T_fx = basic_fit(y=T_in, x=x_in.reset_index(drop=True), f_base=f_base, x2=pd.DataFrame(ss).astype(float), verbose=verbose)
                    except:
                        T_fx = basic_fit(y=T_in, x=pd.DataFrame(ss).astype(float), f_base=f_base, verbose=verbose)
                    if verbose: print (' get means')
                    D = get_pred_mean(T_fx, f_base, x, verbose=verbose)
                    if verbose: print (' second regression')
                    y_fx = basic_fit(y=y_in, x=x_in.reset_index(drop=True), f_base=f_yT, x2=pd.DataFrame(D), verbose=verbose)

                elif estimation == 'Imbens':
                    if verbose: print ('2 polynomial option via Imbens Sec4.4, Eq4.9')
                    x_in0 = np.array(x_in) * subsets_best[idx].reshape(50, 1)
                    x_in1 = np.array(x_in) * ~subsets_best[idx].reshape(50, 1)
                    x_in = pd.DataFrame(np.concatenate([x_in0, x_in1], axis=1))
                    if verbose: print (' regression with', f_yT)
                    y_fx = basic_fit(y=y_in, x=x_in.reset_index(drop=True), f_base=f_yT, x2=T_in, verbose=verbose)

                T_effs[ii], T_bses[ii] = get_pred_variable(y_fx, f_name=f_yT, varnum=0)

            except:
                print ('Failed regress_y_on_T. Test if y has all the same values:', np.all(y_in == y_in[0]), '. y:', np.mean(y_in), np.std(y_in))
                T_effs[ii] = np.nan
                T_bses[ii] = np.nan

        elif tau_method == 'group':
            # Instument T with the a subset
            T_hat = np.copy(T_hat_master)
            if instrument and (obs_model=='normal'):
                T_hat[s0,:] -= beta_0_n[idx]
                T_hat[s1,:] -= beta_1_n[idx]
            elif instrument and (obs_model=='bernoulli'):
                p0 = T_hat[s0,:]
                p1 = T_hat[s1,:]
                q0 = beta_0_n[idx]
                q1 = beta_1_n[idx]
                T_hat[s0,:] = (q0*p0) / (1 - p0 + (q0*p0))
                T_hat[s1,:] = (q1*p1) / (1 - p1 + (q1*p1))

            # Regress on y to fit tau_T
            T_effs[ii], T_bses[ii] = regress_y_on_T(x, y, T_hat, f_yT, neighs, idx, multi=False, verbose=verbose)
            if instrument and (obs_model == 'bernoulli'):
                T_effs[ii] *= -1.0


    # Combine multipe T_effs
    T_eff = aggregate_stats(T_effs, agg_type, llrs, idxs, verbose=verbose)
    T_bse = aggregate_stats(T_bses, agg_type, llrs, idxs, verbose=verbose)
    if verbose: print  ('T effect: T_eff =', T_eff, 'T_bse =', T_bse)

    return T_eff, T_effs, T_bse, T_bses



def T_effect_multi(obs_model, idxs, neighs, x, y, T_hat_master, beta_0_n, beta_1_n, subsets_best, llrs, f_yT='RLM_poly1', agg_type='avg', instrument=True, verbose=False):
    '''
    Compute T effect for all subsets indexed by idx through averaging the betas at eah point
    '''

    if verbose: print ('T_eff estimation multi', f_yT, agg_type)
    n = y.shape[0]
    betas = np.zeros((n,len(idxs)))

    # Collect the betas from relevant neighs
    for ii, idx in enumerate(idxs):
        betas[subset_neigh(neighs[:,idx], subsets_best[idx]),ii]  = -beta_0_n[idx]
        betas[subset_neigh(neighs[:,idx], ~subsets_best[idx]),ii] = -beta_1_n[idx]

    # Average out the betas for all idxs
    if agg_type == 'wavg':
        # llrs_sorted = np.flip(np.sort(llrs[idxs]),0)
        # llrs_sorted /= sum(llrs_sorted)
        llrs_sorted = np.flip(np.sort(llrs[idxs]), 0)
        llrs_sorted = (betas != 0) * llrs_sorted
        llrs_sorted = llrs_sorted / np.sum(llrs_sorted, axis=1).reshape(n, 1)
        betas *= llrs_sorted
        betas_agg = np.sum(betas, axis=1)

    elif agg_type == 'avg':
        betas_agg = np.sum(betas, axis=1) / np.sum(betas!=0, axis=1)

    else:
        assert False, 'Invalid T_eff aggregation type'
    betas_agg[np.isnan(betas_agg)] = 0

    #   get T
    T_hat = np.copy(T_hat_master)
    if instrument and (obs_model == 'normal'):
        T_hat += betas_agg.reshape(n,1)
    elif instrument and (obs_model == 'bernoulli'):
        p = T_hat
        q = betas_agg.reshape(n,1)
        T_hat = -(q*p) / (1 - p + (q*p)) # todo Why add negative sign in front?

    # Regress on y to fit tau_T
    T_eff, T_bse = regress_y_on_T(x, y, T_hat, f_yT, neighs, idxs, multi=True, verbose=verbose)

    if verbose: print ('Mutli T effect: T_eff =', T_eff, 'T_bse =', T_bse)
    return T_eff, T_bse


def regress_y_on_T(x, y, T_hat, f_yT, neighs, idxs, multi, verbose=False):
    '''
    multi (bool): indicator for T_effect_multi()
    '''

    if verbose: 'regress_y_on_T()'

    # Determine inputs
    if multi: nn = np.any(neighs[:,idxs], axis=1)
    else:     nn = neighs[:,idxs]
    y_in = y[nn]
    x_in = pd.DataFrame( np.array(x)[nn,:] )
    T_in = pd.DataFrame( T_hat[nn] )
    
    try:
        # Regress
        y_fx = basic_fit(y_in, x_in, f_yT, x2=T_in, verbose=verbose)
        T_eff, T_bse = get_pred_variable(y_fx, f_name=f_yT, varnum=0)

    except:
        print ('Failed regress_y_on_T. Test if y has all the same values:', np.all(y_in==y_in[0]), '. y:', np.mean(y_in), np.std(y_in))
        T_eff = np.nan
        T_bse = np.nan

    return T_eff, T_bse


def z_effect(obs_model, idxs, beta_0_n, beta_1_n, llrs, T_hat, agg_type='avg', verbose=False):

    if verbose: print ('z_eff estimation', agg_type)
    z_effs = np.zeros(len(idxs))
    # Compute z_eff for each neigh
    for ii, idx in enumerate(idxs):
        if obs_model=='normal':
            z_effs[ii] = abs(beta_0_n[idx] - beta_1_n[idx])
        elif obs_model == 'bernoulli':
            #z_effs[ii] = max(beta_0_n[idx], 1.0/beta_0_n[idx])/2.0 + max(beta_1_n[idx], 1.0/beta_1_n[idx])/2.0
            z_effs[ii] = max(np.log(beta_0_n[idx]) - np.log(beta_1_n[idx]),
                             np.log(beta_1_n[idx]) - np.log(beta_0_n[idx])) / 2.0

    # Combine multipe z_effs
    z_eff = aggregate_stats(z_effs, agg_type, llrs, idxs)
    if verbose: print ('z_eff =', z_eff)

    return z_eff, z_effs


def infogain_subsets(idxs, neighs, discont, subsets_best, llrs, agg_type='avg', verbose=False):
    '''
    Compute information gain of the regions.
    '''

    if verbose: print ('Computing accuracy using', agg_type)
    # Compute acc for each neigh
    igratios = np.zeros(len(idxs))
    for ii, idx in enumerate(idxs):
        neigh = neighs[:, idx]
        s0 = subset_neigh(neigh, subsets_best[idx])
        s1 = subset_neigh(neigh, ~subsets_best[idx])
        k = float(len(subsets_best[idx]))

        #IG = H(discont(neighborhood)) - (np.sum(s0) / k) * H(discont[s0]) - (np.sum(s1) / k) * H(discont[s1])
        ig = entropy(np.sum(discont[neigh]) / k, k) \
            - (np.sum(s0) / k) * entropy(np.sum(discont[s0]) / float(np.sum(s0)), k) \
             - (np.sum(s1) / k) * entropy(np.sum(discont[s1]) / float(np.sum(s1)), k)
        igratios[ii] = ig / entropy(0.5, k)

    # Combine multipe accs
    igratio = aggregate_stats(igratios, agg_type, llrs, idxs)
    if verbose: print ('igratio =', igratio)

    return igratio, igratios


def entropy(p,n):
    p = max(p,1e-6)
    p = min(p, 1-1e-6)
    return -n * (np.dot(p, np.log(p)) + np.dot(1-p, np.log(1-p)))


def aggregate_stats(stats, agg_type, llrs=None, idxs=None, verbose=False):
    '''
    Final two inputs are optional, only for wavg
    '''
    if verbose: 'aggregate_stats', agg_type

    if agg_type == 'wavg':
        llrs_sorted = np.flip(np.sort(llrs[idxs]), 0)

        # if sum(np.exp(llrs_sorted)) == 0: return np.nan
        # stat_agg = np.average(np.squeeze(stats), weights=np.squeeze(np.exp(llrs_sorted) / sum(np.exp(llrs_sorted))))
        if np.any(llrs_sorted < 0):
            print ('WARNING: LLR value < 0, this will mess up the wavg!')
        # todo make this robust to nans
        stat_agg = np.average(np.squeeze(stats), weights=np.squeeze(llrs_sorted / sum(llrs_sorted)))

    elif agg_type == 'avg':
        stat_agg = np.nanmean(stats)
    else:
        assert False, 'Invalid z_eff aggregation type'

    return stat_agg


def validate_subsets(llrs, llr_sig, pivots_best, centers_n, x, z, pval_sig=0.05, top_subsets=1, verbose=False):

    if verbose: print ('Validating subsets for y...')
    # Precompute which idxs are valid to use for TE
    #   randomization testing for significance
    if llr_sig:
        idxs = np.flip(np.argsort(llrs)[np.where(np.sort(llrs) > llr_sig)], 0)
    else:
        idxs = np.flip(np.argsort(llrs), axis=0)

    # Choose maximum the number of top_subsest subsets
    idxs_top = []
    for ii, idx in enumerate(idxs):
        if len(idxs_top) >= top_subsets: break

        #   validate mccray 2008 test of density
        center_i = centers_n[idx]
        pivot = pivots_best[idx]
        projs = project_points(x, z, center_i, pivot, plotting=False)
        projs = projs[~np.isnan(projs)] # remove nans

        # For McCray 2008 tests
        try:
            '''
            mccray_out = mccray.DCdensity(robjects.FloatVector(tuple(projs)), 0.0, plot=False, ext_out=True)
            mccray_pval = mccray_out[3]
            '''
            mccray_pval = pval_sig*10.0 # hardcode for systems without rpy2
            if mccray_pval > pval_sig: # not reject
                idxs_top.append(idx)
            if verbose:
                print ('mccray p-value', mccray_pval)
                # plt.hist(projs[~np.isnan(projs)])
        except:
            # TODO This conservatively means that ANYTIME mccray errors out we remove. this is usually caused by small number of points on one side.
            #      We can make this more robust, or at least include explainations
            continue

    if verbose: 
        print ('Validated', len(idxs_top), 'out of', top_subsets, 'possible top subsets')

    return np.array(idxs_top)


def RDSS_result_stats_sig(obs_model, llrs, idxs, beta_0_n, beta_1_n, subsets_best, neighs, discont, x, y, T, T_hat_master, f_base, f_yT, verbose=False):
    '''
    Wrapper for RDSS_result_stats that computes average over all significant subsets
    '''

    T_eff = np.zeros(8)
    T_bse = np.zeros(8)
    z_eff = np.zeros(2)
    acc = np.zeros(4) #2
    T_placebos = np.zeros((2, x.shape[1], 2))

    # special case of no sig idxs
    if len(idxs)==0:
        return z_eff*np.nan, acc*np.nan, T_eff*np.nan

    T_eff[0], _, T_bse[0], _ = tau_compute('2SLS',    obs_model, idxs, neighs, x, y, T, T_hat_master, beta_0_n, beta_1_n, subsets_best, llrs, estimation='Lee', f_base=f_base, f_yT=f_yT, agg_type='avg', instrument=True, verbose=verbose)
    T_eff[1], _, T_bse[1], _ = tau_compute('nonpara', obs_model, idxs, neighs, x, y, T, T_hat_master, beta_0_n, beta_1_n, subsets_best, llrs, estimation='Lee', f_base=f_base, f_yT=f_yT, agg_type='avg', instrument=True, verbose=verbose)
    T_eff[2], _, T_bse[2], _ = tau_compute('group',   obs_model, idxs, neighs, x, y, T, T_hat_master, beta_0_n, beta_1_n, subsets_best, llrs, estimation='Lee', f_base=f_base, f_yT=f_yT, agg_type='avg', instrument=True, verbose=verbose)
    
    acc[2], _ = infogain_subsets(idxs, neighs, discont, subsets_best, llrs, agg_type='avg', verbose=verbose)
    acc[3], _ = infogain_subsets(idxs, neighs, discont, subsets_best, llrs, agg_type='wavg', verbose=verbose)

    # todo: include BSE in placebo as well
    T_placebos[0,:,:], _ = placebo_testing_linear(obs_model, idxs, neighs, x, y, T, T_hat_master, beta_0_n, beta_1_n, subsets_best, llrs, f_base, f_yT, agg_type='avg', multi=False, verbose=False)
    T_placebos[1,:,:], _ = placebo_testing_linear(obs_model, idxs, neighs, x, y, T, T_hat_master, beta_0_n, beta_1_n, subsets_best, llrs, f_base, f_yT, agg_type='avg', multi=True, verbose=False)

    return z_eff, acc, T_eff, T_bse, T_placebos



def placebo_testing_linear(obs_model, idxs, neighs, x, y, T, T_hat_master, beta_0_n, beta_1_n, subsets_best, llrs, f_base, f_yT,
                    agg_type, multi=False, verbose=False):
    '''
    Wrapper function does linear placebo tests by iteratively considering each x as a potential outcome.
    '''

    if verbose: print ('Placebo testing linear', agg_type)
    T_effs = np.zeros((len(x.columns),2))
    T_bses = np.zeros((len(x.columns), 2))
    for ii, col in enumerate(x.columns):
        placebos = [col]
        T_effs[ii,:], T_bses[ii,:] = placebo_testing(obs_model, idxs, neighs, x, y, T, T_hat_master, beta_0_n, beta_1_n, subsets_best, llrs, f_base, f_yT,
                    placebos, agg_type, multi=multi, verbose=verbose)

    return T_effs, T_bses


def placebo_testing(obs_model, idxs, neighs, x, y, T, T_hat_master, beta_0_n, beta_1_n, subsets_best, llrs, f_base, f_yT,
                    placebos, agg_type, multi=False, verbose=False):
    '''
    Placebo tests on values of x. User specifies which values of x to consider for placebos. All
    other values of x are used for control

    e.g. placebos = [1,2]

    Return:
        T_effs (np.array): treatment effects of placebos and y, in that order
    '''

    if verbose: print ('Placebo testing with', placebos)
    controls = list(set(x.columns) - set(placebos))
    x_in = x[controls]
    placebos.extend(['y'])
    T_effs = np.zeros(len(placebos))
    T_bses = np.zeros(len(placebos))

    for p_idx, placebo in enumerate(placebos):

        if verbose: print ('placebo:', placebo)

        # Consider each placebo and y
        if placebo == 'y':
            y_in = y
        else:
            y_in = np.array(x[placebo])

        # Treatment effect
        if multi:
            T_effs[p_idx], T_bses[p_idx] = T_effect_multi(obs_model, idxs, neighs, x_in, y_in, T_hat_master, beta_0_n, beta_1_n, subsets_best, llrs,
                                           f_yT=f_yT, agg_type=agg_type, verbose=verbose)

        else:
            #T_effs[p_idx], _, T_bses[p_idx], _ = tau_group(obs_model, idxs, neighs, x_in, y_in, T_hat_master, beta_0_n, beta_1_n, subsets_best, llrs,
            #                            f_yT=f_yT, agg_type=agg_type, verbose=verbose)
            T_effs[p_idx], _, T_bses[p_idx], _ = tau_compute('group', obs_model, idxs, neighs, x_in, y_in, T, T_hat_master, beta_0_n, beta_1_n, subsets_best, llrs,
             estimation='Lee', f_base=f_base, f_yT=f_yT, agg_type=agg_type, instrument=True, verbose=verbose)
            #TODO: Allow choice beyond tau_medthod = 'group'
            #TODO: Allow choice for instrument

            

    return T_effs, T_bses


def combine_greedy(neighs, subsets_best, beta_0_n, beta_1_n, llrs, idx_curr, min_llr, min_per=0.0, max_per=1.0, above=np.array([]), below=np.array([])):
    '''
    Combination strategy which adds the points in a neighborhood to either an "above" or "below"
    group depending on which beta value is greater. Then it looks for the intersecting neigh with
    highest LLR and repeats the process on all points not already processed.
    Users can specify where to begin this recursive combination, the lowest llr value to include
    and define the extent of overlap allowed between the included area and a new neigh.
    
    Example usage starting at max_arg, the highest scoring subset:
        above, below = combine_greedy(neighs, subsets_best, beta_0_n, beta_1_n, llrs, idx_curr=np.nanargmax(llrs), min_llr=llrs[max_arg]/1.5)
        plot_neigh(x, z, out=T, out_name='Combined groups in red and black', neigh=[above,below])
    
    Args:
        idx_curr (int): index of datapoint to add 
        min_per (float): minimum percentage of neigh intersecting points
        max_per (float): maximum percentage of neigh intersecting points
        min_llr (float): min llr to use
        above (np.arr): keeps track of points that are in the above subset
        below (np.arr): keeps track of points that are in the below subset
    Return:
        above (np.arr): keeps track of points that are in the above subset
        below (np.arr): keeps track of points that are in the below subset
        
    '''
    # - Intialize the matrix
    if above.size==0:
        above = np.zeros(neighs.shape[0], dtype=bool) # neighs.shape[0] is the number of data points
        below = np.zeros(neighs.shape[0], dtype=bool)

    #- Set those points to either above or below
    if beta_0_n[idx_curr] > beta_1_n[idx_curr]:
        above[subset_neigh(neighs[:,idx_curr], subsets_best[idx_curr])] = True
        below[subset_neigh(neighs[:,idx_curr], ~subsets_best[idx_curr])] = True
    else:
        above[subset_neigh(neighs[:,idx_curr], ~subsets_best[idx_curr])] = True
        below[subset_neigh(neighs[:,idx_curr], subsets_best[idx_curr])] = True

    #- Find highest scoring neighborhood with more than min_per and less than max_per intersection points with 
    #  at least llr of value min_llr
    above_or_below = above | below
    llr_next = -1
    for idx_neigh in range(neighs.shape[1]):
        intersecting = sum(above_or_below & neighs[:,idx_neigh])
        n_curr = sum(neighs[:,idx_neigh])
        if (intersecting >= (min_per*n_curr)) & (intersecting < (max_per*n_curr)): # strictly > max_per to ensure we dont reselect
            if min_llr < llrs[idx_neigh]:
                llr_next = llrs[idx_neigh]
                idx_next = idx_neigh

    # - recurse
    if llr_next > min_llr:
        print ('llr_next', llr_next, 'idx_next', idx_next)
        #above, below = combine_greedy(idx_curr=idx_next, min_per=min_per, max_per=max_per, min_llr=min_llr, above=above, below=below)
        above, below = combine_greedy(neighs, subsets_best, beta_0_n, beta_1_n, llrs, idx_curr=idx_next, min_llr=min_llr, min_per=min_per, max_per=max_per, above=above, below=below)
    return above, below


