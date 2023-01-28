import pandas as pd
import numpy as np
import time
import progressbar

from data_functions import *
from subset_functions import *
from model_functions import *

#import matplotlib.pyplot as plt


def RDSS_residual_multik(obs_model, T,x,z,f_base='RLM', all_points=False, ks=[100], RPD=0, post_shrink=False, verbose=False, plotting=False):
    '''
    Wrapper function to RDSS_residual() enabling search over potentially multiple vlaues for k
    '''

    if verbose: print ('Starting RDSS multi searching...')
    # init dicts for everything
    subsets_best = {}
    pivots_best = {}
    subset_imax = {}
    max_x    = [np.nan] * len(ks)
    llrs     = [np.nan] * len(ks)
    neighs   = [np.nan] * len(ks)
    beta_0_n = [np.nan] * len(ks)
    beta_1_n = [np.nan] * len(ks)
    llrs_n   = [np.nan] * len(ks)
    llrs_a   = [np.nan] * len(ks)
    centers_n = [np.nan] * len(ks)
    T_fx = False

    # run RDSS multiple times
    for ii, k in enumerate(ks):
        output = RDSS_residual(obs_model, T, x, z, f_base=f_base, all_points=all_points, k=k, subsets_best_prev=subsets_best, pivots_best_prev=pivots_best, subset_imax_prev=subset_imax, RPD=RPD, T_fx_prev=T_fx, post_shrink=False, verbose=verbose, plotting=plotting)
        if len(output)==1: return ([-1])
        llrs[ii], neighs[ii], subsets_best, beta_0_n[ii], beta_1_n[ii], T_fx, llrs_n[ii], llrs_a[ii], centers_n[ii], pivots_best, subset_imax = output

    # combine llrs
    llrs = np.concatenate(llrs, axis=0)
    neighs = np.concatenate(neighs, axis=1)
    beta_0_n = np.concatenate(beta_0_n, axis=0)
    beta_1_n = np.concatenate(beta_1_n, axis=0)
    llrs_n = np.concatenate(llrs_n, axis=0)
    llrs_a = np.concatenate(llrs_a, axis=0)
    centers_n = np.concatenate(centers_n, axis=0)

    # check
    assert len(subsets_best) == llrs.shape[0], 'Error'

    return llrs, neighs, subsets_best, beta_0_n, beta_1_n, T_fx, llrs_n, llrs_a, centers_n, pivots_best, subset_imax


def RDSS_residual(obs_model,T,x,z,f_base='OLS', all_points=False, k=100, subsets_best_prev={}, pivots_best_prev={}, subset_imax_prev={}, RPD=0, T_fx_prev=False, post_shrink=False, verbose=False, plotting=False, verbose_bar=True):
    '''
    Compute RD-SS using the residual approach. Search for anomalous transitions in the residuals

    Args:
        obs_model (str): type of observation model {'normal','bernoulli'}
        T: n*1 treatment (binary for now)
        x: n*p covariates where p is the numbr of covariates
        z:     forcing variable
        f_base (str): what type of base function, f(x), to use
        all_points (bool): [False]
        k (int): number of neighbors [100]
        subsets_best_prev:
        pivots_best_prev:
        subset_imax_prev
        RPD (int): placeholder functionality for extensions of the current work. Keep at 0 for now. [0]
        T_fx_prev (bool): [False]
        post_shrink= (bool): [False]
        verbose (bool): [False]
        plotting (bool): [False]
        verbose_bar (bool): [True]
        verbose (bool): plotting and verbose output
    Return:

    '''

    if verbose: print ('Starting RDSS searching for k =',k,'...')

    t0 = time.time()
    (n,px) = x.shape

    if verbose: print ('Getting neighbors...')
    neighs, neighs_n = get_neighs('k_nn', x, z, k=k)
    llrs = np.zeros(neighs_n)
    llrs_n = np.zeros(neighs_n)
    llrs_a = np.zeros(neighs_n)
    beta_0_n = np.zeros(neighs_n)
    beta_1_n = np.zeros(neighs_n)
    centers_n = np.zeros(neighs_n)
    subsets_best = subsets_best_prev.copy()
    pivots_best = pivots_best_prev.copy()
    subset_imax = subset_imax_prev.copy()

    # Fit T = f(X) + \epsilon_i
    if T_fx_prev:
        if verbose: print ('Using previously fitted model...')
        T_fx = T_fx_prev
    else:
        if verbose: print ('Fitting model...')
        T = T.astype(float)
        try:
            T_fx = basic_fit(T, x, f_base, verbose=verbose)
        except:
            print ('WARNING: Failed to bacis_fit() in the search function. Repulling data.')
            return ([-1])

    if obs_model=='normal':

        if verbose: print ('Computing residuals, r_i = \epsilon_i ...')
        r = T - get_pred_mean(T_fx, f_base, x)

        if verbose: print ('Precomputing variance, sigma_i ...')
        
        # Estimate variance locally
        sigma_2_all = np.ones(r.shape) * -1
        for n_i in range(neighs_n):
            sigma_2_all[n_i] = np.var(r[neighs[n_i]])

        assert np.all(sigma_2_all >= 0), 'Negative variance. Either invalid or non-existant'
        sigma_2_all += 1e-6 # add epsilon to avoid divide by zero
        sigma_2_all = sigma_2_all.flatten() # flatten

    elif obs_model=='bernoulli':
        if verbose: print ('Computing bernoulli probabilities ...')
        p = get_pred_mean(T_fx, f_base, x)


    # Progress bar
    if verbose_bar:
        bar = progressbar.ProgressBar(maxval=neighs_n, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

    #if len(T.shape) == 2:
    #    assert T.shape[0] > T.shape[1], 'T shape issue ' + str(T.shape)
    if (obs_model == 'normal'):
        if (len(r.shape) == 2):
            assert r.shape[0] > r.shape[1], 'r shape issue ' + str(r.shape)

    # Plot
    if plotting:
        plot_neigh(x, z, T, 'Treatment as a function of z', neigh=None, D_z=None, z_eff=None)
        if obs_model == 'normal':
            plot_neigh(x, z, r, 'Residual as a function of z', neigh=None, D_z=None, z_eff=None)


    # Find the optimal bisection for each neighborhood
    if verbose: print ('Starting search over neighs...')
    for n_i in range(neighs_n):
        if verbose_bar:
            bar.update(n_i)

        # LL for for the neigh
        x_n = x[neighs[:,n_i]]
        center_i = np.where(np.where(neighs[:,n_i])[0] == n_i)[0][0]
        # Note: If it fails here there may be discrete values and thus distances that are the same. So if
        #       there is dist=0 and more of those than points in a neigh we may not include self in the neigh.

        # For each subset (subdivide k times)
        subsets, subsets_n, dist_pivot, angles = get_vector_neighs(x_n, z, center_i, all_points=all_points, verbose=False)

        llrs_s = np.zeros(subsets_n)
        ll_alt = np.zeros(subsets_n)
        beta_0 = np.zeros(subsets_n)
        beta_1 = np.zeros(subsets_n)

        if obs_model=='normal':

            r_n = r[neighs[:,n_i]] # neigh residual
            sigma_2 = sigma_2_all[neighs[:,n_i]]

            # Flatten residuals
            try:    r_n = r_n.flatten()
            except: r_n = r_n

            if RPD==0:
                llrs_n[n_i] = loglik_null_normal(sigma_2, r_n)
            #elif RPD==1:
            #    Compute later since z_n is a function of the bisection


        elif obs_model == 'bernoulli':
            p_n = p[neighs[:, n_i]]  # neigh probabilities
            T_n = T[neighs[:, n_i]]  # neigh treatments

            beta_n = binary_search_q(T_n, p_n)
            llrs_n[n_i] = loglik_bernoulli_logodds(T_n, p_n, beta_n)


        for s_i in range(subsets_n):

            # indicators for the two subsets w.r.t. all data
            s_i0 = subsets[:, s_i] == 0
            s_i1 = subsets[:, s_i] == 1

            # Compute alternative model
            if RPD==0:
                if obs_model=='normal':
                    beta_0[s_i], beta_1[s_i], ll_alt[s_i], llrs_s[s_i] = alternative_opt_normal(s_i0, s_i1, sigma_2, r_n, llrs_n[n_i])

                elif obs_model=='bernoulli':
                    beta_0[s_i], beta_1[s_i], ll_alt[s_i], llrs_s[s_i] = alternative_opt_bernoulli(s_i0, s_i1, T_n, p_n, llrs_n[n_i])

            elif RPD==1:
                #The null is a function of the bisection to account for any discont aside from one of polynomial order P from RPD

                # Compute distnace from bisection
                #   hack to fix nan in angles for the point that the center goes to
                angles_s = angles[:,s_i]
                if np.sum(np.isnan(angles_s))==1:
                    angles_s[np.isnan(angles_s)] = 0
                dist_n_s = np.linalg.norm(dist_pivot, axis=1) * np.sin(angles_s+np.pi/2)

                # null
                llrs_n_i = loglik_null_normal_1(sigma_2, r_n, dist_n_s)
                
                # alt
                if obs_model=='normal':
                    beta_0[s_i], beta_1[s_i], ALPHA_0_NEED, ALPHA_0_NEED, ll_alt[s_i], llrs_s[s_i] = loglik_alt_normal_1(s_i0, s_i1, sigma_2, r_n, dist_n_s, llrs_n_i)

                elif obs_model=='bernoulli':
                    assert False, 'Not yet implemented!'

            else:
                assert False, 'Proper observation model needed'

        # TODO what is causing nan in ll_alt? Usually just a few

        # Test if all nans
        if np.all(np.isnan(llrs_s)):
            print ('ERROR: All nan in llrs_s')
            if obs_model=='normal':
                print ('neigh residuals', r_n)
            elif obs_model=='bernoulli':
                print ('neigh probabilities', p_n, 'neigh treatments', T_n, 'beta_n', beta_n)
            print ('ll_null', llrs_n[n_i], 'll_alt', ll_alt)
            return ([-1])


        # Get best neigh LLR
        llrs[n_i] = np.nanmax(llrs_s)
        i_max = np.nanargmax(llrs_s)
        centers_n[n_i] = n_i
        pivots_best[len(pivots_best)] = dist_pivot[i_max, :]
        subset_imax[len(subset_imax)] = i_max

        if post_shrink:
            if obs_model=='normal':
                llrs[n_i], beta_0_n[n_i], beta_1_n[n_i], subsets_best[len(subsets_best)], llrs_a[n_i], neighs[:,n_i] = iterate_shrink_neigh(obs_model, x_n, n_i, neighs, dist_pivot, angles[:,i_max], subsets[:, i_max], llrs_n, dist_func='to_bisection', opt1=sigma_2, opt2=r_n)
            elif obs_model=='bernoulli':
                llrs[n_i], beta_0_n[n_i], beta_1_n[n_i], subsets_best[len(subsets_best)], llrs_a[n_i], neighs[:,n_i] = iterate_shrink_neigh(obs_model, x_n, n_i, neighs, dist_pivot, angles[:,i_max], subsets[:, i_max], llrs_n, dist_func='to_bisection', opt1=T_n, r_n=p_n)
        else:
            beta_0_n[n_i] = beta_0[i_max]
            beta_1_n[n_i] = beta_1[i_max]
            subsets_best[len(subsets_best)] = subsets[:, i_max]
            llrs_a[n_i] = ll_alt[i_max]

    # Timing
    t1 = time.time()
    if verbose: print ('Finished RDSS search in', t1-t0, 'seconds')

    # Print and show beta
    if plotting:
        plot_neigh(x, z, llrs, 'LLR -- detected', neigh=None, D_z=None, z_eff=None)
        
        if len(z)==1:
            import matplotlib.pyplot as plt
            plt.scatter(x[z],np.maximum(beta_0_n, beta_1_n),color='red')
            plt.scatter(x[z],np.minimum(beta_0_n, beta_1_n),color='blue')
            plt.title('Beta values')
            plt.show()
        else:
            if verbose: print ('TODO: plot beta0 and beta1 for multiple dim')
            #plot_neigh(x, z, out=T, out_name='Treatment', neigh=subset_neigh(neighs[:,1], subsets[:,0]), D_z=D_z, z_eff=z_eff)

    return (llrs, neighs, subsets_best, beta_0_n, beta_1_n, T_fx, llrs_n, llrs_a, centers_n, pivots_best, subset_imax)


def iterate_shrink_neigh(obs_model, x_n, n_i, neighs, dist_pivot, angles_s, subset, llrs_n, dist_func='to_center', opt1=None, opt2=None):
    '''
    Iteratively shrinks the neigh of interest to find the neigh shape with the highest LLR.

    Usage:
        iterate_shrink_neigh(obs_model, x, n_i, subset_imax, neighs, dist_pivot, angles, i_max, subsets[:, i_max], llrs_n, dist_func='to_bisection', sigma_2=None, r_n=None, T_n=None, p_n=None):
    For Normal:
        opt1=sigma_2, opt2=r_n
    For Bernoulli:
        opt1=T_n, r_n=p_n
    '''

    # Compute distance
    center_i = np.where(np.where(neighs[:,n_i])[0] == n_i)[0][0]
    if dist_func == 'to_center':
        dist = np.linalg.norm(dist_pivot, axis=1) # np.sqrt(np.sum(dist_pivot**2, axis=1))
    elif dist_func == 'to_bisection':
        dist = np.linalg.norm(dist_pivot, axis=1) * np.sin(angles_s+np.pi/2)
        dist = abs(dist)

    # Iteratively remove the farthest point and compute LLR. Keep the best.
    x_ns = np.copy(np.array(x_n))
    subset_curr = np.copy(np.array(subset)) # np.copy(subsets_best[n_i])
    opt1_curr = np.copy(np.array(opt1))
    opt2_curr = np.copy(np.array(opt2))
    neighs_curr = np.copy(neighs[:,n_i])
    neighs_true = np.where(neighs_curr)[0] # list of where the true elements in nieghs are, for speed

    llrs_max = -1000
    while len(dist) > 0:

        # compute LLR [adapted from RDSS_residual()]
        s_i0 = subset_curr == 0
        s_i1 = subset_curr == 1

        if obs_model=='normal':
            llrs_n_curr = loglik_null_normal(opt1_curr, opt2_curr)
            beta_0_curr, beta_1_curr, ll_alt_curr, llrs_s_curr = alternative_opt_normal(s_i0, s_i1, opt1_curr, opt2_curr, llrs_n_curr)
        
        elif obs_model=='bernoulli':
            #beta_n = GD_search_q(opt1_curr, opt2_curr, min_q=0)
            beta_n = binary_search_q(opt1_curr, opt2_curr)
            llrs_n_curr = loglik_bernoulli_logodds(opt1_curr, opt2_curr, beta_n)
            beta_0_curr, beta_1_curr, ll_alt_curr, llrs_s_curr = alternative_opt_bernoulli(s_i0, s_i1, opt1_curr, opt2_curr, llrs_n_curr)
        
        else:
            assert False, 'Proper observation model needed'

        if llrs_s_curr > llrs_max:
            llrs_max = llrs_s_curr
            beta_0_max = beta_0_curr
            beta_1_max = beta_1_curr
            subsets_best = subset_curr
            ll_alt_best = ll_alt_curr
            neighs_best = np.copy(neighs_curr)

        # remove furthest point
        max_curr = np.argmax(dist)
        dist = np.delete(dist, max_curr)
        x_ns = np.delete(x_ns, max_curr, 0)
        subset_curr = np.delete(subset_curr, max_curr)
        opt1_curr = np.delete(opt1_curr, max_curr)
        opt2_curr = np.delete(opt2_curr, max_curr)
        neighs_curr[neighs_true[max_curr]] = False
        neighs_true = np.delete(neighs_true, max_curr)

        #plot_neigh(pd.DataFrame(x_ns), z, out=abs(dist), out_name='dist '+ dist_func)

    return llrs_max, beta_0_max, beta_1_max, subsets_best, ll_alt_best, neighs_best


def loglik_null_normal(sigma_2, r_n):
    beta_n = np.mean(r_n / sigma_2) / np.mean(1 / sigma_2) # think this should be np.sum, no difference though...
    llrs_n = np.sum(-(r_n-beta_n)**2 / (2*sigma_2))
    return llrs_n


def mle_normal_1(sigma_2, r_n, z_n):
    '''
    Compute alpha and beta MLE for RKD (RPD order 1). For individual groups use ____[s_i0] or ____[s_i1] as inputs 
    Input:
        sigma_2 (): sigma_i^2
        r_n (): residuals_i for the neigh
        z_n (): distance_i from the boundary for the neigh
    '''
    nn = r_n.shape[0]

    A = np.sum(r_n / sigma_2) / np.sum(1 / sigma_2)
    B = np.sum(z_n / sigma_2) / np.sum(1 / sigma_2)
    C = np.sum(r_n * z_n / sigma_2) / np.sum((z_n**2) / sigma_2) 
    D = np.sum(      z_n / sigma_2) / np.sum((z_n**2) / sigma_2) 

    alpha_mle = (C - A*D) / (1-B)
    beta_mle = A - (alpha_mle * B)

    return alpha_mle, beta_mle


def loglik_null_normal_1(sigma_2, r_n, z_n):
    '''
    Compute null log likelihood of RKD (RPD order 1)
    '''
    beta0_g0, beta0_g1, beta1_0 = mle_normal_1(sigma_2, r_n, z_n) # <-- need to specify null I think...
    mu0_i = (beta0_g0 * s_i0) + (beta0_g1 * s_i1)
    llrs_n = np.sum(-(r_n - mu0_i - alpha*z_n)**2 / (2*sigma_2))

    return llrs_n


def loglik_alt_normal_1(s_i0, s_i1, sigma_2, r_n, z_n, llrs_n_i):
    '''
    Compute alt log likelihood of RKD (RPD order 1)
    '''
    alpha_0, beta_0 = mle_normal_1(sigma_2[s_i0], r_n[s_i0], z_n[s_i0])
    alpha_1, beta_1 = mle_normal_1(sigma_2[s_i1], r_n[s_i1], z_n[s_i1])

    if np.isnan(beta_0): beta_0 = 0
    if np.isnan(beta_1): beta_1 = 0
    if np.isnan(alpha_0): alpha_0 = 0
    if np.isnan(alpha_1): alpha_1 = 0

    mu_i = (beta_0 * s_i0) + (beta_1 * s_i1)
    nu_i = (alpha_0 * s_i0) + (alpha_1 * s_i1)

    ll_alt_i = np.sum(-( r_n - mu_i - nu_i*z_n )**2 / (2*sigma_2))
    if np.isnan(ll_alt_i):
        llrs_s_i = np.nan
    else:
        llrs_s_i = (ll_alt_i - llrs_n_i)

    return beta_0,  beta_1, alpha_0, alpha_1, ll_alt_i,  llrs_s_i


def alternative_opt_normal(s_i0, s_i1, sigma_2, r_n, llrs_n_i):
    '''
    For Normal observation model, compute the alternative model in a neighborhood given a subset
    Args:
        s_i0 (np.arr): indicator for data points in all data that are in the G0 of the subset
        s_i1 (np.arr): indicator for data points in all data that are in the G1 of the subset
        sigma_2 (np.arr): variance of all data points
        r_n (np.arr): residual of all data points
        llrs_n_i (?): log likelihood of null model for the neighborhood
    Return:
        beta_0_i (np.arr): \beta_0 for subset
        beta_1_i (np.arr): \beta_1 for subset
        ll_alt_i (float): log likelihood of alt model for the neighborhood
        llrs_s_i (float): LLR for subset
    '''

    # Determine parameters
    beta_0_i = np.mean(r_n[s_i0] / sigma_2[s_i0]) / np.mean(1 / sigma_2[s_i0])
    beta_1_i = np.mean(r_n[s_i1] / sigma_2[s_i1]) / np.mean(1 / sigma_2[s_i1])

    if np.isnan(beta_0_i): beta_0_i = 0
    if np.isnan(beta_1_i): beta_1_i = 0

    # Compute LLR
    mu_i = (beta_0_i * s_i0) + (beta_1_i * s_i1)
    ll_alt_i = np.sum(-( r_n - mu_i )**2 / (2*sigma_2))
    if np.isnan(ll_alt_i):
        llrs_s_i = np.nan
    else:
        llrs_s_i = (ll_alt_i - llrs_n_i)

    return beta_0_i,  beta_1_i,  ll_alt_i,  llrs_s_i


def alternative_opt_bernoulli(s_i0, s_i1, T_n, p_n, llrs_n_i):

    # Determine parameters
    #beta_0_i = GD_search_q(T_n[s_i0], p_n[s_i0], min_q=0)
    #beta_1_i = GD_search_q(T_n[s_i1], p_n[s_i1], min_q=0)
    beta_0_i = binary_search_q(T_n[s_i0], p_n[s_i0])
    beta_1_i = binary_search_q(T_n[s_i1], p_n[s_i1])
    
    # Compute LLR
    mu_i = (beta_0_i * s_i0) + (beta_1_i * s_i1)
    ll_alt_i = loglik_bernoulli_logodds(T_n, p_n, mu_i)

    if np.isnan(ll_alt_i):
        llrs_s_i = np.nan
    else:
        llrs_s_i = (ll_alt_i - llrs_n_i)

    return beta_0_i,  beta_1_i,  ll_alt_i,  llrs_s_i


def loglik_bernoulli_logodds(x, p, q):
    '''
    Compute log likelihood of Bernoulli with a multiplicative log odds based probability model
    '''
    # make shapes the same
    if (isinstance(p, np.ndarray) or isinstance(p, pd.DataFrame)) and (isinstance(q, np.ndarray) or isinstance(q, pd.DataFrame)):
        if len(q.shape)>0:
            q = q.reshape(p.shape)
    # compute ll
    qp = q*p
    epsilon=1e-7
    ll_indiv = x*(np.log(epsilon+qp / (1-p+qp))) + (1-x)*(np.log(epsilon+ 1 - (qp / (1-p+qp))))
    ll = np.nansum(ll_indiv)

    if any(np.isnan(ll_indiv)):
        if verbose: print ('WARNING: there are',np.sum(np.isnan(ll_indiv)),'nan values in the loglikelihood of the bernoulli')
    
    return ll



def binary_search_q(T, p):
    '''
    Solve for \beta_0 by binary search. This code is adapted from Zhang and Neill 2017 bias scan code. This 
    actually computes q times the slope, which has the same sign as the slope and is monotonically decreasing with q

    Args:
        T (np.arr): T values in the neigh
        p (np.arr): p(x) values in the neigh
    Returns:
        q_best (float): ideal q [i.e. \beta_0]
    '''
    q_min = 1e-6
    q_max = 1e4

    while np.abs(q_max - q_min) > 0.000001:
        q_mid = (q_min + q_max)/2
        if np.sign(slope_given_q(T, p, q_mid)) > 0:
            q_min = q_min + (q_max - q_min) / 2
        else:
            q_max = q_max - (q_max - q_min) / 2
    
    q_best = (q_min + q_max) / 2

    return q_best

def slope_given_q(T, p, q):
    '''
    Helper function to binary_search_q()
    '''
    return np.sum(T - q*p / (1-p+q*p))


def GD_search_q(x, p, min_q, epsilon=1e-6, stop_cond=1e-2, alpha=0.1, bins=100, debug=False):
    '''
    Solve for \beta_0 by gradient descent
    Args:
        x (np.arr): actually is T
        p (np.arr): p(x) values in the neigh

    '''
    if min_q == 0:
        min_q = max(epsilon, np.sum(x) / np.sum(p / (1 - p + epsilon)))
    elif min_q == 1:
        min_q = max(1.0, np.sum(x) / np.sum(p))
    else:
        assert False, 'invalid min_q'

    q = min_q
    lls = np.ones(bins) * (-np.inf)
    qs = np.zeros(bins) * np.nan
    for ii in range(bins):
        # print 'q', q
        # too high q causes issue
        if q > 1e4:
            return 1e4

        qs[ii] = q
        lls[ii] = loglik_bernoulli_logodds(x, p, q)
        gradient = -np.sum(x / q - (p / (1 - p + q * p)))
        if gradient > 0:
            if ii==0: # special case of all zeros
                return q
            alpha /= 2
            if debug: print ('alpha', alpha)
            q = q_old
            continue
        else:
            q_old = q
            q = q - alpha * gradient
            if abs(q - q_old) < stop_cond: break

    if debug:
        plt.plot(qs, lls)
        plt.title('LLR as function of q')

    return qs[np.nanargmax(lls)]


    
def rand_testing(obs_model, T_fx, T_hat_master, k_samples, iters_rand, alpha, T, x, z, f_base, all_points, k):
    '''
    Rand testing, using local smoothing for unconstrained hetero model.

    Example usage:
        rand_testing(k_samples=100, iters_rand=10, alpha=0.05,
                     obs_model, x, z, f_base, all_points, k)
    
    Args:
        T_hat_master 
        k_samples (int): number of neighs to use for variance estimation
        iters_rand (int): number of randomization iterations
        alpha (float): level of signifcant test
        ...
    Return:
        llr_sig (float): value of singificant llr at alpha level
        llr_max_samples (np.arr): iters_rand*1 array of max llr values from rand tests
    '''
    # Init
    llr_max_samples = -1*np.ones((iters_rand,1))
    llr_all_samples = []

    # Run iter_rand iterations
    rr = 0
    while rr < iters_rand:

        # sample data using local empirical variance
        T_sample,_ = sample_T(T_fx, T_hat_master, T, x, z, k_samples, f_base, verbose=False)
        
        # run RDSS
        output = RDSS_residual_multik(obs_model, T_sample, x, z, f_base=f_base, all_points=all_points, ks=k, verbose=False, plotting=False)
        if len(output) == 1:
            if verbose: print ('Trying to redraw again...')  # todo make sure this doesnt continue in an infinite loop...
            continue
        llrs_sample = output[0] # we only need the first element

        # store results
        llr_max_samples[rr] = np.max(llrs_sample)
        llr_all_samples.append(llrs_sample)
        rr += 1

        # DEBUGGING: Plot the best
        #plot_neigh(x, z, out=T_sample, out_name='Treatment')
    
    # Get alpha level significance
    llr_sig = float( np.sort(llr_max_samples, axis=0)[-max(int(iters_rand*alpha),1)] )
    
    return llr_sig, llr_max_samples, llr_all_samples


def sample_T(T_fx, T_pred, T, x, z, k, f_base, verbose=False):
    '''
    Sample T from a model which only provides mean estimates. Use k-nn local smoothing to estimate variance
    Often used for rand testing.
    
    Sample usage:
        T_sample,T_var = sample_T(T_fx, T_pred, T, x, z, k=100, verbose=True)
    
    Args:
        T_pred (smf): predicted values, often from statsmodel
        T (np.arr): treatment, 1D
        x (pd.df): n*px+1 covariate values
        z ([]): pz index(es) of forcing variables
        k (int): number of nearest-neighbors for knn local smoothing
        verbose (bool): indicator for verbose plotting and output. (False)
    Return:
        T_sample (np.arr): sampled treatment
    '''

    # Regenerate mean and residual
    T_r = T - T_pred
    n = x.shape[0]

    # Estimate variance locally
    neighs, neighs_n = get_neighs('k_nn', x, z, k=k, verbose=False)
    T_var = np.ones(T_pred.shape) * -1
    for ii in range(neighs_n):
        T_var[ii] = np.std(T_r[neighs[ii]]) # todo std or var? I think std

    assert np.all(T_var >= 0), 'Negative variance. Either invalid or non-existant'

    # Sample
    T_sample = T_pred + (T_var * np.random.normal(size=T_pred.shape))

    # Binary
    if np.all((T==1) | (T==0)):
        D_T = np.random.rand(T_sample.shape[0], 1)
        T_sample = T_sample > D_T

    # Plot
    if verbose:
        plot_neigh(x, z, T_sample, 'T sample')
        
    return T_sample, T_var



