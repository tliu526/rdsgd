import pandas as pd
import numpy as np
import scipy as sp

from data_functions import *

#import matplotlib.pyplot as plt


def subset_neigh(neigh, subset):
    '''
    Subset a neigh without changing original array.
    Example use: subset_neigh(neighs[:,1], subsets[:,4])
    
    Args:
        neigh (np.arr): n*1 boolean array indicating neigh
        subset (np.arr): k*1 boolean array indicating subset within neigh
    Return:
        neigh_c (np.arr): n*1 boolean array indicating the intersection or neigh and subset
    '''
    neigh_c = np.copy(neigh)
    to_false = np.where(neigh_c)[0][~subset]
    neigh_c[to_false] = False
    
    return neigh_c


def get_neighs(neigh_type, x, z, k=None, verbose=False):
    '''
    Divide space into neighs.
    Args:
        neigh_type (str): descriptive string of neigh type {'inf_intervals','k_nn'}
        x (pd.df): n*px+1 covariate values
        z ([]): pz index(es) of forcing variables
        k (int): number of nearest-neighbors for knn. (None)
        verbose (bool): indicator for verbose plotting and output. (False)
    Return:
        neighs (np.arr): n*neighs_n boolean array where each col represents one neigh
        neighs_n (int): number of neighs
    '''
    (n,_) = x.shape
    
    if neigh_type == 'inf_intervals':
        # initialize
        neighs_n = n
        neighs = np.zeros((n,neighs_n))
        xz = x[z].values
        
        # neighs from [-inf, z_i)
        for ii in range(n):
            neighs[:,ii] = (xz < xz[ii]).reshape(n)
        
    elif neigh_type == 'k_nn':
        # initialize
        assert k>0, 'k must be a positive integer'
        neighs_n = n
        neighs = np.zeros((n,neighs_n))
        xz = x[z].values
        
        # neighs as the k-nn around a point, inclusive
        for ii in range(n):
            dist = sp.linalg.norm(xz - xz[ii], axis=1)
            k_nn = (-dist).argsort()[-k:][::-1]
            neighs[k_nn,ii] = 1
        
    else:
        assert False, 'Not a valid neigh_type'
        
    # make boolean
    neighs = neighs==1
        
    # TODO: post process to remove redundant neighs, but leave as option to user.
    if verbose: print (neighs.shape, neighs_n)

    return neighs, neighs_n


def get_vector_neighs(x, z, center_i, all_points=False, verbose=False):
    '''
    Divide space into neighs using the vector between the center and each other point.
    Args:
        x (pd.df): n*px+1 covariate values
        z ([]): pz index(es) of forcing variables
        center_i (int): index of center point
        all_points (bool): indicator to translate vectors to all points, leading to O(k^2) subsets. (False)
        verbose (bool): indicator for verbose plotting and output. (False)
    Return:
        neighs (np.arr): n*neighs_n boolean array where each col represents one neigh
        neighs_n (int): number of neighs
    '''
    (n, p) = x.shape

    # distance vector from center to each point
    dist_center = x.values[:, z] - x.values[center_i, z]  # from all points to center

    # all pivot points translate or just from center
    if all_points:
        pivots = range(n)
        neighs_n = n * n
    else:
        pivots = [center_i]
        neighs_n = n

    # initialize
    neighs = np.zeros((n, neighs_n))
    angles = np.zeros((n, neighs_n))

    for pivot in pivots:
        dist_pivot = x.values[:, z] - x.values[pivot, z]  # from all points to pivot (pivot is often the center)
        dist_pivot_norm = np.linalg.norm(dist_pivot)

        for ii in range(n):
            dist_ii = dist_center[ii, :]

            # Special same if distance to same point. Use first basis vector.
            if np.all(dist_ii == 0):
                dist_ii[0] = 1

            # compute the angle for all points
            angle = np.arccos(np.dot(dist_center, dist_pivot[ii, :]) / \
                              (np.linalg.norm(dist_center, axis=1) * np.linalg.norm(dist_pivot[ii, :])))

            # angle within [0,2pi]
            angles[:, ii] = (angle % (2 * np.pi))
            quadrant = angles[:, ii] / (np.pi / 2)

            # compute which side of the threshold each point is on
            neighs[:, ii] = (1 <= quadrant) | (quadrant >= 2)

    # TODO: post process to remove redundant neighs?

    neighs = neighs == 1
    return neighs, neighs_n, dist_pivot, angles



def project_points(x, z, center_i, pivot, plotting=False):
    '''
    Project points to vector given the angles provided.

    Not sure if this works for pivots other than center...
    '''

    center_i = int(center_i)
    dist_center = x.values[:, z] - x.values[center_i, z]  # from all points to center

    angles = np.arccos(np.dot(dist_center, pivot) / \
                       (np.linalg.norm(dist_center, axis=1) * np.linalg.norm(pivot)))
    angles = (angles % (2 * np.pi))

    projected = np.cos(angles) * np.linalg.norm(dist_center, axis=1)

    if plotting:
        plot_neigh(x, z, out=np.linalg.norm(dist_center, axis=1), out_name='|a|');
        print (x.values[center_i, z])
        plot_neigh(x, z, out=angles, out_name='angle')
        plot_neigh(x, z, out=np.cos(angles), out_name='cos(angle)')
        plot_neigh(x, z, out=projected, out_name='projected')

    return projected
