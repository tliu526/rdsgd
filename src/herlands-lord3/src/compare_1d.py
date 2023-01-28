import pandas as pd
import numpy as np
import scipy as sp
import os
import math
import time
import pickle as pkl
import datetime

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

from data_functions import *
from subset_functions import *
from model_functions import *
from search_functions import *
from analysis_functions import *

import matplotlib.pyplot as plt

chgpntR = importr('changepoint')
ecpR = importr('ecp')
ecpR = importr('cpm')

def Rcpt_xcut(x,cptresult):
    '''
    Get the x value at which a changepoint method cuts
    '''
    cut_idx = int(cptresult)
    if cut_idx < x.shape[0]:
        return x.loc[cut_idx][0]
    else:
        return np.nan

def compare_LoRD(obs_model, f_base, T, x, z, k):
    llrs, neighs, subsets_best, beta_0_n, beta_1_n, T_fx, llrs_n, llrs_a, centers_n, angles_best = \
    RDSS_residual(obs_model, T, x, z, f_base=f_base, subset_type='vector', all_points=False, k=k, verbose=False, plotting=False)
    max_arg_lordn = np.nanargmax(llrs)
    return np.array(x)[max_arg_lordn][0]
    
np.random.seed(49)

#######################################################
#######################################################
data_type='cont'; z_effs = np.linspace(0,2.5,10); exps = 50
#data_type='binary';  z_effs = np.linspace(0,8,5); exps = 25
n = 1000
#######################################################
#######################################################

date_now = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
poly_xT=1
discont_type='square' #'linear' # 
f_yT = 'OLS_poly1'; k=50;

# Init data structures
if data_type=='cont':     methods = ['True','LoRD Normal-1','LoRD Normal-2','LoRD Normal-4',                                                         'AMOC mean', 'AMOC var', 'AMOC meanvar','Student-t','Bartlett','Mann-Whitney','Kolmogorov-Smirnov']
elif data_type=='binary': methods = ['True','LoRD Normal-1','LoRD Normal-2','LoRD Normal-4','LoRD Bernoulli-1','LoRD Bernoulli-2','LoRD Bernoulli-4','AMOC mean', 'AMOC var', 'AMOC meanvar','Student-t','Bartlett','Mann-Whitney','Kolmogorov-Smirnov']
methods = ['True', 'AMOC meanvar']
results = np.zeros((exps, len(z_effs), len(methods)))


for exp_i in range(exps):
    for z_eff_i, z_eff in enumerate(z_effs):
        print ('exp_i #', exp_i, 'z_eff_i #', z_eff_i)
        method_i = 0

        x,y,z,T, D_z, z_eff, beta_y_T, discont = data_synthetic(data_type=data_type, n=n, px=1, pz=1, z_eff=z_eff,  poly_xT=poly_xT, discont_type=discont_type, compare1d=True, verbose=False)
        #plt.scatter(x,T); plt.show()
        if 'True' in methods:
            results[exp_i,z_eff_i,method_i] = D_z[0]
            method_i += 1

        # LoRD models
        if 'LoRD Normal-1' in methods:
            results[exp_i,z_eff_i,method_i] = compare_LoRD('normal', 'OLS_poly1', T, x, z, k)
            method_i += 1
        if 'LoRD Normal-2' in methods:
            results[exp_i,z_eff_i,method_i] = compare_LoRD('normal', 'OLS_poly2', T, x, z, k)
            method_i += 1
        if 'LoRD Normal-4' in methods:
            results[exp_i,z_eff_i,method_i] = compare_LoRD('normal', 'OLS_poly4', T, x, z, k)
            method_i += 1

        if 'LoRD Bernoulli-1' in methods:
            results[exp_i,z_eff_i,method_i] = compare_LoRD('bernoulli', 'Logit_poly1', T, x, z, k)
            method_i += 1
        if 'LoRD Bernoulli-2' in methods:
            results[exp_i,z_eff_i,method_i] = compare_LoRD('bernoulli', 'Logit_poly2', T, x, z, k)
            method_i += 1
        if 'LoRD Bernoulli-4' in methods:
            results[exp_i,z_eff_i,method_i] = compare_LoRD('bernoulli', 'Logit_poly4', T, x, z, k)
            method_i += 1

        ro.globalenv['x'] = x
        ro.globalenv['T'] = T
        
        if 'AMOC mean' in methods:
            cptmean = R('cpt.mean(T[1:1000], penalty=\"BIC\", method=\"AMOC\", class=FALSE)')
            results[exp_i,z_eff_i,method_i] = Rcpt_xcut(x,cptmean[0])
            method_i += 1

        if 'AMOC var' in methods:
            cptvar = R('cpt.var(T[1:1000], penalty=\"BIC\", method=\"AMOC\", class=FALSE)')
            results[exp_i,z_eff_i,method_i] = Rcpt_xcut(x,cptvar[0])
            method_i += 1

        if 'AMOC meanvar' in methods:
            cptmeanvar = R('cpt.meanvar(T[1:1000], penalty=\"BIC\", method=\"AMOC\", class=FALSE)')
            results[exp_i,z_eff_i,method_i] = Rcpt_xcut(x,cptmeanvar[0])
            method_i += 1

        if 'Student-t' in methods:
            student_t = R('detectChangePoint(T, \"Student\", ARL0=1000, startup=20)$changePoint')
            results[exp_i,z_eff_i,method_i] = Rcpt_xcut(x, student_t[0])
            method_i += 1

        if 'Bartlett' in methods:
            bartlett = R('detectChangePoint(T, \"Bartlett\", ARL0=1000, startup=20)$changePoint')
            results[exp_i,z_eff_i,method_i] = Rcpt_xcut(x, bartlett[0])
            method_i += 1

        if 'Mann-Whitney' in methods:
            mann_whitney = R('detectChangePoint(T, \"Mann-Whitney\", ARL0=1000, startup=20)$changePoint')
            results[exp_i,z_eff_i,method_i] = Rcpt_xcut(x, mann_whitney[0])
            method_i += 1

        if 'Kolmogorov-Smirnov' in methods:
            k_s = R('detectChangePoint(T, \"Kolmogorov-Smirnov\", ARL0=1000, startup=20)$changePoint')
            results[exp_i,z_eff_i,method_i] = Rcpt_xcut(x, k_s[0])
            method_i += 1

dir_results = '../results'
filename = os.path.join(dir_results, "exp_" + date_now)
pkl.dump([methods, results, exps, z_effs], open(filename + ".p", "wb"))