import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.feasible_gls import GLSHet, GLSHet2
import statsmodels.discrete.discrete_model as smd
import statsmodels.nonparametric.kernel_regression as smk
import pickle
import os
import time

## R function connections
#import pandas.rpy.common as common
#from rpy2.robjects.packages import importr
#from rpy2 import robjects as ro
#from rpy2.robjects import pandas2ri
#base = importr('base')
#stats = importr('stats')
#MASS = importr('MASS')
#robust = importr('robust')
#pandas2ri.activate()
#R = ro.r


def basic_fit(y, x, f_base, x2=pd.DataFrame(), verbose=False):
    '''
    Wrapper function for basic model fitting
    Args:
        y: output
        x: inputs
        f_base (str): specifies the type of model
        x2: optional additional inputs that are strictly linear. (False)
    Return:
        fx (smf): the return value of f_base.fit()
    '''

    if verbose: print ('basic_fit', f_base, x2.empty)

    if f_base == 'OLS':
        x_in = poly_x_create(x, 1, const=True)
        if x2.empty==False:
            x_in = pd.concat([pd.DataFrame(x2), x_in], axis=1)
        fx = sm.OLS(endog=y, exog=x_in).fit()

    elif 'OLS_poly' in f_base:
        poly = int(f_base.split('OLS_poly')[1])
        x_in = poly_x_create(x, poly, const=True)
        if x2.empty==False:
            x_in = pd.concat([pd.DataFrame(x2), x_in], axis=1)
        fx = sm.OLS(endog=np.array(y), exog=np.array(x_in)).fit() # TODO ensure no np casting needed.

    elif f_base == 'kernel': 
        x_in = poly_x_create(x, 1, const=True)
        if x2.empty==False:
            x_in = pd.concat([pd.DataFrame(x2), x_in], axis=1)
        # TODO: var_type is dependant on the imputs -- we to change that to consider binary variables as 'u'
        fxsmk = smk.KernelReg(endog=y, exog=x_in, var_type='c'*x_in.shape[1], bw='cv_ls', reg_type='lc') # for local linear use 'll'
        fx = [fxsmk, fxsmk.fit()[0], fxsmk.bw]

    elif 'Logit_poly' in f_base:
        poly = int(f_base.split('Logit_poly')[1])
        x_in = poly_x_create(x, poly, const=True)
        if x2.empty==False:
            x_in = pd.concat([pd.DataFrame(x2), x_in], axis=1)
        fx = smd.Logit(endog=np.array(y), exog=np.array(x_in)).fit()
        fittedvalues = fx.predict(x_in)
        fx = [fx, fittedvalues]

    elif 'RLM_poly' in f_base:
        poly = int(f_base.split('RLM_poly')[1])
        x_in = poly_x_create(x, poly, const=True)
        if x2.empty==False:
            x_in = pd.concat([pd.DataFrame(x2), x_in], axis=1)
        fx = sm.RLM(endog=np.array(y), exog=np.array(x_in)).fit()

    elif 'robust_poly' in f_base:
        poly = int(f_base.split('robust_poly')[1])
        x_in = poly_x_create(x, poly, const=False)
        if x2.empty == False:
            x_in = pd.concat([pd.DataFrame(x2), x_in], axis=1)
        y_in = pd.DataFrame(y)
        y_in.columns = ['Y']
        x_in.columns = range(x_in.shape[1])
        df_in = pd.concat([y_in, x_in], axis=1)
        formula = 'Y~'
        for col_i, col in enumerate(x_in.columns):
            if col_i==0: formula += 'X' + str(col)
            else:        formula += '+X' + str(col)
        if verbose: print ('formula', formula)

        #print 'df_in', df_in
        fx = R.lmRob(formula, data=df_in)

        if verbose: print ('Adding additional values')
        ro.globalenv['fx_r'] = fx
        ro.globalenv['df_r'] = df_in
        fittedvalues = R('predict(fx_r, df_r)')
        coefs = R('coef(fx_r)')
        #coefs = R('summary(fx_r)$coefficients[, 1]')
        if verbose: print ('coefs', coefs)
        stderrs = R('sqrt(diag(fx_r$cov))')
        #stderrs = R('summary(fx_r)$coefficients[, 2]')
        if verbose: print ('stderrs', stderrs)
        fx = [fx, fittedvalues, coefs, stderrs]


    elif 'robustLog_poly' in f_base:
        poly = int(f_base.split('robustLog_poly')[1])

        x_in = poly_x_create(x, poly, const=False)
        if x2.empty == False:
            x_in = pd.concat([pd.DataFrame(x2), x_in], axis=1)

        y_in = pd.DataFrame(y)
        y_in.columns = ['Y']
        x_in.columns = range(x_in.shape[1])
        df_in = pd.concat([y_in, x_in], axis=1)
        formula = 'Y~'
        for col_i, col in enumerate(x_in.columns):
            if col_i==0: formula += 'X' + str(col)
            else:        formula += '+X' + str(col)
        if verbose: print ('formula', formula)

        fx = R.glmRob(formula, data=df_in)

        if verbose: print ('Adding additional values')
        ro.globalenv['fx_r'] = fx
        ro.globalenv['df_r'] = df_in
        fittedvalues = R('predict.lmRob(fx_r, df_r)')
        coefs = R('coef(fx_r)')
        #coefs = R('summary(fx_r)$coefficients[, 1]')
        if verbose: print ('coefs', coefs)
        stderrs = R('sqrt(diag(fx_r$cov))')
        #stderrs = R('summary(fx_r)$coefficients[, 2]')
        if verbose: print ('stderrs', stderrs)
        fx = [fx, fittedvalues, coefs, stderrs]

    else:
        assert False, 'Undefined f_base'

    if verbose:
        print ('fitted model!')

    return fx


def poly_x_create(x, poly, const=False):
    '''
    Create df with additional columns representing polynomial functions of x. Optionally add a column for a constant.
    Args:
        x: inputs
        poly (int): polynomial (such that poly > 0) of x to construct
        const (bool): option to add a constant column [False]
    Return:
        x_out: x with appended columns
    '''
    assert poly > 0, 'Poly must be positive integer'

    x_out = pd.concat([x.pow(i) for i in range(1, poly + 1)], axis=1)

    # Add ones to start
    if const:
        x_out = pd.concat([pd.DataFrame(np.ones(x.shape[0])), x_out], axis=1)
    return x_out


def get_pred_mean(fx, f_base, x=None, verbose=False): 
    '''
    Retrive the predicted mean of a functional model

    '''
    # todo perhaps use .predict for all the statsmodels functions? Might be a bug in .fittedvalues (such as for logit)
    if verbose: print ('get_pred_mean()', f_base)

    if 'Logit_poly' in f_base:
        pred_mean = np.copy(fx[1])

    elif 'robust_poly' in f_base:
        pred_mean = np.copy(fx[1])

    elif 'robustLog_poly' in f_base:
        pred_mean = np.copy(fx[1])

    elif 'kernel' in f_base:
        pred_mean = np.copy(fx[1])

    else:
        pred_mean = np.copy(fx.fittedvalues)

    return pred_mean.reshape(pred_mean.shape[0],1)


def get_pred_variable(fx, f_name, varnum=0):
    if 'robust_poly' in f_name:
        T_eff = fx[2][varnum+1] # TODO change index to dict indexing
        T_bse = fx[3][varnum+1] 
    elif 'robustLog_poly' in f_name:
        T_eff = fx[2][varnum+1] 
        T_bse = fx[3][varnum+1] 
    elif 'Logit_poly' in f_name:
        T_eff = np.array(fx[0].params)[0]
        T_bse = np.array(fx[0].bse)[0]
    else:
        T_eff = fx.params[varnum]
        T_bse = fx.bse[varnum]
    return T_eff, T_bse



def compare_models(T, x, f_bases=[], criterion='BIC', verbose=False):
    '''
    Compare f_bases models using a criterion, such as BIC or AIC.
    Currently only applicable to statsmodels with native bic or aic. Does not consider heteroskeasticity.

    Example usage:
        compare_models(T, x, f_bases=['OLS_poly1','OLS_poly5','OLS_poly10','OLS_poly50'], criterion='BIC')

    Args:
        T
        x
        f_bases ([str]): names of models
        criterion (str): name of critereon to evaluate models {'BIC','AIC'}
        verbose (str): verbose output
    Return:
        f_best (srt): name of model with best scoring criterion
        scores (np.arr): score of criterion for each model
    '''

    T = T.astype(float)
    scores = np.ones(len(f_bases))*np.inf

    for f_idx, f_base in enumerate(f_bases):
        
        # Fit model    
        try:
            T_fx = basic_fit(T, x, f_base, verbose=verbose)
        except:
            print ('WARNING: Failed to bacis_fit() with f_base:', f_base, 'Providing', criterion, 'of inf. Continuing comparison.')
            continue

        # Evaluate model
        if criterion == 'BIC':
            scores[f_idx] = T_fx.bic
        elif criterion == 'AIC':
            scores[f_idx] = T_fx.aic
        else:
            assert False, 'criterion ' + criterion + ' not yet implemented.'

    best_f = f_bases[np.argmin(scores)]
    return best_f, scores

