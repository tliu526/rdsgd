import pandas as pd
import numpy as np
import os
import json
from model_functions import *

# TODO: remove any pandas strcutres. Just import via pandas but immediately change to numpy.

def data_real(data_type, file_json, subsample=False, plotting=False, verbose=True):
    '''
    Load real data from a csv using the instructions in file_json
    Example usage:
        dir_data = 'AcademicProbation_LSO_2010'
        file_json = 'data_inst.json'
        x, y, z, T, x_cols, inst = data_real(dir_data, file_json)
    
    Args:
        data_type (str): directory of data under /data
        file_json (str): filepath for json instructions under /data/<dir_data>/
        subsample (int): number of entries to subsample, False if full sample. (False)
    Return:
        x (pd.df): n*px+1 covariate values
        y (np.arr): n*1 outcome variable
        z ([]): pz index(es) of forcing variables
        T (np.arr): n*1 T values
        x_cols ([]): name of x columns
        inst ({}): instructions from json
    '''
    # create full data dir path
    dir_data = os.path.join('../data',data_type)

    # Process with json instructions
    with open(os.path.join(dir_data, file_json), 'r') as fp:
        inst = json.load(fp)

    # Get the data
    df = pd.read_csv(os.path.join(dir_data, inst['file'][0]))

    # Remove missing data in any of x, y, or T
    df = df[inst['x'] + inst['y'] + inst['T']]
    if verbose: print ('data originally of size', df.shape)
    df = df.dropna()
    if verbose: print ('dropping missing data', df.shape)

    n = df.shape[0]

    # Create x
    x = df[inst['x']]
    x_cols = list(x.columns)
    x.columns = range(len(x_cols))
    x, x_cols = categorize_x(x, x_cols, inst) # make categorical variables

    # Create y
    y = df[inst['y']] # np.array(df[inst['y'][0]])

    # Create T
    T = np.array(df[inst['T'][0]])
    T = T.reshape(n, 1)

    # Subample data (optional)
    if subsample:
        pp = np.random.permutation(x.shape[0])
        x = x.loc[pp < subsample, :].reset_index(drop=True)
        y = y.loc[pp < subsample, :].reset_index(drop=True)
        T = T[pp < subsample]
        if verbose: print ('data subsamped to', x.shape)
    else:
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)

    # Create z
    z = []
    for z_i in inst['z']:
        z.append(x_cols.index(z_i))

    # Discont
    if (data_type == 'test_score_2012') | (data_type == 'AcademicProbation_LSO_2010'):
        discont = T==1
    else:
        discont = np.squeeze(np.zeros(y.shape) == 0)  # fake discont # todo make np.nan

    return x, y, z, T, x_cols, inst, discont


def categorize_x(x, x_cols, inst):
    '''
    Convert columns of binary/categorical to integers using 1 hot encoding. We use the native pandas
    function called pd.get_dummies. This function adjusts x_cols apropriately.

    Args:
        x (pd.df): n*px+1 covariate values
        x_cols ([]): name of x columns
        inst ({}): instructions from json
    Return:
        x (pd.df): updated with categorical transformations 
        x_cols_new ([]): updated x_cols with categorical cols
    '''
    x_cols_new = []
    for col, coltype in enumerate(x.dtypes):

        # real valued (or already binary/categorical), add to x
        if ('int' in coltype.name) or ('float' in coltype.name):
            x_cols_new.append(x_cols[col]) 

        # binary/categorical
        else:
            print ('binary/categorical col #', col)
            if x_cols[col] in inst['z']:
                assert 'z contains a categorical or binary variable'

            x_dummies = pd.get_dummies(x[col])
            del x[col]
            for cc in x_dummies.columns:
                x.insert(col, 'added_'+str(col)+'_'+str(cc), x_dummies[cc])
                x_cols_new.append('added_'+str(col)+'_'+str(cc)) # add to x

    # Reindex x columns
    x.columns = range(len(x_cols_new))
    
    return x, x_cols_new


def fuzzy_binary_T(T, fuzzy):
    '''
    Make a binary treatment fuzzy

    Args:
        T (np.arr): n*1 T values
        fuzzy (float): fuzziness parameter \in [0.5, 1.0]
    Return:
         T (np.arr): n*1 T values
    '''
    # Confirm T is binary
    assert np.all((T==1) | (T==0)), 'T must be binary'
    assert (fuzzy>=0) and (fuzzy<=1), 'specify fuzzy \in (0,1)'

    # Make fuzzy
    T = T.astype(float)
    T[T == 1] = fuzzy
    T[T == 0] = 1-fuzzy
    D_T = np.random.rand(*T.shape)
    T = T>D_T

    return T


def sigmoid(x):
    '''
    Helper fuction for 1D sigmoid
    '''
    return 1 / (1 + np.exp(-x))


def data_synthetic(data_type, n, px, pz=1, z_eff=0.3, beta_y_T=5.0, poly_xT=1, discont_type='square', RPD=0, compare1d=False, plotting=False, verbose=False):
    '''
    Generate synthetic data with subbstantial customization.

    Args:
        data_type (str): type of T data. 'binary'= T \in {0,1}. 'cont'= T \in [0,1].
        n (int): number of observations
        px (int): number of observable covariates
        pz (int): number of forcing variables (only 1 or 2 for now)
        z_eff (float): additive shift in p(T|z)
        beta_y_T (float): additive shift in p(y|T)
        poly_T (int): polinomial order for x w.r.t. T
        discont_type (str): shape of discontinuity when pz>1 {'square','linear'}. ('square')
        verbose (bool): indicator for verbose plotting and output. (False)
    Return:
        x (pd.df): n*px+1 covariate values
        y (np.arr): n*1 outcome variable
        z ([]): pz index(es) of forcing variables
        T (np.arr): n*1 T values
        D_z ([]): pz threshold(s) for forcing varibales
        z_eff (float): additive shift in p(T|z). User defined.
        beta_y_T (float): additive shift in p(y|T). User defined.
    '''

    D_z = np.random.rand(pz)*1.5-0.75  # dummy discontinuity for z \in [0.1, 0.9]
    x = np.random.rand(n,px)*2-1 # observables in [-1,1]
    u = (np.random.rand(n, 1)*2-1 ) * 2#px  # unobservables
    z = range(pz) # forcing variable, index(es)
    D_T = np.random.rand(n,1) #dummy treshold for T

    if compare1d:
        x = np.linspace(-1,1,n).reshape(n,1)

    # Poly
    if verbose: print ('poly:', poly_xT)
    x_in = poly_x_create(pd.DataFrame(x), poly_xT, const=False)

    
    gamma = np.random.normal(size=(px*poly_xT,1))
    epsilon_T = np.random.normal(size=(n,)) * np.array(0.0+np.mean(x_in,axis=1)) # x[:,z]
    epsilon_T = epsilon_T.reshape((n, 1))
    
    beta_y_x = np.random.normal(size=(px,1))
    epsilon_y = np.random.normal(size=(n,)) * np.array(0.0+np.mean(x_in,axis=1))
    epsilon_y = epsilon_y.reshape((n, 1))

    
    # Define discontinuity shape
    if (pz==1) or (discont_type=='square'):
        discont = np.all(x[:,z]>D_z,axis=1)

        dist_from_discont = np.min(abs(x[:,z] - D_z), axis=1)

        if len(z)>2:
            assert (False, 'Not yet implemented')
            #discont = np.all(x[:,[0,1]]>D_z[0:2],axis=1) # todo fix this.

        # compute distance from discont
        delta = x[:,z] - D_z
        bottom_left = np.all(np.sign(delta)<0, axis=1) # quadrant 3 use hypotenus
        dist_from_discont = abs(np.min(delta, axis=1))
        dist_from_discont[bottom_left] = np.sqrt(np.sum((D_z-x[:,z])**2, axis=1))[bottom_left]

        
    elif discont_type=='linear':
        lower_p = max(np.percentile(x[:,z], 20, axis=0))
        upper_p = min(np.percentile(x[:,z], 80, axis=0))
        b = (upper_p-lower_p) * np.random.rand() + lower_p
        m = np.random.normal(size=(2,1))
        discont = (np.dot(x[:,z],m) + b) < 0
        
    elif 'poly' in discont_type:
        poly = int(discont_type.split('poly')[1])
        discont = np.zeros((x.shape[0],1)) # init
        for pp in range(poly+1):
            beta_discont = np.random.normal(size=(pz,1))
            discont += np.dot(x[:,z]**pp, beta_discont)
        discont = np.all(discont > np.mean(discont), axis=1)
        
    else:
        assert False, 'Invalid discont_type' + str(discont_type)

        
    # Define Treatment data type
    if data_type == 'binary': # {0,1}
        DT_u = discont * np.exp(z_eff/2.0)
        DT_l = ~discont * np.exp(-z_eff/2.0)

        mu = DT_u.reshape((n, 1)) + DT_l.reshape((n, 1))

        # probability of treatment
        p = sigmoid(np.dot(x_in,gamma) + epsilon_T + u)

        # add discontinuity
        p = mu*p / (1-p+mu*p)
        T = np.random.binomial(1, p)

    elif data_type == 'cont': # [0,1]

        # treatment
        T = np.dot(x_in, gamma) + epsilon_T + u

        #  add discontinuity
        if RPD==0:
            DT_u = discont * (z_eff / 2.0)
            DT_l = ~discont * (z_eff / 2.0)
            mu = DT_u.reshape((n, 1)) - DT_l.reshape((n, 1))
            T += mu
        else:
            print (dist_from_discont.shape, discont.shape, T.shape)
            T += (((dist_from_discont ** RPD) * z_eff) * discont).reshape(n,1)

    else:
        assert False, 'Invalid data_type' + str(data_type)

    #y = np.dot(x, beta_y_x) + epsilon_y + beta_y_T * T  # outcome
    y = np.dot(x, beta_y_x) + epsilon_y + beta_y_T * T  + u # outcome
    #y =                        epsilon_y + beta_y_T * T + u  # outcome
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    
    if plotting:
        # Plot discontinuities        
        import matplotlib.pyplot as plt
        plot_neigh(x, z, out=T, out_name='Treatment', neigh=None, D_z=D_z, z_eff=z_eff)
        plot_neigh(x, z, out=y, out_name='Treatment', neigh=None, D_z=D_z, z_eff=z_eff)
    
    return x, y, z, T, D_z, z_eff, beta_y_T, discont


def random_uniform_scaled(a, b, size):
    out = np.zeros(size)
    for ii in range(size[0]):
        out[ii] = (b[ii]-a[ii]) * np.random.random_sample() + a[ii]
    return out

def normalize_xz(x, z):
    '''
    Normalize the columns of x[z]
    '''
    x_means = np.mean(x[z]).values
    x_stds = np.std(x[z]).values
    for ii, z_i in enumerate(z):
        x[z_i] -= x_means[ii]
        x[z_i] /= x_stds[ii]
    return x, x_means, x_stds

def unnormalize_xz(x, z, x_means, x_stds):
    '''
    Unnormalize the columns of x[z]
    '''
    for ii, z_i in enumerate(z):
        x[z_i] *= x_stds[ii]
        x[z_i] += x_means[ii]
    return x


def plot_neigh(x, z, out, out_name, neigh=None, D_z=None, z_eff=None, x_names=[]):
    '''
    Plot data with the option of also plotting neigh locations

    Args:
        x (pd.df): n*px+1 covariate values
        z ([]): pz index(es) of forcing variables
        out (np.arr): n*1 outcome variable. Usually either T or y
        out_name (str): descriptive name of out
        neigh (np.arr): n*1 boolean indicator of neigh membership. (None)
        D_z ([]): pz threshold(s) for forcing varibales. Used for descriptive title. (None)
        z_eff (float): additive shift in p(T|z). Used for descriptive title. (None)
        x_names ([string]): names of the axes corresponding to x. Empty list yeilds no axis names. ([])
    Return: 
        None
    '''

    import matplotlib.pyplot as plt

    # Set title
    if (D_z is not None) and (z_eff is not None):
        title_str = out_name + ': Discontinuity at '+str(D_z)+' of '+str(z_eff)
    else:
        title_str = out_name

    # Plot 1 dimensional discontinuity
    if len(z) == 1:

        plt.scatter(x[z], out)
        plt.title(title_str)
        plt.xlabel(z); plt.ylabel(out_name);
        if neigh is not None:
            if len(neigh) == 2:
                plt.scatter(x[z][neigh[0]], out[neigh[0]], s=50, edgecolor='r', facecolors='none', linewidths=2)
                plt.scatter(x[z][neigh[1]], out[neigh[1]], s=50, edgecolor='m', facecolors='none', linewidths=2)
            else:
                plt.scatter(x[z][neigh], out[neigh], edgecolor='r', facecolors='none', linewidths=2)
        plt.show()

    # Plot 2 dimensional discontinuity
    elif len(z) == 2:
        plt.scatter(x[z[0]], x[z[1]], c=np.array(out).astype(float).flatten(), s=50)
        plt.title(title_str)
        #plt.xlabel(z[0]); plt.ylabel(z[1]);
        if len(x_names)>0:
            plt.xlabel(x_names[0]); plt.ylabel(x_names[1]);
        if neigh is not None:
            if len(neigh) == 2:
                plt.scatter(x[z[0]][neigh[0]], x[z[1]][neigh[0]], s=50, edgecolor='r', facecolors='none', linewidths=2)
                plt.scatter(x[z[0]][neigh[1]], x[z[1]][neigh[1]], s=50, edgecolor='m', facecolors='none', linewidths=2)
            else:
                plt.scatter(x[z[0]][neigh], x[z[1]][neigh], s=50, edgecolor='r', facecolors='none', linewidths=2)
        plt.show()
        
    # Plot >2 dimensional discontinuity
    elif len(z) > 2:
        # Iteratively plot z[0] and z[i]
        for z_ii in range(1,len(z)):
            plt.scatter(x[z[0]], x[z[z_ii]], c=np.array(out).astype(float).flatten(), s=50)
            plt.title(title_str)
            #plt.xlabel(z[0]); plt.ylabel(z[z_ii]);
            if len(x_names)>0:
                plt.xlabel(x_names[0]); plt.ylabel(x_names[z_ii]);
            if neigh is not None:
                if len(neigh) == 2:
                    plt.scatter(x[z[0]][neigh[0]], x[z[z_ii]][neigh[0]], s=50, edgecolor='r', facecolors='none', linewidths=2)
                    plt.scatter(x[z[0]][neigh[1]], x[z[z_ii]][neigh[1]], s=50, edgecolor='m', facecolors='none', linewidths=2)
                else:
                    plt.scatter(x[z[0]][neigh], x[z[z_ii]][neigh], s=50, edgecolor='r', facecolors='none', linewidths=2)
            plt.show()
        
    
