"""
Implements subgroup RDD discovery procedure.
"""
from contextlib import redirect_stderr
from econml.dml import CausalForestDML
from econml.cate_interpreter import SingleTreeCateInterpreter, SingleTreePolicyInterpreter
from linearmodels.iv.model import _OLS
import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf

# user imports
import sys
sys.path.append("../")

def test_discontinuity(df, cutoff, running='x', treat='t', bw=None, kernel="triangular"):
    """
    Tests whether there is a discontinuity present at the cutoff via first stage
    regression.
    
    Assumes a uniform kernel.
    
    Args:
        df (pd.DataFrame): dataframe of candidate data
        running (str): the running variable column
        treat (str): the treatment variable column
    Returns
        (OLSResults, bw) of first stage regression
    """
    # exclude data and generate weights
    if bw is not None:
        sel_df = df[(df[running] < cutoff + bw) & (df[running] > cutoff - bw)].copy()
        weights = pd.Series([1] * sel_df.shape[0])
    else:
        sel_df, bw, weights = gen_bandwidth(df, cutoff=cutoff, running=running, treat=treat, kernel=kernel)

        if sel_df is None:
            return (None, np.nan, np.nan)
    
    # generate covariates needed for first stage regression
    sel_df['z'] = (sel_df[running] >= cutoff).astype(int)
    sel_df['x_lower'] = (1-sel_df['z'])*(sel_df[running] - cutoff) # adjusted x for 2SLS
    sel_df['x_upper'] = sel_df['z']*(sel_df[running] - cutoff) # adjusted x for 2SLS

    ols_formula = f"{treat} ~ 1 + x_lower + x_upper + z"
    try:
        ols_results = _OLS.from_formula(ols_formula, sel_df, weights=weights).fit(cov_type="robust")
        return ols_results, sel_df.shape[0], bw
    
    except Exception as e:
        print(e)
        return (None, np.nan, np.nan)


def first_stage_discovery(df, running_cols, grid_dict=None, treat='t', grid_size=50, alpha=0.01, kernel="triangular", bw=None):
    """
    Runs first stage threshold discovery.
    
    Implements Algorithm A.1.

    Args:
        df (pd.DataFrame): input dataFrame
        running_cols (list): a list of the candidate running variable column names
        grid_dict (dict): optionally provide a grid of cutoffs for each running var
        treat (str): the treatment variable column name
        grid_size (int): the grid size for running variable cutoffs, if grid_dict is not provided
        alpha (float): the nominal FPR, defaults to 0.01
        bw (float): optionally provide a fixed bandwidth
    Returns:
        dict: running_var:sig_results k:v pairs, where sig_results is a dict of cutoff:(OLSResults, n_included, bw)
    """
    running_dict = {}

    for running in running_cols:
        if grid_dict and running in grid_dict:
            cutoffs = grid_dict[running]
        else:
            cutoffs = np.linspace(df[running].min(),
                                  df[running].max(),
                                  grid_size)
        results = []
        # Step 1
        for cutoff in cutoffs:
            # Steps 1a - 1c
            # (OLS_results, n_incl)
            result_tup = test_discontinuity(df, 
                                            cutoff=cutoff,
                                            running=running,
                                            treat=treat,
                                            kernel=kernel,
                                            bw=bw)

            results.append(result_tup)

        # Step 2
        pvals = [res[0].pvalues['z'] if res[0] is not None else np.nan for res in results]
        reject, _ = pg.multicomp(pvals, method="bonf", alpha=alpha)

        # Step 3
        sel_results = {cutoff: results[i] for i, cutoff in enumerate(cutoffs) if reject[i]}
        
        running_dict[running] = sel_results
    
    return running_dict


def gen_bandwidth(df, cutoff, running='x', treat='t', kernel="triangular"):
    """
    Generates all necessary bandwidth information. Calls the more robust R "rdd" package.
    
    Args:
        df (pd.DataFrame): the input data
        cutoff (float): the cutpoint for the running variable
        running (str): the running variable col
        treat (str): the treatment variable col
        kernel (str): optionally provide an alternative kernel (choices are: "rectangular", "triangular")
    
    Returns:
        tuple: (sel_df (pd.df), bw (float), weights (list)), where sel_df is the selected dataframe with a  column
    """
    with redirect_stderr(None):
        try:
            bw = optimal_bandwidth(X=df[running], Y=df[treat], cut=cutoff)
        
        except Exception as r_err:
            return None, None, None
    
    # exclude data
    sel_df = df[(df[running] < cutoff + bw) & (df[running] > cutoff - bw)].copy()
    
    # default to rectangular kernel
    weights = pd.Series([1] * sel_df.shape[0])
    
    if kernel == "triangular":
        weights = (1 - np.abs((sel_df[running] - cutoff) / bw)).reset_index(drop=True)
    
    return sel_df, bw, weights


def optimal_bandwidth(X, Y, cut=0):
    """
    Forked from https://github.com/evan-magnusson/rdd/blob/master/rdd/rdd.py
    DESCRIPTION:
        For a given outcome Y and running variable X, computes the optimal bandwidth
        h using a triangular kernel. For more information, see 
        "OPTIMAL BANDWIDTH CHOICE FOR THE REGRESSION DISCONTINUITY ESTIMATOR",
        by Imbens and Kalyanaraman, at http://www.nber.org/papers/w14726.pdf
    INPUTS:
        Two equal length pandas Series
            Y: the outcome variable
            X: the running variable
        cut: value for the threshold of the rdd (scalar) (default is 0)
    
    OUTPUTS:
        Scalar optimal bandwidth value
    """

    assert X.shape[0] == Y.shape[0], "X and Y are not of the same length"
    assert np.sum(pd.isnull(X)) == 0, "NaNs are present in the running variable X"
    assert np.sum(pd.isnull(Y)) == 0, "NaNs are present in the running variable X"


    # Normalize X
    X = X - cut

    # Step 1
    h1 = 1.84 * X.std() * (X.shape[0]**(-.2))
    Nh1neg = X[(X < 0) & (X > -h1)].shape[0]
    Nh1pos =X[(X >= 0) & (X < h1)].shape[0]
    Ybarh1neg = Y[(X < 0) & (X > -h1)].mean()
    Ybarh1pos = Y[(X >= 0) & (X < h1)].mean()
    fXc = (Nh1neg + Nh1pos) / (2 * X.shape[0] * h1)
    sig2c = (((Y[(X < 0) & (X > -h1)]-Ybarh1neg)**2).sum() +((Y[(X >= 0) & (X < h1)]-Ybarh1pos)**2).sum()) / (Nh1neg + Nh1pos)
    
    # Step 2
    medXneg = X[X<0].median()
    medXpos = X[X>=0].median()
    dat_temp = pd.DataFrame({'Y': Y,'X':X})
    dat_temp = dat_temp.loc[(dat_temp['X'] >= medXneg) & (dat_temp['X'] <= medXpos)]
    dat_temp['treat'] = 0
    dat_temp.loc[dat_temp['X'] >= 0, 'treat'] = 1
    dat_temp['X2'] = X**2
    dat_temp['X3'] = X**3
    eqn = 'Y ~ 1 + treat + X + X2 + X3'
    results = smf.ols(eqn, data=dat_temp).fit()
    m3 = 6*results.params.loc['X3']
    h2pos = 3.56 * (X[X>=0].shape[0]**(-1/7.0)) * (sig2c/(fXc * np.max([m3**2, .01]))) ** (1/7.0)
    h2neg = 3.56 * (X[X<0].shape[0]**(-1/7.0)) * (sig2c/(fXc * np.max([m3**2, .01]))) ** (1/7.0)
    Yplus = Y[(X>=0) & (X<=h2pos)]
    Xplus = X[(X>=0) & (X<=h2pos)]
    dat_temp = pd.DataFrame({'Y': Yplus,'X':Xplus})
    dat_temp['X2'] = X**2
    eqn = 'Y ~ 1 + X + X2'
    results = smf.ols(eqn, data=dat_temp).fit()
    m2pos = 2*results.params.loc['X2']
    Yneg = Y[(X<0) & (X>=-h2neg)]
    Xneg = X[(X<0) & (X>=-h2neg)]
    dat_temp = pd.DataFrame({'Y': Yneg,'X':Xneg})
    dat_temp['X2'] = X**2
    eqn = 'Y ~ 1 + X + X2'
    results = smf.ols(eqn, data=dat_temp).fit()
    m2neg = 2*results.params.loc['X2']
    
    # Step 3
    rpos = 720*sig2c / (X[(X>=0) & (X<=h2pos)].shape[0] * h2pos**4)
    rneg = 720*sig2c / (X[(X<0) & (X>=-h2neg)].shape[0] * h2neg**4)
    CK = 3.4375
    hopt = CK * (2*sig2c/(fXc * ((m2pos - m2neg)**2 + (rpos+rneg))))**.2 * Y.shape[0]**(-.2)
    
    return hopt

def create_feat_df(df, running, cutoff, bw, cutoff_indicator="z"):
    """prepares df for subgroup trees by selecting within bw and creating cutoff indicator column"""
    feat_df = df[(df[running] < cutoff + bw) & (df[running] > cutoff - bw)].copy()
    feat_df[cutoff_indicator] = (feat_df[running] >= cutoff).astype(int)

    return feat_df


def _get_tree_paths_helper(tree, node_tup, sofar, paths):
    """
    Helper function for obtaining all decision paths for the given tree.
    
    Note that this function mutates the paths list.
    
    Args:
        node (int): the node index we're currently at
        sofar (list): the node path we're on so far
        paths (set): the set to populate with all decision paths
        
    Returns:
        None, but mutates the paths list
    """
    if sofar + [node_tup] not in paths:
        paths.append(sofar + [node_tup])
        
    # we're not at a leaf, continue traversing
    if tree.children_left[node_tup[0]] != tree.children_right[node_tup[0]]:
        _get_tree_paths_helper(tree, (tree.children_left[node_tup[0]], "<"), sofar + [(node_tup[0], "<")], paths)
        _get_tree_paths_helper(tree, (tree.children_right[node_tup[0]], ">"), sofar + [(node_tup[0], ">")], paths)        
        
        
def get_tree_rules(interp_tree, covar_cols):
    """
    Builds a list of all the decision rules for the given tree.
    
    Args:
        interp_tree (econml.InterpreterTree)
        covar_cols (list)
    Returns:
        list of lists, where each inner list contains the node_idx path.
    """
    
    tree = interp_tree.tree_model_.tree_
    node_dict = interp_tree.node_dict_
    
    paths = []
    
    _get_tree_paths_helper(tree, (0, "R"), [], paths)
    
    features = tree.feature
    thresholds = tree.threshold
        
    rules = []

    for path in paths:
        rule = []
        for node, path_dir in path:
            feat_name = "leaf"
            if features[node] >= 0:
                feat_name = covar_cols[features[node]]
            rule.append(Rule(node, 
                             feat_name, 
                             path_dir,
                             thresholds[node],
                             node_dict[node])
                       )
    
        rules.append(rule)
    return rules


class RuleList():
    """
    Class for holding a rule path down a tree.
    """
    def __init__(self, rules):
        """
        
        Args:
            rules (list): a list of rules
        """
        self.rules = rules


class Rule():
    """
    Class for holding a tree-specific rule
        """
    
    def __init__(self, node_idx, feature, path_dir, threshold, stat_dict):
        self.node_idx = node_idx
        self.feature = feature
        self.path_dir = path_dir
        self.threshold = threshold
        self.stat_dict = stat_dict
        
    def is_sig(self):
        """Checks if confidence interval contains 0"""
        # if the upper and lower bound product is positive, then we don't contain 0
        return (self.stat_dict['ci'][1][0] * self.stat_dict['ci'][0][0]) > 0
    
    def get_std(self):
        """Returns the treatment assignment standard error"""
        return self.stat_dict['std'][0]
        
    def get_treat_uptake(self):
        """Returns the treatment assignment uptake"""
        return self.stat_dict['mean'][0]
    
    def __str__(self):
        #if self.threshold < 0:
        #    return "leaf"

        if self.feature == "leaf":
            return "leaf"
        
        return f"{self.feature} {self.path_dir} {self.threshold:.3f}"
    
    def __repr__(self):
        return self.__str__()

"""Policy tree functions"""
def compute_neff(df, treat='t', instrument='z'):
    comply_rate = df[(df[instrument] == 1)][treat].mean() - df[(df[instrument] == 0)][treat].mean()
    
    return df.shape[0] * (comply_rate)**2


def _get_policy_tree(feat_df, covars, cost, treat="t", cutoff_indicator="z", 
                      max_depth=None, 
                      min_samples_leaf=100,
                      alpha=0.05,
                      random_state=None):
    """
    Helper function for generating policy tree, allows for extraction of policy tree plot.
    """

    #
    cforest = CausalForestDML(
                       honest=True, 
                       inference=True,
                       discrete_treatment=True,
                       n_jobs=16,
                       random_state=random_state
                      )

    cforest.fit(X=feat_df[covars], T=feat_df[cutoff_indicator], Y=feat_df[treat])
    
    
    interp_tree = SingleTreePolicyInterpreter(include_model_uncertainty=True,
                                              uncertainty_only_on_leaves=False,
                                              max_depth=max_depth, 
                                              min_samples_leaf=min_samples_leaf,
                                              uncertainty_level=alpha,
                                              random_state=random_state
                                             )
    interp_tree.interpret(cforest, feat_df[covars], sample_treatment_costs=cost)

    return interp_tree, cforest

def get_policy_tree_subgroups(feat_df, covars, cost, treat="t", cutoff_indicator="z", 
                      max_depth=None, 
                      min_samples_leaf=100,
                      alpha=0.05,
                      random_state=None):
    """
    Fits a policy tree and returns candidate subgroups.
    
    Node traversal lifted from line 460 of the interpret function: https://github.com/microsoft/EconML/blob/fb2d1139f6c271d4b9a24d9c6d122d4d0891afb0/econml/cate_interpreter/_interpreters.py#L460
    
    Args:
        feat_df (pd.DataFrame): feature df
        covars (list): list of covariate columns
        cost (float): the cost of the non-heterogeneous treatment effect
        treat (str): treatment column (note that this will be the "outcome" of our causal estimator)
        cutoff_indicator (str): cutoff indicator, the "treatment" of our causal estimator
        alpha (float): nominal alpha level for testing
        random_state (int): seed for reproducibility
    Returns
        node_dict, with subgroup definitions at each node
    """
    interp_tree, cforest = _get_policy_tree(feat_df, covars, cost, treat, cutoff_indicator, 
                                            max_depth, 
                                            min_samples_leaf,
                                            alpha,
                                            random_state
                                            )
    
    X = feat_df[covars]
    paths = interp_tree.tree_model_.decision_path(X)
    node_dict = {}
    for node_id in range(paths.shape[1]):
        mask = paths.getcol(node_id).toarray().flatten().astype(bool)
        Xsub = X[mask]
        
        eff_est = cforest.const_marginal_ate_inference(Xsub)
        node_dict[node_id] = {'mean': eff_est.mean_point.item(),
                              'net_benefit': (eff_est.mean_point - cost).item(),
                              'subgroup_mask': mask, #feat_df[mask],
                              'std': (eff_est.std_point).item(),
                              'ci': eff_est.conf_int_mean(alpha=alpha),
                              'ci_len': np.abs(eff_est.conf_int_mean(alpha=alpha)[0].item() \
                                               - eff_est.conf_int_mean(alpha=alpha)[1].item()),
                              'neff': compute_neff(feat_df[mask], treat=treat)
                             }
        
        
    rule_list = get_tree_rules(interp_tree, covars)
    
    # populate the path for each node in tree
    for path in rule_list:
        # the terminal node index is the path that leads to the given node
        node_dict[path[-1].node_idx]['rule_path'] = path
        
    return node_dict

def policy_tree_discovery(df, running_cols, grid_dict, treat='t', alpha=0.05, bw=None, rescale=False, omit_mask=False, random_state=None):
    """
    Runs policy tree discovery across the given running columns.
    
    Args:
        df (pd.DataFrame): input dataFrame
        running_cols (list): a list of the candidate running variable column names
        grid_dict (dict): provide a grid of cutoffs for each running var
        treat (str): the treatment variable column name
        alpha (float): the nominal FPR
        rescale (bool): whether to rescale the running variable to [0, 1], helps with large sample sizes and discrete grid 
        omit_mask (bool): optionally omit the mask from results_dict   
        random_state (int): seed for reproducibility
    Returns:
        dict: running_var:sig_results k:v pairs, where sig_results is a dict of cutoff:subgroup_info
    """
    
    num_tests = 0
    
    subgroup_dict = {}
    
    for running in running_cols:
        running_dict = {}
        cutoffs = grid_dict[running]

        covar_cols = list(df.columns)
        covar_cols.remove(treat)
        covar_cols.remove(running)
        
        # Step 1
        for cutoff in cutoffs:
            cutoff_results = []
            print(f"Testing discontinuity {running}>{cutoff} without heterogeneity...")

            if rescale:
                run_max = max(df[running])
                run_min = min(df[running])
                df['running_rescale'] = (df[running] - run_min) / (run_max - run_min)

                # update both running and cutoff to account for rescaling
                running = 'running_rescale'
                cutoff = (cutoff - run_min) / (run_max - run_min)
            
            # Steps 1a-d
            baseline_fs_results, n, cutoff_bw = test_discontinuity(df, cutoff, running,
                                                      treat=treat, bw=bw)
            
            # this is the estimated treatment uptake without considering heterogeneity
            cost = baseline_fs_results.params['z']
            print("number of samples: {}, cost: {}, # compliers: {}".format(n, cost, cost*n))
            
            feat_df = create_feat_df(df, running, cutoff, cutoff_bw)
            print("\tFitting policy tree subgroups...")
            
            # Steps 1e-1g
            node_dict = get_policy_tree_subgroups(feat_df, covar_cols, cost,
                                                  treat=treat,
                                                  # fixed hyperparameters for interpretable subgroups
                                                  max_depth=3,
                                                  min_samples_leaf=100,
                                                  random_state=random_state
                                                 ) 

            num_tests += len(node_dict)
            print("\tTesting subgroup discontinuities...")
            # Step 1h
            for node_id, node_info in node_dict.items():
                
                # precompute Step 3
                fs_reg_results, _, _ = test_discontinuity(feat_df[node_info['subgroup_mask']], cutoff, running,
                                                          treat=treat, bw=cutoff_bw)
                node_info['llr_results'] = fs_reg_results
                
                if omit_mask:
                    node_info['subgroup_mask'] = None

                cutoff_results.append(node_info)

            if rescale:
                cutoff_key = (cutoff * (run_max - run_min)) + run_min
            else:
                cutoff_key = cutoff
            running_dict[np.round(cutoff_key, 2)] = cutoff_results
    
        # Step 2
        subgroup_dict[running] = running_dict
    
    # Save subgroup_dict
    # Steps 4-5 are computed in separate notebook scripts for checkpointing
    return subgroup_dict, num_tests
