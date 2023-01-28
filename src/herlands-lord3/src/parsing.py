"""
Argument parsers and related code.
"""
import argparse
import numpy as np

def get_default_argparser():
    '''
    Construct the default argument parser.
    '''

    parser = argparse.ArgumentParser(
        description="Running RDSS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose',
        default=1, type=int,
        help="verbosity of the output")

    groups = {}

    # Data (both real and synthetic) arguments    
    groups['data'] = parser.add_argument_group(title="data")
    groups['data'].add_argument('--dir_data',
        default=None, type=str,
        help="Name of directory to load data")
    groups['data'].add_argument('--file_json',
        default=None, type=str,
        help="Name of json file to load for real data instructions")
    groups['data'].add_argument('--synthetic',
        action="store_true",
        help="Synthetic data")
    groups['data'].add_argument('--data_verbose',
        action="store_true",
        help="Verbose output of data formatting")
    groups['data'].add_argument('--data_type',
        default='cont', type=str,
        help="Type of synthetic data. {'cont','binary'}")
    groups['data'].add_argument('--z_eff',
        type=float, default=0.3,
        help="Discontinuity z-effect")
    groups['data'].add_argument('--n_size',
        type=int, default=1000,
        help="Number of observations")
    groups['data'].add_argument('--px',
        type=int, default=2,
        help="Number of observable covariates")
    groups['data'].add_argument('--pz',
        type=int, default=2,
        help="Number of forcing variables (only 1 or 2 for now)")
    groups['data'].add_argument('--poly_xT',
        type=int, default=1,
        help="Poly of x wrt T")

    
    # General model/search arguments
    groups['model'] = parser.add_argument_group(title="model")
    groups['model'].add_argument('--obs_model',
        default='normal', type=str,
        help="observation model. {'normal', 'bernoulli'}")
    groups['model'].add_argument('--f_base',
        default='RLM_poly1', type=str,
        help="base model for search procedure. {'RLM_poly', 'OLS', 'GLSHet', 'Stan'}")
    groups['model'].add_argument('--f_yT',
        default='RLM_poly1', type=str,
        help="base model for regression of instrumented T on y. {'RLM_poly', 'OLS', 'GLSHet', 'Stan'}")
    groups['model'].add_argument('--seed',
        default=1, type=int,
        help="Seed for numpy random generator")
    groups['model'].add_argument('--k',
        default=50, type=int,
        help="Number of neighbors")
    groups['model'].add_argument('--ks',
        default=(0, 0, 0), type=float, nargs=3,
         help="Number of neighbors (uses linspace)")
    groups['model'].add_argument('--all_points',
        action='store_true',
        help="Translate the bisecting hyperplane to each point in the neighborhood")
    groups['model'].add_argument('--subsample',
        default=0, type=int,
        help="Subsample the real data")
    groups['model'].add_argument('--search_verbose',
        action="store_true",
        help="Verbose output of the search")
    groups['model'].add_argument('--top_subsets',
        default=1, type=int,
        help="number of top subsest to consider (instead of rand testing)")
    
    # Randomization testing arguments
    groups['rand'] = parser.add_argument_group(title="rand")
    groups['rand'].add_argument('--iters_rand',
        default=0, type=int,
        help="Number of randomization iterations (0 is none)")
    groups['rand'].add_argument('--k_samples',
        default=50, type=int,
        help="Number of neighbors to use for estiamting empirical variance in randomization testing")
    groups['rand'].add_argument('--alpha',
        default=0.05, type=float,
        help="Alpha level of significance")

    return parser, groups


def parse_input(parser):
    '''
    Parse command line input
    '''

    args = parser.parse_args()
    #args.hash = hashlib.md5(str(args)).hexdigest()
    args.ks = np.linspace(args.ks[0], args.ks[1], args.ks[2]).astype(int)

    return args


def string_to_tuple(s):
    '''
    Helper method to convert a String version of a tuple to a tuple object
    '''
    s = s.strip('(')
    s = s.strip(')')
    s = [int(s) for s in s.split(',')]
    return tuple(s)

