"""
Creates dummy stata files for unit testing.
"""
import configparser
import json
import os
import pandas as pd

# load default configuration
config = configparser.ConfigParser()
config.read("config.ini")
default_config = config['default']

TEST_DIR = os.path.join(default_config['output_dir'], "test", "raw")

with open("json/cols/med.json") as f:
    MED_DICT = json.load(f)

with open("json/cols/lr.json") as f:
    LR_DICT = json.load(f)

def create_breast_cancer_diag():
    """Creates test breast cancer diagnosis data"""

    df = pd.DataFrame(columns=(MED_DICT['idx'] + MED_DICT['diag']))
    df['diag'] = [
        # "diagnosis" task codes
        'V174', # should *not* be selected
        '1C50', # should *not* be selected
        'C50123',
        '1744544',
        'C50',
        '174', 
        # "screen" task codes 
        '177067', # CPT, should *not* be selected
        'Z12312', # ICD, should *not* be selected
        'v7612',  # ICD, should *not* be selected
        'Z1231',
        'Z1239',
        '77063',
        '77067',
        'V7619',
        'V7612',
    ]

    # stata requires no nans be present
    df = df.fillna('')
    
    df.to_stata(os.path.join(TEST_DIR, "diag_test_breast_cancer.dta"))

def create_breast_cancer_proc():
    """Creates test breast cancer procedure data"""

    df = pd.DataFrame(columns=(MED_DICT['idx'] + MED_DICT['proc']))
    df['proc'] = [
        '177067', # CPT, should *not* be selected
        'Z12312', # ICD, should *not* be selected
        'v7612',  # ICD, should *not* be selected
        'Z1231',
        'Z1239',
        '77063',
        '77067',
        'V7619',
        'V7612',
    ]

    # stata requires no nans be present
    df = df.fillna('')
    
    df.to_stata(os.path.join(TEST_DIR, "proc_test_breast_cancer.dta"))


def create_diabetes_diag():
    """Creates test diabetes diagnosis data"""

    df = pd.DataFrame(columns=(MED_DICT['idx'] + MED_DICT['diag']))
    df['diag'] = [
        # "diagnosis" task codes
        "25002",
        "25001", # should *not* be selected
        "25090",
        "25093", # should *not* be selected
        "E1100",
        "5E11", # should *not* be selected
        "E11"
    ]

    # stata requires no nans be present
    df = df.fillna('')
    
    df.to_stata(os.path.join(TEST_DIR, "diag_test_diabetes.dta"))

def create_diabetes_lr():
    """Creates test diabetes lab result data"""

    df = pd.DataFrame(columns=(LR_DICT['idx'] + LR_DICT['lr']))
    df['loinc_cd'] = [
        # a1c
        "4548-4",
        "4549-2",
        "4549-22", # should *not* be selected
        "14548-4", # should *not* be selected

        # glucose
        "1558-6",
        "2345-7",
        "11558-6", # should *not* be selected
        "2345-78"  # should *not* be selected
    ]

    # stata requires no nans be present
    df = df.fillna('')
    
    df.to_stata(os.path.join(TEST_DIR, "lr_test_diabetes.dta"))


def main():
    """Creates all test data"""
    create_breast_cancer_diag()

if __name__ == '__main__':
    main()