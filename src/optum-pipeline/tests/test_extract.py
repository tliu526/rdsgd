"""
Unit tests for extract.py methods.
"""
import configparser
import json
import os
import pandas as pd
import unittest

from optumpipe.extract import code_extract
from make_test_data import *

config = configparser.ConfigParser()
config.read("config.ini")
default_config = config['default']

TEST_DATA_DIR = os.path.join(default_config['output_dir'], "test", "raw")
TEST_OUT_DIR = os.path.join(default_config['output_dir'], "test", "out")

class TestBreastCancer(unittest.TestCase):

    def setUp(self):
        """Set up the unit test class."""
        
        # create synthetic data
        create_breast_cancer_diag()
        create_breast_cancer_proc()

        create_diabetes_diag()
        create_diabetes_lr()


    def test_bc_diag(self):
        """Tests diagnosis and screen extraction from diag table"""
        out_path = os.path.join(TEST_OUT_DIR, "diag")
        
        # create out_path if it doesn't exist yet
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        data_path = os.path.join(TEST_DATA_DIR, "diag_test_breast_cancer.dta")
        
        # load diag_dict for breast cancer
        with open("json/codes/breast_cancer.json") as f:
            breast_cancer_dict = json.load(f)
            bc_diag_dict = breast_cancer_dict['diag']

        diag_dict = {}
        for task in bc_diag_dict:
            codes = []
            for code_type in bc_diag_dict[task]['codes']:
                codes += bc_diag_dict[task][code_type]
            diag_dict[f'bc_{task}'] = codes

        # test execution
        code_extract(data_path, out_path, "test_{}", 
                     'diag', diag_dict, test=False, chunksize=1)

        # load resulting dataframe, check the path matches diag_dict
        test_diagnosis = pd.read_parquet(os.path.join(out_path, 'test_bc_diagnosis.parq'))

        self.assertEqual(4, test_diagnosis.shape[0])
        self.assertTrue(test_diagnosis['diag'].isin(['C50', '174', 'C50123', '1744544']).all())
        self.assertFalse(test_diagnosis['diag'].isin(['1C50', 'V174']).any())

        # load resulting dataframe, check the path matches diag_dict
        test_screen = pd.read_parquet(os.path.join(out_path, 'test_bc_screen.parq'))

        self.assertEqual(6, test_screen.shape[0])
        self.assertTrue(test_screen['diag'].isin([
            'Z1231',
            'Z1239',
            '77063',
            '77067',
            'V7619',
            'V7612',
        ]).all())
        self.assertFalse(test_screen['diag'].isin(['177067', 'Z12312', 'v7612']).any())

    def test_bc_screen_extract(self):
        """Tests screening extraction for breast cancer RDD"""
        
        out_path = os.path.join(TEST_OUT_DIR, "proc")    
        # create out_path if it doesn't exist yet
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        data_path = os.path.join(TEST_DATA_DIR, "proc_test_breast_cancer.dta")

        # load proc_dict for breast cancer
        with open("json/codes/breast_cancer.json") as f:
            breast_cancer_dict = json.load(f)
            bc_proc_dict = breast_cancer_dict['proc']

        codes = []
        for code_type in bc_proc_dict['screen']['codes']:
            codes += bc_proc_dict['screen'][code_type]

        proc_dict = {
            "bc_screen": codes
        }

        # test execution
        code_extract(data_path, out_path, "test_{}", 
                     'proc', proc_dict, test=False, chunksize=1)

        # load resulting dataframe, check the path matches proc_dict
        test_df = pd.read_parquet(os.path.join(out_path, 'test_bc_screen.parq'))

        self.assertEqual(6, test_df.shape[0])
        self.assertTrue(test_df['proc'].isin([
            'Z1231',
            'Z1239',
            '77063',
            '77067',
            'V7619',
            'V7612',
        ]).all())
        self.assertFalse(test_df['proc'].isin(['177067', 'Z12312', 'v7612']).any())


    def test_diabetes_a1c_extract(self):
        """Tests a1c lab extraction for diabetes RDD"""
        
        out_path = os.path.join(TEST_OUT_DIR, "lr")    
        # create out_path if it doesn't exist yet
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        data_path = os.path.join(TEST_DATA_DIR, "lr_test_diabetes.dta")

        # load proc_dict for breast cancer
        with open("json/codes/diabetes.json") as f:
            dm_dict = json.load(f)
            dm_lr_dict = dm_dict['lr']

        codes = []
        lr_dict = {}
        for task in dm_lr_dict:
            codes = []
            for code_type in dm_lr_dict[task]['codes']:
                codes += dm_lr_dict[task][code_type]
            lr_dict[f'dm_{task}'] = codes

        # test execution
        code_extract(data_path, out_path, "test_{}", 
                    'lr', lr_dict, test=False, chunksize=1)

        # load resulting dataframe, check the path matches lr_dict
        test_df = pd.read_parquet(os.path.join(out_path, 'test_dm_a1c.parq'))

        self.assertEqual(2, test_df.shape[0])
        self.assertTrue(test_df['loinc_cd'].isin(['4548-4', '4549-2']).all())
        self.assertFalse(test_df['loinc_cd'].isin(['4549-22', '14548-4']).any())

        # load resulting dataframe, check the path matches lr_dict
        test_df = pd.read_parquet(os.path.join(out_path, 'test_dm_glucose.parq'))

        self.assertEqual(2, test_df.shape[0])
        self.assertTrue(test_df['loinc_cd'].isin(['1558-6', '2345-7']).all())
        self.assertFalse(test_df['loinc_cd'].isin(['11558-6', '2345-78']).any())



    def test_diabetes_diag(self):
        """Tests diagnosis and screen extraction from diag table"""
        out_path = os.path.join(TEST_OUT_DIR, "diag")
        
        # create out_path if it doesn't exist yet
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        data_path = os.path.join(TEST_DATA_DIR, "diag_test_diabetes.dta")
        
        # load diag_dict for breast cancer
        with open("json/codes/diabetes.json") as f:
            dm_dict = json.load(f)
            dm_diag_dict = dm_dict['diag']

        diag_dict = {}
        for task in dm_diag_dict:
            codes = []
            for code_type in dm_diag_dict[task]['codes']:
                codes += dm_diag_dict[task][code_type]
            diag_dict[f'dm_{task}'] = codes

        # test execution
        code_extract(data_path, out_path, "test_{}", 
                     'diag', diag_dict, test=False, chunksize=1)

        # load resulting dataframe, check the path matches diag_dict
        test_diagnosis = pd.read_parquet(os.path.join(out_path, 'test_dm_diagnosis.parq'))

        self.assertEqual(4, test_diagnosis.shape[0])
        self.assertTrue(test_diagnosis['diag'].isin(['25002', '25090', 'E1100', 'E11']).all())
        self.assertFalse(test_diagnosis['diag'].isin(['25001', '25093', '5E11']).any())

      
if __name__ == '__main__':
    unittest.main()