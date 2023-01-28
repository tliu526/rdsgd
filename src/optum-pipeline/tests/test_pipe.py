"""
Unit tests for pipe.py methods.
"""
import configparser
import json
import os
import pandas as pd
import unittest

from optumpipe.pipe import *

# load configuration
config = configparser.ConfigParser()
config.read("config.ini")
default_config = config['default']

class TestOptumPipe(unittest.TestCase):

    def setUp(self):
        pass

    def test_breast_cancer_extract(self):
        """Tests for the correct initialization of breast cancer pipe tasks"""
        test_diag_pipe = OptumPipe("breast_cancer", "diag")

        self.assertTrue('screen' in test_diag_pipe.code_dict)
        self.assertTrue('diagnosis' in test_diag_pipe.code_dict)

        test_diag_pipe = OptumPipe("breast_cancer", "proc")

        self.assertTrue('screen' in test_diag_pipe.code_dict)
        self.assertFalse('diagnosis' in test_diag_pipe.code_dict)


    def test_breast_cancer_merge(self):
        """Tests for the correct initialization of breast cancer merge task"""
        test_merge_pipe = OptumPipe("breast_cancer", "merge")

        self.assertEqual(test_merge_pipe.window, 7)
        
        self.assertEqual(test_merge_pipe.pre_path, os.path.join(
            default_config['output_dir'],
            "breast_cancer",
            "diag",
            "{}q{}_visit.parq"
        ))

        self.assertEqual(test_merge_pipe.post_path, os.path.join(
            default_config['output_dir'],
            "breast_cancer",
            "diag",
            "{}q{}_screen.parq"
        ))

    def test_diabetes_merge(self):
        """Tests for the correct initialization of diabetes merge task"""
        test_merge_pipe = OptumPipe("diabetes", "merge")

        self.assertEqual(test_merge_pipe.window, 30)
        
        self.assertEqual(test_merge_pipe.pre_path, os.path.join(
            default_config['output_dir'],
            "diabetes",
            "lr",
            "{}q{}_a1c.parq"
        ))

        self.assertEqual(test_merge_pipe.post_path, os.path.join(
            default_config['output_dir'],
            "diabetes",
            "diag",
            "{}q{}_diagnosis.parq"
        ))

        self.assertCountEqual(test_merge_pipe.post_cols, ["diag"])

        self.assertCountEqual(test_merge_pipe.pre_cols, [
            "loinc_cd",
            "proc_cd",
            "abnl_cd",
            "rslt_unit_nm",   
            "fmt_lr"
        ])



if __name__ == '__main__':
    unittest.main()
