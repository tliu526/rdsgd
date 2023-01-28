"""
Unit tests for merge.py methods.
"""
import configparser
import json
import numpy as np
import os
import pandas as pd
import unittest

from optumpipe.merge import *

class TestMerge(unittest.TestCase):

    def setUp(self):
        pass

    def test_lr_prep(self):
        """Tests functionality that prepares lr frames for merge"""
        lr_dict = {
            "rslt_nbr": [0, 5, -99999999],
            "rslt_txt": ["5.5", "", ""]
        }

        lr_df = pd.DataFrame(lr_dict)

        result_df = lr_prep(lr_df)

        self.assertEqual(2, result_df.shape[0])
        self.assertEqual(5.5, result_df.iloc[0]["lr_fmt"])
        self.assertEqual(5, result_df.iloc[1]["lr_fmt"])


    def test_gen_diag_indicator(self):
        pre_dict = {
            "patid": [0, 1, 1, 2, 3],
            "fst_dt": [
                '2019-01-01',
                '2020-05-01', # out of order fst_dt
                '2020-04-01',
                '2021-07-01',
                '2022-01-01'
            ]
        }

        pre_df = pd.DataFrame(pre_dict, dtype=str)
        pre_df['fst_dt'] = pd.to_datetime(pre_df['fst_dt'])

        post_dict = {
            "patid": [0, 0, 0, 1, 2, 2],
            "fst_dt": [
                '2019-01-15', # should be a match with a 30 day window 
                '2019-01-16', # should be a match, but dropped during dedup
                '2019-02-15', # should not be a match, outside window 
                '2020-05-15', # should not be a match, outside window 
                '2020-04-30', # should not be a match, before idx date
                '2022-07-01', # should not be a match, outside window
            ]
        }

        post_df = pd.DataFrame(post_dict, dtype=str)
        post_df['fst_dt'] = pd.to_datetime(post_df['fst_dt'])

        window = 30

        expected_dict = {
            "patid": ['0', '1', '2', '3'],
            "fst_dt_pre": [
                '2019-01-01',
                '2020-04-01',
                '2021-07-01',
                '2022-01-01'
            ],
            "fst_dt_post": [
                '2019-01-15',
                np.nan,
                np.nan,
                np.nan
            ],
            "date_diff": [
                pd.to_timedelta(14, unit='d'),
                np.nan,
                np.nan,
                np.nan
            ],
            "indicator": [1, 0, 0, 0]
        }
        expected_df = pd.DataFrame(expected_dict)
        expected_df['fst_dt_pre'] = pd.to_datetime(expected_df['fst_dt_pre'])
        expected_df['fst_dt_post'] = pd.to_datetime(expected_df['fst_dt_post'])

        actual_df = gen_diag_indicator(pre_df, post_df, window)

        print(expected_df)
        print(actual_df)
        pd.testing.assert_frame_equal(expected_df, actual_df,
                                      check_index_type=False)
        

if __name__ == "__main__":
    unittest.main()
