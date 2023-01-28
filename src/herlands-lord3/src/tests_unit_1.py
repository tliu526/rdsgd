import unittest

import numpy as np
from data_functions import *
from search_functions import *
from analysis_functions import *

def fun(x):
	return x + 1

class MyTest(unittest.TestCase):

	def test_cont_normal_OLS(self):
		np.random.seed(1)
		#  generate data
		data_type='cont'; z_eff=3.0; n=100; poly_xT=1; discont_type='square';
		x,y,z,T, D_z, z_eff, beta_y_T, discont = data_synthetic(data_type=data_type, n=n, px=2, pz=2, z_eff=z_eff,  poly_xT=poly_xT, discont_type=discont_type, verbose=False)
		#  search for discontinuity
		obs_model='normal'; f_base='OLS_poly1';  f_yT='OLS_poly1'; k=10; post_shrink=False; 
		llrs, neighs, subsets_best, beta_0_n, beta_1_n, T_fx, llrs_n, llrs_a, centers_n, angles_best, subset_imax = \
		RDSS_residual(obs_model, T, x, z, f_base=f_base, all_points=False, k=k, post_shrink=post_shrink, verbose=False, plotting=False)
		#  result
		max_arg = np.nanargmax(llrs)
		print ('max_arg=',max_arg, 'Correct=42')
		self.assertEqual(max_arg, 42)

	def test_binary_normal_OLS(self):
		np.random.seed(1)
		#  generate data
		data_type='binary'; z_eff=3.0; n=100; poly_xT=1; discont_type='square';
		x,y,z,T, D_z, z_eff, beta_y_T, discont = data_synthetic(data_type=data_type, n=n, px=2, pz=2, z_eff=z_eff,  poly_xT=poly_xT, discont_type=discont_type, verbose=False)
		#  search for discontinuity
		obs_model='normal'; f_base='OLS_poly1';  f_yT='OLS_poly1'; k=10; post_shrink=False; 
		llrs, neighs, subsets_best, beta_0_n, beta_1_n, T_fx, llrs_n, llrs_a, centers_n, angles_best, subset_imax = \
		RDSS_residual(obs_model, T, x, z, f_base=f_base, all_points=False, k=k, post_shrink=post_shrink, verbose=False, plotting=False)
		#  result
		max_arg = np.nanargmax(llrs)
		print ('max_arg=',max_arg, 'Correct=40')
		self.assertEqual(max_arg, 40)

	def test_binary_bernoulli_OLS(self):
		np.random.seed(1)
		#  generate data
		data_type='binary'; z_eff=3.0; n=100; poly_xT=1; discont_type='square';
		x,y,z,T, D_z, z_eff, beta_y_T, discont = data_synthetic(data_type=data_type, n=n, px=2, pz=2, z_eff=z_eff,  poly_xT=poly_xT, discont_type=discont_type, verbose=False)
		#  search for discontinuity
		obs_model='bernoulli'; f_base='OLS_poly1';  f_yT='OLS_poly1'; k=10; post_shrink=False; 
		llrs, neighs, subsets_best, beta_0_n, beta_1_n, T_fx, llrs_n, llrs_a, centers_n, angles_best, subset_imax = \
		RDSS_residual(obs_model, T, x, z, f_base=f_base, all_points=False, k=k, post_shrink=post_shrink, verbose=False, plotting=False)
		#  result
		max_arg = np.nanargmax(llrs)
		print ('max_arg=',max_arg, 'Correct=84')
		self.assertEqual(max_arg, 84)

	def test_binary_bernoulli_Logit(self):
		np.random.seed(1)
		#  generate data
		data_type='binary'; z_eff=3.0; n=100; poly_xT=1; discont_type='square';
		x,y,z,T, D_z, z_eff, beta_y_T, discont = data_synthetic(data_type=data_type, n=n, px=2, pz=2, z_eff=z_eff,  poly_xT=poly_xT, discont_type=discont_type, verbose=False)
		#  search for discontinuity
		obs_model='bernoulli'; f_base='Logit_poly1';  f_yT='OLS_poly1'; k=10; post_shrink=False; 
		llrs, neighs, subsets_best, beta_0_n, beta_1_n, T_fx, llrs_n, llrs_a, centers_n, angles_best, subset_imax = \
		RDSS_residual(obs_model, T, x, z, f_base=f_base, all_points=False, k=k, post_shrink=post_shrink, verbose=False, plotting=False)
		#  result
		max_arg = np.nanargmax(llrs)
		print ('max_arg=',max_arg, 'Correct=40')
		self.assertEqual(max_arg, 40)

	def test_cont_normal_kernel(self):
		np.random.seed(1)
		#  generate data
		data_type='cont'; z_eff=3.0; n=100; poly_xT=1; discont_type='square';
		x,y,z,T, D_z, z_eff, beta_y_T, discont = data_synthetic(data_type=data_type, n=n, px=2, pz=2, z_eff=z_eff,  poly_xT=poly_xT, discont_type=discont_type, verbose=False)
		#  search for discontinuity
		obs_model='normal'; f_base='kernel';  f_yT='OLS_poly1'; k=10; post_shrink=False; 
		llrs, neighs, subsets_best, beta_0_n, beta_1_n, T_fx, llrs_n, llrs_a, centers_n, angles_best, subset_imax = \
		RDSS_residual(obs_model, T, x, z, f_base=f_base, all_points=False, k=k, post_shrink=post_shrink, verbose=False, plotting=False)
		#  result
		max_arg = np.nanargmax(llrs)
		print ('max_arg=',max_arg, 'Correct=9')
		self.assertEqual(max_arg, 9)

	def test_cont_normal_kernel_shrink(self):
		np.random.seed(1)
		#  generate data
		data_type='cont'; z_eff=3.0; n=100; poly_xT=1; discont_type='square';
		x,y,z,T, D_z, z_eff, beta_y_T, discont = data_synthetic(data_type=data_type, n=n, px=2, pz=2, z_eff=z_eff,  poly_xT=poly_xT, discont_type=discont_type, verbose=False)
		#  search for discontinuity
		obs_model='normal'; f_base='kernel';  f_yT='OLS_poly1'; k=10; post_shrink=True; 
		llrs, neighs, subsets_best, beta_0_n, beta_1_n, T_fx, llrs_n, llrs_a, centers_n, angles_best, subset_imax = \
		RDSS_residual(obs_model, T, x, z, f_base=f_base, all_points=False, k=k, post_shrink=post_shrink, verbose=False, plotting=False)
		#  result
		max_arg = np.nanargmax(llrs)
		print ('max_arg=',max_arg, 'Correct=57')
		self.assertEqual(max_arg, 57)

	def test_cont_normal_OLS(self):
		np.random.seed(1)

		### Search procedure
		#  generate data
		data_type='cont'; z_eff=2.0; n=100; poly_xT=1; discont_type='square';
		x,y,z,T, D_z, z_eff, beta_y_T, discont = data_synthetic(data_type=data_type, n=n, px=2, pz=2, z_eff=z_eff,  poly_xT=poly_xT, discont_type=discont_type, verbose=False)
		#  search for discontinuity
		obs_model='normal'; f_base='OLS_poly1';  f_yT='OLS_poly1'; k=10; post_shrink=False; 
		llrs, neighs, subsets_best, beta_0_n, beta_1_n, T_fx, llrs_n, llrs_a, centers_n, angles_best, subset_imax = \
		RDSS_residual(obs_model, T, x, z, f_base=f_base, all_points=False, k=k, post_shrink=post_shrink, verbose=False, plotting=False)
		#  result
		max_arg = np.nanargmax(llrs)
		self.assertEqual(max_arg, 42)

		### Analysis
		top_subsets = 1 # 1000
		T_hat_master = get_pred_mean(T_fx, f_base, x)
		validated_idxs = validate_subsets(llrs, False, angles_best, centers_n, x, z, pval_sig=0.05, top_subsets=top_subsets, verbose=False)
		z_eff, acc, T_eff, T_bse, T_placebos = RDSS_result_stats_sig(obs_model, llrs, validated_idxs, beta_0_n, beta_1_n, subsets_best, neighs, discont, x, y, T, T_hat_master, f_base=f_base, f_yT=f_yT, verbose=False)

		self.assertLess(abs(acc[2]-0.609978), 1e-4)
		self.assertLess(abs(acc[3]-0.609978), 1e-4)

		self.assertLess(abs(T_eff[0]-7.11573625), 1e-4)
		self.assertLess(abs(T_eff[1]-5.38550892), 1e-4)
		self.assertLess(abs(T_eff[2]-1.05245472), 1e-4)

		print ('Passed test_cont_normal_OLS')


	def test_binary_bernoulli_Logit(self):
		np.random.seed(1)

		### Search procedure
		#  generate data
		data_type='binary'; z_eff=6.0; n=100; poly_xT=1; discont_type='square';
		x,y,z,T, D_z, z_eff, beta_y_T, discont = data_synthetic(data_type=data_type, n=n, px=2, pz=2, z_eff=z_eff,  poly_xT=poly_xT, discont_type=discont_type, verbose=False)
		#  search for discontinuity
		obs_model='bernoulli'; f_base='Logit_poly1';  f_yT='OLS_poly1'; k=10; post_shrink=False; 
		llrs, neighs, subsets_best, beta_0_n, beta_1_n, T_fx, llrs_n, llrs_a, centers_n, angles_best, subset_imax = \
		RDSS_residual(obs_model, T, x, z, f_base=f_base, all_points=False, k=k, post_shrink=post_shrink, verbose=False, plotting=False)
		#  result
		max_arg = np.nanargmax(llrs)
		self.assertEqual(max_arg, 93)

		### Analysis
		top_subsets = 1 # 1000
		T_hat_master = get_pred_mean(T_fx, f_base, x)
		validated_idxs = validate_subsets(llrs, False, angles_best, centers_n, x, z, pval_sig=0.05, top_subsets=top_subsets, verbose=False)
		z_eff, acc, T_eff, T_bse, T_placebos = RDSS_result_stats_sig(obs_model, llrs, validated_idxs, beta_0_n, beta_1_n, subsets_best, neighs, discont, x, y, T, T_hat_master, f_base=f_base, f_yT=f_yT, verbose=False)

		self.assertLess(abs(acc[2]-0.32191527), 1e-4)
		self.assertLess(abs(acc[3]-0.32191527), 1e-4)

		self.assertLess(abs(T_eff[0]-1.46770095), 1e-4)
		self.assertLess(abs(T_eff[1]-3.62441678), 1e-4)
		self.assertLess(abs(T_eff[2]-1.7683465), 1e-4)

		print ('Passed test_binary_bernoulli_Logit')

if __name__ == '__main__':
    unittest.main()