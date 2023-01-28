import json
import os

data_type = 'blend_sim' # 'ED_visits_all_np' # 'Inpatient_visits' # 'ED_visits' # 'GPA' # 'test_score' # 'IN_visits_self' # 'IN_visits_medi' # 'IN_visits_o_ins' # 

file_json = 'data_inst.json'

if data_type == 'blend_sim':
    inst = {}
    inst['file'] = 'gap0.2_seed0.csv',
    inst['x'] = ['x', 'covar']
    inst['z'] = ['x', 'covar']
    inst['T'] = ['t']
    inst['y'] = ['t'] # For left school see Table 4. For next term GPA see Table 5. For last 3 see Table 6.
    dir_data = '../data/blend_sim'
    file_json = 'blend_sim.json'


elif data_type == 'GPA':
    inst = {}
    inst['file'] = 'data_orig.csv',
    inst['x'] = ['dist_from_cut','hsgrade_pct','totcredits_year1','sex','age_at_entry','bpl_north_america',
    'mtongue','loc_campus1','loc_campus2']
    inst['z'] = ['dist_from_cut','hsgrade_pct','totcredits_year1','age_at_entry']
    inst['T'] = ['probation_year1']
    inst['y'] = ['left_school','GPA_year2','gradin4','gradin5','gradin6'] # For left school see Table 4. For next term GPA see Table 5. For last 3 see Table 6.
    dir_data = '../data/AcademicProbation_LSO_2010/'

elif data_type == 'ED_visits':
    # The plots in Figure 1 of https://pdfs.semanticscholar.org/cf93/00bbf7dd51a54482a6af05d6438f98514d0a.pdf
    #  use '_np'. However, the paper considers mulitple different settings. Here we consider all only.
    # medi_all: medicaid. priv_all: private insurance. self_all: uninsured. o_ins_all: other insurance
    inst = {}
    inst['file'] = 'P03_ED_Analysis_File.csv',
    inst['x'] = ['months_23']
    inst['z'] = ['months_23']
    inst['T'] = ['priv_all'] 
    inst['y'] = ['all']
    dir_data = '../data/ED_visits/'

elif data_type == 'ED_visits_priv':
    inst = {}
    inst['file'] = 'P03_ED_Analysis_File.csv',
    inst['x']=['months_23']; inst['z']=['months_23']; inst['y']=['all']
    inst['T'] = ['priv_all'] 
    dir_data = '../data/ED_visits/'
    file_json = 'data_inst_priv.json'
elif data_type == 'ED_visits_self':
    inst = {}
    inst['file'] = 'P03_ED_Analysis_File.csv',
    inst['x']=['months_23']; inst['z']=['months_23']; inst['y']=['all']
    inst['T'] = ['self_all'] 
    dir_data = '../data/ED_visits/'
    file_json = 'data_inst_self.json'
elif data_type == 'ED_visits_medi':
    inst = {}
    inst['file'] = 'P03_ED_Analysis_File.csv',
    inst['x']=['months_23']; inst['z']=['months_23']; inst['y']=['all']
    inst['T'] = ['medi_all'] 
    dir_data = '../data/ED_visits/'
    file_json = 'data_inst_medi.json'
elif data_type == 'ED_visits_o_ins':
    inst = {}
    inst['file'] = 'P03_ED_Analysis_File.csv',
    inst['x']=['months_23']; inst['z']=['months_23']; inst['y']=['all']
    inst['T'] = ['o_ins_all'] 
    dir_data = '../data/ED_visits/'
    file_json = 'data_inst_o_ins.json'
elif data_type == 'ED_visits_all_np':
    inst = {}
    inst['file'] = 'P03_ED_Analysis_File.csv',
    inst['x']=['months_23']; inst['z']=['months_23']; inst['y']=['all_np']
    inst['T'] = ['all'] 
    dir_data = '../data/ED_visits/'
    file_json = 'data_inst_all_np.json'

    
elif data_type == 'Inpatient_visits':
    # todo: try different T: 'TOT_priv_ALL', 'TOT_self_ALL', 'TOT_o_ins_ALL'
    inst = {}
    inst['file'] = 'P10_Inpatient_CSV_File.csv',
    inst['x'] = ['months_23']
    inst['z'] = ['months_23']
    inst['T'] = ['TOT_priv_ALL'] 
    inst['y'] = ['TOT_ALL']
    dir_data = '../data/Inpatient_visits/'

elif data_type == 'IN_visits_priv':
    inst = {}
    inst['file'] = 'P10_Inpatient_CSV_File.csv',
    inst['x']=['months_23']; 
    inst['z']=['months_23']; 
    inst['y']=['TOT_ALL']
    inst['T'] = ['TOT_priv_ALL'] 
    dir_data = '../data/Inpatient_visits/'
    file_json = 'data_inst_priv.json'
elif data_type == 'IN_visits_self':
    inst = {}
    inst['file'] = 'P10_Inpatient_CSV_File.csv',
    inst['x']=['months_23']; i
    nst['z']=['months_23']; i
    nst['y']=['TOT_ALL']
    inst['T'] = ['TOT_self_ALL'] 
    dir_data = '../data/Inpatient_visits/'
    file_json = 'data_inst_self.json'
elif data_type == 'IN_visits_medi':
    inst = {}
    inst['file'] = 'P10_Inpatient_CSV_File.csv',
    inst['x']=['months_23']; 
    inst['z']=['months_23']; 
    inst['y']=['TOT_ALL']
    inst['T'] = ['TOT_medi_ALL'] 
    dir_data = '../data/Inpatient_visits/'
    file_json = 'data_inst_medi.json'
elif data_type == 'IN_visits_o_ins':
    inst = {}
    inst['file'] = 'P10_Inpatient_CSV_File.csv',
    inst['x']=['months_23']; 
    inst['z']=['months_23']; 
    inst['y']=['TOT_ALL']
    inst['T'] = ['TOT_o_ins_ALL'] 
    dir_data = '../data/Inpatient_visits/'
    file_json = 'data_inst_o_ins.json'

    
elif data_type == 'test_score':
    # True cutoff is pretest=215. True treatment is 10
    inst = {}
    inst['file'] = 'RDD_Guide_Dataset_0.csv',
    inst['x'] = ['gender', 'sped', 'frlunch', 'esol', 'black', 'white',
           'hispanic', 'asian', 'age', 'pretest']
    inst['z'] = ['age', 'pretest']
    inst['T'] = ['treat']
    inst['y'] = ['posttest']
    dir_data = '../data/ED_visits/AcademicProbation_LSO_2010'
    
else:
    assert False, 'Invalid real data type'


with open(os.path.join(dir_data, file_json), 'w') as fp:
    json.dump(inst, fp)