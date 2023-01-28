# Automated Local Regression Discontinuity Design Discovery

This repository is the official implementation of Automated Local Regression Discontinuity Design Discovery. Details about the methodology can be found in the KDD 2018 paper https://www.kdd.org/kdd2018/accepted-papers/view/automated-local-regression-discontinuity-design-discovery

## Requirements

Python code was developed with Python 2.7 and then migrated to Python 3.8 While it has been tested on Python 3.8 there may still exist some bugs. Please file a report for any other bugs found and they will be fixed promptly! No deep learning libraries (TensorFlow, Torch, etc.) are used and no special hardware is required for executing the code. For certain experiments and visualizations iPython notebooks are used. Connections between R and Python are automatically achieved with rpy2. Since this package is not stable on some deployments, the code is sometimes commented out to ensure it runs on deployements without rpy2. For users who wish to use these functionalities, they can comment that code back in. The required Python packages are specified in the "requirements.txt" document.

This code was developed with R version 3.4.3. The following R libraries are used in the code:

- rdd
- changepoint
- ecp
- cpm

All instructions in this README assume a UNIX machine, though equivilent Windows commands are almost certainly also valid.

## Get Started!

### Synthetic Data 

You can run a suite of synthetic examples through a bbash script. There is a pre-existing bash script in the repo. It can be executed and vizualized as follows. 
1. Navigate to the /src subdirectory.
2. Run multiple testing runs by executing ```eval $ bash scripts/synthetic_tests_OLS_small.txt```
3. Vizulaize results in synthetic_evaluation_viz.ipynb There are preloaded examples using the output of this bbash script from my local machine.

You can modify any of these scripts to experiment with the various settings offered by LoRD3.


### Real Data

#### Exisitng real data

You can use existing real datasets mentioned in the paper through real_data_general.ipynb There are three datasets to use and experiment with.

#### Your real data

If you want to use LoRD3 on your own data, please follow these instrucitons. Our code uses json objects for real data so we include helper scripts to convert csv data into the appropriate json object

1. Place your data in a .csv where each column represents a different dimension of x, z, T, or y. Store that in the /data subdirectory
2. Define which columns should be labeled as x, z, T, or y in prep_data.py The scheme for how to define that is pretty straightforward, but if it is unclear, you should add a conditional statement that looks like this:
```
elif data_type == 'label_for_this_conditional_block':
    inst = {}
    inst['file'] = 'name_of_your_data_file.csv'
    inst['x']=['one_or_more_columns_that_are_x'] 
    inst['z']=['one_or_more_columns_that_are_z'] # z must be a subset of x
    inst['y']=['single_column_that_is_y']
    inst['T'] = ['single_column_that_is_T'] 
    dir_data = 'relative_path_to_your_data_folder'
    file_json = 'name_of_json_file_you_want.json' # this is optional. If not specified, then your json will be named 'data_inst.json'
```
3. Add your 'label_for_this_conditional_block' to the top of prep_data.py as the variable data_type
4. Navigate to the /src subdirectory
5. Execute ```eval $ python prep_data.py```
6. Copy a block from real_data_general.ipynb to run experiments. If z has one or two dimensions the results can be vizulaized with existing functionality.


## Explaination of Each File

#### Files for executing and visulaizations
- scripts: bash scripts for running multiple sets of experiments. To run the lines in a bash script execute ```eval bash [script_name].txt```
- compare_1d.py: script for running changepoint comparison models
- RD_script.py: executabble script to run multiple runs of synthetic or real data experiments
- RD_script.ipynb: Run a single synthetic run of LoRD3 to vizulaize either real valued or bbinary output. Creates equivilent of fiures 2 and 6 of paper
- synthetic_evaluation_viz.ipynb: evaluation and vizualization for synthetic experiments.
- real_data_general.ipynb: Run real data experiments

#### Backend functions
- __init__.py: for Python file sctructure
- analysis_functions.py: various computations including \tau estimatation and NIG computation
- data_functions.py: synthetic and real data loading, basic pre-processing, and helper functions
- model_functions.py: f_base model definitions, training, prediction, and helper functions
- parsing.py: parses command line inputs
- prep_real_data.py: contains definitions for running real data.
- run_processes.py: general helper devops functions for taking in a text file with lines to run and running them.
- search_functions.py: primary RDD search functions
- subset_functions.py: functions dealing with neighborhoods and subset creation and management
- tests_unit_1.py: some unit tests. Not comprehensive and not kept up-to-date. However, they should all pass. Execute as ```python tests_unit_1.py```
- requirements.txt: Python package requirements using Python 3.8
- README.md: This document that you are currently reading


## Contributing

Code is licensed under the MIT License included in this repo.