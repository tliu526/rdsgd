"""
Extracts data from raw Optum files into parquet files with selected columns.

Raw (stata) data path formats for "medical" Optum files, including the new 
diagnosis and procedure tables for the Optum 2020 format: 
    - medical: ses_m<year>q<quarter>.dta
    - diagnosis: ses_diag<year>q<quarter>.dta
    - procedure: ses_proc<year>q<quarter>.dta
"""
import configparser
import json
import os
import pandas as pd

CHUNK_SIZE = 10000

# load default configuration
config = configparser.ConfigParser()
config.read("config.ini")
default_config = config['default']

# load columns from config file, TODO move the data loading into a sep function?
col_dict = {}

# Medical columns
with open("json/cols/med.json") as f:
    med_json = json.load(f)

    for key in ["m", "diag", "proc"]:
        col_dict[key] = med_json['idx'] + med_json[key]

# Lab results columns
with open("json/cols/lr.json") as f:
    lr_json = json.load(f)

    for key in ["lr"]:
        col_dict[key] = lr_json['idx'] + lr_json[key]

# prescription columns
with open("json/cols/rx.json") as f:
    rx_json = json.load(f)

    for key in ["r"]:
        col_dict[key] = rx_json['idx'] + rx_json[key]


def code_extract(in_file, out_path, out_name, code_type, code_dict, chunksize=CHUNK_SIZE, test=False):
    """Extracts information from the code files (diag, proc, lr, rx).

    Filters by the code prefixes provided in code_dict.

    Pre:
        the input stata files have 'diag' column.

    Args:
        in_file (str): input file path
        out_path (str): output file path
        out_name (str): the f string output name
        code_type (str): the code file: diag, proc, lr
        code_dict (dict): dictionary with 
            (out_name, [icd prefixes]) k,v pairs. eg "ht": ["401", "I10"]
        chunksize (int): number of entries in the raw file to read per 
            iteration, don't set too large
        test (bool): enables debugging print statements and logic
    Returns:
        dict: a dictionary of the resulting diag frames
    """

    code_cols = col_dict[code_type]

    code_dfs = {code: pd.DataFrame() for code in code_dict.keys()}

    target_col = code_type
    # TODO a more scalable way to do this?
    if code_type == 'lr':
        target_col = 'loinc_cd'
    if code_type == 'r':
        target_col = 'ndc'

    for chunk in pd.read_stata(in_file, chunksize=chunksize, columns=code_cols):
        if test:
            print(chunk.shape[0])

        for diag, codes in code_dict.items():
            sel_df = chunk[chunk[target_col].str.contains("|".join(codes), regex=True)]
            if sel_df.shape[0] > 0:

                code_dfs[diag] = pd.concat([code_dfs[diag], sel_df])
                #code_dfs[diag] = code_dfs[diag].append(sel_df)

        if test:
            for diag, df in code_dfs.items():
                print(diag)
                print(df.shape)
                print(df.head())
            break
    
    for diag, df in code_dfs.items():
        df = df.reset_index(drop=True)
        f_path = os.path.join(out_path, out_name.format(diag))
        
        if test:
            f_path = f_path + "_test.parq"
        else:
            f_path = f_path + ".parq"
        with open(f_path, "wb") as f:
            #pickle.dump(df, f, -1)
            df.to_parquet(f)


if __name__ == '__main__':
    import time
    import multiprocessing

    config = configparser.ConfigParser()
    config.read("config.ini")
    data_dir = config['default']['data_dir']
    out_dir = config['default']['output_dir']

    print(col_dict)



