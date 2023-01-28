"""
Holds class for building Optum pipeline tasks.
"""

import configparser
import json
import multiprocessing
import os
import pandas as pd
import time

from . import extract
from . import merge

# TODO need to move this into a config file
# better yet, pull dynamically from json/codes
SUPPORTED_TASKS = [
    'breast_cancer',
    'check_up',
    "colon_cancer",
    "diabetes",
    "chd",
    "ckd"
]

SUPPORTED_TABLE_TYPES = [
    'diag',
    'lr',
    #'m',
    'proc',
    'r', # prescriptions, r(x)
    'merge', # not a table type but streamlines argparsing
]
# temporary change the start year for ckd
START_YR = 2005 # 2004
END_YR = 2019

__all__ = [
    'SUPPORTED_TASKS', 
    'SUPPORTED_TABLE_TYPES', 
    'OptumPipe'
]

class OptumPipe():
    """An instance of an Optum data pipeline task."""

    code_dict = {}
    chunksize = 10000

    def __init__(self, task, table_type, n_proc=32, chunksize=10000, test=False):
        """
        TODO think about how to extend to multiple tasks/table types

        Attributes:
            task (str): the RDD setting to target
            table_type (str): the Optum "table" type to target
            n_proc (int): the number of processor to allocate
            chunksize (int): the number of stata rows to read in

            code_dict (dict): maps task to the codes to search for
            out_dir (str): the output directory 
            config (dict): the configuration loaded via config.ini
        """

        if task in SUPPORTED_TASKS:
            self.task = task
        else:
            raise Exception("Provided task {} not supported".format(task))

        if table_type in SUPPORTED_TABLE_TYPES:
            self.table_type = table_type
        else:
            raise Exception("Provided table type {} not supported".format(table_type))

        self.n_proc = n_proc
        self.test = test
        self.chunksize = chunksize

        if self.test:
            self.chunksize = 100
        
        # load configuration
        config = configparser.ConfigParser()
        config.read("config.ini")
        default_config = config['default']
        self.config = default_config

        # prepare configuration settings
        self.load_task_config()

        # prepare output directories
        # TODO should the task be an additional level in the dir hierarchy?
        self.out_dir = os.path.join(default_config['output_dir'], 
                                    self.task, 
                                    self.table_type)
        
        if not os.path.exists(self.out_dir):
            print(f"Created directory {self.out_dir}")
            os.makedirs(self.out_dir)


    def load_task_config(self):
        """
        Loads the configuration files needed for a particular task.

        Assigns the self.code_dict instance dictionary

        Returns:
            None 
        """

        print(f"Loading {self.task} task configuration...")

        with open(f"json/codes/{self.task}.json") as f:
            task_dict = json.load(f)
            table_dict = task_dict[self.table_type]

        if self.table_type != "merge":
            # we're running an extract task, generate code dictionary
            code_dict = {}
            for subtask in table_dict:
                codes = []
                for code_type in table_dict[subtask]['codes']:
                    codes += table_dict[subtask][code_type]

                code_dict[subtask] = codes
            
            self.code_dict = code_dict

        else:
            # we're running a merge task, generate pre and post paths
            self.window = table_dict["window"]
            self.pre_type = table_dict['pre']['table']
            self.post_type = table_dict['post']['table']

            self.pre_path = os.path.join(
                self.config['output_dir'],
                self.task,
                self.pre_type,
                "{{}}q{{}}_{}.parq".format(table_dict['pre']['task'])
            )

            
            self.post_path = os.path.join(
                self.config['output_dir'],
                self.task,
                self.post_type,
                "{{}}q{{}}_{}.parq".format(table_dict['post']['task'])
            )

            # populate columns to propragate in the merge
            # TODO test and refactor _merge
            if "pre_cols" in table_dict:
                self.pre_cols = table_dict["pre_cols"]
            else:
                self.pre_cols = []
            
            if "post_cols" in table_dict:
                self.post_cols = table_dict["post_cols"]  
            else:
                self.post_cols = []

    def __str__(self):
        return f"task: {self.task}, tables: {self.table_type}, # processors: {self.n_proc}"


    def run(self):
        """
        Runs the pipeline.
        """

        start_time = time.time()
        print(f"Running pipeline as {self}")

        if self.table_type == "merge":
            self._merge()
        else:
            self._extract()
        
        end_time = time.time()

        print("Task completed, {:.2f} sec".format(end_time - start_time))


    def _extract(self):
        """Extraction tasks"""
        func_args = []

        for yr in range(START_YR, END_YR + 1):
            for q in range(1, 5):
                args = []

                # file path params
                in_file = os.path.join(self.config['data_dir'], 
                                       f"ses_{self.table_type}{yr}q{q}.dta")
                out_path = self.out_dir
                out_name = f"{yr}q{q}_{{}}"

                args.append(in_file)
                args.append(out_path)
                args.append(out_name)
                
                args.append(self.table_type) # code_type
                args.append(self.code_dict) # code_dict
                args.append(self.chunksize) # chunksize
                args.append(self.test) # test flag
                
                func_args.append(args)
        # TODO double check if this can scale generically to all table types
        #if self.table_type in  ['proc', 'diag', 'lr']:
        extract_func = extract.code_extract

        with multiprocessing.Pool(self.n_proc) as pool:
            pool.starmap(extract_func, func_args)

    def _merge(self):
        """Merge task. 
        
        Uses self.pre/post_cols to optionally provide a list of columns to
        propagate from the pre and post dataframes.

        TODO need to consider memory concerns, and which columns to select when
        reading in.

        TODO when working with lab results, will likely need to include a "cols"
        attribute so that we can properly propagate data through the merge.
        """
        sel_cols = ['patid', 'fst_dt'] 

        # build pre and post_df
        print("Building pre and post dataframes...")
        pre_df = pd.DataFrame()
        post_df = pd.DataFrame()
        for yr in range(START_YR, END_YR):
            if self.test and yr > START_YR:
                break
            for q in range(1,5):
                pre = pd.read_parquet(self.pre_path.format(yr, q),
                                      columns=sel_cols + self.pre_cols)
                pre_df = pd.concat([pre_df, pre])
                
                # TODO temp hack to get rx working
                if self.post_type == 'r':
                    sel_post_cols =  sel_cols[:1] + self.post_cols
                else:
                    sel_post_cols =  sel_cols + self.post_cols
                post = pd.read_parquet(self.post_path.format(yr, q),
                                      columns=sel_post_cols)

                post_df = pd.concat([post_df, post])

        # process pre lr
        if self.pre_type == 'lr':
            pre_df = merge.lr_prep(pre_df)
        
        # process post rx
        if self.post_type == 'r':
            if self.test:
                print("formatting rx...")
            post_df = merge.rx_prep(post_df)

        # get indicator dataframe
        print("Generating indicators for diagnosis matches...")
        indicator_df = merge.gen_diag_indicator(pre_df, post_df, self.window)

        # need to convert date_diff to number for parquet
        indicator_df['date_diff'] = indicator_df['date_diff'].dt.days#.astype(int)

        if self.test:
            print(indicator_df.head(10))
            print(indicator_df.shape)
            hit = indicator_df[indicator_df['indicator'] == 1]
            print(hit.shape)
            print(hit.head(10))

        print("Generating prior diagnosis indicators...")
        indicator_df = merge.gen_prior_diag_indicator(indicator_df, post_df)

        # load ses and left join
        print("Loading and merging ses table...")
        ses_df = pd.read_parquet(os.path.join(
            self.config['output_dir'],
            "ses/ses.parq"
        ))
        indicator_df = merge.left_join_patid(indicator_df, ses_df)

        """     
        ses_df['patid'] = ses_df['patid'].astype(str)
        ses_df = ses_df.sort_values(by="patid")
        indicator_df = indicator_df.merge(ses_df, on="patid", how='left')
        """

        # load mbr and left join
        print("Loading and merging mbr table...")
        
        mbr_df = pd.read_parquet(os.path.join(
            self.config['output_dir'],
            "ses/mbr_first.parq"
        ))

        indicator_df = merge.left_join_patid(
            indicator_df, mbr_df, 
            suffixes=("_ses", "_mbr") # since the ses table was merged first
        )
        """         
        mbr_df['patid'] = mbr_df['patid'].astype(str)
        mbr_df = mbr_df.sort_values(by="patid")
        indicator_df = indicator_df.merge(mbr_df, on="patid", how='left') 
        """
        # compute age
        # this will be according to fst_dt_pre, rounded down to the year
        indicator_df['age'] = indicator_df['fst_dt_pre'].dt.year - indicator_df['yrdob']
        
        test_suffix = "_test" if self.test else ""

        indicator_df.to_parquet(os.path.join(
            self.config["output_dir"],
            self.task,
            self.table_type,
            "{}.parq".format(self.task + test_suffix)
        ))

