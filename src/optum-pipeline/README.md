# Optum-pipeline

A collection of python utilities and data pipeline for processing Optum data.

# Project structure


~~~
optum-pipeline
├── json/             # json config files
|   ├── codes/        # ndc, snomed, icd codes for particular RDDs        
|   └── cols/         # column mappings for the particular Optum table format
|
├── redacted-config/  # REDACTED-specific config files
├── optumpipe/        # module directory
|   ├── __main__.py   # argparsing, data pipe creation and execution
|   ├── med.py        # data extraction for medical Optum tables
|   └── pipe.py       # class for configuring and running data pipelines
|
├── scripts/          # one-off scripts
├── tests/            # unit tests
├── setup.py          # python setup
~~~

# Running the pipeline

To view the Optum pipeline commandline arguments:

~~~bash
$ python -m optumpipe -h
~~~

To run unit tests:

~~~bash
$ python -m unittest discover tests/
~~~

All python scripts should be run from the root of the project directory.

# Getting Started

First, set up VMWare Client and verify that you are able to ssh into REDACTED.

Next, copy `REDACTED-config/.bash_aliases` into your home directory on REDACTED. This will load all the modules needed (python, git) for development once you start an interactive bash session.

Start an interactive session by running `ibash`. This will take you to one of the work servers in the REDACTED clusters. If your `.bash_aliases` file is set up correctly, then python and git should be availble in your interactive session. Verify this by running `python` -- the interpreter session should show that the python version is 3.6.5.

# Connecting to Jupyter notebooks

**Prerequisite:** ensure you have python enabled for your session and your `.bash_aliases` file is set up, as detailed in "Getting Started."

1. ssh into REDACTED and enter an interactive bash session: `ibash`. Note which server you land on, likely `harrower`.
2. `cd` into your working directory of choice.
3. Start a jupyter notebook server by executing `jupyter-notebook --no-browser --port=8889 --ip=0.0.0.0`. Note that the port number may be different if 8889 is already taken.
4. Open a new ssh window inside the VDI, and open up an ssh tunnel by executing: `ssh -fNL 8889:{server}:8889 username@REDACTEDsub`, where `{server}` is the server you landed on in step 1.
5. Open a browser window inside the VDI and navigate to http://localhost:8889. Provide the password token of your notebook server from step 3 and you should be good to go!

**Note:** make sure to save your notebook states often as the REDACTED servers may sometimes close ports after periods of inactivity.

# Data storage and workspace

You should clone this repository into your home directory, but any data files should **not** be stored in your home directory. Instead, store any intermediate data files you generate in `/REDACTED/{yourname}_scratch`. Finalized data files will be stored in `REDACTED` under the appropriate directory.

# Legacy data extraction process

The legacy data extraction process relied largely on the `med_extract.py` script file, which is partially hardcoded with procedure codes and claims. It adhered to the Optum 2017 specification, so there are no guarantees that the code works with the Optum 2019 specification. The process is roughly as follows:

- Use bsub scripts to schedule data pulls by calling `med_extract.py`, filtering specific columns from the stata files and converting to pandas dataframes. These dataframes are of the form {yr}_{qtr}_{table}, mirroring the structure of the raw stata files.
- Use an additional script to merge the pandas dataframes into a single frame. This final frame of analysis will have patients as rows, and relevant outcomes/covariates as columns, and can be used for downstream machine learning/causal inference analysis.
