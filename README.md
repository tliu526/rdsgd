# Automated Detection of Causal Inference Opportunities: Regression Discontinuity Subgroup Discovery

Code repository for results accompanying the paper, accepted to TMLR 2023: [https://openreview.net/forum?id=cdRYoTyHZh](https://openreview.net/forum?id=cdRYoTyHZh)

This project has the following structure:

~~~~
rdsgd/
├── data/                     # processed data files
├── figures/                  # figures for paper
├── results/                  # Notebooks that reproduce all main results and figures
├── src/                      # copies of repositories used for analysis
|    ├── herlands-lord3       # repository from Herlands et al. 2018 for algorithm comparison
|    ├── optum-pipeline/      # repository for processing Optum claims data
|    └── rdd-discovery/       # repository for implementing RDSGD and experiments
~~~~

Note that this project utilizes three separate version-controlled repositories, which we provide copies of under `src/`. We also provide a streamlined set of notebooks in `results/` to easily reproduce the figures and results presented in the paper. Python version >=3.8 and package versions specified in `src/rdd-discovery/requirements.txt` needed to ensure pickle compatability.

**TODO**: Package up `src/rdd-discovery/` as a standalone Python package.

## Steps to reproduce figures and results in main text

1. Run notebooks in `results/`: 
    - `01_neff_simulations.ipynb`
    - `02_sim_single_cov.ipynb`
    - `03_sim_multidim.ipynb`
    - `04_medical_claims.ipynb`
2. Results and figures will be embedded in the Jupyter notebooks, as well as written to `figures/`.

## Steps to reproduce simulated experiments

Note that these experiments take 4-8 hours to run each, depending on the number
of cores available and parallelism used.

### Heterogeneity in one covariate

Run in sequential order: 
1. `src/rdd-discovery/experiments/baseline_experiments.sh`
2. `src/rdd-discovery/experiments/pre_herlands_experiments.py`
3. `src/herlands-lord3/src/herlands_sim.ipynb`
4. `src/rdd-discovery/experiments/run_blend_experiments.py`

### Heterogeneity in multiple covariates

Run `src/rdd-discovery/run_multidim_experiments.py`

## Medical claims case study

Since the claims dataset is private, we cannot provide the source data to replicate the RDSGD search. However notebooks that execute the experiments for the three case studies can be found in:

- `src/rdd-discovery/notebooks/tmlr/05_breast_cancer_discovery.ipynb`
- `src/rdd-discovery/notebooks/tmlr/06_colon_cancer_discovery.ipynb`
- `src/rdd-discovery/notebooks/tmlr/07_diabetes_discovery.ipynb`

## Hyperparameters

Hyperparameter values as specified in the Supplementary Material for all simulated data and model training can be found in the accompanying source code. In particular, see `src/rdd-discovery/utils/rddd.py` for algorithm details and `src/rdd-discovery/utils/sim.py` for simulation details.
