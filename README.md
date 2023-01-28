# Automated Detection of Causal Inference Opportunities: Regression Discontinuity Subgroup Discovery

This project has the following structure:

~~~~
aaai-rdd-subgroup-discovery
├── data/                     # processed data files
├── figures/                  # figures for paper
├── results/                  # Notebooks that reproduce all main results and figures
├── src/                      # copies of repositories used for analysis
|    ├── herlands-lord3       # repository from Herlands et al. 2018 for algorithm comparison
|    ├── optum-pipeline/      # repository for processing Optum claims data
|    └── rdd-discovery/       # repository for implementing RDSGD and experiments
~~~~

Note that this project utilizes three separate version-controlled repositories, which we provide anonymized versions of under `src/`. We also provide a streamlined set of notebooks in `results/` to easily reproduce the figures and results presented in the paper. Python version >=3.8 and package versions specified in `src/rdd-discovery/requirements.txt` needed to ensure pickle compatability.

## Steps to reproduce figures and results in main text

1. Run notebooks in `results/` in sequential order: 
    - `01_sim_single_cov.ipynb`
    - `02_sim_multiple_cov.ipynb`
    - `03_medical_claims.ipynb`
    - `04_app_tau_power.ipynb`
2. Results and figures will be embedded in the Jupyter notebooks, as well as written to `figures/`.

## Steps to reproduce simulated experiments

Note that these experiments take 4-8 hours to run each, depending on the number
of cores available and parallelism used.

### Heterogeneity in one covariate

Run in sequential order: 
1. `src/rdd-discovery/experiments/baseline_experiments.sh`
2. `src/rdd-discovery/experiments/pre_herlands_experiments.py`
3. `src/herlands-lord3/src/herlands_sim.ipynb`
4. `src/rdd-discovery/notebooks/aaai/01_single_sim_results.ipynb`

### Heterogeneity in multiple covariates

Run `src/rdd-notebooks/aaai/02_multi_sim_results.ipynb`

## Medical claims case study

Since the claims dataset is private, we cannot provide the source data to replicate the RDSGD search. However notebooks that execute the experiments for the three case studies can be found in:

- `src/rdd-discovery/notebooks/aaai/03_breast_cancer_discovery.ipynb`
- `src/rdd-discovery/notebooks/aaai/04_colon_cancer_discovery.ipynb`
- `src/rdd-discovery/notebooks/aaai/05_diabetes_discovery.ipynb`

## Hyperparameters

Hyperparameter values as specified in the Technical Appendix for all simulated data and model training can be found in the accompanying source code. In particular, see `src/rdd-discovery/utils/rddd.py` for algorithm details and `src/rdd-discovery/utils/sim.py` for simulation details.
