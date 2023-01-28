"""
Script for preparing data for Herlands et al. experiments.

Once data is generated, run herlands_sim.ipynb in /src/herlands-lord3/src to run 
experiment.
"""

import os
import sys
sys.path.append("../")
from utils.sim import generate_cont_blended_rdd

if __name__ == "__main__":
    HERLANDS_DIR = "../../herlands-lord3/data/blend_sim"

    gaps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    seeds = range(0, 500)
    n_samples = 1000

    for gap in gaps:
        for seed in seeds:
            df = generate_cont_blended_rdd(
                n=n_samples,
                seed=seed,
                fuzzy_gap=gap
            )
            
            df.to_csv(os.path.join(HERLANDS_DIR, f"gap{gap}_seed{seed}.csv"))