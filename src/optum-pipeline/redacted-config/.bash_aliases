#!/bin/bash
# Loads application modules for use in the interactive ibash, sets up env variables
module load emacs/25.3;
module load vim/7.4;
module load git/2.17.0;
module load python/3.6.5;
module load R;

export XDG_RUNTIME_DIR=""
export DISPLAY=localhost:0

