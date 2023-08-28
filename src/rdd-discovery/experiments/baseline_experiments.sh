#!/bin/bash
for offset in {0..400..100}
do
    echo "$offset"
    time python run_blend_experiments.py $offset ../results/kdd/baseline_discovery/ baseline 
done