#!/bin/bash

# Activate Conda environment
conda activate crm-syncom

# Create output folder if not existing
OUTPUT_FOLDER=figures
mkdir -p $OUTPUT_FOLDER

DATA_FOLDER=data

#= Figure 2 =#

python figure2/plot_figures.py --data_dir $DATA_FOLDER --out $OUTPUT_FOLDER

#= Figure 3 =#



#= Figure 4 =#

python figure4/plot_figures.py --data_dir $DATA_FOLDER --out $OUTPUT_FOLDER

#= Figure 5 =#

python figure5/plot_figures.py --epistasis $DATA_FOLDER/epistasis_vals.csv --out $OUTPUT_FOLDER

#= Figure 6 =#



