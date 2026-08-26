#!/bin/bash
# Usage: bash plot_all.sh [output_dir_for_figs]   (default: figures)
 
# Output folder: first argument if given, otherwise "figures"
OUTPUT_FOLDER=${1:-figures}
DATA_FOLDER=data

echo "=============================================="
echo " Regenerating all manuscript figures"
echo "=============================================="

# The conda environment should already be active (see README):
#   conda env create -f environment.yml
#   conda activate crm-syncom

ENV_NAME=crm-syncom

if [ "$(basename "${CONDA_PREFIX:-}")" != "$ENV_NAME" ]; then
    echo "[error] The '$ENV_NAME' conda environment is not active."
    echo "[error] Please run the following first:"
    echo "[error]     conda env create -f environment.yml   # first time only"
    echo "[error]     conda activate $ENV_NAME"
    exit 1
fi

echo "[setup] Conda environment: $ENV_NAME (active)"
echo "[setup] Python:            $(which python)"
 
echo "[setup] Data folder:   $DATA_FOLDER"

if [ -n "$1" ]; then
    echo "[setup] Output folder: $OUTPUT_FOLDER (provided)"
else
    echo "[setup] Output folder: $OUTPUT_FOLDER (default; pass one as: bash plot_all.sh <output_dir>)"
fi
 
if [ ! -d "$OUTPUT_FOLDER" ]; then
    echo "[setup] Output folder does not exist -- creating it."
    mkdir -p "$OUTPUT_FOLDER"
fi

DATA_FOLDER=data

#= Figure 2 =#
echo "----------------------------------------------"
echo "[Figure 2] Generating..."
python figure2/plot_figures.py --data_dir $DATA_FOLDER --out $OUTPUT_FOLDER
echo "[Figure 2] Done."

#= Figure 3 =#
echo "----------------------------------------------"
echo "[Figure 3] Generating..."
python figure3/plot_figures.py --data_dir $DATA_FOLDER --out $OUTPUT_FOLDER
echo "[Figure 3] Done."

#= Figure 4 =#
echo "----------------------------------------------"
echo "[Figure 4] Generating..."
python figure4/plot_figures.py --data_dir $DATA_FOLDER --out $OUTPUT_FOLDER
echo "[Figure 4] Done."

#= Figure 5 =#
echo "----------------------------------------------"
echo "[Figure 5] Generating..."
python figure5/plot_figures.py --epistasis $DATA_FOLDER/epistasis_vals.csv --out $OUTPUT_FOLDER
echo "[Figure 5] Done."

#= Figure 6 =#
echo "----------------------------------------------"
echo "[Figure 6] Generating..."
python figure6/plot_figures.py --data_dir $DATA_FOLDER --out $OUTPUT_FOLDER
echo "[Figure 6] Done."


echo ""
echo "=============================================="
echo " All figures generated!"
echo " Files written to: $(cd "$OUTPUT_FOLDER" && pwd)"
echo "=============================================="