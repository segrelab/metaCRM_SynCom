"""
Run simulations for the multistability analysis.
"""

import sys
import os
import pandas as pd
import numpy as np
import random

# Load data consistently using new versions of the code.
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../figure4/"))  # Import code to make fig 4E

import figure4.process_data as process_data
import utils

from argparse import ArgumentParser
from pathlib import Path

def progress_bar(current, total, bar_length=40):
    # Calculate percentage and fill length
    fraction = current / total
    arrow = int(fraction * bar_length - 1) * '=' + '>'
    padding = int(bar_length - len(arrow)) * ' '
    
    # Format ending to handle complete state cleanly
    ending = '\n' if current == total else ''
    
    # Overwrite the current line
    print(f"Progress: [{arrow}{padding}] {fraction:.1%}", end='\r', flush=True)
    if ending:
        print(ending, end='')

def shuffle_labels(t0_abun: list):
    t0_pseudo = [x if x > 0 else x + 0.0001 for x in t0_abun]
    n_perm = 100

    t0_shuffles = [random.sample(t0_pseudo, len(t0_pseudo)) for _ in range(n_perm)]
    return t0_shuffles

if __name__ == "__main__":
    parser = ArgumentParser(description="Run multistability analysis.")
    parser.add_argument("--out", required=True, help="Directory to save output data.")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    random.seed(42)

    t0_abun = utils.t0_wc_data()

    t0_shuffles = shuffle_labels(t0_abun.values.tolist()[0])

    passage_list = []
    sp_abun_list = []

    for i, sp_abun in enumerate(t0_shuffles):
        passages, _ = process_data.simulate_whole_community_exp(tfs=4, crossfeeding=True, t0_abun=sp_abun)
        passage_list.append(passages)
        progress_bar(i+1, n_perm)

    all_passages = pd.concat(passage_list, keys=np.arange(n_perm)).reset_index(level=0, names="permutation")
    all_passages.to_csv(os.path.join(args.out, "sim_whole_comm/t0_shuffles.csv"))