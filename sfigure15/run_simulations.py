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

def shuffle_labels(t0_abun: list, n_perm):
    t0_pseudo = [x if x > 0 else x + 0.0001 for x in t0_abun]

    t0_shuffles = [random.sample(t0_pseudo, len(t0_pseudo)) for _ in range(n_perm)]
    return t0_shuffles

if __name__ == "__main__":
    parser = ArgumentParser(description="Run multistability analysis.")
    parser.add_argument("--out", required=True, help="Directory to save output data.")
    parser.add_argument("--run_perms", required=True, help="Option to run large n permutations.")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    random.seed(42)
    n_perm = 100

    if args.run_perms == True:
        t0_abun = utils.t0_wc_data()

        t0_shuffles = shuffle_labels(t0_abun.values.tolist()[0], n_perm)

        passage_list = []
        sp_abun_list = []

        for i, sp_abun in enumerate(t0_shuffles):
            passages, _ = process_data.simulate_whole_community_exp(tfs=4, crossfeeding=True, t0_abun=sp_abun)
            passage_list.append(passages)
            progress_bar(i+1, n_perm)

        all_passages = pd.concat(passage_list, keys=np.arange(n_perm)).reset_index(level=0, names="permutation")
        all_passages.to_csv(os.path.join(args.out, "sim_whole_comm/t0_shuffles.csv"))

    # Run simulations for different normalization schemes
    t0 = utils.t0_wc_data()
    deviation_factor = utils.deviation_factor()
    cp_num_df = utils.cn_16s()
    od_cfu_df = utils.od_to_cfu()

    norm_factors = pd.concat([t0, deviation_factor, cp_num_df, od_cfu_df])
    norm_factors.loc['OD_cfu_16s'] = norm_factors.loc['Average CFU ml^-1'] * norm_factors.loc['16s_copy_number']

    sp_t0_df, _ = process_data.simulate_whole_community_exp(crossfeeding=True, t0_abun=norm_factors.loc['crm_t0'])
    sp_cfu_df, _ = process_data.simulate_whole_community_exp(crossfeeding=True, od_cfu_conv=norm_factors.loc['Average CFU ml^-1'])
    sp_t0_cfu_df, _ = process_data.simulate_whole_community_exp(crossfeeding=True, t0_abun=norm_factors.loc['crm_t0'], od_cfu_conv=norm_factors.loc['Average CFU ml^-1'])

    # Save data
    os.makedirs(os.path.join(args.out, "sim_whole_comm/normalizations"), exist_ok=True)
    sp_t0_df.to_csv(os.path.join(args.out, "sim_whole_comm/normalizations/sp_t0.csv"))
    sp_cfu_df.to_csv(os.path.join(args.out, "sim_whole_comm/normalizations/sp_cfu.csv"))
    sp_t0_cfu_df.to_csv(os.path.join(args.out, "sim_whole_comm/normalizations/sp_t0_cfu.csv"))