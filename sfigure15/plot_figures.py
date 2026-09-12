import sys
import os
import pandas as pd
import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import seaborn as sns
import statistics
import scipy
from scipy.stats import spearmanr, pearsonr, linregress

# Load data consistently using new versions of the code.
sys.path.append(os.path.abspath("/projectnb/cometsfba/rowanon/projects/metaCRM_SynCom/"))
sys.path.append(os.path.abspath("/projectnb/cometsfba/rowanon/projects/metaCRM_SynCom/figure4/"))  # Import code to make fig 4E

import figure4.process_data as process_data


from argparse import ArgumentParser
from pathlib import Path

import utils

INV_MAP = {1:'1319',2:'1320',3:'1321',4:'1323',5:'1324',6:'1325',7:'1327',8:'1329',9:'1330',10:'1331',11:'1334',12:'1337',13:'1338',14:'1336',15:'1538',16:'1602',17:'1597'}

def plot_final_abundances(final_abun, outfile):
    colormap = utils.get_species_colormap(False)

    fig, ax = plt.subplots(figsize=(18, 6))

    bottom = np.zeros(n_perm)
    x = np.arange(n_perm)

    for i, species in enumerate(species_cols):
        values = final_abun[:, i]
        ax.bar(x, values, bottom=bottom, color=colormap[species], label=species, width=0.9)
        bottom += values

    ax.set_xlabel('Simulation run')
    ax.set_ylabel('Relative abundance')
    ax.set_title('Final community composition across 100 simulation runs')
    ax.set_xlim(-0.5, n_perm - 0.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=1, style="italic")

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile=os.path.join(args.out, "SFig15a.pdf"))

def plot_distributions(final_abun, strategies, outfile=None):
    arr = np.asarray(final_abun)
    n_sp = arr.shape[1]

    fig, axes = plt.subplots(3, 5, figsize=(13, 7), constrained_layout=True)

    for i, ax in enumerate(axes.flat[:n_sp]):
        ax.violinplot(arr[:, i], positions=[0], showextrema=False,
                    widths=0.8, vert=False)

        for (name, vals) in strategies.items():
            ax.scatter(np.asarray(vals)[i], 0, s=22, zorder=3,
                    label=name if i == 0 else None)

        ax.set_title(utils.sps_names[i], fontsize=9)
        ax.set_yticks([])
        ax.set_ylim(-0.6, 0.6)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
        ax.tick_params(axis="x", labelsize=8)

    for ax in axes.flat[n_sp:]:
        ax.set_visible(False)

    fig.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    if outfile:
        plt.savefig(outfile=os.path.join(args.out, "SFig15b.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser(description="Plot 16s normalizations.")
    parser.add_argument("--out", required=True, help="Directory to save output figures.")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    passage_list = pd.read_csv("./data/sim_whole_comm/t0_shuffles.csv")

    species_cols = passage_list.columns.drop(['passage', 'permutation'])
    final_abun = passage_list.loc[passage_list.groupby('permutation')['passage'].idxmax(), species_cols].to_numpy()

    t0 = utils.t0_wc_data()
    deviation_factor = utils.deviation_factor()
    cp_num_df = utils.cn_16s()
    od_cfu_df = utils.od_to_cfu()

    norm_factors = pd.concat([t0, deviation_factor, cp_num_df, od_cfu_df])

    norm_factors.loc['OD_cfu_16s'] = norm_factors.loc['Average CFU ml^-1'] * norm_factors.loc['16s_copy_number']
    norm_factors

    sp_df = pd.read_csv('./data/sim_whole_comm/wc_sp_sim.csv')
    sp_nocross_df = pd.read_csv('./data/sim_whole_comm/wc_sp_sim_nc.csv')
    sp_t0_df = pd.read_csv('./data/sim_whole_comm/normalizations/sp_t0.csv')
    sp_cfu_df = pd.read_csv('./data/sim_whole_comm/normalizations/sp_cfu.csv')
    sp_t0_cfu_df = pd.read_csv('./data/sim_whole_comm/normalizations/sp_t0.csv')

    sp_final = sp_df.iloc[-1]
    nocross_final = sp_nocross_df.iloc[-1]
    t0_final = sp_t0_df.iloc[-1]
    odcfu_final = sp_cfu_df.iloc[-1]
    t0cfu_final = sp_t0_cfu_df.iloc[-1]

    strategies = {"equal": sp_final, "no crossfeeding": nocross_final, "measured t0": t0_final, "equal OD": odcfu_final, "t0 and OD": t0cfu_final}

    plot_final_abundances(final_abun, outfile=os.path.join(args.out, "Sfig_16a.pdf"))
    plot_distributions(final_abun, strategies, outfile=os.path.join(args.out, "Sfig_16b.pdf"))