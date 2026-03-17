"""

"""

# All
import sys
import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import textalloc as ta
import statistics
from scipy.stats import sem
import seaborn as sns
from matplotlib.lines import Line2D 

# Fig 3A
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Circle

# Incl helper functions from utils
sys.path.append(os.path.abspath("../"))
import utils

def conversion(x):
    """Handles mixed datatype input from csv"""
    try:
        val = eval(x)
        return val if isinstance(val, list) else float(val)
    except:
        return float(x)

def plot_sim_v_exp_abun(sim_results: pd.DataFrame, exp_results: pd.DataFrame, outfig: Path) -> None:
    n = len(utils.sps)
    fig = plt.figure(figsize=(12, 10))
    
    colormap = utils.get_species_colormap()
    species_ids = utils.sps
    species_names = utils.sps_names
    
    #upper-triangle layout
    gs = plt.GridSpec(n, n, figure=fig, wspace=0.2, hspace=0.2)

    #track where to put x-labels
    bottom_subplot_for_col = {}
    for i in range(n):
        for j in range(i+1, n):
            bottom_subplot_for_col[j] = i

    light_hatch = (0, 0, 0, 0.25)
    
    #plot co-culture bars
    for i in range(n):
        for j in range(i+1, n):
            i_id = species_ids[i]
            j_id = species_ids[j]

            ax = fig.add_subplot(gs[i, j])

            color_i = colormap[species_names[i]]
            color_j = colormap[species_names[j]]

            #get sim final abundances
            sim_i_norm = sim_results.loc[j_id, i_id]
            sim_j_norm = sim_results.loc[i_id, j_id]

            #get exp final abundances
            sp_i_exp = exp_results.loc[j_id, i_id]
            sp_j_exp = exp_results.loc[i_id, j_id]
            
            exp_i_norm = sum(sp_i_exp) / len(sp_i_exp) if isinstance(sp_i_exp, list) else sp_i_exp
            exp_j_norm = sum(sp_j_exp) / len(sp_j_exp) if isinstance(sp_j_exp, list) else sp_j_exp


            #plot bars with hatch pattern for simulation
            ax.bar(0, sim_i_norm, color=color_i, edgecolor=light_hatch, hatch='//')
            ax.bar(0, sim_j_norm, bottom=sim_i_norm, color=color_j, edgecolor=light_hatch,hatch='//')
            ax.bar(1, exp_i_norm, color=color_i)
            ax.bar(1, exp_j_norm, bottom=exp_i_norm, color=color_j)

            #remove ticks, sum to 1
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(0, 1)
            
            # Add row species label with colored circle
            if j == n-1:
                # transform from axes -> figure coordinates
                x_fig, y_fig = ax.transAxes.transform((1.05, 0.5))
                x_fig, y_fig = fig.transFigure.inverted().transform((x_fig, y_fig))
                fig.text(x_fig + 0.03, y_fig, species_names[i], va='center', ha='left', fontsize=10)
                circle = Circle((x_fig+0.01, y_fig), 0.005, color=color_i, transform=fig.transFigure, clip_on=False)
                fig.patches.append(circle)

            # Add column species label with colored circle
            if i == bottom_subplot_for_col[j]:
                x_fig, y_fig = ax.transAxes.transform((0.5, -0.3))
                x_fig, y_fig = fig.transFigure.inverted().transform((x_fig, y_fig))
                fig.text(x_fig, y_fig, species_names[j], va='top', ha='center', fontsize=10, rotation=60)
                circle = Circle((x_fig, y_fig), 0.005, color=color_j, transform=fig.transFigure, clip_on=False)
                fig.patches.append(circle)

    # legend only for Simulation vs Experimental
    legend_handles = [
        Patch(facecolor='white', hatch='//', label='Simulation', edgecolor=light_hatch),
        Patch(facecolor='white', label='Experimental', edgecolor=light_hatch)
    ]

    fig.legend(
        handles=legend_handles,
        loc='lower left',
        bbox_to_anchor=(0.18, 0.55),
        fontsize=12
    )
    fig.text(0.4, 0.9, 'Pairwise Co-Culture Simulated vs. Experimental', fontsize=14)
    
    plt.savefig(outfig, dpi=600, bbox_inches='tight', pad_inches=0.1)

def plot_average_abundance(sim_results: pd.DataFrame, exp_results: pd.DataFrame, outfig: Path) -> None:
    avg_dict = {}
    avg_dict['sp'] = utils.sps
    avg_dict['exp_avg'] = []
    avg_dict['sim_avg'] = []
    
    avg_dict['exp_std'] = []
    avg_dict['sim_std'] = []
    color_map = utils.get_species_colormap(False)
    
    for sp in utils.sps:
        sp_i_sim = sim_results.loc[sim_results.index != sp, sp].tolist()
        exp_data_i = exp_results.loc[exp_results.index != sp, sp].tolist()
        sp_i_exp = [sum(i) / len(i) if isinstance(i, list) else i for i in exp_data_i]
        
        avg_dict['exp_avg'].append(statistics.mean(sp_i_exp))
        avg_dict['sim_avg'].append(statistics.mean(sp_i_sim))
        
        avg_dict['exp_std'].append(sem(sp_i_exp))
        avg_dict['sim_std'].append(sem(sp_i_sim))
        
        
    avg_df = pd.DataFrame.from_dict(avg_dict)
    
    avg_df["color"] = avg_df["sp"].map(color_map)
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    
    
    for x, y, xerr, yerr, c in zip(avg_df.sim_avg,
                                   avg_df.exp_avg,
                                   avg_df.sim_std,
                                   avg_df.exp_std,
                                   avg_df.color):
        ax.errorbar(x, y,
                    xerr=xerr, yerr=yerr,
                    fmt='None',
                    ecolor=c,
                    elinewidth=1,
                    capsize=2,
                    linestyle='--',
                    zorder=2)
        
        
    ax.scatter(avg_df.sim_avg, avg_df.exp_avg, c=avg_df.color, s=50, label=avg_df.sp)
    
    ax.axline((0,0), (1,1), ls='--', c="grey")
    
    
    ax.set_title("Average Abundance in Co-Culture by Species", size=14)
    
    ax.set_xlabel('Average Predicted Species Abundance', size=12)
    ax.set_ylabel('Average Measured Species Abundance', size=12)
    
    plt.savefig(outfig, dpi=600, bbox_inches='tight', pad_inches=0.1)

def plot_pearson_corr(pearson_rs: pd.DataFrame, outfig: Path) -> None:
    fig_corr = plt.figure(figsize=(5,5))
    ax_corr = fig_corr.add_subplot(111)

    pearson_no_leak_arr = pearson_rs['leakage off'].tolist()
    pearson_leak_arr = pearson_rs['leakage on'].tolist()
    
    text_list = list(utils.get_species_colormap().values())
    
    sc = ax_corr.scatter(pearson_no_leak_arr, pearson_leak_arr, c=text_list, s=70)

    ax_corr.set_xlabel('Cross-feeding off', size=14)
    ax_corr.set_ylabel('Cross-feeding on', size=14)
    ax_corr.set_title('Pearson Correlations by Species', size=16)

    xy_min = np.nanmin([pearson_no_leak_arr, pearson_leak_arr])
    xy_max = np.nanmax([pearson_no_leak_arr, pearson_leak_arr])

    ax_corr.axline((0,0), (1,1), ls='--', c="grey")

    ax_corr.set_xlim(0,1)
    ax_corr.set_ylim(0,1)
        
    plt.savefig(outfig, dpi=600, bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type = Path, required=True, help="Directory with input CSV files.")
    parser.add_argument("--out", type = Path, required=True, help="Directory to save output figures.")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Handle complex data
    converters = {str(sp):conversion for sp in utils.sps}
    
    # Load processed data
    exp_ratios = pd.read_csv(args.data_dir / "exp_pairwise_coculture/exp_coculture_ratios.csv", index_col=0, converters=converters)
    exp_ratios.columns = exp_ratios.columns.astype(str)
    exp_ratios.index = exp_ratios.index.astype(str)
    
    sim_ratios = pd.read_csv(args.data_dir / "sim_pairwise_coculture/sim-coculture-ratio_leakage.csv", index_col=0)
    sim_ratios.columns = sim_ratios.columns.astype(str)
    sim_ratios.index = sim_ratios.index.astype(str)
    
    pearson_rs = pd.read_csv(args.data_dir / "sim_pairwise_coculture/sim-exp_pearson.csv", index_col=0)
    
    # Plot figures
    plot_sim_v_exp_abun(sim_ratios, exp_ratios, args.out / "Mfig3a.png")
    plot_average_abundance(sim_ratios, exp_ratios, args.out / "Mfig3b.png")
    plot_pearson_corr(pearson_rs, args.out / "Mfig3c.png")