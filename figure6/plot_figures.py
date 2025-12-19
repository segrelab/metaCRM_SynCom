"""
Figure 6: Plot all main and Supplementary figures for this section
"""
from argparse import ArgumentParser
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind
import numpy as np
sys.path.append(".")
import utils


def plot_shannon_hist(sim_subcom, exp_subcom, outfile=None):
    #plot distribution of shannon diversity from all 3-species simulations
    fig = sns.histplot(sim_subcom['Shannon diversity sim'], bins=30)

    #subset to tested communities to draw lines on histogram plot
    subset_comms = exp_subcom['Subcommunity species'].unique()  # get unique values from df2
    sim_subcom_selected = sim_subcom[sim_subcom['Subcommunity species'].isin(subset_comms)]

    for value in sim_subcom_selected['Shannon diversity sim'].values:
        plt.axvline(value, color='red', linestyle='dashed', linewidth=1, ymax=0.25)

    plt.xlabel("Shannon Entropy", fontsize=12)
    plt.ylabel("Number of Communities", fontsize=12)

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=500)

    plt.show()
    return

def add_bracket(ax, x_start, x_end, y, text, text_offset=0.03):
    """
    Draws a bracket from x_start to x_end at height y.
    text_offset determines how far above the bracket the text appears.
    """
    ax.plot([x_start, x_end], [y, y], color='black', lw=1.5)
    ax.plot([x_start, x_start], [y, y - text_offset], color='black', lw=1.5)
    ax.plot([x_end, x_end], [y, y - text_offset], color='black', lw=1.5)

    ax.text(
        (x_start + x_end) / 2,
        y + text_offset,
        text,
        ha='center',
        va='bottom',
        fontsize=12
    )

def plot_stacked_bars(exp_subcom, outfile=None):

    #get species colormap
    species_cmap = utils.get_species_colormap(name_key=False)   

    #get species otu cols only
    meta_cols = ["OD", "community_idx", "Subcommunity species", "Shannon diversity exp"]
    species_cols = [c for c in exp_subcom.columns if c not in meta_cols]
    n = len(exp_subcom)

    #norm each community to 1
    exp_subcom[species_cols] = exp_subcom[species_cols].div(
        exp_subcom[species_cols].sum(axis=1), axis=0
    )
    x = np.arange(n)

    #remove outer spines
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    #plot stacked bars
    bottom = np.zeros(n)

    for sp in species_cols:
        values = exp_subcom[sp].values
        if np.any(values > 0): 
            ax.bar(
                x,
                values,
                bottom=bottom,
                color=species_cmap.get(sp, "#999999"),
                edgecolor="none",
                width=0.6
            )
            bottom += values

    #set labels and ticks
    ax.set_xticks(x)
    ax.set_xticklabels(range(n+1, n))   # labels 1..n
    ax.set_ylabel("Relative Abundance", fontsize=12)
    ax.set_xlabel("Inoculum", fontsize=12)

    #add brackets for high and low groups
    max_y = 1.0   
    bracket_y = max_y + 0.05   
    add_bracket(ax, 0, 9, bracket_y, "Low entropy")
    add_bracket(ax, 10, 19, bracket_y, "High entropy")

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=500)
    plt.show()
    return

def plot_shannon_barplots(exp_subcom, sim_subcom, outfile=None):

    #Merge experimental and simulated data into one clean dataframe
    exp_shannon_df = exp_subcom[['community_idx', 'Subcommunity species', 'Shannon diversity exp']]
    shannon_df = exp_shannon_df.merge(sim_subcom[['Subcommunity species', 'Shannon diversity sim']], on='Subcommunity species', how='left')

    #assign group labels
    shannon_df['Group'] = np.where(shannon_df['community_idx'] >= 11, 'Low entropy', 'High entropy')

    #melt to long form for seaborn
    plot_df = shannon_df.melt(
        id_vars=['community_idx', 'Group'],
        value_vars=['Shannon diversity exp', 'Shannon diversity sim'],
        var_name='Type',
        value_name='Shannon'
    )

    #perform t-tests across groups
    low = shannon_df[shannon_df['Group']=="Low entropy"]
    high = shannon_df[shannon_df['Group']=="High entropy"]
    t_exp = ttest_ind(low['Shannon diversity exp'], high['Shannon diversity exp'])
    t_sim = ttest_ind(low['Shannon diversity sim'], high['Shannon diversity sim'])

    def p_to_stars(p):
        print(p)
        if p < 0.0001: return "****"
        elif p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        else: return "ns"
    star_exp = p_to_stars(t_exp.pvalue)
    star_sim = p_to_stars(t_sim.pvalue)

    #plot boxplot and stripplot
    plt.figure(figsize=(6,6))

    ax = sns.boxplot(
        data=plot_df,
        x='Group',
        y='Shannon',
        hue='Type',
        width=0.65,
        boxprops=dict(facecolor="none", edgecolor="black", linewidth=1.4),
        whiskerprops=dict(color="black", linewidth=1.4),
        capprops=dict(color="black", linewidth=1.4),
        medianprops=dict(color="black", linewidth=1.4),
        showcaps=True
    )
    sns.stripplot(data=plot_df, x='Group', y='Shannon', hue='Type',
                  dodge=True, alpha=0.8, jitter=True, edgecolor='none', size=8)

    #legend and axes settings
    handles, labels = ax.get_legend_handles_labels()
    label_map = {
        "Shannon diversity exp": "Experimental",
        "Shannon diversity sim": "Simulated"
    }
    clean_labels = [label_map[l] for l in labels[:2]]
    ax.legend(handles[:2], clean_labels, title="", fontsize=14)
    ax.set_xlabel("Group", fontsize=16)
    ax.set_ylabel("Shannon Entropy", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    #add significance brackets
    def add_bracket(ax, x1, x2, y, text):
        ax.plot([x1, x1, x2, x2], [y, y+0.01, y+0.01, y], lw=1.2, c='k')
        ax.text((x1+x2)/2, y+0.015, text, ha='center', va='bottom')
    y_max = plot_df['Shannon'].max()
    add_bracket(ax, 0-0.15, 1-0.15, y_max + 0.02, star_exp)
    add_bracket(ax, 0+0.15, 1+0.15, y_max + 0.10, star_sim)

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=500)
    plt.show()
    return

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir", type=Path,
        help="Processed data directory"
    )
    parser.add_argument(
        "--out", type=Path, default="data",
        help="Directory to save figures"
    )
    args = parser.parse_args()

    #add output folder structure
    args.out.mkdir(exist_ok=True, parents=True)

    #Load data needed for plots
    sim_subcom = pd.read_csv(os.path.join(args.data_dir, "simulated_3sp_communities.csv"), index_col=0)
    exp_subcom = pd.read_csv(os.path.join(args.data_dir, "experimental_3sp_otu.csv"))

    #Plot figs
    plot_shannon_hist(sim_subcom, exp_subcom, outfile=os.path.join(args.out, "Mfig_6a.png"))
    plot_stacked_bars(exp_subcom, outfile=os.path.join(args.out, "Mfig_6b.png"))
    plot_shannon_barplots(exp_subcom, sim_subcom, outfile=os.path.join(args.out, "Mfig_6c.png"))   
