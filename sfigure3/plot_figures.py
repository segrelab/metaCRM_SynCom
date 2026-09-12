"""
Code for plotting supplementary figure 3, carbon consumption / secretion
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

import re
import sys
import os

sys.path.append(os.path.abspath("/projectnb/cometsfba/rowanon/projects/metaCRM_SynCom/"))
import utils

from argparse import ArgumentParser
from pathlib import Path

# Constants
C_RE = re.compile(r"C(?![a-z])(\d*)")  # Match just "C" or "C1" not "Ca"...

def carbon_count(formula):
    m = C_RE.search(str(formula))
    return int(m.group(1) or 1) if m else 0

def plot_carbon_balance(agg, ncol=3):
    species = agg["species"].unique()
    nrow = int(np.ceil(len(species) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    lines = []
    for ax, sp in zip(axes.flat, species):
        g = agg.loc[agg["species"] == sp].sort_values("time")
        lines = [
            ax.plot(g["time"], g["consumed_C"], "o-", color="tab:blue",
                    label="C consumed")[0],
            ax.plot(g["time"], g["secreted_C"], "s-", color="tab:orange",
                    label="C secreted")[0],
            ax.plot(g["time"], g["net_C"], "^--", color="k",
                    label="net C uptake")[0],
        ]
        ax.axhline(0, lw=0.8, color="grey", zorder=0)
        ax.set_title(sp, style="italic")
        ax.set_xlabel("time")
        ax.set_ylabel("carbon (mmol C)")
    for ax in axes.flat[len(species):]:
        ax.set_visible(False)
    axes[0, 0].legend(handles=lines, fontsize=8, loc="best")
    fig.suptitle("Carbon consumption vs. secretion per strain")
    fig.tight_layout()
    return fig

def plot_net_distributions_by_class(per_met, class_colors, class_order, seed=0):
    rng = np.random.default_rng(seed)
    species = sorted(per_met["species"].unique())
    cls = per_met["metabolite_class"].where(
        per_met["metabolite_class"].isin(class_order), "Others")
    cols = [("net", "net flux (mmol)"), ("net_C", "net carbon (mmol C)")]
    fig, axes = plt.subplots(1, 2, figsize=(5 + 0.7 * len(species), 4.4))
    for ax, (col, label) in zip(axes, cols):
        data = [per_met.loc[per_met["species"] == sp, col].to_numpy()
                for sp in species]
        ax.boxplot(data, widths=0.55, showfliers=False, medianprops=dict(color="k"))
        for i, sp in enumerate(species, start=1):
            m_sp = (per_met["species"] == sp).to_numpy()
            xj = i + rng.uniform(-0.14, 0.14, m_sp.sum())
            y = per_met.loc[m_sp, col].to_numpy()
            c_sp = cls.to_numpy()[m_sp]
            for c in class_order:
                m = c_sp == c
                if m.any():
                    ax.scatter(xj[m], y[m], s=26, alpha=0.8, zorder=3,
                               linewidths=0.3, edgecolor="k",
                               color=class_colors[c])
        ax.axhline(0, lw=0.8, color="grey", zorder=0)
        ax.set_xticks(range(1, len(species) + 1))
        ax.set_xticklabels(species, rotation=45, ha="right", style="italic")
        ax.set_ylabel(label)
        ax.set_title(label.split(" (")[0])
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=6,
                          markeredgecolor="k", markeredgewidth=0.3,
                          color=class_colors[c], label=c) for c in class_order]
    fig.legend(handles=handles, fontsize=8, loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False, title="class")
    fig.suptitle("Net flux per metabolite by strain (>0 secreted, <0 consumed)")
    fig.tight_layout()
    return fig

def plot_net_C_vs_growth(gr, x_label, sp_color_palette):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.scatter(gr["g"], gr["net_C"], s=70, zorder=3, linewidth=0.5,
               c=[sp_color_palette[sp] for sp in gr["species"]])
    ax.axhline(0, lw=0.8, color="grey", zorder=0)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(gr["g"], gr["net_C"])
    ax.axline((0,intercept), slope=slope, ls='--', c="grey")
    ax.text(.60, 0.98, f'r={r_value:.3f}, p={p_value:.1e}', 
             transform=ax.transAxes, fontsize=11, verticalalignment='top')
    ax.set_xlabel(x_label)
    ax.set_ylabel("net carbon (mmol C)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=sp_color_palette[sp], label=sp)
               for sp in per_sp["species"]]
    fig.legend(handles=handles, fontsize=8, loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    #### Formula information ####
    # Information about # of carbons for NLDM metabolites
    formulas = pd.read_csv(os.path.join(args.data_dir, "NLDM-formulas.csv"))
    # Change names to lowercase
    formulas['Name'] = formulas['Name'].str.lower()

    # Metabolite timecourse data
    met_time = pd.read_csv(os.path.join(args.data_dir, "monoculture_exp/met_time_df.csv"))
    met_time['metabolite'] = met_time['metabolite'].str.lower()  # There's also some capital case metabolites in here...

    #### Plot and save figures ####
    fig_props = {
        bbox_inches: 'tight',
        transparent: 'True'
    }

    # Get C count for formulas
    formulas = formulas.assign(n_C=formulas["Formula"].map(carbon_count))

    # Merge dataframes
    met_time = met_time.merge(
            formulas.rename(columns={"Name": "metabolite"})[["metabolite", "n_C", "Concentration (uM)"]],
            on="metabolite", how="left", validate="many_to_one",
    )

    met_time["median_usage"] = met_time["median_usage"] * met_time["Concentration (uM)"]
    met_time["median_val"] = met_time["median_val"] * met_time["Concentration (uM)"]

    # identify consumed vs. secreted metabolites for each species
    met_time["consumed"] = met_time["median_usage"].clip(upper=0).abs()
    met_time["secreted"] = met_time["median_usage"].clip(lower=0)

    # Bias by C
    met_time["consumed_C"] = met_time["consumed"] * met_time["n_C"]
    met_time["secreted_C"] = met_time["secreted"] * met_time["n_C"]

    # sort
    agg = (
        met_time.groupby(["species", "time"], observed=True)
                [["consumed", "consumed_C", "secreted", "secreted_C"]]
                .sum(min_count=1)
                .reset_index()
    )
    agg = agg.assign(net_C=agg["consumed_C"] - agg["secreted_C"])  

    # Panel a
    carbon_balance = plot_carbon_balance(agg)
    carbon_balance.savefig(outfile=os.path.join(args.out, "SFig3a.pdf"), **fig_props)

    # Panel b
    per_met = (
        met_time.loc[met_time["n_C"].notna()]
        .assign(net=lambda d: d["secreted"] - d["consumed"])
        .groupby(["species", "metabolite", "metabolite_class"], observed=True)
        .agg(net=("net", "sum"), n_C=("n_C", "first"))
        .reset_index()
        .assign(net_C=lambda d: d["net"] * d["n_C"])
    )

    class_order = ["Sugar", "Organic_Acid", "Amino_Acid", "Nucleobase", "Others"]
    palette = sns.color_palette("husl", n_colors=len(class_order))
    class_colors = dict(zip(class_order, palette))

    net_dist = plot_net_distributions_by_class(per_met, class_colors, class_order)
    net_dist.savefig(outfile=os.path.join(args.out, "SFig3b.pdf"), **fig_props)

    # Panel c
    # Plot net carbon flux by growth rate
    glist = pd.read_csv(os.path.join(args.data_dir, 'final_crm_params/glist_fitted.csv', index_col=0))
    translate = utils.sps_to_name  # Translation dict
    normg = glist
    normg = pd.DataFrame(normg)
    normg.reset_index(inplace=True)
    normg.rename(columns={normg.columns[0]: "sp", normg.columns[1]: "g"}, inplace=True)
    normg["species"] = normg["sp"].astype(str).map(translate)

    # Final biomass
    final = pd.read_csv(os.path.join(args.data_dir, "monoculture_exp/growth_df_clean.csv"), skiprows=[1,2,3]).transpose().rename(columns={0: 'g'}, index=translate).reset_index(names=['species'])

    # Merge info for plot
    gr = per_sp.merge(normg[["species", "g"]], on="species",
                    how="inner", validate="one_to_one")

    bf = per_sp.merge(final[["species", "g"]], on="species",
                    how="inner", validate="one_to_one")

    grpredictor = plot_net_C_vs_growth(gr, "Fitted growth rate (g)", sp_color_palette)
    bfpredictor = plot_net_C_vs_growth(bf, "Final OD600", sp_color_palette)

    grpredictor.savefig(outfile=os.path.join(args.out, "SFig3ca.pdf"), **fig_props)
    bfpredictor.savefig(outfile=os.path.join(args.out, "SFigcb.pdf"), **fig_props)