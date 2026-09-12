import sys
import os
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import seaborn as sns
import statistics
import scipy
from scipy.stats import spearmanr, pearsonr, linregress

#include helper functions for loading data consistently
sys.path.append(os.path.abspath("../"))

#include figure 4 plotting code
sys.path.append(os.path.abspath("../figure4/")) 

import process_data
import utils

from argparse import ArgumentParser
from pathlib import Path

def plot_t0_correlations(df, x_row, x_label, panels,
                         color_map=None, labels=None, legend=True, outfile=None):
    tidy = df.T.apply(pd.to_numeric, errors="coerce")   
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.9))
    axes, fits = np.atleast_1d(axes), {}


    for ax, (y_row, y_label, logy) in zip(axes, panels):
        d = tidy[[x_row, y_row]].dropna().copy()
        if logy:
            d[y_row] = np.log10(d[y_row])

        # fit line only - points drawn below so each can take its own colour
        sns.regplot(data=d, x=x_row, y=y_row, ax=ax, ci=None, scatter=False,
                    line_kws=dict(color="#52514e", lw=1.8))

        colors = ([color_map.get(str(sp), "0.6") for sp in d.index]
                  if color_map else "#2a78d6")
        ax.scatter(d[x_row], d[y_row], s=58, c=colors,
                   edgecolor="white", linewidths=0.8, zorder=3)

        res = linregress(d[x_row], d[y_row])
        fits[y_row] = res
        # sit the text in whichever top corner the fit slopes away from
        ax.annotate(f"$R^2$ = {res.rvalue**2:.2f}\n$p$ = {res.pvalue:.3f}",
                    xy=(0.8, 0.97), xycoords="axes fraction",
                    va="top", ha="left", fontsize=9, color="k")

        if logy:
            ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"$10^{{{v:g}}}$"))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    if legend and color_map:
        handles = [Line2D([], [], marker="o", ls="", markersize=7,
                          markerfacecolor=c, markeredgecolor="white",
                          label=(labels or {}).get(sp, sp))
                   for sp, c in color_map.items()]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
                   frameon=False, fontsize=9)

    sns.despine(fig=fig)
    fig.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=400, bbox_inches="tight")
    return 

def plot_passage_composition(dfs, color_map, passage=5, labels=None,
                             drop_empty=True, ax=None, outfile=None, title=None):
    # one row per scheme, one column per species
    comp = pd.DataFrame({k: v.loc[passage] for k, v in dfs.items()}).T
    comp.columns = comp.columns.astype(str)
    comp = comp.apply(pd.to_numeric, errors="coerce").fillna(0.0)


    comp = comp.div(comp.sum(axis=1), axis=0)
    if drop_empty:
        comp = comp.loc[:, comp.sum() > 0]

    species = [s for s in color_map if s in comp.columns]
    comp = comp[species]

    if ax is None:
        _, ax = plt.subplots(figsize=(0.6 * len(comp) + 3, 6))

    x = np.arange(len(comp))
    bottom = np.zeros(len(comp))
    for sp in species:
        vals = comp[sp].to_numpy()
        ax.bar(x, vals, bottom=bottom, width=0.65,
               color=color_map.get(sp, "0.6"),
               edgecolor="k", linewidth=0.2,
               label=(labels or {}).get(sp, sp))
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(comp.index, rotation=60, ha="right")
    ax.set_ylabel("Relative abundance")
    ax.set_title(title)
    ax.set_xlim(-0.6, len(comp) - 0.4)

    # reverse so the legend reads top-to-bottom in the same order as the stack
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=9)

    sns.despine(ax=ax)
    ax.figure.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=400, bbox_inches="tight")
    return 

def correlate_compositions(exp_dfs, sim_dfs, method="pearson", transform=None):
    """
    Correlate every experiment x simulation pair at every shared passage.
    exp_dfs, sim_dfs : {label: df}  - index = passage, columns = species code
    method    : "pearson" | "spearman"
    transform : None | "log" | "clr"
    Returns tidy: experiment, simulation, passage, r, p, n
    """
    fn = {"pearson": pearsonr, "spearman": spearmanr}[method]
    species = sorted(set.intersection(
        *[set(d.columns.astype(str)) for d in (*exp_dfs.values(), *sim_dfs.values())]))

    def prep(d):
        d = d.copy()
        d.columns = d.columns.astype(str)
        d = d[species].apply(pd.to_numeric, errors="coerce")
        if transform in ("log", "clr"):
            nz = d.values[d.values > 0]
            d = np.log10(d + (nz.min() / 2 if nz.size else 1e-9))
            if transform == "clr":
                d = d.sub(d.mean(axis=1), axis=0)
        return d

    exp_p = {k: prep(v) for k, v in exp_dfs.items()}
    sim_p = {k: prep(v) for k, v in sim_dfs.items()}

    rows = []
    for e_name, e in exp_p.items():
        for s_name, s in sim_p.items():
            for p in e.index.intersection(s.index):
                a, b = e.loc[p].to_numpy(float), s.loc[p].to_numpy(float)
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
                    r = p_val = np.nan
                else:
                    r, p_val = fn(a[ok], b[ok])
                rows.append(dict(experiment=e_name, simulation=s_name,
                                 passage=p, r=r, p=p_val, n=int(ok.sum())))
    return pd.DataFrame(rows)

def plot_agreement_heatmap(corr, passage=None, exp_order=None, sim_order=None,
                           label="Pearson $r$", title_extra="", full_range=True, ax=None,
                           outfile=None):
    if passage is None:
        passage = corr["passage"].max()
    grid = (corr[corr["passage"] == passage]
            .pivot(index="experiment", columns="simulation", values="r")
            .reindex(index=exp_order, columns=sim_order))

    if ax is None:
        _, ax = plt.subplots(figsize=(1.15 * grid.shape[1] + 3.2,
                                      0.62 * grid.shape[0] + 2.4))
    lo, hi = np.nanmin(grid.values), np.nanmax(grid.values)
    vmin, vmax = ((-1, 1) if lo < 0 else (0, 1)) if full_range else (lo, hi)
    sns.heatmap(grid, ax=ax, annot=True, fmt=".2f", annot_kws=dict(fontsize=9),
                cmap="vlag" if lo < 0 else "Blues", center=0 if lo < 0 else None,
                vmin=vmin, vmax=vmax, linewidths=2, linecolor="white",
                cbar_kws=dict(label=label, shrink=0.8))
    ax.set_title(f"Passage {passage} composition agreement{title_extra}")
    ax.set_xlabel("simulation"); ax.set_ylabel("experiment")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.figure.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=400, bbox_inches="tight")
    return ax.figure, grid


def plot_agreement_passages(corr, exp_order=None, sim_order=None, sim_colors=None,
                            label="Pearson $r$", title_extra="", full_range=Tru,
                            outfile=None):
    exps = exp_order or corr["experiment"].unique().tolist()
    sims = sim_order or corr["simulation"].unique().tolist()
    colors = sim_colors or dict(zip(sims, SIM_COLORS))

    ncol = min(3, len(exps))
    nrow = int(np.ceil(len(exps) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow),
                             sharex=True, sharey=True, squeeze=False)

    for ax, e_name in zip(axes.ravel(), exps):
        sub = corr[corr["experiment"] == e_name]
        for s_name in sims:
            d = sub[sub["simulation"] == s_name].sort_values("passage")
            ax.plot(d["passage"], d["r"], marker="o", ms=6, lw=2,
                    color=colors[s_name], label=s_name)
        ax.axhline(0, color="#c9c8c4", lw=1, zorder=0)
        ax.set_title(e_name, fontsize=10)
        ax.set_xticks(sorted(corr["passage"].unique()))
        if full_range:
            ax.set_ylim(0, 1.05)
    for ax in axes.ravel()[len(exps):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("passage")
    for ax in axes[:, 0]:
        ax.set_ylabel(label)

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=9, title="simulation")
    fig.suptitle(f"Agreement across passages{title_extra}", y=1.0)
    sns.despine(fig=fig)
    fig.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=400, bbox_inches="tight")
    return fig

if __name__ == "__main__":
    parser = ArgumentParser(description="Plot 16s normalizations.")
    parser.add_argument("--out", required=True, help="Directory to save output figures.")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)


    t0 = utils.t0_wc_data()
    deviation_factor = utils.deviation_factor()
    cp_num_df = utils.cn_16s()
    od_cfu_df = utils.od_to_cfu()

    norm_factors = pd.concat([t0, deviation_factor, cp_num_df, od_cfu_df])
    #add 16S_copy_number * cfu-od
    norm_factors.loc['OD_cfu_16s'] = norm_factors.loc['Average CFU ml^-1'] * norm_factors.loc['16s_copy_number']

    row_val, label = "crm_t0", "Relative Abundance at $t_0$"

    # (row name in df, y-axis label, plot/fit on log10 scale?)
    subplots = [
        ("16s_copy_number",   "16S rRNA copy number",              False),
        ("Average CFU ml^-1", "Species-specific OD-CFU Conversion",False),
        ("OD_cfu_16s",        "OD-CFU Conversion * 16S copy",      False),
    ]

    plot_t0_correlations(
        norm_factors, "crm_t0", "Relative Abundance at $t_0$ (Exp B)", subplots,
        color_map=utils.get_species_colormap(name_key=False), labels=utils.sps_to_name,
        outfile=os.path.join(args.out, "Sfig_13abc.pdf"))


    "Experimental abundances and normalized abundances"

    exp_a = pd.read_csv("../data/exp_whole_comm/df_a.csv", index_col=0)
    exp_b = pd.read_csv("../data/exp_whole_comm/df_b.csv", index_col=0)

    #divide by deviation factor
    exp_a_devfac = exp_a.div(norm_factors.loc['deviation_factor'])
    exp_a_devfac = exp_a_devfac.div(exp_a_devfac.sum(axis=1), axis=0)
    exp_b_devfac = exp_b.div(norm_factors.loc['deviation_factor'])
    exp_b_devfac = exp_b_devfac.div(exp_b_devfac.sum(axis=1), axis=0)

    #divide by 16S copy number
    exp_a_16scn = exp_a.div(norm_factors.loc['16s_copy_number'])
    exp_a_16scn = exp_a_16scn.div(exp_a_16scn.sum(axis=1), axis=0)
    exp_b_16scn = exp_b.div(norm_factors.loc['16s_copy_number'])
    exp_b_16scn = exp_b_16scn.div(exp_b_16scn.sum(axis=1), axis=0)

    exp_dfs = {"Exp A (raw)": exp_a, "Exp A (16S Copy)": exp_a_16scn, "Exp A (t0 Deviation Factor)": exp_a_devfac,
        "Exp B (raw)": exp_b, "Exp B (16S Copy)": exp_b_16scn, "Exp B (t0 Deviation Factor)": exp_b_devfac}

    plot_passage_composition(exp_dfs, utils.get_species_colormap(name_key=False), passage=5, labels=utils.sps_to_name, title="Experimental final passage composition",
                             outfile=os.path.join(args.out, "Sfig_13d.pdf"))

    sp_df, _ = process_data.simulate_whole_community_exp(crossfeeding=True)
    sp_nocross_df, _ = process_data.simulate_whole_community_exp(crossfeeding=False)
    sp_t0_df, _ = process_data.simulate_whole_community_exp(crossfeeding=True, t0_abun=norm_factors.loc['crm_t0'])
    sp_cfu_df, _ = process_data.simulate_whole_community_exp(crossfeeding=True, od_cfu_conv=norm_factors.loc['Average CFU ml^-1'])
    sp_t0_cfu_df, _ = process_data.simulate_whole_community_exp(crossfeeding=True, t0_abun=norm_factors.loc['crm_t0'], od_cfu_conv=norm_factors.loc['Average CFU ml^-1'])

    sim_dfs = {"Initial equal abundance": sp_df, "Initial equal abundance (l=0)": sp_nocross_df, "T0 16S abundance": sp_t0_df,
            "Equal OD, OD-CFU": sp_cfu_df, "T0 16S abundance, OD-CFU": sp_t0_cfu_df}

    plot_passage_composition(dfs, utils.get_species_colormap(name_key=False), passage=5, labels=utils.sps_to_name, title="Simulated final passage composition",
                             outfile=os.path.join(args.out, "Sfig_14e.pdf"))

    METHOD = "spearman"          # <- the two knobs
    LABEL = {"pearson": "Pearson $r$", "spearman": "Spearman $\\rho$"}[METHOD]
    SIM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                "#008300", "#4a3aa7", "#e34948"]

    corr = correlate_compositions(exp_dfs, sim_dfs, method=METHOD)

    fig3, grid = plot_agreement_heatmap(corr, passage=5, exp_order=list(exp_dfs),
                                        sim_order=list(sim_dfs), label=LABEL, outfile=os.path.join(args.out, "Sfig_15a.pdf"))
    
    fig4 = plot_agreement_passages(corr, exp_order=list(exp_dfs), sim_order=list(sim_dfs),
                                label=LABEL, outfile=os.path.join(args.out, "Sfig_15b.pdf"))