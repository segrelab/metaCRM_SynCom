"""
Figure 4: Whole Community and Leave-one-out plots
"""
import sys
import os
import pandas as pd
import numpy as np
sys.path.append(".")
import utils
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import math
from argparse import ArgumentParser
from pathlib import Path

def plot_Mfig_4a(df_long, outfile=None):
    """
    Plot Passages and replicates as stacked bars for whole community experiment
    """
    
    # Map species code to readable names
    species_names = {code: utils.get_species_name(code) for code in df_long["species"].unique()}
    df_long["species_name"] = df_long["species"].map(species_names)

    # Sort by passage and replicate
    df_long["rep_index"] = df_long["replicate"].astype(str).str.extract("(\d+)").astype(int)
    df_long = df_long.sort_values(["Passage", "rep_index"])
    species_order = sorted(df_long["species_name"].unique())
    palette = utils.get_species_colormap()
    passages = sorted(df_long["Passage"].unique())
    reps_per_passage = df_long["rep_index"].nunique()
    
    bar_width = 0.8 / reps_per_passage
    fig, ax = plt.subplots(figsize=(8, 6))

    #loop through passages and plot bars
    for i, passage in enumerate(passages):
        passage_data = df_long[df_long["Passage"] == passage]
        for j, rep in enumerate(sorted(passage_data["rep_index"].unique())):
            rep_data = passage_data[passage_data["rep_index"] == rep]
            #keep species order consistent across all bars
            rep_data = rep_data.groupby("species_name")["value"].sum().reindex(species_order, fill_value=0)
            bottom = 0
            xpos = i + j * bar_width
            for species in species_order:
                height = rep_data[species]
                ax.bar(
                    xpos, height,
                    bottom=bottom,
                    color=palette[species],
                    width=bar_width,
                    edgecolor='black',
                    linewidth=0.2
                )
                bottom += height

    ax.set_xticks([i + (reps_per_passage - 1) * bar_width / 2 for i in range(len(passages))])
    ax.set_xticklabels(passages, rotation=0)

    #final formatting
    ax.set_xlabel("Passage", fontsize=12)
    ax.set_ylabel("Relative Abundance (16S)", fontsize=12)
    ax.set_title("Experiment A Abundances")
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[sp]) for sp in species_order]
    ax.legend(handles, species_order, bbox_to_anchor=(1, 1), loc="upper left", title="Species")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    
    if outfile:
        plt.savefig(outfile, dpi=600)
    plt.show()

    #return species_order to be used for the other bar plots of similar style
    return species_order

def plot_whole_community_exp2(exp2_df_passages, sp_order, outfile=None):
    """
    Plot Passages and replicates as stacked bars for whole community experiment (2).
    """

    df = exp2_df_passages.copy()
    df["Passage"] = df["Passage"].astype(str)

    # Map species code to readable names
    species_name_map = {code: utils.get_species_name(code) for code in exp2_df_passages.columns if code not in ["Passage", "rep"]}
    long_df = df.melt(id_vars=["Passage", "rep"], var_name="species", value_name="value")
    long_df["species_name"] = long_df["species"].map(species_name_map)

    long_df["rep_index"] = long_df["rep"].str.extract("(\d+)").astype(int)
    long_df = long_df.sort_values(["Passage", "rep_index"])

    #sort by passage and replicate
    passages = sorted(long_df["Passage"].unique())[1:6]
    reps_per_passage = long_df["rep_index"].nunique()
    bar_width = 0.8 / reps_per_passage

    palette = utils.get_species_colormap()
    fig, ax = plt.subplots(figsize=(8, 6))

    #loop through passages and replicates and plot stacked bars
    for i, passage in enumerate(passages):
        passage_data = long_df[long_df["Passage"] == passage]
        for j, rep in enumerate(sorted(passage_data["rep_index"].unique())):
            rep_data = passage_data[passage_data["rep_index"] == rep]
            #keep species order consistent across bars
            rep_data = rep_data.groupby("species_name")["value"].sum().reindex(sp_order, fill_value=0)
            bottom = 0
            xpos = i + j * bar_width
            for species in sp_order:
                height = rep_data[species]
                ax.bar(
                    xpos, height,
                    bottom=bottom,
                    color=palette[species],
                    width=bar_width,
                    edgecolor='black',
                    linewidth=0.2
                )
                bottom += height

    #other plot formatting options
    ax.set_xticks([i + (reps_per_passage - 1) * bar_width / 2 for i in range(len(passages))])
    ax.set_xticklabels(passages, rotation=0)
    ax.set_xlabel("Passage", fontsize=12)
    ax.set_ylabel("Relative Abundance (16S)", fontsize=12)
    ax.set_title("Experiment B Abundances")
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[sp]) for sp in sp_order]
    ax.legend(handles, sp_order, bbox_to_anchor=(1, 1), loc="upper left", title="Species")

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=500)
    plt.show()

    return

def plot_whole_community_simulation(df_pivot, species_order_names, outfile=None):
    """
    Plot Passages and replicates as stacked bars for simulated whole community.
    """
    #map species codes to names and codes to names
    species_map = {sp: utils.get_species_name(sp) for sp in df_pivot.columns}
    species_order_names = [sp for sp in species_order_names if sp in species_map.values()]
    code_order = []
    for name in species_order_names:
        for code, cname in species_map.items():
            if cname == name:
                code_order.append(code)
                break

    palette = utils.get_species_colormap()
    fig, ax = plt.subplots(figsize=(5, 6))
    passages = df_pivot.index
    bar_width = 0.6

    #loop through passages and plot stacked bars
    for i, passage in enumerate(passages):
        bottom = 0
        #keep order of species consistent across passages and other plots
        for sp_code in code_order:
            species_name = species_map[sp_code]
            height = df_pivot.loc[passage, sp_code]
            ax.bar(
                i, height, bottom=bottom, color=palette[species_name],
                width=bar_width, edgecolor='black', linewidth=0.2
            )
            bottom += height

    #other plotting parameters/options
    ax.set_xticks(range(len(passages)))
    ax.set_xticklabels(passages, rotation=0)
    ax.set_xlabel("Passage", fontsize=12)
    ax.set_ylabel("Relative Abundance", fontsize=12)
    ax.set_title("Simulated Abundances")
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[sp]) for sp in species_order_names]
    ax.legend(handles, species_order_names, bbox_to_anchor=(1, 1), loc="upper left", title="Species")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=600)
    plt.show()

def plot_whole_community_correlation(df_exp1, df_exp2, df_sim, df_sim_no_cf, outfile=None):
    
    # Assume all dataframes have same shape: passages x species
    passages = df_exp2.index

    corrs = {
        "passage": [],
        "correlation": [],
        "Simulation Mode": [],
        "Exp Dataset": []
    }
    df_exp1.index = df_exp1.index.astype(int)
    df_exp2.index = df_exp2.index.astype(int)
    df_sim.index = df_sim.index.astype(int)
    df_sim_no_cf.index = df_sim_no_cf.index.astype(int)

    #calculate correlations between experiments and simulations for each passage
    for exp_label, df_exp in zip(["ExpA", "ExpB"], [df_exp1, df_exp2]):
        for p in passages:
            r1, _ = scipy.stats.spearmanr(df_exp.loc[p], df_sim.loc[p])
            r2, _ = scipy.stats.spearmanr(df_exp.loc[p], df_sim_no_cf.loc[p])

            corrs["passage"].append(p)
            corrs["correlation"].append(r1)
            corrs["Simulation Mode"].append("Cross-feeding")
            corrs["Exp Dataset"].append(exp_label)

            corrs["passage"].append(p)
            corrs["correlation"].append(r2)
            corrs["Simulation Mode"].append("No Cross-feeding")
            corrs["Exp Dataset"].append(exp_label)

    #save together as DataFrame
    corr_df = pd.DataFrame(corrs)

    #plot
    plt.figure(figsize=(7, 5))
    sns.lineplot(
        data=corr_df,
        x="passage",
        y="correlation",
        hue="Simulation Mode",
        style="Exp Dataset",   # line style separates Exp1 vs Exp2
        markers=True,
        dashes=True,
        palette={"Cross-feeding": "orange", "No Cross-feeding": "green"}
    )
    plt.title("Correlation of Simulated vs. Experimental Abundances")
    plt.ylim(0, 1)
    plt.ylabel("Spearman Correlation", fontsize=14)
    plt.xlabel("Passage", fontsize=14)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=600)
    plt.show()
    
    return 

def distribution_overlap(a, b, bins=10):
    """Return overlap coefficient between two 1D distributions."""
    hist_a, bin_edges = np.histogram(a, bins=bins, density=True)
    hist_b, _ = np.histogram(b, bins=bin_edges, density=True)
    overlap = np.sum(np.minimum(hist_a, hist_b)) * np.diff(bin_edges)[0]
    
    if np.isnan(overlap):
        overlap = 0.0

    return overlap

def plot_all_loo_effects(loo_sim_vals, loo_exp_vals, outfile=None):

    loo_sim_vals = loo_sim_vals.rename(columns={'$W_i$': 'W_i'})
    loo_exp_vals = loo_exp_vals.rename(columns={'$W_i$': 'W_i'})

    loo_sim_vals['source'] = 'Simulation'
    loo_exp_vals['source'] = 'Experiment'

    combined = pd.concat([loo_sim_vals, loo_exp_vals], ignore_index=True)
    species_to_drop = ['Methylobacterium', 'Rhodococcus', 'Brevibacillus', 'Marmoricola', 'Paenibacillus']
    combined = combined[~combined['sp'].isin(species_to_drop)]
    combined['log_W_i'] = np.log10(combined['W_i'])
    combined = combined[np.isfinite(combined['log_W_i'])]

    plt.figure(figsize=(14,8))
    ax = plt.gca()

    # Get unique species (x categories) in plotting order
    species = combined['sp'].unique()
    for i in range(0, len(species), 2):
        ax.axvspan(i - 0.5, i + 0.5, color='lightgrey', alpha=0.2, zorder=0)

    # Plot the faint horizontal line
    plt.axhline(y=1e-9, color='grey', linestyle='--', linewidth=2)

    # Plot the data
    sns.stripplot(
        data=combined,
        x='sp',
        y='log_W_i',
        hue='source',
        palette={'Simulation': 'green', 'Experiment': 'orange'},
        jitter=0.25,
        dodge=True,
        size=6,
        alpha=0.7
    )

    sns.boxplot(
        data=combined,
        x='sp',
        y='log_W_i',
        hue='source',
        palette={'Simulation': 'green', 'Experiment': 'orange'},
        fliersize=0,
        linewidth=1,
        boxprops={'alpha':0.2},
        whiskerprops={'alpha':0.2},
        capprops={'alpha':0.2},
        medianprops={'alpha':0.5, 'color':'black'},
        dodge=True
    )

    for j, sp in enumerate(species):
        sim_sub = loo_sim_vals[loo_sim_vals['sp'] == sp]['W_i']
        exp_sub = loo_exp_vals[loo_exp_vals['sp'] == sp]['W_i']
        overlap = distribution_overlap(sim_sub, exp_sub)
        ax.text(
            j,
            combined['log_W_i'].max(),
            f"S = {overlap:.2f}",
            ha='center',
            va='bottom',
            fontsize=14,
            color='black'
        )
    

    # Legend cleanup
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower left', fontsize=16)

    # Aesthetics
    plt.xticks(rotation=40, fontsize=16)
    plt.yticks(fontsize=14)
    plt.xlabel('')
    plt.ylabel('Leave-out Abun. / Rescaled Community Abun.', fontsize=16)
    plt.title('Leave-out Effects on Community Abundance By Species', fontsize=18)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=600)
    plt.show()
    return

def plot_loo_abundance_comparison(df_sim, df_exp, outfile=None):
    """
    Plot leave-one-out resulting abundances by species in a boxplot, show whole community abundance for reference as well.
    """

    #drop species with low/no growth in whole community
    no_growth = {"1330", "1336", "1338"}  # keep as strings
    df_sim = df_sim.drop(columns=[c for c in df_sim.columns if str(c) in no_growth], errors="ignore")
    df_exp = df_exp.drop(columns=[c for c in df_exp.columns if str(c) in no_growth], errors="ignore")

    #remove whole-community row ('5') from boxplots, to be plotted differently
    df_sim = df_sim.copy()
    df_exp = df_exp.copy()
    df_sim_loo = df_sim[df_sim.index.astype(str) != "5"]
    df_exp_loo = df_exp[df_exp.index.astype(str) != "5"]

    #extract whole-community abundance for markers
    whole_sim = df_sim.loc["5"]
    whole_exp = df_exp.loc["5"]

    #melt into long format
    df_sim_loo = df_sim_loo.reset_index().rename(columns={df_sim_loo.index.name or "index": "sp_left_out"})
    df_exp_loo = df_exp_loo.reset_index().rename(columns={df_exp_loo.index.name or "index": "sp_left_out"})
    sim_long = df_sim_loo.melt(id_vars="sp_left_out", var_name="species_code", value_name="abundance")
    exp_long = df_exp_loo.melt(id_vars="sp_left_out", var_name="species_code", value_name="abundance")

    #differentiate experiment vs. simulated
    sim_long["source"] = "Simulation"
    exp_long["source"] = "Experiment"
    combined = pd.concat([sim_long, exp_long], ignore_index=True)
    combined["species_code"] = combined["species_code"].astype(str)
    combined["sp_left_out"] = combined["sp_left_out"].astype(str)

    #convert species codes to names
    combined["species"] = combined["species_code"].apply(utils.get_species_name).astype(str)
    combined["left_out_species"] = combined["sp_left_out"].apply(utils.get_species_name).astype(str)

    #remove diagonals (effect of species i when i is left out)
    combined = combined[combined["species"] != combined["left_out_species"]]

    #remove zero abundances to log transform
    combined = combined[combined["abundance"] > 0]
    combined["log_abun"] = np.log10(combined["abundance"])
    combined = combined[np.isfinite(combined["log_abun"])]

    #plot
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    species_order = combined["species"].unique()
    ax.set_xlim(-0.5, len(species_order) - 0.5)

    #alternating background
    for i in range(0, len(species_order), 2):
        ax.axvspan(i - 0.5, i + 0.5, color="lightgrey", alpha=0.2)

    #stripplot
    sns.stripplot(
        data=combined,
        x="species",
        y="log_abun",
        hue="source",
        dodge=True,
        jitter=0.25,
        alpha=0.6,
        size=5,
        palette={"Simulation": "green", "Experiment": "orange"},
    )

    #boxplot
    sns.boxplot(
        data=combined,
        x="species",
        y="log_abun",
        hue="source",
        dodge=True,
        fliersize=0,
        linewidth=1,
        boxprops={"alpha": 0.2},
        whiskerprops={"alpha": 0.2},
        capprops={"alpha": 0.2},
        medianprops={"color": "black", "alpha": 0.6},
        palette={"Simulation": "green", "Experiment": "orange"},
    )

    #plot whole-community abundances as stars
    plt.draw()  
    ymin, ymax = ax.get_ylim()

    for i, sp in enumerate(species_order): 
        code = combined[combined["species"] == sp]["species_code"].iloc[0]
        sim_val = whole_sim.get(code, np.nan)
        if pd.notna(sim_val):
            if sim_val > 0:
                y = np.log10(sim_val)
            else:
                y = ymin  # place star at bottom if abundance is zero
            ax.scatter(
                i - 0.2, y,
                color="green", s=80, marker="*",
                edgecolor="black", zorder=6, alpha=0.9
            )
        exp_val = whole_exp.get(code, np.nan)

        if pd.notna(exp_val):
            if exp_val > 0:
                y = np.log10(exp_val)
            else:
                y = ymin  # place star at bottom if abundance is zero
            ax.scatter(
                i + 0.2, y,
                color="orange", s=80, marker="*",
                edgecolor="black", zorder=6, alpha=0.9
            )

    for i, sp in enumerate(species_order):
        sim_vals = combined[(combined["species"] == sp) & (combined["source"] == "Simulation")]["abundance"]
        exp_vals = combined[(combined["species"] == sp) & (combined["source"] == "Experiment")]["abundance"]

        #perform Mann–Whitney U test (two-sided) between expeirmental and simulated distributions
        if len(sim_vals) > 0 and len(exp_vals) > 0:
            stat, p_value = scipy.stats.mannwhitneyu(sim_vals, exp_vals, alternative='two-sided')
        else:
            p_value = np.nan

        #determine significance stars
        if np.isnan(p_value):
            stars = ""
        elif p_value < 0.001:
            stars = "***"
        elif p_value < 0.01:
            stars = "**"
        elif p_value < 0.05:
            stars = "*"
        else:
            stars = ""

    #legend cleanup
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower left", fontsize=12)

    plt.xticks(rotation=40, fontsize=14)
    plt.yticks(fontsize=12)
    plt.xlabel("")
    plt.ylabel("Leave-out Log Abundance", fontsize=16)
    plt.title("Leave-One-Out Abundances: Experiment vs Simulation", fontsize=18)

    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=600)

    plt.show()

def plot_whole_community_correlation_arth(df_exp1, df_exp2, df_sim, df_sim_no_cf, df_sim_no_arth, outfile=None):
    # Assume all dataframes have same shape: passages x species
    passages = df_exp2.index

    corrs = {
        "passage": [],
        "correlation": [],
        "Simulation Mode": [],
        "Exp Dataset": []
    }

    #For each whole community experiment, perform correlations with different simulation modes, asave in dataframe
    for exp_label, df_exp in zip(["Exp1", "Exp2"], [df_exp1, df_exp2]):
        for p in passages:
            r1, _ = scipy.stats.spearmanr(df_exp.loc[p], df_sim.loc[p])
            r2, _ = scipy.stats.spearmanr(df_exp.loc[p], df_sim_no_cf.loc[p])
            r3, _ = scipy.stats.spearmanr(df_exp.loc[p], df_sim_no_arth.loc[p])

            corrs["passage"].append(p)
            corrs["correlation"].append(r1)
            corrs["Simulation Mode"].append("Cross-feeding")
            corrs["Exp Dataset"].append(exp_label)

            corrs["passage"].append(p)
            corrs["correlation"].append(r2)
            corrs["Simulation Mode"].append("No Cross-feeding")
            corrs["Exp Dataset"].append(exp_label)

            corrs["passage"].append(p)
            corrs["correlation"].append(r3)
            corrs["Simulation Mode"].append("No Arthrobacter")
            corrs["Exp Dataset"].append(exp_label)

    corr_df = pd.DataFrame(corrs)

    #plot correlations across passages
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for ax, exp_label in zip(axes, ["Exp1", "Exp2"]):
        sns.lineplot(
            data=corr_df[corr_df["Exp Dataset"] == exp_label],
            x="passage",
            y="correlation",
            hue="Simulation Mode",
            markers=True,
            dashes=True,
            palette={
                "Cross-feeding": "orange",
                "No Cross-feeding": "green",
                "No Arthrobacter": "grey"
            },
            ax=ax
        )
        ax.set_title(f"Correlation to {exp_label}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Spearman Correlation", fontsize=12)
        ax.set_xlabel("Passage", fontsize=12)

        #place legend only on the first subplot to avoid duplicates
        if exp_label == "Exp1":
            ax.legend(
                fontsize=10,
                title_fontsize=12,
                loc="upper left",
                bbox_to_anchor=(0.45, 0.28)
            )
        else:
            ax.get_legend().remove()

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=600)
    plt.show()
    return corr_df

def plot_loo_onesp(df_exp, df_sim, sp, outfile=None):
    """
    df_exp, df_sim: rows = species left out, columns = species codes
    sp: species code of interest (string)
    """

    # --- Extract the values for the species of interest ---
    df_exp.index = df_exp.index.map(utils.get_species_name)
    df_exp.columns = df_exp.columns.map(utils.get_species_name)

    df_sim.index = df_sim.index.map(utils.get_species_name)
    df_sim.columns = df_sim.columns.map(utils.get_species_name)
    
    df_exp = df_exp.drop(index=sp, errors="ignore")
    df_sim = df_sim.drop(index=sp, errors="ignore")
    exp_vals = df_exp[sp]
    sim_vals = df_sim[sp]

    # --- Create side-by-side bars ---
    x = np.arange(len(df_exp))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width/2, exp_vals, width, label="Experimental")
    ax.bar(x + width/2, sim_vals, width, label="Simulated")

    ax.set_xticks(x)
    ax.set_xticklabels(df_exp.index, rotation=60, ha='right')

    ax.set_ylabel("Relative Abundance")
    ax.set_xlabel("Species Left Out")
    ax.set_title(f"Effect on {sp}")

    ax.legend()
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=500)
    plt.show()

    return fig, ax

def load_exp_loo_effects(path_to_epistasis_vals):
    #Load epistasis values df and subset to LOO W_i values
    #Some are under W_j and some are under W_i, need a complete set
    
    full_epistasis_vals = pd.read_csv(path_to_epistasis_vals)
    full_epistasis_vals.columns
    extra_vals = full_epistasis_vals[['j', 'k', '$W_j$']].drop_duplicates()
    extra_vals.columns = [['i', 'sp', '$W_i$']]
    loo_exp_vals = full_epistasis_vals[['i', 'k', '$W_i$']].drop_duplicates()
    loo_exp_vals.columns = [['i', 'sp', '$W_i$']]
    loo_exp_vals = pd.concat([loo_exp_vals, extra_vals], ignore_index=True).drop_duplicates()
    loo_exp_vals.columns = [c[0] if isinstance(c, tuple) else c for c in loo_exp_vals.columns]
    
    return loo_exp_vals


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate all figures from CSV data.")
    parser.add_argument("--data_dir", required=True, help="Directory with input CSV files.")
    #parser.add_argument("--epistasis_data", required=True, help="Directory with experimental LOO effects data.")
    parser.add_argument("--out", required=True, help="Directory to save output figures.")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    #LOAD PROCESSED DATA
    exp_a_reps = pd.read_csv(os.path.join(args.data_dir, "exp_whole_comm/df_a_reps.csv"))
    exp_a_clean = pd.read_csv(os.path.join(args.data_dir, "exp_whole_comm/df_a.csv"), index_col=0)
    exp_b_reps = pd.read_csv(os.path.join(args.data_dir, "exp_whole_comm/df_b_reps.csv"))
    exp_b_clean = pd.read_csv(os.path.join(args.data_dir, "exp_whole_comm/df_b.csv"), index_col=0)

    wc_sp_sim = pd.read_csv(os.path.join(args.data_dir, "sim_whole_comm/wc_sp_sim.csv"), index_col=0)
    wc_sp_nc = pd.read_csv(os.path.join(args.data_dir, "sim_whole_comm/wc_sp_sim_nc.csv"), index_col=0)   
    wc_sp_noarth = pd.read_csv(os.path.join(args.data_dir, "sim_whole_comm/df_sp_noarth.csv"), index_col=0)
    
    exp_loo_df = pd.read_csv(os.path.join(args.data_dir, "exp_loo_df.csv"), index_col=0)
    sim_loo_effects = pd.read_csv(os.path.join(args.data_dir, "sim_loo/sim_loo_effects.csv"))
    sim_loo_nc = pd.read_csv(os.path.join(args.data_dir, "sim_loo/sp_loo_nc.csv"))
    sim_loo = pd.read_csv(os.path.join(args.data_dir, "sim_loo/sp_loo.csv"), index_col=0)

    #load experimental epistasis values
    exp_loo_effects = load_exp_loo_effects(os.path.join(args.data_dir, "epistasis_vals.csv"))

    #add whole community row to LOO df
    whole_sp_sim = wc_sp_sim.copy()
    row_to_append = whole_sp_sim.loc[5, sim_loo.columns]
    sim_loo_plot = pd.concat([sim_loo, pd.DataFrame([row_to_append], index=['5'])])
    row_to_append = exp_a_clean.loc[5, exp_loo_df.columns]
    exp_loo_plot = pd.concat([exp_loo_df, pd.DataFrame([row_to_append], index=['5'])])
    
    #PLOT ALL FIGURES
    col_order = plot_Mfig_4a(exp_a_reps, outfile=os.path.join(args.out, "Mfig_4a_a.png"))
    plot_whole_community_exp2(exp_b_reps, col_order, outfile=os.path.join(args.out, "Mfig_4a_b.png"))
    plot_whole_community_simulation(wc_sp_sim, col_order, outfile=os.path.join(args.out, "Mfig_4b.png"))
    plot_whole_community_simulation(wc_sp_nc, col_order, outfile=os.path.join(args.out, "Mfig_4c.png"))
    plot_whole_community_correlation(exp_a_clean, exp_b_clean, wc_sp_sim, wc_sp_nc, outfile=os.path.join(args.out, "Mfig_4d.png"))
    plot_whole_community_correlation_arth(exp_a_clean, exp_b_clean, wc_sp_sim, wc_sp_nc, wc_sp_noarth, outfile=os.path.join(args.out, "Sfig_8.png"))
    plot_all_loo_effects(sim_loo_effects, exp_loo_effects, outfile=os.path.join(args.out, "loo_effects_bar.png"))
    plot_loo_abundance_comparison(sim_loo_plot, exp_loo_plot, outfile=os.path.join(args.out, "Mfig_4e.png"))
    plot_loo_onesp(exp_loo_df.copy(), sim_loo.copy(), sp='Burkholderia', outfile=os.path.join(args.out, "Sfig_7b.png"))
    plot_loo_onesp(exp_loo_df.copy(), sim_loo.copy(), sp='Mucilaginibacter', outfile=os.path.join(args.out, "Sfig_7a.png"))

