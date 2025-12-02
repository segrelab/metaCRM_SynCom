"""
Figure 2: Plot all main and Supplementary figures for this section
"""
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec 
import os
import sys
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
sys.path.append('.')
import utils

def plot_Mfig_2a(met_class_df, met_time_df, outfile=None):
    """
    Plot resource consumption/production at last timepoint.
    Cluster species based on metabolite usage.
    """
    
    #last timepoint
    last_tp_df = met_time_df[met_time_df['time'].notna()].copy()
    last_tp_df = last_tp_df.groupby(['species', 'metabolite', 'metabolite_class']).median_usage.last().reset_index()
    
    #rows=metabolites, cols=species
    plot_df = last_tp_df.pivot(index='metabolite', columns='species', values='median_usage')
    plot_matrix = plot_df[plot_df.index != 'spermidine']

    #order metabolites by class
    class_order = ["Sugar", "Organic_Acid", "Amino_Acid", "Nucleobase", "Others"]
    row_order = []
    for cls in class_order:
        mets_in_class = met_class_df.loc[met_class_df.metabolite_class == cls, "metabolite"].tolist()
        mets_in_class = [m for m in mets_in_class if m in plot_matrix.index]
        row_order.extend(mets_in_class)
    plot_matrix = plot_matrix.loc[row_order]
    
    #assign colors to classes in the same order as rows
    palette = sns.color_palette("husl", n_colors=len(class_order))
    lut = {}
    class_colors = {}
    for i, cls in enumerate(class_order):
        class_colors[cls] = palette[i]
        for met in met_class_df.loc[met_class_df.metabolite_class == cls, "metabolite"]:
            if met in plot_matrix.index:
                lut[met] = palette[i]
    row_colors = plot_matrix.index.map(lut)

    plot_matrix.index.name = None
    plot_matrix.columns.name = None
    
    #plot clustered heatmap
    g = sns.clustermap(
        plot_matrix, row_colors=row_colors,
        figsize=(6,10), row_cluster=False,
        dendrogram_ratio=(.05, .05), cbar_pos=(0.85, .3, .05, .2),
        cmap='vlag', center=0, vmax=1, vmin=-1
    )
    g.ax_heatmap.set_yticklabels([])
    g.ax_heatmap.set_yticks([])
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    g.fig.suptitle('Experimental Metabolite Usage', fontsize=14, y=1.02, x=0.42)
    
    #modify colorbar
    g.cax.set_title('Metabolite\nUsage\n', fontsize=10, pad=15)
    cbar = g.cax
    ticks = cbar.get_yticks()
    tick_labels = [f'{int(t)}+' if t==1 else str(t) for t in ticks]
    cbar.set_yticklabels(tick_labels)
    
    #custom row colors with class labels
    class_label_positions = []
    for cls, metabolites in met_class_df.groupby('metabolite_class')['metabolite']:
        class_rows = [i for i, met in enumerate(plot_matrix.index) if met in metabolites.values]
        if class_rows:
            mid = (min(class_rows) + max(class_rows)) / 2
            class_label_positions.append((mid, cls))
    for pos, cls in class_label_positions:
        g.ax_row_colors.text(-0.5, pos, cls.replace("_"," "), rotation=90, va='center', ha='right', fontsize=10, color='black')
    
    col_order = [plot_matrix.columns[i] for i in g.dendrogram_col.reordered_ind]
    
    if outfile:
        plt.savefig(outfile, dpi=600, bbox_inches='tight')
    
    return col_order

def plot_Sfig_1(met_class_df, met_time_df, outfile=None):
    """
    Plot the consumption of each metabolite over time for each species.
    Log10 scale. Figure: species (rows) x metabolite classes (columns)
    """
    species_list = met_time_df['species'].unique()
    classes = met_class_df['metabolite_class'].unique()
    num_species = len(species_list)
    fig, axes = plt.subplots(num_species, len(classes), figsize=(20, num_species*2))
    
    #color map per metabolite class
    color_map = {}
    for cls in classes:
        mets = met_class_df[met_class_df['metabolite_class']==cls]['metabolite']
        fixed_colors = sns.color_palette("Spectral", n_colors=len(mets))
        color_map[cls] = {met: fixed_colors[i%len(fixed_colors)] for i, met in enumerate(mets)}

    #loop over species and metabolites, separated by metabolite class
    for s, sp in enumerate(species_list):
        sp_df = met_time_df[met_time_df['species']==sp]
        for c, cls in enumerate(classes):
            ax = axes[s,c] if num_species>1 else axes[c]
            mets = met_class_df[met_class_df['metabolite_class']==cls]['metabolite']
            for met in mets:
                met_df = sp_df[sp_df['metabolite']==met]
                if met_df.empty:
                    continue
                x = [0] + met_df['time'].tolist()
                y = [1] + met_df['median_val'].tolist()
                ax.plot(x, np.log10(y), linewidth=1, color=color_map[cls][met], label=met)
                ax.set_ylim([-6,1])
            if c != 0:
                ax.set_yticklabels([])
            if s != num_species-1:
                ax.set_xticklabels([])
            ax.set_title(cls if s==0 else "")
            ax.set_ylabel(sp if c==0 else "", fontsize=12)
    
    #long legend
    legend_elements = []
    for cls in classes:
        legend_elements.append(Patch(color='none', label=f"----- {cls} -----", linewidth=0))
        for met in met_class_df[met_class_df['metabolite_class']==cls]['metabolite']:
            legend_elements.append(Patch(color=color_map[cls][met], label=met))
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.15,0.5),
               title="Metabolite Classes", fontsize=11, title_fontsize=12)
    
    fig.supxlabel("Time (hr)", fontsize=16)
    fig.supylabel("Log10 Metabolite Ratio", fontsize=16)
    plt.subplots_adjust(left=0.1, bottom=0.05, wspace=0.2, hspace=0.1, right=0.98)
    
    if outfile:
        plt.savefig(outfile, dpi=600, bbox_inches='tight')
    return

def plot_Sfig_2(met_class_df, met_dR_df, outfile=None):
    """
    Plot dR/R/dt over N for each species and metabolite class.
    """
    species_list = met_dR_df['species'].unique()
    classes = met_class_df['metabolite_class'].unique()
    num_species = len(species_list)
    
    fig, axes = plt.subplots(num_species, len(classes), figsize=(20, num_species*2))
    
    #color map per metabolite class
    color_map = {}
    for cls in classes:
        mets = met_class_df[met_class_df['metabolite_class']==cls]['metabolite']
        fixed_colors = sns.color_palette("Spectral", n_colors=len(mets))
        color_map[cls] = {met: fixed_colors[i%len(fixed_colors)] for i, met in enumerate(mets)}
    
    for s, sp in enumerate(species_list):
        sp_df = met_dR_df[met_dR_df['species']==sp]
        row_y_values = []
        for c, cls in enumerate(classes):
            ax = axes[s,c] if num_species>1 else axes[c]
            mets = met_class_df[met_class_df['metabolite_class']==cls]['metabolite']
            for met in mets:
                met_df_met = sp_df[sp_df['metabolite']==met]
                if met_df_met.empty:
                    continue

                #ensure initial point (0,0)
                x = [0] + met_df_met['N'].tolist()
                y = [0] + met_df_met['dR/Rdt'].tolist()

                ax.plot(x, y, color=color_map[cls][met], linewidth=1)
                ax.scatter(x, y, color=color_map[cls][met], s=5)
                row_y_values.extend(y)
            if c != 0:
                ax.set_yticklabels([])
                ax.set_xticklabels([])
            ax.set_title(cls if s==0 else "")
            ax.set_ylabel(sp if c==0 else "", fontsize=12)
        if row_y_values:
            y_min = min(row_y_values) - 0.1*abs(min(row_y_values))
            y_max = max(row_y_values) + 0.1*abs(max(row_y_values))
            for c in range(len(classes)):
                ax = axes[s,c] if num_species>1 else axes[c]
                ax.set_ylim(y_min, y_max)
    
    #legend
    legend_elements = []
    for cls in classes:
        legend_elements.append(Patch(color='none', label=f"----- {cls} -----", linewidth=0))
        for met in met_class_df[met_class_df['metabolite_class']==cls]['metabolite']:
            legend_elements.append(Patch(color=color_map[cls][met], label=met))
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.15,0.5),
               title="Metabolite Classes", fontsize=11, title_fontsize=12)
    
    fig.supxlabel("N ($10^9$ CFU/ml)", fontsize=16)
    fig.supylabel("dR/R*dt (1/hr)", fontsize=16)
    plt.subplots_adjust(left=0.08, bottom=0.03, wspace=0.05, hspace=0.2, right=0.98)
    
    if outfile:
        plt.savefig(outfile, dpi=600, bbox_inches='tight')

    return

def plot_Sfig_3(od_time_df, growth_df, ncols=5, figsize=(12, 8),
                         subset_line_color="red", full_line_color="blue", outfile=None):
    """
    Plots combined growth curves for multiple species.
    Plots OD measured from monoculture growth experiments and OD measured from monoculture exometabolomics sampling experiments.
    """

    growth_df = growth_df.copy()

    species_list = sorted(od_time_df['species'].unique())
    n_species = len(species_list)
    nrows = int(np.ceil(n_species /ncols))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, species in enumerate(species_list):
        ax = axes[i]

        #OD for each species during metabolomics experiment
        subset_data = od_time_df[od_time_df['species'] == species].sort_values('time')
        ax.plot(subset_data['time'], subset_data['OD'], '-o',
                color=subset_line_color, label='Exometab samples', lw=2)
        for t, od in zip(subset_data['time'], subset_data['OD']):
            ax.axvline(x=t, color=subset_line_color, linestyle=':', alpha=0.7)
            ax.plot(t, od, 'o', color=subset_line_color)

        #full growth curve from unperturbed monoculture growth
        full_data = growth_df[growth_df['species'] == species].sort_values('Time')
        ax.plot(full_data['Time'], full_data['OD'], '-',
                color=full_line_color, label='growth curve', lw=1.5, alpha=0.8)

        ax.set_title(species, fontsize=12)
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel("OD600")
        ax.grid(False)
        ax.set_xlim(0,60)

    #hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=600, bbox_inches='tight')

    return

def plot_Sfig_4(df, outfile=None):
    """
    Plot growth and metabolite parameters used to analytically estimate CRM growth parameters.
    """
    #subplot for each species
    species_list = df["species_code"].unique()
    N = len(species_list)
    fig = plt.figure(figsize=(10, 5))
    for i, sp in enumerate(species_list):
        ax = fig.add_subplot(3, 5, i+1)

        sub = df[df["species_code"] == sp].sort_values("point_index")

        #plot points
        ax.scatter(sub["x"], sub["y"], color="k", s=10)

        #plot fitted line
        ax.plot(sub["x"], sub["yfit"], color="grey")

        #species name
        ax.set_title(sub["species_name"].iloc[0], fontsize=10)

        ax.tick_params(labelsize=7)
        ax.xaxis.get_offset_text().set_size(7)

        #R^2 (from r_value)
        r2 = sub["r_value"].iloc[0] ** 2
        ax.text(
            0.05, 0.95, f"$R^2 = {r2:.2f}$",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="left"
        )

    fig.supxlabel(r'$\sum_{\alpha} \omega_{\alpha} C_{i\alpha} R_{\alpha}$', fontsize=12)
    fig.supylabel("dN/N·dt", fontsize=12)
    plt.subplots_adjust(wspace=0.4, hspace=0.6, left=0.07, bottom=0.17)

    if outfile:
        plt.savefig(outfile, dpi=600)

    return

def plot_Sfig_5b(C1, C2, outfile=None):
    """
    Compare Cmatrix values for metabolite uptake derived from fitting vs. analytical methods.
    """
    sps = utils.sps
    fig = plt.figure(figsize=(8, 5))
    x, y = [], []
    palette = utils.get_species_colormap(name_key=False)
    ax = fig.add_subplot(111)
    
    #loop through each row and assign a species-specific color
    for i in range(C2.shape[0]):
        x = C2[i]
        y = C1[i]
        ax.scatter(x, y, label=utils.get_species_name(sps[i]), fc='none', ec=palette[sps[i]])

    #figure plotting specifications
    min_val = min(C1.min(), C2.min())  
    max_val = max(C1.max(), C2.max())  
    ax.plot([min_val, max_val], [min_val, max_val], color='grey', linestyle='--', linewidth=1, label="1:1")
    ax.set_xlabel(r'$C_{i\alpha}$ derived from analytical method (mL/hr)')
    ax.set_ylabel(r'$C_{i\alpha}$ fitted from simulated annealing (mL/hr)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='center right', bbox_to_anchor=(1.45, 0.5), fontsize=9)
    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=500)

    return

def plot_Mfig_2b(Cmatrix, glist, l, met_class_df, sp_order, outfile=None):
    """
    Plot Fitted C-Matrix parameters in Metabolite Classes.
    Use same species order as 2a plot, scale by energy conversion factor.
    """
    #species_name -> species_code
    name_to_code = {
        utils.get_species_name(code): code
        for code in Cmatrix.index
    }
    #convert names to codes
    sp_order_codes = [str(name_to_code[name]) for name in sp_order]
    
    #load fitted Cmatrix
    Cmatrix, D_dict, l, glist, cfu = utils.load_fitted_params()
    plot_l = pd.DataFrame(l['sucrose'])

    Cmatrix = Cmatrix.loc[sp_order_codes]
    glist = glist.loc[sp_order_codes]
    plot_l = plot_l.loc[sp_order_codes]
    plot_matrix = Cmatrix

    #scale by energy conversion factor w, constant for all metabolites
    plot_matrix = Cmatrix * 10**12

    #build ordered column list based on class order
    class_order = ["Sugar", "Organic_Acid", "Amino_Acid", "Nucleobase", "Others"]
    ordered_cols = []
    for cls in class_order:
        mets = (
            met_class_df
            .loc[met_class_df.metabolite_class == cls, "metabolite"]
            .tolist()
        )
        mets = [m for m in mets if m in plot_matrix.columns]
        ordered_cols.extend(mets)

    #force this order on Cmatrix
    plot_matrix = plot_matrix[ordered_cols]

    #row colors based on metabolite classes
    palette = sns.color_palette("husl", n_colors=len(class_order))
    class_colors = {cls: palette[i] for i, cls in enumerate(class_order)}

    lut = {}
    for cls in class_order:
        color = class_colors[cls]
        mets = met_class_df.loc[met_class_df.metabolite_class == cls, "metabolite"]
        for m in mets:
            if m in ordered_cols:
                lut[m] = color

    col_colors = plot_matrix.columns.map(lut)

    #plot main clustermap
    g = sns.clustermap(
        plot_matrix,
        col_colors=col_colors,
        figsize=(12, 6),
        row_cluster=False,
        col_cluster=False,
        cmap='Blues',
        norm=LogNorm(vmin=1e-8, vmax=max(plot_matrix.max()))
    )

    g.ax_heatmap.set_yticklabels([])
    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.set_xticks([])

    ax = g.ax_heatmap

    #add metab category labels/colors
    start = 0
    boundaries = []

    for cls in class_order:
        mets = [m for m in ordered_cols if lut[m] == class_colors[cls]]
        n = len(mets)
        end = start + n
        boundaries.append((cls, start, end))
        ax.axvline(end - 0.05, color='dimgray', linewidth=2)
        start = end

    for cls, start, end in boundaries:
        mid = (start + end) / 2
        ax.text(mid, -1.5, cls.replace("_", " "), ha='center', va='center', fontsize=12)

    #shift heatmap to include g and l parameter columns
    g.gs.update(left=0.01, right=0.90)
    gs = gridspec.GridSpec(1, 2, right=0.16, left=0.1, top=0.8, wspace=0.5)

    # keep vmin/vmax for shared (g,l) scale
    vmin, vmax = 0, 1.2

    #plot g parameters by species
    ax2 = g.fig.add_subplot(gs[0])
    sns.heatmap(
        glist, cmap="Blues", cbar=False, ax=ax2,
        vmin=vmin, vmax=vmax,
        yticklabels=[utils.get_species_name(idx) for idx in glist.index]
    )
    ax2.set_xticks([])
    ax2.set_xlabel("$g_i$", fontsize=12)
    ax2.xaxis.set_label_position('top')

    #plot l parameters by species
    ax3 = g.fig.add_subplot(gs[1])
    sns.heatmap(
        plot_l, cmap="Blues", cbar=False, ax=ax3,
        vmin=vmin, vmax=vmax
    )
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlabel(r"$l_{\alpha}$", fontsize=12)
    ax3.xaxis.set_label_position('top')

    #other plotting options
    g.cax.set_title('Energy\nUptake Rate\n$C_{i\\alpha}\\cdot\\omega_{\\alpha}$',
                    fontsize=11, pad=15)
    g.fig.text(
        0.5, 0.2,
        '            Fitted CRM Metabolite Usage Parameters',
        ha='center', va='top', fontsize=14
    )
    g.ax_cbar.set_position([0.93, 0.25, 0.03, 0.4])

    if outfile:
        plt.savefig(outfile, dpi=600, bbox_inches='tight')

    return

def plot_Mfig_2c(init_sp_x, fit_sp_x, exp_abundance, outfile=None):
    """
    Plot final growth abundance from simulating CRm monoculture growth with initial and fitted 
    parameter estimated, compare to observed experimental final growth yields.
    """
    
    #assume constant OD to cfu conversion for all strains
    cfu_conv = 10**9

    #load experimental monoculture data
    od_final = exp_abundance.iloc[-1]
    exp_cfu_final = cfu_conv * od_final
    init_cfu_final = init_sp_x.loc[50]
    fitted_cfu_final = fit_sp_x.loc[50]  

    #calculate pearson correlation
    i_r, i_p = scipy.stats.pearsonr(exp_cfu_final.values.astype('float'), init_cfu_final.values.astype('float'))
    f_r, f_p = scipy.stats.pearsonr(exp_cfu_final.values.astype('float'), fitted_cfu_final.values.astype('float'))

    fig, axes = plt.subplots(2, 1, figsize=(5, 8), sharex=True)
    palette = utils.get_species_colormap()

    for idx, val in enumerate(init_cfu_final):
        axes[0].scatter(exp_cfu_final.values[idx]/10**9, val/10**9, color='royalblue')
    axes[0].axline((0, 0), slope=1, c='grey', ls='--')
    axes[0].set_title('Initial Model Parameters', fontsize=12)
    axes[0].text(0.05, 1.8, f'r = {i_r:.2f}\np = {i_p:.2f}', 
             transform=axes[1].transAxes, fontsize=11)

    for idx, val in enumerate(fitted_cfu_final):
        axes[1].scatter(exp_cfu_final.values[idx]/10**9, val/10**9, label=utils.get_species_name(fitted_cfu_final.index.to_list()[idx]), color='royalblue')
    axes[1].axline((0, 0), slope=1, c='grey', ls='--')
    axes[1].set_xlabel('Measured Population Abundance ($10^9$ cfu/mL)', fontsize=12)
    axes[1].set_title('Optimized Model Parameters', fontsize=12)
    axes[1].set_ylim(0,1.6)
    axes[1].text(0.05, 0.75, f'r = {f_r:.2f}\np = {f_p:.2e}', 
             transform=axes[1].transAxes, fontsize=11)

    fig.text(-0.02, 0.5, 'Predicted Population Abundance ($10^9$ cfu/mL)', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, bbox_inches='tight', dpi=500)

    return

def plot_Mfig_2d(fit_met_df, metab_time_df, met_class_df, outfile=None):
    """
    Plot final metabolite abundance from simulating CRM monoculture growth with fitted params
    compared to observed experimental final timepoint metabolite abundances.
    """
    
    #metabolite classes map to colors
    classes = met_class_df['metabolite_class'].unique()
    class_colors = {
        cls: c for cls, c in zip(classes, sns.color_palette("husl", n_colors=len(classes)))
    }
    met_to_class = dict(
        zip(met_class_df['metabolite'], met_class_df['metabolite_class'])
    )

    #extract metabolite list
    rep_sp = fit_met_df['species'].unique()[0]
    rep_df = fit_met_df[fit_met_df['species'] == rep_sp]
    metabolite_cols = [
        c for c in rep_df.columns if c not in ['species', 'time']
    ]

    #assign colors for each metabolite
    colors = [
        class_colors.get(met_to_class.get(met, "Unknown"), "gray")
        for met in metabolite_cols
    ]


    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 9), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    axes = axes.flatten()

    #loop over species, one species per subplot
    for sp_i, sp_code in enumerate(utils.sps):
        ax = axes[sp_i]

        #convert species code to species name
        sp_name = utils.get_species_name(sp_code)

        #extract final timepoint per species
        exo_df = metab_time_df[metab_time_df['species'] == sp_name]

        #find final experimental exometabolomics timepoint
        final_time_exp = exo_df['time'].max()
        exo_last = exo_df[exo_df['time'] == final_time_exp]
        exo_usage_map = dict(zip(exo_last['metabolite'], exo_last['median_usage']))
        exo_usage = np.array([
            exo_usage_map.get(m, np.nan) for m in metabolite_cols
        ], dtype=float)

        #subset simulated metabolite data by species 
        sim_df = fit_met_df[fit_met_df['species'] == int(sp_code)].copy()
        
        if sim_df.empty:
            ax.set_title(f"{sp_name} (no sim data)")
            print('sim_df is empty')
            continue

        sim_df['time'] = sim_df['time'].astype(float)

        #find equivalent final timepoint in simulated data
        final_row = sim_df.loc[
            np.isclose(sim_df['time'], final_time_exp, atol=1e-6),
            metabolite_cols
        ]

        #simulated abundance at the start of the simulation
        init_row = sim_df.loc[
            np.isclose(sim_df['time'], 0.0, atol=1e-6),
            metabolite_cols
        ]

        end_resource_abundance = final_row.values.flatten().astype(float)
        init_resource_abundance = init_row.values.flatten().astype(float)

        #normalize the simulated final abundance by the initial, same normalization done in experiments with blank media
        with np.errstate(divide='ignore', invalid='ignore'):
            sim_usage = (end_resource_abundance - init_resource_abundance) / init_resource_abundance

        #plot simulated vs. measured usage
        ax.scatter(sim_usage, exo_usage,
                   color=colors, linewidth=1.5, s=60, alpha=0.5)

        ax.set_title(sp_name, fontsize=15)

        #pearson correlation
        mask = ~np.isnan(sim_usage) & ~np.isnan(exo_usage)
        if np.sum(mask) > 1:
            r_val, p_val = scipy.stats.pearsonr(sim_usage[mask], exo_usage[mask])
            corr_text = f"r = {r_val:.2f}"
        else:
            corr_text = "r = --"

        ax.text(0.95, 0.05, corr_text,
                ha='right', va='bottom',
                transform=ax.transAxes, fontsize=14)

        #axes formatting
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(y=0, c='grey', linewidth=0.6)
        ax.axvline(x=0, c='grey', linewidth=0.6)


    fig.text(0.5, 0.04, 'Simulated Resource Usage', ha='center', fontsize=18)
    fig.text(0.07, 0.5, 'Measured Resource Usage', va='center',
             rotation='vertical', fontsize=18)

    if outfile:
        plt.savefig(outfile, dpi=600)

    return


if __name__ == "__main__": 
    parser = ArgumentParser(description="Generate all figures from CSV data.")
    parser.add_argument("--data_dir", required=True, help="Directory with input CSV files.")
    parser.add_argument("--out", required=True, help="Directory to save output figures.")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    #load processed data
    metab_class_df = pd.read_csv(os.path.join(args.data_dir, "met_class_df.csv"))
    metab_time_df = pd.read_csv(os.path.join(args.data_dir, "monoculture_exp/met_time_df.csv"))
    metab_dR_df = pd.read_csv(os.path.join(args.data_dir, "monoculture_exp/met_dR_df.csv"))
    od_time_df = pd.read_csv(os.path.join(args.data_dir, "monoculture_exp/od_time_df.csv"))
    growth_df_all_timepoints = pd.read_csv(os.path.join(args.data_dir, "monoculture_exp/growth_df_all_timepoints.csv"))
    cmat_fitted = pd.read_csv(os.path.join(args.data_dir, "final_crm_params/cmat_fitted.csv"), index_col=0)
    cmat_init = pd.read_csv(os.path.join(args.data_dir, "init_crm_params/cmat_init.csv"), index_col=0)
    init_sp_mono = pd.read_csv(os.path.join(args.data_dir, "monoculture_sim/init_mono_sp_df.csv"), index_col=0)
    fit_sp_mono = pd.read_csv(os.path.join(args.data_dir, "monoculture_sim/fit_mono_sp_df.csv"), index_col=0)
    growth_df_clean = pd.read_csv(os.path.join(args.data_dir, "monoculture_exp/growth_df_clean.csv"))
    glist_fitted = pd.read_csv(os.path.join(args.data_dir, "final_crm_params/glist_fitted.csv"))
    l_fitted = pd.read_csv(os.path.join(args.data_dir, "final_crm_params/l_fitted.csv"),)
    gparam_df = pd.read_csv(os.path.join(args.data_dir, "init_crm_params/gparam_df.csv"))
    fit_met_df = pd.read_csv(os.path.join(args.data_dir, "monoculture_sim/fit_mono_met_df.csv"))
    init_met_df = pd.read_csv(os.path.join(args.data_dir, "monoculture_sim/init_mono_met_df.csv"))

    #plot figs
    plot_Sfig_1(metab_class_df, metab_time_df, outfile=os.path.join(args.out, "Sfig_1.png"))
    col_order = plot_Mfig_2a(metab_class_df, metab_time_df, outfile=os.path.join(args.out, "Mfig_2a.png"))
    plot_Sfig_2(metab_class_df, metab_dR_df, outfile=os.path.join(args.out, "Sfig_2.png"))
    plot_Sfig_3(od_time_df, growth_df_all_timepoints, outfile=os.path.join(args.out, "Sfig_3.png"))
    plot_Sfig_5b(np.array(cmat_fitted), np.array(cmat_init), outfile=os.path.join(args.out, "Sfig_5b.png"))
    plot_Mfig_2c(init_sp_mono, fit_sp_mono, growth_df_clean, outfile=os.path.join(args.out, "Mfig_2c.png"))
    plot_Mfig_2b(cmat_fitted, glist_fitted, l_fitted, metab_class_df, col_order, outfile=os.path.join(args.out, "Mfig_2b.png"))
    plot_Sfig_4(gparam_df, outfile=os.path.join(args.out, "Sfig_4.png"))
    plot_Mfig_2d(fit_met_df, metab_time_df, metab_class_df, outfile=os.path.join(args.out, "Mfig_2d_fit.png"))
    plot_Mfig_2d(init_met_df, metab_time_df, metab_class_df, outfile=os.path.join(args.out, "Mfig_2d_init.png"))
