"""
Figure 5: Epistasis
"""
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (8, 6)


palette = plt.get_cmap('bwr').with_extremes(under = "gray")
hue_norm = plt.Normalize(-1.0, 1.0)
direction_to_marker = {
  'beneficial': 'o',
  'deleterious': 's',
  'reciprocal':'^',
}


def plot_epistasis_distribution(epistasis_df: pd.DataFrame) -> plt.Figure:
  """Figure 5a"""
  ax = sns.histplot(epistasis_df, binwidth = 0.024, x = '$\epsilon$', hue = 'direction', multiple = 'stack')
  ax.set_xlabel("$\epsilon$")
  ax.set_title("Distribution of Epistatic Interaction Values")
  ax.get_legend().set_title('effect direction')
  return ax.figure


def plot_epistasis_legend() -> plt.Figure:
  legend_fig = plt.figure()
  cbar_ax = legend_fig.add_axes([0.05, 0.44, 0.05, 0.5])
  legend_fig.colorbar(cm.ScalarMappable(hue_norm, palette), cax = cbar_ax, orientation = "vertical")
  cbar_ax.tick_params(labelsize = 12)
  cbar_ax.set_title(r'$\epsilon$', fontsize = 12)
  cbar_ax.legend([mpl.patches.Patch(facecolor = palette.get_under(), edgecolor = 'black', linewidth = 0.75)], ['0 / 0'], frameon = False, 
                fontsize = 12,  bbox_to_anchor = [-0.38, -0.03], loc = 'upper left', handlelength = 2.5, handleheight = 1.4)
  label_to_handle = { 
    direction: mpl.lines.Line2D([0], [0], marker = marker, markersize = 12, color = 'black', linestyle = 'None')
    for direction, marker in direction_to_marker.items()
  }
  legend_fig.legend(label_to_handle.values(), label_to_handle.keys(), frameon = False, title = 'effect direction', 
                  fontsize = 12, title_fontsize = 12, bbox_to_anchor = [0, 0.35], loc = 'upper left')
  return legend_fig


def plot_pairwise_epistasis_values(epistasis_df: pd.DataFrame, k: str) -> plt.Figure:
  """Figure 5b (Arthrobacter)"""
  epistasis_zero_df = epistasis_df.copy()
  # Encode zero vals
  epistasis_zero_df.loc[epistasis_zero_df[["$W_{ij}$", "$W_i*W_j$"]].abs().sum(axis = 1) == 0.0, "$\epsilon$"] = -1.5
  ax = None
  # Add per-marker
  for direction, marker in direction_to_marker.items():
    if k in epistasis_zero_df[epistasis_zero_df['direction'] == direction].index.get_level_values('k'):
      ax = sns.stripplot(epistasis_zero_df[epistasis_zero_df['direction'] == direction].loc[:, :, k], x = 'j', y = 'i', hue = '$\epsilon$', 
                    hue_norm = hue_norm, palette = palette, jitter = False, edgecolor = 'black', marker = marker, linewidth = 1, size = 18,
                    ax = ax, legend = False)

  # Figure parameters
  ax.set_xlabel('Leave-out species j')
  ax.set_ylabel('Leave-out species i')
  ax.set_aspect('equal')
  ax.tick_params(axis = "x", rotation = 90)
  ax.set_title(f"Epistatic interaction effects on {k}")
  epistasis_df['0 growth'] = epistasis_zero_df['$\epsilon$'] == -1.5

  # Add histplot to lower left
  subax = inset_axes(ax, width = "40%", height = "40%", loc = "lower left", bbox_to_anchor = (0.075, 0.05, 1, 1), bbox_transform = ax.transAxes)
  sns.histplot(epistasis_df.loc[:, :, k], x = '$\epsilon$', hue = '0 growth', palette = { True: "gray", False: "steelblue" }, multiple = 'stack', 
                stat = 'density', hue_order = [True, False], binrange = (-1.0, 1.0), bins = 11, ax = subax, legend = False)
  sns.kdeplot(epistasis_df.loc[:, :, k], x = '$\epsilon$', clip = (-1.0, 1.0), ax = subax, legend = False)
  subax.set_xlabel("")
  subax.set_ylabel("")
  subax.spines['top'].set_visible(False)
  subax.spines['right'].set_visible(False)

  ax.figure.tight_layout()
  
  return ax.figure


def plot_leaveout_fitness_comparison(epistasis_df: pd.DataFrame, k: str) -> plt.Figure:
  """Figure 5c (Burkholderia) and 5d (Arthrobacter)"""
  ax = sns.scatterplot(epistasis_df.loc[:, :, k], x = '$W_i*W_j$', y = '$W_{ij}$', hue = '$\epsilon$', 
                     style = 'direction', markers = direction_to_marker, linewidth = 1, edgecolor = "black", 
                     hue_norm = hue_norm, palette = palette, legend = False)

  ax.set_title(f"Single and double-leaveout fitnesses, {k}")
  ax.set_xscale("symlog", linthresh = 0.001)
  ax.set_yscale("symlog", linthresh = 0.001)

  ax.axline((0.5, 0.5), (0.51, 0.51), color = "black", linestyle = "--", zorder = -1)
  ax.axhline(1, color = "black", linestyle = "--", zorder = -1)
  ax.axvline(1, color = "black", linestyle = "--", zorder = -1)
  ax.set_ylim(-0.0003, 100)
  ax.set_xlim(ax.get_ylim())
  ax.set_aspect('equal')

  return ax.figure


def plot_all_pairwise_epistasis(epistasis_df: pd.DataFrame) -> plt.Figure:
  """Supplemental Figure 11"""
  epistasis_zero_df = epistasis_df.copy()
  k_sps = epistasis_zero_df.index.get_level_values('k').unique()

  # Encode zero vals
  epistasis_zero_df.loc[epistasis_zero_df[["$W_{ij}$", "$W_i*W_j$"]].abs().sum(axis = 1) == 0.0, "$\epsilon$"] = -1.5
  grid = sns.catplot(epistasis_zero_df[epistasis_zero_df['direction'] == 'deleterious'], kind = 'strip', x = 'j', y = 'i', hue = '$\epsilon$', col = 'k', hue_norm = hue_norm, palette = palette,
                    jitter = False, edgecolor = 'black', marker = 's', linewidth = 1, size = 18, col_wrap = 5, col_order = sorted(k_sps))
  grid.legend.get_texts()[0].set_text("0 / 0")
  grid.add_legend({ direction: mpl.lines.Line2D([0], [0], marker = marker, markersize = 12, color = 'black', linestyle = 'None')
                  for direction, marker in direction_to_marker.items() }, 
                  title = 'direction', bbox_to_anchor = (1, 0.3))
  grid.tick_params(axis = "x", rotation = 90)
  grid.set_titles("Epistatic interaction effects on {col_name}")
  grid.set_axis_labels("Leave-out species j", "Leave-out species i")
  epistasis_df['0 growth'] = epistasis_zero_df['$\epsilon$'] == -1.5
  # Add histplot to lower left
  for ax, k in zip(grid.axes, grid.col_names):
    for direction, marker in direction_to_marker.items():
      if direction == 'deleterious': continue
      if k in epistasis_zero_df[epistasis_zero_df['direction'] == direction].index.get_level_values('k'):
        sns.stripplot(epistasis_zero_df[epistasis_zero_df['direction'] == direction].loc[:, :, k], x = 'j', y = 'i', hue = '$\epsilon$', hue_norm = hue_norm, palette = palette,
                      jitter = False, edgecolor = 'black', marker = marker, linewidth = 1, size = 18, ax = ax, legend = False)
      
    subax = inset_axes(ax, width = "40%", height = "40%", loc = "lower left", bbox_to_anchor = (0.075, 0.05, 1, 1), bbox_transform = ax.transAxes)
    sns.histplot(epistasis_df.loc[:, :, k], x = '$\epsilon$', hue = '0 growth', palette = { True: "gray", False: "steelblue" }, multiple = 'stack', 
                stat = 'density', hue_order = [True, False], binrange = (-1.0, 1.0), bins = 11, ax = subax, legend = False)
    sns.kdeplot(epistasis_df.loc[:, :, k], x = '$\epsilon$', clip = (-1.0, 1.0), ax = subax, legend = False)
    subax.set_xlabel("")
    subax.set_ylabel("")
    subax.spines['top'].set_visible(False)
    subax.spines['right'].set_visible(False)

  grid.tight_layout()
  grid.figure.set_size_inches(26, 12)

  return grid.figure


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument("--epistasis", help = "Filepath to the CSV containing epistasis calculation results", type = Path, default = "data/epistasis_vals.csv")
  parser.add_argument("--out", help = "Folder to output figure plots to", type = Path, default = "figures")

  args = parser.parse_args()

  outf = args.out
  epistasis_df = pd.read_csv(args.epistasis, index_col = list('ijk'))

  fig5a = plot_epistasis_distribution(epistasis_df)
  fig5a.savefig(f'{outf}/Fig5a.pdf')
  plt.close(fig5a)
  
  fig5b = plot_pairwise_epistasis_values(epistasis_df, 'Arthrobacter')
  fig5b.savefig(f'{outf}/Fig5b.pdf')
  plt.close(fig5b)

  fig5c = plot_leaveout_fitness_comparison(epistasis_df, 'Burkholderia')
  fig5c.savefig(f'{outf}/Fig5c.pdf')
  plt.close(fig5c)

  fig5d = plot_leaveout_fitness_comparison(epistasis_df, 'Arthrobacter')
  fig5d.savefig(f'{outf}/Fig5d.pdf')
  plt.close(fig5d)

  fig5legend = plot_epistasis_legend()
  fig5legend.savefig(f'{outf}/Fig5legend.pdf')
  plt.close(fig5legend)

  figS11 = plot_all_pairwise_epistasis(epistasis_df)
  figS11.savefig(f'{outf}/FigS11.pdf')
  plt.close(figS11)