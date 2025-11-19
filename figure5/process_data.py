"""
Figure 5: Epistasis
"""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.append('.')
from utils import sps_names, load_leaveout_table, load_passage_table


def calc_growth(abundance_df: pd.DataFrame, sps_in_exp: list[str], column: str, val) -> pd.Series:
  """
  Calculate growth of each species in given scenario as a percent of total community growth
  """
  filtered = abundance_df[abundance_df[column] == val]
  filtered_pcts = filtered[sps_in_exp].T / filtered['Total']
  avg_growth = filtered_pcts.mean(axis = 1)
  return avg_growth


def calculate_epistasis(whole_community: pd.DataFrame, leave_one: pd.DataFrame, leave_two: pd.DataFrame) -> pd.DataFrame:
  sps_in_exp = [c for c in whole_community.columns if c in sps_names]

  # calculate fitness of time passage communities (no leaveouts)
  whole_community_growth = calc_growth(whole_community, sps_in_exp, 'P', "P5")

  # Drop low-growth species from k consideration
  k_sps = [*sps_in_exp]
  k_sps.remove("Brevibacillus")
  k_sps.remove("Marmoricola")
  k_sps.remove("Methylobacterium")
  k_sps.remove("Paenibacillus")
  k_sps.remove("Rhodococcus")

  # calculate fitness of communities with leaveouts = leaveout growth over wild-type growth
  leaveout_one_fitness = pd.DataFrame({ i: (calc_growth(leave_one, sps_in_exp, 'leaveout', (i, i)) / 
                                            whole_community_growth * 
                                            (1 - whole_community_growth[i]))[k_sps] 
                                            for i in sps_in_exp }).T
  leaveout_one_fitness.index.set_names("i", inplace = True)
  leaveout_two_fitness = pd.DataFrame({ (i, j): (calc_growth(leave_two, sps_in_exp, 'leaveout', (i, j)) / 
                                                whole_community_growth * 
                                                (1 - (whole_community_growth[i] + whole_community_growth[j])))[k_sps] 
                                                for j in sps_in_exp for i in sps_in_exp if j > i }).T
  leaveout_two_fitness.index.set_names(["i", "j"], inplace = True)

  # Merge values of interest into one dataframe
  w_i = leaveout_one_fitness.melt(var_name = "k", value_name = "$W_i$", ignore_index = False).set_index("k", append = True)
  w_i = w_i[w_i.index.map(lambda ik: ik[0] != ik[1])]
  w_ij = leaveout_two_fitness.melt(var_name = "k", value_name = "$W_{ij}$", ignore_index = False).set_index("k", append = True)
  w_ij = w_ij[w_ij.index.map(lambda ijk: ijk[0] != ijk[2] and ijk[1] != ijk[2])]

  epistasis_df = pd.merge(w_i, w_ij, left_index = True, right_index = True).reorder_levels(["i", "j", "k"]).sort_index()
  epistasis_df["$W_j$"] = epistasis_df.index.map(lambda ijk: w_i.loc[ijk[1], ijk[2]]["$W_i$"])

  # Calculate epistasis values and direction
  epistasis_df["$W_i*W_j$"] = epistasis_df["$W_i$"] * epistasis_df["$W_j$"]
  epistasis_df = epistasis_df[["$W_i$", "$W_j$", "$W_i*W_j$", "$W_{ij}$"]]
  epistasis_df["$\epsilon$"] = (epistasis_df["$W_{ij}$"] - epistasis_df["$W_i*W_j$"]) / (epistasis_df["$W_{ij}$"] + epistasis_df["$W_i*W_j$"]).replace(0, 1)
  epistasis_df["direction"] = "deleterious"
  epistasis_df.loc[epistasis_df["$W_{ij}$"] > 1, "direction"] = "beneficial"
  epistasis_df.loc[np.sign(epistasis_df["$W_{ij}$"] - 1) != np.sign(epistasis_df["$W_i*W_j$"] - 1), "direction"] = "reciprocal"

  return epistasis_df



if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument("--whole-community", help = "Filepath to the txt containing whole-community abundance data", type = Path)
  parser.add_argument("--leave-one-out", help = "Filepath to the txt containing leave-one-out abundance data", type = Path)
  parser.add_argument("--leave-two-out", help = "Filepath to the txt containing leave-two-out abundance data", type = Path)
  parser.add_argument("--out", help = "Folder to output processed data to", type = Path, default = "data")

  args = parser.parse_args()

  # load data
  leave_one = load_leaveout_table(args.leave_one_out)
  leave_two = load_leaveout_table(args.leave_two_out)
  whole_community = load_passage_table(args.whole_community)
  
  # generate epistasis values
  epistasis_df = calculate_epistasis(whole_community, leave_one, leave_two)

  # save
  epistasis_df.to_csv(f'{args.out}/epistasis_vals.csv')