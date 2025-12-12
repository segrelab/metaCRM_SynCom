'''
Figure 6. Sub-community design for increased Shannon Entropy in 3-species communities
'''
import sys
import os
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np
import itertools
import ast
sys.path.append(".")
import utils

def calc_shannon_diversity (array):
	"""
	determine shannon diveristy from an array of species abundance
	"""
	N = sum(array)
	H = 0
	for el in array: 
		if el != 0:
			H += - (el/N) * np.log(el/N)
	return H

def simulate_subcommunities(N, time=48):
    w = 10**12
    results = [] 
    idx = 0

    # initial metabolite concentrations
    init_met_conc = utils.load_met_conc()
    init_met_conc.index.name = None
    init_met_conc.rename(columns={"Concentration (g/mL)": "x0"}, inplace=True)

    for community in itertools.combinations(utils.sps, 3):
        Cmatrix, D_dict, l, glist, cfu = utils.load_fitted_params(sps=community)

        #initial species abundance
        init_sp_abun = pd.Series([10**9 * 0.01] * len(glist), index=community, name='x0')

        #run simulation
        x0_combined = pd.concat([init_sp_abun, init_met_conc])
        params = utils.mcrm_params('eye', x0_combined, Cmatrix, D_dict, l, glist, cfu, w, k=0.04, old=False, time=time)
        sp_x, met_x = utils.run_mcrm(params, old=False)

        #get final timepoint abundances, total yield and diversity
        end_abundance = sp_x.iloc[-1].values               # numpy array of abundances
        sd = calc_shannon_diversity(end_abundance.tolist())
        fitness = np.sum(end_abundance)

        #save results
        results.append({
            "Idx": idx,
            "Subcommunity species": ",".join(map(str, community)),
            "Fitness": fitness,
            "Shannon diversity sim": sd,
            "End abundance": ",".join(map(str, end_abundance))
        })

        idx += 1

    #convert results to dataframe
    results_df = pd.DataFrame(results)

    return results_df

def process_subcommunity_otu(subcomm_data):
    exp_subcomm = pd.read_table(subcomm_data, index_col=0)
    exp_subcomm['community_idx'] = exp_subcomm['Community'].str.extract(r'^S(\d+)_').astype(int)
    exp_subcomm['clean'] = exp_subcomm['Community'].str.split('_', n=1).str[1].apply(ast.literal_eval)
    exp_subcomm['Subcommunity species'] = exp_subcomm['clean'].apply(lambda x: ",".join(x))
    exp_subcomm = exp_subcomm[exp_subcomm['Subcommunity species'].apply(lambda x: len(x.split(',')) < 4)]
    exp_subcomm = (
        exp_subcomm[exp_subcomm["Passage"] == "P7"]
        .sort_values("community_idx", ascending=False)
        .copy()
    )
    species_cols = exp_subcomm.columns[4:19]  
    exp_subcomm['Shannon diversity exp'] = exp_subcomm[species_cols].apply(lambda row: calc_shannon_diversity(row.values), axis=1)

    exp_subcomm.drop(['Community', 'clean', 'Passage', 'Replicate'], axis=1, inplace=True)

    return exp_subcomm

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--subcomm_data", type=Path,
        help="Directory with 16S results from sub-community experiments"
    )
    parser.add_argument(
        "--out", type=Path, default="data",
        help="Directory to save output CSVs"
    )
    args = parser.parse_args()

    #add output folder structure
    args.out.mkdir(exist_ok=True, parents=True)

    #Simulate subcommunities with fitted-CRM
    sim_subcom = simulate_subcommunities(N=3)

    #Load experimental subcommunity OTUs
    #subcomm_data = '/projectnb/cometsfba/sibald/CRM_syncom/results/ilija/OTU_subcommunity.txt'
    exp_subcom = process_subcommunity_otu(args.subcomm_data)

    #Save dataframes
    sim_subcom.to_csv(os.path.join(args.out,'simulated_3sp_communities.csv'), index=False)
    exp_subcom.to_csv(os.path.join(args.out,'experimental_3sp_otu.csv'), index=False)
