"""
Process data for Figure 3, pairwise co-culture experiments and simulations.
"""

#Modified passages
#Import necessary packages
import sys
from pathlib import Path
import os
import pandas as pd
import numpy as np

# Save and load dictionaries
import pickle
from argparse import ArgumentParser

import statistics
import scipy.stats

# Incl helper functions from utils
sys.path.append(os.path.abspath("../"))
import utils

# Global variables..

# Map each species to an integer
INV_MAP = {1:'1319',2:'1320',3:'1321',4:'1323',5:'1324',6:'1325',7:'1327',8:'1329',9:'1330',10:'1331',11:'1334',12:'1337',13:'1338',14:'1336',15:'1538',16:'1602',17:'1597'}

##### FUNCTIONS ###############################################################

def pairs_list(sps: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for i in range(15):
        for j in range(i, 15):
            pairs.append((sps[i], sps[j]))

    return pairs

def process_otu(OTU_dict: None) -> pd.DataFrame:
    """
    Processes OTU table and produces a dataframe with OTU information, trimming contaminated samples.
    """
    global INV_MAP
    
    OTU_dict = utils.load_otu_data()
    
    otu_df = pd.DataFrame.from_dict(OTU_dict)
    
    remove = [int(item) - 1 for item in utils.removed_samples]  # Remove contaminated wells
    
    otu_df.drop(otu_df.columns[remove], axis=1, inplace=True)
    
    otu_df_cols = otu_df.iloc[0].tolist()
    
    col_ids = [
        (INV_MAP[item[0]], INV_MAP[item[1]])
        for item in otu_df_cols
    ]
    
    otu_df.columns = col_ids
    otu_df.drop(labels='sp', axis=0, inplace=True)
    
    return otu_df

def process_ratios_exp(OTU_dict: None) -> pd.DataFrame:
    """
    Processes OTU table and produces a dataframe with information on
    relative abundance ratios for each pair.
    """
    global INV_MAP
    # Calculate ratios from OTU dict
    ratio_dict = utils.calc_ratios(OTU_dict)
    
    ratio_df = pd.DataFrame.from_dict(ratio_dict)
    ratio_df.sort_index(inplace=True)
    ratio_df.rename(index=INV_MAP, columns=INV_MAP, inplace=True)
    ratio_df.drop(['1597', '1602'], axis=0, inplace=True)

    np.fill_diagonal(ratio_df.values, 0)  # 0 instead of 1

    return ratio_df

def process_ratios_sim(abun_df: pd.DataFrame) -> pd.DataFrame:
    # New dataframe where column, index are species
    ratio_df = pd.DataFrame(columns=utils.sps, index=utils.sps)

    # Create a dataframe with the last column
    growth_final = abun_df.iloc[:, -1:]

    # Should look like mi, sp1 and sp2 with ods in final column
    # For each pair in the matrix, calc
    for (id1, id2), row in growth_final.iterrows():
        tup = row[14399]
        if len(tup) >= 2:
            value = tup[0] / (tup[0] + tup[1])
            ratio_df.loc[id1, id2] = 1 - value
            ratio_df.loc[id2, id1] = value
        else:  # Monoculture
            ratio_df.loc[id1, id2] = 0
    
    return ratio_df
    
def simulate_pairs(
    sps: list[str],
    Cmatrix: np.ndarray,
    D_dict: np.ndarray,
    lmatrix: np.ndarray,
    mu: np.ndarray,
    cfu: np.ndarray,
    w: float,
    x0_r: np.ndarray,
    tfs: int,
    l_zero_mode: bool) -> pd.DataFrame:
    """
    Function to run the CRM for each pair.
    Uses "old" nomenclature for running the mcrm (arrays, rather than dataframes).
    """    
    global INV_MAP
    # All pairs
    pairs = pairs_list(sps)

    # Time points
    tps = np.arange(14400)
    
    # Empty dataframe to populate results into
    mi = pd.MultiIndex.from_tuples(pairs, names=['sp1','sp2'])
    
    results_df = pd.DataFrame(index=mi, columns=tps)
                
    for i in range(15):
        for j in range(15):
            # Only do each pair once...
            if j > i:
                #For each pair, grab only the necessary parameters for running the simulation.
                Cmatrix_pair = Cmatrix[[i,j], :]
                mu_pair = [mu[i][0], mu[j][0]]
                cfu_pair = [cfu[i][0], cfu[j][0]]
                Dmatrix_dict = {0:D_dict[i], 1:D_dict[j]}
                l_pair = lmatrix[[i,j], :]
                
                # If leakage is set to zero (TRUE) then make l matrix 0.
                if l_zero_mode:
                    l_pair = np.zeros_like(l_pair)
                    l_pair.fill(0.0)

                # Set timepoint zero resources and species.
                x0_s = 0.01 * 10**9
                x0 = np.array(list(np.full((1, 2), x0_s)[0]) + list(x0_r))
                
                #Input params, run.
                params = utils.mcrm_params('eye', x0, Cmatrix_pair, Dmatrix_dict, l_pair, mu_pair, cfu_pair, w, k=0.04, old=True)
                x = utils.run_mcrm(params, old=True)
                y = x[:,0:2] # Retrieve the species abundances from the resulting list. 

                # Simulate transfers.
                for tf in range(tfs-1):
                    x_s = y[-1]*0.02
                    x0 = np.array(list(np.full((1, 2), x_s)[0]) + list(x0_r))
                    params = utils.mcrm_params('eye', x0, Cmatrix_pair, Dmatrix_dict, l_pair, mu_pair, cfu_pair, w, k=0.04, old=True)
                    x_tf = utils.run_mcrm(params, old=True)
                    y = np.concatenate((y, x_tf[:,0:2]), axis=0)

                key = (INV_MAP[i+1],INV_MAP[j+1])
                compress = list(map(tuple, y))
                results_df.loc[key, :] = compress
            
            # Simulated monoculture
            elif i == j:
                # Monoculture is a similar story to above, just with one species.
                Cmatrix_i = Cmatrix[[i], :]
                mu_i = [mu[i]]
                cfu_i = [cfu[i]]
                Dmatrix_dict = {0:D_dict[i]}
                l_single = lmatrix[[i], :]
                
                if l_zero_mode:
                    l_single = np.zeros_like(l_single)
                    l_single.fill(0.0)

                x0_s = 0.01 * 10**9
                x0 = np.array(list(np.full((1, 1), x0_s)[0]) + list(x0_r))
                params = utils.mcrm_params('eye', x0, Cmatrix_i, Dmatrix_dict, l_single, mu_i, cfu_i, w,k=0.04, old=True)
                
                x =  utils.run_mcrm(params, old=True)
                y = x[:,0:1]

                for tf in range(tfs-1):
                    x_s = y[-1]*0.02
                    x0 = np.array(list(np.full((1, 1), x_s)[0]) + list(x0_r))
                    params = utils.mcrm_params('eye', x0, Cmatrix_i, Dmatrix_dict, l_single, mu_i, cfu_i, w,k=0.04, old=True)
                    x_tf = utils.run_mcrm(params, old=True)
                    y = np.concatenate((y, x_tf[:,0:1]), axis=0)
                
                key = (INV_MAP[i+1],INV_MAP[j+1])
                compress = list(map(tuple, y))
                results_df.loc[key, :] = compress
                
    return results_df

def compute_pearson_correlations(exp_ratios: pd.DataFrame, sim_ratios_l: pd.DataFrame, sim_ratios_nl: pd.DataFrame) -> pd.DataFrame:
    # Modify experimental ratios dataframe to get the average
    # Calculate average ratio of dataframe
    pearson_df = pd.DataFrame(index=utils.sps, columns=['leakage on', 'leakage off'])
    
    for i in range(len(utils.sps)):
        sp_i = utils.sps[i]
        
        # Get the column 
        sim_l  = sim_ratios_l[sp_i].tolist()
        sim_nl = sim_ratios_nl[sp_i].tolist()
        
        # Get average ratio 
        exp = [sum(item) / len(item) if isinstance(item, list) else item for item in exp_ratios[sp_i].tolist()]

        pearson_no_leak = scipy.stats.pearsonr(sim_nl, exp)

        pearson_leak = scipy.stats.pearsonr(sim_l, exp)

        species_i_name = utils.get_species_name(sps[i])

        pearson_df.loc[sp_i, 'leakage off'] = pearson_no_leak[0]
        pearson_df.loc[sp_i, 'leakage on'] = pearson_leak[0]

    return pearson_df


##############################
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pairwise_coculture", type = Path, help = "Filepath to the .txt containing pairwise abundance data (OTUs)")
    parser.add_argument("--out", type = Path, default = "results", help = "Directory to save processed output.")
    args = parser.parse_args()

    # Create output directory if it doesn't exist.
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Load params
    OTU_dict = utils.load_otu_data(args.pairwise_coculture)

    # removed_samples, Samples that seemed to have contamination across wells. These pairs will have only one repeat as opposed to two.
    removed_samples = utils.removed_samples
    sps = utils.sps

    # estimated ratio of each species in co-culture (based on 16S and OD of co-culture)
    tfs = 3
    w = 10**12
    x0_r = utils.load_met_conc(old=True)

    Cmatrix, D_dict, lmatrix, glist, cfu = utils.load_fitted_params(old=True)

    # Run coculture
    sim_results_df_l  = simulate_pairs(sps, Cmatrix, D_dict, lmatrix, glist, cfu, w, x0_r, tfs, False)
    sim_results_df_nl = simulate_pairs(sps, Cmatrix, D_dict, lmatrix, glist, cfu, w, x0_r, tfs, True)

    # Process experimental results
    exp_results_otus   = process_otu(OTU_dict)
    exp_results_ratios = process_ratios_exp(OTU_dict)

    # Process simulation results
    sim_ratio_l  = process_ratios_sim(sim_results_df_l)
    sim_ratio_nl = process_ratios_sim(sim_results_df_nl) 
    
    # Calculate pearson correlations
    pearson_df = compute_pearson_correlations(exp_results_ratios, sim_ratio_l, sim_ratio_nl)
    
    # Save results
    sim_coculture = args.out / "sim_pairwise_coculture"
    exp_coculture = args.out / "exp_pairwise_coculture"
    
    # Create directories if they don't exist
    sim_coculture.mkdir(exist_ok=True)
    exp_coculture.mkdir(exist_ok=True)

    exp_results_ratios.to_csv(exp_coculture / "exp_coculture_ratios.csv")
    exp_results_otus.to_csv(exp_coculture / "exp_coculture_OTUs.csv")

    pearson_df.to_csv(sim_coculture / "sim-exp_pearson.csv")
    sim_ratio_l.to_csv(sim_coculture / "sim-coculture-ratio_leakage.csv")
    sim_ratio_nl.to_csv(sim_coculture / "sim-coculture-ratio_no-leakage.csv")
    
    sim_results_df_l.to_csv(sim_coculture / "sim-coculture-timeseries_leakage.csv")
    sim_results_df_nl.to_csv(sim_coculture / "sim-coculture-timeseries_no-leakage.csv")