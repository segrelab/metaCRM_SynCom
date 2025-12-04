"""
Figure 4: Process CRM 16S and simulation data for leave-one-out and whole-community experiments
"""
import sys
import os
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.append(".")
import utils


def load_OTU_table(inputfile):
    """ load processed OTU table """
    OTU_dict = {}
    lines = [open(inputfile, 'r').read().strip("\n")][0].split('\n')
    for line in lines[1:]:
        tokens = line.split('\t')
        s_id = tokens[0]
        try:
            sp1, sp2 = int(tokens[1]), int(tokens[2])
        except ValueError: 
            sp1, sp2 = int(tokens[1][1:]), int(tokens[2][1:])
        sample_data = {'sp': (sp1,sp2)}
        for ind, token in enumerate(tokens[3:], 1):
            try:
                sample_data[ind] = float(token) 
            except ValueError: 
                sample_data[ind] = token
        OTU_dict[s_id] = sample_data
    return OTU_dict

def make_df_all(OTU_dict):
    """Average replicates at each time point to make a df, and include replicate-specific output for r1–r4."""

    sp_data = {}
    sp_data_r1, sp_data_r2, sp_data_r3, sp_data_r4 = {}, {}, {}, {}
    sps = list(range(1, 16))
    ps = list(range(1, 8))

    for sp in sps:
        sp_t, sp_r1_t, sp_r2_t, sp_r3_t, sp_r4_t = [], [], [], [], []
        for p in ps:
            r_d = []
            for r in range(1, 5):
                for idx in OTU_dict:
                    passage = int(OTU_dict[idx]['sp'][0])
                    replicate = int(OTU_dict[idx]['sp'][1])
                    if p == passage and r == replicate:
                        total = sum([OTU_dict[idx][s] for s in sps])
                        if total > 0:
                            r_d.append(100 * OTU_dict[idx][sp] / total)
                        else:
                            r_d.append(0.0)
            # Ensure we have exactly 4 replicates
            while len(r_d) < 4:
                r_d.append(0.0)
            avg_r = np.average(r_d)
            sp_t.append(avg_r)
            sp_r1_t.append(r_d[0])
            sp_r2_t.append(r_d[1])
            sp_r3_t.append(r_d[2])
            sp_r4_t.append(r_d[3])
        sp_data[sp] = sp_t
        sp_data_r1[sp] = sp_r1_t
        sp_data_r2[sp] = sp_r2_t
        sp_data_r3[sp] = sp_r3_t
        sp_data_r4[sp] = sp_r4_t

    # Normalize average data
    norm_sp_data = {sp: [] for sp in sps}
    for p in ps:
        abun_list = [sp_data[sp][p - 1] for sp in sps]
        norm_abun_list = [el / sum(abun_list) for el in abun_list]
        for sp in sps:
            norm_sp_data[sp].append(norm_abun_list[sp - 1])

    # Convert to DataFrame with species labels
    species_labels = utils.sps  # list of 15 species names
    df = pd.DataFrame(norm_sp_data, index=ps)
    df.columns = species_labels
    df.index.name = 'Passage'

    df = df.loc[1:5]

    # Normalize and store each replicate separately in a single long-form DataFrame
    dfs = []
    for label, sp_data_r in zip(
        ['r1', 'r2', 'r3', 'r4'],
        [sp_data_r1, sp_data_r2, sp_data_r3, sp_data_r4]
    ):
        norm_data_r = {sp: [] for sp in sps}
        for p in ps:
            abun_list = [sp_data_r[sp][p - 1] for sp in sps]
            norm_abun_list = [abun / sum(abun_list) for abun in abun_list]
            for sp in sps:
                norm_data_r[sp].append(norm_abun_list[sp - 1])
        df_r = pd.DataFrame(norm_data_r, index=ps)
        df_r.columns = species_labels
        df_r.index.name = 'Passage'
        df_r = df_r.loc[1:5]
        df_r = df_r.reset_index().melt(id_vars="Passage", var_name="species", value_name="value")
        df_r["replicate"] = label
        dfs.append(df_r)

    df_replicates_long = pd.concat(dfs, ignore_index=True)
    
    return df, df_replicates_long

def load_experiment_2_df(path):

    cols = ['sp_left_out', 'Passage', 'rep', '1319', '1320', '1321', '1323', '1324', 
        '1325', '1327', '1329', '1330', '1331', '1334', '1337', '1338', '1336', '1538', 'misc']
    df = pd.read_table(path, skiprows=1, names=cols)
    df.drop(['sp_left_out', 'misc'], axis=1, inplace=True)
    numeric_cols = df.columns.difference(['Passage', 'rep'])
    df[numeric_cols] = df[numeric_cols].div(df[numeric_cols].sum(axis=1),axis=0)
    df.Passage = df.Passage.str.replace("P",'').astype(int)

    #only passages 1-5 to match experiment A
    df_filtered = df[df["Passage"].isin(range(1, 6))]
    numeric_cols = df_filtered.columns.difference(['Passage', 'rep'])
    df_avg = df_filtered.drop(columns="rep").groupby("Passage").mean()
    #renorm to sum to 1
    df_avg = df_avg.div(df_avg.sum(axis=1), axis=0)
    df_final = df_avg

    return df_final, df

def simulate_whole_community_exp(tfs=4, time=48, crossfeeding=True, old=False, t0_abun=True, no_arth=False):
    """
    Simulate the whole community experiment.
    Returns two long-form DataFrames: sp_abun_long and met_abun_long.
    Columns: 'passage', 'species'/'metabolite', 'value'
    """
    
    t = time
    w = 10**12
    Cmatrix, D_dict, l, glist, cfu = utils.load_fitted_params()
    if crossfeeding == False:
        l = pd.DataFrame(0.0, index=l.index, columns=l.columns)

    # initial metabolite concentrations
    init_met_conc = utils.load_met_conc()
    init_met_conc.index.name = None
    init_met_conc.rename(columns={"Concentration (g/mL)": "x0"}, inplace=True)

    # initial species abundance
    init_sp_abun = pd.Series([10**9 * 0.01] * len(glist), index=utils.sps, name='x0')
    if t0_abun:
        init_abuns = [0.019849146, 0.093290988, 0.038507344, 0.30567686,
                      0.132195316, 0.132195316, 0.083366415, 0.003572846, 
                      0.001587932, 0.010718539, 0.026200873, 0.0, 0.0, 0, 0.138547042]
        init_sp_abun = pd.Series([(i + 0.00005) * 10**9 for i in init_abuns], index=utils.sps, name='x0')
    if no_arth:
        init_sp_abun.loc['1331'] = 0.0

    # helper function to convert wide DF to long form
    def to_long(df, passage):
        df_long = df.reset_index().melt(id_vars="index", var_name="species", value_name="value")
        df_long = df_long.rename(columns={"index": "time"})
        df_long["passage"] = passage
        return df_long

    sp_long_dfs = []
    met_long_dfs = []

    # first passage
    x0_combined = pd.concat([init_sp_abun, init_met_conc])
    params = utils.mcrm_params('eye', x0_combined, Cmatrix, D_dict, l, glist, cfu, w, k=0.04, old=old, time=t)
    sp_x, met_x = utils.run_mcrm(params, old=old)
    sp_long_dfs.append(to_long(sp_x, 1))
    met_long_dfs.append(to_long(met_x, 1))

    # subsequent passages
    for tf in range(tfs):
        x_s = pd.Series(sp_x.loc[47.99] * 0.02, index=utils.sps, name='x0')
        x0_combined = pd.concat([x_s, init_met_conc])
        params = utils.mcrm_params('eye', x0_combined, Cmatrix, D_dict, l, glist, cfu, w, k=0.04, old=old, time=t)
        sp_x, met_x = utils.run_mcrm(params, old=old)
        sp_long_dfs.append(to_long(sp_x, tf + 2))
        met_long_dfs.append(to_long(met_x, tf + 2))

    # concatenate all passages
    sp_abun_long = pd.concat(sp_long_dfs, ignore_index=True)
    met_abun_long = pd.concat(met_long_dfs, ignore_index=True)

    # clean up species abundance to sum to 1 and have t value per passage
    sp_abun_long = (sp_abun_long[sp_abun_long['time'] == 47.99]
            .pivot(index='passage', columns='species', values='value')
            .pipe(lambda d: d.div(d.sum(axis=1), axis=0)))

    return sp_abun_long, met_abun_long


def simulate_leave_one_out_exp(tfs=4, time=48, crossfeeding=True, old=False):
    '''
    Perform Leave-one-out simulations with all 15 strains. Save in final dataframe with index for species_left_out.
    '''

    t=time
    w=10**12

    #save results
    sp_abun = {}
    met_abun = {}
    
    #LEAVE-ONE-OUT    
    for sp in utils.sps:
        Cmatrix, D_dict, l, glist, cfu = utils.load_fitted_params()
        if crossfeeding==False:
            l = pd.DataFrame(0.0, index=l.index, columns=l.columns)

        #set initial metabolite concentrations
        init_met_conc = utils.load_met_conc()
        init_met_conc.index.name = None
        init_met_conc.rename(columns={"Concentration (g/mL)": "x0"}, inplace=True)

        #set initial species abundance
        init_sp_abun = pd.Series([10**9 * 0.01] * len(glist), index=utils.sps, name='x0')

        #combine into the initial conditions vector
        x0_combined = pd.concat([init_sp_abun, init_met_conc])
    
        for df in [x0_combined, Cmatrix, l, glist, cfu]:
            df.drop(sp, inplace=True)
        D_dict.pop(sp, None)

        #set up and run
        sp_abun[sp] = {}
        met_abun[sp] = {}
        params = utils.mcrm_params('eye', x0_combined, Cmatrix, D_dict, l, glist, cfu, w, k=0.04, old=old, time=t)
        sp_x, met_x = utils.run_mcrm(params, old=old)
        sp_abun[sp]['1'] = sp_x
        met_abun[sp]['1'] = met_x

        for tf in range(tfs):
            x_s = pd.Series(sp_x.loc[47.99]*0.02, index=utils.sps, name='x0')
            x0_combined = pd.concat([x_s, init_met_conc])
            x0_combined.drop(sp, inplace=True)
            params = utils.mcrm_params('eye', x0_combined, Cmatrix, D_dict, l, glist, cfu, w, k=0.04, old=old, time=t)
            sp_x, met_x = utils.run_mcrm(params, old=old)
            sp_abun[sp][str(tf+2)] = sp_x
            met_abun[sp][str(tf+2)] = met_x

    #transform sp_abun into a df with the final loo abundances
    sp_abun = {s: time_dict['5'] for s, time_dict in sp_abun.items() if '5' in time_dict}
    rows = []
    for s, df in sp_abun.items():
        # Get the last row
        last_row = df.iloc[-1]
        last_row_df = pd.DataFrame(last_row).T
        last_row_df.index = [df.index[-1]]
        # Add a column for species_left_out
        last_row_df['species_left_out'] = s
        rows.append(last_row_df)
    #concatenate rows, set index
    result_df = pd.concat(rows)
    result_df.set_index('species_left_out', inplace=True)
    result_df = result_df.fillna(0)
    result_df = result_df.div(result_df.sum(axis=1), axis=0)

    return result_df, met_abun


def process_leave_out_experiment(loo_data):
    
    loo_data = pd.read_table(loo_data, index_col=0)
    df = loo_data

    #convert column values to species
    df_main = df.drop(columns=["-1", "-2"], errors="ignore").copy()
    column_index_to_species = {
        (i+1): col.split("-")[1] for i, col in enumerate(df_main.columns)
        if "-" in col and col != "Total"
    }
    species_cols = [col for col in df_main.columns if "-" in col and col != "Total"]
    df_abundances = df_main[species_cols].copy()
    df_abundances.columns = [col.split("-")[1] for col in species_cols] 

    #renormalize to sum to 1 for each sample
    row_totals = df_abundances.sum(axis=1)
    df_abundances_normalized = df_abundances.div(row_totals, axis=0)

    #get left-out species from original df using column index mapping
    left_out_species = []
    for _, row in df.iterrows():
        species1 = column_index_to_species.get(int(row["-1"]), None)
        species2 = column_index_to_species.get(int(row["-2"]), None)
        left_out_species.append((species1, species2))

    #add left_out column
    df_abundances_normalized["left_out"] = left_out_species
    species_dict = {}
    unique_species = pd.unique([s for pair in df_abundances_normalized["left_out"] for s in pair])

    for species in unique_species:
        mask = df_abundances_normalized["left_out"].apply(lambda pair: species in pair)
        subset = df_abundances_normalized[mask].drop(columns=["left_out"])
        mean_series = subset.median()
        mean_series.index = [str(idx) for idx in mean_series.index]
        species_dict[species] = mean_series

    final_df = pd.DataFrame.from_dict(species_dict, orient='index')
    final_df = final_df.reset_index().rename(columns={'index': 'species_left_out'})
    final_df.set_index('species_left_out', inplace=True)
    final_df = final_df.div(final_df.sum(axis=1), axis=0)

    #fill diagonals with 0s (pair is same species twice)
    exp_loo_df = final_df.fillna(0)

    return exp_loo_df

def get_rescaled_abun(whole, row):
    '''
    Artificuially leave out one species from the whole community dataframe, used for normalizing LOO data.
    '''
    dropped = whole.drop(row['i'], errors='ignore')
    rescaled = dropped / dropped.sum()
    val = rescaled.get(row['sp'])
    return float(val) if val is not None else float('nan')

def calc_sim_loo_effects(sim_loo_df, whole_comm_sim_df):
    '''
    Calculate W_i effect on species j when i is left out. Use whole community data to 
    normalize, part of the process of calculating episttasis effects.
    '''

    loo_sim_vals = sim_loo_df.melt(ignore_index=False).reset_index()
    loo_sim_vals.columns = [['i', 'sp', 'abun_sp']]
    loo_sim_vals[['i', 'sp']] = loo_sim_vals[['i', 'sp']].applymap(utils.get_species_name)

    sim_abun = whole_comm_sim_df.loc[5]
    sim_abun.index = [utils.get_species_name(sp) for sp in sim_abun.index]
    sim_abun.astype(float)

    whole = sim_abun.astype(float)

    loo_sim_vals['sp_abun_in_rescaled_whole'] = loo_sim_vals.apply(lambda row: get_rescaled_abun(whole, row), axis=1)
    loo_sim_vals.dropna(inplace=True)

    loo_sim_vals['$W_i$'] = loo_sim_vals['abun_sp'].iloc[:, 0] / loo_sim_vals['sp_abun_in_rescaled_whole'].iloc[:, 0]
    loo_sim_vals.columns = [c[0] if isinstance(c, tuple) else c for c in loo_sim_vals.columns]

    return loo_sim_vals

def calc_coculture_interactions(coculture_data):
    '''
    From experimental co-culture data, calculate species-species interactions. 
    If W_ii is the fitness (determined by OD) of a double inoculation of strain i, 
    and W_ij is the fitness of i with j present, then the interaction of j on i is then defined as
    R_ij = W_ij - W_ii / (W_ij + W_ii)
    '''
    #translate numbers in co-culture data to species codes
    sp_key = {
        's1': '1319','s2': '1320','s3': '1321','s4': '1323','s5': '1324',
        's6': '1325','s7': '1327','s8': '1329','s9': '1330','s10': '1331',
        's11': '1334','s12': '1337','s13': '1338','s14': '1336','s15': '1538',
    }

    #load co-culture data file
    co_culture_df = pd.read_csv(coculture_data, index_col=0)

    #Change column names to be same as abbreviated species_1/species_2 names
    new_columns = [
        'species_1', 'species_2', 'final_od', 's1', 's2', 's3', 's4', 's5', 's6', 
        's7', 's8', 's9', 's10', 's11', 's12','s13', 's14', 's15', 's16', 's17'
    ]
    col_dict = {i: j for i, j in zip(co_culture_df.columns.to_list(), new_columns)}
    co_culture_df.rename(columns=col_dict, inplace=True)

    #remove samples with signficant contamination from 16S results
    co_culture_df = co_culture_df[
        ~(co_culture_df['species_1'].isin(['s16', 's17', 's18'])) | 
        (co_culture_df['species_2'].isin(['s16', 's17', 's18']))
    ]

    #separate metadata and 16S data, turn 16S counts to relative abundance to evaluate contamination
    excl_samples = [187, 200, 243, 244, 245, 246, 91, 94, 96, 119, 128, 131] #exclude these samples, not co-culture experiment
    co_culture_md = co_culture_df[['species_1', 'species_2', 'final_od']]
    co_culture_otu = co_culture_df.drop(['species_1', 'species_2', 'final_od'], axis=1)
    sp_cols = co_culture_otu.columns.to_list()
    sample_id = co_culture_md.index
    co_culture_otu = np.array(co_culture_otu.astype(float))
    co_culture_prop = pd.DataFrame(
        co_culture_otu / co_culture_otu.sum(1, keepdims=True),
        index=sample_id,
        columns=sp_cols
    )
    co_culture_prop = co_culture_prop.rename(index={'sample_id': 'sample.id'})
    co_culture_prop = co_culture_md.merge(co_culture_prop, on='sample.id')
    co_culture_prop['prop_pure'] = co_culture_prop.apply(
        lambda row: (row[row['species_1']] + row[row['species_2']]),
        axis=1
    )

    #Samples where 2 co-cultured species make up < 80% of 16S Reads -> contamination
    flagged_samples = co_culture_prop[co_culture_prop['prop_pure'] < 0.8].index

    #keep only relevant, uncontaminated samples in df
    remove_samples = list(set(flagged_samples).union(set(excl_samples)))
    clean_co_culture_prop = co_culture_prop[~co_culture_prop.index.isin(remove_samples)]
    clean_co_culture_prop = clean_co_culture_prop.drop(['s16', 's17', 'prop_pure'], axis=1)
    df = clean_co_culture_prop.copy()

    #get the list of species from the relative abundance columns
    abun_cols = df.columns.difference(['species_1', 'species_2', 'final_od'])
    species = abun_cols.tolist()

    # Helper to get all ij experiments regardless of order
    def match_ij(df, i, j):
        return df[
            ((df['species_1'] == i) & (df['species_2'] == j)) |
            ((df['species_1'] == j) & (df['species_2'] == i))
        ]

    #compute self-growth for each species (ii experiments)
    self_growth = {}
    for sp in species:
        ii_expts = match_ij(df, sp, sp)
        if not ii_expts.empty:
            # Use median in case of replicates
            self_growth[sp] = np.median(ii_expts['final_od'] * ii_expts[sp])
        else:
            self_growth[sp] = np.nan  # will help detect missing data
        if self_growth[sp] < 0.05:
            self_growth[sp] = 0.05

    #calculate the interaction matrix
    species_names = [sp_key[sp] for sp in species]
    interaction_matrix = pd.DataFrame(index=species_names, columns=species_names, dtype=float)

    for i in species:
        for j in species:
            if i == j:
                interaction = np.nan
                interaction_matrix.loc[sp_key[i], sp_key[j]] = interaction
                continue
            ij_expts = match_ij(df, i, j)
            if not ij_expts.empty and self_growth[i] is not np.nan:
                mean_effect = np.mean(ij_expts['final_od'] * ij_expts[i])
                if (mean_effect + self_growth[i]) == 0:
                    interaction = np.nan
                else:
                    #interaction metric!
                    interaction = (mean_effect - self_growth[i]) / (mean_effect + self_growth[i])
                interaction_matrix.loc[sp_key[i], sp_key[j]] = interaction

    #prepare df for saving with informative columns, index labels
    interaction_matrix = interaction_matrix.astype(float)
    interaction_matrix = pd.DataFrame(
        interaction_matrix.values,
        columns=[utils.get_species_name(i) for i in interaction_matrix.columns], 	
        index=[utils.get_species_name(i) for i in interaction_matrix.columns]
    )

    #remove rows where all values are Nan
    interaction_matrix.dropna(how='all', inplace=True)
    
    return interaction_matrix

def calc_loo_interactions(whole_community_abun, loo_abun):

    #load data
    whole_community = pd.read_csv(whole_community_abun, sep="\t", index_col=0)
    leave_one_out = pd.read_csv(loo_abun, sep="\t", index_col=0)

    #get species columns
    species_ids = utils.sps
    species_cols = [col for col in whole_community.columns.to_list()[2:-1]]

    #create a map from species ID to its full colname like '1-1319'
    species_id_to_colname = {
        species_id: f"{i+1}-{species_id}"
        for i, species_id in enumerate(species_ids)
    }


    #normalize species abundances to relative abundances
    whole_community_rel = whole_community[species_cols].div(whole_community['Total'], axis=0)
    leave_one_out_rel = leave_one_out[species_cols].div(leave_one_out['Total'], axis=0)

    species_code_to_row = {
        code: row_num for col in leave_one_out_rel.columns
        for row_num, code in [col.split('-')]
    }

    #subset whole community data to only last passage samples
    p5_mask = whole_community['-1'] == 'P5'
    whole_p5_avg = whole_community_rel[p5_mask].groupby(whole_community.loc[p5_mask, '-2']).mean().mean()

    #compute average across replicates for each leave-one-out case
    loo_interactions = pd.DataFrame(index=species_ids, columns=species_ids, dtype=float)
    col_to_species = {col: col.split('-')[1] for col in species_cols}

    #loop through leave-one-out experiments
    for species_j in species_ids:
        #rows where that species is left out
        row_number = species_code_to_row[species_j]  
        mask = leave_one_out['-1'].astype(str) == row_number
        loo_avg = leave_one_out_rel[mask].mean()

        #get the whole community profile and renormalize with that species artificially removed
        wc_without = whole_p5_avg.drop(f"{list(col_to_species.keys())[species_ids.index(species_j)]}")
        wc_without = wc_without / wc_without.sum()

        #compute interactions with all other species
        for species_i in species_ids:
            if species_i == species_j:
                continue
            #skip species with very little whole community growth
            if species_i in ('1334','1337','1338'):
                continue
            colname = species_id_to_colname[species_i]
            w_ij = loo_avg[colname]
            w_i_wt = wc_without[colname]
        
            if (w_ij + w_i_wt) == 0.0:
                R_ij = np.nan
            else:
                #interaction metric!
                R_ij = (w_i_wt - w_ij) / (w_i_wt + w_ij)
            
            loo_interactions.loc[species_i, species_j] = R_ij

    #fill missing vals and diagonals with NaN
    np.fill_diagonal(loo_interactions.values, np.nan)
    loo_interactions = pd.DataFrame(loo_interactions.values, 
                                    columns=[utils.get_species_name(i) for i in loo_interactions.columns], 
                                    index=[utils.get_species_name(i) for i in loo_interactions.index])

    #remove rows where all values are Nan
    loo_interactions.dropna(how='all', inplace=True)

    return loo_interactions

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--otu_table", type=Path, required=True, help="Path to whole-community OTU table")
    parser.add_argument("--exp2_table", type=Path, required=True, help="Path to experiment 2 OTU table")
    parser.add_argument("--loo_data", type=Path, required=True, help="Path to leave-one-out OTU data")
    parser.add_argument("--coculture_data", type=Path, required=True, help="Path to co-culture OTU/OD data")
    parser.add_argument("--out", type=Path, default="results", help="Directory to save processed output")
    args = parser.parse_args()

    #SETUP OUTPUT DIR AND FILE STRUCTURE
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    exp_whole_comm = args.out / "exp_whole_comm"
    sim_whole_comm = args.out / "sim_whole_comm"
    sim_loo = args.out / "sim_loo"
    exp_whole_comm.mkdir(exist_ok=True)
    sim_whole_comm.mkdir(exist_ok=True)
    sim_loo.mkdir(exist_ok=True)

    #PROCESS EXPERIMENTAL DATA
    OTU_dict = load_OTU_table(args.otu_table)
    df_a, df_a_reps = make_df_all(OTU_dict)
    df_b, df_b_reps = load_experiment_2_df(args.exp2_table)

    #SIMULATE CRM FOR WHOLE COMMUNITY
    df_sp_sim, df_met_sim = simulate_whole_community_exp(tfs=4, crossfeeding=True, t0_abun=False)
    df_sp_sim_nc, df_met_sim_nc = simulate_whole_community_exp(tfs=4, crossfeeding=False, t0_abun=False)
    df_sp_noarth, met_noarth = simulate_whole_community_exp(tfs=4, crossfeeding=True, t0_abun=False, no_arth=True)

    #PROCESS AND SIMULATE LEAVE-ONE-OUT DATA
    exp_loo_df = process_leave_out_experiment(args.loo_data)
    sp_loo, met_loo = simulate_leave_one_out_exp()
    sp_nc, met_nc = simulate_leave_one_out_exp(crossfeeding=False)
    sim_loo_effects = calc_sim_loo_effects(sp_loo, df_sp_sim)

    #CALCULATE SPECIES-SPECIES INTERACTIONS
    coculture_interactions = calc_coculture_interactions(args.coculture_data)
    loo_interactions = calc_loo_interactions(args.otu_table, args.loo_data)

    #SAVE PROCESSED DATA
    df_a.to_csv(exp_whole_comm / "df_a.csv")
    df_a_reps.to_csv(exp_whole_comm / "df_a_reps.csv", index=False)
    df_b.to_csv(exp_whole_comm / "df_b.csv")
    df_b_reps.to_csv(exp_whole_comm / "df_b_reps.csv", index=False)
    df_sp_sim.to_csv(sim_whole_comm / "wc_sp_sim.csv")
    df_sp_sim_nc.to_csv(sim_whole_comm / "wc_sp_sim_nc.csv")
    df_sp_noarth.to_csv(sim_whole_comm / "df_sp_noarth.csv")
    sp_loo.to_csv(sim_loo / "sp_loo.csv") 
    sp_nc.to_csv(sim_loo / "sp_loo_nc.csv")
    exp_loo_df.to_csv(out_dir / "exp_loo_df.csv")
    sim_loo_effects.to_csv(sim_loo / "sim_loo_effects.csv", index=False)
    coculture_interactions.to_csv(out_dir / "coculture_interactions.csv")
    loo_interactions.to_csv(out_dir / "loo_interactions.csv")

