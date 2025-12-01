"""
Figure 4: Process CRM 16S and simulation data for leave-one-out and whole-community experiments
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(".."))
import utils
from argparse import ArgumentParser

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

    #only passages 1-5
    df_filtered = df[df["Passage"].isin(range(1, 6))]
    numeric_cols = df_filtered.columns.difference(['Passage', 'rep'])
    df_avg = df_filtered.drop(columns="rep").groupby("Passage").mean()
    #renorm
    df_avg = df_avg.div(df_avg.sum(axis=1), axis=0)
    df_final = df_avg
    # Get first 5 passages and convert "P1" → 1, etc.
    #df_final.index = df_final.index.str.replace("P", "").astype(int)

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
        # Convert Series to DataFrame with 1 row
        last_row_df = pd.DataFrame(last_row).T
        # Set the index to the value of the original last row's index
        last_row_df.index = [df.index[-1]]
        # Add a column for species_left_out
        last_row_df['species_left_out'] = s
        rows.append(last_row_df)
    # Concatenate all rows
    result_df = pd.concat(rows)
    # Set index to species_left_out
    result_df.set_index('species_left_out', inplace=True)
    result_df = result_df.fillna(0)
    result_df = result_df.div(result_df.sum(axis=1), axis=0)

    return result_df, met_abun


def process_leave_out_experiment(loo_data):
    loo_data = pd.read_table(loo_data, index_col=0)
    df = loo_data
    df_main = df.drop(columns=["-1", "-2"], errors="ignore").copy()
    column_index_to_species = {
        (i+1): col.split("-")[1] for i, col in enumerate(df_main.columns)
        if "-" in col and col != "Total"
    }

    species_cols = [col for col in df_main.columns if "-" in col and col != "Total"]
    df_abundances = df_main[species_cols].copy()
    df_abundances.columns = [col.split("-")[1] for col in species_cols]  # → '1319', etc.
    row_totals = df_abundances.sum(axis=1)
    df_abundances_normalized = df_abundances.div(row_totals, axis=0)

    # Step 5: Get left-out species from original df using column index mapping
    left_out_species = []
    for _, row in df.iterrows():
        species1 = column_index_to_species.get(int(row["-1"]), None)
        species2 = column_index_to_species.get(int(row["-2"]), None)
        left_out_species.append((species1, species2))

    # Step 6: Add left_out column
    df_abundances_normalized["left_out"] = left_out_species

    # Step 7: Build dictionary
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

    exp_loo_df = final_df.fillna(0)

    return exp_loo_df

def get_rescaled_abun(whole, row):
    dropped = whole.drop(row['i'], errors='ignore')
    rescaled = dropped / dropped.sum()
    val = rescaled.get(row['sp'])
    return float(val) if val is not None else float('nan')

def calc_sim_loo_effects(sim_loo_df, whole_comm_sim_df):
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

def main(args):
    OTU_dict = load_OTU_table(args.otu_table)
    df_a, df_a_reps = make_df_all(OTU_dict)
    df_b, df_b_reps = load_experiment_2_df(args.exp2_table)

    df_sp_sim, df_met_sim = simulate_whole_community_exp(tfs=4, crossfeeding=True, t0_abun=False)
    df_sp_sim_nc, df_met_sim_nc = simulate_whole_community_exp(tfs=4, crossfeeding=False, t0_abun=False)
    df_sp_noarth, met_noarth = simulate_whole_community_exp(tfs=4, crossfeeding=True, t0_abun=False, no_arth=True)

    exp_loo_df = process_leave_out_experiment(args.loo_data)
    sp_loo, met_loo = simulate_leave_one_out_exp()
    sp_nc, met_nc = simulate_leave_one_out_exp(crossfeeding=False)
    sim_loo_effects = calc_sim_loo_effects(sp_loo, df_sp_sim)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)
    df_a.to_csv(out_dir / "df_a.csv")
    df_a_reps.to_csv(out_dir / "df_a_reps.csv", index=False)
    df_b.to_csv(out_dir / "df_b.csv")
    df_b_reps.to_csv(out_dir / "df_b_reps.csv", index=False)
    df_sp_sim.to_csv(out_dir / "wc_sp_sim.csv")
    df_met_sim.to_csv(out_dir / "wc_met_sim.csv", index=False)
    df_sp_sim_nc.to_csv(out_dir / "wc_sp_sim_nc.csv")
    df_met_sim_nc.to_csv(out_dir / "wc_met_sim_nc.csv", index=False)
    df_sp_noarth.to_csv(out_dir / "df_sp_noarth.csv")
    met_noarth.to_csv(out_dir / "met_noarth.csv", index=False)
    sp_loo.to_csv(out_dir / "sp_loo.csv") 
    sp_nc.to_csv(out_dir / "sp_loo_nc.csv")
    exp_loo_df.to_csv(out_dir / "exp_loo_df.csv")
    sim_loo_effects.to_csv(out_dir / "sim_loo_effects.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--otu-table", type=Path, required=True, help="Path to whole-community OTU table")
    parser.add_argument("--exp2-table", type=Path, required=True, help="Path to experiment 2 OTU table")
    parser.add_argument("--loo-data", type=Path, required=True, help="Path to leave-one-out OTU data")
    parser.add_argument("--out", type=Path, default="results", help="Directory to save processed output")
    args = parser.parse_args()
    main(args)

