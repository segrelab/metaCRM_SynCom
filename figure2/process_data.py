"""
Figure 2: Monoculture Metabolite Processing and CRM Fitting/Simulation
"""
from argparse import ArgumentParser
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
import statistics
sys.path.append('../.')
import utils

growth_dir = '/projectnb/cometsfba/Archived_misc/mCAFEs/Jing_syncom_project_Nov2023/DATA/Growth'

def process_all_metabolite_data(
        data,           # utils.load_all_metdata()
        met_classes,    # dict: {class: [list of metabolites]}
        ODs,            # utils.load_od_data()
        tps,            # utils.tps, e.g. ['1','2','3','4']
        alpha_df        # utils.load_met_conc()
):
    """
    Processes the nested metabolite data structure and returns:

      1. met_class_df – tidy metabolite → class mapping
      2. met_time_df  – longitudinal metabolite data
      3. met_dR_df    – dR/R/dt
      4. od_time_df   – longitudinal OD data (species × time × OD)

    Species codes are converted to species names using utils.get_species_name().
    """

    # ---------------------------------------------------------
    # Build tidy metabolite-class dataframe
    # ---------------------------------------------------------
    met_class_list = [(m, cls) for cls, mets in met_classes.items() for m in mets]
    met_class_df = pd.DataFrame(met_class_list, columns=["metabolite", "metabolite_class"])
    met_to_class = dict(zip(met_class_df.metabolite, met_class_df.metabolite_class))

    # ---------------------------------------------------------
    # Output DataFrames
    # ---------------------------------------------------------
    time_rows, dR_rows, od_rows = [], [], []

    # Flatten alpha table
    alpha_values = alpha_df.values.flatten().tolist()

    # =========================================================
    # Loop through species
    # =========================================================
    for species_code, sdata in data.items():
        species_name = utils.get_species_name(species_code)

        # Time intervals using sucrose
        suc_times = [sdata["sucrose"][tp]["Time"] for tp in tps if "sucrose" in sdata and tp in sdata["sucrose"]]
        tintervals = utils.calc_interval(suc_times) if suc_times else None

        # ---------------------------
        # Build OD dataframe
        # ---------------------------
        for tp in tps:
            met_for_time = next((m for m in met_to_class if m in sdata and tp in sdata[m]), None)
            if met_for_time and f"T{tp}" in ODs[species_code]:
                od_rows.append({
                    "species": species_name,
                    "time": sdata[met_for_time][tp].get("Time", None),
                    "OD": ODs[species_code][f"T{tp}"]
                })

        # ---------------------------
        # Loop over metabolites
        # ---------------------------
        for met, cls in met_to_class.items():
            if met not in sdata:
                continue

            # Metabolite-time dataframe
            for tp in tps:
                if tp not in sdata[met]:
                    continue
                entry = sdata[met][tp]
                reps = [entry[r] for r in ["R1","R2","R3","R4"] if r in entry]
                if not reps:  # silently skip empty lists
                    continue
                time_rows.append({
                    "species": species_name,
                    "metabolite": met,
                    "metabolite_class": cls,
                    "time": entry.get("Time", None),
                    "median_val": statistics.median(reps)
                })

            # dR/R/dt dataframe
            try:
                a_index = list(met_class_df.metabolite).index(met)
                alpha = alpha_values[a_index]
            except:
                alpha = 1.0

            available_tps = [tp for tp in tps if tp in sdata[met]]
            if not available_tps:
                continue

            R_vals = []
            for tp in available_tps:
                reps = [sdata[met][tp][r] for r in ["R1","R2","R3","R4"] if r in sdata[met][tp]]
                if reps:
                    R_vals.append(statistics.median(reps) * alpha)
                else:
                    R_vals.append(np.nan)  # silently use NaN if no replicates

            dt_vals = tintervals[:len(R_vals)] if tintervals else [np.nan]*len(R_vals)

            R_prev = alpha
            x_vals, y_vals = [0], [0]

            for i, tp in enumerate(available_tps):
                R = R_vals[i]
                N = float(ODs[species_code][f"T{tp}"])
                dt = dt_vals[i] if i < len(dt_vals) else dt_vals[0]

                dR_Rdt = (R - R_prev) / ((R + R_prev)/2 * dt) if dt and (R + R_prev)/2 else 0.0
                x_vals.append(N)
                y_vals.append(dR_Rdt)
                R_prev = R

            for N_val, dR_val in zip(x_vals[1:], y_vals[1:]):
                dR_rows.append({
                    "species": species_name,
                    "metabolite": met,
                    "metabolite_class": cls,
                    "N": N_val,
                    "dR/Rdt": dR_val
                })

    met_time_df = pd.DataFrame(time_rows)
    met_time_df['median_usage'] = met_time_df['median_val'] - 1
    met_dR_df = pd.DataFrame(dR_rows)
    od_time_df = pd.DataFrame(od_rows)

    return met_class_df, met_time_df, met_dR_df, od_time_df

def load_tab_data(file):
    """Load plate reader .txt data as a tidy dataframe."""
    df = pd.read_csv(file, sep='\t')
    df = df.melt(id_vars=df.columns[0], var_name='Strain', value_name='OD')
    df = df.rename(columns={df.columns[0]: 'Time'})
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df = df.dropna(subset=['Time'])
    return df


def load_manual_growth(file):
    """Load manually measured growth data (csv) with R1-R3 replicates."""
    df = pd.read_csv(file)
    df = df.melt(id_vars=['Strain', 'Time'], var_name='Replicate', value_name='OD')
    df['Strain'] = df['Strain'].astype(str)
    df['Time'] = pd.to_numeric(df['Time'])
    df['OD'] = pd.to_numeric(df['OD'], errors='coerce')
    return df


def compile_growth_data(sps, growth_data_dir):
    """Compile growth data for each strain from multiple sources."""
    R0 = load_tab_data(os.path.join(growth_data_dir,'NLDM-30C-12well.txt'))
    R1 = load_tab_data(os.path.join(growth_data_dir, 'NLDM-30C-R1.txt'))
    R2 = load_tab_data(os.path.join(growth_data_dir, 'NLDM-30C-R2.txt'))
    R3 = load_tab_data(os.path.join(growth_data_dir, 'NLDM-30C-R3.txt'))
    R4 = load_tab_data(os.path.join(growth_data_dir, 'NLDM-30C-R4.txt'))
    R5 = load_manual_growth(os.path.join(growth_data_dir,'121122_mcafes_growth.csv'))

    all_data = []
    for sp in sps:
        if sp in ['1338']:
            df = R3[R3['Strain'] == sp].copy()
        elif sp in ['1330', '1336']:
            df = R0[R0['Strain'] == sp].copy()
        else:
            df = R5[R5['Strain'] == sp].copy()
        df['Strain'] = sp
        all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)
    df_all['species'] = df_all['Strain'].apply(utils.get_species_name)
    df_all['Time'] = df_all['Time'] / 60.0
    
    return df_all

def gparam_dict_to_df(g_param_dict):
    """Convert nested g_param_dict into tidy DataFrame."""
    records = []
    for sp_code, d in g_param_dict.items():
        species_name = utils.get_species_name(sp_code)
        for i, (x, y, yfit) in enumerate(zip(d["x"], d["y"], d["yfit"])):
            records.append({
                "species_code": sp_code,
                "species_name": species_name,
                "point_index": i,
                "x": x,
                "y": y,
                "yfit": yfit,
                "g": d["g"],
                "r_value": d["r_value"]
            })
    return pd.DataFrame(records)


def mcrm_initialparams_run(sps=utils.sps, cfu_conv=1e9, plot=False, time=500, step=0.01):
    """Simulate monoculture growth using initial parameters."""
    w = 1e12
    Cmatrix = utils.derive_cmatrix()
    D_dict = utils.derive_Dmatrix_perspecies()
    glist = utils.derive_g()
    l = pd.DataFrame(0.2, index=Cmatrix.index, columns=Cmatrix.columns)
    init_met_conc = utils.load_met_conc().rename(columns={"Concentration (g/mL)": "x0"})
    init_sp = pd.DataFrame(0.01 * cfu_conv, index=glist.index, columns=['x0'])

    metab_x_dict = {}
    for i, sp in enumerate(sps):
        C_i = Cmatrix.loc[[sp]]
        g_i = glist.loc[[sp]]
        l_i = l.loc[[sp]]
        x0_combined = pd.concat([init_sp.loc[[sp]]['x0'], init_met_conc])
        params = utils.mcrm_params('eye', x0_combined, C_i, D_dict, l_i, g_i,
                                   1e9, w, 1e9, 0, 0.04, time, step)
        sp_x, met_x = utils.run_mcrm(params)
        metab_x_dict[sp] = met_x
        if i == 0:
            tot_sp_x = sp_x
        else:
            tot_sp_x[sp] = sp_x.values
        if plot:
            print(utils.get_species_name(sp))

    # Convert D_dict-style metabolite dictionaries to long dataframe
    metab_df = pd.concat([df.assign(species=sp, time=df.index) for sp, df in metab_x_dict.items()],
                         axis=0).reset_index(drop=True)

    return tot_sp_x, metab_df


def run_monoculture(sps=utils.sps, cfu_conv=1e9, plot=False, time=500, step=0.01):
    """Simulate monoculture growth using fitted parameters."""
    w = 1e12
    metab_x_dict = {}
    for i, sp in enumerate(sps):
        Cmatrix, D_dict, l, glist, cfu = utils.load_fitted_params(sps=[sp])
        init_sp = pd.DataFrame(0.01 * cfu_conv, index=glist.index, columns=['x0'])
        x0_combined = pd.concat([init_sp['x0'], utils.load_met_conc().rename(columns={"Concentration (g/mL)": "x0"})])
        params = utils.mcrm_params('eye', x0_combined, Cmatrix, D_dict, l, glist,
                                   cfu_conv, w, cfu_conv, 0, 0.04, time, step)
        sp_x, met_x = utils.run_mcrm(params)
        metab_x_dict[sp] = met_x
        if i == 0:
            tot_sp_x = sp_x
        else:
            tot_sp_x[sp] = sp_x.values

    metab_df = pd.concat([df.assign(species=sp, time=df.index) for sp, df in metab_x_dict.items()],
                         axis=0).reset_index(drop=True)

    return tot_sp_x, metab_df


def main():
    # -------------------------
    # ARGUMENTS
    # -------------------------
    growth_dir = '/projectnb/cometsfba/Archived_misc/mCAFEs/Jing_syncom_project_Nov2023/DATA/Growth'

    parser = ArgumentParser()
    parser.add_argument(
        "--growth_data_dir", type=Path, default=growth_dir,
        help="Directory with plate reader OD data for monoculture experiments"
    )
    parser.add_argument(
        "--out", type=Path, default="processed_data",
        help="Directory to save output CSVs"
    )
    args = parser.parse_args()

    # Make main output folder
    args.out.mkdir(exist_ok=True, parents=True)

    # Make subfolders for CRM parameter outputs
    init_crm_dir = args.out / "init_crm_params"
    final_crm_dir = args.out / "final_crm_params"
    init_crm_dir.mkdir(exist_ok=True)
    final_crm_dir.mkdir(exist_ok=True)

    # -------------------------
    # PROCESS METABOLITE DATA
    # -------------------------
    metab_class_df, metab_time_df, metab_dR_df, od_time_df = process_all_metabolite_data(
        utils.load_all_metdata(),         
        utils.load_met_classes(),   
        utils.load_od_data(),          
        utils.tps,          
        utils.load_met_conc()       
    )

    # -------------------------
    # PROCESS GROWTH DATA
    # -------------------------
    growth_df_all_timepoints = compile_growth_data(utils.sps, growth_dir)
    growth_df_clean = pd.DataFrame.from_dict(
        utils.load_od_data(path=os.path.join(args.growth_data_dir, 'od_timepoints_1211.csv')),
        orient='columns'
    )

    # -------------------------
    # LOAD CRM PARAMETERS
    # -------------------------
    # Final fitted
    cmat_fitted, d_dict_fitted, l_fitted, glist_fitted, cfu = utils.load_fitted_params()
    
    # Initial
    gparam_df = gparam_dict_to_df(utils.derive_g(for_plot=True))
    cmat_init = utils.derive_cmatrix(norm=False)
    D_dict_init = utils.derive_Dmatrix_perspecies()
    glist_init = utils.derive_g()
    l_init = pd.DataFrame(0.2, index=cmat_init.index, columns=cmat_init.columns)

    # -------------------------
    # RUN MONOCULTURE CRM SIMULATIONS
    # -------------------------
    init_sp_mono, init_met_df = mcrm_initialparams_run()
    fit_sp_mono, fit_met_df = run_monoculture()
    #save only 0.1 intervals for metabolite df
    def is_multiple_of_point1(x, tol=1e-6):
        return abs((x / 0.1) - round(x / 0.1)) < tol

    init_met_df = init_met_df[init_met_df['time'].apply(is_multiple_of_point1)]
    fit_met_df  = fit_met_df[fit_met_df['time'].apply(is_multiple_of_point1)]
    init_met_df['time'] = init_met_df['time'].round(1)
    fit_met_df['time'] = fit_met_df['time'].round(1)
    
    # -------------------------
    # SAVE DATAFRAMES
    # -------------------------

    # Experimental metabolomics and growth outputs
    metab_class_df.to_csv(args.out / "met_class_df.csv", index=False)
    metab_time_df.to_csv(args.out / "met_time_df.csv", index=False)
    metab_dR_df.to_csv(args.out / "met_dR_df.csv", index=False)
    od_time_df.to_csv(args.out / "od_time_df.csv", index=False)
    growth_df_all_timepoints.to_csv(args.out / "growth_df_all_timepoints.csv", index=False)
    growth_df_clean.to_csv(args.out / "growth_df_clean.csv", index=False)

    # Initial CRM parameters
    gparam_df.to_csv(init_crm_dir / "gparam_df.csv", index=False)
    cmat_init.to_csv(init_crm_dir / "cmat_init.csv")
    pd.concat([df.assign(species=sp) for sp, df in D_dict_init.items()]).to_csv(init_crm_dir / "d_dict_init.csv", index=False)
    l_init.to_csv(init_crm_dir / "l_init.csv")
    glist_init.to_csv(init_crm_dir / "glist_init.csv")

    # Final CRM parameters
    cmat_fitted.to_csv(final_crm_dir / "cmat_fitted.csv")
    pd.concat([df.assign(species=sp) for sp, df in d_dict_fitted.items()]).to_csv(final_crm_dir / "d_dict_fitted.csv", index=False)
    l_fitted.to_csv(final_crm_dir / "l_fitted.csv")
    glist_fitted.to_csv(final_crm_dir / "glist_fitted.csv")

    # Monoculture simulations
    init_sp_mono.to_csv(args.out / "init_mono_sp_df.csv")
    fit_sp_mono.to_csv(args.out / "fit_mono_sp_df.csv")
    init_met_df.to_csv(args.out / "init_mono_met_df.csv", index=False)
    fit_met_df.to_csv(args.out / "fit_mono_met_df.csv", index=False)

if __name__ == "__main__":
    main()
