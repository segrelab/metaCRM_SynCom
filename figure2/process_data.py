"""
Figure 2: Monoculture Metabolite Processing and CRM Fitting/Simulation
"""
from argparse import ArgumentParser
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
import pickle
import statistics
sys.path.append('.')
import utils

import random

def process_all_metabolite_data(
        data,           # utils.load_all_metdata()
        met_classes,    # dict: {class: [list of metabolites]}
        ODs,            # utils.load_od_data()
        tps,            # utils.tps
        alpha_df        # utils.load_met_conc()
):
    """
    Processes experimental metabolite data and returns:

      1. met_class_df – utility metabolite → class mapping
      2. met_time_df  – longitudinal metabolite data
      3. met_dR_df    – dR/R/dt - used for fitting CRM growth parameter
      4. od_time_df   – longitudinal OD data (species × time × OD)

    """

    #build metabolite-category df from dict
    met_class_list = [(m, cls) for cls, mets in met_classes.items() for m in mets]
    met_class_df = pd.DataFrame(met_class_list, columns=["metabolite", "metabolite_class"])
    met_to_class = dict(zip(met_class_df.metabolite, met_class_df.metabolite_class))

    #output dfs
    time_rows, dR_rows, od_rows = [], [], []

    #flatten initial metabolite concentrations to list
    alpha_values = alpha_df.values.flatten().tolist()

    #for each species
    for species_code, sdata in data.items():

        species_name = utils.get_species_name(species_code)

        #calculate time between exometabolomics sampling points
        suc_times = [sdata["sucrose"][tp]["Time"] for tp in tps if "sucrose" in sdata and tp in sdata["sucrose"]]
        tintervals = utils.calc_interval(suc_times) if suc_times else None

        #construct OD per time point df
        for tp in tps:
            met_for_time = next((m for m in met_to_class if m in sdata and tp in sdata[m]), None)
            if met_for_time and f"T{tp}" in ODs[species_code]:
                od_rows.append({
                    "species": species_name,
                    "time": sdata[met_for_time][tp].get("Time", None),
                    "OD": ODs[species_code][f"T{tp}"]
                })

        #construct normalized metabolite abundance timecourse dataframe
        for met, cls in met_to_class.items():
            if met not in sdata:
                continue

            for tp in tps:
                if tp not in sdata[met]:
                    continue
                entry = sdata[met][tp]
                reps = [entry[r] for r in ["R1","R2","R3","R4"] if r in entry]
                if not reps:  #skip empty lists
                    continue

                #take the median abundance across replicates
                time_rows.append({
                    "species": species_name,
                    "metabolite": met,
                    "metabolite_class": cls,
                    "time": entry.get("Time", None),
                    "median_val": statistics.median(reps)
                })

            #construct dR/R/dt dataframe
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
                    R_vals.append(np.nan)  #use NaN if no replicates

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

    #switch from normalized metabolite abundance (value of 1 means no change in abundance compared to blank)
    #instead use metabolite usage where + means production and - means consumption, still relative to blank abundances
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

    #Select continuous sampling data from slow growing species
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

    #define model parameters
    w = 1e12
    Cmatrix = utils.derive_cmatrix()
    D_dict = utils.derive_Dmatrix_perspecies()
    glist = utils.derive_g()
    l = pd.DataFrame(0.2, index=Cmatrix.index, columns=Cmatrix.columns)
    init_met_conc = utils.load_met_conc().rename(columns={"Concentration (g/mL)": "x0"})
    init_sp = pd.DataFrame(0.01 * cfu_conv, index=glist.index, columns=['x0'])

    #save abundances of metabolites and species at each time step
    metab_x_dict = {}

    #subset to one species at a time to simulate monoculture growth
    for i, sp in enumerate(sps):
        C_i = Cmatrix.loc[[sp]]
        g_i = glist.loc[[sp]]
        l_i = l.loc[[sp]]

        #run simulation
        x0_combined = pd.concat([init_sp.loc[[sp]]['x0'], init_met_conc])
        params = utils.mcrm_params('eye', x0_combined, C_i, D_dict, l_i, g_i, 1e9, w, 1e9, 0, 0.04, time, step)
        sp_x, met_x = utils.run_mcrm(params)
        metab_x_dict[sp] = met_x
        
        if i == 0:
            tot_sp_x = sp_x
        else:
            tot_sp_x[sp] = sp_x.values
        if plot:
            print(utils.get_species_name(sp))

    #convert D_dict-style metabolite dictionaries to long dataframe
    metab_df = pd.concat([df.assign(species=sp, time=df.index) for sp, df in metab_x_dict.items()],
                         axis=0).reset_index(drop=True)

    return tot_sp_x, metab_df


def run_monoculture(sps=utils.sps, cfu_conv=1e9, plot=False, time=500, step=0.01):
    """Simulate monoculture growth using fitted parameters."""
    w = 1e12
    metab_x_dict = {}
    
    #subset to one species at a time to simulate monoculture growth
    for i, sp in enumerate(sps):
        
        #load fitted parameters
        Cmatrix, D_dict, l, glist, cfu = utils.load_fitted_params(sps=[sp])

        #simulate, save species and metabolite abundance at each timestep
        init_sp = pd.DataFrame(0.01 * cfu_conv, index=glist.index, columns=['x0'])
        x0_combined = pd.concat([init_sp['x0'], utils.load_met_conc().rename(columns={"Concentration (g/mL)": "x0"})])
        params = utils.mcrm_params('eye', x0_combined, Cmatrix, D_dict, l, glist, cfu_conv, w, cfu_conv, 0, 0.04, time, step)
        sp_x, met_x = utils.run_mcrm(params)
        metab_x_dict[sp] = met_x
        if i == 0:
            tot_sp_x = sp_x
        else:
            tot_sp_x[sp] = sp_x.values

    #convert D_dict-style metabolite dictionaries to long dataframe
    metab_df = pd.concat([df.assign(species=sp, time=df.index) for sp, df in metab_x_dict.items()],
                         axis=0).reset_index(drop=True)

    return tot_sp_x, metab_df


############################################################################
###   Simulated Annealing, Not run here but placed here for reference.   ###
############################################################################

### fit C and D matrices using exometabolomics and single species growth ###

def isNaN(num):
	return num!= num

def residue(sp, x0, mu_i, w_alpha, Cmatrix_i, D, l_i, R_dict, sample_times, sample_ods, growth, finalOD, od_to_cfu, cfu_c, w1, w2):
	""" 
	set up a cost function to calculate the distance between simulated mets at each timepoints vs. exometabolomics measurement 
	"""
	# simulate dynamics of species growth and metabolite changes
	params = mcrm_params ('eye', x0, Cmatrix_i, D, l_i, [mu_i], [cfu_c], w_alpha)
	alpha = x0[params['varIdx']['resources']]
	x = run_mcrm(params)

	# determine the cost (difference from the measurements)
	cost = 0
	# difference in final OD
	weight = 10    
	cost = weight*(cfu_c*(x[-1].tolist())[0]/10**9 - finalOD*od_to_cfu/10**9)**2   # only consider the final OD

	try: 
		cost += weight*sum([(float(x[int(float(sample_times[sp][T]))].tolist()[0]/10**9) - float(sample_ods[sp][T]))**2 for T in ['T1', 'T2', 'T3'] if sample_ods[sp][T] != ''])   # compare to measured OD at sampling points

	# difference in metabolites at each timepoint
	weight2 = w2     
	tps = [sample_times[sp][t] for t in sample_times[sp]]

	for t_idx, t in enumerate(['1','2','3','4']):
		if tps[t_idx] != '':
			mets_sim = x[int(float(tps[t_idx]))][1:]
			mets_frc_sim = [(alpha[met_idx] - met_abun)/alpha[met_idx] for met_idx, met_abun in enumerate(mets_sim)]
			mets_ex = R_dict[sp][t]

			for el_i, el in enumerate(mets_ex):
				if not isNaN(el): 
					if t != '4':
						cost += weight2 * (mets_frc_sim[el_i] - el)**2
					else: 
						cost += weight2 * (mets_frc_sim[el_i] - el)**2

	return cost 

def small_step_perturbation(array, normed):
	"""
	perturbing each element of C by 5-10%
	"""
	new_array = np.zeros (shape=(array.shape[0], array.shape[1]))

	for idx, el in enumerate(list(array[0])):
		if el != 0 : 
			perm = random.gauss (0, el*0.1)
		else: 
			perm = random.gauss (0, (el+10**-14)*0.1)
		if el + perm > 0: new_array[0][idx] = el + perm
		else: new_array[0][idx] = 0

	if normed: 
		# normalize distribution for each species
		sum_of_rows = new_array.sum(axis=1)
		new_array = new_array / sum_of_rows[:, np.newaxis]
	return np.array (new_array)

def small_step_perturbation_D(D, normed):
	"""
	pertubing D matrix 
	"""
	idx_mets = [i for i in range(D.shape[0]) if max(D[i].tolist())!=0]  # get a list of metabolites produced
	new_D = np.zeros (shape=(D.shape[0], D.shape[1])) 
	for i in idx_mets: 
		array = D[i]
		for j, el in enumerate(list(array)):
			if i!=j: 
				perm = random.gauss (0, el*0.1)
				new_D[i][j] = el + perm 
	# renorm
	if normed: 
		new_D = new_D / new_D.sum(0)
		new_D = np.nan_to_num(new_D, copy=True, nan=0.0)
	return new_D

def small_step_perturbation_g(g):
	"""
	perturbing leakage fraction 
	"""
	g = random.gauss (0, 0.01*g) + g
	return g

def perturbation_cfu(c): 
	"""
	perturbing CFU conversion factor
	"""
	stepsize = 0.1
	bounds = np.asarray([[0.001, 1]])
	new_c = c + randn(len(bounds)) * stepsize
	return new_c[0]

def small_step_perturbation_lmatrix(l):
	"""
	perturbing l matrix
	"""
	stepsize = 0.01
	bounds = np.asarray([[0.01, 1]])
	new_l = np.zeros (shape=(l.shape[0], l.shape[1]))
	for i in range(l.shape[0]):
		l_p = l[i][0] + randn(len(bounds)) * stepsize
		if l_p >= 0: 
			for j in range(l.shape[1]):
				new_l[i][j] = l_p[0]
		else: 
			for j in range(l.shape[1]):
				new_l[i][j] = l[i][j]
	return new_l

def initialize_Dmatrix(sps, Cmatrix):
	"""
	initialize D matrix 
	"""
	R_dict, Tintervals = utils.get_timestep_resource_usage(sps) #use of each resource per species at 4 timepoints 
	alpha = utils.load_met_conc()
	D_dict = {}

	for i, sp in enumerate(sps):
		Cmatrix_i = Cmatrix[[i],:]
		D_p = utils.derive_Dmatrix_perspecies(R_dict[sp], alpha, Cmatrix_i, True)
		D_dict[i] = D_p 
	return D_dict

def simulated_annealing_randominitialization(sps, Cmatrix, w, growth_data, sa_params, modelname, runs):
	"""
	In N number of runs, initialize random Cmatrix, D matrix, and g, and perform simulated annealing for each species 
	"""
	# initialize Cmatrix
	Cmatrix2 = np.zeros (shape=(Cmatrix.shape[0], Cmatrix.shape[1]))
	C_params = []
	for i in range (len(sps)):
		C_array = Cmatrix[[i],:]
		C_params.extend([el for el in list(C_array[0]) if el >0])
	minC, maxC = min(C_params), max(C_params)
	for i in range (len(sps)):
		for j in range (Cmatrix.shape[1]):
			new_C = random.uniform (minC, maxC)
			Cmatrix2[i][j] = new_C
	
	# initialize mu
	new_mu = [0]*len(sps)
	for i in range (len(sps)):
		new_mu[i] = random.uniform (0.1, 50)
		new_mu[i] = random.randint (0, 50)

	# initalize Dmatrix 
	D_dict = {}
	for i in range(len(sps)):
		D = np.random.rand (Cmatrix.shape[1], Cmatrix.shape[1])
		D_normed = D / D.sum(0)  # normalize D matrix 
		D_dict[i] = D_normed    

	# perform simulated annealing
	simulated_annealing(sps, Cmatrix2, w, new_mu, growth_data, sa_params, modelname)


def simulated_annealing(sps, Cmatrix, w, mu, growth_data, sa_params, modelname):
	""" use simulated annealing approach to fit C matrix with the input of exometabolomics data and final OD abundance of each species in mono-culture growth """
	# load measurements 
	R_dict, Tintervals = utils.get_timestep_resource_usage(sps) #use of each resource per species at 4 timepoints 
	OTU_dict = utils.load_otu_data()
	sps_map = {1:'1319', 2:'1320', 3:'1321', 4:'1323', 5:'1324', 6:'1325', 7:'1327', 8:'1329', 9:'1330', 10:'1331', 11:'1334', 12:'1337', 13:'1338', 14:'1336', 15:'1538'}
	inv_map = {v: k for k, v in sps_map.items()}

	Times = utils.load_timepoints()
	ODs   = utils.load_od_data()
	alpha = utils.load_met_conc()

	# simulated annealing parameters
	n_iterations = sa_params[0]
	step_size = sa_params[1] 
	temp = sa_params[2]
	w1, w2 = sa_params[3], sa_params[4]

	for i in range (15):
		sp = sps[i]
		sp_idx = inv_map[sp]
		od_to_cfu = 10**9
		finalOD = float(ODs[sp]['T4']) * od_to_cfu / 10**9
		growth = growth_data[sp]
		x0_s = 0.01 * od_to_cfu
		x0 = np.array (list(np.full((1, 1), x0_s)[0]) + list(alpha))

		for run in range(1):
			Cmatrix_i = Cmatrix[[i],:]
			mu_i = 0.1*mu[i] # analytical solutions
			D_p = fitDmatrix_perspecies (R_dict[sp], alpha, Cmatrix_i, True)  # analytical solutions 
			D_p_normed = D_p / D_p.sum(0)  # normalize D matrix 
			D = {0:D_p_normed}           # get species i D matrix
			l = np.zeros (shape=(Cmatrix.shape[0], Cmatrix.shape[1]))    # leakage fraction ia
			l.fill (0.2)   # initialize l
			l_i = l[[i],:] 
			cfu_c = 1  
			# plot initial guess
			init_params = mcrm_params('eye', x0, Cmatrix_i, D, l_i, [mu_i], [cfu_c], w)

			best_theta = [mu_i, l_i, D[0], Cmatrix_i]   # permute mu, l and D 
			best_sqDiff = residue (sp, x0, mu_i, w, Cmatrix_i, D, l_i, R_dict, Times, ODs, growth, finalOD, od_to_cfu, cfu_c, w1, w2)
			curr, curr_sD = best_theta, best_sqDiff
			temps, scores = list(), list()
			for step in range(n_iterations):

				candidate = [small_step_perturbation_g(curr[0]), small_step_perturbation_lmatrix(curr[1]), small_step_perturbation_D(curr[2], True), small_step_perturbation(curr[3], False)]

				candidate_sD = residue (sp, x0, candidate[0], w, candidate[3],{0:candidate[2]}, candidate[1], R_dict, Times, ODs, growth, finalOD, od_to_cfu, cfu_c, w1, w2)

				if candidate_sD < best_sqDiff: 
					# store new best point
					best_theta, best_sqDiff = candidate, candidate_sD
                    
				# diff between candidate and current point evaluation
				diff = candidate_sD - curr_sD
                
				# calculate temperature for current epoch
				t = temp / float(step+1)

				if diff < 0: 
					# store the new current point 
					curr, curr_sD = candidate, candidate_sD 
				else: 
					# calcualte metropolis acceptance criterion
					try: 
						metropolis = math.exp (-diff/t)
						if np.random.rand() < metropolis: 
							curr, curr_sD = candidate, candidate_sD  # store the new current point 

					except OverflowError: 
						pass
				
				temps.append (t)
				scores.append (curr_sD)
		
			best_params = mcrm_params('eye', x0, best_theta[3], {0:best_theta[2]}, best_theta[1], [best_theta[0]], [cfu_c], w)

			pickle.dump(best_params, open(outpath+modelname+sp+'_run0.p', "wb"))


def load_simulated_annealing_scores(inputfile):
	"""
	load results of all simulated annealing runs
	"""
	lines = [open(inputfile, 'r').read().strip("\n")][0].split('\n')
	data = {}
	for line in lines[1:]: 
		tokens = line.split('\t')
		sp, run = tokens[0], int(tokens[1]) 
		temp = json.loads(tokens[2])
		scores = json.loads(tokens[3])
		if sp not in data: 
			data[sp] = {run:{'T':temp, 'S': scores}}
		else: 
			data[sp][run] = {'T':temp, 'S': scores}
	return data


def get_best_param(sp_i, sp, sa_params, params_list, growth_data):
	""" 
	from all runs, get the param with the lowest score
	"""
	min_score, best_param = 1000, ''

	# load measurements 
	R_dict, Tintervals = utils.get_timestep_resource_usage(sps) #use of each resource per species at 4 timepoints 
	OTU_dict = utils.load_otu_data()
	sps_map = {1:'1319', 2:'1320', 3:'1321', 4:'1323', 5:'1324', 6:'1325', 7:'1327', 8:'1329', 9:'1330', 10:'1331', 11:'1334', 12:'1337', 13:'1338', 14:'1336', 15:'1538'}
	inv_map = {v: k for k, v in sps_map.items()}

	Times = utils.load_timepoints()
	ODs   = utils.load_od_data()
	alpha = utils.load_met_conc()

	x0_s = 0.01 * 10**9
	x0 = np.array (list(np.full((1, 1), x0_s)[0]) + list(alpha))
	# load simulated annealing parameters
	w1, w2 = sa_params[3], sa_params[4]
	# load species specific data
	finalOD = od_dict[inv_map[sp]][inv_map[sp]]
	growth = growth_data[sp]
	w = 10**12	
	for params in params_list: 
		C = params['C']
		Cmatrix_i = params['C'] [[0],:]
		mu_i = params['g'][0]
		D = {0:params['D'][0]}
		l_i = params['l'][[0],:]
		cfu_i = params['cfu'][0]

		score = residue(sp, x0, mu_i, w, Cmatrix_i, D, l_i, R_dict, Times, ODs, growth, finalOD, cfu_i, w1, w2)
		if score < min_score: 
			min_score = score
			best_param = params 
	return best_param

######################################################################
###   Derive and Save CRM Parameters and Monoculture Data cleanly  ###
######################################################################


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--growth_data_dir", type=Path,
        help="Directory with plate reader OD data for monoculture experiments"
    )
    parser.add_argument(
        "--out", type=Path, default="data",
        help="Directory to save output CSVs"
    )
    args = parser.parse_args()

    #add output folder structure
    args.out.mkdir(exist_ok=True, parents=True)
    
    init_crm_dir = args.out / "init_crm_params"
    final_crm_dir = args.out / "final_crm_params"
    mono_sim_dir = args.out / "monoculture_sim"
    mono_exp_dir = args.out / "monoculture_exp"
    
    init_crm_dir.mkdir(exist_ok=True)
    final_crm_dir.mkdir(exist_ok=True)
    mono_sim_dir.mkdir(exist_ok=True)
    mono_exp_dir.mkdir(exist_ok=True)

    #process experimental monoculture metabolite data
    metab_class_df, metab_time_df, metab_dR_df, od_time_df = process_all_metabolite_data(
        utils.load_all_metdata(),         
        utils.load_met_classes(),   
        utils.load_od_data(),          
        utils.tps,          
        utils.load_met_conc()       
    )

    #PROCESS DATA
    
    #process experimental monoculture growth data
    growth_df_all_timepoints = compile_growth_data(utils.sps, args.growth_data_dir)
    growth_df_clean = pd.DataFrame.from_dict(
        utils.load_od_data(path=os.path.join(args.growth_data_dir, 'od_timepoints_1211.csv')),
        orient='columns'
    )

    #derive initial CRM parameter estimates
    gparam_df = gparam_dict_to_df(utils.derive_g(for_plot=True))
    cmat_init = utils.derive_cmatrix(norm=False)
    D_dict_init = utils.derive_Dmatrix_perspecies()
    glist_init = utils.derive_g()
    l_init = pd.DataFrame(0.2, index=cmat_init.index, columns=cmat_init.columns)

    #load simulated annealing fitted CRM parameters
    cmat_fitted, d_dict_fitted, l_fitted, glist_fitted, cfu = utils.load_fitted_params()

    #run monoculture simulations with initial and fitted parameters
    init_sp_mono, init_met_df = mcrm_initialparams_run()
    fit_sp_mono, fit_met_df = run_monoculture()
    
    #save only 0.1 intervals for metabolite df, very large
    def is_multiple_of_point5(x, tol=1e-6):
        return abs((x / 0.5) - round(x / 0.5)) < tol

    init_met_df = init_met_df[init_met_df['time'].apply(is_multiple_of_point5)]
    fit_met_df  = fit_met_df[fit_met_df['time'].apply(is_multiple_of_point5)]
    init_met_df['time'] = init_met_df['time'].round(1)
    fit_met_df['time'] = fit_met_df['time'].round(1)

    #SAVE DATA

    #experimental metabolomics and growth outputs
    metab_class_df.to_csv(args.out / "met_class_df.csv", index=False)
    metab_time_df.to_csv(mono_exp_dir / "met_time_df.csv", index=False)
    metab_dR_df.to_csv(mono_exp_dir / "met_dR_df.csv", index=False)
    od_time_df.to_csv(mono_exp_dir / "od_time_df.csv", index=False)
    growth_df_all_timepoints.to_csv(mono_exp_dir / "growth_df_all_timepoints.csv", index=False)
    growth_df_clean.to_csv(mono_exp_dir / "growth_df_clean.csv", index=False)

    #initial CRM parameters
    gparam_df.to_csv(init_crm_dir / "gparam_df.csv", index=False)
    cmat_init.to_csv(init_crm_dir / "cmat_init.csv")
    pd.concat([df.assign(species=sp) for sp, df in D_dict_init.items()]).to_csv(init_crm_dir / "d_dict_init.csv", index=False)
    l_init.to_csv(init_crm_dir / "l_init.csv")
    glist_init.to_csv(init_crm_dir / "glist_init.csv")

    #final CRM parameters
    cmat_fitted.to_csv(final_crm_dir / "cmat_fitted.csv")
    pd.concat([df.assign(species=sp) for sp, df in d_dict_fitted.items()]).to_csv(final_crm_dir / "d_dict_fitted.csv", index=False)
    l_fitted.to_csv(final_crm_dir / "l_fitted.csv")
    glist_fitted.to_csv(final_crm_dir / "glist_fitted.csv")

    #monoculture simulations
    init_sp_mono.to_csv(mono_sim_dir / "init_mono_sp_df.csv")
    fit_sp_mono.to_csv(mono_sim_dir / "fit_mono_sp_df.csv")
    init_met_df.to_csv(mono_sim_dir / "init_mono_met_df.csv", index=False)
    fit_met_df.to_csv(mono_sim_dir / "fit_mono_met_df.csv", index=False)

