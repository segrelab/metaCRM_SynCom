import pandas as pd
import numpy as np
import warnings
import glob
import os
import pickle
import re
import scipy.stats
import statistics
import glob
# Used for some old versions of functions.
import csv
from scipy.integrate import odeint

#DATA PATHS AND GLOBAL VARIABLES
data_dir = '/projectnb/cometsfba/Archived_misc/mCAFEs/Jings_data_backup_06112024'
#od_data_path = data_dir+'/Growth/od_timepoints_1211.csv'
od_data_path = data_dir+'/Exometabolomics/od_timepoints.csv'
met_conc_path = data_dir+'/Exometabolomics/met_concentration.csv'
met_classes_path = data_dir + '/Exometabolomics/met_category.csv'
met_data_path = data_dir + "/Exometabolomics/*_processed.txt"
timepoints_path = data_dir + '/Exometabolomics/timepoints.csv'
cfu_conv_path = data_dir + '/Growth/cfu.csv'
sim_anneal_model_name = 'model3.5_0209'
otu_table = data_dir + '/16S/co-culture/OTU-table.editied.csv'
model_dir = '/projectnb/cometsfba/Archived_misc/mCAFEs/Jing_syncom_project_Nov2023'
growth_files = glob.glob(data_dir + '/Growth' + '/NLDM*')

sps_to_name = {
  '1319': 'Lysobacter',
  '1320': 'Burkholderia',
  '1321': 'Variovarax',
  '1323': 'Chitinophaga',
  '1324': 'Niastella',
  '1325': 'Mucilaginibacter',
  '1327': 'Rhizobium',
  '1329': 'Bosea',
  '1330': 'Methylobacterium',
  '1331': 'Arthrobacter',
  '1334': 'Rhodococcus',
  '1337': 'Brevibacillus',
  '1338': 'Paenibacillus',
  '1336': 'Marmoricola',
  '1538': 'Bacillus'
}
sps = list(sps_to_name.keys())
sps_names = list(sps_to_name.values())
#Samples that were removed from the co-culture OTU table due to contamination
removed_samples=['187', '200', '243', '244', '245', '246', '91', '94', '96', '119', '128', '131']
tps = ['1', '2', '3', '4']




################################################
####  Functions for Loading Processed Data  ####
################################################

def load_otu_data(path=otu_table):
    """
    Load processed OTU data into a dictionary.

    Args: 
        str: path to processed OTU data. If no path is given, the default OTU data 
        path is used (preferred)

    Returns: 
        dict: A dictionary of dictionaries with samples, species by species tuples,
        od for experiment, and OTUs.
    
    Example: 
    {
      'OTU_1': {
        'sp': (12, 23),
        'od': 0.45,
        1: 3,
        2: 0,
        3: 1,
        4: 2
      }, ...
    }
    """
    
    # Load OTU file #
    OTU_dict = {}
    # Tokenize lines in csv
    lines = [open(path, 'r').read().strip("\n")][0].split('\n')
    for line in lines[1:]:
        tokens = line.split(',')
        # Sample id as a label
        s_id = tokens[0]
        
        # Tuple of species ids
        sp1, sp2 = int(tokens[1][1:]), int(tokens[2][1:])
        
        # Total OD of a given sample
        od = float(tokens[3])
        sample_data = {'sp': (sp1,sp2), 'od': od}
        
        # Create list of OTU data
        for ind, token in enumerate(tokens[4:], 1):
            sample_data[ind] = int(token)
            
        # Add to final dictionary
        OTU_dict[s_id] = sample_data
    return OTU_dict

def load_leaveout_table(infile: str) -> pd.DataFrame:
  """
  Loads a leaveout OTU experiment from a file, annotating with species labels
  
  :param infile: File containing OTU experiments to load
  :type infile: str
  :return: A dataframe with a multiseries index containing the leaveout species, where each row is one experiment
  :rtype: DataFrame
  """
  otu_df = pd.read_csv(infile, sep="\t", index_col=0)
  col_to_sp = {c: get_species_name(c.split("-")[1]) for c in otu_df.columns if re.match(r"[0-9]+-[0-9]+", c)}
  num_to_sp = {int(c.split("-")[0]): sp for c, sp in col_to_sp.items()}
  otu_df['leaveout'] = [(min(num_to_sp[i], num_to_sp[j]), max(num_to_sp[i], num_to_sp[j])) for _, (i, j) in otu_df[['-1', '-2']].iterrows()]
  otu_df.drop(columns = ["-1", "-2"], inplace=True)
  if "Total" not in otu_df:
    otu_df["Total"] = [sum(vals) for _, vals in otu_df.drop(columns=['leaveout']).iterrows()]
  otu_df.rename(columns = col_to_sp, inplace=True)
  return otu_df

def load_passage_table(infile: str) -> pd.DataFrame:
  """
  Loads a passaged OTU experiment from a file, annotating with species labels
  
  :param infile: File containing OTU experiments to load
  :type infile: str
  :return: A dataframe with a multiseries index containing the passage number and replicate, where each row is one measurement
  :rtype: DataFrame
  """
  otu_df = pd.read_csv(infile, sep="\t", index_col=0)
  otu_df.rename(columns = {"-1": "P", "-2": "R"}, inplace=True)
  if "Total" not in otu_df:
    otu_df["Total"] = [sum(vals) for _, vals in otu_df.drop(columns=["P", "R"]).iterrows()]
  col_to_sp = {c: get_species_name(c.split("-")[1]) for c in otu_df.columns if re.match(r"[0-9]+-[0-9]+", c)}
  otu_df.rename(columns = col_to_sp, inplace=True)
  return otu_df

def load_od_data(path=od_data_path):
    """
    Load processed OD data into a dictionary.

    Args: 
        str: path to processed OD data. If no path is given, the default OD data path is 
        used (preferred)

    Returns: 
        dict: structured {'sp_id':{'T1':OD, ...}...} Empty strings are assigned to 
        timepoints without OD measurements.
    """

    #provide user a warning if default OD data is not used
    if path != od_data_path:
        warnings.warn("USING DIFFERENT DATA PATH THAN DEFAULT. Providing function with no defined data path will result in the use of the default path (preferred).", UserWarning)
    
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df = df.applymap(lambda x: '' if pd.isna(x) else x)
    return df.to_dict(orient='index')

def load_met_conc(path=met_conc_path, old=False):
    """
    Load Metabolite Concentration data into a df. Spermidine excluded. uM of metabolite x
    g/mol (molecular weight of metab) x 10^-9 for unit conversion. Resulting units are 
    concentration in g/mL
    
    Args: 
        str: path to metabolite data. If no path is given, the default path is used (preferred)
        old: Uses an old version of the metabolite loading code for backwards compat-
        ability with some functions.

    Returns: 
        pandas df: Metabolite names as index, Concentration(g/mL) as first column. Spermidine excluded    
    """
    if path != met_conc_path:
        warnings.warn("USING DIFFERENT DATA PATH THAN DEFAULT. Providing function with no defined data path will result in the use of the default path (preferred).", UserWarning)
        
    if old == True:
        conc, mw = {}, {}
        data_reader = csv.reader(open(path, 'r'), delimiter='\t')
        # Ignore header
        header = next(data_reader)
        # Process each line
        for row in data_reader:
            d = row[0].split(',')
            if len(row) == len(header) and d[0]!='':
                conc[d[1]] = d[2]
                mw[d[1]]   = d[3]

        mets = list(conc.keys())
        num_resources = len(mets)-1
        alpha = np.zeros (num_resources)
        for m_idx, met in enumerate(mets[:-1]):
            alpha[m_idx] = float(conc[met]) * float(mw[met]) * 10**(-9)

        return alpha
    else:
        df = pd.read_csv(path)
        #multiply metabolite concentration by molecular weight and converting to get units g/mL
        df['Concentration (g/mL)'] = df['Concentration (uM)'].astype(float)* df['MW (g/mol)'].astype(float) * 10**(-9)
        conc_df = df[['Metabolite', 'Concentration (g/mL)']].set_index('Metabolite', drop=True)

        #exclude spermidine
        conc_df.drop('spermidine', inplace=True, axis=0)

        return conc_df

def load_met_classes(path=met_classes_path, exclude_spermidine=True):
    """
    Load Metabolite Class data from csv file
    
    Args: 
        path (str): path to metabolite data. If no path is given, the default path is used (preferred)
        exclude_spermidine (bool): include spermidine in the metabolite class dict or not (default=True)

    Returns: 
        dict: {'class1(str)':[metab1(str), metab2(str), ...]...} 
    """
    if path != met_classes_path:
        warnings.warn("USING DIFFERENT DATA PATH THAN DEFAULT. Providing function with no defined data path will result in the use of the default path (preferred).", UserWarning)

    
    met_class_df = pd.read_csv(path)
    if exclude_spermidine:
        met_class_df.replace('spermidine', np.nan, inplace=True)
    return {col: [val for val in met_class_df[col].values if not pd.isna(val)] for col in met_class_df.columns}

def load_processed_data (infile):
    """
    Load processed exometabolomics data for a given species.
    Output: Dictionary with metabolite data. [met, timepoint, rep] = {value}.
    """
    lines = [open(infile, 'r').read().strip("\n")][0].split('\n')
    header = lines[0].split('\t')
    data = {}
    for line in lines[1:]:
        tokens = line.split('\t')
        met = tokens[0]
        if met not in data: 
            data[met] = {}
        data[met][tokens[1]] = {}
        # For each replicate
        for idx, token in enumerate (tokens[2:], 2):
            data[met][tokens[1]][header[idx]] = float(token) 
    return data 

def load_all_metdata(sps=sps, path=met_data_path): 
    """
    Loads met data for all species into a dictionary from processed files.
    Assumes a standard filename for the unprocessed and processed data.

    Args: 
        sps (list of str): list of species to load metabolite data for
        path (str): path to processed exometabolomics directory. If no path is given, the default path is used (preferred)

    Returns: 
        dict: species code as keys
    """

    if path != met_data_path:
        warnings.warn("USING DIFFERENT DATA PATH THAN DEFAULT. Providing function with no defined data path will result in the use of the default path (preferred).", UserWarning)
    
    metDict = {}
    files = [f for f in glob.glob(path) if os.path.isfile(f)]
    for sp in sps:
        for f in files:
            if os.path.basename(f).split("_")[0] == sp:
                metDict[sp] = load_processed_data(f)

    return metDict

def load_timepoints(path=timepoints_path):
    """
    Load the timepoints(hours) that correspond to T1, T2, T3, T4 for each species exometabolomics data.
    """
    return pd.read_csv(timepoints_path, index_col=0, keep_default_na=False, dtype='string')

def load_cfu_conversion(path=cfu_conv_path, sps=sps):
    """
    Load the species-specific conversions between OD and CFU/ml.

    Args:
        path (str): Path to processed CFU file. If no path is given, the default path is used (preferred).

    Returns:
        dict: {sp code: {cfu/mL: conversion value}}
    """
    df = pd.read_csv(path, dtype={'Species': 'str'})
    cfu_dict = {}
    cfu_dict = df.set_index('Species').to_dict(orient='index')
    cfu_sub_dict = {k: cfu_dict[k] for k in sps}
    
    return cfu_sub_dict

def load_fitted_params(sps=sps, model_name=sim_anneal_model_name, path=model_dir, old=False):
    """
    Load fitted parameters after running simulated annealing step for given species.

    Args:
        sps: list of species 4 digit codes. (default: all 15)
        model_name: identifier for model version to load (default: model3.5_0209)
        path: path to directory with identified version of model paramters to load
        old: Return either an old version of the matrix ()

    Returns:
        Cmatrix: pandas DataFrame of consumer matrix with species as rows and metabolites as columns. 
                Matrix is unnormalized and quantifies the comsumption rate of each metabolite by each species
        D_dict: Dictionary with species codes as keys. Each species has its own pandas DataFrame that encodes the 
                resource transformations for that species, which resources turn into other resources upon consumption.
        lmatrix: Species x Metabolite pandas DataFrame that assigns the leakage parameter of each metabolite for each 
                species. The leakage parameter is consistent across all metabolites but different between species. 
        glist: pandas DataFrame that specifies the growth parameter for each species.
        cfu: pandas DataFrame that specifies the od to cfu conversion parameter for each species.
        
    """
    if model_name != sim_anneal_model_name or path != model_dir:
        warnings.warn(
            "USING DIFFERENT MODEL PARAMETERS THAN DEFAULT. Providing function with no defined model name will result in the use of the default model (preferred).",
            UserWarning,
        )
    Cmatrix = np.zeros(shape=(len(sps), 64))
    D_dict = {} 
    D_dict_pd = {}
    glist = np.zeros(shape=(len(sps), 1))
    lmatrix = np.zeros(shape=(Cmatrix.shape[0], Cmatrix.shape[1]))
    cfu = np.zeros(shape=(len(sps), 1))

    #get metabolite list for column labels
    metData = load_all_metdata()
    mets = list(metData['1319'].keys())[:-1]

    for sp_i, sp in enumerate(sps):
        run_files = []
        params = pickle.load(
            open(path + '/OUTPUTS/' + model_name + '/model3_sa_params_sp' + sp + '_run0.p', 'rb')
        )
        C = params['C']
        Cmatrix[sp_i] = C
        D_dict[sp_i] = params['D'][0]
        D_dict_pd[sp] = pd.DataFrame(params['D'][0], index=mets, columns=mets)
        lmatrix[sp_i] = params['l']
        glist[sp_i] = params['g']
        cfu[sp_i] = params['cfu']

    if old == True:
        return Cmatrix, D_dict, lmatrix, glist, cfu
        
    else:
        Cmatrix = pd.DataFrame(Cmatrix, index=sps, columns=mets)
        lmatrix = pd.DataFrame(lmatrix, index=sps, columns=mets)
        glist = pd.DataFrame(glist, index=sps)
        cfu = pd.DataFrame(cfu, index=sps)

        return Cmatrix, D_dict_pd, lmatrix, glist, cfu

#############################
####  Utility Functions  ####
#############################

def get_species_list(sps):
    return sps

def get_timepoints(tps):
    return tps

def get_model_name(sim_anneal_model_name):
    return sim_anneal_model_name

def get_species_name(species_number:  str):
    """
    Given a species number, return the corresponding species name.
    """

    # Return the species name if it's in the dictionary, else return None
    return sps_to_name.get(str(species_number), None)

def get_species_colormap(name_key=True):
    if name_key:
        sp_color_palette = {'Arthrobacter': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                             'Bacillus': (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
                             'Bosea': (1.0, 0.4980392156862745, 0.054901960784313725),
                             'Brevibacillus': (1.0, 0.7333333333333333, 0.47058823529411764),
                             'Burkholderia': (0.17254901960784313,0.6274509803921569,0.17254901960784313),
                             'Chitinophaga': (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
                             'Lysobacter': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
                             'Marmoricola': (1.0, 0.596078431372549, 0.5882352941176471),
                             'Methylobacterium': (0.5803921568627451,0.403921568627451, 0.7411764705882353),
                             'Mucilaginibacter': (0.7725490196078432,0.6901960784313725,0.8352941176470589),
                             'Niastella': (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
                             'Paenibacillus': (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
                             'Rhizobium': (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
                             'Rhodococcus': (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
                             'Variovarax': (0.4980392156862745, 0.4980392156862745, 0.4980392156862745)}
    else:
        sp_color_palette = {'1331': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                             '1538': (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
                             '1329': (1.0, 0.4980392156862745, 0.054901960784313725),
                             '1337': (1.0, 0.7333333333333333, 0.47058823529411764),
                             '1320': (0.17254901960784313,0.6274509803921569,0.17254901960784313),
                             '1323': (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
                             '1319': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
                             '1336': (1.0, 0.596078431372549, 0.5882352941176471),
                             '1330': (0.5803921568627451,0.403921568627451, 0.7411764705882353),
                             '1325': (0.7725490196078432,0.6901960784313725,0.8352941176470589),
                             '1324': (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
                             '1338': (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
                             '1327': (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
                             '1334': (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
                             '1321': (0.4980392156862745, 0.4980392156862745, 0.4980392156862745)}
    return sp_color_palette

def calc_interval(l): 
	""" 
	for a list of numbers, calculate the interval between two numbers 
	"""
	intervals = []
	for idx, el in enumerate(l):
		if idx == 0:
			intervals.append (el)
		else: 
			intervals.append (el - l[idx-1])
	return intervals

def get_timestep_resource_usage(sps=sps, tps=tps, metData=load_all_metdata()):
	"""
	C matrix at each timepoint is determined based on the usage of each resource over dt, and averaged across all timepoints
	"""
	mets = list(metData['1319'].keys())
	resource_usage_dict = {}
	Tintervals_dict = {}
	for i in range(len(sps)):
		metdata = metData[sps[i]]
		times = [metdata['sucrose'][t]['Time'] for t in ['1', '2', '3', '4'] if t in metdata['sucrose']]
		tintervals = calc_interval(times)
		Tintervals_dict[sps[i]] = tintervals
		resource_usage_dict[sps[i]] = {}
		for t in tps:
			if t in metdata['sucrose']:
				resource_usage_dict[sps[i]][t] = []
				for j, met in enumerate(mets[:-1]):
					ratio = np.median([metdata[met][t][R] for R in ['R1', 'R2', 'R3', 'R4'] if R in metdata[met][t].keys()])
					used_frac = ratio - 1   # if negative, met is being produced
					resource_usage_dict[sps[i]][t].append(used_frac)
	return resource_usage_dict, Tintervals_dict
    
def calc_od(OTU_dict, excl=removed_samples, use_norm_ratio=False):
    """
    Create a dictionary that maps the od of each species in absence or presence of another species.
    
    Args:
        dictionary: Dictionary of OTUs with included od for coculture.
        str list: List of sample ids that are contaminated.
        
    Returns:
        dictionary: 2d dictionary with same species od and od with other species.
    
    example:
    data = {
    data['sp1']['sp1'] = ...
    data['sp1']['sp2'] = ..., ... (Equal to num rep)
    }
    """
    
    od_dict = {}
    samples = OTU_dict.keys() - excl
    for sp in range(1, 18):
        od_dict[sp] = dict()
        for sample in samples:
            data = OTU_dict[sample]
            
            # od in absence of other species
            if data['sp'] == (sp, sp):
                od_dict[sp][sp] = data['od']
            
            # od in presence of other species
            elif data['sp'][0] == sp or data['sp'][1] == sp:
                sp2 = [i for i in data['sp'] if i != sp][0]
                if use_norm_ratio:
                    ratio = norm_ratio ([data[sp], data[sp2]])
                od = data['od'] * (data[sp] / (data[sp] + data[sp2]))
                # adjusted fraction based on extraction efficiency
                # od = data['od'] * ratio[0]   
                if sp2 not in od_dict[sp]:
                    od_dict[sp][sp2] = [od]
                else:
                    od_dict[sp][sp2].append(od)
    return od_dict
                    
def calc_ratios(OTU_dict, excl=removed_samples, use_norm_ratio=False):
    """
    Create a dictionary that maps the ratio of each species in absence 
    or presence of another species
    
    """ 
    ratio_dict = {}
    samples = OTU_dict.keys() - excl

    for sp in range(1, 16):
        ratio_dict [sp] = dict()
        for sample in samples:
            data = OTU_dict[sample]
            # od in absence of other species
            if data['sp'] == (sp, sp):
                ratio_dict[sp][sp] = 1
            # od in presence of other species
            elif data['sp'][0] == sp or data['sp'][1] == sp:
                sp2 = [i for i in data['sp'] if i != sp][0]
                ratio = data[sp] / (data[sp] + data[sp2])
                if use_norm_ratio:
                    ratio = norm_ratio ([data[sp], data[sp2]])[0]
                if sp2 not in ratio_dict[sp]:
                    ratio_dict[sp][sp2] = [ratio]
                else:
                    ratio_dict[sp][sp2].append(ratio)
    return ratio_dict

def norm_ratio (sp_frac):
    """
    normalize species abundance according to cell lysis efficiency (P0 ratio)
    sp_frac: fraction of species in a list
    """
    ab =[0.059732959, 0.047083626, 0.035839775, 0.26598735, 0.084680253, 0.168306395, 0.125790583, 0.002459592, 0.001756852, 0.008784259, 0.05130007, 1/15, 0.00878426, 1/15, 0.137385805]
    adj_ab = [sp_frac[i]/(len(sp_frac)*ab[i]) for i in range(len(sp_frac))]
    norm_ab = [el/sum(adj_ab) for el in adj_ab]
    return norm_ab
    
###################################################
####  Parameter Inference and CRM Derivations  ####
###################################################

def derive_cmatrix(sps=sps, metData=load_all_metdata(), ODs=load_od_data(), alpha_df=load_met_conc(), cfu_conv=load_cfu_conversion(), norm=False):
    '''
    DERIVE C-MATRIX PARAMETERS ANALYTICALLY

    Args:
        sps: list of species IDs (default: all 15 strains)
        metData: dict of metabolite concentrations at different timepoints. Species ID's are keys. 
                    In the form of output from load_all_metdata() (default: all 15 strains) 
        ODs: dict of OD measurements from strain monoculture experiments. Species ID's are keys.
                In the form of output from load_od_data() (default: all 15 strains)
        alpha: metabolite concentrations in blank NLDM media. output of load_met_conc() 
                (default: excludes sperimidine)
        cfu_conv: OD-to-cfu conversion to be used. Dict with species codes as keys, values are conversion values (default: species-specific values)
        norm (bool): If True, returns a normalized version of the C-matrix such that each strains 
                    consumption of metabolites sums to 1 (default: True)

    Returns:
        Cmatrix: Pandas data frame with species as rows and metabolites as columns. Values represent resource uptake parameter for each species-resource pair. Equations used to calculate this can be found in the Supplement (Eq S6)

    
    '''

    # Define time points, metabolites and metabolite concentrations in media
    tps = ['1', '2', '3', '4']
    mets = list(metData['1319'].keys())
    alpha = alpha_df.values.tolist()
    alpha = [a for sp in alpha for a in sp]

    # Initialize the C matrix with zeros
    numSpecies = len(sps)
    numResources = len(mets) - 1  # Excluding spermidine
    Cmatrix = np.zeros(shape=(numSpecies, numResources))

    # Iterate over each species
    for i in range(len(sps)):
        
        sp = sps[i]
        metdata = metData[sp]

        # Extract times for each metabolite and calculate time intervals
        times = [metdata['sucrose'][t]['Time'] for t in tps if t in metdata['sucrose']]
        tintervals = calc_interval(times)

        # Conversion factor from OD to CFU
        od_to_cfu = cfu_conv[sp]['cfu/ml']

        # Iterate over each metabolite
        for j, met in enumerate(mets[:-1]):
            x, y = [0], [0]
            t_idx = 0

            # Iterate over each time point
            for t, tp in enumerate(tps):
                if tp in metdata[met]:
                    # Calculate the median ratio for the metabolite across replicates
                    met_ratio = np.median([metdata[met][tp][R] for R in ['R1', 'R2', 'R3', 'R4'] if R in metdata[met][tp].keys()])

                    R = met_ratio * alpha[j]

                    # Check for non-NaN values
                    if not np.isnan(R):
                        N = float(ODs[sp]['T'+tps[t]]) * od_to_cfu
                        # Calculate R0 for the previous time point, median across replicates or initial value
                        if t != 0 and tps[t-1] in metdata[met]:
                            R0 = np.median([metdata[met][tps[t-1]][R] for R in ['R1', 'R2', 'R3', 'R4'] if R in metdata[met][tps[t-1]].keys()]) * alpha[j]
                        else:
                            R0 = alpha[j]

                        # For each time interval, calculate dR/Rdt, and N -- slope of this is Cia
                        if not np.isnan(R0):
                            dt = tintervals[t_idx] * 60
                            t_idx += 1
                            x.append(N)
                            y.append((R - R0) / (((R + R0) / 2) * dt))

            # Calculate the derivative of y with respect to x
            dydx = np.diff(y) / np.diff(x)
            neg_val = [el for el in dydx if el < 0]
    
            # Update the C matrix with maximum Cia across all time intervals
            if neg_val != []:
                Cmatrix[i][j] = -10 * min(neg_val)
            else:
                Cmatrix[i][j] = 0

    # Normalize the C matrix 
    norm_matrix = Cmatrix
    if norm:
        sum_of_rows = Cmatrix.sum(axis=1)
        norm_matrix = Cmatrix / sum_of_rows[:, np.newaxis]
    norm_matrix = pd.DataFrame(norm_matrix, index=sps, columns = mets[:-1])
    
    return norm_matrix

def derive_g (sps=sps, ODs=load_od_data(), alpha_df=load_met_conc(), metData=load_all_metdata(), Cmatrix=derive_cmatrix(norm=False), for_plot=False):
    '''
    DERIVE GROWTH(g) PARAMETERS ANALYTICALLY

    Args:
        sps: list of species IDs (default: all 15 strains)
        metData: dict of metabolite concentrations at different timepoints. Species ID's are keys. 
                    In the form of output from load_all_metdata() (default: all 15 strains) 
        ODs: dict of OD measurements from strain monoculture experiments. Species ID's are keys.
                In the form of output from load_od_data() (default: all 15 strains)
        alpha: metabolite concentrations in blank NLDM media. output of load_met_conc() 
                (default: excludes sperimidine)
        Cmatrix: Pandas dataframe with species as rows and metabolites as columns. Values represent 
                    resource uptake parameter for each species-resource pair. Output from derive_cmatrix. 
                    Equations used to calculate this can be found in the Supplement (Eq S6)

    Returns:
        g_list: Pandas series with species as index and growth parameter as values. 
                Equations used to calculate this can be found in the Supplement (Eq S7).

    if for_plot=True, more than just the g_list is returned to be used to plot supplementary figure 3
    '''

    # Conversion factor from OD to CFU
    od_to_cfu = 10**9	
    
    # Energy content of each resource, constant
    w = 10**12
    
    tps = ['1', '2', '3', '4']
    mets = list(metData['1319'].keys())[:-1]
    alpha = alpha_df.values.tolist()
    alpha = [a for sp in alpha for a in sp]

	# norm C matrix
    Cmatrix = np.array(Cmatrix)
    glist = []
    g_param_dict = {}

    for i in range(len(sps)):
        sp = sps[i]
        metdata = metData[sp]
        times = [metdata['sucrose'][t]['Time'] for t in ['1', '2', '3', '4'] if t in metdata['sucrose']]
        tintervals = calc_interval (times)
        x, y = [0], [0]
        t_idx = 0
        for t, tp in enumerate(tps):
            if ODs[sp]['T'+tps[t]] != '':
                sum_met = 0
                N = float(ODs[sp]['T'+tps[t]]) * od_to_cfu
                dt = tintervals[t_idx]
                if t == 0:
                    N0 = 0
                else:
                    if ODs[sp]['T'+tps[t-1]] != '':
                        N0 = float(ODs[sp]['T'+tps[t-1]]) * od_to_cfu
                    else:
                        N0 = 0
                for j, met in enumerate(mets):
                    if tp in metdata[met]:
                        R = np.median([metdata[met][tp][R] for R in ['R1', 'R2', 'R3', 'R4'] if R in metdata[met][tp].keys()]) * alpha[j]
                        if not np.isnan(R):   # if R is not null
                            sum_met += w * Cmatrix[i][j] * R

                y.append ( (N-N0)/(((N+N0)/2)*dt) )
                x.append ( sum_met )

		# derive slope
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        glist.append(slope)
        yfit = [intercept + slope * xi for xi in x]

        #save x, y, yfit, and r_value into dict for plotting
        g_param_dict[sp] = {'x': x, 'y': y, 'yfit': yfit, 'g':slope, 'r_value':r_value}

    g_list = pd.Series(glist, index=sps)

    #return more than just the g_list for plotting supplementary figure
    if for_plot:
        return g_param_dict

    return g_list

def derive_Dmatrix_perspecies(sps=sps, Cmatrix=derive_cmatrix(), alphas_df=load_met_conc(), norm=False):
    '''
    Provided the derived consumer matrix and data about production/consumption of metabolites in 
    monoculture experiments, derive the resource transformation matrix for each species independently. 
    Matrix describes what metabolites get turned into and leaked as upon consumption.

    Args: 
        sps: list of species codes (default: all 15)
        Cmatrix: non-normalized consumer matrix (default: output from derive_cmatrix())
        alphas_df: pandas DataFrame that describes the metabolite concentrations at time zero in NLDM 
        norm (bool): If true, the D matrices will be normalized such that each column sums to 1 (default: False)

    Returns:
        D_dict: dictionary with species codes as keys, and values are a pandas DataFrame encoding the 
        resource transformation matrix (Dmatrix) for that species
    
    '''
    R_dict, Tintervals = get_timestep_resource_usage()  # get the use of each resource by each species at 4 different timepoints
    D_dict = {}
    alphas = alphas_df.values.tolist()
    alphas = [a for sp in alphas for a in sp]
    mets=Cmatrix.columns.to_list()
    Cmatrix = np.array(Cmatrix)
    
    for i, sp in enumerate(sps):
        Cmatrix_i = Cmatrix[[i],:]

        numResources = Cmatrix_i.shape[1]   # number of columns in C matrix

        Dmatrix = np.zeros (shape=(numResources, numResources))#for each species, make a unique D matrix
        Dt_dict = {}  # a dictionary for D matrix at 4 time intervals
	    # get list of metabolites being produced at any timepoints
        for t in R_dict[sp]:
            D_t = np.zeros (shape=(numResources, numResources))
            produced_idx = [idx for idx in range(len(R_dict[sp][t])) if R_dict[sp][t][idx] <0] # met being produced at this time interval
            consumed_idx = [idx for idx in range(len(R_dict[sp][t])) if R_dict[sp][t][idx] >0] # met being consumed at this time interval
            for idx in produced_idx:
                produced_amt = R_dict[sp][t][idx] * alphas[idx]
                for alpha in consumed_idx:
                    consumed_sum = sum([R_dict[sp][t][i] * alphas[i] for i in consumed_idx])
                    D_t[idx][alpha] = produced_amt * (R_dict[sp][t][alpha] * alphas[alpha]/consumed_sum)    
                    # partition the production according to the consumption level of different mets
		    # normalize distribution for each species
            D_t = D_t/D_t.sum(0)
            D_t = np.nan_to_num(D_t, copy=True, nan=0.0)
            Dt_dict[t] = D_t
            
	    # average four timepoints
        for i in range(Dmatrix.shape[0]):
            for j in range(Dmatrix.shape[1]):
                Dmatrix[i][j] = statistics.median ([Dt_dict[t][i][j] for t in Dt_dict])
                if ( Dmatrix[i][j] > 0 ) == False:
                    Dmatrix[i][j] = 0


        if norm:
            # normalize distribution for each species
            Dmatrix = Dmatrix / Dmatrix.sum(0)
            Dmatrix = np.nan_to_num(Dmatrix, copy=True, nan=0.0)

        D_dict[sp] = pd.DataFrame(Dmatrix, columns=mets, index=mets)
        
    return D_dict

#################################################
####  Functions for Running CRM Simulations  ####
#################################################


def mcrm_params(qual, x0, Cmatrix, D_dict, leakage, mu, cfu_c, w_alpha=10**8, cfu_conv=load_cfu_conversion(), m=0, k=0.04, time=48, step=0.01, old=False):
    """ 
    Construct the param structure for the MCRM ODE solver.
    Args: 
        qual: Type of resource qualities. String, "eye" or "ranked". Recommend using 'eye'
        
        x0: Initial availability of resources. Concatenated DataFrame of species initial 
            conditions with metabolite initial conditions to follow.
            
        Cmatrix: Consumer matrix of form output by derive_cmatrix()
        
        D_dict: Dictionary of species-specific resource transformation matrices. In form output by derive_Dmatrix_perspecies()
        
        leakage: pandas DataFrame (species x metabolites) with values representing the leakage parameter.
        
        mu: pandas DataFrame with species-specific growth parameters
        
        cfu_c : From param import. -- UNCLEAR IF THIS IS BEING USED
        
        w_alpha: Energy conversion rate, uniform for all metabolites/species
        
        cfu_conv: conversion factor from OD to CFU/mL -- UNCLEAR IF THIS IS BEING USED
        
        m: maintainance value (default = 0).
        
        k: velocity for resource uptake (default = 0.4)
        
        time: Number of hours to run simulation (default = 48)
        
        step: The step size for the simulation (default = 0.01, ~ 36 sec)
    
    Returns:
        params: Dictionary of parameters required to run the CRM simulation. 
                Form that can be passed to run_mcrm() function
    
    """
    num_species = Cmatrix.shape[0]

    num_resources = Cmatrix.shape[1]


    # set up variable indexes for ODE variable
    if old == True:
        var_idx = dict()
        var_idx['species'] = range(num_species)
        var_idx['resources'] = range(num_species, num_resources+num_species)
    else:
        var_idx = dict()
        var_idx['species'] = np.arange(num_species)
        var_idx['resources'] = np.arange(num_species, num_resources+num_species)
        params = {}

    # define each resource to be of higher quality than the next if qual is 'ranked'
    # i.e. for R1, R2, ..., Rn ==> w1 > w2 > ... > wn
    # if qual is 'eye', W is an identity matrix
    if qual == 'ranked': 
        w = range(1, num_resources+1)[::-1]
        w = np.diag(w)
    if qual == 'eye': 
        W = np.identity(num_resources)
    W = w_alpha * W

    # mu is g, or Conversion factor from energy uptake to growth rate for organism i
    # This is repetitive but leaving it in for clarity.
    g = mu

    # Conversion factor from species abundance to CFU
    cfu_c = cfu_c

    params = { 'num_species': num_species, 
                'num_resources': num_resources,
                'var_idx': var_idx,
                'C': Cmatrix,
                'D': D_dict,
                'W': W,
                'k': k,
                'g': mu,
                'cfu': cfu_c,
                'od_to_cfu': cfu_conv,
                'l': leakage,              
                'm': m, 
                'x0': x0,
                'time': time,
                'step': step
    }

    return params

def run_mcrm(params, old=False):
    """
    Given a list of parameters, run a consumer resource model simulation.

    Args: 
        params: Look at mcrm_params() function. params variable should be 
                in form of output of this function.
        old: Choose whether to return the old version of the results
                (a single list of both metabolites and population)
                or new (separate, informative dataframes).

    Returns:
        sp_x: population size/species dynamics for each species over 
                all timepoints in simulated experiment.
        met_x: metabolite concentration/resource dynamics for each 
                metabolite over all timepoints in simulated experiment.
    """
    # parse input
    num_species = params['num_species']
    num_resources = params['num_resources']
    var_idx = params['var_idx']
    C = params['C']
    D = params['D']
    W = params['W']
    k = params['k']
    g = params['g']
    l = params['l']
    m = params['m']
    x0 = params['x0']
    time = params['time']
    step = params['step']

    # time interval 
    t = np.arange(0, time, step)

    # Integrate ODE 
    if old == True:
        args = (num_species, num_resources, C, D, W, g, l, k, m, var_idx)
        x = odeint(population_dynamics2_old, x0, t, args=args, mxstep=50)
        return x
        
    else:
        args = (num_species, num_resources, C.values, D, W, g.values, l.values, k, m, var_idx, C.index)
        x = odeint(population_dynamics2, x0['x0'], t, args=args, mxstep=50)
    
        sp_x = pd.DataFrame(x[:,:num_species], index=t, columns=x0.index[:num_species])
        met_x = pd.DataFrame(x[:,num_species:], index=t, columns=x0.index[num_species:])
        
        return sp_x, met_x
        
def population_dynamics2(x, t, num_species, num_resources, C, D, W, g, l, k, m, var_idx, sps):

    '''
    Function used internally by run_mcrm() to perform the simulated experiment defined 
    by params passed to run_mcrm(). Should not typically be called outside of run_mcrm()
    '''
    
    dx = np.zeros(num_species+num_resources)

    negs = x < 0 # find which y values are negative
    x_pos = np.maximum(x, np.zeros(x.shape)) # set them to 0

    # value of species 
    x_s = x_pos[var_idx['species']]

    # value of resources
    x_r = x_pos[var_idx['resources']]

    # when D is unique for individual species
    growth_rate_multiplier = np.zeros(shape=(1, len(var_idx['species'])))
    consumption = np.zeros(shape=(len(var_idx['resources']), 1))
    production  = np.zeros(shape=(len(var_idx['resources']), 1))
    for sp in var_idx['species']:
        x_sp = np.zeros(shape=(len(var_idx['species']), 1))
        x_sp[sp][0] = x_s[sp]
        
        growth_rate_multiplier_sp = (1-l[sp][0])*np.dot(np.dot(C, W), x_r.T/(x_r.T + k)) - m   # with cross-feeding

        growth_rate_multiplier[0][sp] = growth_rate_multiplier_sp[sp]
        
        consumption_sp = np.dot(np.dot(C, np.diag(x_r/(x_r + k))).T, x_sp)

        consumption += consumption_sp

        production += np.dot(D[sps[sp]], l[sp][0]*consumption_sp)

    dx[var_idx['species']] = x_s*g.flatten()*growth_rate_multiplier
    dx[var_idx['resources']] = production.T[0] - consumption.T[0]

    dx[negs] = np.maximum(dx[negs], np.zeros(negs.sum()))  # ensure that none of the negative y values will decrease

    return dx

def population_dynamics2_old (x, t, num_species, num_resources, C, D, W, g, l, k, m, varIdx):
    """
    Old version of the population dynamics function. For backwards compatibility with some functions.
    """
    dx = np.zeros(num_species+num_resources)

    negs = x < 0                             # find which y values are negative
    x_pos = np.maximum(x, np.zeros(x.shape)) # set them to 0

    # value of species
    x_s = x_pos[varIdx['species']]

    # value of resources
    x_r = x_pos[varIdx['resources']]

    # when D is unique for individual species
    growth_rate_multiplier = np.zeros(shape=(1, len(varIdx['species'])))
    consumption = np.zeros(shape=(len(varIdx['resources']), 1))
    production  = np.zeros(shape=(len(varIdx['resources']), 1))
    for sp in varIdx['species']:

        x_sp = np.zeros(shape=(len(varIdx['species']), 1))
        x_sp[sp][0] = x_s[sp]

        growth_rate_multiplier_sp = (1-l[sp][0])*np.dot(np.dot(C, W), x_r.T/(x_r.T + k)) - m   # with cross-feeding

        growth_rate_multiplier[0][sp] = growth_rate_multiplier_sp[sp]

        consumption_sp = np.dot(np.dot(C, np.diag(x_r/(x_r + k))).T, x_sp)

        consumption += consumption_sp

        production += np.dot(D[sp], l[sp][0]*consumption_sp)

    dx[varIdx['species']] = x_s*g*growth_rate_multiplier
    dx[varIdx['resources']] = production.T[0] - consumption.T[0]

    dx[negs] = np.maximum(dx[negs], np.zeros(negs.sum()))  # ensure that none of the negative y values will decrease

    return dx


def main():
    #Use for testing newly added functions
    return

if __name__ == "__main__":
    main()