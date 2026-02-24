# metaCRM_SynCom
This repository accompanies the manuscript:

**“Metabolic blueprints of monocultures enable prediction and design of synthetic microbial consortia”**  
https://doi.org/10.64898/2026.01.11.698878

It contains all processed data, model parameters, and code required to reproduce the simulations and figures presented in the manuscript.

## Abstract
Synthetic microbial ecology aims at designing communities with desired properties based on mathematical models of individual organisms. It is unclear whether simplified models harbor enough detail to predict the composition of synthetic communities in metabolically complex environments. Here, we use longitudinal exometabolite data of monocultures for 15 rhizosphere bacteria to parametrize a consumer-resource model, which we use to predict pairwise co-cultures and higher order communities. The capacity to artificially “switch off” cross-feeding interactions in the model demonstrates their importance in ecosystem structure. Leave-one-out and leave-two-out experiments display that pairwise co-cultures do not necessarily capture inter-species interactions within larger communities and broadly highlight the nonlinearity of interactions. Finally, we illustrate that our model can be used to identify new sub-communities of three strains with high likelihood of coexistence. Our results establish hybrid mechanistic and data-driven metabolic models as a promising and extendable framework for predicting and engineering microbial communities.

### Repo Structure
- `data/`: Processed data ready for plotting, generated according to `figureX/process_data.py`.
- `utils.py`: Utility Python constants, filepaths, and functions used across data processing and plotting scripts.
- `figureX/`: Each figure has its own folder containing two files:
  - `plot_figures.py`: A script that generates plots associated with the given figure and saves them to `figures/`, including any supplemental figures related to this figure.
  - `process_data.py`: A script that takes the raw data as input to perform processing and analysis and outputs files in the correct format for `plot_figures.py`.
- `plot_all.sh`: A Bash script that runs all `figureX/plot_figures.py` scripts to generate all figures and place them in `figures/`.
- `environment.yml`: A Conda environment file containing software versions used for data processing and figure generation.

## Conda Environment
All analyses were performed within a specified Conda environment to ensure computational reproducibility. All simulations, parameter fitting, data processing, and figure generation were executed within this environment. The complete dependency specification, including exact package versions, is provided in `environment.yml`.

To recreate the environment:
```bash
conda env create -f environment.yml
conda activate metaCRM_SynCom
```

## Data
Raw exometabolomics and 16S data is in the process of being deposited in formal data repositories, but here there is all of the clean and processed data needed for recreating the simulations and figures in the manuscript.
All processed data is within the data directory:
- `exp_whole_comm/`: OTU tables across replicates and passages for the 2 experiments with all 15 syncom members
- `final_crm_params/`: Fitted C matrix, D marices, growth and leakage parameters after our two-step parameter fitting procedure. These are the model parameters used for a majority of the simulated experiments.
- `init_crm_params/`: Initial parameters estimates yielded from analytical solutions from CRM model equations and timecourse exometabolomics data.
- `monoculture_exp/`: Experimental measurements of OD and exometabolomics across timepoints for monocultures of 15 strains
- `monoculture_sim/`: Simulated monoculture growth with out fitted CRM (both initial and final parameters are tested)
- `sim_loo/`: Simulated leave-one-out community composition over time
- `sim_whole_comm/`: Simulated whole community experiments with the 15 member syncom
- `coculture_interactions.csv`, `loo_interactions.csv`, `epistasis_vals.csv` : Interaction metrics derived from experimental measurements
- `exp_loo_df.csv`: Experimental leave-one-out community compositions
- `experimental_3sp_otu.csv`, `simulated_3sp_communities.csv` : experimental and simulated composition and shannon entropy for the designed subcommunities (Figure 6)

## Data Processing
Each figure subdirectory contains a `process_data.py` script that documents how raw experimental data were transformed into the processed datasets included in this repository, and show how our CRM simulations are implemented.

## Figure Generation
All figures can be reproduced if you run:
```bash
bash plot_all.sh <output_directory>
```
Each individual figure panel and associated supplementary figures can also be generated from running the corresponding `plot_figure.py` within each figure directory.
