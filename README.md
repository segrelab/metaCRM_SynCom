# metaCRM_SynCom
Metabolomics-informed consumer-resource modeling of a 15-member synthetic microbial community.

### Project Structure
- `data/`: A folder containing the processed data ready for plotting, generated according to `figureX/process_data.py`.
- `utils.py`: Utility Python constants, filepaths, and functions used across data processing and plotting scripts.
- `figureX/`: Each figure has its own folder containing two files:
  - `process_data.py`: A script that takes the raw data as input to perform processing and analysis and outputs files in the correct format for `plot_figures.py`.
  - `plot_figures.py`: A script that generates plots associated with the given figure and saves them to `figures/`, including any supplemental figures related to this figure.
- `plot_all.sh`: A Bash script that runs all `figureX/plot_figures.py` scripts to generate all figures and place them in `figures/`.
- `environment.yml`: A Conda environment file containing software versions used for data processing and figure generation.
