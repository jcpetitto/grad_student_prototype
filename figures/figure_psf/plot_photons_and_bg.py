"""
Title: Signal and Background Analysis for Yeast Strains Using Microscopy Data

Description:
This script processes microscopy tracking data from yeast cells of different strains (GFA1, MYO2, TRA1) to analyze the distribution of signal and background photon counts. It performs the following steps:

1. **Data Loading and Processing**:
   - Loads tracking data for each yeast strain from CSV and pickle files.
   - Processes the data to swap certain column values (e.g., in/out states).
   - Stores the processed data for each strain in a list.

2. **Data Concatenation and Binning**:
   - Concatenates all background (`bg`) and signal (`signal`) data from all strains.
   - Calculates bins for histograms based on the combined data to ensure consistency across strains.

3. **Visualization**:
   - Creates histograms of the background and signal distributions for each strain.
   - Fits a Log-Normal distribution to the signal data (commented out in the plotting section).
   - Plots the histograms with appropriate labels and legends.
   - Saves the final figure as an SVG file.

Dependencies:
- numpy
- pandas
- matplotlib
- seaborn
- pickle
- scipy
- scienceplots
- extract_results (custom module)
- cycler

Usage:
- Ensure all required data files are available at the specified paths.
- Adjust file paths and configurations as needed.
- Run the script to generate the histograms and save the figure.

Notes:
- The script assumes that the data files are organized in specific folders and have certain naming conventions.
- The `yeast_extractresults` function is assumed to be defined in the `extract_results` module.

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import stats
from matplotlib.animation import FuncAnimation
from extract_results import yeast_extractresults
import scienceplots
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import splprep, splev, UnivariateSpline
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.signal import convolve
from cycler import cycler
from scipy.interpolate import splrep, BSpline
from scipy.stats import gamma, lognorm
# Apply a scientific plotting style
plt.style.use('science')

result_folder_list = ['results_tracks_gapclosing2', 'results_tracks_gapclosing5', 'results_tracks_gapclosing10',
                      'results_tracks_gapclosing15']
result_folder_list = ['../results_tracks_gapclosing10']

# Base configuration
base_cfg = {
    'digits': '',
    'mainpath': '/media/pieter/Extreme SSD/Yeast_tracking_data2023/Simulation',
    'resultdir': "v3GLRT_0_95",
    'trackdata': 'tracks.data',
    'pixelsize': 128,
    'moviename': '/media/pieter/Extreme SSD/Yeast_tracking_data2023/',
    'frametime': 0.02,
    'columns_to_swap': ['may_inout', 'def_inout', 'may_inout_spline', 'def_inout_spline']
}


# Function definitions
def load_and_process_tracks(cfg):
    all_tracks = pd.read_csv(cfg['all_tracks_path'])
    for col in cfg['columns_to_swap']:
        all_tracks.loc[(all_tracks[col] == 0) & (all_tracks[col].notna()), col] = 2
        all_tracks.loc[(all_tracks[col] == 1) & (all_tracks[col].notna()), col] = 0
        all_tracks.loc[(all_tracks[col] == 2) & (all_tracks[col].notna()), col] = 1
    return all_tracks


def load_other_data(cfg):
    file_lookup = pd.read_csv(cfg['file_lookup_path'])
    ne_lookup = pd.read_csv(cfg['ne_lookup_path'])
    with open(cfg['nedata_path'], 'rb') as file:
        ne_data = pickle.load(file)
    return file_lookup, ne_lookup, ne_data


def process_for_config(cfg, make_velocity=True, succes_threshold=120):
    all_tracks = load_and_process_tracks(cfg)
    file_lookup, ne_lookup, ne_data = load_other_data(cfg)
    yr = yeast_extractresults(cfg)
    yr.all_tracks = all_tracks
    return all_tracks


digits_list = [823, 820, 822]
all_results = []
succes_threshold = 150

for result_folder in result_folder_list:
    for digits in digits_list:
        cfg = base_cfg.copy()
        cfg.update({
            'digits': str(digits),
            'all_tracks_path': f'../{result_folder}/all_tracks_{digits}.csv',
            'ne_lookup_path': f'../{result_folder}/ne_lookup{digits}.csv',
            'file_lookup_path': f'../{result_folder}/file_lookup_{digits}.csv',
            'nedata_path': f'../{result_folder}/nedata{digits}.pkl',
        })
        all_tracks = process_for_config(cfg, make_velocity=False, succes_threshold=succes_threshold)
        all_results.append(all_tracks)

label = ['GFA1', 'MYO2', 'TRA1']
CB_color_cycle = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#fbafe4', '#949494', '#ece133',
                  '#56b4e9']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=CB_color_cycle)
from cycler import cycler
from scipy.interpolate import splrep, BSpline

custom_color_cycle = [CB_color_cycle[i] for i in [0, 1, 2]]

# Concatenate all results to get the combined data for calculating bins
all_bg = np.concatenate([df['bg'] for df in all_results])
all_signal = np.concatenate([df['signal'] for df in all_results])

# Calculate bins using the combined data
bg_bins = np.linspace(all_bg.min(), all_bg.max(), 55)
signal_bins = np.linspace(all_signal.min(), all_signal.max(), 55)

# Create subplots
fig, axs = plt.subplots(len(all_results), 2, dpi=400, figsize=(4, 4), sharex='col')

for i, strain_data in enumerate(all_results):
    # Background histogram
    axs[i, 1].hist(strain_data['bg'], bins=30, density=True, color=CB_color_cycle[i % len(CB_color_cycle)],
                   label=label[i],range=[0,300])
    if i == len(all_results) - 1:
        axs[i, 1].set_xlabel('Mean background\n [photons/pixel]')
    axs[i, 1].set_ylabel('Probability')
    axs[i, 1].legend()

    # Signal histogram and fit distribution
    signal_data = strain_data['signal']
    axs[i, 0].hist(signal_data, bins=30, density=True, color=CB_color_cycle[i % len(CB_color_cycle)],
                   label=label[i],range=[0,1000])

    # Fit a Log-Normal distribution to the signal data
    params_lognorm = lognorm.fit(signal_data)

    # Plot the fitted Gamma distribution
    xmin, xmax = axs[i, 0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    # p_gamma = gamma.pdf(x, *params_gamma)
    # axs[i, 0].plot(x, p_gamma, 'k', linewidth=2, label=f'Gamma Fit: params={params_gamma}')

    # Plot the fitted Log-Normal distribution
    p_lognorm = lognorm.pdf(x, *params_lognorm)
    #axs[i, 0].plot(x, p_lognorm, 'r', linewidth=2, label=f'LogNorm Fit: params={params_lognorm}')

    if i == len(all_results) - 1:
        axs[i, 0].set_xlabel('Signal [photons]')
    axs[i, 0].set_ylabel('Probability')
    axs[i, 0].legend()

plt.tight_layout(pad=0.1)
plt.savefig('signal_bg_allstrains.svg', format='svg')
plt.show()