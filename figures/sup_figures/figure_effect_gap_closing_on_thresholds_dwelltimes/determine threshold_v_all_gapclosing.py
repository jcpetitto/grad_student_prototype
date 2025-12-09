"""
Title: Threshold Determination and Visualization for Yeast Cell Transport Events

Description:
This script analyzes transport events in yeast cells by varying the threshold for detecting transport events. It processes data from multiple result folders with different gap closing times and computes thresholds for nuclear and cytoplasmic transport events. The main steps include:

1. **Data Loading and Processing**:
   - Loads tracking data for different yeast strains from CSV and pickle files.
   - Processes the data using custom functions to swap certain column values and load additional data.

2. **Threshold Analysis**:
   - For each result folder (representing different gap closing times), the script iterates over yeast strains.
   - It computes transport events and fits diffusion models to determine binding parameters.
   - Varies the threshold over a range and computes the corresponding parameters (`mu1`, `mu2`, etc.).

3. **Visualization**:
   - Plots the difference between the binding parameters and the transport threshold for nuclear and cytoplasmic events.
   - Identifies the intersection points where the difference crosses zero, indicating optimal thresholds.
   - Stores the thresholds along with their uncertainties.

4. **Data Saving**:
   - Compiles the computed thresholds into a DataFrame.
   - Saves the thresholds to a CSV file (`thresholds.csv`).

Dependencies:
- numpy
- pandas
- matplotlib
- seaborn
- pickle
- scipy
- scienceplots
- Custom modules:
  - `extract_results` (contains `yeast_extractresults`)

Usage:
- Ensure that the required data files are available and paths are correctly set in the configuration.
- Adjust the `result_folder_list`, `digits_list`, and other parameters as needed.
- Run the script to perform the analysis and generate plots.
- The final threshold data will be saved to `thresholds.csv`.

Notes:
- The script assumes a specific directory structure and naming convention for data files.
- Custom functions like `yeast_extractresults` should be defined in the `extract_results` module.
- The script uses a specific plotting style (`science`) for better visualization.

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
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import convolve
from cycler import cycler
from scipy.interpolate import splrep, BSpline
import os
import shutil
import glob

# Apply a scientific plotting style
plt.style.use('science')

# List of result folders
result_folder_list = ['results_tracks_gapclosing2', 'results_tracks_gapclosing5',
                      'results_tracks_gapclosing10', 'results_tracks_gapclosing15']

# Initialize a list to collect threshold data
threshold_data = []

for result_folder in result_folder_list:
    gap_closing_time = int(result_folder.split('gapclosing')[-1])

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

    # Function definitions (load_and_process_tracks, load_other_data, calculate_velocity_splines, etc.)
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
        # Load and process data
        max_number_intersections = 3
        all_tracks = load_and_process_tracks(cfg)
        file_lookup, ne_lookup, ne_data = load_other_data(cfg)

        # Initialize yeast extract results object
        yr = yeast_extractresults(cfg)

        yr.all_tracks = all_tracks
        yr.lookup = file_lookup
        yr.nelookup = ne_lookup
        yr.nedata = ne_data

        # Perform your analysis and plotting here
        thresrange = np.linspace(49, 140, 20)
        mu1_list = []
        mu2_list = []
        error1_list = []
        error2_list = []
        transport_events_gfa, success_export_df, unsuccess_export_df, success_import_df, unsuccess_import_df, retrogate_export_df, retrogate_import_df, _ \
            = yr.find_transports_v2(thressuccess_transport=[-succes_threshold, succes_threshold],
                                    thres_transport=[-thresrange[0], thresrange[0]],
                                    max_number_intersections=max_number_intersections, use_spline=True)
        for qq in thresrange:
            # Continue with your existing analysis and plotting...
            distance_total, num_bins, x_fit, yfit, mu1, error1, mu2, error2 = yr.plot_diffusion_vs_distance(
                [transport_events_gfa, ],
                xlim=[-250, 250],
                num_bins=20, label_list=[cfg['digits']], ylim=[0, 0.0035],
                return_params=True, threshold=qq)
            mu1_list.append(mu1)
            error1_list.append(error1)
            mu2_list.append(mu2)
            error2_list.append(error2)

        sample_distances = np.arange(-250, 250, 4)
        if make_velocity:
            mean_velocities, mean_velocities_proj, bin_centers_velocity = yr.plot_velocity([transport_events_gfa],
                                                                                           sample_distances=sample_distances,
                                                                                           upsample_factor=100,
                                                                                           name_list=[cfg['digits']])
        else:
            mean_velocities, mean_velocities_proj, bin_centers_velocity = [], [], []

        results = {
            'transport_events_gfa': transport_events_gfa,
            'success_export_df': success_export_df,
            'unsuccess_export_df': unsuccess_export_df,
            'success_import_df': success_import_df,
            'unsuccess_import_df': unsuccess_import_df,
            'retrogate_export_df': retrogate_export_df,
            'retrogate_import_df': retrogate_import_df,
            'distance_total': distance_total,
            'num_bins': num_bins,
            'x_fit': x_fit,
            'y_fit': yfit,
            'mean_velocities': mean_velocities,
            'mean_velocities_proj': mean_velocities_proj,
            'bin_centers_velocity': bin_centers_velocity,
            'mu1_list': mu1_list,
            'error1_list': error1_list,
            'error2_list': error2_list,
            'mu2_list': mu2_list,
            'thresrange': thresrange,
            # Add other variables as needed
        }
        return results

    # List of digit sets to loop over
    digits_list = [823, 820, 822]
    all_results = []
    succes_threshold = np.inf

    for digits in digits_list:
        # Update cfg paths for the current set of digits
        cfg = base_cfg.copy()  # Start with the base configuration
        cfg.update({
            'digits': str(digits),
            'all_tracks_path': '../' + result_folder + f'/all_tracks_{digits}.csv',
            'ne_lookup_path': '../' + result_folder + f'/ne_lookup{digits}.csv',
            'file_lookup_path': '../' + result_folder + f'/file_lookup_{digits}.csv',
            'nedata_path': '../' + result_folder + f'/nedata{digits}.pkl',
        })

        # Process data for this configuration
        result = process_for_config(cfg, make_velocity=False, succes_threshold=succes_threshold)
        all_results.append(result)

    # Labels for plotting
    labels = ['GFA1', 'MYO2', 'TRA1']
    CB_color_cycle = ['#0173b2',
                      '#de8f05',
                      '#029e73',
                      '#d55e00',
                      '#cc78bc',
                      '#ca9161',
                      '#fbafe4',
                      '#949494',
                      '#ece133',
                      '#56b4e9']
    # Set the color cycle for Matplotlib
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=CB_color_cycle)

    # Function to find intersections
    def find_intersections(thresrange, y_values):
        intersection_points = np.argwhere(np.diff(np.sign(y_values)) != 0).flatten()
        intersections = []

        for i in intersection_points:
            # Linear interpolation to find the zero crossing
            x1, x2 = thresrange[i], thresrange[i + 1]
            y1, y2 = y_values[i], y_values[i + 1]

            # Interpolate the crossing
            intersection = x1 - y1 * (x2 - x1) / (y2 - y1)
            intersections.append(intersection)

        return np.array(intersections)

    # Create subplots with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=400, figsize=(2, 2.5), sharex=True)
    custom_color_cycle = [CB_color_cycle[i] for i in [0, 1, 2]]
    ax1.set_prop_cycle(cycler('color', custom_color_cycle))
    ax2.set_prop_cycle(cycler('color', custom_color_cycle))

    # Initialize a list to hold entries for this gap_closing_time
    gap_time_entries = []

    # First plot for binding, nuclear
    for i, result in enumerate(all_results):
        negative_sum1 = -np.array(result['mu1_list']) - result['thresrange']
        ax1.plot(result['thresrange'], negative_sum1, '-', label=f"{labels[i]} ")
        ax1.fill_between(result['thresrange'], negative_sum1 - np.array(result['error1_list']),
                         negative_sum1 + np.array(result['error1_list']), alpha=0.3)

        y_values = -np.array(result['mu1_list']) - result['thresrange']
        y_errors = np.array(result['error1_list'])
        y_lower = y_values - y_errors
        y_upper = y_values + y_errors

        intersections_mean = find_intersections(result['thresrange'], y_values)
        intersections_lower = find_intersections(result['thresrange'], y_lower)
        intersections_upper = find_intersections(result['thresrange'], y_upper)

        for j in range(len(intersections_mean)):
            print(
                f'Intersection {labels[i]} at {intersections_mean[j]:.2f} (range: {intersections_lower[j]:.2f} to {intersections_upper[j]:.2f})')

        # Get thresholds
        nuclear_threshold = intersections_mean[0] if len(intersections_mean) > 0 else np.nan
        nuclear_lower = intersections_lower[0] if len(intersections_lower) > 0 else np.nan
        nuclear_upper = intersections_upper[0] if len(intersections_upper) > 0 else np.nan

        # Create entry
        threshold_entry = {
            'gap_closing_time': gap_closing_time,
            'label': labels[i],
            'nuclear_threshold': nuclear_threshold,
            'nuclear_lower': nuclear_lower,
            'nuclear_upper': nuclear_upper,
            # Will fill in cytoplasmic thresholds later
        }

        # Append to gap_time_entries
        gap_time_entries.append(threshold_entry)

    # Second plot for binding, cytoplasmic, combined with transport effects
    for i, result in enumerate(all_results):
        negative_sum2 = np.array(result['mu2_list']) - result['thresrange']
        ax2.plot(result['thresrange'], negative_sum2, label=f"{labels[i]} ")
        ax2.fill_between(result['thresrange'], negative_sum2 - np.array(result['error2_list']),
                         negative_sum2 + np.array(result['error2_list']), alpha=0.3)

        y_values = result['mu2_list'] - result['thresrange']
        y_errors = np.array(result['error2_list'])
        y_lower = y_values - y_errors
        y_upper = y_values + y_errors

        intersections_mean = find_intersections(result['thresrange'], y_values)
        intersections_lower = find_intersections(result['thresrange'], y_lower)
        intersections_upper = find_intersections(result['thresrange'], y_upper)

        for j in range(len(intersections_mean)):
            print(
                f'Intersection {labels[i]} at {intersections_mean[j]:.2f} (range: {intersections_lower[j]:.2f} to {intersections_upper[j]:.2f})')

        # Get thresholds
        cytoplasmic_threshold = intersections_mean[0] if len(intersections_mean) > 0 else np.nan
        cytoplasmic_lower = intersections_lower[0] if len(intersections_lower) > 0 else np.nan
        cytoplasmic_upper = intersections_upper[0] if len(intersections_upper) > 0 else np.nan

        # Update the corresponding entry in gap_time_entries
        gap_time_entries[i]['cytoplasmic_threshold'] = cytoplasmic_threshold
        gap_time_entries[i]['cytoplasmic_lower'] = cytoplasmic_lower
        gap_time_entries[i]['cytoplasmic_upper'] = cytoplasmic_upper

    # After processing both nuclear and cytoplasmic thresholds, append entries to threshold_data
    threshold_data.extend(gap_time_entries)

    ax1.set_ylabel(r'$ -\mu_\text{binding, nuc} \\- T_\text{transport} $ [nm]')
    ax2.axhline(0, color='grey', linestyle='--')
    ax1.axhline(0, color='grey', linestyle='--')
    ax2.set_xlabel(r'$T_\text{transport} $[nm]')
    ax2.set_ylabel(r'$ \mu_\text{binding, cyto} \\- T_\text{transport} $ [nm]')
    ax2.legend()

    fig.tight_layout(pad=0.2)
    fig.savefig('../' + result_folder + r'/threshold_transportevent.svg', format='svg')
    fig.show()
    plt.close('all')

    # Copy SVG files to destination folder
    # [Your existing code for copying files]

# After processing all result folders, create a DataFrame and save to CSV
df_thresholds = pd.DataFrame(threshold_data)
df_thresholds.to_csv('thresholds.csv', index=False)
print("Threshold data saved to 'thresholds.csv'")
