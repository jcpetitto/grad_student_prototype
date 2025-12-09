"""
Title: Dwell Time and Binding Parameter Analysis for Yeast Cell Transport Events

Description:
This script analyzes transport events in yeast cells using thresholds determined from previous analyses. It processes data from multiple result folders with different gap closing times, applies the calculated thresholds to identify successful transport events, and computes dwell times and binding parameters for each yeast strain. The main steps include:

1. **Data Loading and Processing**:
   - Reads thresholds from a CSV file generated in earlier analyses (`thresholds.csv`).
   - Loads tracking data, nuclear envelope data, and other necessary information for each yeast strain.
   - Processes the data using custom functions to swap certain column values and load additional data.

2. **Data Analysis**:
   - For each result folder (representing different gap closing times), the script iterates over yeast strains.
   - It applies the thresholds to identify successful transport events.
   - Fits a double Gaussian model to the histogram of distances to identify binding sites and their parameters.
   - Computes dwell times in the nuclear, cytoplasmic, and combined compartments using exponential decay fitting.

3. **Visualization**:
   - Plots histograms of the distance to the membrane and overlays the fitted double Gaussian model.
   - Prints the optimized parameters and their errors for each strain.

4. **Data Compilation and Saving**:
   - Compiles the computed dwell times and binding parameters into DataFrames.
   - Saves the dwell times and binding parameters to CSV files (`dwell_times_table.csv` and `binding_params_table.csv`).

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
- Adjust the `result_folder_list`, `labels`, `digits_list`, and other parameters as needed.
- Run the script to perform the analysis and generate plots.
- The final dwell time and binding parameter data will be saved to CSV files.

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
from cycler import cycler
from scipy.interpolate import splrep, BSpline
from scipy.optimize import curve_fit
import os
import shutil
import glob

# Apply a scientific plotting style
plt.style.use('science')

# Read the thresholds from the CSV file generated earlier
thresholds_df = pd.read_csv('thresholds.csv')

# List of result folders
result_folder_list = ['results_tracks_gapclosing2', 'results_tracks_gapclosing5',
                      'results_tracks_gapclosing10', 'results_tracks_gapclosing15']

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

def process_for_config(cfg, make_velocity=True, thres_success_transport=[-200, 200]):
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
    transport_events_gfa, success_export_df, unsuccess_export_df, success_import_df, unsuccess_import_df, retrogate_export_df, retrogate_import_df, _ \
        = yr.find_transports_v2(thressuccess_transport=thres_success_transport, thres_transport=thres_success_transport,
                                max_number_intersections=max_number_intersections, use_spline=True)
    # Continue with your existing analysis and plotting...
    distance_total, num_bins, x_fit, yfit = yr.plot_diffusion_vs_distance([transport_events_gfa,], xlim=[-200, 200],
                                  num_bins=20, label_list=[cfg['digits']], ylim=[0, 0.0035])
    sample_distances = np.arange(-250, 250, 4)
    if make_velocity:
        mean_velocities, mean_velocities_proj, bin_centers_velocity = yr.plot_velocity([transport_events_gfa],
                                                                                      sample_distances=sample_distances, upsample_factor=100, name_list=[cfg['digits']])
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
        # Add other variables as needed
    }
    return results

def double_gaussian_model(x, width1, center1, scale1, width2, center2, scale2, offset, sigma):
    return (scale1 * np.exp(-(x - center1) ** 2 / (2 * (width1 ** 2 + sigma ** 2))) +
            scale2 * np.exp(-(x - center2) ** 2 / (2 * (width2 ** 2 + sigma ** 2))) +
            offset)

def fit_model(x, width1, center1, scale1, width2, center2, scale2, offset):
    return double_gaussian_model(x, width1, center1, scale1, width2, center2, scale2, offset, measurement_sigma)

# List of labels and corresponding digits
labels = ['GFA1', 'MYO2', 'TRA1']
digits_list = ['823', '820', '822']

# Initialize lists to collect data
dwell_time_data = []
binding_params_data = []

# Loop over result folders
for result_folder in result_folder_list:
    gap_closing_time = int(result_folder.split('gapclosing')[-1])
    print(f"Processing gap closing time: {gap_closing_time}")

    # Now get the thresholds for this gap_closing_time
    thresholds_sub_df = thresholds_df[thresholds_df['gap_closing_time'] == gap_closing_time]

    # Initialize empty list for all_results
    all_results = []

    for i, digits in enumerate(digits_list):
        label_name = labels[i]

        # Get thresholds for this label and gap_closing_time
        threshold_row = thresholds_sub_df[thresholds_sub_df['label'] == label_name]
        if not threshold_row.empty:
            nuclear_threshold = threshold_row['nuclear_threshold'].values[0]
            nuclear_lower = threshold_row['nuclear_lower'].values[0]
            nuclear_upper = threshold_row['nuclear_upper'].values[0]
            cytoplasmic_threshold = threshold_row['cytoplasmic_threshold'].values[0]
            cytoplasmic_lower = threshold_row['cytoplasmic_lower'].values[0]
            cytoplasmic_upper = threshold_row['cytoplasmic_upper'].values[0]

            # Compute negative_threshold and positive_threshold
            negative_threshold = -nuclear_threshold
            positive_threshold = cytoplasmic_threshold

            thresh = [negative_threshold, positive_threshold]
        else:
            # Handle missing data
            print(f"Thresholds not found for label {label_name} and gap_closing_time {gap_closing_time}")
            thresh = [np.nan, np.nan]

        print(f"Using thresholds for {label_name}: {thresh}")

        # Update cfg paths for the current set of digits
        cfg = base_cfg.copy()  # Start with the base configuration
        cfg.update({
            'digits': digits,
            'all_tracks_path': '../' + result_folder + f'/all_tracks_{digits}.csv',
            'ne_lookup_path': '../' + result_folder + f'/ne_lookup{digits}.csv',
            'file_lookup_path': '../' + result_folder + f'/file_lookup_{digits}.csv',
            'nedata_path': '../' + result_folder + f'/nedata{digits}.pkl',
        })

        # Process data for this configuration
        result = process_for_config(cfg, make_velocity=False, thres_success_transport=thresh)
        all_results.append(result)

        # Make binding plot
        hist, bin_edges = np.histogram(result['distance_total'], bins=30, range=(-250, 250))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Fit the model using curve_fit
        measurement_sigma = 49 / np.sqrt(2)  # Your measurement sigma
        initial_params = [20, -80, 80000, 20, 80, 80000, 1000]  # Adjust these initial params as needed
        try:
            popt, pcov = curve_fit(fit_model, bin_centers, hist, p0=initial_params)
            perr = np.sqrt(np.diag(pcov))  # Standard errors of the parameters

            # Extract parameters
            width1, center1, scale1 = popt[0], popt[1], popt[2]
            width2, center2, scale2 = popt[3], popt[4], popt[5]
            offset = popt[6]
            # Errors
            width1_err, center1_err = perr[0], perr[1]
            width2_err, center2_err = perr[3], perr[4]

            # Collect binding parameters for the table
            binding_params_data.append({
                'gap_closing_time': gap_closing_time,
                'Label': label_name,
                'width1': width1,
                'width1_err': width1_err,
                'center1': center1,
                'center1_err': center1_err,
                'width2': width2,
                'width2_err': width2_err,
                'center2': center2,
                'center2_err': center2_err
            })

            # Round parameter values and ceil errors
            popt_rounded = np.round(popt).astype(int)
            perr_ceiled = np.ceil(perr).astype(int)

            # Plotting for each strain
            x_plot = np.linspace(-250, 250, 1000)
            plt.figure(figsize=(3, 1.5))
            plt.bar(bin_centers, hist, width=np.diff(bin_edges)[0], alpha=0.6,
                    label='data ' + labels[i])
            plt.plot(x_plot, double_gaussian_model(x_plot, *popt, 0), 'k--', label='binding sites')
            plt.plot(x_plot, double_gaussian_model(x_plot, *popt, measurement_sigma), 'k:',
                     label=r'$\sigma_\text{binding}$ + $\sigma_\text{CoLoc}$ ')
            plt.xlabel('Distance to membrane [nm]')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            plt.show()

            print(f"{labels[i]} Optimized parameters and their errors:")
            print(f"Width1 (sigma) = {popt[0]:.2f} ± {perr[0]:.2f}")
            print(f"Center1 = {popt[1]:.2f} ± {perr[1]:.2f}")
            print(f"Scale1 = {popt[2]:.2f} ± {perr[2]:.2f}")
            print(f"Width2 (sigma) = {popt[3]:.2f} ± {perr[3]:.2f}")
            print(f"Center2 = {popt[4]:.2f} ± {perr[4]:.2f}")
            print(f"Scale2 = {popt[5]:.2f} ± {perr[5]:.2f}")
            print(f"Offset = {popt[6]:.2f} ± {perr[6]:.2f}")
        except RuntimeError:
            print(f"Error - curve_fit failed for {label_name} at gap closing time {gap_closing_time}")
            # Set parameters and errors to NaN
            width1 = center1 = width2 = center2 = np.nan
            width1_err = center1_err = width2_err = center2_err = np.nan
            # Collect binding parameters with NaNs
            binding_params_data.append({
                'gap_closing_time': gap_closing_time,
                'Label': label_name,
                'width1': width1,
                'width1_err': width1_err,
                'center1': center1,
                'center1_err': center1_err,
                'width2': width2,
                'width2_err': width2_err,
                'center2': center2,
                'center2_err': center2_err
            })
            continue  # Skip to next iteration

        # Define thresholds for dwell times using the fitted parameters
        now_thres_dwell = [center1 - 2 * width1, center2 + 2 * width2]

        # Now proceed to compute the dwell times using these thresholds
        # For each label, calculate dwell times
        export = result['success_export_df']

        # Use the thresholds 'now_thres_dwell' for filtering
        filtered_df = export[export['dist_cor_spline'].between(now_thres_dwell[0], now_thres_dwell[1])]
        filtered_df_nuc = export[export['dist_cor_spline'].between(now_thres_dwell[0], 0)]
        filtered_df_cyto = export[export['dist_cor_spline'].between(0, now_thres_dwell[1])]

        # Initialize dictionaries to store filtered data
        filtered_dfs = {
            'nuc': filtered_df_nuc,
            'cyto': filtered_df_cyto,
            'export': filtered_df,
        }

        # Define the exponential decay function
        def exp_func(x, a, b):
            return a * np.exp(-x / b)

        max_time = 50  # Maximum time (in frames) for the histograms

        for filter_key, filtered_df_comp in filtered_dfs.items():
            # Step 2: Group by 'id' and count the occurrences
            count_by_id = filtered_df_comp.groupby('id').size()

            # Step 3: Transform counts so each time contributes to all preceding bins up to that time
            histogram_data = np.zeros(max_time + 1)
            avg_time = []
            for time in count_by_id:
                avg_time.append(time)
                time = int(time)
                if time >= max_time:
                    time = max_time - 1  # To avoid index out of bounds
                histogram_data[1:time + 1] += 1

            # Define x values for fitting
            x_values = np.arange(1, len(histogram_data))

            # Fit the exponential model
            try:
                popt_exp, pcov_exp = curve_fit(exp_func, x_values, histogram_data[1:], p0=(histogram_data[1], 10), maxfev=5000)
                # Calculate the standard errors of the parameters
                perr_exp = np.sqrt(np.diag(pcov_exp))
                # Round the dwell time and ceil the error
                dwell_time = int(np.round(popt_exp[1] * 20))  # Convert tau to ms and round
                dwell_time_error = int(np.ceil(perr_exp[1] * 20))  # Convert error to ms and ceil
            except RuntimeError:
                # If the curve_fit fails, set dwell time and error to NaN
                dwell_time = np.nan
                dwell_time_error = np.nan

            # Collect dwell times for the table
            dwell_time_data.append({
                'gap_closing_time': gap_closing_time,
                'Label': label_name,
                'Compartment': filter_key,
                'Dwell_Time_ms': dwell_time,
                'Error_ms': dwell_time_error,
            })

            # Optionally, you can plot the histograms and fits here
            # (Omitted for brevity)

# After processing all result folders, create DataFrame for dwell times
df_dwell_times = pd.DataFrame(dwell_time_data)

# Pivot the table to have compartments as columns
df_pivot_dwell = df_dwell_times.pivot_table(index=['gap_closing_time', 'Label'], columns='Compartment', values=['Dwell_Time_ms', 'Error_ms']).reset_index()

# Display the dwell times table
print("\nDwell Times Table:")
print(df_pivot_dwell)

# Save dwell times to CSV
df_pivot_dwell.to_csv('dwell_times_table.csv', index=False)
print("Dwell times data saved to 'dwell_times_table.csv'")

# Create DataFrame for binding parameters
df_binding_params = pd.DataFrame(binding_params_data)

# Display the binding parameters table
print("\nBinding Parameters Table:")
print(df_binding_params)

# Save binding parameters to CSV
df_binding_params.to_csv('binding_params_table.csv', index=False)
print("Binding parameters data saved to 'binding_params_table.csv'")
