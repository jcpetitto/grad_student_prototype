"""
Title: Yeast Tracking Data Analysis Script for Dwell Time Distributions

Description:
This script analyzes yeast tracking data to study dwell time distributions in different compartments (nuclear, cytoplasmic, and export regions) during transport events. It processes the tracking data, identifies successful export events, filters data based on specified distance thresholds, and fits exponential decay models to the dwell time data. The script generates plots to visualize the dwell time distributions and saves them as SVG files.

Functions:
- load_and_process_tracks(cfg): Loads and processes tracking data from CSV files. It modifies specified columns by swapping values to correct the data format.
- load_other_data(cfg): Loads additional data required for the analysis, including file lookups and nuclear envelope (NE) data.
- process_for_config(cfg, make_velocity=True, thres_success_transport=[-200, 200]): Processes data for a given configuration, performs analysis, and returns a dictionary containing the results.
- exp_func(x, a, b): Defines the exponential decay function used for curve fitting.

Main Workflow:
1. Sets up the base configuration and result folder.
2. Defines functions for data loading and processing.
3. Processes data using the `process_for_config` function for each configuration in the `digits_list`.
4. Identifies successful export events and filters data based on distance thresholds.
5. Calculates dwell times in nuclear, cytoplasmic, and export compartments.
6. Fits exponential decay models to the dwell time data.
7. Generates and saves plots for the dwell time distributions.
8. Collects dwell times into a table and displays or saves it.

Dependencies:
- numpy
- pandas
- matplotlib
- pickle
- scipy
- extract_results (custom module)
- scienceplots
- cycler

Usage:
- Ensure that the required data files are located in the paths specified in the configuration.
- Adjust the `digits_list` and `threshold_list` as needed for your data.
- Run the script to perform the analysis and generate the plots.
- The results are saved in the specified `result_folder`.

Notes:
- The script uses a custom module `yeast_extractresults` which should be available in the Python path.
- The plotting style is set to 'science' using the `scienceplots` package.
- Measurement sigma and initial parameters for curve fitting are set based on assumptions; adjust as necessary.
- The code includes steps for both individual and combined analysis of dwell times across different labels.
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

# Apply a scientific plotting style
plt.style.use('science')

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

def process_for_config(cfg, make_velocity = True,
thres_success_transport = [-200,200] ):

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
    # For example:
    transport_events_gfa, success_export_df, unsuccess_export_df, success_import_df, unsuccess_import_df, retrogate_export_df, retrogate_import_df,_ \
        = yr.find_transports_v2(thressuccess_transport=thres_success_transport, thres_transport=thres_success_transport,
                                max_number_intersections=max_number_intersections, use_spline=True)
    # Continue with your existing analysis and plotting...
    distance_total,num_bins,x_fit, yfit = yr.plot_diffusion_vs_distance([transport_events_gfa,], xlim=[-200, 200],
                                  num_bins=20, label_list=[cfg['digits']],ylim=[0,0.0035])
    sample_distances = np.arange(-250, 250, 4)
    if make_velocity:
        mean_velocities, mean_velocities_proj,bin_centers_velocity = yr.plot_velocity([transport_events_gfa],
                                                                  sample_distances=sample_distances, upsample_factor=100, name_list=[cfg['digits']])
    else:
        mean_velocities, mean_velocities_proj, bin_centers_velocity = [],[],[]

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
# List of digit sets to loop over
digits_list = ['823','820','822']
result_folder = '../results_tracks_gapclosing10'
all_results = []
threshold_list = [[-76.24, 94.76], [-89.75,78.61], [-69.74,77.45]]
for i, digits in enumerate(digits_list):
    # Update cfg paths for the current set of digits
    cfg = base_cfg.copy()  # Start with the base configuration
    cfg.update({
        'digits': digits,
        'all_tracks_path': '../'+result_folder+f'/all_tracks_{digits}.csv',
        'ne_lookup_path': '../'+result_folder+f'/ne_lookup{digits}.csv',
        'file_lookup_path': '../'+result_folder+f'/file_lookup_{digits}.csv',
        'nedata_path': '../'+result_folder+f'/nedata{digits}.pkl',
    })

    # Process data for this configuration
    result= process_for_config(cfg, make_velocity = False,thres_success_transport=threshold_list[i])
    all_results.append(result)
label = ['GFA1','MYO2', 'TRA1']
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

from cycler import cycler
from scipy.interpolate import splrep, BSpline

from scipy.optimize import curve_fit


# Define the exponential decay function
def exp_func(x, a, b):
    return a * np.exp(-x / b)




#
# label = ['GFA1', 'MYO2', 'TRA1']
# for i, result in enumerate(all_results):
#     export = result['success_export_df']
#
#     # Step 1: Filter the DataFrame where 'dist_cor_spline' is within the range -a to a
#     filtered_df = export[export['dist_cor_spline'].between(-thres_success_transport, thres_success_transport)]
#
#     # Step 2: Group by 'id' and count the occurrences
#     count_by_id = filtered_df.groupby('id').size()
#
#     # Step 3: Transform counts so each time contributes to all preceding bins up to that time
#     max_time = 50
#     histogram_data = np.zeros(max_time + 1)
#     avg_time = []
#     for time in count_by_id:
#         avg_time.append(time)
#         histogram_data[1:time + 1] += 1
#
#     # Define x values for fitting
#     x_values = np.arange(1, len(histogram_data))-1
#
#     # Fit the exponential model
#     popt, pcov = curve_fit(exp_func, x_values, histogram_data[1:], p0=(50, 10))
#
#     # Calculate the standard errors of the parameters
#     perr = np.sqrt(np.diag(pcov))
#
#     # Step 4: Plot the histogram and the fit
#     plt.figure(figsize=(3, 3))
#     plt.bar(x_values*20, histogram_data[1:], color='blue', label=label[i],width=20)
#     plt.plot(x_values*20, exp_func(x_values, *popt), 'k--',
#              label='Fit: b=%.2f±%.2f' % (popt[1]*20, perr[1]*20))
#     plt.xlabel('Time [ms]')
#     plt.ylabel('Counts')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'export_time_{label[i]}.svg', format='svg')
#     plt.show()
#     print('avg time = ', np.mean(np.array(avg_time))*20, ' time exp. = ',popt[1]*20 )
#
#


label = ['GFA1', 'MYO2', 'TRA1']
list_exp = [[-(75+2*35), (97+2*47)],[-(90+2*41), (78+2*38)],[-(69+2*27), (77+2*34)]]
for i, result in enumerate(all_results):
    export = result['success_export_df']
    thresh = list_exp[i]
    # Step 1: Filter the DataFrame where 'dist_cor_spline' is within the range -a to a
    filtered_df = export[export['dist_cor_spline'].between(thresh[0], thresh[1])]
    filtered_df_nuc = export[export['dist_cor_spline'].between(thresh[0], 0)]
    filtered_df_cyto = export[export['dist_cor_spline'].between(-0, thresh[1])]

    # Step 2: Group by 'id' and count the occurrences
    count_by_id = filtered_df.groupby('id').size()

    # Step 3: Transform counts so each time contributes to all preceding bins up to that time
    max_time = 50
    histogram_data = np.zeros(max_time + 1)
    avg_time = []
    for time in count_by_id:
        avg_time.append(time)
        histogram_data[1:time + 1] += 1

    # Define x values for fitting
    x_values = np.arange(1, len(histogram_data))-1

    # Fit the exponential model
    popt, pcov = curve_fit(exp_func, x_values, histogram_data[1:], p0=(50, 10))

    # Calculate the standard errors of the parameters
    perr = np.sqrt(np.diag(pcov))

    # Step 4: Plot the histogram and the fit
    plt.figure(figsize=(3, 3))
    plt.bar(x_values*20, histogram_data[1:], color='blue', label=label[i],width=20)
    plt.plot(x_values*20, exp_func(x_values, *popt), 'k--',
             label='Fit: b=%.2f±%.2f' % (popt[1]*20, perr[1]*20))
    plt.xlabel('Time [ms]')
    plt.ylabel('Counts')
    plt.legend()
    plt.tight_layout()
    plt.savefig('graphs/'+rf'/export_time_{label[i]}.svg', format='svg')
    plt.show()
    print('avg time = ', np.mean(np.array(avg_time))*20, ' time exp. = ',popt[1]*20 )
#
# label = ['GFA1', 'MYO2', 'TRA1']
# for i, result in enumerate(all_results):
#     export = result['success_export_df']
#
#     # Create dictionaries to store filtered data
#     filtered_dfs = {
#         'cyto': export[export['dist_cor_spline'].between(0, thres_success_transport)],
#         'nuc': export[export['dist_cor_spline'].between(-thres_success_transport, 0)],
#         'full': export[export['dist_cor_spline'].between(-thres_success_transport, thres_success_transport)]
#     }
#
#     # Iterate over each filtered DataFrame
#     for filter_key, filtered_df in filtered_dfs.items():
#         # Step 2: Group by 'id' and count the occurrences
#         count_by_id = filtered_df.groupby('id').size()
#
#         # Step 3: Transform counts so each time contributes to all preceding bins up to that time
#         max_time = 50
#         histogram_data = np.zeros(max_time + 1)
#         avg_time = []
#         for time in count_by_id:
#             avg_time.append(time)
#             histogram_data[1:time + 1] += 1
#
#         # Define x values for fitting
#         x_values = np.arange(1, len(histogram_data))
#
#         # Fit the exponential model
#         popt, pcov = curve_fit(exp_func, x_values, histogram_data[1:], p0=(50, 10))
#
#         # Calculate the standard errors of the parameters
#         perr = np.sqrt(np.diag(pcov))
#
#         # Step 4: Plot the histogram and the fit
#         plt.figure(figsize=(3, 3))
#         plt.bar(x_values * 20, histogram_data[1:], color='blue', label=f'{label[i]} {filter_key}', width=20)
#         plt.plot(x_values * 20, exp_func(x_values, *popt), 'k--',
#                  label=f'Fit: b={popt[1]*20:.2f}±{perr[1]*20:.2f}')
#         plt.xlabel('Time [ms]')
#         plt.ylabel('Counts')
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(f'export_time_{label[i]}_{filter_key}.svg', format='svg')
#         plt.show()
#         print(f'avg time for {filter_key} = ', np.mean(np.array(avg_time)) * 20, 'ms, time exp. = ', popt[1] * 20, 'ms')



label = ['GFA1', 'MYO2', 'TRA1']
labels = ['GFA1', 'MYO2', 'TRA1']
list_exp = [[-(75+2*35), (97+2*47)],[-(90+2*41), (78+2*38)],[-(69+2*27), (77+2*34)]]
for i, result in enumerate(all_results):
    export = result['success_export_df']
    thresh = list_exp[i]
    # Step 1: Filter the DataFrame where 'dist_cor_spline' is within the range -a to a
    filtered_df = export[export['dist_cor_spline'].between(thresh[0], thresh[1])]
    filtered_df_nuc = export[export['dist_cor_spline'].between(thresh[0], 0)]
    filtered_df_cyto = export[export['dist_cor_spline'].between(-0, thresh[1])]
dwell_time_data = []

for i, result in enumerate(all_results):
    export = result['success_export_df']
    thresh = list_exp[i]

    # Create dictionaries to store filtered data
    filtered_dfs = {
       # 'full': export[export['dist_cor_spline'].between(-thres_success_transport, thres_success_transport)],
        'nuc': export[export['dist_cor_spline'].between(thresh[0], 0)],
        'cyto': export[export['dist_cor_spline'].between(0, thresh[1])],
        'export': export[export['dist_cor_spline'].between(thresh[0], thresh[1])],

    }
    print(f"For {label[i]}:")
    print(f"Number of traces used for nuclear dwell time: {len(np.unique(filtered_dfs['nuc']['id']))}")
    print(f"Number of traces used for cytoplasmic dwell time: {len(np.unique(filtered_dfs['cyto']['id']))}")
    print(f"Number of traces for export dwell time: {len(np.unique(filtered_dfs['export']['id']))}")

    # Create subplots with a shared x-axis
    fig, axs = plt.subplots(3, 1, figsize=(3, 3), sharex=True)

    for ax, (filter_key, filtered_df) in zip(axs, filtered_dfs.items()):
        # Step 2: Group by 'id' and count the occurrences
        count_by_id = filtered_df.groupby('id').size()

        # Step 3: Transform counts so each time contributes to all preceding bins up to that time
        max_time = 50
        histogram_data = np.zeros(max_time + 1)
        avg_time = []
        for time in count_by_id:
            avg_time.append(time)
            histogram_data[1:time + 1] += 1

        # Define x values for fitting
        x_values = np.arange(1, len(histogram_data))

        # Fit the exponential model
        popt, pcov = curve_fit(exp_func, x_values, histogram_data[1:], p0=(50, 10))

        # Calculate the standard errors of the parameters
        perr = np.sqrt(np.diag(pcov))
        # Round the dwell time and ceil the error
        dwell_time = int(np.round(popt[1] * 20))  # Convert tau to ms and round
        dwell_time_error = int(np.ceil(perr[1] * 20))  # Convert error to ms and ceil

        # Collect dwell times for the table
        dwell_time_data.append({
            'Label': label[i],
            'Compartment': filter_key,
            'Dwell Time (ms)': f"{dwell_time} ± {dwell_time_error}"
        })

        # Plot the histogram and the fit
        ax.bar(x_values * 20, histogram_data[1:], color=CB_color_cycle[i], label=f"{label[i]} $\\tau_\\text{{{filter_key}}}$" , width=20)
        ax.plot(x_values * 20, exp_func(x_values, *popt), 'k--',
                label=f'$\\tau_\\text{{{filter_key}}}$={popt[1]*20:.2f}±{perr[1]*20:.2f}')
        print(f'{label[i]}$\\tau_\\text{{{filter_key}}}$={popt[1]*20:.2f}±{perr[1]*20:.2f}')
        ax.set_ylabel('Counts')
        plt.xlim(0,400)
        ax.legend()

    # Set common labels
    plt.xlabel('Time [ms]')
    plt.tight_layout()
    plt.savefig('graphs/'+rf'/export_time_{label[i]}_all_combined.svg', format='svg')
    plt.show()
    # for filter_key in filtered_dfs:
    #     print(f'avg time for {filter_key} = ', np.mean(np.array(avg_time)) * 20, 'ms, time exp. = ', popt[1] * 20, 'ms')
# Create a DataFrame for the dwell times
df_dwell_times = pd.DataFrame(dwell_time_data)

# Pivot the table to have compartments as columns
df_pivot = df_dwell_times.pivot(index='Label', columns='Compartment', values='Dwell Time (ms)').reset_index()

# Display the table
print("\nDwell Times Table:")
print(df_pivot)

# Initialize a list to store dwell times for the table
dwell_time_data = []

# Initialize dictionaries to store histogram_data for each compartment across labels
compartments = ['nuc', 'cyto', 'export']
histogram_data_dict = {comp: {} for comp in compartments}

max_time = 50  # Maximum time (in frames) for the histograms

for i, result in enumerate(all_results):
    export = result['success_export_df']
    thresh = list_exp[i]
    label_name = labels[i]  # Use label_name instead of label

    # Create dictionaries to store filtered data
    filtered_dfs = {
        'nuc': export[export['dist_cor_spline'].between(thresh[0], 0)],
        'cyto': export[export['dist_cor_spline'].between(0, thresh[1])],
        'export': export[export['dist_cor_spline'].between(thresh[0], thresh[1])],
    }

    for filter_key, filtered_df in filtered_dfs.items():
        # Step 2: Group by 'id' and count the occurrences
        count_by_id = filtered_df.groupby('id').size()

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
        popt, pcov = curve_fit(exp_func, x_values, histogram_data[1:], p0=(histogram_data[1], 10))

        # Calculate the standard errors of the parameters
        perr = np.sqrt(np.diag(pcov))
        # Round the dwell time and ceil the error
        dwell_time = int(np.round(popt[1] * 20))  # Convert tau to ms and round
        dwell_time_error = int(np.ceil(perr[1] * 20))  # Convert error to ms and ceil

        # Collect dwell times for the table
        dwell_time_data.append({
            'Label': label_name,
            'Compartment': filter_key,
            'Dwell Time (ms)': f"{dwell_time} ± {dwell_time_error}"
        })

        # Store histogram data for plotting later
        histogram_data_dict[filter_key][label_name] = {
            'x_values': x_values * 20,  # Convert to ms
            'histogram_data': histogram_data[1:],  # Skip the zero bin
            'popt': popt,
            'perr': perr
        }

# Now, for each compartment, plot the cumulative histograms for all labels
for filter_key in compartments:
    for density_setting in [False, True]:  # First plot counts, then density
        plt.figure(figsize=(4,4))
        for i, label_name in enumerate(labels):
            data = histogram_data_dict[filter_key].get(label_name)
            if data:
                x_values = data['x_values']
                histogram_data = data['histogram_data']
                popt = data['popt']
                perr = data['perr']
                dwell_time = int(np.round(popt[1] * 20))  # Convert tau to ms and round
                dwell_time_error = int(np.ceil(perr[1] * 20))  # Convert error to ms and ceil

                # Normalize the histogram data if density is True
                if density_setting:
                    total_counts = np.sum(histogram_data)
                    if total_counts > 0:
                        histogram_data_normalized = histogram_data / total_counts
                    else:
                        histogram_data_normalized = histogram_data  # Avoid division by zero
                    y_values = histogram_data_normalized
                    ylabel = 'Density'
                else:
                    y_values = histogram_data
                    ylabel = 'Counts'

                # Plot histogram data
                plt.bar(x_values, y_values, width=20, alpha=0.5, color=CB_color_cycle[i],
                        label=rf"{label_name} $\tau={dwell_time}\pm{dwell_time_error}$ ms")

                # Plot exponential fit
                if density_setting:
                    # Normalize the exponential fit as well
                    exp_fit = exp_func(x_values / 20, *popt)
                    exp_fit_normalized = exp_fit / np.sum(exp_fit)
                    plt.plot(x_values, exp_fit_normalized, 'k--')
                else:
                    plt.plot(x_values, exp_func(x_values / 20, *popt), 'k--')

        plt.xlabel('Time [ms]')
        plt.ylabel(ylabel)
        density_str = 'Density' if density_setting else 'Counts'
        plt.title(f'Dwell Time Distribution ({filter_key.capitalize()}, {density_str})')
        plt.xlim(0, 400)
        plt.legend()
        plt.tight_layout()
        density_label = 'density_true' if density_setting else 'density_false'
        plt.savefig(f'graphs/export_time_{filter_key}_all_labels_{density_label}.svg', format='svg')
        plt.show()

# Create a DataFrame for the dwell times
df_dwell_times = pd.DataFrame(dwell_time_data)

# Pivot the table to have compartments as columns
df_pivot = df_dwell_times.pivot(index='Label', columns='Compartment', values='Dwell Time (ms)').reset_index()

# Display the table
print("\nDwell Times Table:")
print(df_pivot)

# Optionally, save the table to a CSV file
df_pivot.to_csv('dwell_times_table.csv', index=False)