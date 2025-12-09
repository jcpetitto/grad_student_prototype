"""
Title: Yeast Tracking Data Analysis Script

Description:
This script performs data analysis on yeast tracking simulation results. It processes tracking data to analyze transport events, localization probabilities, and binding interactions within yeast cells. The script generates various plots to visualize the results of the analysis.

Functions:
- load_and_process_tracks(cfg): Loads and processes track data from CSV files. It modifies specified columns by swapping values to correct the data format.
- load_other_data(cfg): Loads additional data required for the analysis, including file lookups and nuclear envelope (NE) data.
- process_for_config(cfg, make_velocity=True, succes_threshold=120): Processes data for a given configuration, performs analysis, and returns a dictionary containing the results.

Main Workflow:
1. Sets up the base configuration and result folder list.
2. Iterates over a list of result folders and digit identifiers to process data for each configuration.
3. Processes the data using the `process_for_config` function.
4. Collects results from each configuration.
5. Generates plots for:
   - Localization probabilities.
   - Binding interactions vs. transport thresholds.
   - Threshold transport events.
6. Saves the generated plots as SVG files.
7. Copies the SVG files to a designated folder for further use.

Dependencies:
- numpy
- pandas
- matplotlib
- seaborn
- pickle
- scipy
- extract_results (custom module)
- scienceplots
- cycler
- glob
- os
- shutil

Usage:
- Ensure that the required data files are located in the paths specified in the configuration.
- Run the script to perform the analysis and generate the plots.

Notes:
- The script uses a custom module `yeast_extractresults` which should be available in the Python path.
- The plotting style is set to 'science' using the `scienceplots` package.
- The script is tailored to specific datasets and configurations; modifications may be required for different datasets.
"""


import pandas as pd

import pickle

from extract_results import yeast_extractresults
import scienceplots

import numpy as np
import matplotlib.pyplot as plt

# Apply a scientific plotting style
plt.style.use('science')
result_folder_list = ['results_tracks_gapclosing2','results_tracks_gapclosing5','results_tracks_gapclosing10',
                      'results_tracks_gapclosing15']
result_folder_list = ['results_tracks_gapclosing10']
for result_folder in result_folder_list:
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

    def process_for_config(cfg, make_velocity = True,succes_threshold=120):

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
        thresrange = np.linspace(49,140, 20)
        mu1_list = []
        mu2_list = []
        error1_list = []
        error2_list =[]
        transport_events_gfa, success_export_df, unsuccess_export_df, success_import_df, unsuccess_import_df, retrogate_export_df, retrogate_import_df, _ \
            = yr.find_transports_v2(thressuccess_transport=[-succes_threshold, succes_threshold], thres_transport=[-thresrange[0], thresrange[0]],
                                    max_number_intersections=max_number_intersections, use_spline=True)
        for qq in thresrange:
            # Continue with your existing analysis and plotting...
            distance_total,num_bins,x_fit, yfit,mu1,error1, mu2,error2 = yr.plot_diffusion_vs_distance([transport_events_gfa,],
                                                                                                       xlim=[-250, 250],
                                          num_bins=20, label_list=[cfg['digits']],ylim=[0,0.0035],
                                                                           return_params=True,threshold=qq)
            mu1_list.append(mu1)
            error1_list.append(error1)
            mu2_list.append(mu2)
            error2_list.append(error2)


        sample_distances = np.arange(-250, 250, 4)
        if make_velocity:
            mean_velocities, mean_velocities_proj,bin_centers_velocity = yr.plot_velocity([transport_events_gfa],
                                                                      sample_distances=sample_distances, upsample_factor=100,
                                                                                          name_list=[cfg['digits']])
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
            'mu1_list': mu1_list,
            'error1_list': error1_list,
            'error2_list': error2_list,
            'mu2_list': mu2_list,
            'thresrange':thresrange,
            # Add other variables as needed
        }
        return results
    # List of digit sets to loop over
    digits_list = [823, 820,822]
    all_results = []
    succes_threshold = np.inf

    for digits in digits_list:
        # Update cfg paths for the current set of digits
        cfg = base_cfg.copy()  # Start with the base configuration
        cfg.update({
            'digits': str(digits),
            'all_tracks_path': '../../'+result_folder+f'/all_tracks_{digits}.csv',
            'ne_lookup_path': '../../'+result_folder+f'/ne_lookup{digits}.csv',
            'file_lookup_path': '../../'+result_folder+f'/file_lookup_{digits}.csv',
            'nedata_path': '../../'+result_folder+f'/nedata{digits}.pkl',
        })

        # Process data for this configuration
        result= process_for_config(cfg, make_velocity = False,succes_threshold=succes_threshold)
        all_results.append(result)
    label = ['GFA1', 'MYO2', 'TRA1']
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

    custom_color_cycle = [CB_color_cycle[i] for i in [0, 0, 1, 1,  2, 2]]


    fig, ax = plt.subplots(dpi=400, figsize=(2.2, 1.5))
    fig2, ax2 = plt.subplots(dpi=400, figsize=(2.2, 1.5))

    ax2.set_prop_cycle(cycler('color', custom_color_cycle))
    custom_color_cycle = [CB_color_cycle[i] for i in [0, 1,  2]]
    ax.set_prop_cycle(cycler('color', custom_color_cycle))
    label = ['GFA1']#, 'MYO2', 'TRA1']
    for i, result in enumerate(all_results):
        if i == 0:

        # Plot mean velocities and their projections
            ax.plot(result['x_fit'], result['y_fit'], label=label[i], linestyle='-',linewidth=2)
            ax2.plot(result['x_fit'], result['y_fit'], label=label[i], linestyle='-',linewidth=2)
            ax2.hist(result['distance_total'], bins = result['num_bins'],range=[-250,250], density=True, alpha=0.3)
    ax.set_xlabel('Distance to membrane [nm]')
    ax.set_ylabel(r'Prob. loc.')
    ax2.set_xlabel('Distance to membrane [nm]')
    ax2.set_ylabel(r'Prob. loc.')
    fig.legend()

    fig.tight_layout(pad=0.2)
    fig.savefig('graphs/localzi.svg',format='svg')
    fig.show()
    fig2.tight_layout(pad=0.2)
    fig2.savefig(r'graphs/localizationprob_onlygfa.svg',format='svg')
    fig2.legend()
    fig2.show()

    # Create two subplots with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=400, figsize=(2,2.5), sharex=True)

    # Custom color cycle
    custom_color_cycle = [CB_color_cycle[i] for i in [0, 0, 1, 1, 2, 2,]]
    labels = ['GFA1', 'MYO2', 'TRA1']

    # First plot
    for i, result in enumerate(all_results):
        thresrange = np.array(result['thresrange'])
        mu1_list = np.array(result['mu1_list'])
        error1_list = np.array(result['error1_list'])

        ax1.plot(thresrange, mu1_list, '-', label=labels[i])
        ax1.fill_between(thresrange, mu1_list - error1_list, mu1_list + error1_list, alpha=0.3)

    ax1.set_ylabel(r'$\mu_\text{binding, nuc}$ [nm]')
    #ax1.legend()

    # Second plot
    for i, result in enumerate(all_results):
        thresrange = np.array(result['thresrange'])
        mu1_list = -np.array(result['mu1_list']) - thresrange
        error1_list = np.array(result['error1_list'])

        ax2.plot(thresrange, mu1_list, '-', label=labels[i], linewidth=1)
        ax2.fill_between(thresrange, mu1_list - error1_list, mu1_list + error1_list, alpha=0.3)

    ax2.axhline(0, color='grey', linestyle='--')
    ax2.set_xlabel(r'$T_\text{transport} $[nm]')
    ax2.set_ylabel(r'$-\mu_\text{binding, nuc}\\-T_\text{transport} $ [nm]')
    ax2.legend()

    fig.tight_layout(pad=0.2)
    fig.savefig('graphs/threshold_binding_nuc.svg', format='svg')
    fig.show()


    # Create two subplots with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=400, figsize=(2, 2.5), sharex=True)

    # Custom color cycle
    custom_color_cycle = [CB_color_cycle[i] for i in [0, 0, 1, 1, 2, 2,]]
    labels = ['GFA1', 'MYO2', 'TRA1']
    # First plot
    for i, result in enumerate(all_results):
        thresrange = np.array(result['thresrange'])
        mu2_list = np.array(result['mu2_list'])
        error2_list = np.array(result['error2_list'])

        ax1.plot(thresrange, mu2_list, '-', label=labels[i])
        ax1.fill_between(thresrange, mu2_list - error2_list, mu2_list + error2_list, alpha=0.3)

    ax1.set_ylabel(r'$\mu_\text{binding, cyto}$ [nm]')
    #ax1.legend()

    # Second plot
    for i, result in enumerate(all_results):
        thresrange = np.array(result['thresrange'])
        mu2_list = np.array(result['mu2_list']) - thresrange
        error2_list = np.array(result['error2_list'])

        ax2.plot(thresrange, mu2_list, '-', label=labels[i], linewidth=1)
        ax2.fill_between(thresrange, mu2_list - error2_list, mu2_list + error2_list, alpha=0.3)

    ax2.axhline(0, color='grey', linestyle='--')
    ax2.set_xlabel(r'$T_\text{transport} $[nm]')
    ax2.set_ylabel(r'$\mu_\text{binding, cyto}\\-T_\text{transport} $ [nm]')
    ax2.legend()

    fig.tight_layout(pad=0.2)
    fig.savefig('graphs/threshold_binding_cyto.svg', format='svg')
    fig.show()

    # Create subplots with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=400, figsize=(2, 2.5), sharex=True)
    custom_color_cycle = [CB_color_cycle[i] for i in [0, 1,  2]]
    ax1.set_prop_cycle(cycler('color', custom_color_cycle))
    ax2.set_prop_cycle(cycler('color', custom_color_cycle))
    labels = ['GFA1', 'MYO2', 'TRA1']


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

    ax1.set_ylabel(r'$ -\mu_\text{binding, nuc} \\- T_\text{transport} $ [nm]')
    ax2.axhline(0, color='grey', linestyle='--')
    ax1.axhline(0, color='grey', linestyle='--')
    ax2.set_xlabel(r'$T_\text{transport} $[nm]')
    ax2.set_ylabel(r'$ \mu_\text{binding, cyto} \\- T_\text{transport} $ [nm]')
    ax2.legend()

    fig.tight_layout(pad=0.2)
    fig.savefig('graphs/threshold_transportevent.svg', format='svg')
    fig.show()
    plt.close('all')


