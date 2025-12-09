"""
Title: Yeast Tracking Data Analysis Script for Localization Density and Failed Transport Events

Description:
This script analyzes yeast tracking data to study localization density distributions and failed transport events within yeast cells. It processes tracking data to identify transport events, computes localization densities using Gaussian Kernel Density Estimation (KDE), and fits exponential decay models to time data of failed imports and exports. The script generates heatmaps and plots to visualize the results.

Functions:
- load_and_process_tracks(cfg): Loads and processes tracking data from CSV files. It modifies specified columns by swapping values to correct the data format.
- load_other_data(cfg): Loads additional data required for the analysis, including file lookups and nuclear envelope (NE) data.
- process_for_config(cfg, make_velocity=True): Processes data for a given configuration, performs analysis, and generates plots.
- preprocess_data(data): Preprocesses time data for exponential fitting by counting cumulative events over time.
- exp_func(x, a, b): Defines the exponential function used for curve fitting.
- plot_histogram_and_fit(data, save_title, col='red', xlabel=''): Plots histogram of cumulative counts and fits the exponential decay model.

Main Workflow:
1. Sets up the base configuration.
2. Defines functions for data loading and processing.
3. Processes data using the `process_for_config` function for a given configuration.
4. Identifies transport events and sequences.
5. Transforms sequences to extract RNA positions and distances.
6. Fits exponential decay models to time data for failed imports and exports.
7. Filters RNA data based on a distance threshold.
8. Computes and plots localization density using Gaussian KDE for nuclear and cytoplasmic RNA.
9. Generates heatmaps and saves the figures.

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
- tqdm

Usage:
- Ensure that the required data files are located in the paths specified in the configuration.
- Adjust the `digits_list` and other parameters as needed for your data.
- Run the script to perform the analysis and generate the plots.

Notes:
- The script uses a custom module `yeast_extractresults` which should be available in the Python path.
- The plotting style is set to 'science' using the `scienceplots` package.
- The code includes placeholders and commented sections that may need to be completed or adjusted for different datasets.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import gaussian_kde
import tqdm
from scipy.stats import stats
from matplotlib.animation import FuncAnimation
from extract_results import yeast_extractresults
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial import Delaunay
import scienceplots
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import splprep, splev, UnivariateSpline
from scipy.interpolate import UnivariateSpline
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
from cycler import cycler
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

# Genetic Algorithm setup

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

def process_for_config(cfg, make_velocity = True):
    plotting_data = {
        'transport': None,
        'export': None,
        'import': None
    }
    plotting_data_distri = {
        'transport': None,
        'export': None,
        'import': None
    }
    # Load and process data
    max_number_intersections = 3
    all_tracks = load_and_process_tracks(cfg)
    file_lookup, ne_lookup, ne_data = load_other_data(cfg)
    test = file_lookup[file_lookup['id'] == 3414]
    test2 = all_tracks[all_tracks['id']==3414]
    # Initialize yeast extract results object
    yr = yeast_extractresults(cfg)
    yr.all_tracks = all_tracks
    yr.lookup = file_lookup
    yr.nelookup = ne_lookup
    yr.nedata = ne_data

    # Perform your analysis and plotting here
    # For example:



    transport_events, success_export_df, unsuccess_export_df, success_import_df, unsuccess_import_df, retrogate_export_df, retrogate_import_df, sequences \
        = yr.find_transports_v2(thressuccess_transport=[-250, 250], thres_transport=[-84, 81],
                                max_number_intersections=max_number_intersections, use_spline=True)

    nuc_rna, dist_between_nuc,cyto_rna,dist_between_cyto,time_nuc, time_cyto = yr.transform_sequences(sequences)
    from scipy.optimize import curve_fit

    def preprocess_data(data):
        max_time = 40
        # Create an array to store the modified counts
        processed_counts = np.zeros(max_time)
        # Increment counts for all bins from 0 to the value at each data point
        for time in data:
            if time < max_time:
                processed_counts[:int(time) + 1] += 1
        return processed_counts

    # Define the exponential function
    def exp_func(x, a, b):
        return a * np.exp(-b * x)

    def plot_histogram_and_fit(data, save_title,col='red',xlabel = ''):
        plt.figure(figsize=(2.5,2.5))
        processed_counts = preprocess_data(data)
        x_values = np.arange(len(processed_counts))
        plt.bar(x_values*20, processed_counts, width=20, edgecolor=col,color=col)

        # Fit the exponential model
        popt, pcov = curve_fit(exp_func, x_values, processed_counts,
                               p0=(max(processed_counts), 0.1) )
        perr = np.sqrt(np.diag(pcov))
        tau = 1/popt[1]
        tau_ms = tau*20
        rel_error = perr[1]/popt[1]
        # Plot the fitted curve
        plt.plot(x_values*20, exp_func(x_values, *popt), 'k--')
        plt.xlabel(xlabel)
        plt.ylabel('Counts')
        plt.title(f"b = {tau_ms:.2f} ± {tau_ms*rel_error:.2f}")
        #plt.legend()
        plt.tight_layout()
        plt.savefig('graphs/'+save_title+'.svg',format='svg')
        plt.show()

    # Plot for time_nuc
    plot_histogram_and_fit(time_nuc, 'time_fail_nuc'+cfg['digits'], col = "blue", xlabel=r'Time for failed import')

    # Plot for time_cyto
    plot_histogram_and_fit(time_cyto, 'time_fail_cyto'+cfg['digits'], col = "#DAA520", xlabel=r'Time for failed export')


    threshold = 2 # Replace YOUR_THRESHOLD_VALUE with your specific threshold
    # Filter nuc_rna and cyto_rna based on the threshold condition
    filtered_nuc_rna = [nuc for nuc, dist in zip(nuc_rna, dist_between_nuc) if dist < threshold]
    filtered_cyto_rna = [cyto for cyto, dist in zip(cyto_rna, dist_between_cyto) if dist < threshold]
    filtered_nuc_rna_np = np.concatenate(filtered_nuc_rna,axis=-1).T*128
    filtered_cyto_rna_np = np.concatenate(filtered_cyto_rna,axis=-1).T*128


    # Label and combine data
    nuc_data = np.hstack((filtered_nuc_rna_np, np.full((filtered_nuc_rna_np.shape[0], 1), 'nuc')))
    cyto_data = np.hstack((filtered_cyto_rna_np, np.full((filtered_cyto_rna_np.shape[0], 1), 'cyto')))

    # Combine into a single array
    combined_data = np.vstack((nuc_data, cyto_data))

    # Create a DataFrame
    df = pd.DataFrame(combined_data, columns=['x', 'y', 'type'])

    # Convert x and y to numeric types as they might be recognized as objects initially
    df['x'] = pd.to_numeric(df['x'])
    df['y'] = pd.to_numeric(df['y'])


    # Create a figure with two subplots, sharing the y-axis
    fig, axes = plt.subplots(1, 2, figsize=(3.6, 2), sharey=True)
    fig.subplots_adjust(wspace=0.01)  # Adjust the width space

    # Plot types
    types = ['nuc', 'cyto']
    xlims = [(-300, 40), (-40, 300)]  # Custom x-limits for each plot
    contour_list = []
    bootstrap_num=1
    total_area_list = []
    for ax, plot_type, xlim in zip(axes, types, xlims):
        data = df[df['type'] == plot_type][['x', 'y']]
        values = data.values.T

        # Gaussian KDE
        kde = gaussian_kde(values, bw_method=0.15)  # Reduced smoothing
        x_grid = np.linspace(min(xlim), max(xlim), 100)
        y_grid = np.linspace(-250, 250, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_coords = np.vstack([X.ravel(), Y.ravel()])

        for boots in range(bootstrap_num):
            # Number of columns in the original array
            num_columns = values.shape[1]

            # Randomly choose column indices with replacement
            bootstrap_indices = np.random.choice(num_columns, size=num_columns, replace=True)

            # Select columns based on the chosen indices
            bootstrap_sample = values[:, bootstrap_indices]
            kde = gaussian_kde(bootstrap_sample, bw_method=0.15)  # Re
            Z = kde(grid_coords).reshape(X.shape)


            # Cumulative density
            sorted_density = np.sort(Z.ravel())
            cumulative_density = np.cumsum(sorted_density)
            cumulative_density /= max(cumulative_density)  # Normalize
            density_68 = np.interp(0.32, cumulative_density, sorted_density)
            area_covered = np.sum(Z >= density_68)
            total_area = (x_grid[1] - x_grid[0]) * (
                    y_grid[1] - y_grid[0]) * area_covered  # Grid cell area * number of cells
            total_area_list.append(total_area)

        kde = gaussian_kde(values, bw_method=0.15)  # Reduced smoothing
        # Dense grid
        x_grid = np.linspace(min(xlim), max(xlim), 1000)
        y_grid = np.linspace(-250, 250, 1000)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_coords = np.vstack([X.ravel(), Y.ravel()])

        # KDE over grid
        Z = kde(grid_coords).reshape(X.shape)

        # Cumulative density
        sorted_density = np.sort(Z.ravel())
        cumulative_density = np.cumsum(sorted_density)
        cumulative_density /= max(cumulative_density) # Normalize

        # CDF 10% increments
        cdf_levels = np.linspace(0, 1, 11)
        density_levels_at_cdf = np.interp(cdf_levels, cumulative_density, sorted_density)


        print("Density level at 64%:", density_68)
        print("Area covered by the first 64% of the KDE:", total_area)
        # Define your custom colormap once, used for both plots
        if plot_type == 'nuc':
            colormap = LinearSegmentedColormap.from_list("white_to_blue", ["white", "blue"], N=256)
            nuc_area = np.mean(np.array(total_area_list))
            nuc_std_area = np.std(np.array(total_area_list))
        else:
            dark_yellow = "#DAA520"  # This is a gold-like color, you can adjust the hex code to any dark yellow you prefer

            # Create the colormap from white to the specified dark yellow
            colormap = LinearSegmentedColormap.from_list("white_to_dark_yellow", ["white", dark_yellow],
                                                                     N=256)
            cyto_area = np.mean(np.array(total_area_list))
            cyto_std_area = np.std(np.array(total_area_list))

        # Plot with custom colormap
        contour = ax.contourf(X, Y, Z, levels=density_levels_at_cdf, extend='max', cmap=colormap)
        contour_lines = ax.contour(X, Y, Z, levels=[density_68], colors='black')  # Black contour line at 68%

        contour_list.append(contour)
        # Adjust plot
        ax.set_xlim(xlim)
        ax.set_ylim(-250, 250)
        ax.set_xlabel('x [nm]')
        if ax == axes[0]:  # Only add y label to first subplot
            ax.set_ylabel('y [nm]')

    # Add color bars for each subplot
    # make both at the right of the whole figure
    # Add color bars for each subplot, both at the right of the whole figure

    #cbar1.set_label('Density Percentile')
    ticks = np.linspace(0.1, 1, 10)  # Generates values from 0.1 to 1 in 10 steps

    cbar2 = fig.colorbar(contour_list[1], ax=axes.ravel().tolist())
    cbar2.set_label(r'Localization density [\%]')
    mn, mx = contour_list[1].get_clim()  # Retrieve the color limits from the contour plot
    cbar2.set_ticks([])  # Set 11 ticks to include both ends of the range
    #cbar2.set_ticklabels(['0%','10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])

    # test = np.linspace(mn,mx,11)# colorbar max value
    # cbar2.set_ticks(np.linspace(mn,mx,11))
    # cbar2.set_ticklabels([])

    cbar1 = fig.colorbar(contour_list[0], ax=axes.ravel().tolist()                         )

    cbar1.set_ticks([])  # This hides the ticks
    plt.title('nuc_area ='+str(np.round(nuc_area,0))+'+-' +str(np.round(nuc_std_area,0))  + '\ncyto_area='+str(np.round(cyto_area,0))+'+-' +str(np.round(cyto_std_area,0)) ,fontsize=6)
    plt.tight_layout()

    plt.savefig('graphs/heatmap'+cfg['digits']+'.svg', format='svg')
    plt.show()











    ##############################################################################33

    # fig = plt.figure(figsize=(3,3),dpi=300)
    # using_mpl_scatter_density(fig, filtered_cyto_rna_np[:,0], filtered_cyto_rna_np[:,1])
    # plt.xlim(-100,300)
    # plt.ylim(-250,250)
    # plt.xlabel('x [nm]')
    # plt.ylabel('y [nm]')
    # plt.title('Cytoplasmic RNA ' + cfg['digits'])
    # plt.savefig('./test_figs/cyto' + cfg['digits'] + 'density.png', dpi=400)
    # plt.show()





    results = {
        'transport_events': transport_events,
        'success_export_df': success_export_df,
        'unsuccess_export_df': unsuccess_export_df,
        'success_import_df': success_import_df,
        'unsuccess_import_df': unsuccess_import_df,
        'retrogate_export_df': retrogate_export_df,
        'retrogate_import_df': retrogate_import_df,
        'plotting_data_distri': plotting_data_distri,
        'plotting_data': plotting_data,
        'sequences': sequences,
        'filtered_nuc_rna_np':
            filtered_nuc_rna_np,
        'filtered_cyto_rna_np':
            filtered_cyto_rna_np,
        # Add other variables as needed
    }
    return results
# List of digit sets to loop over
digits_list = ['823','820','822']
#digits_list = ['822']
all_results = []
custom_color_cycle = [default_colors[i] for i in [0, 0, 1, 1,  2, 2]]

for digits in digits_list:
    # Update cfg paths for the current set of digits
    cfg = base_cfg.copy()  # Start with the base configuration
    cfg.update({
        'digits': str(digits),
        'all_tracks_path': '../../results_tracks_gapclosing10/all_tracks_'+digits+'.csv',
        'ne_lookup_path': f'../../results_tracks_gapclosing10/ne_lookup'+digits+'.csv',
        'file_lookup_path': f'../../results_tracks_gapclosing10/file_lookup_'+digits+'.csv',
        'nedata_path': f'../../results_tracks_gapclosing10/nedata'+digits+'.pkl',
    })

    # Process data for this configuration
    result= process_for_config(cfg, make_velocity = False)
    all_results.append(result)
