"""
Title: Analysis of Gap Closing Times and Transport Events in Yeast Cell Tracking Data

Description:
This script analyzes yeast cell tracking data to investigate gap closing times, transport events, and the spatial-temporal characteristics of nuclear import and export events. The main objectives are:

1. **Gap Closing Time Analysis**:
   - Identifies interpolated segments in tracking data by detecting linear sequences in the particle positions.
   - Computes the lengths of these interpolated sequences (gap closing times).
   - Fits an exponential decay model to the histogram of gap closing times to characterize the distribution.

2. **Transport Event Analysis**:
   - Processes tracking data to identify successful and unsuccessful import and export events.
   - Merges data to find common cells/movies where both import and export events occur.
   - Calculates the Euclidean distances between nuclear envelope (NE) crossings for import and export events.
   - Analyzes the time differences between import and export NE crossings.

3. **Visualization**:
   - Plots histograms of gap closing times with exponential fit overlays.
   - Plots histograms of distances between import and export NE crossings.
   - Plots histograms of time differences between import and export NE crossings.

Functions:
- `load_and_process_tracks(cfg)`: Loads and processes tracking data, swapping specific column values.
- `load_other_data(cfg)`: Loads additional data files required for analysis.
- `process_for_config(cfg, make_velocity, success_threshold, labelss)`: Main function that performs data processing and analysis for a given configuration.

Usage:
- Update the `base_cfg` dictionary with the correct paths to your data files.
- Define the list of strains (`digits_list`) and their corresponding labels.
- Set the success thresholds for transport events in the `success_threshold` list.
- Run the script to perform the analysis and generate plots.
- The script will output results to the console and save figures in SVG format.

Dependencies:
- numpy
- pandas
- matplotlib
- seaborn
- pickle
- scipy
- tqdm
- scienceplots
- Custom modules:
  - `extract_results` (should contain `yeast_extractresults`)

Notes:
- Ensure that all required data files are available and paths are correctly set.
- The script assumes a specific directory structure and naming convention for data files.
- The `yeast_extractresults` class should be properly implemented in the `extract_results` module.
- Random seeds are not set in this script; results may vary between runs.

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
import tqdm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import splprep, splev, UnivariateSpline
from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import convolve
from scipy.stats import expon
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

def process_for_config(cfg, make_velocity = True,succes_threshold=[-100,100],labelss=''):

    # Load and process data
    max_number_intersections = 3
    all_tracks = load_and_process_tracks(cfg)
    file_lookup, ne_lookup, ne_data = load_other_data(cfg)
    unique_movies = np.unique(file_lookup['filename'].values)

    # Drop duplicates based on 'filename' and 'NE'
    unique_combinations = ne_lookup.drop_duplicates(subset=['filename', 'NE'])

    # Count the unique combinations
    print('strain = ', cfg['digits'])
    number_of_unique_combinations = unique_combinations.shape[0]
    print("Number of movies:", len(unique_movies))
    print("Number of cells:", number_of_unique_combinations)
    # Group data by 'id' and count the number of frames for each track
    track_lengths = all_tracks.groupby('id').size()

    # Calculate the number of unique tracks
    number_of_tracks = track_lengths.count()

    # Calculate the average length of each track
    average_track_length = track_lengths.mean()

    print("Number of unique tracks:", number_of_tracks)
    print("Average length of tracks (in frames):", average_track_length)

    # Initialize yeast extract results object
    yr = yeast_extractresults(cfg)
    yr.all_tracks = all_tracks
    yr.lookup = file_lookup
    yr.nelookup = ne_lookup
    yr.nedata = ne_data
    all_tracks = all_tracks.sort_values(by=['id', 'frame'])

    # Convert columns to numpy arrays for faster computation
    x = all_tracks['x'].to_numpy()
    ids = all_tracks['id'].to_numpy()

    # Calculate the difference in 'x' values
    x_diff = np.diff(x)
    id_diff = np.diff(ids)

    # Initialize a tolerance for detecting linear segments
    tolerance = 1e-3

    # Find indices where the difference in 'x' is approximately constant and within the same particle
    interpolated_mask = np.isclose(x_diff[:-1], x_diff[1:], atol=tolerance) & (id_diff[:-1] == 0)

    #interpolated_mask = np.isclose(x_diff[:-1], x_diff[1:], atol=tolerance)
    # Get the start and end indices of contiguous True sequences
    change_indices = np.diff(np.concatenate(([0], interpolated_mask.view(np.int8), [0])))
    starts = np.where(change_indices == 1)[0]
    ends = np.where(change_indices == -1)[0]

    # Calculate lengths of these sequences
    sequence_lengths = ends - starts

    # Define the exponential function for fitting
    def exponential_func(x, a, b):
        return a * np.exp(-b * x)

    # Generate the histogram data
    hist_counts, bin_edges = np.histogram(sequence_lengths, bins=np.arange(0.5, 11.5, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Center of each bin

    # Use curve_fit to fit the exponential function to the histogram data
    params, covariance = curve_fit(exponential_func, bin_centers, hist_counts, p0=(1, 1))

    # Extract the fitted parameters
    a_fitted, b_fitted = params
    a_fitted_sci = f"{a_fitted:.1e}"

    # Generate x values for plotting the fitted curve
    x_fit = np.linspace(1, 10, 100)
    y_fit = exponential_func(x_fit, a_fitted, b_fitted)

    # Plotting the histogram of sequence lengths and the fitted exponential curve
    plt.figure(figsize=(2.5, 2.5), dpi=300)
    #plt.hist(sequence_lengths, bins=np.arange(0.5, 11.5, 1), edgecolor='black', label='Data', alpha=0.7)
    plt.plot(x_fit, y_fit, 'r-',
             label=f'Exponential fit:\n$y = {a_fitted_sci} \cdot e^{{-{b_fitted:.2f}x}}$')

    plt.hist(sequence_lengths, bins=np.arange(0.5, 11.5, 1), edgecolor='black')  # Centering the bins
    plt.legend()
    plt.xlabel('Gap closing time [frames]')
    plt.ylabel('Counts')
    plt.title(labelss)
    plt.tight_layout(pad=0.4)
    plt.savefig('gapclosinghistogram_'+labelss+'.svg', format='svg' )
    plt.show()

    max(sequence_lengths)

    # Perform your analysis and plotting here
    # For example:
    transport_events_gfa, success_export_df, unsuccess_export_df, success_import_df, unsuccess_import_df,\
        retrogate_export_df, retrogate_import_df,sequences \
        = yr.find_transports_v2(thressuccess_transport=succes_threshold, thres_transport=[-1, 1],
                                max_number_intersections=max_number_intersections, use_spline=True)

    unique_crossings = len(np.unique(transport_events_gfa['id'].values))
    unique_exports = len(np.unique(success_export_df['id'].values))
    unique_imports = len(np.unique(success_import_df['id'].values))
    merged_df1 = pd.merge(success_export_df, file_lookup, on='id')
    merged_df2 = pd.merge(success_import_df, file_lookup, on='id')

    # Step 1: Get the unique combinations of 'filename' and 'NE_y' in merged_df1
    unique_combinations_df1 = merged_df1[['filename', 'NE_y']].drop_duplicates()

    # Step 2: Get the unique combinations of 'filename' and 'NE_y' in merged_df2
    unique_combinations_df2 = merged_df2[['filename', 'NE_y']].drop_duplicates()

    # Step 3: Check how many unique combinations in merged_df1 are also present in merged_df2
    common_combinations = pd.merge(unique_combinations_df1, unique_combinations_df2, on=['filename', 'NE_y'])

    # Step 4: Count the number of unique combinations in both dataframes
    total_unique_df1 = len(unique_combinations_df1)
    total_unique_df2 = len(unique_combinations_df2)
    total_common_combinations = len(common_combinations)

    # Filter intersect points
    intersect_df1 = merged_df1[merged_df1['intersect_spline'] == 1]
    intersect_df2 = merged_df2[merged_df2['intersect_spline'] == 1]

    # Merge these filtered dataframes on 'filename' and 'NE_y' to find matching intersect points
    intersect_merged = pd.merge(intersect_df1, intersect_df2, on=['filename', 'NE_y'], suffixes=('_df1', '_df2'))

    # Calculate the distance between intersect points
    def calculate_distance(row):
        return np.sqrt((row['x_df1'] - row['x_df2']) ** 2 + (row['y_df1'] - row['y_df2']) ** 2)

    # Apply the distance calculation
    intersect_merged['distance'] = intersect_merged.apply(calculate_distance, axis=1)
    plt.hist( np.array(intersect_merged['distance'])*128,bins=40 )
    plt.xlim([0,20*128])
    plt.xlabel('Euclidean distance between crossings\n import and export traces [nm]')
    plt.ylabel('Counts')
    plt.title(labelss)
    plt.tight_layout(pad=0.4)
    plt.savefig('Distance_between_crossings' + labelss + '.svg', format='svg')
    plt.show()

    # Initialize the list to store shortest time differences
    shortest_time_diffs = []

    # Iterate through intersect_df2 (export traces) and find the closest increasing frame in intersect_df1 (import traces)
    i = 0  # Pointer for intersect_df1
    for j, row_df2 in intersect_df2.iterrows():
        # Find the smallest frame difference that is increasing
        while i < len(intersect_df1):
            row_df1 = intersect_df1.iloc[i]
            frame_diff = row_df2['frame'] - row_df1['frame']

            if frame_diff >= 0:  # Ensure the frame difference is non-negative (increasing)
                shortest_time_diffs.append(frame_diff)
                i += 1  # Move to the next point in intersect_df1
                break
            i += 1

    # Plotting the histogram of shortest time differences
    plt.hist(np.array(shortest_time_diffs)*20, bins=20)  # Adjust bins if needed
    plt.xlabel('Time between export and \n import NE crossings [ms]')
    plt.ylabel('Counts')
    plt.title(labelss)
    plt.tight_layout(pad=0.4)
    plt.savefig('Time_between_crossings' + labelss + '.svg', format='svg')

    plt.show()
    # Output the results
    print("Number of exports [binding to binding]", unique_exports)
    print("Number of imports [binding to binding]", unique_imports)
    print(f"Number of cells/movies in which there are exports: {total_unique_df1}")
    print(f"Number of cells/movies in which there are imports: {total_unique_df2}")
    print(f"Number of cells/movies that match: {total_common_combinations}")
    print("Number of unique tracks that crossed N", unique_crossings)



    positive_count = 0
    negative_count = 0

    for df in sequences:
        # Check the first value in the 'dist_cor_spline' column
        if df['dist_cor_spline'].iloc[0] > 0:
            positive_count += 1
        elif df['dist_cor_spline'].iloc[0] < 0:
            negative_count += 1

    print("Number of failed export events:", positive_count)
    print("Number of failed import events:", negative_count)

    # Continue with your existing analysis and plotting...
    distance_total,num_bins,x_fit, yfit = yr.plot_diffusion_vs_distance([transport_events_gfa,], xlim=[-250, 250],
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
digits_list = [823, 820,822]
all_results = []
succes_threshold = [[-74,98],[-89,78],[-68,77]]
label = ['GFA1', 'MYO2', 'TRA1']
for itter, digits in enumerate(digits_list):
    # Update cfg paths for the current set of digits
    cfg = base_cfg.copy()  # Start with the base configuration
    cfg.update({
        'digits': str(digits),
        'all_tracks_path': f'../../results_tracks_gapclosing10/all_tracks_{digits}.csv',
        'ne_lookup_path': f'../../results_tracks_gapclosing10/ne_lookup{digits}.csv',
        'file_lookup_path': f'../../results_tracks_gapclosing10/file_lookup_{digits}.csv',
        'nedata_path': f'../../results_tracks_gapclosing10/nedata{digits}.pkl',
    })

    # Process data for this configuration
    result= process_for_config(cfg, make_velocity = False,succes_threshold=succes_threshold[itter], labelss =label[itter] )
    all_results.append(result)

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

