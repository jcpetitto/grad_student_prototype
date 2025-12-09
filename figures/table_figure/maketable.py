"""
Title: Analysis of Yeast Cell Transport Events with Variable Intersection Thresholds

Description:

This script performs a comprehensive analysis of yeast cell transport events, focusing on how varying the maximum number of intersections affects the results. The key objectives of the script are:

1. **Data Loading and Preprocessing**:
   - Load tracking data and related information for different yeast strains.
   - Process the data to prepare it for analysis, including swapping specific column values.

2. **Transport Event Analysis**:
   - Use a range of maximum intersection thresholds to explore how the number of allowed intersections impacts the identification of transport events.
   - Identify successful and unsuccessful import and export events based on specified thresholds.
   - Calculate various statistics, such as the number of unique tracks, cells, and movies involved in the events.

3. **Visualization**:
   - Generate histograms and plots to visualize the distribution of gap closing times, distances between crossings, and time differences between import and export events.
   - Save the plots with filenames that include labels indicating the specific configurations used.

4. **Result Compilation**:
   - Collect results into a summary DataFrame.
   - Save the summary results to a CSV file for further analysis or reporting.

Dependencies:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `pickle`
- `scipy`
- `tqdm`
- `scienceplots`
- Custom module: `extract_results` (should contain `yeast_extractresults`)

Usage:

- Update the `base_cfg` dictionary with the correct paths to your data files.
- Ensure that the custom module `extract_results` is available and properly implemented.
- Adjust the lists `digits_list`, `success_thresholds`, and `labels` to match the yeast strains you are analyzing.
- Run the script to perform the analysis and generate plots.
- The script will save the figures and a summary CSV file in the specified directories.

Notes:

- The script uses the `science` plotting style for better visualization.
- The analysis explores different ranges for the maximum number of intersections to understand their impact on the results.
- Ensure that all required data files are available and paths are correctly set in the configuration.

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
from scipy.optimize import curve_fit



from scipy.signal import convolve
from scipy.stats import expon
from cycler import cycler
from scipy.interpolate import splrep, BSpline

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

def process_for_config(cfg, max_number_intersections_range, make_velocity=True, success_threshold=[-100, 100], label=''):
    min_intersections, max_intersections = max_number_intersections_range

    # Load and process data
    all_tracks = load_and_process_tracks(cfg)
    file_lookup, ne_lookup, ne_data = load_other_data(cfg)
    unique_movies = np.unique(file_lookup['filename'].values)

    # Drop duplicates based on 'filename' and 'NE'
    unique_combinations = ne_lookup.drop_duplicates(subset=['filename', 'NE'])

    # Count the unique combinations
    print('Strain =', cfg['digits'])
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

    # Perform your analysis and plotting here
    transport_events_gfa, success_export_df, unsuccess_export_df, success_import_df, unsuccess_import_df, \
        retrogate_export_df, retrogate_import_df, sequences = \
        yr.find_transports_v2(
            thressuccess_transport=success_threshold,
            thres_transport=[-1, 1],
            max_number_intersections=(min_intersections, max_intersections),
            use_spline=True
        )
    all_transport = pd.concat((success_export_df,success_import_df))
    distance_total,num_bins,x_fit, yfit = yr.plot_diffusion_vs_distance([all_transport,], xlim=[-250, 250],
                                  num_bins=20, label_list=[cfg['digits']],ylim=[0,0.0035],label_intersection=f'intersections: {min_intersections}-{max_intersections}')

    unique_crossings = len(np.unique(transport_events_gfa['id'].values))
    unique_exports = len(np.unique(success_export_df['id'].values))
    unique_imports = len(np.unique(success_import_df['id'].values))

    # Merge dataframes for analysis
    merged_df1 = pd.merge(success_export_df, file_lookup, on='id')
    merged_df2 = pd.merge(success_import_df, file_lookup, on='id')

    # Unique combinations and counts
    unique_combinations_df1 = merged_df1[['filename', 'NE_y']].drop_duplicates()
    unique_combinations_df2 = merged_df2[['filename', 'NE_y']].drop_duplicates()
    common_combinations = pd.merge(unique_combinations_df1, unique_combinations_df2, on=['filename', 'NE_y'])
    total_unique_df1 = len(unique_combinations_df1)
    total_unique_df2 = len(unique_combinations_df2)
    total_common_combinations = len(common_combinations)

    # Calculate positive and negative counts
    positive_count = sum(1 for df in sequences if df['dist_cor_spline'].iloc[0] > 0)
    negative_count = sum(1 for df in sequences if df['dist_cor_spline'].iloc[0] < 0)

    # Output the results
    print(f"Results for {label}:")
    print("Number of exports [binding to binding]:", unique_exports)
    print("Number of imports [binding to binding]:", unique_imports)
    print(f"Number of cells/movies with exports: {total_unique_df1}")
    print(f"Number of cells/movies with imports: {total_unique_df2}")
    print(f"Number of cells/movies that match: {total_common_combinations}")
    print("Number of unique tracks that crossed NE:", unique_crossings)
    print("Number of failed export events:", positive_count)
    print("Number of failed import events:", negative_count)

    # Now, include the plotting code and save figures with filenames that include the label

    ### Example Plot 1: Gap Closing Histogram ###
    # Compute sequence_lengths as per your original code
    x = all_tracks['x'].to_numpy()
    ids = all_tracks['id'].to_numpy()
    x_diff = np.diff(x)
    id_diff = np.diff(ids)
    tolerance = 1e-3
    interpolated_mask = np.isclose(x_diff[:-1], x_diff[1:], atol=tolerance) & (id_diff[:-1] == 0)
    change_indices = np.diff(np.concatenate(([0], interpolated_mask.view(np.int8), [0])))
    starts = np.where(change_indices == 1)[0]
    ends = np.where(change_indices == -1)[0]
    sequence_lengths = ends - starts

    # Fit exponential function to the histogram data
    def exponential_func(x, a, b):
        return a * np.exp(-b * x)

    hist_counts, bin_edges = np.histogram(sequence_lengths, bins=np.arange(0.5, 11.5, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    params, covariance = curve_fit(exponential_func, bin_centers, hist_counts, p0=(1, 1))
    a_fitted, b_fitted = params
    a_fitted_sci = f"{a_fitted:.1e}"
    x_fit = np.linspace(1, 10, 100)
    y_fit = exponential_func(x_fit, a_fitted, b_fitted)

    # Plotting the histogram and the fitted exponential curve
    plt.figure(figsize=(2.5, 2.5), dpi=300)
    plt.plot(x_fit, y_fit, 'r-',
             label=f'Exponential fit:\n$y = {a_fitted_sci} \cdot e^{{-{b_fitted:.2f}x}}$')
    plt.hist(sequence_lengths, bins=np.arange(0.5, 11.5, 1), edgecolor='black')
    plt.legend()
    plt.xlabel('Gap closing time [frames]')
    plt.ylabel('Counts')
    plt.title(f"{label}\nIntersections: {min_intersections}-{max_intersections}")
    plt.tight_layout(pad=0.4)
    plt.savefig(f'/home/pieter/development/yeast_processor_v3/table_figure/gapclosinghistogram_{label}.svg', format='svg')
    plt.close()

    ### Example Plot 2: Distance Histogram ###
    # Filter intersect points
    intersect_df1 = merged_df1[merged_df1['intersect_spline'] == 1]
    intersect_df2 = merged_df2[merged_df2['intersect_spline'] == 1]

    # Merge these filtered dataframes on 'filename' and 'NE_y' to find matching intersect points
    intersect_merged = pd.merge(intersect_df1, intersect_df2, on=['filename', 'NE_y'], suffixes=('_df1', '_df2'))

    # Calculate the distance between intersect points
    def calculate_distance(row):
        return np.sqrt((row['x_df1'] - row['x_df2']) ** 2 + (row['y_df1'] - row['y_df2']) ** 2)

    intersect_merged['distance'] = intersect_merged.apply(calculate_distance, axis=1)

    plt.figure()
    plt.hist(np.array(intersect_merged['distance']) * 128, bins=40)
    plt.xlim([0, 20 * 128])
    plt.xlabel('Euclidean distance between crossings\n import and export traces [nm]')
    plt.ylabel('Counts')
    plt.title(f"{label}\nIntersections: {min_intersections}-{max_intersections}")
    plt.tight_layout(pad=0.4)
    plt.savefig(f'/home/pieter/development/yeast_processor_v3/table_figure/distance_histogram_{label}.svg', format='svg')
    plt.close()

    ### Example Plot 3: Time Differences Histogram ###
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
    plt.figure()
    plt.hist(np.array(shortest_time_diffs) * 20, bins=20)  # Adjust bins if needed
    plt.xlabel('Time between export and \n import NE crossings [ms]')
    plt.ylabel('Counts')
    plt.title(f"{label}\nIntersections: {min_intersections}-{max_intersections}")
    plt.tight_layout(pad=0.4)
    plt.savefig(f'/home/pieter/development/yeast_processor_v3/table_figure/time_differences_histogram_{label}.svg', format='svg')
    plt.close()

    # Continue with additional analysis and plotting...
    # For example, plotting diffusion vs distance or velocity if needed

    # Collect results into a dictionary to be saved later
    results = {
        'digits': cfg['digits'],
        'label': label,
        'min_intersections': min_intersections,
        'max_intersections': max_intersections,
        'unique_exports': unique_exports,
        'unique_imports': unique_imports,
        'total_unique_cells_with_exports': total_unique_df1,
        'total_unique_cells_with_imports': total_unique_df2,
        'total_common_cells': total_common_combinations,
        'unique_tracks_crossed_NE': unique_crossings,
        'failed_export_events': positive_count,
        'failed_import_events': negative_count,
        'number_of_movies': len(unique_movies),
        'number_of_cells': number_of_unique_combinations,
        'number_of_unique_tracks': number_of_tracks,
        'average_track_length_frames': average_track_length,
        # Add other variables as needed
    }
    return results

# List of max_number_intersections ranges
max_intersections_list = [
    (0, 3),
    (0, 5),
    (0, 10),
    (0, np.inf),
    (3, 10),
    (3, np.inf)
]

# List of digit sets to loop over
digits_list = [823, 820, 822]
all_results = []
success_thresholds = [[-76.24, 94.76], [-89.75,78.61], [-69.74,77.45]]
labels = ['GFA1', 'MYO2', 'TRA1']

CB_color_cycle = [
    '#0173b2',
    '#de8f05',
    '#029e73',
    '#d55e00',
    '#cc78bc',
    '#ca9161',
    '#fbafe4',
    '#949494',
    '#ece133',
    '#56b4e9'
]

# Set the color cycle for Matplotlib
plt.rcParams['axes.prop_cycle'] = cycler(color=CB_color_cycle)

# Initialize a list to collect summary results for CSV
summary_results = []

for itter, digits in enumerate(digits_list):
    # Update cfg paths for the current set of digits
    cfg = base_cfg.copy()  # Start with the base configuration
    cfg.update({
        'digits': str(digits),
        'all_tracks_path': f'../results_tracks_gapclosing10/all_tracks_{digits}.csv',
        'ne_lookup_path': f'../results_tracks_gapclosing10/ne_lookup{digits}.csv',
        'file_lookup_path': f'../results_tracks_gapclosing10/file_lookup_{digits}.csv',
        'nedata_path': f'../results_tracks_gapclosing10/nedata{digits}.pkl',
    })

    # Loop over the max_number_intersections ranges
    for max_intersection_range in max_intersections_list:
        # Create a unique label for each combination
        min_intersections, max_intersections = max_intersection_range

        # Handle infinity in labels
        if max_intersections == np.inf:
            max_label = 'inf'
        else:
            max_label = str(int(max_intersections))

        label = f"{labels[itter]}_intersections_{min_intersections}_{max_label}"

        # Process data for this configuration and intersection range
        result = process_for_config(
            cfg,
            max_number_intersections_range=max_intersection_range,
            make_velocity=False,
            success_threshold=success_thresholds[itter],
            label=label
        )

        # Store the result in the summary_results list
        summary_results.append(result)

# Create a DataFrame from the summary results
summary_df = pd.DataFrame(summary_results)

# Save the DataFrame to a CSV file
summary_df.to_csv('/home/pieter/development/yeast_processor_v3/table_figure/summary_results.csv', index=False)

print("Summary results have been saved to 'summary_results.csv'. You can open this file in Excel to view the data.")
print("All figures have been saved with filenames that include the label variable to identify configurations.")
