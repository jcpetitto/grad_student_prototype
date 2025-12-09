"""
This script defines the `yeast_extractresults` class and associated utility functions for analyzing yeast transport events.
It provides comprehensive methods for extracting, preprocessing, analyzing, and visualizing tracking data related to yeast cell transport dynamics.
The implementation leverages various libraries such as NumPy, Pandas, Matplotlib, SciPy, Shapely, and others for efficient data manipulation and visualization.

Modules and Dependencies:
- **ast**: Abstract Syntax Trees for parsing strings into Python objects.
- **scipy.interpolate (splprep, splev, UnivariateSpline)**: For spline fitting and evaluation.
- **os**: Operating system interfaces for file and directory operations.
- **itertools.groupby**: For grouping data.
- **numpy**: Numerical operations on arrays.
- **matplotlib.pyplot**: Plotting and visualization.
- **datetime**: Handling date and time.
- **pandas**: Data manipulation and analysis.
- **pickle**: Serializing and deserializing Python objects.
- **scipy.optimize (curve_fit)**: Optimization and curve fitting.
- **skimage.measure**: Image processing and measurements.
- **tqdm**: Progress bars for loops.
- **scipy.stats (norm)**: Statistical functions.
- **scipy.spatial.distance (cdist)**: Computing distance matrices.
- **io**: Handling I/O streams.
- **tifffile**: Reading and writing TIFF files.
- **shapely.geometry (Polygon, Point)**: Geometric objects and operations.
- **shapely.ops (nearest_points)**: Geometric operations on Shapely objects.

Classes:
- **yeast_extractresults**:
    A class designed to extract, preprocess, and analyze tracking results from yeast cell transport experiments.

    **Attributes**:
    - `path`: Main directory path containing data.
    - `resultdir`: Directory where results are stored.
    - `trackdata`: Filename of the tracking data.
    - `pixelsize`: Size of a pixel in nanometers.
    - `moviefilename`: Base name for generated movies.
    - `frametime`: Time interval between frames in seconds.
    - `list_inout`: List of column names related to in/out spline data.
    - `dist_str`: Column name for distance spline.
    - `intersect_str`: Column name for intersection spline.
    - `dist_cor_str`: Column name for corrected distance spline.
    - `x_str`, `y_str`: Column names for x and y spline coordinates.
    - `may_inout_str`, `def_inout_str`: Column names for may/def in/out spline data.

    **Methods**:
    - `__init__(self, cfg)`: Initializes the class with configuration parameters.
        - **Parameters**:
            - `cfg` (dict): Configuration dictionary containing paths and parameters.

    - `find_length(self)`: Calculates the total length of all tracks by loading tracking data.
        - **Returns**:
            - `tot_len` (int): Total number of track points.

    - `gather_tracks(self, length=0)`: Gathers and consolidates tracking data from multiple files.
        - **Parameters**:
            - `length` (int, optional): Predefined length (default is 0).
        - **Returns**:
            - `trackdataframe` (pd.DataFrame): Consolidated tracking data.
            - `fileidlookup` (pd.DataFrame): Lookup table mapping files to track IDs.
            - `neidlookup` (pd.DataFrame): Lookup table for NE points.
            - `nedata` (list): List of NE point data.
            - `number_off_cells_all` (int): Counter for cells (placeholder for future use).
            - `number_off_detections_all` (int): Counter for detections (placeholder).
            - `number_off_detections_tot_all` (int): Total detections counter (placeholder).

    - `preprocess_track(self, track, thres_transport, thressuccess_transport, spline=True)`: Processes individual tracks to categorize transport events.
        - **Parameters**:
            - `track` (pd.DataFrame): Single track data.
            - `thres_transport` (list): Thresholds for transport events.
            - `thressuccess_transport` (list): Thresholds for successful transport.
            - `spline` (bool, optional): Whether to use spline fitting (default is True).
        - **Returns**:
            - `processed` (dict): Dictionary containing categorized track data.

    - `find_transports_v2(self, bootstrap=False, thres_transport=[-100, 100], thressuccess_transport=[-250, 250],
                           max_number_intersections=8, use_spline=True)`: Identifies transport events within the tracking data.
        - **Parameters**:
            - `bootstrap` (bool, optional): Whether to perform bootstrapping (default is False).
            - `thres_transport` (list, optional): Thresholds for transport events.
            - `thressuccess_transport` (list, optional): Thresholds for successful transport.
            - `max_number_intersections` (int or tuple, optional): Maximum allowed intersections.
            - `use_spline` (bool, optional): Whether to use spline fitting (default is True).
        - **Returns**:
            - Tuple of DataFrames containing various categorized transport events.

    - `find_nuclear_detections(self, min_dist=np.inf)`: Finds nuclear detections based on specified conditions.
        - **Parameters**:
            - `min_dist` (float, optional): Minimum distance threshold (default is infinity).
        - **Returns**:
            - `non_transport_df` (pd.DataFrame): DataFrame containing non-transport detections.

    - `find_nuclear_detectionsv2(self)`: Updated method for finding nuclear detections.
        - **Returns**:
            - `tracks_potential` (pd.DataFrame): DataFrame with potential nuclear detections.

    - `movie_dataset(self, trackingdf, lookup, fps=2)`: Generates movies from tracking data.
        - **Parameters**:
            - `trackingdf` (pd.DataFrame): Tracking data.
            - `lookup` (pd.DataFrame): Lookup table.
            - `fps` (int, optional): Frames per second for the movie (default is 2).

    - `movie_transportevents(self, fps=2)`: Generates movies specifically for transport events.
        - **Parameters**:
            - `fps` (int, optional): Frames per second for the movie (default is 2).

    - `cvtFig2Numpy(self, fig)`: Converts a Matplotlib figure to a NumPy array.
        - **Parameters**:
            - `fig` (matplotlib.figure.Figure): Matplotlib figure to convert.
        - **Returns**:
            - `image` (np.ndarray): Image array.

    - `makevideoFromArray(self, movieName, array, fps=25)`: Creates a video file from an image array.
        - **Parameters**:
            - `movieName` (str): Name/path of the output video file.
            - `array` (np.ndarray): Array of images.
            - `fps` (int, optional): Frames per second for the video (default is 25).

    - `line_segments_intersect(self, A, B, C, D)`: Checks if two line segments intersect.
        - **Parameters**:
            - `A`, `B`, `C`, `D` (np.ndarray): Endpoints of the two line segments.
        - **Returns**:
            - `bool`: True if segments intersect, else False.

    - `find_intersections_and_normals(self, nepoints, x_coordinates_rna, y_coordinates_rna)`: Finds intersections between NE and RNA paths and computes normals.
        - **Parameters**:
            - `nepoints` (np.ndarray): Coordinates of the nuclear envelope.
            - `x_coordinates_rna` (np.ndarray): X coordinates of RNA path.
            - `y_coordinates_rna` (np.ndarray): Y coordinates of RNA path.
        - **Returns**:
            - `crossings` (list): List of intersecting segments.
            - `normals` (list): List of normal vectors at intersections.

    - `calculate_distance_midpoint_and_normal_along_envelope(self, nepoints, intersections)`: Calculates distance, midpoint, and normal along the envelope.
        - **Parameters**:
            - `nepoints` (np.ndarray): Coordinates of the nuclear envelope.
            - `intersections` (list): List of intersecting segments.
        - **Returns**:
            - `total_distance` (float): Total distance along the envelope.
            - `midpoint_position` (np.ndarray): Position of the midpoint.
            - `normal_vector` (np.ndarray): Normal vector at the midpoint.

    - `calculate_distance_along_envelope(self, nepoints, intersections)`: Calculates the distance along the envelope between intersections.
        - **Parameters**:
            - `nepoints` (np.ndarray): Coordinates of the nuclear envelope.
            - `intersections` (list): List of intersecting segments.
        - **Returns**:
            - `total_distance` (float): Total distance along the envelope.

    - `rotate_points(self, points, angle)`: Rotates points by a given angle around the origin.
        - **Parameters**:
            - `points` (np.ndarray): Points to rotate.
            - `angle` (float): Rotation angle in radians.
        - **Returns**:
            - `rotated_points` (np.ndarray): Rotated points.

    - `transform_coordinate_system(self, nepoints, x_coordinates_rna, y_coordinates_rna,
                                    intersections, normals, normal_direction='negative')`: Transforms the coordinate system based on intersections and normals.
        - **Parameters**:
            - `nepoints` (np.ndarray): Coordinates of the nuclear envelope.
            - `x_coordinates_rna` (np.ndarray): X coordinates of RNA path.
            - `y_coordinates_rna` (np.ndarray): Y coordinates of RNA path.
            - `intersections` (list): List of intersecting segments.
            - `normals` (list): List of normal vectors at intersections.
            - `normal_direction` (str, optional): Desired normal direction ('negative' or 'positive').
        - **Returns**:
            - `nepoints_rotated` (np.ndarray): Rotated nuclear envelope points.
            - `rna_rotated` (np.ndarray): Rotated RNA path points.

    - `plot_velocity(self, df_list, sample_distances, xlabel='Distance to membrane [nm]',
                      ylabel=r'Velocity [$\mu$m/s] ', name_list=['GFA', 'MYO', 'TRA'], linking_distance=512,
                      bootstrap=True, upsample_factor=10, smoothfactor=1.6, return_raw=False)`: Plots velocity as a function of distance to the membrane.
        - **Parameters**:
            - `df_list` (list of pd.DataFrame): List of tracking dataframes.
            - `sample_distances` (array-like): Distances for sampling.
            - `xlabel` (str, optional): Label for the x-axis.
            - `ylabel` (str, optional): Label for the y-axis.
            - `name_list` (list of str, optional): Labels for different datasets.
            - `linking_distance` (int, optional): Maximum linking distance.
            - `bootstrap` (bool, optional): Whether to perform bootstrapping for error estimates.
            - `upsample_factor` (int, optional): Factor for upsampling spline fits.
            - `smoothfactor` (float, optional): Smoothing factor for spline fitting.
            - `return_raw` (bool, optional): Whether to return raw velocity data.
        - **Returns**:
            - Depending on `return_raw`, returns either processed velocity data or nothing.

    - `plot_all_diffusion(self, df_list, num_bins=10, xlim=[-250, 250], mode='projection1',
                           xlabel='Distance to membrane [nm]', ylabel=r'Velocity [$\mu$m/s] ',
                           name_list=['GFA', 'MYO', 'TRA'], linking_distance=512, bootstrap=True, use_spline=True)`: Plots diffusion coefficients for all datasets.
        - **Parameters**:
            - `df_list` (list of pd.DataFrame): List of tracking dataframes.
            - `num_bins` (int, optional): Number of bins for histogram.
            - `xlim` (list, optional): X-axis limits.
            - `mode` (str, optional): Mode of analysis.
            - `xlabel` (str, optional): Label for the x-axis.
            - `ylabel` (str, optional): Label for the y-axis.
            - `name_list` (list of str, optional): Labels for different datasets.
            - `linking_distance` (int, optional): Maximum linking distance.
            - `bootstrap` (bool, optional): Whether to perform bootstrapping for error estimates.
            - `use_spline` (bool, optional): Whether to use spline fitting.
        - **Returns**:
            - Arrays containing diffusion analysis results.

    - `plot_diffusion_vs_distance(self, trackdf_list, xlim=[-1000, 1000],
                                   num_bins=50, label_list='GFA',
                                   ylim=[0.001, 0.0035], title='', return_params=False, threshold=None,
                                   label_intersection=None)`: Plots diffusion coefficients versus distance with bimodal fitting.
        - **Parameters**:
            - `trackdf_list` (list of pd.DataFrame): List of tracking dataframes.
            - `xlim` (list, optional): X-axis limits.
            - `num_bins` (int, optional): Number of bins for histogram.
            - `label_list` (list of str, optional): Labels for different datasets.
            - `ylim` (list, optional): Y-axis limits.
            - `title` (str, optional): Plot title.
            - `return_params` (bool, optional): Whether to return fitted parameters.
            - `threshold` (float, optional): Threshold for filtering data.
            - `label_intersection` (str, optional): Label for intersections.
        - **Returns**:
            - Depending on `return_params`, returns fitted parameters or nothing.

    - `plot_mean_diffusion_histogram(self, transport_events)`: Plots a histogram of mean diffusion per track.
        - **Parameters**:
            - `transport_events` (pd.DataFrame): DataFrame containing transport events.
        - **Returns**:
            - Displays a histogram plot.

    - `analyze_transport_events(self, transport_events, thres_transport)`: Analyzes transport events to categorize import/export statuses.
        - **Parameters**:
            - `transport_events` (pd.DataFrame): DataFrame containing transport events.
            - `thres_transport` (float): Threshold for transport categorization.
        - **Returns**:
            - `all_in_one` (np.ndarray): Array containing analysis results for each transport event.

    - `make_diffusion_mmsplot(self, df_list, name_list=['GFA', 'MYO', 'TRA'], mode='', linking_distance=5,
                                min_distance=20)`: Creates multiple moment scaling plots for diffusion analysis.
        - **Parameters**:
            - `df_list` (list of pd.DataFrame): List of tracking dataframes.
            - `name_list` (list of str, optional): Labels for different datasets.
            - `mode` (str, optional): Mode of analysis.
            - `linking_distance` (int, optional): Maximum linking distance.
            - `min_distance` (int, optional): Minimum distance threshold.
        - **Returns**:
            - Arrays containing diffusion and scaling analysis results.

    - `compute_distance_combinations(self, label='GFA')`: Computes distances between NE crossings.
        - **Parameters**:
            - `label` (str, optional): Label for the dataset.
        - **Returns**:
            - Plots histograms of distance combinations and counts.

    - `compute_export_events(self, transport_threshold, release_threshold)`: Computes and categorizes export events.
        - **Parameters**:
            - `transport_threshold` (float): Threshold for transport categorization.
            - `release_threshold` (float): Threshold for release categorization.
        - **Returns**:
            - Resized image arrays for nuclear docking, cytoplasmic release, and full export events.

    - `compute_import_events(self, transport_threshold, release_threshold)`: Computes and categorizes import events.
        - **Parameters**:
            - `transport_threshold` (float): Threshold for transport categorization.
            - `release_threshold` (float): Threshold for release categorization.
        - **Returns**:
            - Resized image arrays for cytoplasmic docking, nuclear release, and full export events.

    - `compute_docking_events(self, docking, release_threshold)`: Computes docking events based on thresholds.
        - **Parameters**:
            - `docking` (pd.DataFrame): DataFrame containing docking events.
            - `release_threshold` (float): Threshold for release categorization.
        - **Returns**:
            - Resized image arrays for nuclear docking, cytoplasmic release, and full export events.

    - `plot_dwell_time(self, df_list, num_bins=30, range_docking=[-250, 0],
                       xlabel='Dwell time [ms]', ylabel='Counts',
                       name_list=['GFA', 'MYO', 'TRA'], title='Transport events')`: Plots dwell time histograms with exponential fitting.
        - **Parameters**:
            - `df_list` (list of pd.DataFrame): List of tracking dataframes.
            - `num_bins` (int, optional): Number of bins for histogram.
            - `range_docking` (list, optional): Range for docking time.
            - `xlabel` (str, optional): Label for the x-axis.
            - `ylabel` (str, optional): Label for the y-axis.
            - `name_list` (list of str, optional): Labels for different datasets.
            - `title` (str, optional): Plot title.
        - **Returns**:
            - Displays histogram plots with fitted exponential curves.

Functions:
- **check_intersection(A, B, C, D)**:
    Checks if two line segments AB and CD intersect.
    - **Parameters**:
        - `A`, `B`, `C`, `D` (np.ndarray): Endpoints of the two line segments.
    - **Returns**:
        - `bool`: True if segments intersect, else False.

- **compute_normal(A, B)**:
    Computes the normalized normal vector for line segment AB.
    - **Parameters**:
        - `A`, `B` (np.ndarray): Endpoints of the line segment.
    - **Returns**:
        - `normal` (np.ndarray): Normalized normal vector.

- **find_intersections_and_normals(nepoints, x_coordinates_rna, y_coordinates_rna)**:
    Finds intersections between nuclear envelope (NE) and RNA paths and computes normals at these intersections.
    - **Parameters**:
        - `nepoints` (np.ndarray): Coordinates of the nuclear envelope.
        - `x_coordinates_rna` (np.ndarray): X coordinates of RNA path.
        - `y_coordinates_rna` (np.ndarray): Y coordinates of RNA path.
    - **Returns**:
        - `intersections` (list): List of intersecting NE segments.
        - `normals` (list): List of normalized normal vectors at intersections.

- **calculate_distance_midpoint_and_normal_along_envelope(nepoints, intersections)**:
    Calculates the total distance along the envelope, the midpoint position, and the normal vector at the midpoint.
    - **Parameters**:
        - `nepoints` (np.ndarray): Coordinates of the nuclear envelope.
        - `intersections` (list): List of intersecting NE segments.
    - **Returns**:
        - `total_distance` (float): Total distance along the NE between intersections.
        - `midpoint_position` (np.ndarray): Position of the midpoint.
        - `normal_vector` (np.ndarray): Normal vector at the midpoint.

- **calculate_distance_along_envelope(nepoints, intersections)**:
    Calculates the cumulative distance along the nuclear envelope between intersecting segments.
    - **Parameters**:
        - `nepoints` (np.ndarray): Coordinates of the nuclear envelope.
        - `intersections` (list): List of intersecting NE segments.
    - **Returns**:
        - `total_distance` (float): Cumulative distance along the envelope.

- **rotate_points(points, angle)**:
    Rotates a set of points by a specified angle around the origin.
    - **Parameters**:
        - `points` (np.ndarray): Array of points to rotate.
        - `angle` (float): Rotation angle in radians.
    - **Returns**:
        - `rotated_points` (np.ndarray): Rotated points.

- **transform_coordinate_system(nepoints, x_coordinates_rna, y_coordinates_rna,
                                intersections, normals, normal_direction='negative')**:
    Transforms the coordinate system based on intersections and normals to align RNA paths with the nuclear envelope.
    - **Parameters**:
        - `nepoints` (np.ndarray): Coordinates of the nuclear envelope.
        - `x_coordinates_rna` (np.ndarray): X coordinates of RNA path.
        - `y_coordinates_rna` (np.ndarray): Y coordinates of RNA path.
        - `intersections` (list): List of intersecting NE segments.
        - `normals` (list): List of normalized normal vectors at intersections.
        - `normal_direction` (str, optional): Desired normal direction ('negative' or 'positive').
    - **Returns**:
        - `nepoints_rotated` (np.ndarray): Rotated nuclear envelope coordinates.
        - `rna_rotated` (np.ndarray): Rotated RNA path coordinates.

Usage:
--------
The `yeast_extractresults` class is designed for analyzing transport events in yeast cell imaging data. Below is an example workflow:

1. **Initialize the Class**:
    ```python
    import yaml

    # Load configuration from a YAML file or define it as a dictionary
    cfg = {
        'mainpath': '/path/to/data',
        'resultdir': '/path/to/results',
        'trackdata': 'trackdata.pkl',
        'pixelsize': 100,  # in nm
        'moviename': 'yeast_movie',
        'frametime': 0.5  # in seconds
    }

    extractor = yeast_extractresults(cfg)
    ```

2. **Gather Tracks**:
    ```python
    trackdf, fileidlookup, neidlookup, nedata, _, _, _ = extractor.gather_tracks()
    ```

3. **Find Transport Events**:
    ```python
    transport, success_export, unsuccess_export, success_import, unsuccess_import, retrogate_export, retrogate_import, sequences = extractor.find_transports_v2()
    ```

4. **Analyze Transport Events**:
    ```python
    all_in_one = extractor.analyze_transport_events(transport, thres_transport=100)
    ```

5. **Plot Velocity vs. Distance**:
    ```python
    mean_vel, mean_vel_proj, bin_centers = extractor.plot_velocity(
        df_list=[transport],
        sample_distances=np.linspace(-250, 250, 50),
        name_list=['GFA']
    )
    ```

6. **Generate Movies**:
    ```python
    extractor.movie_transportevents(fps=2)
    ```

7. **Compute and Plot Dwell Times**:
    ```python
    extractor.plot_dwell_time([transport], num_bins=30, range_docking=[-250, 0],
                              xlabel='Dwell time [ms]', ylabel='Counts',
                              name_list=['GFA'], title='Transport events')
    ```

Notes:
--------
- **Data Formats**: Ensure that tracking data files (`trackdata.pkl`, `npc_points*.data`, etc.) are correctly formatted and accessible.
- **Dependencies**: All required libraries must be installed. Some may require additional installation steps (e.g., `shapely`, `tifffile`).
- **Performance**: Some methods involve heavy computations and may benefit from parallel processing or GPU acceleration where applicable.
- **Error Handling**: Methods include basic error handling (e.g., try-except blocks). Depending on data quality, additional validations may be necessary.
- **Visualization**: Generated plots and movies are saved or displayed using Matplotlib and image processing libraries. Ensure that the environment supports GUI operations or modify code to save plots without displaying.
- **Customization**: Parameters such as thresholds (`thres_transport`, `thressuccess_transport`), smoothing factors, and bin sizes can be adjusted based on specific experimental requirements.

Best Practices:
-----------------
- **Data Validation**: Always validate the structure and content of input data before processing to prevent runtime errors.
- **Resource Management**: Close figures and free up memory when dealing with large datasets or generating numerous visualizations.
- **Modularity**: Utilize the class methods in a modular fashion to build flexible analysis pipelines.
- **Documentation**: Keep method docstrings updated to reflect any changes in parameters or functionality.
- **Version Control**: Use version control systems (e.g., Git) to track changes and collaborate effectively.
- **Testing**: Implement unit tests for critical methods to ensure reliability and facilitate debugging.

"""




import ast
from scipy.interpolate import splprep, splev
import os
from itertools import groupby

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
import pandas as pd
import pickle
import scipy.optimize as opt
import skimage.measure
import tqdm
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.spatial.distance import cdist
import io
import tifffile
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points
class yeast_extractresults:

    """
    Yeast result extraction.
    """
    def __init__(self, cfg):
        self.path = cfg['mainpath']
        self.resultdir = cfg['resultdir']
        self.trackdata = cfg['trackdata']
        self.pixelsize = cfg['pixelsize']
        self.moviefilename = cfg['moviename']
        self.frametime = cfg['frametime']
        
       
        self.list_inout = ['may_inout_spline', 'def_inout_spline']
        self.dist_str = 'dist_spline'
        self.intersect_str = 'intersect_spline'
        self.dist_cor_str = 'dist_cor_spline'
        self.x_str = 'x_spline'
        self.y_str= 'y_spline'
        self.may_inout_str = 'may_inout_spline'
        self.def_inout_str = 'def_inout_spline'
        
    def find_length(self):
        import tqdm
        trackdataframe = pd.DataFrame()
        fileidlookup = pd.DataFrame(columns=['filename', 'NE', 'id'])
        ID = 0
        tot_len = 0
        for folder_name in tqdm.tqdm(os.listdir(self.path)):
            path = self.path + '/' + folder_name
            for folder_name_lower in os.listdir(path):
                if folder_name_lower == 'results':
                    for full_path, dirname, files in os.walk(path + '/' + folder_name_lower):
                        for file in files:
                            if file == self.trackdata:
                                with open(full_path + '/' + self.trackdata, 'rb') as inputfile:
                                    tracklist = pickle.load(inputfile)
                                    for listnum in range(len(tracklist)):
                                        tracks = tracklist[listnum]
                                        for tracknum in range(len(tracks)):
                                            df = tracks[tracknum]

                                            ID = ID + 1

                                            #trackdataframe = pd.concat([trackdataframe, df], axis=0, ignore_index=True)
                                            tot_len = tot_len + len(df)
        return tot_len


    def gather_tracks(self, length=0):
        trackdata = []
        fileidlookup_list = []
        neidlookup_list = []
        nedata = []
        ID = 0
        # Assuming these variables will be used or updated later
        number_off_cells_all = 0
        number_off_detections_all = 0
        number_off_detections_tot_all = 0

        for folder_name in tqdm.tqdm(os.listdir(self.path)):
            path = os.path.join(self.path, folder_name)
            results_path = os.path.join(path, 'results')
            if os.path.exists(results_path):
                for full_path, _, files in os.walk(results_path):
                    if self.trackdata in files:
                        with open(os.path.join(full_path, self.trackdata), 'rb') as inputfile:
                            tracklist = pickle.load(inputfile)
                            for listnum in range(len(tracklist)):
                                nefile_path = os.path.join(full_path, f'npc_points{listnum}.data')
                                tracks = tracklist[listnum]
                                try:
                                    with open(nefile_path, 'rb') as nefile:
                                        nepoints = pickle.load(nefile)
                                        neidlookup_list.append(
                                            {'filename': full_path, 'NE': listnum, 'nepoints': np.array(nepoints)[0]})
                                        nedata.append(nepoints)
                                except Exception as e:
                                    print(f"Error loading NE points from {nefile_path}: {e}")
                                    continue

                                for track in tracks:
                                    track['id'] = ID
                                    fileidlookup_list.append({'filename': full_path, 'NE': listnum, 'id': ID})
                                    trackdata.append(track)
                                    ID += 1

        trackdataframe = pd.concat(trackdata, ignore_index=True)
        fileidlookup = pd.DataFrame(fileidlookup_list)
        neidlookup = pd.DataFrame(neidlookup_list)

        self.all_tracks = trackdataframe
        self.lookup = fileidlookup
        # Assuming nedata and the counters are used later
        return trackdataframe, fileidlookup, neidlookup, nedata, number_off_cells_all, number_off_detections_all, number_off_detections_tot_all
    def preprocess_track(self, track, thres_transport, thressuccess_transport, spline=True):
        # Initialize a dictionary to store processed track information
        processed = {
            "transport": [],
            "success_export": [],
            "success_import": [],
            "unsuccess_export": [],
            "unsuccess_import": [],
            "retrogate_export": [],
            "retrogate_import": []
        }

        # Process self.may_inout_str and  self.def_inout_str if they are not NaN


        for inout_col in self.list_inout:
            if track[inout_col].notna().all():
                inout = track[inout_col].to_numpy() * 2 - 1
                dist_arr = inout * track[self.dist_str].to_numpy() * self.pixelsize

                track.loc[:,self.dist_cor_str] = dist_arr

                # Initialize a list to store sequences
                sequences = []
                current_sequence = []
                flag = 0
                flag2 = 0
                # Iterate through the DataFrame
                for index, row in track.iterrows():
                    # Check if the value in the current row is 1
                    if flag2 == 1 and flag==1:
                        if len(sequences)>1:
                            sequences[int(len(sequences)-1)] =   pd.concat([sequences[int(len(sequences)-1)], pd.DataFrame(row).transpose()])
                    flag2 = 0
                    if row[self.intersect_str] == 1:
                        if flag == 1:
                            current_sequence.append(pd.DataFrame(row).transpose())

                        flag = 1

                        # End the sequence one index earlier
                        if len(current_sequence[::])>0:
                            sequences.append(pd.concat(current_sequence))
                            flag2 = 1
                        current_sequence = []

                    # Append the current row to the current sequence
                    current_sequence.append(pd.DataFrame(row).transpose())

                if len(sequences)>0:
                    del (sequences[0])
                processed["sequences"] = sequences
                # Define conditions for categorizing the track
                success_condition = np.min(dist_arr) < thressuccess_transport[0] and np.max(dist_arr) > thressuccess_transport[1]
                transport_condition = np.min(dist_arr) < thres_transport[0] and np.max(dist_arr) > thres_transport[1]

                if success_condition:
                    processed["transport"].append(track)
                    if inout[0] == 1 and inout[-1] == -1 and dist_arr[-1] < thressuccess_transport[0]:
                        processed["success_import"].append(track)
                    if inout[0] == 1 and inout[-1] == -1 and dist_arr[-1] > thressuccess_transport[0]:
                        processed["unsuccess_import"].append(track)

                    if inout[0] == -1 and dist_arr[-1] > thressuccess_transport[1]:
                        processed["success_export"].append(track)
                    if inout[0] == -1 and dist_arr[-1] < thressuccess_transport[1]:
                        processed["unsuccess_export"].append(track)

                elif transport_condition:
                    processed["transport"].append(track)


                    if track[self.dist_cor_str].values[0]>0:
                        if track[self.dist_cor_str].values[-1]>0:
                            processed["retrogate_import"].append(track)
                        elif track[self.dist_cor_str].values[-1]<0:
                            processed["unsuccess_import"].append(track)

                    if track[self.dist_cor_str].values[0]<0:
                        if track[self.dist_cor_str].values[-1]<0:
                            processed["retrogate_export"].append(track)
                        elif track[self.dist_cor_str].values[-1]>0:
                            processed["unsuccess_export"].append(track)

        return processed


    def find_transports_v2(self, bootstrap = False, thres_transport=[-100,100], thressuccess_transport=[-250,250],
                           max_number_intersections=8, use_spline=True):
        result_dfs = {
            "transport": [],
            "success_export": [],
            "success_import": [],
            "unsuccess_export": [],
            "unsuccess_import": [],
            "retrogate_export": [],
            "retrogate_import": [],
            "sequences": [],
        }

        tracks = self.all_tracks.copy()
        if bootstrap:
            # Get unique 'id's
            unique_ids = tracks['id'].unique()

            # Sample 'id's with replacement
            bootstrapped_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)

            # Filter tracks to only include bootstrapped 'id's
            # This involves creating a mask that checks for each row if its 'id' is in the bootstrapped_ids
            mask = tracks['id'].isin(bootstrapped_ids)
            tracks = tracks[mask].copy()

        tracks[self.intersect_str] = tracks[self.intersect_str].fillna(0).astype(int)
        non_zero_intersect_spline = tracks[tracks[self.intersect_str] != 0]

        # Now, drop duplicates based on the 'id' column to find unique IDs
        unique_ids_with_non_zero_intersect = non_zero_intersect_spline.drop_duplicates(subset='id')['id'].values

        tracks_potential = tracks[tracks['id'].isin(unique_ids_with_non_zero_intersect)]
        unique_ids_list = unique_ids_with_non_zero_intersect.tolist()
        all_retrogate_events = []
        for i in tqdm.tqdm(unique_ids_list,desc="find transport events..."):
            track = tracks_potential[tracks_potential['id'] == i].copy()
            if track.empty or (track['closest_index'].isin([0, 1]).any()):
                continue

            if isinstance(max_number_intersections, int) or isinstance(max_number_intersections, float):
                if 0 < track[self.intersect_str].sum() < max_number_intersections:
                    processed = self.preprocess_track(track, thres_transport, thressuccess_transport,spline=use_spline)
                    for key in result_dfs.keys():
                        result_dfs[key].extend(processed[key])
            else:
                if max_number_intersections[0] < track[self.intersect_str].sum() < max_number_intersections[1]:
                    processed = self.preprocess_track(track, thres_transport, thressuccess_transport,spline=use_spline)
                    for key in result_dfs.keys():
                        result_dfs[key].extend(processed[key])

        # Convert lists of DataFrames into single DataFrames
        for key in result_dfs.keys():
            if key == 'sequences':
                result_dfs[key] = result_dfs[key]
            elif result_dfs[key]:
                result_dfs[key] = pd.concat(result_dfs[key], ignore_index=True)
            else:
                result_dfs[key] = pd.DataFrame()

        return (result_dfs["transport"], result_dfs["success_export"], result_dfs["unsuccess_export"],
                result_dfs["success_import"], result_dfs["unsuccess_import"], result_dfs["retrogate_export"],
                result_dfs["retrogate_import"], result_dfs["sequences"])

    def find_nuclear_detections(self, min_dist = np.inf):
        pd.options.mode.chained_assignment = None
        non_transport_dfs = []
        tracks = self.all_tracks

        for i in tqdm.tqdm(range(max(tracks['id']) + 1)):
            track = tracks[tracks['id'] == i]
            intersectarr = track['intersect_spline'].to_numpy()
            dist_arr_ori = track['dist_spline'].to_numpy()
            if len(intersectarr) > 0:
                intersectarr[np.isnan(np.float64(intersectarr))] = 0

                if sum(intersectarr == 1) == 0 and sum(dist_arr_ori<min_dist)>0:
                    if 'may_inout_spline' in track.columns and np.sum(np.isnan(track['may_inout_spline'])) == 0:
                        inout = track['may_inout_spline'].to_numpy()
                        unique_vals = np.unique(inout)
                        if len(unique_vals) >1:
                            continue
                    elif 'def_inout_spline' in track.columns and np.sum(np.isnan(track['def_inout_spline'])) == 0:
                        inout = track['def_inout_spline'].to_numpy()
                    else:
                        raise ValueError('Something is going wrong')

                    inout = inout * 2 - 1
                    dist_arr = inout * track['dist_spline'].to_numpy()

                    track['dist_cor_spline'] = dist_arr
                    non_transport_dfs.append(track)

        non_transport_df = pd.concat(non_transport_dfs, ignore_index=True)
        return non_transport_df

    def find_nuclear_detectionsv2(self):
        pd.options.mode.chained_assignment = None
        # Assuming 'tracks' is your DataFrame and has been properly defined.
        tracks = self.all_tracks

        # Fill NaN and convert data type for intersect column
        tracks[self.intersect_str] = tracks[self.intersect_str].fillna(0).astype(int)
        # Identify tracks that have 'intersect_str' equal to 1
        tracks_with_intersect_one = tracks[tracks[self.intersect_str] == 1]['id'].unique()

        # Exclude these tracks
        tracks_potential = tracks[~tracks['id'].isin(tracks_with_intersect_one)]
        # Filter tracks potential based on these IDs


        # Define a function to apply complex conditions
        inout1 = np.array(tracks_potential['may_inout'])
        inout2 = np.array(tracks_potential['def_inout'])
        # Collect tracks that were successfully processed
        combined_inout = np.where(np.isnan(inout1), inout2, inout1)

        inout = combined_inout * 2 - 1
        dist_arr = inout * tracks_potential['dist_spline'].to_numpy()

        tracks_potential['dist_cor_spline'] = dist_arr


        return tracks_potential
    def movie_dataset(self, trackingdf, lookup, fps=2):
        import tifffile

        import pickle
        from tqdm import tqdm



        unique_id = np.unique(trackingdf['id'].values)


        for id in tqdm(unique_id, desc='transport evennts '):
            arr = []
            arr2 = []
            arr3 = []
            arr4 = []
            track = trackingdf[trackingdf['id'] == id]*1
            minframe = min(track['frame'])
            maxframe = max(track['frame'])
            frames = np.arange(minframe, maxframe + 1, 1)
            pathdf = lookup[lookup['id'] ==id]
            track[self.x_str] -= track['xdrift']
            track[self.y_str] -= track['ydrift']
            #time_list_npc = np.load(pathdf['filename'].values[0] + '/timepoints.npy')

            NE_number = pathdf['NE'].values[0]
            bbox_NE = np.load((pathdf['filename'].values[0]) + '/bbox.npy')
            npc_points_for_time = pickle.load(
                open(pathdf['filename'].values[0] + '/npc_points' + str(NE_number) + '.data', 'rb'))

            image = np.load(
                (pathdf['filename'].values[0]).replace('results/', '') + '/boundingbox' + str(NE_number) + '.npy')
            fullnpcimg = tifffile.imread(
                (pathdf['filename'].values[0]).replace('results/', '') + '/RNAgreen'+''.join(filter(str.isdigit, (pathdf['filename'].values[0]).replace('results/', '')[-5::]))+'.tiff').mean(0)

            for ii in range(np.shape(frames)[0]):
                qq = frames[ii]
                frame = image[qq, :, :]

                npc = npc_points_for_time

                plt.ioff()
                plt.cla()
                plt.clf()
                plt.close('all')
                fig = plt.figure(1, figsize=(4, 4))

                if len((track[track['frame']==qq]['xdrift']).values)!=0:
                    xdrift = (track[track['frame']==qq]['xdrift']).values
                    ydrift = (track[track['frame']==qq]['ydrift']).values

                for npc_segments in npc:
                    plt.plot(npc_segments[0, :]- xdrift- bbox_NE[NE_number][2], npc_segments[1, :] -ydrift - bbox_NE[NE_number][0], linewidth=3, color='red')

                plt.imshow(frame)
                import trackpy as tp

                if np.size(track.query('frame<={0}'.format(qq) + '& frame>={0}'.format(max(0, qq - 20)))) != 0:
                    axes = tp.plot_traj(track.query('frame<={0}'.format(qq) + '& frame>={0}'.format(max(0, qq - 20))),
                                        plot_style={'linewidth': 5})
                from PIL import Image

                arr.append(self.cvtFig2Numpy(fig))
                plt.close('all')
                plt.ioff()
                plt.cla()
                plt.clf()
                plt.close('all')
                fig3 = plt.figure(1, figsize=(4, 4))
                plt.imshow(frame)
                arr3.append(self.cvtFig2Numpy(fig3))
                plt.close('all')
                plt.ioff()
                plt.cla()
                plt.clf()
                plt.close('all')

                fig4 = plt.figure(1, figsize=(4, 4))
                plt.imshow(fullnpcimg)
                arr4.append(self.cvtFig2Numpy(fig4))
                plt.close('all')
                plt.ioff()
                plt.cla()
                plt.clf()
                plt.close('all')

                img = Image.open(pathdf['filename'].values[0] + '/refined_spline.png').convert('RGBA')
                arrnpc = np.array(img)
                fig2 = plt.figure(2, figsize=(4, 4))

                plt.imshow(arrnpc)

                arr2.append(self.cvtFig2Numpy(fig2))
                plt.close('all')
                plt.ioff()
                plt.cla()
                plt.clf()
                plt.close('all')
            array_final1 = np.concatenate((arr, arr2), axis=2)
            array_final2 = np.concatenate((arr3, arr4), axis=2)
            array_final = np.concatenate((array_final1, array_final2), axis=1)
            self.makevideoFromArray(self.moviefilename + 'transport_id_'+str(id) + '.mp4', array_final, fps)

    def movie_transportevents(self, fps=2):
        import tifffile

        import pickle
        from tqdm import tqdm


        trackingdf = self.transport_events*1
        unique_id = np.unique(trackingdf['id'].values)


        for id in tqdm(unique_id, desc='transport evennts '):
            arr = []
            arr2 = []
            arr3 = []
            arr4 = []
            track = trackingdf[trackingdf['id'] == id]*1
            minframe = min(track['frame'])
            maxframe = max(track['frame'])
            frames = np.arange(minframe, maxframe + 1, 1)
            pathdf = self.lookup[self.lookup['id'] ==id]
            track[self.x_str] -= track['xdrift']
            track[self.y_str] -= track['ydrift']
            #time_list_npc = np.load(pathdf['filename'].values[0] + '/timepoints.npy')

            NE_number = pathdf['NE'].values[0]
            bbox_NE = np.load((pathdf['filename'].values[0]) + '/bbox.npy')
            npc_points_for_time = pickle.load(
                open(pathdf['filename'].values[0] + '/npc_points' + str(NE_number) + '.data', 'rb'))

            image = np.load(
                (pathdf['filename'].values[0]).replace('results/', '') + '/boundingbox' + str(NE_number) + '.npy')
            fullnpcimg = tifffile.imread(
                (pathdf['filename'].values[0]).replace('results/', '') + '/RNAgreen'+''.join(filter(str.isdigit, (pathdf['filename'].values[0]).replace('results/', '')[-5::]))+'.tiff').mean(0)

            for ii in range(np.shape(frames)[0]):
                qq = frames[ii]
                frame = image[qq, :, :]

                npc = npc_points_for_time

                plt.ioff()
                plt.cla()
                plt.clf()
                plt.close('all')
                fig = plt.figure(1, figsize=(4, 4))

                if len((track[track['frame']==qq]['xdrift']).values)!=0:
                    xdrift = (track[track['frame']==qq]['xdrift']).values
                    ydrift = (track[track['frame']==qq]['ydrift']).values

                for npc_segments in npc:
                    plt.plot(npc_segments[0, :]- xdrift- bbox_NE[NE_number][2], npc_segments[1, :] -ydrift - bbox_NE[NE_number][0], linewidth=3, color='red')

                plt.imshow(frame)
                import trackpy as tp

                if np.size(track.query('frame<={0}'.format(qq) + '& frame>={0}'.format(max(0, qq - 20)))) != 0:
                    axes = tp.plot_traj(track.query('frame<={0}'.format(qq) + '& frame>={0}'.format(max(0, qq - 20))),
                                        plot_style={'linewidth': 5})
                from PIL import Image

                arr.append(self.cvtFig2Numpy(fig))
                plt.close('all')
                plt.ioff()
                plt.cla()
                plt.clf()
                plt.close('all')
                fig3 = plt.figure(1, figsize=(4, 4))
                plt.imshow(frame)
                arr3.append(self.cvtFig2Numpy(fig3))
                plt.close('all')
                plt.ioff()
                plt.cla()
                plt.clf()
                plt.close('all')

                fig4 = plt.figure(1, figsize=(4, 4))
                plt.imshow(fullnpcimg)
                arr4.append(self.cvtFig2Numpy(fig4))
                plt.close('all')
                plt.ioff()
                plt.cla()
                plt.clf()
                plt.close('all')

                img = Image.open(pathdf['filename'].values[0] + '/refined_spline.png').convert('RGBA')
                arrnpc = np.array(img)
                fig2 = plt.figure(2, figsize=(4, 4))

                plt.imshow(arrnpc)

                arr2.append(self.cvtFig2Numpy(fig2))
                plt.close('all')
                plt.ioff()
                plt.cla()
                plt.clf()
                plt.close('all')
            array_final1 = np.concatenate((arr, arr2), axis=2)
            array_final2 = np.concatenate((arr3, arr4), axis=2)
            array_final = np.concatenate((array_final1, array_final2), axis=1)
            self.makevideoFromArray(self.moviefilename + 'transport_id_'+str(id) + '.mp4', array_final, fps)

    def cvtFig2Numpy(self, fig):
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        canvas = FigureCanvas(fig)
        canvas.draw()

        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height.astype(np.uint32),
                                                                            width.astype(np.uint32), 3)

        return image


    def makevideoFromArray(self, movieName, array, fps=25):
        import imageio


        imageio.mimwrite(movieName, array, fps=fps)

    def line_segments_intersect(self,A, B, C, D):
        """Check if line segments AB and CD intersect."""

        def ccw(A, B, C):
            """Check if the points A, B, and C are in counter-clockwise order."""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def find_simple_crossings_with_normals(self,nepoints, x_coordinates_rna, y_coordinates_rna):
        crossings = []
        normals = []

        # Convert RNA points to line segments
        rna_segments = [(np.array([x_coordinates_rna[i], y_coordinates_rna[i]]),
                         np.array([x_coordinates_rna[i + 1], y_coordinates_rna[i + 1]]))
                        for i in range(len(x_coordinates_rna) - 1)]

        # Check each segment of the nuclear membrane
        for i in range(nepoints.shape[1] - 1):
            A = nepoints[:, i]
            B = nepoints[:, i + 1]
            segment_direction = B - A
            normal_direction = np.array(
                [-segment_direction[1], segment_direction[0]])  # Rotate 90 degrees to get normal

            # Check each RNA segment
            for C, D in rna_segments:
                if self.line_segments_intersect(A, B, C, D):
                    crossings.append((A, B, C, D))  # Simplified; actual crossing point calculation omitted
                    normals.append(normal_direction / np.linalg.norm(normal_direction))  # Normalize the normal vector

        return crossings, normals

    def transform_sequences(self, sequences):
        time_nuc = []
        nuc_rna = []
        dist_between_nuc = []
        cyto_rna = []
        dist_between_cyto= []
        time_cyto = []

        for i in tqdm.tqdm(range(len(sequences)),desc="transform sequences (align normals)"):
            sequence = sequences[i]
            if len(sequence)>0:
                if sequence[self.dist_cor_str].values[1] < 0:
                    lookup = self.lookup[self.lookup['id'] == int(sequence['id'].values[0])]
                    filename, ne = lookup[['filename']].values[0][0], lookup[['NE']].values[0][0]

                    index_true_condition = self.nelookup.loc[
                        (self.nelookup['filename'] == filename) & (self.nelookup['NE'] == ne)].index

                    index_true_condition.tolist()
                    nepoints = self.nedata[index_true_condition[0]][0]
                    bbox = self.nelookup.loc[index_true_condition[0], 'bbox']
                    # Convert the string to a Python list first
                    x_coordinates_rna = sequence[self.x_str].values
                    y_coordinates_rna = sequence[self.y_str].values
                    bbox = np.array(ast.literal_eval(bbox.replace('\n', ',').replace('.', ',')))
                    flag = 0
                    for box in range(np.shape(bbox)[0]):
                        bbox_iter = bbox[box, :]
                        nepoints_potx = nepoints[0, :] - bbox_iter[2]
                        nepoints_poty = nepoints[1, :] - bbox_iter[0]
                        nepoints_pot = np.array([nepoints_potx, nepoints_poty])
                        # Generate new upsampled frames

                        pos = np.array([x_coordinates_rna, y_coordinates_rna])
                        closest_distances = []
                        for x, y in zip(x_coordinates_rna, y_coordinates_rna):
                            point = np.array([[x, y]])
                            distances = cdist(point, nepoints_pot.T)  # Calculate distance to each membrane point
                            closest_distance = np.min(distances)
                            closest_distances.append(closest_distance)
                            # Find the closest distance
                        if sum(abs(np.array(closest_distances) - sequence['dist_spline'])) < 1:
                            flag = 1
                            break

                    if flag == 0:
                        print('ERROR')

                    intersections_npc, normals_npc = find_intersections_and_normals(nepoints_pot, x_coordinates_rna,
                                                                                    y_coordinates_rna)
                    distance_along_envelope = calculate_distance_along_envelope(nepoints_pot, intersections_npc)
                    distance, midpoint, normal = calculate_distance_midpoint_and_normal_along_envelope(nepoints_pot,
                                                                                                       intersections_npc)

                    # Plot nuclear envelope
                    # Plot everything including the corrected starting points for normals
                    nepoints_rotated, rna_rotated = transform_coordinate_system(nepoints_pot, x_coordinates_rna,
                                                                                y_coordinates_rna,
                                                                                midpoint, normal, normal_direction='negative')

                    nuc_rna.append(rna_rotated[:,1:-1])

                    dist_between_nuc.append(distance_along_envelope)
                    time_nuc.append(len(sequence)-3)



                else:
                    lookup = self.lookup[self.lookup['id'] == int(sequence['id'].values[0])]
                    filename, ne = lookup[['filename']].values[0][0], lookup[['NE']].values[0][0]

                    index_true_condition = self.nelookup.loc[
                        (self.nelookup['filename'] == filename) & (self.nelookup['NE'] == ne)].index

                    index_true_condition.tolist()
                    nepoints = self.nedata[index_true_condition[0]][0]
                    bbox = self.nelookup.loc[index_true_condition[0], 'bbox']
                    # Convert the string to a Python list first
                    x_coordinates_rna = sequence[self.x_str].values
                    y_coordinates_rna = sequence[self.y_str].values
                    bbox = np.array(ast.literal_eval(bbox.replace('\n', ',').replace('.', ',')))
                    flag = 0
                    for box in range(np.shape(bbox)[0]):
                        bbox_iter = bbox[box, :]
                        nepoints_potx = nepoints[0, :] - bbox_iter[2]
                        nepoints_poty = nepoints[1, :] - bbox_iter[0]
                        nepoints_pot = np.array([nepoints_potx, nepoints_poty])
                        # Generate new upsampled frames

                        pos = np.array([x_coordinates_rna, y_coordinates_rna])
                        closest_distances = []
                        for x, y in zip(x_coordinates_rna, y_coordinates_rna):
                            point = np.array([[x, y]])
                            distances = cdist(point, nepoints_pot.T)  # Calculate distance to each membrane point
                            closest_distance = np.min(distances)
                            closest_distances.append(closest_distance)
                            # Find the closest distance
                        if sum(abs(np.array(closest_distances) - sequence['dist_spline'])) < 1:
                            flag = 1
                            break

                    if flag == 0:
                        print('ERROR')

                    intersections_npc, normals_npc = find_intersections_and_normals(nepoints_pot, x_coordinates_rna,
                                                                                    y_coordinates_rna)
                    distance_along_envelope = calculate_distance_along_envelope(nepoints_pot, intersections_npc)
                    distance, midpoint, normal = calculate_distance_midpoint_and_normal_along_envelope(nepoints_pot,
                                                                                                       intersections_npc)
                    nepoints_rotated, rna_rotated = transform_coordinate_system(nepoints_pot, x_coordinates_rna,
                                                                                y_coordinates_rna,
                                                                                midpoint, normal,
                                                                                normal_direction='negative')

                    cyto_rna.append(rna_rotated[:,1:-1])
                    dist_between_cyto.append(distance_along_envelope)
                    time_cyto.append(len(sequence)-3)
                    # Plot nuclear envelope
                    # Plot everything including the corrected starting points for normals
                    # plt.figure(figsize=(8, 6))
                    # len_line = 3
                    # for iteration, normal in enumerate(normals_npc):
                    #     intersection = intersections_npc[iteration][0]
                    #     linex = [intersection[0], intersection[0] + normal[0] * len_line]
                    #     liney = [intersection[1], intersection[1] + normal[1] * len_line]
                    #     plt.plot(linex, liney, 'k', label='normal', linewidth=2)
                    # plt.plot(*nepoints_pot, 'b-', label='Nuclear Envelope')
                    # plt.plot(x_coordinates_rna, y_coordinates_rna, 'g-', label='RNA Path')
                    #
                    # plt.xlabel('X Coordinate')
                    # plt.ylabel('Y Coordinate')
                    # plt.legend()
                    # plt.axis('equal')
                    # plt.title('Distance' + str(distance_along_envelope))
                    #
                    # plt.show()
                    #
                    # # Plotting for verification
                    # plt.figure(figsize=(8, 6))
                    # plt.plot(nepoints_rotated[0, :], nepoints_rotated[1, :], 'b-', label='Nuclear Envelope Transformed')
                    # plt.plot(rna_rotated[0,1:-1], rna_rotated[1, 1:-1], 'g-', label='RNA Path Transformed')
                    # plt.plot(0, 0, 'ro', label='First Intersection')  # Mark the new origin
                    #
                    # plt.xlabel('X Coordinate')
                    # plt.ylabel('Y Coordinate')
                    # plt.legend()
                    # plt.axis('equal')
                    # plt.title('Transformed Coordinate System')
                    # plt.show()
                    # test = 0

        return nuc_rna, dist_between_nuc,cyto_rna,dist_between_cyto, time_nuc, time_cyto

    def plot_velocity(self, df_list, sample_distances, xlabel='Distance to membrane [nm]',
                      ylabel=r'Velocity [$\mu$m/s] ', name_list=['GFA', 'MYO', 'TRA'], linking_distance=512,
                      bootstrap=True,   upsample_factor = 10, smoothfactor= 1.6, return_raw=False):
        def bootstrap_mean_and_error(data, num_samples=1000):
            means = []

            for _ in range(num_samples):
                # Bootstrap resampling with replacement
                bootstrap_sample = np.random.choice(data, size=len(data), replace=True)

                # Compute mean of the bootstrap sample
                mean = np.mean(bootstrap_sample)
                means.append(mean)

            # Calculate mean and error bars
            mean_of_means = np.mean(means)
            std_of_means = np.std(means)

            return mean_of_means, std_of_means
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        from cycler import cycler

        # Create a custom color cycle based on your pattern 1, 1, 2, 2, 3, 3 from the default cycle
        # Note: Python indexing starts at 0, so "1, 1, 2, 2, 3, 3" corresponds to "0, 0, 1, 1, 2, 2" in zero-based indexing
        custom_color_cycle = [default_colors[i] for i in [0, 0,0,0, 1, 1,1,1, 2, 2,2,2]]

        # Create a new figure and axes


        # Set the custom property cycle

        fig, ax = plt.subplots(dpi=400, figsize=(4, 4))
        ax.set_prop_cycle(cycler('color', custom_color_cycle))
        for df, label in zip(df_list, name_list):
            all_velocities = []
            all_velocities_proj = []
            all_distances = [] # Dict to store sampled velocities
            for id in tqdm.tqdm(df['id'].unique()):
                single_track = df[df['id'] == id]
                if single_track.empty or len(single_track) < 3:  # Need at least 3 points to fit a spline
                    continue
                # Fit splines for x and y coordinates vs. frame number (time)
                frames = single_track['frame'].values

                x_spline = UnivariateSpline(frames, single_track['x'].values, s=smoothfactor)
                y_spline = UnivariateSpline(frames, single_track['y'].values, s=smoothfactor)
                lookup = self.lookup[self.lookup['id'] == id]
                filename, ne = lookup[['filename']].values[0][0], lookup[['NE']].values[0][0]

                index_true_condition = self.nelookup.loc[
                    (self.nelookup['filename'] == filename) & (self.nelookup['NE'] == ne)].index

                index_true_condition.tolist()
                nepoints = self.nedata[index_true_condition[0]][0]
                bbox = self.nelookup.loc[index_true_condition[0], 'bbox']
                # Convert the string to a Python list first

                bbox = np.array(ast.literal_eval(bbox.replace('\n', ',').replace('.', ',')))
                # nepoints[0, :] -= bbox[ne,2]
                # nepoints[1, :] -= bbox[ne, 0]
                flag=0
                for box in range(np.shape(bbox)[0]):
                    bbox_iter = bbox[box,:]
                    nepoints_potx = nepoints[0,:]- bbox_iter[2]
                    nepoints_poty= nepoints[1,:]- bbox_iter[0]
                    nepoints_pot = np.array([nepoints_potx,nepoints_poty])
                    # Generate new upsampled frames

                    new_frames = np.linspace(frames.min(), frames.max(), len(frames) * 1)

                    # Calculate new positions from spline
                    new_x = x_spline(new_frames)
                    new_y = y_spline(new_frames)

                    pos = np.array([new_x, new_y])
                    closest_distances = []
                    for x, y in zip(new_x, new_y):
                        point = np.array([[x, y]])
                        distances = cdist(point, nepoints_pot.T)  # Calculate distance to each membrane point
                        closest_distance = np.min(distances)
                        closest_distances.append(closest_distance)
                        # Find the closest distance
                    test = 0
                    if sum(abs(np.array(closest_distances) -single_track['dist_spline']))<10:
                        flag=1
                        break

                if flag == 0:
                    print('ERROR')
                new_frames = np.linspace(frames.min(), frames.max(), len(frames) * upsample_factor)
                new_x = x_spline(new_frames)
                new_y = y_spline(new_frames)
                pos = np.array([new_x, new_y])
                dx = np.diff(new_x)  # Difference in x between consecutive points
                dy = np.diff(new_y)  # Difference in y between consecutive points
                dt = np.diff(new_frames)  # Time difference between consecutive points
                velocity = np.sqrt(dx ** 2 + dy ** 2) / dt
                velocity = np.append(velocity, velocity[-1])
                # Calculate velocity (first derivative of the spline)
                # Calculate closest distance to the membrane for each new position
                # Prepare membrane points
                x_membrane, y_membrane = nepoints_pot

                # Fit a spline to the membrane points
                if sum(np.isnan(single_track[ self.def_inout_str])) == 0:
                    tck_mem, u_mem = splprep([x_membrane, y_membrane], s=0,
                                     per=True,quiet=3)  # s=0 for no smoothing, per=True for a periodic spline
                else:
                    tck_mem, u_mem= splprep([x_membrane, y_membrane], s=0,
                                             per=False,quiet=3)
                    # Evaluate the spline on a dense grid for a smooth curve
                u_fine = np.linspace(0, 1, 10000)
                x_fine, y_fine = splev(u_fine, tck_mem)  # Corrected to unpack two values

                nepoints_fine = np.array([x_fine, y_fine])


                closest_distances = []
                for x, y in zip(new_x, new_y):
                    point = np.array([[x, y]])
                    distances = cdist(point, nepoints_fine.T)  # Calculate distance to each membrane point
                    closest_distance = np.min(distances)  # Find the closest distance
                    closest_distances.append(closest_distance)

                closest_distances = np.array(closest_distances)

                if sum(np.isnan(single_track[ self.def_inout_str])) ==0:
                    in_out_array = skimage.measure.points_in_poly(pos.T, np.array([nepoints_pot[0, :],
                                                          nepoints_pot[1, :]]).T).astype(int)*-2+1
                elif sum(np.isnan(single_track[self.may_inout_str])) ==0:
                    tck, u = splprep([nepoints_pot[0, :], nepoints_pot[1, :]], s=0, per=True,
                                     quiet=3)  # z=None because we don't have z-values
                    spl_values = splev(np.linspace(0, 1, 1000), tck)
                    in_out_array = skimage.measure.points_in_poly(pos.T, np.array([spl_values[0],
                                                                                   spl_values[1]]).T).astype(int)*-2+1

                closest_distances *= in_out_array
                dd = abs(np.diff(closest_distances))
                vel_proj = abs(dd / dt)
                vel_proj = np.append(vel_proj, vel_proj[-1])


                all_distances.extend(closest_distances)
                all_velocities.extend(velocity)
                all_velocities_proj.extend(vel_proj)
            all_distances = np.array(all_distances)
            all_velocities = np.array(all_velocities)*self.pixelsize/1000/self.frametime
            all_velocities_proj = np.array(all_velocities_proj)*self.pixelsize/1000/self.frametime
            # Use np.histogram to bin the distances
            counts, edges = np.histogram(all_distances, bins=sample_distances/128, density=False)

            # Initialize an array to hold mean velocity for each bin
            mean_velocities = np.ones(len(edges) - 1)
            mean_velocities_proj = np.ones(len(edges) - 1)
            std_velocities = np.zeros(len(edges) - 1)
            std_velocities_proj = np.zeros(len(edges) - 1)
            all_velocities_list = []
            # Calculate mean velocity for each bin
            for i in range(len(edges) - 1):

                # Indices of velocities within the current bin
                indices = (all_distances >= edges[i]) & (all_distances < edges[i + 1])
                velocities_in_bin = all_velocities[indices]
                all_velocities_list.append(velocities_in_bin)
                velocities_in_bin = velocities_in_bin[velocities_in_bin<4] # filter out spurious
                velocities_proj_in_bin = all_velocities_proj[indices]
                velocities_proj_in_bin = velocities_proj_in_bin[velocities_proj_in_bin < 4]  # filter out spurious
                mean_velocities[i] = np.mean(velocities_in_bin)
                std_velocities[i] = 0 #np.std(velocities_in_bin)
                mean_velocities_proj[i] = np.mean(velocities_proj_in_bin)
                std_velocities_proj[i] = 0# np.std(velocities_proj_in_bin)


                # mean_velocities[i], std_velocities[i] = bootstrap_mean_and_error(velocities_in_bin)
                # mean_velocities_proj[i],std_velocities_proj[i] = bootstrap_mean_and_error(velocities_proj_in_bin)


            # Plotting mean velocity as a function of the bin centers
            bin_centers = (edges[:-1] + edges[1:]) / 2
            ax.plot(bin_centers*self.pixelsize, mean_velocities, label=label, linestyle='-')
            ax.plot(bin_centers*self.pixelsize, mean_velocities_proj, label=label, linestyle='--')
            # plot the standard deviation of the velocities too (not as error bar but as coninoues area)
            # ax.fill_between(bin_centers, mean_velocities - std_velocities, mean_velocities + std_velocities, alpha=0.2)
            # ax.fill_between(bin_centers, mean_velocities_proj - std_velocities_proj,
            #                 mean_velocities_proj + std_velocities_proj, alpha=0.2)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.tight_layout()
        plt.close('all')
        if return_raw:
            return mean_velocities,mean_velocities_proj,bin_centers,all_velocities_list
        else:
            return mean_velocities,mean_velocities_proj,bin_centers
    def plot_all_diffusion(self, df_list, num_bins=10, xlim=[-250, 250], mode='projection1',
                           xlabel = 'Distance to membrane [nm]', ylabel = r'Velocity [$\mu$m/s] ',
                           name_list=['GFA', 'MYO', 'TRA'],linking_distance=512, bootstrap=True,use_spline=True):

        def bootstrap_mean_and_error(data, num_samples=10000):
            means = []

            for _ in range(num_samples):
                # Bootstrap resampling with replacement
                bootstrap_sample = np.random.choice(data, size=len(data), replace=True)

                # Compute mean of the bootstrap sample
                mean = np.mean(bootstrap_sample)
                means.append(mean)

            # Calculate mean and error bars
            mean_of_means = np.mean(means)
            std_of_means = np.std(means)

            return mean_of_means, std_of_means

        # Initialize lists to store data for each dataframe

        mean_values_list = []
        std_values_list = []
        mean_proj_values_list = []
        std_proj_values_list = []

        fig, (ax,ax1) = plt.subplots(1,2,dpi=400, figsize=(3, 3))
        fig3, ax3 = plt.subplots(1, 1, dpi=400, figsize=(3, 3))
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        from cycler import cycler

        # Create a custom color cycle based on your pattern 1, 1, 2, 2, 3, 3 from the default cycle
        # Note: Python indexing starts at 0, so "1, 1, 2, 2, 3, 3" corresponds to "0, 0, 1, 1, 2, 2" in zero-based indexing
        custom_color_cycle = [default_colors[i] for i in [0, 0, 1, 1, 2, 2]]

        # Create a new figure and axes


        # Set the custom property cycle
        ax.set_prop_cycle(cycler('color', custom_color_cycle))
        #ax1.set_prop_cycle(cycler('color', custom_color_cycle))
        ax3.set_prop_cycle(cycler('color', custom_color_cycle))
        for df, label in zip(df_list,name_list ):
            unique_ids = df['id'].unique()
            dif_total = np.array([])
            dif_proj_total = np.array([])
            distance_total = np.array([])

            for id in tqdm.tqdm(unique_ids):
                single_track = df[df['id'].values == id]
                framestep = (single_track['frame'].values)[1::] - (single_track['frame'].values)[:-1]
                x_displacement = (single_track[x_str].values)[1::] - (single_track[x_str].values)[:-1]
                y_displacement = (single_track[y_str].values)[1::] - (single_track[y_str].values)[:-1]
                displacement = np.sqrt(x_displacement ** 2 + y_displacement ** 2)

                displacement_proj =  abs((single_track[dist_cor_str].values)[1::] - (single_track[dist_cor_str].values)[:-1])

                diffusion = displacement / framestep
                diffusion_proj = displacement_proj / framestep

                if np.sum(np.isnan(single_track[self.may_inout_str].values)) == 0:
                    inout = (single_track[self.may_inout_str].values * 2) - 1
                    distance_from_ne = single_track[self.dist_str].values * inout
                    distance_from_ne = (distance_from_ne[1::] + distance_from_ne[:-1]) / 2
                else:
                    inout = (single_track[self.def_inout_str].values * 2) - 1
                    distance_from_ne = single_track[self.dist_str].values * inout
                    distance_from_ne = (distance_from_ne[1::] + distance_from_ne[:-1]) / 2
                dif_proj_total = np.concatenate((dif_proj_total, diffusion_proj))
                dif_total = np.concatenate((dif_total, diffusion))
                distance_total = np.concatenate((distance_total, distance_from_ne))

            distance_total = distance_total * self.pixelsize
            dif_total = dif_total * self.pixelsize
            filter_distance = dif_total < linking_distance
            dif_total = dif_total[filter_distance]
            dif_proj_total = dif_proj_total[filter_distance]* self.pixelsize

            distance_total = distance_total[filter_distance]
            hist, edges = np.histogram(distance_total, bins=num_bins, range=(xlim[0], xlim[1]))
            mean_values = []
            std_values = []
            mean_proj_values = []
            std_proj_values = []

            for i in range(num_bins):
                mask = (distance_total >= edges[i]) & (distance_total < edges[i + 1])
                if bootstrap:
                    mean, error = bootstrap_mean_and_error(dif_total[mask])
                    mean_proj, error_proj = bootstrap_mean_and_error(dif_proj_total[mask])
                    mean_values.append(mean*self.frametime)
                    std_values.append(error*self.frametime)
                    mean_proj_values.append(mean_proj*self.frametime)
                    std_proj_values.append(error_proj*self.frametime)
                else:
                    mean_values.append(np.mean(dif_total[mask])*self.frametime)
                    std_values.append(np.std(dif_total[mask])*self.frametime/np.sqrt(len(dif_total[mask])))
                    mean_proj_values.append(np.mean(dif_proj_total[mask])*self.frametime)
                    std_proj_values.append(np.std(dif_proj_total[mask])*self.frametime/np.sqrt(len(dif_total[mask])))

            mean_values_list.append(mean_values)
            std_values_list.append(std_values)
            mean_proj_values_list.append(mean_proj_values)
            std_proj_values_list.append(std_proj_values)

            # Plot the line with error bars
            ax3.errorbar((edges[:-1] + edges[1:]) / 2, mean_values, yerr=std_values, marker='o', linestyle='-',
                        label=label,markersize=4,capsize=4)

            ax3.errorbar((edges[:-1] + edges[1:]) / 2, mean_proj_values, yerr=std_proj_values, marker='o', linestyle=':',
                         markersize=4, capsize=4)

            ax.errorbar((edges[:-1] + edges[1:]) / 2, mean_values, yerr=std_values, marker='o', linestyle='-',
                        label=label,markersize=4,capsize=4)
            ax.errorbar((edges[:-1] + edges[1:]) / 2, mean_proj_values, yerr=std_proj_values, marker='o', linestyle=':',
                          markersize=4, capsize=4)

            ax1.hist(distance_total, bins=num_bins, range=(xlim[0], xlim[1]), label=label, alpha=0.5,
                    )
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel('Counts')
            ax1.legend()

        # ax.set_ylim([0, 100])
        # ax.set_xlabel(xlabel)
        # ax.set_ylabel(ylabel)
        # ax.legend()
        ax3.set_ylim([15*self.frametime, 100*self.frametime])
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(ylabel)
        ax3.legend()
        fig.tight_layout()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"../results/diffustion1_{current_time}.svg"
        # plt.title('Diffusion Coefficients for All Datasets')
        fig.savefig(file_path, format='svg')
        #plt.title('Diffusion Coefficients for All Datasets')
        fig.show()
        fig3.tight_layout()

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"../results/diffustion2_{current_time}.svg"
        # plt.title('Diffusion Coefficients for All Datasets')
        fig3.savefig(file_path, format='svg')
        fig3.show()


    def plot_diffusion_vs_distance(self,trackdf_list, xlim=[-1000,1000],
                                   num_bins = 50, label_list='GFA',
                                   ylim=[0.001, 0.0035],title='',return_params=False,threshold = None, label_intersection=None):
        fig, ax2 = plt.subplots(figsize=(3, 3))
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        from cycler import cycler

        # Create a custom color cycle based on your pattern 1, 1, 2, 2, 3, 3 from the default cycle
        # Note: Python indexing starts at 0, so "1, 1, 2, 2, 3, 3" corresponds to "0, 0, 1, 1, 2, 2" in zero-based indexing
        custom_color_cycle = [default_colors[i] for i in [0, 0, 1, 1, 2, 2]]

        # Create a new figure and axes

        # Set the custom property cycle
        ax2.set_prop_cycle(cycler('color', custom_color_cycle))
        for trackdf, label in zip(trackdf_list,label_list):


            unique_ids = trackdf['id'].unique()
            unique_ids_list = unique_ids.tolist()
            dif_total = np.array([])
            distance_total = np.array([])

            for id in tqdm.tqdm(unique_ids_list):
                single_track = trackdf[trackdf['id'].values == id]
                if threshold is not None:
                    if min(single_track['dist_cor_spline'].values) > -threshold or \
                            max(single_track['dist_cor_spline'].values) <threshold:
                        continue


                framestep = (single_track['frame'].values)[1::] - (single_track['frame'].values)[:-1]
                x_displacement = (single_track[self.x_str].values)[1::] - (single_track[self.x_str].values)[:-1]
                y_displacement = (single_track[self.y_str].values)[1::] - (single_track[self.y_str].values)[:-1]
                displacement = np.sqrt(x_displacement**2 + y_displacement**2)
                diffusion = displacement / framestep

                distance_from_ne = np.array(single_track['dist_cor_spline'])
                midpoints = (distance_from_ne[:-1] + distance_from_ne[1:]) / 2

                dif_total = np.concatenate((dif_total, diffusion))
                distance_total = np.concatenate((distance_total, midpoints))

            # Define the number of bins on the x-axis

            distance_total = distance_total
            dif_total = dif_total *self.pixelsize

            # Bin distance_total on the x-axis
            hist, edges = np.histogram(distance_total, bins=num_bins, range=(xlim[0], xlim[1]),density=True)
            bin_centers = (edges[1:] + edges[:-1]) / 2
            # Find the peaks of the histogram
            peak1 = bin_centers[np.argmax(hist[:len(hist) // 2])]
            peak2 = bin_centers[np.argmax(hist[len(hist) // 2:]) + len(hist) // 2]

            # Estimate initial parameters
            initial_amplitude1 = np.sum(hist)*7  # Half of the maximum value
            initial_amplitude2 = np.sum(hist)*7  # Half of the maximum value
            initial_mean1 = -100
            initial_mean2 = 100
            initial_std1 = 120# Estimated from data
            initial_std2 = 120  # Estimated from data
            initial_offset = min(hist)
            initial_params = [initial_amplitude1, initial_mean1, initial_std1, initial_amplitude2, initial_mean2,
                              initial_std2, initial_offset]

            # Define the bimodal function
            def bimodal(x, amplitude1, mean1, std1, amplitude2, mean2, std2, offset):
                return amplitude1 * norm.pdf(x, mean1, std1) + amplitude2 * norm.pdf(x, mean2, std2) + offset

            # Fit the bimodal function to the data
            try:
                params, cov = curve_fit(bimodal, bin_centers, hist, p0=initial_params, method='dogbox')
                perr = np.sqrt(np.diag(cov))
            except:
                params = [1,1,1,1,1,1,1]
                perr = [0,0,0,0,0,0,0]
            print('label = ', label)
            print('mean1 = ', params[1], 'error = ', perr[1])
            print('std1 = ', params[2], 'error = ', perr[2])
            print('area1 = ',params[0]*params[2]/0.3989)
            print('mean2 = ', params[4], 'error = ', perr[4])
            print('std2 = ', params[5], 'error = ', perr[5])
            print('area2 = ',params[3]*params[5]/0.3989)

            print('ratio cyto/nuc= ', params[3]*params[5]/(params[0]*params[2]) )
            print('\n')
            # Plot the histogram
            x_fit = np.linspace(xlim[0], xlim[1], 1000)

            # Calculate mean and standard deviation of dif_total for each bin
            mean_values = []
            std_values = []
            for i in range(num_bins):
                mask = (distance_total >= edges[i]) & (distance_total < edges[i + 1])
                mean_values.append(np.mean(dif_total[mask]))
                std_values.append(np.std(dif_total[mask]))

            # Create the line plot
        fig, ax2 = plt.subplots(figsize=(2, 0.5))

        ax2.hist(distance_total, bins=num_bins, range=(xlim[0], xlim[1]), label=label,alpha=0.5,density=True,color='#0173b2')
        ax2.plot(x_fit, bimodal(x_fit, *params),color='#0173b2')
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        # Update the colo   rs of the labels
        #ax2.set_xlabel('Distance to membrane [nm]')
        #ax.set_ylabel('Diffusion coefficient [nm/frame]', color='blue')
        #ax2.set_ylabel('Loc. prob.')
        ax2.set_ylim([0.0005,0.0035])
       # ax2.set_title(title)
        #fig.legend()

        #ax2.set_facecolor('white')
        # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # file_path = f"../results/bi_modal_{current_time}.svg"
        # # plt.title('Diffusion Coefficients for All Datasets')
        # plt.tight_layout(pad=0.1)
        # if label_intersection is None:
        #     plt.savefig(file_path, bbox_inches='tight', format='svg')
        # else:
        #     file_path = os.path.join('/home/pieter/development/yeast_processor_v3/table_figure',
        #                              label + label_intersection+ 'binding_site.svg')
        #     plt.savefig(file_path, bbox_inches='tight', format='svg')
        #
        # plt.close()
        if return_params:
            return distance_total, num_bins, x_fit, bimodal(x_fit, *params),params[1],perr[1], params[4],perr[4]
        else:
            return distance_total,num_bins,x_fit, bimodal(x_fit, *params)

    def plot_mean_diffusion_histogram(self, transport_events):
        unique_ids = transport_events['id'].unique()
        unique_ids_list = unique_ids.tolist()
        dif_pertrack = np.array([])

        for id in tqdm.tqdm(unique_ids_list):
            single_track = transport_events[transport_events['id'] == id]
            framestep = (single_track['frame'].values)[1::] - (single_track['frame'].values)[:-1]
            x_displacement = (single_track[self.x_str].values)[1::] - (single_track[self.x_str].values)[:-1]
            y_displacement = (single_track[self.y_str].values)[1::] - (single_track[self.y_str].values)[:-1]
            displacement = np.sqrt(x_displacement ** 2 + y_displacement ** 2)
            diffusion = displacement / framestep

            dif_pertrack = np.append(dif_pertrack, np.mean(diffusion))

        plt.hist(dif_pertrack, bins=20, color='blue', alpha=0.7)
        plt.xlabel('Mean Diffusion per Track [nm/frame]')
        plt.ylabel('Frequency')
        plt.title('Histogram of Mean Diffusion per Track transport events')

        plt.show()

    def analyze_transport_events(self, transport_events, thres_transport):

        unique_ids = transport_events['id'].unique()
        unique_ids_list = unique_ids.tolist()

        import_time_list = np.array([])  # Use NumPy array
        export_time_list = np.array([])  # Use NumPy array
        status_export = np.array([])  # Use NumPy array
        status_import = np.array([])  # Use NumPy array
        all_in_one = np.zeros((len(unique_ids), 4))

        for index, id_iter in enumerate(unique_ids_list):

            single_track = transport_events[transport_events['id'] == id_iter]
            outside_release = np.array(single_track['dist_cor'] < -thres_transport / self.pixelsize)
            inside_release = np.array(single_track['dist_cor'] > thres_transport / self.pixelsize)
            # Find the indices of the first True in each array
            index_inside = np.where(inside_release)[0]
            index_outside = np.where(outside_release)[0]
            all_in_one[index, 0] = id_iter

            # Determine which array has the first True event
            if len(index_inside) > 0 and (len(index_outside) == 0 or index_inside[0] < index_outside[0]):

                print("Particle exported")
                all_in_one[index, 1] = 0
                # Particle starts inside
                first_outside = index_outside[0]
                last_inside = index_inside[index_inside < first_outside][-1]

                export_time = first_outside - last_inside
                all_in_one[index, 3] = export_time
                export_time_list = np.append(export_time_list, export_time)  # Append to NumPy array
                print('export time =', export_time)
                # Now we have several options after the import:
                # 1. Did it go back inside the nucleus

                if len(inside_release) > 0 and np.any(inside_release[first_outside:]):
                    print("Option 1: The particle went back inside the nucleus.")

                    export_after_entry = None
                    for i, is_outside in enumerate(outside_release[first_outside + 1:]):
                        if is_outside:
                            export_after_entry = i + first_outside
                            break

                    if export_after_entry is not None:
                        print("molecule  event detected after going back outside again the nucleus at frame:",
                              export_after_entry)
                        print("If this happens frequently, we need to think about it")
                        status_export = np.append(status_export, 4)
                        all_in_one[index, 2] = 4
                        # Append to NumPy array
                    else:
                        status_export = np.append(status_export, 3)  # Append to NumPy array
                        all_in_one[index, 2] = 3
                # 2. did it release completely inside the nucleus (is the last frame in the nucleus > threshold)
                elif outside_release[-1]:
                    print('particle got released')
                    status_export = np.append(status_export, 1)  # Append to NumPy array
                    all_in_one[index, 2] = 1

                # 3. did it go back in 'docking' and stayed there (is the last frame in the nucleus < threshold)
                elif not outside_release[-1]:
                    print('particle got back in docking')
                    status_export = np.append(status_export, 2)  # Append to NumPy array
                    all_in_one[index, 2] = 2
                else:
                    assert ('error!')

            else:
                print("Particle imported")
                all_in_one[index, 1] = 1
                # Particle starts outside
                # Find the first time it was outside above the threshold
                first_inside = index_inside[0]
                # Find the latest time it was inside, before it was outside
                last_outside = index_outside[index_outside < first_inside][-1]
                import_time = first_inside - last_outside
                all_in_one[index, 3] = import_time
                import_time_list = np.append(import_time_list, import_time)
                print('import time =', import_time)
                # Now we have several options after the import:
                # 1. Did it go back outside the nucleus, i.e., is there an outside_release after this and stayed there?
                if len(outside_release) > 0 and np.any(outside_release[first_inside:]):
                    print("Option 3: The particle went back outside the nucleus.")

                    import_after_exit = None
                    for i, is_inside in enumerate(inside_release[first_inside + 1:]):
                        if is_inside:
                            import_after_exit = i + first_inside
                            break

                    if import_after_exit is not None:
                        print("molecule  event detected after going back inside again the nucleus at frame:",
                              import_after_exit)
                        print("If this happens frequently, we need to think about it")
                        status_import = np.append(status_import, 4)  # Append to NumPy array
                        all_in_one[index, 2] = 4
                    else:
                        status_import = np.append(status_import, 3)  # Append to NumPy array
                        all_in_one[index, 2] = 3

                # 2. did it release completely inside the nucleus (is the last frame in the nucleus > threshold)
                elif inside_release[-1]:
                    print('particle got released')
                    status_import = np.append(status_import, 1)  # Append to NumPy array
                    all_in_one[index, 2] = 1
                # 3. did it go back in 'docking' and stayed there (is the last frame in the nucleus < threshold)
                elif not inside_release[-1]:
                    print('particle got back in docking')
                    status_import = np.append(status_import, 2)  # Append to NumPy array
                    all_in_one[index, 2] = 2
                else:
                    assert ('error!')
        return all_in_one


    def make_diffusion_mmsplot(self, df_list, name_list=['GFA', 'MYO', 'TRA'], mode='',linking_distance=5,min_distance=20):

        # Initialize lists to store data for each dataframe
        mean_values_list = []
        std_values_list = []

        fig, (ax, ax1) = plt.subplots(1, 2, dpi=400, figsize=(5, 2.5))
        fig3, ax3 = plt.subplots(1, 1, dpi=400)
        secondary_diffusion = []
        alfa_secondary = []
        MSS_slope = []
        max_distance_from_ne_list = []
        id_list = []
        for df, label in zip(df_list,name_list ):

            filtered_ids = df.groupby('id').filter(
                lambda x: (x['dist_cor_spline'] > 400/self.pixelsize).all() and x['def_inout_spline'].notna().all())

            # Extract unique ids from the filtered DataFrame
            unique_ids = filtered_ids['id'].unique()
            #unique_ids = df['id'].unique()

            dif_total = np.array([])
            distance_total = np.array([])

            for id in tqdm.tqdm(unique_ids):
                try:
                    single_track = df[df['id'].values == id]
                    framestep = (single_track['frame'].values)[1::] - (single_track['frame'].values)[:-1]
                    x_displacement = (single_track[self.x_str].values)[1::] - (single_track[self.x_str].values)[:-1]
                    y_displacement = (single_track[self.y_str].values)[1::] - (single_track[self.y_str].values)[:-1]
                    displacement = np.sqrt(x_displacement ** 2 + y_displacement ** 2)
                    if mode == 'projection':
                        displacement =  abs((single_track['dist_cor'].values)[1::] - (single_track['dist_cor'].values)[:-1])

                    stepsize = displacement / framestep

                    if np.sum(np.isnan(single_track[self.may_inout_str].values)) == 0:
                        inout = (single_track[self.may_inout_str].values * 2) - 1
                        distance_from_ne = single_track['dist'].values * inout
                        distance_from_ne = (distance_from_ne[1::] + distance_from_ne[:-1]) / 2
                    else:
                        inout = (single_track[ self.def_inout_str].values * 2) - 1
                        distance_from_ne = single_track['dist'].values * inout
                        distance_from_ne = (distance_from_ne[1::] + distance_from_ne[:-1]) / 2
                    filter_distance = stepsize < linking_distance
                    if sum(~filter_distance) >0 or len(x_displacement)<min_distance:
                        continue

                    def diffusion_curve(x,D,alfa):
                        return 4*D*x**alfa
                    max_distance_NE = np.max(abs(distance_from_ne)) # but without outliers
                    moment_list = [1,2,3,4,5,6]
                    alfa_list = []
                    for moment in moment_list:

                        track_len = len(stepsize)
                        time_lag = np.arange(1,int(np.round(len(stepsize)-1)))
                        msd_plot = np.zeros(len(time_lag))
                        for lagnum, lag in enumerate(time_lag):
                            moment_per_lag = 0
                            for time in range(track_len-lag):
                                displacement_step = np.sqrt((single_track[self.x_str].values[time+lag] -  single_track[self.x_str].values[time])**2 +
                                               (single_track[self.y_str].values[time+lag] -  single_track[self.y_str].values[time])**2)
                                displacement_step = displacement_step* self.pixelsize/1000
                                if mode == 'projection':
                                    displacement_step = abs(
                                        (single_track['dist_cor'].values)[time+lag] - (single_track['dist_cor'].values)[time])

                                moment_per_lag += displacement_step**moment

                            moment_per_lag = moment_per_lag/(track_len-lag)
                            msd_plot[lagnum] = moment_per_lag
                        time_lag = time_lag*self.frametime
                        parameters, covariance = curve_fit(diffusion_curve, time_lag, msd_plot)
                        generalized_diff_constant =parameters[0]
                        alfa_per_moment = parameters[1]
                        alfa_list.append(alfa_per_moment)

                        if moment ==2:
                            second_dif = generalized_diff_constant
                            second_alfa = alfa_per_moment

                        # x_fit = np.linspace(min(time_lag), max(time_lag), 100)
                        # y_fit = diffusion_curve(x_fit, *parameters)
                        #
                        # plt.scatter(time_lag, msd_plot)
                        # plt.plot(x_fit,y_fit)
                        # plt.show()
                    linear = np.polyfit(moment_list,alfa_list,1)
                    # x_fit = np.linspace(0, max(moment_list), 100)
                    # y_fit = np.polyval(linear,x_fit)
                    #
                    # plt.scatter(moment_list,alfa_list)
                    # plt.plot(x_fit,y_fit)
                    MSS_slope.append(linear[0])
                    max_distance_from_ne_list.append(max_distance_NE)
                    secondary_diffusion.append(second_dif)
                    alfa_secondary.append(second_alfa)
                    id_list.append(id)
                except:
                    print('fitting failed')
                    MSS_slope.append(0)
                    max_distance_from_ne_list.append(0)
                    secondary_diffusion.append(0)
                    alfa_secondary.append(0)
                    id_list.append(id)
        return np.array(secondary_diffusion), np.array(alfa_secondary), np.array(MSS_slope), np.array(max_distance_from_ne_list), np.array(id_list)

    def compute_distance_combinations(self,label='GFA'):
        unique_ids = np.unique(self.transport_events['id'].values)
        file_lookup_filtered = self.lookup[self.lookup['id'].isin(unique_ids)]

        count_per_combination = file_lookup_filtered.groupby(['filename', 'NE']).size().reset_index(name='count')
        counts = count_per_combination['count'].values
        distance_combinations = []

        for i, count in enumerate(counts):
            if count > 1:
                filename = count_per_combination['filename'].values[i]
                NE = count_per_combination['NE'].values[i]
                filtered_rows = file_lookup_filtered[
                    (file_lookup_filtered['filename'] == filename) & (file_lookup_filtered['NE'] == NE)]

                # Extract the 'id' values
                ids = filtered_rows['id'].values
                tracks_temp = self.transport_events[self.transport_events['id'].isin(ids)]
                intersectpoints = tracks_temp[tracks_temp[self.intersect_str] == 1]
                x_coordinates = intersectpoints[self.x_str].values
                y_coordinates = intersectpoints[self.y_str].values
                # Create an array of coordinates
                coordinates = np.column_stack((x_coordinates, y_coordinates))

                # Compute the Euclidean distance matrix
                distance_matrix = cdist(coordinates, coordinates, metric='euclidean')
                lower_triangle_distance = np.tril(distance_matrix, k=-1)
                lower_triangle_flatten = lower_triangle_distance[lower_triangle_distance != 0].flatten()
                distance_combinations.append(lower_triangle_flatten)

        distance_combinations = np.concatenate(distance_combinations)

        # Plot histogram for distances
        plt.figure(dpi=400)
        plt.hist(distance_combinations * self.pixelsize, label=label)
        plt.xlabel('Distance between NE crossings [nm]')
        plt.ylabel('Counts')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot histogram for counts
        plt.figure(dpi=400)
        plt.hist(count_per_combination['count'].values, bins=range(1, max(count_per_combination['count']) + 2),
                 edgecolor='black', align='left', label=label)
        plt.xlabel('Number of transport events per cell')
        plt.ylabel('Counts')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def compute_export_events(self, transport_threshold, release_threshold):

        unique_ids = self.transport_events['id'].unique()
        unique_ids_list = unique_ids.tolist()
        number_export = 0
        cytoplasmic_release = []
        nuclear_docking = []
        retrogate_export = 0
        retrogate_unsucessful_export = 0
        retrogate_export_full_export = 0
        image_list_cytoplasmic_release = []
        image_list_nuclear_docking = []
        image_list_full_export = []
        for index, id_iter in tqdm.tqdm(enumerate(unique_ids_list)):
            lookup = self.lookup[self.lookup['id'] == id_iter]
            filename, ne = lookup[['filename']].values[0][0],lookup[['NE']].values[0][0]
            index_true_condition = self.nelookup.loc[
                (self.nelookup['filename'] == filename) & (self.nelookup['NE'] == ne)].index
            index_true_condition.tolist()
            nedata = self.nedata[index_true_condition[0]]
            single_track = self.transport_events[self.transport_events['id'] == id_iter]
            outside_trans = np.array(single_track['dist_cor'] < -transport_threshold / self.pixelsize)
            inside_trans = np.array(single_track['dist_cor'] > transport_threshold / self.pixelsize)

            inside_release = np.array(single_track['dist_cor'] > release_threshold / self.pixelsize)
            outside_release = np.array(single_track['dist_cor'] < -release_threshold / self.pixelsize)

            if np.sum(outside_trans) > 0 and np.sum(inside_trans) > 0:
                # Define conditions
                conditions = [

                    (single_track['dist_cor'].values > 0) & (
                            single_track['dist_cor'].values <= release_threshold / self.pixelsize),  # Nuclear docking
                    single_track['dist_cor'].values > release_threshold / self.pixelsize,  # Nuclear diffusion
                    (single_track['dist_cor'].values < 0) & (
                            single_track['dist_cor'].values >= -release_threshold / self.pixelsize),  # cytoplasmic docking
                    single_track['dist_cor'].values < -release_threshold / self.pixelsize  # # cytoplasmic diffusion
                ]

                # Define corresponding states
                state_values = [ 2, 3, 4, 5]
                state_labels = {
                    2: 'Nuclear Docking',
                    3: 'Nuclear Diffusion',
                    4: 'Cytoplasmic Docking',
                    5: 'Cytoplasmic Diffusion'
                }
                # Use np.select to create the state_array
                state_array = np.select(conditions, state_values, default=0)

                index_inside = np.where(inside_trans)[0]
                index_outside = np.where(outside_trans)[0]

                # Determine which array has the first True event - starts inside
                if len(index_inside) > 0 and index_inside[0] < index_outside[0]:
                    first_outside = index_outside[0]
                    # check for release
                    inside_release = np.array(single_track['dist_cor'] > release_threshold / self.pixelsize)
                    outside_release = np.array(single_track['dist_cor'] < -release_threshold / self.pixelsize)
                    frame_alpha_ori = (single_track['frame'].values - min(single_track['frame'].values))

                    # compute cytoplasmic release
                    if np.sum(outside_release) > 0:
                        last_inside = np.where(inside_trans)[0][-1]
                        last_release = np.where(outside_release)[0][-1]
                        if last_inside > last_release:
                            retrogate_export += 1
                        else:
                            cytoplasmic_temp = np.sum((single_track['dist_cor'].values < 0) & (
                                    single_track['dist_cor'].values > -release_threshold / self.pixelsize))
                            cytoplasmic_release.append(cytoplasmic_temp)

                            filter_arr = (single_track['dist_cor'].values < 0) & (
                                    single_track['dist_cor'].values > -release_threshold / self.pixelsize)
                            eucdist = np.sqrt((single_track[self.x_str].values) ** 2 + (single_track[self.y_str].values) ** 2)
                            eucdist_tot = np.cumsum(abs(eucdist[1::] - eucdist[0:-1]))
                            distance_tot = np.round(eucdist_tot[-1], 2)

                            number_export += 1

                            plt.figure(dpi=200, figsize=(3, 3))
                            frame_ori = (single_track['frame'].values - min(single_track['frame'].values))

                            for state_value in set(state_values):
                                bol = np.array(state_array == state_value)
                                frame = frame_ori[bol]

                                x_values = single_track.loc[state_array == state_value, self.x_str]
                                y_values = single_track.loc[state_array == state_value, self.y_str]

                                plt.scatter(
                                    x_values,
                                    y_values,
                                    label=state_labels[state_value],
                                    alpha=0.8  # Adjust alpha for transparency if needed
                                )
                                plt.plot(nedata[0,:],nedata[1,:], 'k')
                                # Add text labels to each spot using frame_alpha_ori values
                                for i, (x, y, alpha_value) in enumerate(zip(x_values, y_values, frame)):
                                    plt.text(x, y, f'{alpha_value}', fontsize=8, color='black', ha='center',
                                             va='center')

                            legend = plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
                            plt.title(
                                f'Cytoplasmic release Time\n {cytoplasmic_temp} frames \n Distance = \n{distance_tot} px')
                            # plt.text(1, 1, title_text, fontsize=12, ha='left', va='bottom')
                            #skimage.measure.points_in_poly(np.array([x_values,y_values]), )
                            plt.xlabel('x [px]')
                            plt.ylabel('y [px]')
                            # plt.title('Scatter Plot with States')
                            plt.tight_layout(pad=0.1)
                            # Save the figure to the buffer
                            buffer = io.BytesIO()
                            plt.savefig(buffer, format='tiff', bbox_inches='tight')
                            plt.close()  # Close the figure to free up resources

                            # Rewind the buffer
                            buffer.seek(0)

                            # Read the image from the buffer and append it to the list
                            img = tifffile.imread(buffer)
                            image_list_cytoplasmic_release.append(img)
                    # compute nuclear docking
                    if np.sum(inside_release) > 0:
                        last_oustide = np.where(outside_trans)[0][-1]
                        first_release = np.where(inside_release)[0][0]
                        last_release = np.where(inside_release)[0][-1]
                        if last_release > last_oustide:
                            retrogate_unsucessful_export += 1
                        else:
                            nuclear_temp = np.sum((single_track['dist_cor'].values > 0) & (
                                    single_track['dist_cor'].values < release_threshold / self.pixelsize))

                            filter_arr = (single_track['dist_cor'].values > 0) & (
                                    single_track['dist_cor'].values < release_threshold / self.pixelsize)
                            eucdist = np.sqrt((single_track[self.x_str].values) ** 2 + (single_track[self.y_str].values) ** 2)
                            eucdist_tot = np.cumsum(abs(eucdist[1::] - eucdist[0:-1]))
                            distance_tot = np.round(eucdist_tot[-1], 2)

                            nuclear_docking.append(nuclear_temp)
                            plt.figure(dpi=200, figsize=(3, 3))
                            frame_ori = (single_track['frame'].values - min(single_track['frame'].values))

                            for state_value in set(state_values):
                                bol = np.array(state_array == state_value)
                                frame = frame_ori[bol]

                                x_values = single_track.loc[state_array == state_value, self.x_str]
                                y_values = single_track.loc[state_array == state_value, self.y_str]

                                plt.scatter(
                                    x_values,
                                    y_values,
                                    label=state_labels[state_value],
                                    alpha=0.8  # Adjust alpha for transparency if needed
                                )

                                # Add text labels to each spot using frame_alpha_ori values
                                for i, (x, y, alpha_value) in enumerate(zip(x_values, y_values, frame)):
                                    plt.text(x, y, f'{alpha_value}', fontsize=8, color='black', ha='center',
                                             va='center')
                            legend = plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
                            plt.title(f'Nuclear Docking Time\n {nuclear_temp} frames \n Distance = \n{distance_tot} px')
                            # plt.text(1, 1, title_text, fontsize=12, ha='left', va='bottom')

                            plt.xlabel('x [px]')
                            plt.ylabel('y [px]')
                            # plt.title('Scatter Plot with States')
                            plt.tight_layout(pad=0.1)
                            # Save the figure to the buffer
                            buffer = io.BytesIO()
                            plt.savefig(buffer, format='tiff', bbox_inches='tight')
                            plt.close()  # Close the figure to free up resources

                            # Rewind the buffer
                            buffer.seek(0)

                            # Read the image from the buffer and append it to the list
                            img = tifffile.imread(buffer)
                            image_list_nuclear_docking.append(img)
                        # counts_between

                # define full export event
                if np.sum(inside_release) > 0 and np.sum(outside_release) > 0:
                    index_inside = np.where(inside_release)[0]
                    index_outside = np.where(outside_release)[0]
                    #####################################################
                    last_oustide = np.where(outside_release)[0][-1]
                    first_release = np.where(inside_release)[0][0]
                    last_release = np.where(inside_release)[0][-1]
                    if last_release > last_oustide:
                        retrogate_export_full_export += 1
                    else:
                        plt.figure(dpi=200, figsize=(3, 3))
                        frame_ori = (single_track['frame'].values - min(single_track['frame'].values))

                        for state_value in set(state_values):
                            bol = np.array(state_array == state_value)
                            frame = frame_ori[bol]

                            x_values = single_track.loc[state_array == state_value, self.x_str]
                            y_values = single_track.loc[state_array == state_value, self.y_str]

                            plt.scatter(
                                x_values,
                                y_values,
                                label=state_labels[state_value],
                                alpha=0.8  # Adjust alpha for transparency if needed
                            )

                            # Add text labels to each spot using frame_alpha_ori values
                            for i, (x, y, alpha_value) in enumerate(zip(x_values, y_values, frame)):
                                plt.text(x, y, f'{alpha_value}', fontsize=8, color='black', ha='center',
                                         va='center')
                        legend = plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
                        # plt.text(1, 1, title_text, fontsize=12, ha='left', va='bottom')

                        plt.xlabel('x [px]')
                        plt.ylabel('y [px]')
                        # plt.title('Scatter Plot with States')
                        plt.tight_layout(pad=0.1)
                        # Save the figure to the buffer
                        buffer = io.BytesIO()
                        plt.savefig(buffer, format='tiff', bbox_inches='tight')
                        plt.close()  # Close the figure to free up resources

                        # Rewind the buffer
                        buffer.seek(0)

                        # Read the image from the buffer and append it to the list
                        img = tifffile.imread(buffer)
                        image_list_full_export.append(img)

        min_height = min(img.shape[0] for img in image_list_nuclear_docking)
        min_width = min(img.shape[1] for img in image_list_nuclear_docking)
        image_nucdock_resized = [img[:min_height, :min_width, :] for img in image_list_nuclear_docking]
        # Create a 3D NumPy array by stacking along a new dimension
        image_nucdock_resized = np.stack(image_nucdock_resized, axis=0)
        min_height = min(img.shape[0] for img in image_list_cytoplasmic_release)
        min_width = min(img.shape[1] for img in image_list_cytoplasmic_release)
        image_cyto_release_resized = [img[:min_height, :min_width, :] for img in image_list_cytoplasmic_release]
        image_cyto_release_resized = np.stack(image_cyto_release_resized, axis=0)

        min_height = min(img.shape[0] for img in image_list_full_export)
        min_width = min(img.shape[1] for img in image_list_full_export)
        image_export_resized = [img[:min_height, :min_width, :] for img in image_list_full_export]
        image_export_resized = np.stack(image_export_resized, axis=0)

        return image_nucdock_resized,image_cyto_release_resized,image_export_resized, cytoplasmic_release


    def compute_import_events(self, transport_threshold, release_threshold):

        unique_ids = self.transport_events['id'].unique()
        unique_ids_list = unique_ids.tolist()
        number_export = 0
        cytoplasmic_docking = []
        nuclear_release = []
        retrogate_export = 0
        retrogate_unsucessful_export = 0
        retrogate_export_full_export = 0
        image_list_cytoplasmic_docking = []
        image_list_nuclear_release = []
        image_list_full_export = []
        for index, id_iter in tqdm.tqdm(enumerate(unique_ids_list)):

            single_track = self.transport_events[self.transport_events['id'] == id_iter]
            outside_trans = np.array(single_track['dist_cor'] < -transport_threshold / self.pixelsize)
            inside_trans = np.array(single_track['dist_cor'] > transport_threshold / self.pixelsize)

            inside_release = np.array(single_track['dist_cor'] > release_threshold / self.pixelsize)
            outside_release = np.array(single_track['dist_cor'] < -release_threshold / self.pixelsize)

            if np.sum(outside_trans) > 0 and np.sum(inside_trans) > 0:
                # Define conditions
                conditions = [
                    single_track['dist'].values < transport_threshold / self.pixelsize,  # transition
                    (single_track['dist_cor'].values > transport_threshold / self.pixelsize) & (
                            single_track['dist_cor'].values < release_threshold / self.pixelsize),  # Nuclear docking
                    single_track['dist_cor'].values > release_threshold / self.pixelsize,  # Nuclear diffusion
                    (single_track['dist_cor'].values < -transport_threshold / self.pixelsize) & (
                            single_track['dist_cor'].values > -release_threshold / self.pixelsize),  # cytoplasmic docking
                    single_track['dist_cor'].values < -release_threshold / self.pixelsize  # # cytoplasmic diffusion
                ]

                # Define corresponding states
                state_values = [1, 2, 3, 4, 5]
                state_labels = {
                    1: 'Transition',
                    2: 'Nuclear Docking',
                    3: 'Nuclear Diffusion',
                    4: 'Cytoplasmic Docking',
                    5: 'Cytoplasmic Diffusion'
                }
                # Use np.select to create the state_array
                state_array = np.select(conditions, state_values, default=0)

                index_inside = np.where(inside_trans)[0]
                index_outside = np.where(outside_trans)[0]

                # Determine which array has the first True event - starts outside
                if len(index_inside) > 0 and index_inside[0] > index_outside[0]:
                    first_outside = index_outside[0]
                    # check for release
                    inside_release = np.array(single_track['dist_cor'] > release_threshold / self.pixelsize)
                    outside_release = np.array(single_track['dist_cor'] < -release_threshold / self.pixelsize)

                    # compute cytoplasmic release
                    if np.sum(outside_release) > 0:
                        last_inside = np.where(inside_trans)[0][-1]
                        last_release = np.where(outside_release)[0][-1]
                        if last_inside > last_release:
                            retrogate_export += 1
                        else:
                            cytoplasmic_temp = np.sum((single_track['dist_cor'].values < 0) & (
                                    single_track['dist_cor'].values > -release_threshold / self.pixelsize))
                            cytoplasmic_docking.append(cytoplasmic_temp)

                            filter_arr = (single_track['dist_cor'].values < 0) & (
                                    single_track['dist_cor'].values > -release_threshold / self.pixelsize)
                            eucdist = np.sqrt((single_track[self.x_str].values) ** 2 + (single_track[self.y_str].values) ** 2)
                            eucdist_tot = np.cumsum(abs(eucdist[1::] - eucdist[0:-1]))
                            distance_tot = np.round(eucdist_tot[-1], 2)

                            number_export += 1

                            plt.figure(dpi=200, figsize=(3, 3))
                            for state_value in set(state_values):
                                plt.scatter(
                                    single_track.loc[state_array == state_value, self.x_str],
                                    single_track.loc[state_array == state_value, self.y_str],
                                    label=state_labels[state_value],
                                    alpha=0.7  # Adjust alpha for transparency if needed
                                )
                            legend = plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
                            plt.title(
                                f'Cytoplasmic release Time\n {cytoplasmic_temp} frames \n Distance = \n{distance_tot} px')
                            # plt.text(1, 1, title_text, fontsize=12, ha='left', va='bottom')

                            plt.xlabel('x [px]')
                            plt.ylabel('y [px]')
                            # plt.title('Scatter Plot with States')
                            plt.tight_layout(pad=0.1)
                            # Save the figure to the buffer
                            buffer = io.BytesIO()
                            plt.savefig(buffer, format='tiff', bbox_inches='tight')
                            plt.close()  # Close the figure to free up resources

                            # Rewind the buffer
                            buffer.seek(0)

                            # Read the image from the buffer and append it to the list
                            img = tifffile.imread(buffer)
                            image_list_cytoplasmic_docking.append(img)
                    # compute nuclear docking
                    if np.sum(inside_release) > 0:
                        last_oustide = np.where(outside_trans)[0][-1]
                        first_release = np.where(inside_release)[0][0]
                        last_release = np.where(inside_release)[0][-1]
                        if last_release > last_oustide:
                            retrogate_unsucessful_export += 1
                        else:
                            nuclear_temp = np.sum((single_track['dist_cor'].values > 0) & (
                                    single_track['dist_cor'].values < release_threshold / self.pixelsize))

                            filter_arr = (single_track['dist_cor'].values > 0) & (
                                    single_track['dist_cor'].values < release_threshold / self.pixelsize)
                            eucdist = np.sqrt((single_track[self.x_str].values) ** 2 + (single_track[self.y_str].values) ** 2)
                            eucdist_tot = np.cumsum(abs(eucdist[1::] - eucdist[0:-1]))
                            distance_tot = np.round(eucdist_tot[-1], 2)

                            nuclear_release.append(nuclear_temp)
                            plt.figure(dpi=200, figsize=(3, 3))
                            for state_value in set(state_values):
                                plt.scatter(
                                    single_track.loc[state_array == state_value, self.x_str],
                                    single_track.loc[state_array == state_value, self.y_str],
                                    label=state_labels[state_value],
                                    alpha=0.7  # Adjust alpha for transparency if needed
                                )
                            legend = plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
                            plt.title(f'Nuclear Docking Time\n {nuclear_temp} frames \n Distance = \n{distance_tot} px')
                            # plt.text(1, 1, title_text, fontsize=12, ha='left', va='bottom')

                            plt.xlabel('x [px]')
                            plt.ylabel('y [px]')
                            # plt.title('Scatter Plot with States')
                            plt.tight_layout(pad=0.1)
                            # Save the figure to the buffer
                            buffer = io.BytesIO()
                            plt.savefig(buffer, format='tiff', bbox_inches='tight')
                            plt.close()  # Close the figure to free up resources

                            # Rewind the buffer
                            buffer.seek(0)

                            # Read the image from the buffer and append it to the list
                            img = tifffile.imread(buffer)
                            image_list_nuclear_release.append(img)
                        # counts_between

                # define full export event
                if np.sum(inside_release) > 0 and np.sum(outside_release) > 0:
                    index_inside = np.where(inside_release)[0]
                    index_outside = np.where(outside_release)[0]
                    #####################################################
                    last_oustide = np.where(outside_release)[0][-1]
                    first_release = np.where(inside_release)[0][0]
                    last_release = np.where(inside_release)[0][-1]
                    if last_release < last_oustide:
                        retrogate_export_full_export += 1
                    else:
                        plt.figure(dpi=200, figsize=(3, 3))
                        for state_value in set(state_values):
                            plt.scatter(
                                single_track.loc[state_array == state_value, self.x_str],
                                single_track.loc[state_array == state_value, self.y_str],
                                label=state_labels[state_value],
                                alpha=0.7  # Adjust alpha for transparency if needed
                            )
                        legend = plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
                        # plt.text(1, 1, title_text, fontsize=12, ha='left', va='bottom')

                        plt.xlabel('x [px]')
                        plt.ylabel('y [px]')
                        # plt.title('Scatter Plot with States')
                        plt.tight_layout(pad=0.1)
                        # Save the figure to the buffer
                        buffer = io.BytesIO()
                        plt.savefig(buffer, format='tiff', bbox_inches='tight')
                        plt.close()  # Close the figure to free up resources

                        # Rewind the buffer
                        buffer.seek(0)

                        # Read the image from the buffer and append it to the list
                        img = tifffile.imread(buffer)
                        image_list_full_export.append(img)

        min_height = min(img.shape[0] for img in image_list_nuclear_release)
        min_width = min(img.shape[1] for img in image_list_nuclear_release)
        image_nucdock_resized = [img[:min_height, :min_width, :] for img in image_list_nuclear_release]
        # Create a 3D NumPy array by stacking along a new dimension
        image_nucdock_resized = np.stack(image_nucdock_resized, axis=0)
        min_height = min(img.shape[0] for img in image_list_cytoplasmic_docking)
        min_width = min(img.shape[1] for img in image_list_cytoplasmic_docking)
        image_cyto_release_resized = [img[:min_height, :min_width, :] for img in image_list_cytoplasmic_docking]
        image_cyto_release_resized = np.stack(image_cyto_release_resized, axis=0)

        min_height = min(img.shape[0] for img in image_list_full_export)
        min_width = min(img.shape[1] for img in image_list_full_export)
        image_export_resized = [img[:min_height, :min_width, :] for img in image_list_full_export]
        image_export_resized = np.stack(image_export_resized, axis=0)

        return image_nucdock_resized,image_cyto_release_resized,image_export_resized


    def compute_docking_events(self, docking, release_threshold):

        unique_ids = self.transport_events['id'].unique()
        unique_ids_list = unique_ids.tolist()
        number_export = 0
        cytoplasmic_docking = []
        nuclear_release = []
        retrogate_export = 0
        retrogate_unsucessful_export = 0
        retrogate_export_full_export = 0
        image_list_cytoplasmic_docking = []
        image_list_nuclear_release = []
        image_list_full_export = []
        for index, id_iter in tqdm.tqdm(enumerate(unique_ids_list)):

            single_track = self.transport_events[self.transport_events['id'] == id_iter]
            outside_trans = np.array(single_track['dist_cor'] < -transport_threshold / self.pixelsize)
            inside_trans = np.array(single_track['dist_cor'] > transport_threshold / self.pixelsize)

            inside_release = np.array(single_track['dist_cor'] > release_threshold / self.pixelsize)
            outside_release = np.array(single_track['dist_cor'] < -release_threshold / self.pixelsize)

            if np.sum(outside_trans) > 0 and np.sum(inside_trans) > 0:
                # Define conditions
                conditions = [
                    single_track['dist'].values < transport_threshold / self.pixelsize,  # transition
                    (single_track['dist_cor'].values > transport_threshold / self.pixelsize) & (
                            single_track['dist_cor'].values < release_threshold / self.pixelsize),  # Nuclear docking
                    single_track['dist_cor'].values > release_threshold / self.pixelsize,  # Nuclear diffusion
                    (single_track['dist_cor'].values < -transport_threshold / self.pixelsize) & (
                            single_track['dist_cor'].values > -release_threshold / self.pixelsize),  # cytoplasmic docking
                    single_track['dist_cor'].values < -release_threshold / self.pixelsize  # # cytoplasmic diffusion
                ]

                # Define corresponding states
                state_values = [1, 2, 3, 4, 5]
                state_labels = {
                    1: 'Transition',
                    2: 'Nuclear Docking',
                    3: 'Nuclear Diffusion',
                    4: 'Cytoplasmic Docking',
                    5: 'Cytoplasmic Diffusion'
                }
                # Use np.select to create the state_array
                state_array = np.select(conditions, state_values, default=0)

                index_inside = np.where(inside_trans)[0]
                index_outside = np.where(outside_trans)[0]

                # Determine which array has the first True event - starts outside
                if len(index_inside) > 0 and index_inside[0] > index_outside[0]:
                    first_outside = index_outside[0]
                    # check for release
                    inside_release = np.array(single_track['dist_cor'] > release_threshold / self.pixelsize)
                    outside_release = np.array(single_track['dist_cor'] < -release_threshold / self.pixelsize)

                    # compute cytoplasmic release
                    if np.sum(outside_release) > 0:
                        last_inside = np.where(inside_trans)[0][-1]
                        last_release = np.where(outside_release)[0][-1]
                        if last_inside > last_release:
                            retrogate_export += 1
                        else:
                            cytoplasmic_temp = np.sum((single_track['dist_cor'].values < 0) & (
                                    single_track['dist_cor'].values > -release_threshold / self.pixelsize))
                            cytoplasmic_docking.append(cytoplasmic_temp)

                            filter_arr = (single_track['dist_cor'].values < 0) & (
                                    single_track['dist_cor'].values > -release_threshold / self.pixelsize)
                            eucdist = np.sqrt((single_track[self.x_str].values) ** 2 + (single_track[self.y_str].values) ** 2)
                            eucdist_tot = np.cumsum(abs(eucdist[1::] - eucdist[0:-1]))
                            distance_tot = np.round(eucdist_tot[-1], 2)

                            number_export += 1

                            plt.figure(dpi=200, figsize=(3, 3))
                            for state_value in set(state_values):
                                plt.scatter(
                                    single_track.loc[state_array == state_value, self.x_str],
                                    single_track.loc[state_array == state_value, self.y_str],
                                    label=state_labels[state_value],
                                    alpha=0.7  # Adjust alpha for transparency if needed
                                )
                            legend = plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
                            plt.title(
                                f'Cytoplasmic release Time\n {cytoplasmic_temp} frames \n Distance = \n{distance_tot} px')
                            # plt.text(1, 1, title_text, fontsize=12, ha='left', va='bottom')

                            plt.xlabel('x [px]')
                            plt.ylabel('y [px]')
                            # plt.title('Scatter Plot with States')
                            plt.tight_layout(pad=0.1)
                            # Save the figure to the buffer
                            buffer = io.BytesIO()
                            plt.savefig(buffer, format='tiff', bbox_inches='tight')
                            plt.close()  # Close the figure to free up resources

                            # Rewind the buffer
                            buffer.seek(0)

                            # Read the image from the buffer and append it to the list
                            img = tifffile.imread(buffer)
                            image_list_cytoplasmic_docking.append(img)
                    # compute nuclear docking
                    if np.sum(inside_release) > 0:
                        last_oustide = np.where(outside_trans)[0][-1]
                        first_release = np.where(inside_release)[0][0]
                        last_release = np.where(inside_release)[0][-1]
                        if last_release > last_oustide:
                            retrogate_unsucessful_export += 1
                        else:
                            nuclear_temp = np.sum((single_track['dist_cor'].values > 0) & (
                                    single_track['dist_cor'].values < release_threshold / self.pixelsize))

                            filter_arr = (single_track['dist_cor'].values > 0) & (
                                    single_track['dist_cor'].values < release_threshold / self.pixelsize)
                            eucdist = np.sqrt((single_track[self.x_str].values) ** 2 + (single_track[self.y_str].values) ** 2)
                            eucdist_tot = np.cumsum(abs(eucdist[1::] - eucdist[0:-1]))
                            distance_tot = np.round(eucdist_tot[-1], 2)

                            nuclear_release.append(nuclear_temp)
                            plt.figure(dpi=200, figsize=(3, 3))
                            for state_value in set(state_values):
                                plt.scatter(
                                    single_track.loc[state_array == state_value, self.x_str],
                                    single_track.loc[state_array == state_value, self.y_str],
                                    label=state_labels[state_value],
                                    alpha=0.7  # Adjust alpha for transparency if needed
                                )
                            legend = plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
                            plt.title(f'Nuclear Docking Time\n {nuclear_temp} frames \n Distance = \n{distance_tot} px')
                            # plt.text(1, 1, title_text, fontsize=12, ha='left', va='bottom')

                            plt.xlabel('x [px]')
                            plt.ylabel('y [px]')
                            # plt.title('Scatter Plot with States')
                            plt.tight_layout(pad=0.1)
                            # Save the figure to the buffer
                            buffer = io.BytesIO()
                            plt.savefig(buffer, format='tiff', bbox_inches='tight')
                            plt.close()  # Close the figure to free up resources

                            # Rewind the buffer
                            buffer.seek(0)

                            # Read the image from the buffer and append it to the list
                            img = tifffile.imread(buffer)
                            image_list_nuclear_release.append(img)
                        # counts_between

                # define full export event
                if np.sum(inside_release) > 0 and np.sum(outside_release) > 0:
                    index_inside = np.where(inside_release)[0]
                    index_outside = np.where(outside_release)[0]
                    #####################################################
                    last_oustide = np.where(outside_release)[0][-1]
                    first_release = np.where(inside_release)[0][0]
                    last_release = np.where(inside_release)[0][-1]
                    if last_release < last_oustide:
                        retrogate_export_full_export += 1
                    else:
                        plt.figure(dpi=200, figsize=(3, 3))
                        for state_value in set(state_values):
                            plt.scatter(
                                single_track.loc[state_array == state_value, self.x_str],
                                single_track.loc[state_array == state_value, self.y_str],
                                label=state_labels[state_value],
                                alpha=0.7  # Adjust alpha for transparency if needed
                            )
                        legend = plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
                        # plt.text(1, 1, title_text, fontsize=12, ha='left', va='bottom')

                        plt.xlabel('x [px]')
                        plt.ylabel('y [px]')
                        # plt.title('Scatter Plot with States')
                        plt.tight_layout(pad=0.1)
                        # Save the figure to the buffer
                        buffer = io.BytesIO()
                        plt.savefig(buffer, format='tiff', bbox_inches='tight')
                        plt.close()  # Close the figure to free up resources

                        # Rewind the buffer
                        buffer.seek(0)

                        # Read the image from the buffer and append it to the list
                        img = tifffile.imread(buffer)
                        image_list_full_export.append(img)

        min_height = min(img.shape[0] for img in image_list_nuclear_release)
        min_width = min(img.shape[1] for img in image_list_nuclear_release)
        image_nucdock_resized = [img[:min_height, :min_width, :] for img in image_list_nuclear_release]
        # Create a 3D NumPy array by stacking along a new dimension
        image_nucdock_resized = np.stack(image_nucdock_resized, axis=0)
        min_height = min(img.shape[0] for img in image_list_cytoplasmic_docking)
        min_width = min(img.shape[1] for img in image_list_cytoplasmic_docking)
        image_cyto_release_resized = [img[:min_height, :min_width, :] for img in image_list_cytoplasmic_docking]
        image_cyto_release_resized = np.stack(image_cyto_release_resized, axis=0)

        min_height = min(img.shape[0] for img in image_list_full_export)
        min_width = min(img.shape[1] for img in image_list_full_export)
        image_export_resized = [img[:min_height, :min_width, :] for img in image_list_full_export]
        image_export_resized = np.stack(image_export_resized, axis=0)

        return image_nucdock_resized,image_cyto_release_resized,image_export_resized

    def plot_dwell_time(self, df_list, num_bins=30,range_docking=[-250,0],
                           xlabel='Dwell time [ms]', ylabel='Counts',
                           name_list=['GFA', 'MYO', 'TRA'], title='Transport events'):

        def find_sequence_lengths(boolean_array):
            boolean_array = np.asarray(boolean_array)

            # Find True positions
            true_positions = np.where(boolean_array)[0]
            if np.sum(true_positions)>0:
                # Find differences between consecutive True positions
                diffs = np.diff(true_positions)

                # Find indices where differences are greater than 1 (indicating a new sequence)
                sequence_starts = np.where(diffs > 1)[0] + 1

                # Include the start and end indices of the sequences
                sequence_indices = np.concatenate([[0], sequence_starts, [len(boolean_array)]])

                # Calculate lengths of sequences
                sequence_lengths = np.diff(sequence_indices)
            else:
                sequence_lengths = []
            return sequence_lengths

        # Initialize lists to store data for each dataframe
        mean_values_list = []
        std_values_list = []

        plt.figure(dpi=300)
        for df, label in zip(df_list, name_list):
            unique_ids = df['id'].unique()
            dif_total = np.array([])
            docking_total = np.array([])

            for id in tqdm.tqdm(unique_ids):
                single_track = df[df['id'].values == id]
                distance_from_ne = single_track[self.dist_cor_str]
                docking = np.logical_and(range_docking[0]  < distance_from_ne,
                                         distance_from_ne < range_docking[1] )

                sequences = np.split(docking, np.where(np.diff(docking) != 0)[0] + 1)
                times = []
                for seq in sequences:
                    if np.array(seq)[0] == True:
                        times.append(len(seq))

                timepoint = []
                cumul_times=[]
                for timepoint in times:
                    cumul_times_iter = np.arange(1,timepoint+1)

                    cumul_times = np.concatenate((cumul_times,cumul_times_iter) )
                docking_total = np.concatenate((docking_total, cumul_times))
            docking_total = docking_total * self.frametime* 1000
            hist, bins, patches = plt.hist(docking_total , bins=num_bins,label=label,alpha=0.5,range=[0,2000])
            hist_color = patches[0].get_facecolor()
            # Define the exponential function
            def exponential_func(x, a, b):
                return a * np.exp(-b * x)

            # Calculate bin centers
            bin_centers = (bins[:-1] + bins[1:]) / 2
            # Initial estimate based on the data
            initial_a = np.max(hist)
            initial_b = 1 / np.mean(docking_total)

            # Fit the exponential curve to the histogram data with initial estimates
            params, covariance = curve_fit(exponential_func, bin_centers, hist, p0=[initial_a, initial_b])

            # Plot the fitted curve and include parameters in the label
            fitted_curve_label = f' {label} $a \\cdot e^{{-x/b}}$\n $b = {1/params[1]:.3f}$'
            plt.plot(bin_centers, exponential_func(bin_centers, *params), color=hist_color, linestyle='-', label=fitted_curve_label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()

        plt.tight_layout()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"../results/dwell_time_{current_time}.svg"
        plt.savefig(file_path, format='svg')
        plt.show()







def check_intersection(A, B, C, D):
    """Return true if line segments AB and CD intersect."""

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def compute_normal(A, B):
    """Compute the normal vector for line segment AB."""
    AB = B - A
    normal = np.array([-AB[1], AB[0]])
    return normal / np.linalg.norm(normal)


def find_intersections_and_normals(nepoints, x_coordinates_rna, y_coordinates_rna):
    intersections = []
    normals = []
    center_of_mass = np.mean(nepoints, axis=1)

    # Prepare RNA segments
    rna_segments = [(np.array([x_coordinates_rna[i], y_coordinates_rna[i]]),
                     np.array([x_coordinates_rna[i + 1], y_coordinates_rna[i + 1]]))
                    for i in range(len(x_coordinates_rna) - 1)]

    # Check intersections with nuclear envelope
    for i in range(nepoints.shape[1] - 1):
        A = nepoints[:, i]
        B = nepoints[:, i + 1]

        for C, D in rna_segments:
            if check_intersection(A, B, C, D):
                midpoint = (A + B) / 2
                normal = compute_normal(A, B)

                # Ensure the normal points outward from the center of mass
                if np.dot(normal, midpoint - center_of_mass) > 0:
                    normal = -normal  # Reverse the normal if it points towards the center

                intersections.append((A,B))  # Using midpoint for visualization
                normals.append(normal)

    return intersections, normals


def calculate_distance_midpoint_and_normal_along_envelope(nepoints, intersections):
    total_distance = 0
    intersection_indices = []
    center_of_mass = np.mean(nepoints, axis=1)

    # Find closest NPC segment for each intersection
    for inter in intersections:
        C, D = inter
        midpoint_rna = (C + D) / 2
        min_dist = np.inf
        closest_segment_index = None

        for i in range(nepoints.shape[1] - 1):
            A = nepoints[:, i]
            B = nepoints[:, i + 1]
            midpoint_npc = (A + B) / 2
            dist = np.linalg.norm(midpoint_rna - midpoint_npc)
            if dist < min_dist:
                min_dist = dist
                closest_segment_index = i

        if closest_segment_index is not None:
            intersection_indices.append(closest_segment_index)

    intersection_indices.sort()

    # Calculate distances along the envelope
    distances = [0]
    for i in range(intersection_indices[0], intersection_indices[1]):
        A = nepoints[:, i]
        B = nepoints[:, i + 1]
        segment_length = np.linalg.norm(B - A)
        total_distance += segment_length
        distances.append(total_distance)

    # Find the midpoint's index along the NE points
    half_distance = total_distance / 2
    midpoint_index = None
    for i, d in enumerate(distances):
        if d >= half_distance:
            midpoint_index = intersection_indices[0] + i - 1
            break

    # Linearly interpolate to find the exact midpoint position
    if midpoint_index is not None and midpoint_index + 1 < nepoints.shape[1]:
        A = nepoints[:, midpoint_index]
        B = nepoints[:, midpoint_index + 1]

        distance_diff = distances[i] - distances[i - 1]
        if distance_diff != 0:  # Check to prevent division by zero
            ratio = (half_distance - distances[i - 1]) / distance_diff
        else:
            ratio = 0  # If A and B
        midpoint_position = A + ratio * (B - A)

        # Calculate the tangent vector at the midpoint
        tangent_vector = B - A
        # Normalize the tangent vector
        tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)
        # Calculate the normal vector by rotating the tangent vector 90 degrees
        normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
    else:
        midpoint_position = np.array([np.nan, np.nan])  # Fallback in case of an error
        normal_vector = np.array([np.nan, np.nan])
    if np.dot(normal_vector, midpoint_position - center_of_mass) > 0:
        normal_vector = -normal_vector  # Reverse the normal if it points towards the center

    return total_distance, midpoint_position, normal_vector
def calculate_distance_along_envelope(nepoints, intersections):
    total_distance = 0
    intersection_indices = []

    # Find closest NPC segment for each intersection
    for inter in intersections:
        C, D = inter
        midpoint_rna = (C + D) / 2
        min_dist = np.inf
        closest_segment_index = None

        for i in range(nepoints.shape[1] - 1):
            A = nepoints[:, i]
            B = nepoints[:, i + 1]
            midpoint_npc = (A + B) / 2
            dist = np.linalg.norm(midpoint_rna - midpoint_npc)
            if dist < min_dist:
                min_dist = dist
                closest_segment_index = i

        if closest_segment_index is not None:
            intersection_indices.append(closest_segment_index)

    # Ensure the indices are sorted to calculate the distance correctly
    intersection_indices.sort()

    # Calculate distance along the envelope
    for i in range(intersection_indices[0], intersection_indices[1]):
        A = nepoints[:, i]
        B = nepoints[:, i + 1]
        total_distance += np.linalg.norm(B - A)

    return total_distance

# Re-defining the necessary functions due to reset
def rotate_points(points, angle):
    """Rotate points by a given angle around the origin."""
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    return np.dot(points, rotation_matrix.T)

def transform_coordinate_system(nepoints, x_coordinates_rna, y_coordinates_rna,
                                intersections, normals, normal_direction='negative'):
    # Translation to bring the first intersection to the origin
    com = np.mean(nepoints, axis=1)
    #first_intersection = (intersections[0][0] + intersections[0][1]) / 2
    first_intersection = intersections
    vector_to_com = com - first_intersection
    # if np.dot(normals[0], vector_to_com) < 0 and normal_direction=='negative':
    #     print('Error! The normal vector does not point away from the CoM as expected.')
    # if np.dot(normals[0], vector_to_com) > 0 and normal_direction=='positve':
    #     print('Error! The normal vector does not point away from the CoM as expected.')
    if np.dot(normals, vector_to_com) < 0 and normal_direction=='negative':
        print('Error! The normal vector does not point away from the CoM as expected.')
    if np.dot(normals, vector_to_com) > 0 and normal_direction=='positve':
        print('Error! The normal vector does not point away from the CoM as expected.')

    nepoints_translated = nepoints - first_intersection.reshape(2, 1)
    rna_translated = np.vstack((x_coordinates_rna, y_coordinates_rna)) - first_intersection.reshape(2, 1)



    # Determine the angle for rotation
    first_normal = normals
    angle_to_x_axis = np.arctan2(first_normal[1], first_normal[0])


    # Rotate the coordinate system such that the first normal points along the x-axis
    nepoints_rotated = rotate_points(nepoints_translated.T, -angle_to_x_axis).T
    rna_rotated = rotate_points(rna_translated.T, -angle_to_x_axis).T

    first_normal_rotated = rotate_points(first_normal.reshape(1, 2), -angle_to_x_axis)

    if (normal_direction == 'negative' and first_normal_rotated[0,0]>0)   or (normal_direction == 'positive'and first_normal_rotated[0,0]<0):
        # If the normal doesn't point as specified, rotate everything 180 degrees
        angle_correction = np.pi
        nepoints_rotated = rotate_points(nepoints_rotated.T, -angle_correction).T
        rna_rotated = rotate_points(rna_rotated.T, -angle_correction).T
        # No need to rotate COM and normal again, as the 180-degree rotation will just reverse their direction
    else:
        print('error!!!!!!!!!!!!!')
    return nepoints_rotated, rna_rotated