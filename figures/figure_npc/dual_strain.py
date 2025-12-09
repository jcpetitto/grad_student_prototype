"""
Title: Yeast NPC Localization and Distance Analysis Script

Description:
This script analyzes the spatial relationship between different nucleoporins (components of the Nuclear Pore Complex, NPC) in yeast cells. It processes microscopy images to detect NPCs labeled with different fluorescent markers, aligns the images, and calculates the distances between NPC components across different strains. The script performs statistical analysis and visualizations to compare the distributions of distances among strains.

Functions:
- `extract_digits(folder_name)`: Extracts numerical digits from a folder name.
- `compute_tangent_slopes(data)`: Computes tangent slopes of a 2D array using central difference approximation.
- `closest_point_index(x, y, line_points)`: Finds the index of the closest point on a line to a given point.
- `gaussian(x, amplitude, mean, stddev)`: Defines a Gaussian function for curve fitting.
- `fit_gaussian(data, bins)`: Fits a Gaussian to histogram data.
- `plot_histogram_with_gaussian(data, xlabel, binwidth, x_lim, color, label)`: Plots a histogram with a fitted Gaussian curve.
- `remove_outliers(df, column_name, group_key)`: Removes outliers from a DataFrame based on the interquartile range (IQR).

Main Workflow:
1. **Initialization**: Sets up configurations and initializes variables.
2. **Data Processing**:
   - Iterates through specified yeast strains and their respective data directories.
   - For each cell folder, performs image registration to align images from different channels.
   - Detects NPCs using the `Yeast_processor` class.
   - Refines NPC fits and computes the normal vectors at each point on the NPC contour.
   - Calculates distances between NPC components along the normals.
3. **Data Aggregation**: Collects distance measurements across all cells and strains.
4. **Statistical Analysis**:
   - Fits Gaussian distributions to the distance data for each strain.
   - Plots histograms with fitted Gaussian curves.
   - Creates violin and box plots to visualize the distribution of distances.
   - Performs ANOVA and t-tests to assess statistical significance between strains.

Dependencies:
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- scikit-image (skimage)
- tqdm
- torch
- re
- os
- scienceplots
- colorcet (for color maps)
- Custom modules:
  - `Yeast_processor` from `utils.Yeast_processor`
  - `intersection` module for finding line intersections

Usage:
- Ensure that all dependencies, including custom modules, are installed and accessible in the Python environment.
- Adjust the `cfg` dictionary with the correct paths and file names for the data.
- Update the `strains` list with the strains to analyze.
- Run the script to process the data, perform analysis, and generate plots.

Notes:
- The script uses a custom `Yeast_processor` class, which should be defined in `utils.Yeast_processor`.
- The `intersection` function is assumed to be available from the `intersect` module.
- The script requires access to microscopy image data and may need adjustment based on the specific data format.
- The plotting style is set to 'science' using the `scienceplots` package.
- Statistical tests include ANOVA and t-tests to compare distance distributions between strains.
"""



import skimage
from sympy.physics.quantum.identitysearch import scipy

from utils.Yeast_processor import Yeast_processor
import os
import tqdm
import re
import torch
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.optimize import curve_fit
import numpy as np
from intersect import intersection
import scienceplots
plt.style.use('science')
def show_napari(img):
    import napari
    viewer = napari.imshow(img)
def show_napari_points(img, points):
    import napari
    viewer = napari.Viewer()
    viewer.add_image(img)
    viewer.add_points(points, symbol='cross', size=3, face_color='red', edge_width=0 )

def closest_point_index(x,y, line_points):
    """
    Find the index of the closest point on a line to a given point.

    Parameters:
    - point: Tuple or array containing the coordinates of the given point (e.g., (x, y))
    - line_points: 2D numpy array representing the line points (each column is a point, e.g., np.array([[x1, x2], [y1, y2]]))

    Returns:
    - index: Index of the closest point on the line to the given point
    """
    point = np.array([x,y])
    distances = np.linalg.norm(line_points - point, axis=0)
    index = np.argmin(distances)
    return index
folder = '/home/pieter/Data/Yeast/july_7_exposure_20ms_emgain300'

cfg = {
    'path': '/home/pieter/Data/Yeast/july_7_exposure_20ms_emgain300/',
    'fn_reg_npc': 'BFred21.tiff',
    'fn_reg_rnp': 'BFgreen19.tiff',
    'fn_track_rnp': 'NEgreen19.tiff',
    'fn_track_npc': 'NEred19.tiff',
    'roisize':8, # even number please
    'sigma': 1.3,
    'frames': [0, 2000],
    'frames_npcfit': [0,250],
    'drift_bins': 10,
     'resultdir': "/results/",
    'gain': '/home/pieter/Data/Yeast/bright_images_20ms.tiff',
    'offset': '/home/pieter/Data/Yeast/dark_images_20ms.tiff',
    'model_NE': '../trained_networks/Modelweights_NE_segmentation.pt',
    'model_bg': '../trained_networks/model_wieghts_background_psf.pth',
    'pixelsize': 128
}
def extract_digits(folder_name):
    # Use regular expression to extract digits from the folder name
    digits_match = re.search(r'\d+', folder_name)
    if digits_match:
        return digits_match.group()
    return None

def compute_tangent_slopes(data):
    """
    Compute the tangent slopes of a 2D array using central difference approximation.

    Parameters:
    - data: 2D numpy array of shape (2, N) where N is the number of points

    Returns:
    - tangent_slopes: 2D numpy array of tangent slopes, one for x and one for y
    """
    h = 1  # Adjust this value based on your specific data spacing

    # Central difference for x-axis
    dx = (data[0, 2:] - data[0, :-2]) / (2 * h)
    dx = np.concatenate(([dx[0]], dx, [dx[-1]]))  # Padding the edges

    # Central difference for y-axis
    dy = (data[1, 2:] - data[1, :-2]) / (2 * h)
    dy = np.concatenate(([dy[0]], dy, [dy[-1]]))  # Padding the edges

    tangent_slopes = np.vstack((dx, dy))

    return tangent_slopes
diff_all_strains = []
diff_all_cells = []
strains = ['BMY_1408','BMY_1409','BMY_1410', 'BMY_1914']
#strains = ['BMY_1914']
num_cells  = np.zeros(np.size(strains))
if __name__ == "__main__":
    for num_strain, strain in enumerate(strains):
        dif_list = []
        base_dir = "/media/pieter/Extreme SSD/dual_strain/" + strain
        root_dirs = os.listdir(base_dir)
        threshold_regist = 0.5
        full_paths = [os.path.join(base_dir, item) for item in root_dirs]

        # root_dirs = ["/media/pieter/Extreme SSD/Yeast_tracking_data2023/BMY823/BMY823_7_16_23_aqsettings1_batchA",
        #             "/home/pieter/Data/Yeast/tracking data/BMY823_7_16_23_aqsettings1_batchB",
        #              "/home/pieter/Data/Yeast/tracking data/BMY823_7_20_23_aqsettings1_batchB"]
        count = 0
        distance_list = []
        for root_dir in full_paths:

            for folder_name in tqdm.tqdm(os.listdir(root_dir)):

                folder_path = os.path.join(root_dir, folder_name)
                #if folder_name == 'cell 20':
                if os.path.isdir(folder_path):
                    # if count >10:
                    #     break
                    try:
                        digits = extract_digits(os.path.basename(folder_path))
                        number_off_cells = 0
                        number_off_detections = 0
                        number_off_detections_tot = []

                        if digits:
                            print(folder_path)
                            torch.cuda.empty_cache()
                            # Modify the cfg dictionary for the current folder
                            cfg['fn_reg_npc1'] = '/BF1red' + digits + '.tiff'
                            cfg['fn_reg_rnp1'] = '/BF1green' + digits + '.tiff'
                            cfg['fn_reg_npc2'] = '/BF2red' + digits + '.tiff'
                            cfg['fn_reg_rnp2'] = '/BF2green' + digits + '.tiff'
                            cfg['fn_track_rnp'] = '/RNAgreen' + digits + '.tiff'
                            cfg['fn_track_npc'] = '/NEred' + digits + '.tiff'
                            # Update the path in the configuration with the current folder path
                            cfg['path'] = folder_path
                            # Perform the processing steps for the current folder
                            Yp = Yeast_processor(cfg)

                            Yp.compute_registration(regmode=1)
                            translation1 = Yp.registration['tvec']
                            Yp.compute_registration(regmode=2)
                            translation2 = Yp.registration['tvec']
                            np.sqrt(abs(translation1[0] - translation2[0]) ** 2 + abs(translation1[0] - translation2[0]) ** 2)

                            if np.sqrt(abs(translation1[0] - translation2[0]) ** 2 + abs(
                                    translation1[0] - translation2[0]) ** 2) < threshold_regist:
                            #Yp.compute_drift(save_fig=True)
                                logits = Yp.detect_npc(save_fig=False,count_good_label=40,gap_closing_distance=20,
                                                       threshold=0.1, oldmethod=False,usegreen=False)
                                points_allcells1, errors1, values1 = Yp.refinement_npcfit_movie_new(movie=False, registration = True,
                                                                                                       smoothness = 5,
                                                                                  Lambda=1e-2,length_line=12,estimate_prec=False,
                                                                                  number_mean=250, dual_strain=False, save_fig=False,iterations=300, max_signs=np.inf)

                                points_allcells2, errors2, values2, deriv_allcells2 = Yp.refinement_npcfit_movie_new(movie=False, registration=False,
                                                                                                    smoothness=5,
                                                                                                    Lambda=1e-2, length_line=12,
                                                                                                    estimate_prec=False,
                                                                                                            number_mean=250,
                                                                                            dual_strain=True, save_fig=False,iterations=300, max_signs=np.inf)
                                num_cells[num_strain] += len(points_allcells2)
                                print(num_cells)
                                for i in range(min(len(points_allcells2),len(points_allcells1))):
                                    tangent_slopes2 = deriv_allcells2[i][0]
                                    tangent_slopes = compute_tangent_slopes(points_allcells2[i][0])
                                    normal_slopes = np.array([-tangent_slopes[1, :], tangent_slopes[0, :]])
                                    # normal_slopes is the normal for each point x1,y1
                                    x1 = points_allcells2[i][0][0, :]
                                    y1 = points_allcells2[i][0][1, :]
                                    x2 = points_allcells1[i][0][0,:]
                                    y2 = points_allcells1[i][0][1, :]

                                    # Initialize arrays to store intersection points and distances
                                    intersection_points = []
                                    distances_along_normal = []
                                    lenght_line=3

                                    ### only for inout measurement
                                    tck, u = scipy.interpolate.splprep(
                                        [x1, y1], s=0, per=True,
                                        quiet=3)

                                    # evaluate the spline fits for 1000 evenly spaced distance values
                                    x_spline, y_spline = scipy.interpolate.splev(np.linspace(0, 1, 1000),
                                                                     tck)
                                    # total distance of green
                                    distances_betweenpoints = np.sqrt(
                                        np.sum(np.diff(np.array([x1,y1]).T, axis=0) ** 2, axis=1))
                                    cumulative_distances = np.cumsum(np.insert(distances_betweenpoints, 0, 0))[-1]

                                    # Loop over each point in x1, y1
                                    for j in range(len(x1)):
                                        normal_slope = normal_slopes[:, j] / np.linalg.norm(normal_slopes[:, j])
                                        point = np.array([x1[j],y1[j]])
                                        start = point - lenght_line * normal_slope
                                        end = point + lenght_line * normal_slope
                                        # -- Extract the line...
                                        # Make a line with "num" points...
                                        x0line, y0line = start[0], start[1]
                                        x1line, y1line = end[0], end[1]
                                        x, y = np.linspace(x0line, x1line, 100), np.linspace(y0line, y1line, 100)


                                        intersect = intersection(x,y,x2,y2)
                                        distance = False
                                        if len(intersect[0])==1 and cumulative_distances>5:
                                            distance = np.linalg.norm(np.array(intersect).T-point)


                                            ####
                                            inout = skimage.measure.points_in_poly(np.array(intersect).T, np.array([x_spline,y_spline]).T).astype(int)
                                            if inout == 0:
                                                distance = distance*-1
                                            distance_list.append(distance)

                                        if False:#j%1e9 == 0:

                                            plt.figure(dpi=400)
                                            if distance:
                                                plt.title('distance = ' + str(distance) + '\n total lengt green: '+str(cumulative_distances) )

                                            plt.plot(x,y, color='k', label='normal')
                                            plt.ylabel('x [pixel]')
                                            plt.xlabel('y [pixel]')
                                            plt.plot(x2,y2,label='red channel', color='red')

                                            plt.plot(x_spline,y_spline,':',color='green')
                                            plt.plot(points_allcells2[i][0][0, :], points_allcells2[i][0][1, :],label='green channel', color='green')
                                            plt.legend()
                                            plt.tight_layout()
                                            plt.show()



                            # for listlen in range(len(params1)):
                            #     params1_temp = params1[listlen]
                            #     params1_temp = params1_temp[0]
                            #     mune1=params1_temp[:,7]
                            #     params2_temp = params2[listlen]
                            #     params2_temp = params2_temp[0]
                            #     mune2 = params2_temp[:, 7]
                            #     diff = mune1-mune2
                            #     diff_all_cells.append(diff)
                            # import numpy as np

                                count +=1
                                print(count)
                    except:
                        print('no file')

        diff_all_strains.append(distance_list)

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2 / (2 * stddev ** 2)))

def fit_gaussian(data, bins):
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=[max(hist), 0.0, 10])
    amplitude, mean, stddev = popt
    perr = np.sqrt(np.diag(pcov))  # Calculate the standard deviation errors
    return amplitude, mean, stddev, perr

def plot_histogram_with_gaussian(data, xlabel, binwidth=50, x_lim=(-200, 250), color='blue', label=''):
    # Calculate the number of bins based on the desired binwidth
    data_range = max(data) - min(data)
    num_bins = int(data_range / binwidth)
    counts, bin_edges = np.histogram(data, bins=num_bins, range=[min(data), max(data)], density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.hist(data, bins=num_bins, range=[min(data), max(data)], color=color, alpha=0.5, label=label, density=True)
    plt.xlabel(xlabel)

    amplitude, mean, stddev, perr = fit_gaussian(data, num_bins)

    x_fit = np.linspace(min(data), max(data), 5000)
    y_fit = gaussian(x_fit, amplitude, mean, stddev)
    plt.plot(x_fit, y_fit, color=color)
    print(label, f' Gaussian fit, $\sigma={stddev:.2f} \pm {perr[2]:.2f}$ nm, $\mu={mean:.2f} \pm {perr[1]:.2f}$ nm')
    plt.ylabel('Probability')
    plt.xlim(x_lim[0], x_lim[1])
    plt.legend(loc='upper right')

print('number of cells are', num_cells)
# Assuming you have three different strains in diff_all_strains
strains = ['Nup60', 'Nup82', 'Nup170', 'Dbp5']

# Create a figure
fig = plt.figure(figsize=(2, 2))

# Use colorblind-friendly colors from colorcet
colors = cc.glasbey

# Loop through each strain and plot the histogram with the fitted curve
for i, data in enumerate(diff_all_strains):
    if i < 3:
        plot_histogram_with_gaussian(np.array(data), 'Distance to Dbp5 [nm]', binwidth=30, color=colors[i], label=strains[i])

plt.tight_layout(pad=0.2)
plt.savefig('graphs/threestrains.svg', format='svg')
plt.show()

import seaborn as sns
import pandas as pd
# Prepare the data
data = []
for strain_index, strain_data in enumerate(diff_all_strains):
    for value in strain_data:

        data.append({'Strain': strains[strain_index], 'Distance to Dbp5 [nm]': value})
df = pd.DataFrame(data)


def remove_outliers(df, column_name, group_key):
    # Calculate the IQR and determine outliers for each group
    def is_outlier(group):
        Q1 = group.quantile(0.25)
        Q3 = group.quantile(0.75)
        IQR = Q3 - Q1
        return ~group.between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # Apply the outlier detection to each group and get a boolean Series
    outlier_mask = df.groupby(group_key)[column_name].transform(is_outlier)

    # Filter the DataFrame based on the inverse of the outlier mask
    return df[~outlier_mask]

# Applying the corrected function
df_no_outliers = remove_outliers(df, 'Distance to Dbp5 [nm]', 'Strain')
plt.figure(figsize=(2, 2))

sns.violinplot(x='Strain', y='Distance to Dbp5 [nm]', data=df_no_outliers, palette='colorblind',
               order=['Nup82', 'Nup170', 'Nup60'], bw=0.4, inner=None)

# Calculate the medians for each strain
medians = df_no_outliers.groupby('Strain')['Distance to Dbp5 [nm]'].median().reindex(['Nup82', 'Nup170', 'Nup60'])

# Overlay black bars for the medians
for i, strain in enumerate(['Nup82', 'Nup170', 'Nup60']):
    median_val = medians[strain]
    # Draw the median line; adjust the xmin and xmax to control the width of the median bar
    plt.hlines(median_val, i - 0.25, i + 0.25, color='black', lw=2)
plt.tight_layout(pad=0.1)
plt.xlabel("Second NPC label")

plt.savefig('graphs/threestrains_violinplot_no_outliers.svg', format='svg')
plt.show()

from scipy.stats import f_oneway

# Assuming df_no_outliers is your DataFrame and it's already filtered to exclude outliers

# Filter data for each strain
data_nup82 = df_no_outliers[df_no_outliers['Strain'] == 'Nup82']['Distance to Dbp5 [nm]']
data_nup170 = df_no_outliers[df_no_outliers['Strain'] == 'Nup170']['Distance to Dbp5 [nm]']
data_nup60 = df_no_outliers[df_no_outliers['Strain'] == 'Nup60']['Distance to Dbp5 [nm]']

# Perform one-way ANOVA
f_stat, p_value = f_oneway(data_nup82, data_nup170, data_nup60)

print(f"ANOVA test results: F-statistic = {f_stat}, p-value = {p_value}")


from scipy.stats import ttest_ind
# Pairwise comparisons
pairs = [('Nup82', 'Nup170'), ('Nup82', 'Nup60'), ('Nup170', 'Nup60')]
p_values = {}
for strain1, strain2 in pairs:
    group1 = df[df['Strain'] == strain1]['Distance to Dbp5 [nm]']
    group2 = df[df['Strain'] == strain2]['Distance to Dbp5 [nm]']
    _, p_value = ttest_ind(group1, group2)
    p_values[(strain1, strain2)] = p_value

# Function to determine annotation text based on p-value
def p_value_annotation_text(p):
    if p < 0.00001:
        return "****"
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"  # not significant

# Create the box plot
plt.figure(figsize=(2, 2))
ax = sns.boxplot(x='Strain', y='Distance to Dbp5 [nm]', data=df, palette='colorblind', order=['Nup82', 'Nup170', 'Nup60'], showfliers=False)

# Hide the x-axis labels
#ax.set_xticklabels([])
# Optionally, hide the x-axis label (if any)
ax.set_xlabel("NPC label")

plt.tight_layout(pad=0.1)
plt.savefig('graphs/threestrains_boxplot.svg', format='svg')
plt.show()