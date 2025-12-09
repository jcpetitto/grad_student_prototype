## Script to make figures for drift precision

import os
import re
import numpy as np
import torch.cuda
import matplotlib.pyplot as plt
from utils.Yeast_processor import Yeast_processor
from scipy.stats import norm
from scipy.optimize import curve_fit
import scienceplots
plt.style.use('science')
import tqdm
from scipy.stats import poisson
import math

# Functions for visualization using Napari
def show_napari(img):
    """
    Display an image using Napari.
    """
    import napari
    viewer = napari.imshow(img)

def show_tensor(img):
    """
    Display a tensor image using Napari.
    """
    import napari
    viewer = napari.imshow(img.detach().cpu().numpy())

def show_napari_points(img, points):
    """
    Display an image with overlaid points using Napari.
    """
    import napari
    viewer = napari.Viewer()
    viewer.add_image(img)
    viewer.add_points(points, symbol='cross', size=3, face_color='red', edge_width=0)

# Configuration dictionary for processing
cfg = {
    'path': '/home/pieter/Data/Yeast/july_7_exposure_20ms_emgain300/cell 21/',  # Data path
    'fn_reg_npc1': 'BF1red{}.tiff',    # Filename pattern for first NPC registration
    'fn_reg_rnp1': 'BF1green{}.tiff',  # Filename pattern for first RNP registration
    'fn_reg_npc2': 'BF2red{}.tiff',    # Filename pattern for second NPC registration
    'fn_reg_rnp2': 'BF2green{}.tiff',  # Filename pattern for second RNP registration
    'fn_track_rnp': 'RNAgreen{}.tiff', # Filename pattern for RNP tracking
    'fn_track_npc': 'NEred{}.tiff',    # Filename pattern for NPC tracking
    'roisize':16,                      # ROI size (even number please)
    'sigma': 1.3,
    'frames': [0, 1000],
    'frames_npcfit': [0,250],
    'drift_bins': 2,
    'resultdir': "/results/",
    'gain': '/home/pieter/Data/Yeast/bright_images_20ms.tiff',      # Gain calibration image
    # 'gain': '/home/pieter/Data/astigmatist_cal_100mlam_1_MMStack_Default.ome.tif',  # Alternative gain calibration image
    'offset': '/home/pieter/Data/Yeast/dark_images_20ms.tiff',      # Offset calibration image
    'model_NE': '',            # Model for NE
    'model_bg': '',                   # Background model
    'pixelsize': 128                                                # Pixel size in nm
}

def extract_digits(folder_name):
    """
    Extracts digits from a folder name using regular expressions.
    """
    # Use regular expression to extract digits from the folder name
    digits_match = re.search(r'\d+', folder_name)
    if digits_match:
        return digits_match.group()
    return None

def process_folder(folder_path):
    """
    Processes a single folder to compute drift differences.

    Parameters:
    - folder_path: Path to the folder to process.

    Returns:
    - diff_x: Array of x-coordinate drift differences (excluding the first index).
    - diff_y: Array of y-coordinate drift differences (excluding the first index).
    """
    global cfg
    Yp = Yeast_processor(cfg)
    # Extract the digits from the folder name
    digits = extract_digits(os.path.basename(folder_path))
    if digits:
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
        # Yp.calibrate(savefig=True)  # Calibration step (commented out)
        trans1, trans2 = Yp.compute_drift(save_fig=False, estimate_precision=True)
        diff = trans1 - trans2
        diff_x = diff[:,0]
        diff_y = diff[:, 1]
        return diff_x[1::], diff_y[1::]  # Exclude the first index (always zero)

if __name__ == "__main__":
    # Lists to accumulate drift differences
    difx_list = []
    dify_list = []
    base_dir = "/media/pieter/Extreme SSD/Yeast_tracking_data2023/BMY823" # change to your folder
    root_dirs = os.listdir(base_dir)
    full_paths = [os.path.join(base_dir, item) for item in root_dirs]

    # Loop over each root directory
    for root_dir in full_paths:
        # Loop over each folder in the root directory
        for folder_name in tqdm.tqdm(os.listdir(root_dir)):
            # if len(difx_list) > 20:
            #     break  # Limit the number of data points
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                try:
                    print(folder_path)
                    # Process the folder and get drift differences
                    diffx, diffy = process_folder(folder_path)
                    difx_list = np.append(difx_list, diffx)
                    dify_list = np.append(dify_list, diffy)
                    print(len(difx_list))
                except:
                    print('error, probably file not found')

    import numpy as np  # Redundant import (already imported at the top)

    def gaussian(x, amplitude, mean, stddev):
        """
        Gaussian function for curve fitting.
        """
        return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)

    def fit_gaussian(data, bins):
        """
        Fits a Gaussian curve to histogram data.

        Parameters:
        - data: The data to fit.
        - bins: Number of bins in the histogram.

        Returns:
        - amplitude: Fitted amplitude of the Gaussian.
        - mean: Fitted mean of the Gaussian.
        - stddev: Fitted standard deviation of the Gaussian.
        """
        hist, bin_edges = np.histogram(data, bins=bins)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[250, 0.0, 10])
        amplitude, mean, stddev = popt
        return amplitude, mean, stddev

    import matplotlib.pyplot as plt  # Redundant import (already imported at the top)
    plt.close('all')

    def plot_histogram_with_gaussian(data, xlabel, binwidth=10, ylim=250):
        """
        Plots a histogram of the data with a fitted Gaussian curve.

        Parameters:
        - data: The data to plot.
        - xlabel: Label for the x-axis.
        - binwidth: Width of the histogram bins.
        - ylim: Limit for the y-axis.
        """
        # Calculate the number of bins based on the desired binwidth
        data_range = max(data) - min(data)
        num_bins = int(data_range / binwidth)
        counts, _ = np.histogram(data, range=[min(data), max(data)], bins=num_bins)
        plt.figure(figsize=[2,3])
        plt.hist(data, range=[min(data), max(data)], bins=num_bins)
        plt.xlabel(xlabel)
        plt.xlim(-30, 30)
        plt.ylim(0, ylim)
        plt.yticks([0,100,200])
        amplitude, mean, stddev = fit_gaussian(data, num_bins)

        x_fit = np.linspace(-50, 50, 1000)
        y_fit = gaussian(x_fit, amplitude, mean, stddev)
        plt.plot(x_fit, y_fit, label='s=' + str(round(stddev, 2)) + 'nm,\n mu=' + str(round(mean, 2)) + 'nm')

        plt.ylabel('Counts')
        plt.legend()  # Adjust the 'loc' parameter as needed

        plt.tight_layout()

    # Convert drift differences to numpy arrays
    translation_x = np.array(difx_list, dtype=float)
    translation_y = np.array(dify_list, dtype=float)

    # Calculate the overall drift magnitude and filter out values >= 5
    sigma_reg = np.sqrt(translation_x**2 + translation_y**2)
    sigma_reg = sigma_reg[sigma_reg < 5]

    # Plot histogram and Gaussian fit for translation x
    plot_histogram_with_gaussian(translation_x * 128, r'$D_{1,x} - D_{2,x}$ [nm]', binwidth=2, ylim=200)
    plt.savefig('graphs/driftx.svg', format='svg')
    plt.show()

    # Plot histogram and Gaussian fit for translation y
    plot_histogram_with_gaussian(translation_y * 128, r'$D_{1,y} - D_{2,y}$ [nm]', binwidth=2, ylim=200)
    plt.savefig('graphs/drifty.svg', format='svg')
    plt.show()
    plt.show()
