"""
Script to compute and plot histograms of scale and angle deviations from image registration.

This script processes image data folders, extracts scale factors and angles from registration,
collects the data, and plots histograms with Gaussian fits for the scale factors and angles.

Requirements:
- numpy
- torch.cuda
- matplotlib
- Yeast_processor from utils
- scipy
- scienceplots
- tqdm
- math
- napari (for visualization)
"""

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
    'drift_bins': 4,
    'resultdir': "/results/",
    'gain': '/home/pieter/Data/Yeast/bright_images_20ms.tiff',      # Gain calibration image
    # 'gain': '/home/pieter/Data/astigmatist_cal_100mlam_1_MMStack_Default.ome.tif',  # Alternative gain calibration image
    'offset': '/home/pieter/Data/Yeast/dark_images_20ms.tiff',      # Offset calibration image
    'model_NE': 'trained_networks/FPNresnet34_yeast.pt',            # Model for NE
    'model_bg': 'trained_networks/noleakyv3.pth',                   # Background model
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
    Processes a single folder to compute scale and angle from image registration.

    Parameters:
    - folder_path: Path to the folder to process.

    Returns:
    - scale: Scale factor obtained from registration.
    - angle: Angle obtained from registration.
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
        # Compute registration with regmode=1
        scale, angle, mean_img = Yp.compute_registration(regmode=1, plotfig=True, print_update=False)
        translation1 = Yp.registration['tvec']
        # Compute registration with regmode=2, using scale and angle from previous step
        Yp.compute_registration(regmode=2, plotfig=False, print_update=True, scale=scale, angle=angle)
        translation2 = Yp.registration['tvec']
        diff = translation2 - translation1

        return scale, angle

if __name__ == "__main__":
    # Lists to accumulate scale factors and angles
    scale_list = []
    alfa_list = []
    base_dir = "/media/pieter/Extreme SSD/Yeast_tracking_data2023/BMY822"
    root_dirs = os.listdir(base_dir)

    full_paths = [os.path.join(base_dir, item) for item in root_dirs]

    # Alternative root directories (commented out)
    # root_dirs = ["/media/pieter/Extreme SSD/Yeast_tracking_data2023/BMY823/BMY823_7_16_23_aqsettings1_batchA",
    #             "/home/pieter/Data/Yeast/tracking data/BMY823_7_16_23_aqsettings1_batchB",
    #              "/home/pieter/Data/Yeast/tracking data/BMY823_7_20_23_aqsettings1_batchB"]
    meanimg_list = []

    # Loop over each root directory
    for root_dir in full_paths:
        # if len(alfa_list) > 10:
        #     break  # Limit the number of data points
        # Loop over each folder in the root directory
        for folder_name in tqdm.tqdm(os.listdir(root_dir)):
            # if len(dif_list)>100:
            #     break
            folder_path = os.path.join(root_dir, folder_name)
            # if folder_name == 'cell 20':
            if os.path.isdir(folder_path):
                try:
                    print(folder_path)
                    # Process the folder and get scale and angle
                    scale, angle = process_folder(folder_path)
                    scale_list = np.append(scale_list, scale)
                    alfa_list = np.append(alfa_list, angle)
                    print(len(alfa_list))
                except:
                    print('error, probably file not found')

    import numpy as np  # Redundant import (already imported at the top)

    # Gaussian function for curve fitting
    def gaussian(x, amplitude, mean, stddev):
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
        plt.figure(dpi=400, figsize=[2, 3])
        plt.hist(data, range=[min(data), max(data)], bins=num_bins)
        plt.xlabel(xlabel)
        # plt.xlim(-1 * 70, 1 * 70)
        # plt.ylim(0, ylim)
        amplitude, mean, stddev = fit_gaussian(data, num_bins)
        # plt.yticks([0,100,200])
        x_fit = np.linspace(-2 * 125, 2 * 125, 1000)
        y_fit = gaussian(x_fit, amplitude, mean, stddev)
        plt.plot(x_fit, y_fit, label=r'$\sigma={:.2f}$nm,\n$\mu={:.2f}$ nm'.format(stddev, mean))
        plt.ylabel('Counts')
        plt.legend(loc='upper right')  # Adjust the 'loc' parameter as needed
        plt.tight_layout()

    # Convert lists to numpy arrays
    scale_arr = np.array(scale_list, dtype=float)
    angle_arr = np.array(alfa_list, dtype=float)

    # Gaussian function for curve fitting
    def gaussian(x, mu, sigma, amplitude):
        return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Function to plot histogram with Gaussian fit
    def plot_hist_with_gaussian(data, bins, range, xlabel, ylabel, filename):
        """
        Plots a histogram of the data with a Gaussian fit and saves it.

        Parameters:
        - data: The data to plot.
        - bins: Number of bins in the histogram.
        - range: The range for the histogram bins.
        - xlabel: Label for the x-axis.
        - ylabel: Label for the y-axis.
        - filename: Filename to save the figure.
        """
        plt.figure(figsize=(2, 2))

        # Histogram
        counts, bin_edges = np.histogram(data, bins=bins, density=True, range=range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Fit Gaussian
        popt, _ = curve_fit(gaussian, bin_centers, counts, p0=[np.mean(data), np.std(data), max(counts)])
        mu, sigma, amplitude = popt

        # Plot histogram
        plt.hist(data, bins=bins, density=True, range=range, alpha=0.6, color='g')

        # Plot Gaussian fit
        x = np.linspace(range[0], range[1], 100)
        plt.plot(x, gaussian(x, *popt), 'k', linewidth=2)
        title = r'$\mu = %.3f$' % mu + '\n' + r'$\sigma = %.3f$' % sigma

        # Add legend
        plt.legend([title], loc='upper right')

        # Labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Tight layout and save as SVG
        plt.tight_layout(pad=0.1)
        plt.savefig(filename, format='svg')
        plt.show()

    # Plotting and saving the figures
    plot_hist_with_gaussian(
        scale_arr,
        bins=30,
        range=[0.98, 1.02],
        xlabel=r'Scale factor $S$',
        ylabel='Probability',
        filename='graphs/scale_histogram.svg'
    )
    plot_hist_with_gaussian(
        angle_arr,
        bins=30,
        range=[-1, 1],
        xlabel=r'Angle $\alpha$ [$^\circ$]',
        ylabel='Probability',
        filename='graphs/angle_histogram.svg'
    )
