"""
Script to Compute and Plot Histograms of Translation Differences from Image Registrations.

This script processes image data folders to compute translation differences between two registration
modes. It collects the translation differences and mean images, saves a stack of mean images, and
plots histograms with Gaussian fits for the translation differences in the x and y directions, as
well as the overall translation magnitude.

Key Functionalities:
- Extracts numerical identifiers from folder names to process corresponding image files.
- Computes registration between images using specified modes.
- Accumulates translation differences and mean images from multiple folders.
- Saves a stack of mean images as a TIFF file.
- Plots histograms of translation differences with Gaussian fits.

Requirements:
- Python 3.x
- numpy
- torch.cuda
- matplotlib
- Yeast_processor from utils
- scipy
- scienceplots
- tqdm
- tifffile
- math
- napari (for visualization)

Usage:
Run the script directly. Ensure that the configuration paths and parameters are correctly set.
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
plt.style.use('science')  # Apply science plotting style
import tqdm
from scipy.stats import poisson
import math

# Visualization Functions Using Napari
def show_napari(img):
    """
    Display an image using Napari.

    Parameters:
    - img (numpy.ndarray): Image data to display.
    """
    import napari
    viewer = napari.imshow(img)

def show_tensor(img):
    """
    Display a tensor image using Napari.

    Parameters:
    - img (torch.Tensor): Tensor image to display.
    """
    import napari
    viewer = napari.imshow(img.detach().cpu().numpy())

def show_napari_points(img, points):
    """
    Display an image with overlaid points using Napari.

    Parameters:
    - img (numpy.ndarray): Image data to display.
    - points (numpy.ndarray): Points to overlay on the image.
    """
    import napari
    viewer = napari.Viewer()
    viewer.add_image(img)
    viewer.add_points(points, symbol='cross', size=3, face_color='red', edge_width=0)

# Configuration Dictionary for Processing
cfg = {
    'path': '/home/pieter/Data/Yeast/july_7_exposure_20ms_emgain300/cell 21/',  # Base data path
    'fn_reg_npc1': 'BF1red{}.tiff',    # Filename pattern for first NPC registration
    'fn_reg_rnp1': 'BF1green{}.tiff',  # Filename pattern for first RNP registration
    'fn_reg_npc2': 'BF2red{}.tiff',    # Filename pattern for second NPC registration
    'fn_reg_rnp2': 'BF2green{}.tiff',  # Filename pattern for second RNP registration
    'fn_track_rnp': 'RNAgreen{}.tiff', # Filename pattern for RNP tracking
    'fn_track_npc': 'NEred{}.tiff',    # Filename pattern for NPC tracking
    'roisize': 16,                      # ROI size (must be an even number)
    'sigma': 1.3,                       # Sigma value for processing
    'frames': [0, 1000],                # Frame range for processing
    'frames_npcfit': [0, 250],         # Frame range for NPC fitting
    'drift_bins': 4,                    # Number of bins for drift calculation
    'resultdir': "/results/",           # Directory to save results
    'gain': '/home/pieter/Data/Yeast/bright_images_20ms.tiff',      # Gain calibration image
    # 'gain': '/home/pieter/Data/astigmatist_cal_100mlam_1_MMStack_Default.ome.tif',  # Alternative gain calibration image
    'offset': '/home/pieter/Data/Yeast/dark_images_20ms.tiff',      # Offset calibration image
    'model_NE': 'trained_networks/FPNresnet34_yeast.pt',            # Model path for NE
    'model_bg': 'trained_networks/noleakyv3.pth',                   # Background model path
    'pixelsize': 128                                                # Pixel size in nm
}

def extract_digits(folder_name):
    """
    Extract digits from a folder name using regular expressions.

    Parameters:
    - folder_name (str): Name of the folder.

    Returns:
    - str or None: Extracted digits as a string if found, else None.
    """
    digits_match = re.search(r'\d+', folder_name)
    if digits_match:
        return digits_match.group()
    return None

def process_folder(folder_path):
    """
    Process a single folder to compute translation differences and collect mean images.

    Parameters:
    - folder_path (str): Path to the folder to process.

    Returns:
    - tuple: Translation difference (numpy.ndarray) and mean image (numpy.ndarray).
    """
    global cfg
    Yp = Yeast_processor(cfg)  # Initialize Yeast_processor with current configuration

    # Extract digits from the folder name to identify corresponding files
    digits = extract_digits(os.path.basename(folder_path))
    if digits:
        torch.cuda.empty_cache()  # Clear CUDA cache to free memory

        # Update configuration filenames with extracted digits
        cfg['fn_reg_npc1'] = f'/BF1red{digits}.tiff'
        cfg['fn_reg_rnp1'] = f'/BF1green{digits}.tiff'
        cfg['fn_reg_npc2'] = f'/BF2red{digits}.tiff'
        cfg['fn_reg_rnp2'] = f'/BF2green{digits}.tiff'
        cfg['fn_track_rnp'] = f'/RNAgreen{digits}.tiff'
        cfg['fn_track_npc'] = f'/NEred{digits}.tiff'

        # Update the path in the configuration to the current folder
        cfg['path'] = folder_path

        # Re-initialize Yeast_processor with updated configuration
        Yp = Yeast_processor(cfg)

        # Perform image registration in mode 1 and retrieve translation vector
        scale, angle, mean_img = Yp.compute_registration(regmode=1, plotfig=True, print_update=False)
        translation1 = Yp.registration['tvec']

        # Perform image registration in mode 2 using scale and angle from mode 1
        Yp.compute_registration(regmode=2, plotfig=False, print_update=True, scale=scale, angle=angle)
        translation2 = Yp.registration['tvec']

        # Compute the difference between the two translation vectors
        diff = translation2 - translation1

        return diff, mean_img

    # Return None if digits are not found
    return None, None

def gaussian(x, amplitude, mean, stddev):
    """
    Gaussian function for curve fitting.

    Parameters:
    - x (float or numpy.ndarray): Independent variable.
    - amplitude (float): Amplitude of the Gaussian.
    - mean (float): Mean of the Gaussian.
    - stddev (float): Standard deviation of the Gaussian.

    Returns:
    - float or numpy.ndarray: Computed Gaussian value(s).
    """
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)

def fit_gaussian(data, bins):
    """
    Fit a Gaussian curve to histogram data.

    Parameters:
    - data (numpy.ndarray): Data to fit.
    - bins (int): Number of bins in the histogram.

    Returns:
    - tuple: Fitted amplitude, mean, and standard deviation of the Gaussian.
    """
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[250, 0.0, 10])
    amplitude, mean, stddev = popt
    return amplitude, mean, stddev

def plot_histogram_with_gaussian(data, xlabel, binwidth=10, ylim=250):
    """
    Plot a histogram of the data with a fitted Gaussian curve.

    Parameters:
    - data (numpy.ndarray): Data to plot.
    - xlabel (str): Label for the x-axis.
    - binwidth (int, optional): Width of the histogram bins. Defaults to 10.
    - ylim (int, optional): Upper limit for the y-axis. Defaults to 250.
    """
    # Calculate the number of bins based on the desired binwidth
    data_range = max(data) - min(data)
    num_bins = int(data_range / binwidth)
    counts, _ = np.histogram(data, range=[min(data), max(data)], bins=num_bins)

    plt.figure(dpi=400, figsize=[2, 3])
    plt.hist(data, range=[min(data), max(data)], bins=num_bins, color='skyblue', edgecolor='black')
    plt.xlabel(xlabel)
    plt.xlim(-70, 70)  # Set x-axis limits
    # plt.ylim(0, ylim)  # Uncomment to set y-axis limits

    # Fit and plot Gaussian
    amplitude, mean, stddev = fit_gaussian(data, num_bins)
    x_fit = np.linspace(-250, 250, 1000)
    y_fit = gaussian(x_fit, amplitude, mean, stddev)
    plt.plot(x_fit, y_fit, label=f'σ={stddev:.2f} nm,\nμ={mean:.2f} nm', color='red')

    plt.ylabel('Counts')
    plt.legend(loc='upper right')  # Position the legend
    plt.tight_layout()
    plt.show()

def plot_hist_with_gaussian(data, bins, range_vals, xlabel, ylabel, filename):
    """
    Plot a histogram of the data with a Gaussian fit and save the figure.

    Parameters:
    - data (numpy.ndarray): Data to plot.
    - bins (int): Number of bins in the histogram.
    - range_vals (list or tuple): Range for the histogram bins.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - filename (str): Filename to save the figure.
    """
    plt.figure(figsize=(2, 2))

    # Histogram
    counts, bin_edges = np.histogram(data, bins=bins, density=True, range=range_vals)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit Gaussian
    popt, _ = curve_fit(gaussian, bin_centers, counts, p0=[np.mean(data), np.std(data), max(counts)])
    mu, sigma, amplitude = popt

    # Plot histogram
    plt.hist(data, bins=bins, density=True, range=range_vals, alpha=0.6, color='g', edgecolor='black')

    # Plot Gaussian fit
    x = np.linspace(range_vals[0], range_vals[1], 100)
    plt.plot(x, gaussian(x, *popt), 'k', linewidth=2)
    title = f'μ = {mu:.3f}\nσ = {sigma:.3f}'

    # Add legend
    plt.legend([title], loc='upper right')

    # Labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Tight layout and save as SVG
    plt.tight_layout(pad=0.1)
    plt.savefig(filename, format='svg')
    plt.show()

if __name__ == "__main__":
    # Initialize lists to accumulate translation differences and mean images
    dif_list = []
    meanimg_list = []

    # Base directory containing all root directories
    base_dir = "/media/pieter/Extreme SSD/Yeast_tracking_data2023/BMY822"
    root_dirs = os.listdir(base_dir)
    full_paths = [os.path.join(base_dir, item) for item in root_dirs]

    # Loop over each root directory
    for root_dir in full_paths:
        # Loop over each folder in the root directory with a progress bar
        for folder_name in tqdm.tqdm(os.listdir(root_dir), desc=f"Processing {root_dir}"):
            # if len(dif_list) > 50:
            #     break  # Limit the number of data points to prevent excessive processing

            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                try:
                    print(f"Processing folder: {folder_path}")
                    # Process the folder to get translation differences and mean image
                    diff, meanimg = process_folder(folder_path)

                    if diff is not None and meanimg is not None:
                        dif_list = np.append(dif_list, diff)  # Append translation differences
                        meanimg_list.append(meanimg)         # Append mean image
                        print(f"Accumulated {len(dif_list)} translation differences.")
                    else:
                        print(f"No digits found in folder name: {folder_name}")
                except Exception as e:
                    print(f"Error processing folder {folder_path}: {e}")

    # Concatenate all mean images into a single stack
    if meanimg_list:
        stack = np.concatenate(meanimg_list)
        # Save the stack as a TIFF file, starting from the 101st image to exclude initial frames
        import tifffile
        tifffile.imwrite('bf_stack_gfa.tiff', stack[100::, ...])
        print("Saved mean image stack as 'bf_stack_gfa.tiff'.")
    else:
        print("No mean images to concatenate and save.")

    # Replace None values in dif_list with 10 to handle missing data
    dif_list = np.where(dif_list == None, 10, dif_list)

    # Plotting Function for Histograms with Gaussian Fit
    def plot_histogram_with_gaussian(data, xlabel, binwidth=10, ylim=250):
        """
        Plots a histogram of the data with a fitted Gaussian curve.

        Parameters:
        - data (numpy.ndarray): Data to plot.
        - xlabel (str): Label for the x-axis.
        - binwidth (int, optional): Width of the histogram bins. Defaults to 10.
        - ylim (int, optional): Upper limit for the y-axis. Defaults to 250.
        """
        # Calculate the number of bins based on the desired binwidth
        data_range = max(data) - min(data)
        num_bins = int(data_range / binwidth)
        counts, _ = np.histogram(data, range=[min(data), max(data)], bins=num_bins)

        plt.figure(dpi=400, figsize=[2, 3])
        plt.hist(data, range=[min(data), max(data)], bins=num_bins, color='skyblue', edgecolor='black')
        plt.xlabel(xlabel)
        plt.xlim(-70, 70)  # Set x-axis limits
        # plt.ylim(0, ylim)  # Uncomment to set y-axis limits

        # Fit and plot Gaussian
        amplitude, mean, stddev = fit_gaussian(data, num_bins)
        x_fit = np.linspace(-250, 250, 1000)
        y_fit = gaussian(x_fit, amplitude, mean, stddev)
        plt.plot(x_fit, y_fit, label=f'sd={stddev:.2f} nm,\n mean={mean:.2f} nm', color='red')

        plt.ylabel('Counts')
        plt.legend(loc='upper right')  # Position the legend
        plt.tight_layout()
        plt.show()

    # Convert translation differences to numpy arrays for x and y directions
    translation_x = np.array(dif_list[::2], dtype=float)  # Even indices for x
    translation_y = np.array(dif_list[1::2], dtype=float)  # Odd indices for y

    # Calculate the overall translation magnitude (sigma_reg) and filter out values >= 5
    sigma_reg = np.sqrt(translation_x**2 + translation_y**2)
    sigma_reg = sigma_reg[sigma_reg < 5]

    # Plot histogram and Gaussian fit for translation in x-direction
    plot_histogram_with_gaussian(
        translation_x * 125,  # Scale translation to nanometers
        r'$t_{1,x} - t_{2,x}$ [nm]',
        binwidth=6,
        ylim=250
    )
    plt.savefig('graphs/registx.svg', format='svg')  # Save the figure
    plt.close('all')  # Close the figure to free memory

    # Plot histogram and Gaussian fit for translation in y-direction
    plot_histogram_with_gaussian(
        translation_y * 125,  # Scale translation to nanometers
        r'$t_{1,y} - t_{2,y}$ [nm]',
        binwidth=6,
        ylim=250
    )
    plt.savefig('graphs/registy.svg', format='svg')  # Save the figure
    plt.close('all')  # Close the figure to free memory

    # Plot histogram and Gaussian fit for overall translation magnitude
    plot_histogram_with_gaussian(
        sigma_reg * 125,  # Scale magnitude to nanometers
        r'$\| t_2 - t_1 \|$ [nm]',
        binwidth=6,
        ylim=250
    )
    plt.savefig('graphs/sigma_reg.svg', format='svg')  # Save the figure
    plt.close('all')  # Close the figure to free memory
