"""
Title: Yeast Cell Image Processing and Calibration Script

Description:

This script is designed to process yeast cell imaging data for the purpose of gain calibration and background correction. It performs the following key steps:

1. **Configuration Setup**:
   - Defines a configuration dictionary `cfg` containing file paths and processing parameters.
   - Includes paths to images for registration, tracking, gain, and offset correction.

2. **Utility Functions**:
   - `extract_digits(folder_name)`: Extracts digits from folder names to identify specific datasets.
   - `show_napari(img)`, `show_tensor(img)`, `show_napari_points(img, points)`: Visualization functions using Napari for image and point display.

3. **Folder Processing**:
   - `process_folder(folder_path, check_mask=False, use_old_mask=False)`: Main function that processes each folder containing imaging data.
     - Modifies the configuration based on the folder's content.
     - Performs background correction and bleaching normalization.
     - Computes the mean-variance relationship for gain calibration.
     - Saves a calibration plot showing the gain estimation.

4. **Main Execution**:
   - Iterates over specified root directories containing yeast cell data.
   - Processes each folder that meets the criteria (e.g., `cell 12`).
   - Outputs progress using `tqdm` for visualization.

Dependencies:

- `os`
- `re`
- `pickle`
- `scipy`
- `torch`
- `matplotlib`
- `numpy`
- `tqdm`
- `scienceplots`
- `tifffile`
- `napari`
- Custom module: `utils.Yeast_processor` (assumed to be defined elsewhere)

Usage:

- Update the `cfg` dictionary with the correct paths to your data files.
- Ensure that the required images for gain and offset correction are available.
- Run the script to process the folders containing yeast cell imaging data.
- The script will output calibration plots and save them as SVG files.

Notes:

- The script assumes a specific directory structure and naming convention for the data files.
- Ensure that the `Yeast_processor` class is properly implemented and available in the `utils` module.
- The script uses the `science` style for plotting to improve visualization.
- Paths in the `cfg` dictionary should be adjusted to match your system's file structure.

"""

import os
import re
import pickle

import scipy
import torch.cuda
import matplotlib.pyplot as plt
from utils.Yeast_processor import Yeast_processor
import tqdm
import scienceplots
import numpy as np
import os
plt.style.use('science')


def show_napari(img):
    import napari
    viewer = napari.imshow(img)

def show_tensor(img):
    import napari
    viewer = napari.imshow(img.detach().cpu().numpy())

def show_napari_points(img, points):
    import napari
    viewer = napari.Viewer()
    viewer.add_image(img)
    viewer.add_points(points, symbol='cross', size=3, face_color='red', edge_width=0)

cfg = {
    'path': '/home/pieter/Data/Yeast/july_7_exposure_20ms_emgain300/cell 21/',
    'fn_reg_npc1': 'BF1red{}.tiff',
    'fn_reg_rnp1': 'BF1green{}.tiff',
    'fn_reg_npc2': 'BF2red{}.tiff',
    'fn_reg_rnp2': 'BF2green{}.tiff',
    'fn_track_rnp': 'RNAgreen{}.tiff',
    'fn_track_npc': 'NEred{}.tiff',
    'roisize':16, # even number please
    'sigma': 0.92,
    'frames': [0, 1000],
    'frames_npcfit': [0,250],
    'drift_bins': 4,
     'resultdir': "/results/",
    'gain': '/media/pieter/Extreme SSD/Yeast_tracking_data2023/bright_images_green_channel_20ms_300EM.tiff',
    # 'gain': '/home/pieter/Data/astigmatist_cal_100mlam_1_MMStack_Default.ome.tif',
    'offset': '/media/pieter/Extreme SSD/Yeast_tracking_data2023/dark_images_green_channel_20ms_300EM.tiff',
    'model_NE': 'trained_networks/FPNresnet34_yeastvAnyee.pt',
    'model_bg': 'trained_networks/noleakyv4_v_vectorpsf.pth',
    'pixelsize':128
}
threshold_regist = 3 # threshold for registration difference before and after (in pixels)
def extract_digits(folder_name):
    # Use regular expression to extract digits from the folder name
    digits_match = re.search(r'\d+', folder_name)
    if digits_match:
        return digits_match.group()
    return None
import time
def process_folder(folder_path, check_mask=False, use_old_mask=False):
    def extract_digits(folder_name):
        # Use regular expression to extract digits from the folder name
        digits_match = re.search(r'\d+', folder_name)

        if digits_match:
            return digits_match.group()
        return None

    Yp = Yeast_processor(cfg)

    # Extract the digits from the folder name
    digits = extract_digits(os.path.basename(folder_path))

    if digits:
        torch.cuda.empty_cache()
        channel = 2
        # Modify the cfg dictionary for the current folder
        cfg['fn_reg_npc1'] = '/BF1red' + digits + '.tiff'
        cfg['fn_reg_rnp1'] = '/BF1green' + digits + '.tiff'
        cfg['fn_reg_npc2'] = '/BF2red' + digits + '.tiff'
        cfg['fn_reg_rnp2'] = '/BF2green' + digits + '.tiff'
        cfg['fn_track_rnp'] = '/RNAgreen' + digits + '.tiff'
        cfg['fn_track_npc'] = '/NEred' + digits + '.tiff'

        # Update the path in the configuration with the current folder path
        cfg['path'] = folder_path
        print('folder path = ', folder_path)
        # Perform the processing steps for the current folder
        import tifffile
        Yp = Yeast_processor(cfg)
        err = Yp.check_files()
        if channel ==1:
            Yp.fn_bright_image = '/media/pieter/Extreme SSD/Yeast_tracking_data2023/bright_images_green_channel_20ms_300EM.tiff'
            Yp.fn_dark_image = '/media/pieter/Extreme SSD/Yeast_tracking_data2023/dark_images_green_channel_20ms_300EM.tiff',

        if channel ==2:
            Yp.fn_bright_image = '/media/pieter/Extreme SSD/Yeast_tracking_data2023/bright_images_red_channelapril2024.tiff'
            Yp.fn_dark_image = '/media/pieter/Extreme SSD/Yeast_tracking_data2023/red_dark300.tif',

        dark_image = tifffile.imread(Yp.fn_dark_image)
        track_rnp = tifffile.imread(Yp.fn_bright_image)
        varbg = np.var(dark_image)
        dark_mean = np.mean(dark_image, 0)
        bg_corrected = track_rnp[:, :, :] - np.mean(dark_mean)
        print(np.mean(dark_mean))
        # Normalization of data (bleaching correction)
        firstplane_avg = np.mean(bg_corrected[0, :, :])
        mean_array = np.mean(bg_corrected, (1, 2))

        dim_array = np.ones((1, bg_corrected.ndim), int).ravel()
        dim_array[0] = -1
        b_reshaped = (firstplane_avg / mean_array).reshape(dim_array)

        bg_corrected = bg_corrected * b_reshaped

        variance = np.var(bg_corrected, 0)
        mean = np.mean(bg_corrected, 0)

        meanvarplot = scipy.stats.binned_statistic(mean.flatten(), variance.flatten(), bins=100, statistic='mean')
        weights, _ = np.histogram(mean.flatten(), bins=meanvarplot.bin_edges)
        weights = weights / np.sum(weights)

        center = (meanvarplot.bin_edges[1:] + meanvarplot.bin_edges[:-1]) / 2
        # remove nans and fit
        nanvalues = np.isnan(meanvarplot.statistic)
        fit = np.polyfit(center[~nanvalues], meanvarplot.statistic[~nanvalues], 1, w=weights[~nanvalues])

        fig, ax = plt.subplots(figsize=(3,3))

        ax.plot(center, np.polyval(fit, center), color='darkred',
                label='fit')
        ax.scatter(center, meanvarplot.statistic, marker='x', label='Mean variance in bin', s=10)
        ax.set_xlabel('Mean [ADU]')
        ax.set_ylabel('Variance [ADU]')
        ax2 = ax.twinx()
        ax2.plot(center, weights, label='Weights', color='tab:blue')
        ax2.set_ylabel(r"Weights (prop. to \#pixels)")
        ax2.set_yscale('log')
        fig.legend()
        plt.title('gain '+ str(np.round(1 / fit[0], 4)))
        plt.tight_layout()

        fig.savefig('calibration_'+str(channel) +'.svg', format='svg')
        plt.close('all')


           

    # Command-line argument parser

    #return pos, params_list, roi_pos_list, smpconcat, mu, Yp.bbox_NE, Yp
if __name__ == "__main__":
    count_errors = 0
    #"/home/pieter/Data/Yeast/tracking data/BMY823_7_16_23_aqsettings1_batchA",
    root_dirs = ["/media/pieter/Extreme SSD/Yeast_tracking_data2023/BMY823"]

    number_off_cells_all = 0
    number_off_detections_all = 0
    number_off_detections_tot_all = []
    for root_dir in root_dirs:
        countsss = 0
        for folder_name_upper in tqdm.tqdm(os.listdir(root_dir)):

            folder_path_upper = os.path.join(root_dir, folder_name_upper)

            for folder_name in tqdm.tqdm(os.listdir(folder_path_upper)):
                if folder_name == 'cell 12':
                    folder_path = os.path.join(folder_path_upper, folder_name)

                    if os.path.isdir(folder_path):
                        print(folder_path)
                        process_folder(folder_path, use_old_mask=False)


                        plt.close('all')
                countsss += 1
    import numpy as np
