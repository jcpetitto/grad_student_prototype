"""
Script to Compute and Plot Histograms of Translation Differences and Registration Metrics from Image Registrations.
Vybrant Violet stain vs brightfield

This script processes image data folders to compute translation differences between two registration modes.
It collects the translation differences, scale factors, and angles, along with mean images from multiple folders.
The script saves a stack of mean images and plots histograms with Gaussian fits for:
- Translation differences in the x and y directions.
- Overall translation magnitude.
- Differences in scale factors and angles between registrations.

Key Functionalities:
- Extracts numerical identifiers from folder names to process corresponding image files.
- Computes registration between images using specified modes.
- Accumulates translation differences, scale factors, and angles from multiple folders.
- Saves a stack of mean images as a TIFF file.
- Plots histograms of translation differences, scale differences, and angle differences with Gaussian fits.

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
plt.style.use('science')
import tqdm
from scipy.stats import poisson
import math
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
    'sigma': 1.3,
    'frames': [0, 1000],
    'frames_npcfit': [0,250],
    'drift_bins': 4,
     'resultdir': "/results/",
    'gain': '/home/pieter/Data/Yeast/bright_images_20ms.tiff',
    # 'gain': '/home/pieter/Data/astigmatist_cal_100mlam_1_MMStack_Default.ome.tif',
    'offset': '/home/pieter/Data/Yeast/dark_images_20ms.tiff',
    'model_NE': 'trained_networks/FPNresnet34_yeast.pt',
    'model_bg': 'trained_networks/noleakyv3.pth',
    'pixelsize': 128
}

def extract_digits(folder_name):
    # Use regular expression to extract digits from the folder name
    digits_match = re.search(r'\d+', folder_name)
    if digits_match:
        return digits_match.group()
    return None

def process_folder(folder_path):
    global cfg
    Yp = Yeast_processor(cfg)
    # Extract the digits from the folder name
    digits = extract_digits(os.path.basename(folder_path))
    if digits:
        torch.cuda.empty_cache()
        # Modify the cfg dictionary for the current folder
        cfg['fn_reg_npc1'] = '/BF1red' + digits + '.tiff'
        cfg['fn_reg_rnp1'] = '/BF1green' + digits + '.tiff'
        cfg['fn_reg_npc2'] = '/NEred' + digits + '.tiff'
        cfg['fn_reg_rnp2'] = '/RNAgreen' + digits + '.tiff'
        cfg['fn_track_rnp'] = '/RNAgreen' + digits + '.tiff'
        cfg['fn_track_npc'] = '/NEred' + digits + '.tiff'
        # Update the path in the configuration with the current folder path
        cfg['path'] = folder_path
        # Perform the processing steps for the current folder
        Yp = Yeast_processor(cfg)
       # Yp.calibrate(savefig=True)
        scale, angle, mean_img = Yp.compute_registration(regmode=1,plotfig=True,print_update = False)
        translation1 = Yp.registration['tvec']
        Yp.compute_registration(regmode=2,plotfig=False,print_update = True, scale = scale, angle=angle)
        translation2 = Yp.registration['tvec']
        Yp.compute_registration(regmode=2, plotfig=False, print_update=True)
        angle2 = Yp.registration['angle']*1
        scale2 = Yp.registration['scale']*1
        diff = translation2 - translation1

        return diff, mean_img,translation1,translation2, scale, scale2, angle,angle2
if __name__ == "__main__":
    dif_list = []
    base_dir = "/media/pieter/Extreme SSD/Dropbox (UMass Medical School)/Yeast-vybrant-violet/BMY_823 Vibrant Violet"
    root_dirs = os.listdir(base_dir)

    full_paths = [os.path.join(base_dir, item) for item in root_dirs]

    # root_dirs = ["/media/pieter/Extreme SSD/Yeast_tracking_data2023/BMY823/BMY823_7_16_23_aqsettings1_batchA",
    #             "/home/pieter/Data/Yeast/tracking data/BMY823_7_16_23_aqsettings1_batchB",
    #              "/home/pieter/Data/Yeast/tracking data/BMY823_7_20_23_aqsettings1_batchB"]
    dif_list = np.array([])  # Assuming dif_list is a numpy array as in the original code
    meanimg_list = []
    translation1_list = []
    translation2_list = []
    scale_list = []
    scale2_list = []
    angle_list = []
    angle2_list = []

    for root_dir in full_paths:
        # if len(dif_list) > 100:
        #     break
        for folder_name in tqdm.tqdm(os.listdir(root_dir)):
            # if len(dif_list)>100:
            #     break
            folder_path = os.path.join(root_dir, folder_name)
            # if folder_name == 'cell 20':
            if os.path.isdir(folder_path):
                try:
                    print(folder_path)
                    # pos,params_list, roi_pos_list, smpconcat, mu,boundingbox_coordinates, Yp = process_folder(folder_path)
                    diff, meanimg, translation1, translation2, scale, scale2, angle, angle2 = process_folder(
                        folder_path)

                    # Append the results to the respective lists
                    dif_list = np.append(dif_list, diff)
                    meanimg_list.append(meanimg)
                    translation1_list.append(translation1)
                    translation2_list.append(translation2)
                    scale_list.append(scale)
                    scale2_list.append(scale2)
                    angle_list.append(angle)
                    angle2_list.append(angle2)

                    # t_nump = traces[0].detach().cpu().numpy()
                    # t_nump1 = traces[1].detach().cpu().numpy()
                    #Yp.tracking(movie=False)
                    print(len(dif_list))
                except:
                    print('error, probably file not found')

    import numpy as np

stack = np.concatenate(meanimg_list)

import tifffile
#tifffile.imwrite('bf_stack_gfa.tiff',stack[100::,...] )
dif_list[dif_list == None] = 10

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)

def fit_gaussian(data, bins):
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[250, 0.0, 10])
    amplitude, mean, stddev = popt
    return amplitude, mean, stddev




def plot_histogram_with_gaussian(data, xlabel, binwidth=10, ylim=250, xlim=70,str_unit=' nm'):
    # Calculate the number of bins based on the desired binwidth
    data_range = max(data) - min(data)
    num_bins = int(data_range / binwidth)
    counts, _ = np.histogram(data,range=[min(data), max(data)], bins=num_bins)
    plt.figure(dpi=400,figsize=[2,3])
    plt.hist(data, range=[min(data), max(data)], bins=num_bins)
    plt.xlabel(xlabel)
    plt.xlim(-1 * xlim, 1 * xlim)
    #plt.ylim(0,ylim)
    amplitude, mean, stddev = fit_gaussian(data, num_bins)
    #plt.yticks([0,100,200])
    x_fit = np.linspace(-2 * xlim, 2 * xlim, 1000)
    y_fit = gaussian(x_fit, amplitude, mean, stddev)
    plt.plot(x_fit, y_fit, label=r'$\sigma={:.2f}$'.format(stddev)+str_unit +'\n'+'$\mu={:.2f}$'.format(mean)+str_unit)
    plt.ylabel('Counts')
    plt.legend(loc='upper right')  # Adjust the 'loc' parameter as needed
    plt.tight_layout()




# Assuming dif_list contains [x1, y1, x2, y2, ...] values
translation_x = np.array(dif_list[::2], dtype=float)
translation_y = np.array(dif_list[1::2], dtype=float)

dif_angle= np.array(angle_list, dtype=float)- np.array(angle2_list, dtype=float)
dif_scale = np.array(scale_list, dtype=float)- np.array(scale2_list, dtype=float)
# Now you can calculate sigma_reg without errors
sigma_reg = np.sqrt(translation_x**2 + translation_y**2)
sigma_reg = sigma_reg[sigma_reg<5]

# Plot histogram and Gaussian fit for translation x
plot_histogram_with_gaussian(translation_x*125, r'$t_{vybrant,x} - t_{BF,x}$ [nm]', binwidth=6, ylim=250)
plt.savefig('graphs/registx_vybrant.svg', format = 'svg')
plt.show()
plt.close('all')
# Plot histogram and Gaussian fit for translation y
plot_histogram_with_gaussian(translation_y*125,  r'$t_{vybrant,y} - t_{BF,y}$ [nm]', binwidth=6, ylim=250)
plt.savefig('graphs/registy_vybrant.svg', format = 'svg')
plt.show()
plt.close('all')
# Plot histogram and Gaussian fit for translation y
# Plot histogram and Gaussian fit for translation x
plot_histogram_with_gaussian(dif_angle, r'$\alpha_{vybrant} - \alpha_{BF} \quad [^{\circ}]$ ', binwidth=0.05,
                             ylim=250,xlim=1, str_unit=' $^\circ$')
plt.savefig('graphs/angle_vybrant.svg', format = 'svg')
plt.show()
plt.close('all')
# Plot histogram and Gaussian fit for translation y
plot_histogram_with_gaussian(dif_scale,  r'$S_{vybrant} - S_{BF}$',
                             binwidth=0.001, ylim=250,xlim=0.02)
plt.savefig('graphs/scale_vybrant.svg', format = 'svg')
plt.show()
plt.close('all')
# Plot histogram and Gaussian fit for translation y
