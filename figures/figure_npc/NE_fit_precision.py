"""
Title: Estimation of Localization Precision in Yeast Nuclear Envelope Fitting

Description:
This script processes microscopy images of yeast cells to estimate the localization precision of nuclear envelope (NE) fitting. It performs the following steps:

1. **Configuration and Setup**:
   - Imports necessary libraries and modules.
   - Sets plotting styles for consistent visualization.
   - Defines helper functions for visualization using Napari (optional).

2. **Data Processing**:
   - Defines a configuration dictionary `cfg` with parameters for data processing.
   - Iterates over directories containing yeast cell images.
   - Extracts digits from folder names to identify image files.
   - Processes each folder using the `Yeast_processor` class (assumed to be defined in `utils.Yeast_processor`).

3. **NE Fitting and Parameter Extraction**:
   - Detects nuclear pore complexes (NPCs) using `detect_npc`.
   - Refines NPC fits and extracts parameters such as mean positions (`mune1`, `mune2`) and sigma values (`sigma1`, `sigma2`).
   - Calculates the difference in mean positions (`diff`) between two fits.

4. **Data Aggregation**:
   - Collects the extracted parameters across all cells.
   - Counts the number of processed cells.

5. **Statistical Analysis and Visualization**:
   - Defines functions for fitting Gaussian and Cauchy distributions to the data.
   - Plots histograms of the localization differences and sigma values.
   - Fits a Gaussian to the histogram and overlays the fit on the plot.
   - Saves the plots as SVG files for high-quality vector graphics.

Usage:
- Update the `cfg` dictionary with the correct file paths and parameters specific to your data.
- Ensure all required data files and models are available in the specified paths.
- Run the script to process the data and generate the histograms.
- The resulting plots will help assess the localization precision of NE fitting in yeast cells.

Dependencies:
- numpy
- torch
- matplotlib
- tqdm
- re
- os
- scienceplots
- scipy
- Custom module: `Yeast_processor` from `utils.Yeast_processor`

Notes:
- The script uses a custom `Yeast_processor` class, which must be defined in a module accessible to the script.
- File paths in the configuration should be updated to reflect the correct locations of your data and models.
- The plotting style is set to 'science' for high-quality figures suitable for publications.
- Adjust the directory paths and file names according to your dataset.
- The script includes error handling to skip folders without the required data.

"""

import os
import tqdm
import re
import torch
import matplotlib.pyplot as plt
from utils.Yeast_processor import Yeast_processor
from scipy.stats import norm
from scipy.optimize import curve_fit
import numpy as np

import scienceplots
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('science')


def show_napari(img):
    import napari
    viewer = napari.imshow(img)
def show_napari_points(img, points):
    import napari
    viewer = napari.Viewer()
    viewer.add_image(img)
    viewer.add_points(points, symbol='cross', size=3, face_color='red', edge_width=0 )

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
    'model_NE': '../trained_networks/FPNresnet34_yeastvAnyee.pt',
    'model_bg': '../trained_networks/noleakyv2.pth',
    'pixelsize': 128
}
def extract_digits(folder_name):
    # Use regular expression to extract digits from the folder name
    digits_match = re.search(r'\d+', folder_name)
    if digits_match:
        return digits_match.group()
    return None
diff_all_cells = []
sigma1_all_cells = []
sigma2_all_cells = []
if __name__ == "__main__":
    dif_list = []
    base_dir = "/media/pieter/Extreme SSD/Yeast_tracking_data2023/BMY823"
    root_dirs = os.listdir(base_dir)

    full_paths = [os.path.join(base_dir, item) for item in root_dirs]

    # root_dirs = ["/media/pieter/Extreme SSD/Yeast_tracking_data2023/BMY823/BMY823_7_16_23_aqsettings1_batchA",
    #             "/home/pieter/Data/Yeast/tracking data/BMY823_7_16_23_aqsettings1_batchB",
    #              "/home/pieter/Data/Yeast/tracking data/BMY823_7_20_23_aqsettings1_batchB"]
    count = 0
    numcells = 0
    for root_dir in full_paths:
        # if numcells > 1000:
        #     break
        for folder_name in tqdm.tqdm(os.listdir(root_dir)):
            # if numcells > 20:
            #     break
            # if count>20:
            #     break
            folder_path = os.path.join(root_dir, folder_name)
            #if folder_name == 'cell 20':
            if os.path.isdir(folder_path):

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

                        # Yp.calibrate(savefig=False)
                        #Yp.compute_registration()
                        #Yp.compute_drift(save_fig=True)
                        logits = Yp.detect_npc(save_fig=False, count_good_label=40, gap_closing_distance=10,
                                               threshold=0.05)

                        import numpy as np


                        params1, params2 = Yp.refinement_npcfit_movie_new(movie=False, registration = False,
                                                                                               smoothness = 10,
                                                                          Lambda=0.001,length_line=12,estimate_prec=True, number_mean=250, save_fig=False,max_signs=np.inf,iterations=300)

                        for listlen in range(len(params1)):
                            params1_temp = params1[listlen]
                            params1_temp = params1_temp[0]
                            mune1=params1_temp[:,7]
                            sigma1=params1_temp[:,8]
                            params2_temp = params2[listlen]
                            params2_temp = params2_temp[0]
                            mune2 = params2_temp[:, 7]
                            sigma2 = params2_temp[:, 8]
                            diff = mune1-mune2

                            sigma1_all_cells.append(sigma1)
                            sigma2_all_cells.append(sigma2)
                            diff_all_cells.append(diff)
                        import numpy as np

                        numcells = numcells + len(params1)
                        print(' number of cells = ', numcells)
                        count +=1
                        print(count)
                except:
                    print('no file')



def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)
def cauchy(x, amplitude, x0, gamma):
    return amplitude / (np.pi * gamma * (1 + ((x - x0) / gamma)**2))
def fit_gaussian(data, bins,p0=[1, 0.0, 10]):
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=p0, method='lm')
    amplitude, mean, stddev = popt
    return amplitude, mean, stddev




def plot_histogram_with_gaussian(data, xlabel, binwidth=10, ylim=250,
                                 xlim=[-150,150],p0=[1, 0.0, 10]):
    # Calculate the number of bins based on the desired binwidth
    data_range = max(data) - min(data)
    num_bins = int(data_range / binwidth)
    counts, _ = np.histogram(data,range=[min(data), max(data)], bins=num_bins,density=True)
    plt.figure(dpi=400,figsize=(2,2))
    plt.hist(data, range=[min(data), max(data)], bins=num_bins,density=True)
    plt.xlabel(xlabel)
    plt.xlim(xlim)
    #plt.ylim(0,ylim)
    #plt.yticks([0,100,200,300])
    amplitude, mean, stddev = fit_gaussian(data, num_bins,p0=p0)

    x_fit = np.linspace(xlim[0], xlim[1], 5000)
    y_fit = gaussian(x_fit, amplitude, mean, stddev)
    plt.plot(x_fit, y_fit, label=r'Gaussian fit,' + '\n' +r'$\sigma={:.2f}$nm'.format(stddev)+ '\n'+r'$\mu={:.2f}$nm'.format( mean))
    plt.ylabel('Probability')
    plt.legend(loc='upper right')  # Adjust the 'loc' parameter as needed
    plt.tight_layout(pad=0.2)
    plt.savefig('graphs/precisicion_graph.svg', format='svg')
    plt.show()

diff_all_cells_array = np.array(np.concatenate(diff_all_cells))
sigma1_array = np.array(np.concatenate(sigma1_all_cells))
#np.save('precision.npy',diff_all_cells_array )
# Plot histogram and Gaussian fit for translation x
plot_histogram_with_gaussian(diff_all_cells_array*128, r'$\Delta$ peak NE fit [nm]', binwidth=5, ylim=200)

plot_histogram_with_gaussian(diff_all_cells_array*128, r'$\Delta$ peak NE fit [nm]', binwidth=5, ylim=200)

plot_histogram_with_gaussian(sigma1_array*128, r'Sigma [nm]', binwidth=20,xlim=[0,400],p0=[1,300,100])

