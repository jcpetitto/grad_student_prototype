"""
Title: Visualization of GLRT Results and ROIs in PSF Detection Using Ground Truth Background

Description:
This script simulates point spread function (PSF) images with varying background noise levels, performs a Generalized Likelihood Ratio Test (GLRT) for PSF detection using the ground truth background, and visualizes the results. The primary goal is to demonstrate the performance of the GLRT in detecting PSFs in noisy images and to visualize the regions of interest (ROIs) involved in the test.

The script performs the following steps:

1. **Imports and Initialization**:
   - Imports necessary libraries and modules.
   - Sets random seeds for reproducibility.
   - Defines helper functions for visualization.

2. **Simulation Parameters**:
   - Sets up parameters for simulating spots (PSFs) and background noise.
   - Defines the number of spots, batch size, ROI size, PSF parameters, and background noise parameters.

3. **Data Generation**:
   - Generates simulated ROIs with noise using the `generate_rois_withnoise` function.
   - Scales the simulated data by a normalization factor.

4. **GLRT Computation**:
   - Performs the GLRT on both background-only images and images containing PSFs.
   - Uses the ground truth background in the GLRT (`mode = ['GT']`).

5. **Visualization**:
   - Concatenates the ground truth background, simulated images, estimated PSF images, and estimated background images.
   - Selects a specific ROI for visualization.
   - Displays and saves the concatenated ROIs as an image.

Functions:
- `show_tensor(image)`: Visualizes a tensor image using Napari.
- `show_napari(img)`: Visualizes a NumPy array image using Napari.
- `normcdf(x, sigma, mu)`: Computes the cumulative distribution function of the normal distribution.

Usage:
- Ensure that all required modules and data files are available.
- Adjust the paths and parameters as needed (e.g., number of spots, noise levels).
- Run the script to generate the simulated data, perform GLRT, and visualize the ROIs.
- The final image will be saved as 'allrois<mode_iter>.svg' and can be displayed using Napari.

Dependencies:
- numpy
- torch
- matplotlib
- tqdm
- scipy
- scienceplots
- Custom modules:
  - `utils.Neural_networks`: Contains neural network models and functions (e.g., `glrtfunction`).
  - `utils_psf`: Contains the `generate_rois_withnoise` function for data simulation.

Notes:
- The script uses a fixed PSF sigma value of 0.9 pixels.
- The background noise parameters are set to simulate a challenging detection scenario.
- The script sets random seeds for reproducibility.
- The `mode` variable is set to `['GT']` to use the ground truth background.

"""

import torch
import numpy as np
import torch.utils.data
import torch.nn as nn
from utils.Neural_networks import ConvolutionalNeuralNet, Unet_pp,Unet_pp_timeseries, glrtfunction
import matplotlib.pyplot as plt
from utils_psf import generate_rois_withnoise
from tqdm import tqdm
from scipy.stats import chi2
import scienceplots
plt.style.use('science')
def show_tensor(image):
    import napari
    viewer = napari.Viewer()
    viewer.add_image(image.detach().cpu().numpy())
def show_napari(img):
    import napari
    viewer = napari.Viewer()  # that's it!
    viewer.add_image(img)
def normcdf(x, sigma=1.0, mu=0.0):
    import scipy
    return 0.5 * (1 + scipy.special.erf((x-mu) / (sigma*np.sqrt(2))))

np.random.seed(42)  # Set the seed to an arbitrary value, e.g., 42
torch.manual_seed(42)  # Set the seed to an arbitrary value, e.g., 42
torch.cuda.manual_seed(42)
# general parameters
numspots =1000 # for trainging
#numspots =300# for 100 channel

numchannels = 20
iterationss =1
dev = 'cuda'
roisize = 16 # 16 is neccesary for downsampling 3 steps! - otherwise errors later on

# parameters for spot
bg_psf = (0, 0)
photons = (200, 200)
sigma = (0.9,0.9)
vector = False

number_of_psf = 10  # number of 0=0%, 2=50%, 3=33%, 10 = 100%

# parameters for background
factorx = (-1.0, 1.0)
factory = (-1.0, 1.0)

mode = ['GT']

posx = (roisize / 2 -8, roisize / 2 +8)
posy = (roisize / 2 -8, roisize / 2 +8) # for NOISe!!!
border_from_roi = 8  # for PSF
roi_small = 8
# loop over to save batches in folder
with torch.no_grad():

    delta_noise = (80, 80)
    offset_noise = (1, 1)
    offset_perlin = (80, 80)
    smp, target, target_mu, smp_bg, SNR, non_uniformity, norm, photons_m = generate_rois_withnoise(numspots,
                           roisize, factorx, factory, posx, posy, delta_noise, offset_noise,offset_perlin,
                        border_from_roi, photons, bg_psf, sigma, dev, number_of_psf,numchannels = numchannels,
                       beaddim=[], vector=False,random=False)
    smp_bg = smp_bg*norm
    smp = smp*norm
    target = target*norm
    for mode_iter in mode:
        if mode_iter == 'GT':
            estimated_bg = target*1
        elif mode_iter == 'DL':
            pass
        elif mode_iter == 'None':
            estimated_bg= None
        else:
            print('ERRORRRR')

    bounds = torch.tensor([[0, roi_small], [0, roi_small], [0, 1e5], [0, 1e5]]).to(dev)
    torch.cuda.empty_cache()

    # initial guess
    initial = torch.zeros((smp_bg.size(0), 4))
    initial[:, :2] = torch.tensor([roi_small/2, roi_small/2])  # position
    initial[:, 2] = 0  # photons
    initial[:, 3] = torch.mean(smp_bg[:,0,...], dim=(-1, -2))  # bg

    initial_ = torch.Tensor(initial).to(dev)
    if estimated_bg == None:
        estimated_bg_small=None
    else:
        estimated_bg_small = estimated_bg[:, 0, 4:12, 4:12]

    smp_bg_small = smp_bg[:,0,4:12, 4:12]
    smp_small = smp[:, 0, 4:12, 4:12]
    gt_small = (target[:, 0, 4:12, 4:12]+target_mu[:, 4:12, 4:12]*norm)
    ratio, _, _, mu_psf_bg, mu_bg, traces_bg_all, traces_int_all = glrtfunction(smp_bg_small, 100000, bounds, initial_, roi_small, sigma[1],
                                                     tol=torch.tensor([1e-4, 1e-4]), bg_constant=estimated_bg_small)

    ratio_psf, _, _, mu_psf_psf, mu_bg_psf, traces_bg_all_psf, traces_int_all_psf = glrtfunction(smp_small, 100000, bounds, initial_, roi_small, sigma[1],
                                                     tol=torch.tensor([1e-4, 1e-4]), bg_constant=estimated_bg_small)

    bg_rois = torch.concatenate((target[:, 0, 4:12, 4:12],smp_bg_small,mu_psf_bg,mu_bg),dim=-1)
    psf_rois = torch.concatenate((gt_small,smp_small,mu_psf_psf,mu_bg_psf),dim=-1)

    allrois = torch.concatenate((bg_rois,psf_rois),dim=-2)
    show_tensor(allrois)
    allrois_np = allrois[779,...].detach().cpu().numpy()
    plt.figure()  # You can adjust the figure size as needed
    plt.imshow(allrois_np, aspect='equal', cmap='gray')  # Choose a colormap that fits your data
    plt.axis('off')  # Turn off the axis for a cleaner look
    plt.tight_layout()
    # Save the figure as SVG
    plt.savefig("allrois"+mode_iter+".svg", format='svg', bbox_inches='tight')
    plt.close()