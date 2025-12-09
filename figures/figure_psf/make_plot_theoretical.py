"""
Title: Generalized Likelihood Ratio Test (GLRT) Simulation for PSF Detection

Description:
This script simulates point spread function (PSF) images with specified background noise characteristics and performs a Generalized Likelihood Ratio Test (GLRT) to detect the presence of a signal (PSF) against the background noise. The script aims to demonstrate that the test statistic under the null hypothesis (background only) follows a chi-squared distribution with one degree of freedom (χ²₁). It also visualizes the test statistics for both background and signal cases.

The main steps include:
1. **Simulation of PSF Images**:
   - Generates synthetic images with and without PSF signals.
   - Adds background noise with specified parameters.

2. **GLRT Computation**:
   - Performs GLRT on the simulated images to compute test statistics for background and signal cases.
   - Uses custom functions to fit the PSF and compute likelihood ratios.

3. **Statistical Analysis**:
   - Computes the distribution of the test statistic under the null hypothesis.
   - Compares the empirical distribution with the theoretical chi-squared distribution.

4. **Visualization**:
   - Plots histograms of the test statistics for both background and signal.
   - Overlays the theoretical chi-squared probability density function (PDF).
   - Marks the threshold corresponding to a 5% probability of false alarm (PFA).

Functions:
- `show_tensor(image)`: Visualizes a tensor image using Napari (optional).
- `show_napari(img)`: Visualizes a NumPy array image using Napari (optional).
- `normcdf(x, sigma, mu)`: Computes the cumulative distribution function of the normal distribution.

Usage:
- Ensure that all required modules and custom functions are available.
- Adjust the parameters as needed (e.g., number of spots, noise levels).
- Run the script to perform the GLRT simulation and generate the plots.
- The resulting plot will be saved as an SVG file with the filename `perfectpfaplot<mode>.svg`.

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
- The `mode` variable can be adjusted to include different background estimation methods.
- The script sets random seeds for reproducibility.

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
numspots =50000 # for trainging
#numspots =300# for 100 channel

numchannels = 20
iterationss =1
dev = 'cuda'
roisize = 16 # 16 is neccesary for downsampling 3 steps! - otherwise errors later on

# parameters for spot
bg_psf = (0, 0)
photons = (150, 150)
sigma = (0.9,0.9)
vector = False

number_of_psf = 10  # number of 0=0%, 2=50%, 3=33%, 10 = 100%

# parameters for background
factorx = (-1.0, 1.0)
factory = (-1.0, 1.0)

mode = ['None']

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

    bounds = torch.tensor([[0, roi_small], [0, roi_small], [-1e5, 1e5], [-1e5, 1e5]]).to(dev)
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
    ratio, _, _, _, mu_bg, traces_bg_all, traces_int_all = glrtfunction(smp_bg_small, 100000, bounds, initial_, roi_small, sigma[1],
                                                     tol=torch.tensor([1e-4, 1e-4]), bg_constant=estimated_bg_small)

    ratio_psf, _, _, _, mu_bg_psf, traces_bg_all_psf, traces_int_all_psf = glrtfunction(smp_small, 100000, bounds, initial_, roi_small, sigma[1],
                                                     tol=torch.tensor([1e-4, 1e-4]), bg_constant=estimated_bg_small)




    test_traces = traces_int_all.detach().cpu().numpy()

    ratio_np = ratio.detach().cpu().numpy()
    ratio_psf_np = ratio_psf.detach().cpu().numpy()
    plt.figure(figsize=(2.5,2.5))
    if mode_iter == 'GT':
        label = r'background'
        label_psf = r'signal'
    if mode_iter == 'None':
        label = r'background'
        label_psf = r'signal'
    plt.hist(ratio_np,bins=np.linspace(0,20,100), density=True, alpha=0.6,label=label)
    plt.hist(ratio_psf_np, bins=np.linspace(0, 20, 100), density=True, alpha=0.6, label=label_psf)
    pfa = 2 * normcdf(-np.sqrt(ratio_np))
    gamma = 1/normcdf(0.05/2)**2
    num_exceeding = np.sum(ratio_np > gamma)

    # Compute the percentage of points exceeding the gamma threshold
    percentage_exceeding = (num_exceeding / len(ratio_np)) * 100
    # Generate values for the x axis
    x = np.linspace(0.02, 20, 500)
    # Generate the chi squared distribution for these x values
    chi_squared_dist = chi2.pdf(x, 1)

    plt.plot(x, chi_squared_dist, lw=1, label=r'$\chi_1^2$ PDF')
    plt.vlines(gamma,0,3, label=r'5\% $P_\text{FA}$ thresh.',color='black', linestyle=':' )
    plt.ylabel('Probability')
    plt.xlabel('Test statistic $T_G$')
    plt.ylim(0,1)
    plt.xlim(0,20)
    plt.legend()
    plt.tight_layout()
    plt.savefig('perfectpfaplot'+mode_iter+'.svg', format='svg')
    plt.show()