"""
Title: Statistical Analysis of GLRT Test Statistics under Different Background Conditions

Description:

This script performs a statistical analysis of the Generalized Likelihood Ratio Test (GLRT) test statistics under different background conditions. It aims to evaluate the impact of background modeling on the detection performance by:

1. **Data Simulation**:
   - Generating synthetic Regions of Interest (ROIs) with varying levels of noise and background using the `generate_rois_withnoise` function.
   - Simulating both background-only images and images containing a Point Spread Function (PSF) signal.

2. **GLRT Application**:
   - Applying the GLRT to background-only images with the background included in the model.
   - Applying the GLRT to background-subtracted images to assess the effect of background removal.

3. **Statistical Analysis**:
   - Calculating the Probability of False Alarm (PFA) for both cases by comparing the test statistics to a theoretical threshold derived from the chi-squared distribution.
   - Collecting statistics over multiple iterations to compute the mean and standard deviation of the PFA.

4. **Visualization**:
   - Plotting histograms of the test statistics for both cases.
   - Comparing the empirical distributions with the theoretical chi-squared distribution.
   - Displaying the mean and standard deviation of the PFA across iterations.

Dependencies:

- `numpy`
- `torch`
- `matplotlib`
- `tqdm`
- `scipy`
- `scienceplots`
- Custom modules:
  - `Neural_networks` (should contain `ConvolutionalNeuralNet`, `Unet_pp`, `Unet_pp_timeseries`, `glrtfunction`)
  - `utils` (should contain `generate_rois_withnoise`)

Usage:

- Ensure that the required custom modules are available and correctly imported.
- Adjust the parameters in the script (e.g., number of spots, noise levels) as needed.
- Run the script to perform the analysis and generate the plots.
- The script will save the generated plots as SVG files.

Notes:

- Random seeds are set for both NumPy and PyTorch to ensure reproducibility.
- The script runs multiple simulations (iterations) to collect statistics on the PFA.
- Two scenarios are compared:
  - **Background Included in Model**: The background is included in the GLRT model during detection.
  - **Background Subtracted**: The background is subtracted from the data before applying the GLRT.
- The GLRT function and data generation functions are assumed to be defined in the custom modules.

"""


import torch
import numpy as np
import torch.utils.data
import torch.nn as nn
from utils.Neural_networks import ConvolutionalNeuralNet, Unet_pp,Unet_pp_timeseries, glrtfunction
import matplotlib.pyplot as plt
import os
os.path
from utils_sup import generate_rois_withnoise
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

mode = ['GT']

posx = (roisize / 2 -8, roisize / 2 +8)
posy = (roisize / 2 -8, roisize / 2 +8) # for NOISe!!!
border_from_roi = 8  # for PSF
roi_small = 8
# loop over to save batches in folder

list1 = []
list2 = []

for _ in range(50):
    with torch.no_grad():

        delta_noise = (0, 200)
        offset_noise = (0, 200)
        offset_perlin = (0, 200)
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
        smp_bg_small = torch.clamp(smp_bg_small-estimated_bg_small+torch.amin(estimated_bg_small,dim=[-1,-2])[:,None,None],0)
        initial[:, 3] = torch.mean(smp_small[:,0,...], dim=(-1, -2))  # bg

        ratio_psf, _, _, _, mu_bg_psf, traces_bg_all_psf, traces_int_all_psf = glrtfunction(smp_bg_small, 100000, bounds, initial_, roi_small, sigma[1],
                                                         tol=torch.tensor([1e-4, 1e-4]), bg_constant=None)




        test_traces = traces_int_all.detach().cpu().numpy()

        ratio_np = ratio.detach().cpu().numpy()
        ratio_psf_np = ratio_psf.detach().cpu().numpy()
        pfa = 2 * normcdf(-np.sqrt(ratio_np))
        gamma = 1 / normcdf(0.05 / 2) ** 2
        num_exceeding = np.sum(ratio_np > gamma)
        percentage_exceeding = (num_exceeding / len(ratio_np)) * 100

        pfa2 = 2 * normcdf(-np.sqrt(ratio_psf_np))
        gamma = 1 / normcdf(0.05 / 2) ** 2
        num_exceeding = np.sum(ratio_psf_np > gamma)
        percentage_exceeding2 = (num_exceeding / len(ratio_psf_np)) * 100
        list1.append(percentage_exceeding)
        list2.append(percentage_exceeding2)


plt.figure(figsize=(5, 2.5))  # Adjusted for two subplots side by side

# Create a subplot for the first histogram
ax1 = plt.subplot(1, 2, 1)
if mode_iter == 'GT' or mode_iter == 'None':
    label = 'background'
    label_psf = 'signal'

# Plot the first histogram
ax1.hist(ratio_np, bins=np.linspace(0, 10, 100), density=True, alpha=0.6, label='bg in model')


x = np.linspace(0.02, 20, 500)
chi_squared_dist = chi2.pdf(x, 1)
ax1.plot(x, chi_squared_dist, lw=1)
ax1.vlines(gamma, 0, 3, color='black', linestyle=':')
ax1.set_ylabel('Probability')
ax1.set_xlabel('Test statistic $T_G$')
ax1.set_ylim(0, 1)
ax1.set_title('mean =' + str(np.round(np.mean(np.array(list1)),3)) + ' std = '+ str(np.round(np.std(np.array(list1)),3)))
ax1.set_xlim(0, 10)
ax1.legend()

# Create a subplot for the second histogram, sharing the y-axis with the first
ax2 = plt.subplot(1, 2, 2, sharey=ax1)
# Plot the second histogram
ax2.hist(ratio_psf_np, bins=np.linspace(0, 10, 100), density=True, alpha=0.6, label='bg subtracted')
ax2.set_xlabel('Test statistic $T_G$')
ax2.plot(x, chi_squared_dist, lw=1, label=r'$\chi_1^2$ PDF')
ax2.vlines(gamma, 0, 3, label=r'5\% $P_\text{FA}$ thresh.', color='black', linestyle=':')
ax2.set_title('mean =' + str(np.round(np.mean(np.array(list2)),3)) + 'std = '+ str(np.round(np.std(np.array(list2)),3)))
# Only need to set the label for the y-axis on the first subplot
# ax2.set_ylabel('Probability')  # Not needed, shared with ax1
ax2.set_xlim(0, 10)
ax2.legend()

plt.tight_layout()
plt.savefig('subtractplot_shared_y' + mode_iter + '.svg', format='svg')
plt.show()


