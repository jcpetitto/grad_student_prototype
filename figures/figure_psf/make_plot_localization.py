"""
Title: Localization Precision Analysis with Background Estimation in PSF Fitting

Description:
This script analyzes the impact of background estimation methods on the localization precision in point spread function (PSF) fitting of single-molecule localization microscopy (SMLM) data. It compares three different background estimation modes:

1. **None**: No background estimation is used.
2. **Ground Truth (GT)**: The true background is known and used.
3. **Deep Learning (DL)**: A neural network estimates the background.

The script performs the following steps:

1. **Imports and Initial Setup**:
   - Imports necessary libraries and custom modules.
   - Sets random seeds for reproducibility.
   - Defines helper functions for visualization and statistical calculations.

2. **Simulation Parameters**:
   - Specifies the total number of spots (`numspots_total`) to simulate and the batch size (`batch_size`).
   - Defines parameters for the simulated PSF, background, and noise.
   - Sets up the modes for background estimation to be tested.

3. **Data Generation and Processing Loop**:
   - Loops over each background estimation mode.
   - For each mode, loops over batches to simulate data, estimate background, perform PSF fitting, and calculate localization precision.
   - Collects localization precision and background standard deviation data for analysis.

4. **Model Fitting**:
   - Fits a 2D Gaussian PSF with fixed sigma to the simulated data using maximum likelihood estimation (MLE).
   - Uses different background estimates depending on the mode.

5. **Analysis and Visualization**:
   - Calculates the localization precision (in pixels) for each spot.
   - Bins the data based on background standard deviation (`δ_bg`) and computes the mean localization precision and standard error in each bin.
   - Plots the localization precision as a function of background standard deviation for each background estimation mode.

6. **Results**:
   - Generates and saves a plot showing how background estimation affects localization precision under varying background noise conditions.

Functions:
- `show_tensor(image)`: Visualizes a tensor image using Napari.
- `show_napari(img)`: Visualizes an image using Napari.
- `normcdf(x, sigma, mu)`: Computes the cumulative distribution function of a normal distribution.

Usage:
- Ensure all required custom modules and data are available.
- Adjust the simulation parameters as needed.
- Run the script to perform the analysis.
- The resulting plot (`localization.svg`) will be saved in the current directory.

Dependencies:
- numpy
- torch
- matplotlib
- tqdm
- scienceplots
- scipy
- Custom modules:
  - `utils.Neural_networks`: Contains neural network models and functions.
  - `utils.psf_fit_utils`: Contains PSF fitting utilities.
  - `utils_psf`: Contains the `generate_rois_withnoise` function for data simulation.

Notes:
- The script uses a fixed PSF sigma value of 0.9 pixels.
- The localization precision is calculated in pixels and converted to nanometers (assuming a pixel size of 128 nm) for plotting.
- The neural network model used for background estimation (`noleakyv4_v_vectorpsf.pth`) should be trained and available at the specified path.
- The `generate_rois_withnoise` function simulates single-molecule images with specified noise characteristics.
- The script sets random seeds for reproducibility.

"""

import torch
import numpy as np
import torch.utils.data
import torch.nn as nn
from utils.Neural_networks import ConvolutionalNeuralNet, Unet_pp, Unet_pp_timeseries, glrtfunction
from utils.psf_fit_utils import Gaussian2DFixedSigmaPSF,LM_MLE_with_iter
import matplotlib.pyplot as plt
from utils_psf import generate_rois_withnoise
from tqdm import tqdm
from scipy.stats import chi2
import scienceplots

plt.style.use('science')
torch.manual_seed(42)
np.random.seed(42)


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
    return 0.5 * (1 + scipy.special.erf((x - mu) / (sigma * np.sqrt(2))))


# Set up the total number of spots and batch size
numspots_total = 200000  # Total number of spots you want to generate and process
batch_size = 40000  # Number of spots to process in each batch
num_batches = numspots_total // batch_size + (numspots_total % batch_size > 0)


numchannels = 20
iterationss = 1
dev = 'cuda'
roisize = 16  # 16 is neccesary for downsampling 3 steps! - otherwise errors later on
modelname = '../../trained_networks/model_wieghts_background_psf.pth'


# parameters for spot
bg_psf = (0, 0)
photons = (400, 400)
sigma = (0.9, 0.9)
vector = False

number_of_psf = 10 # number of 0=0%, 2=50%, 3=33%, 10 = 100%

# parameters for background
factorx = (-1.0, 1.0)
factory = (-1.0, 1.0)

mode = ['None',  'GT','DL']

posx = (roisize / 2 - 8, roisize / 2 + 8)
posy = (roisize / 2 - 8, roisize / 2 + 8)  # for NOISe!!!
border_from_roi = 8  # for PSF
roi_small = 8
num_channels=20

stdbg_list_tot = []

prec_list_tot = []


# loop over to save batches in folder
for mode_iter in mode:
    print(mode_iter)
    stdbg_list = []
    prec_list = []
    for batch_num in tqdm(range(num_batches)):
        with torch.no_grad():
            numspots = min(batch_size, numspots_total - batch_num * batch_size)
            delta_noise = (0, 100)
            offset_noise = (40, 20)
            offset_perlin = (0, 100)
            smp, target, target_mu, smp_bg, SNR, non_uniformity, norm, photons_m = generate_rois_withnoise(numspots,
                                                                                                           roisize, factorx,
                                                                                                           factory, posx, posy,
                                                                                                           delta_noise,
                                                                                                           offset_noise,
                                                                                                           offset_perlin,
                                                                                                           border_from_roi,
                                                                                                           photons, bg_psf,
                                                                                                           sigma, dev,
                                                                                                           number_of_psf,
                                                                                                           numchannels=numchannels,
                                                                                                           beaddim=[],
                                                                                                           vector=False, random=False)
            #show_tensor(smp)
            smp_bg = smp_bg * norm
            smp = smp * norm
            target = target * norm
            GT = False
            if mode_iter == 'GT':
                estimated_bg = target * 1
                GT=True
            elif mode_iter == 'DL':
                Unet_pp_instance = Unet_pp_timeseries(num_channels).to(dev)
                Unet_pp_instance.load_state_dict(torch.load(modelname)['model_state_dict'])
                Unet_pp_instance.eval()
                estimated_bg = Unet_pp_instance(smp/norm)*norm
            elif mode_iter == 'None':
                estimated_bg = None
            else:
                print('ERRORRRR')

            bounds = torch.tensor([[0, roi_small], [0, roi_small], [-1e5, 1e5], [-1e5, 1e5]]).to(dev)
            torch.cuda.empty_cache()

            # initial guess
            initial = torch.zeros((smp_bg.size(0), 4))
            initial[:, :2] = torch.tensor([roi_small / 2, roi_small / 2])  # position
            initial[:, 2] = 200  # photons
            initial[:, 3] = torch.mean(smp_bg[:, 0, ...], dim=(-1, -2))  # bg

            initial_ = torch.Tensor(initial).to(dev)
            if estimated_bg == None:
                estimated_bg_small=None
            else:
                estimated_bg_small = estimated_bg[:, 0, 4:12, 4:12]

            smp_bg_small = smp_bg[:, 0, 4:12, 4:12]
            stdbg = torch.std(target[:,0,4:12, 4:12], dim=(-1,-2))
            smp_small = smp[:, 0, 4:12, 4:12]

            # Fit the model
            model = Gaussian2DFixedSigmaPSF(roi_small, sigma=sigma[0])

            initial_[:, 3] = initial_[:, 3]
            mle = LM_MLE_with_iter(model, lambda_=1e-3, iterations=100, param_range_min_max=bounds,
                                   tol=torch.tensor([1e-3, 1e-3, 1e-2, 1e-2]).to(dev))
            #mle = torch.jit.script(mle)
            params_, loglik_, traces_ = mle.forward(smp_small.type(torch.float32),
                                                   initial_.type(torch.float32),
                                                   estimated_bg_small)

            traces_ = torch.permute(traces_,(1,0,2))
            traces = traces_.detach().cpu().numpy()
            x_deviation = (initial_[:, 0] - params_[:, 0]).detach().cpu().numpy()
            y_deviation = (initial_[:, 1] - params_[:, 1]).detach().cpu().numpy()
            loc_precision = np.sqrt(x_deviation**2 + y_deviation**2)

            stdbg_list.extend(stdbg.detach().cpu().numpy())
            prec_list.extend(loc_precision)
    stdbg_list_tot.append(stdbg_list)
    prec_list_tot.append(prec_list)
mu,jac = model.forward(params_, bg_constant=estimated_bg_small)
show_tensor(torch.concatenate((smp_small, mu),dim=-1))
bins = np.linspace(0, 20, num=11)
n_bootstrap_samples = 1000

# Setup for plotting
fig, ax = plt.subplots(figsize=(2, 2))
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
from cycler import cycler

custom_color_cycle = [default_colors[i] for i in [2, 3, 4]]
ax.set_prop_cycle(cycler('color', custom_color_cycle))

bin_width = 0.25  # Width of the bars
bin_centers = np.arange(1, len(bins))
modelabels = [r'$\text{bg}_\text{co}$', r'$\text{bg}_\text{GT}$', r'$\text{bg}_\text{NN}$']
for mode_index, (mode_label) in enumerate(modelabels):
    stdbg_np = np.array(stdbg_list_tot[mode_index])
    prec_np = np.array(prec_list_tot[mode_index])
    filter_mask = stdbg_np < 20
    prec_np = prec_np[filter_mask]
    stdbg_np = stdbg_np[filter_mask]

    bin_indices = np.digitize(stdbg_np, bins)
    prec_per_bin = np.zeros(len(bins) - 1)
    std_err_per_bin = np.zeros(len(bins) - 1)

    for i in range(1, len(bins)):
        prec_in_bin = prec_np[bin_indices == i]
        bootstrap_prec = []
        for _ in range(n_bootstrap_samples):
            if len(prec_in_bin) > 0:
                resampled_prec= np.random.choice(prec_in_bin, size=len(prec_in_bin), replace=True)

            else:
                print('EERRRRORRRR')
            bootstrap_prec.append(resampled_prec)

        prec_per_bin[i - 1] = np.mean(bootstrap_prec)
        std_err_per_bin[i - 1] = np.std(bootstrap_prec)

    # Adjust the position of bars for each mode to avoid overlap
    position_adjustment = (mode_index - 0.5) * bin_width
    positions = bin_centers + position_adjustment

    ax.bar(positions, prec_per_bin*128, width=bin_width, yerr=std_err_per_bin*128,
           capsize=1, label=mode_label)
ax.set_xticks(bin_centers)
bin_labels = [f"{bins[i]:.0f}-{bins[i + 1]:.0f}" for i in range(len(bins) - 1)]
ax.set_xticklabels(bin_labels, rotation=45)

ax.set_xlabel(r'$\delta_\text{bg}$ [photons]')
ax.set_ylabel('Loc. precision [nm]')
ax.set_ylim(0)
ax.legend()
plt.tight_layout(pad=0.1)
plt.savefig('localization.svg',format='svg')
plt.show()