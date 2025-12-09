"""
Title: Simulation and Analysis of Background Estimation Methods in GLRT Detection

Description:
This script simulates image data with varying background noise levels to compare the performance of different background estimation methods in Generalized Likelihood Ratio Test (GLRT) detection. It evaluates the probability of false alarms (P_FA) for three modes of background estimation:

1. **No Background Estimation (`'None'`)**: Uses the raw image data without any background correction.
2. **Ground Truth Background (`'GT'`)**: Uses the true background used in the simulation as the background estimate.
3. **Deep Learning Background Estimation (`'DL'`)**: Uses a trained neural network (U-Net) to estimate the background.

The script performs the following steps:

1. **Setup and Initialization**:
   - Imports necessary libraries and modules.
   - Sets random seeds for reproducibility.
   - Defines helper functions for visualization and statistical calculations.

2. **Simulation Parameters**:
   - Defines parameters for spot generation, background, and detection settings.
   - Specifies the number of spots, batch size, and number of channels.

3. **Data Generation and Processing**:
   - Iterates over the different background estimation modes.
   - Generates synthetic image data with noise using `generate_rois_withnoise`.
   - For each mode, processes the data:
     - If using ground truth background (`'GT'`), it uses the true background.
     - If using the neural network (`'DL'`), it loads the pre-trained U-Net model to estimate the background.
     - If no background estimation (`'None'`), it proceeds without background correction.
   - Performs GLRT detection using `glrtfunction` with the estimated background.
   - Collects statistics such as the standard deviation of the background and detection ratios.

4. **Statistical Analysis**:
   - Calculates the probability of false alarms (P_FA) for each mode.
   - Uses bootstrapping to estimate the standard error of the P_FA in different background noise bins.

5. **Visualization**:
   - Plots a bar chart comparing the P_FA across different background noise levels for each background estimation mode.
   - Plots histograms of the signal (photons) and mean background to show their distributions.

Functions:
- `show_tensor(image)`: Displays a tensor image using Napari (optional).
- `show_napari(img)`: Displays an image using Napari (optional).
- `normcdf(x, sigma, mu)`: Computes the cumulative distribution function (CDF) of a normal distribution.

Usage:
- Ensure that all required modules and data are available.
- Adjust the simulation parameters and paths to models as needed.
- Run the script to perform the simulation, detection, and analysis.
- The results will be displayed as plots and can be saved as SVG files.

Dependencies:
- numpy
- matplotlib
- tqdm
- torch
- scienceplots
- scipy
- Custom modules:
  - `utils.Neural_networks`: Contains `ConvolutionalNeuralNet`, `Unet_pp`, `Unet_pp_timeseries`, `glrtfunction`.
  - `utils_psf`: Contains `generate_rois_withnoise`.

Notes:
- The script uses a pre-trained neural network model for background estimation. Ensure that the model file path is correct.
- The `generate_rois_withnoise` function generates synthetic data with noise; adjust parameters as needed.
- Visualization functions using Napari are optional and require the `napari` package to be installed.
- The plotting style is set to 'science' using the `scienceplots` package for publication-quality figures.
- The script includes bootstrapping for statistical estimation, which may increase computation time.

Important Variables:
- `mode`: A list specifying the background estimation modes to compare.
- `numspots_total`: Total number of spots to simulate.
- `batch_size`: Number of spots to process per batch.
- `modelname`: Path to the pre-trained U-Net model for background estimation.

"""

import torch
import numpy as np
import torch.utils.data
import torch.nn as nn
from utils.Neural_networks import ConvolutionalNeuralNet, Unet_pp, Unet_pp_timeseries, glrtfunction
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
batch_size = 20000  # Number of spots to process in each batch
num_batches = numspots_total // batch_size + (numspots_total % batch_size > 0)


numchannels = 20
iterationss = 1
dev = 'cuda'
roisize = 16  # 16 is neccesary for downsampling 3 steps! - otherwise errors later on
modelname = '../../trained_networks/model_wieghts_background_psf.pth'

# parameters for spot
bg_psf = (0, 0)
photons = (50, 600) # not used!
sigma = (0.9, 0.9)
vector = False

number_of_psf = 3 # number of 0=0%, 2=50%, 3=33%, 10 = 100%

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

ratio_list_tot = []
photon_list = []
bg_list = []

# loop over to save batches in folder
for mode_iter in mode:
    print(mode_iter)
    stdbg_list = []
    ratio_list = []

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
                                                                                                           vector=False)

            smp_bg = smp_bg * norm
            smp = smp * norm
            target = target * norm
            mean_target = torch.mean(target, dim=(1, 2, 3))
            signal = photons_m[:,0]

            GT = False
            if mode_iter == 'GT':
                estimated_bg = target * 1
                GT=True
                photon_list.append(signal.detach().cpu().numpy())
                bg_list.append(mean_target.detach().cpu().numpy())
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
            initial[:, 2] = 0  # photons
            initial[:, 3] = torch.mean(smp_bg[:, 0, ...], dim=(-1, -2))  # bg

            initial_ = torch.Tensor(initial).to(dev)
            if estimated_bg == None:
                estimated_bg_small=None
            else:
                estimated_bg_small = estimated_bg[:, 0, 4:12, 4:12]

            smp_bg_small = smp_bg[:, 0, 4:12, 4:12]
            stdbg = torch.std(target[:,0,4:12, 4:12], dim=(-1,-2))

            # plt.hist(stdbg.detach().cpu().numpy(),bins=50)
            # plt.show()
            ratio, _, _, _, mu_bg, traces_bg_all, traces_int_all = glrtfunction(smp_bg_small, 10000000, bounds, initial_,
                                                                                roi_small, sigma[1],
                                                                                tol=torch.tensor([1e-4, 1e-4]),
                                                                                bg_constant=estimated_bg_small, GT=GT)


            test_traces = traces_int_all.detach().cpu().numpy()
            ratio_np = ratio.detach().cpu().numpy()
            pfa = 2 * normcdf(-np.sqrt(ratio_np))
            gamma = 1 / normcdf(0.05 / 2) ** 2
            stdbg_list.extend(stdbg.detach().cpu().numpy())
            ratio_list.extend(ratio_np)
    stdbg_list_tot.append(stdbg_list)
    ratio_list_tot.append(ratio_list)

gamma = 1 / normcdf(0.05 / 2) ** 2
num_exceeding = np.sum(ratio_np > gamma)
percentage_exceeding = (num_exceeding / len(ratio_np)) * 100

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
    ratios_np = np.array(ratio_list_tot[mode_index])
    filter_mask = stdbg_np < 20
    ratios_np = ratios_np[filter_mask]
    stdbg_np = stdbg_np[filter_mask]

    bin_indices = np.digitize(stdbg_np, bins)
    percentage_exceeding_per_bin = np.zeros(len(bins) - 1)
    std_err_per_bin = np.zeros(len(bins) - 1)

    for i in range(1, len(bins)):
        ratios_in_bin = ratios_np[bin_indices == i]
        bootstrap_percentages = []
        for _ in range(n_bootstrap_samples):
            if len(ratios_in_bin) > 0:
                resampled_ratios = np.random.choice(ratios_in_bin, size=len(ratios_in_bin), replace=True)
                num_exceeding = np.sum(resampled_ratios > gamma)
                percentage_exceeding = (num_exceeding / len(resampled_ratios)) * 100
            else:
                percentage_exceeding = 0
            bootstrap_percentages.append(percentage_exceeding)

        percentage_exceeding_per_bin[i - 1] = np.mean(bootstrap_percentages)
        std_err_per_bin[i - 1] = np.std(bootstrap_percentages)

    # Adjust the position of bars for each mode to avoid overlap
    position_adjustment = (mode_index - 0.5) * bin_width
    positions = bin_centers + position_adjustment
    # If you have a single bar or category and just want the first color

    ax.bar(positions, percentage_exceeding_per_bin, width=bin_width, yerr=std_err_per_bin,
           capsize=1, label=mode_label)
ax.axhline(y=5, color='black', linestyle='--', linewidth=1, label='Set 5\% threshold')

ax.set_xticks(bin_centers)
bin_labels = [f"{bins[i]:.0f}-{bins[i + 1]:.0f}" for i in range(len(bins) - 1)]
ax.set_xticklabels(bin_labels, rotation=45)

ax.set_xlabel(r'$\delta_\text{bg}$ [photons]',fontdict={'size':8})
ax.set_ylabel(r'$P_\text{FA}^,$ [\%]')
ax.legend()
plt.tight_layout(pad=0.1)
plt.savefig('detection_improv.svg',format='svg')
plt.show()

# Create subplots
fig, axs = plt.subplots(1, 2, dpi=400, figsize=(4, 2), sharex='col')

# Plot histogram for photon_list
axs[0].hist(np.array(photon_list).flatten(), bins=30, density=True, alpha=0.6, color='gray',range=[0,1000])
axs[0].set_xlabel('Signal [photons]')
axs[0].set_ylabel('Probability')
#axs[0].set_title('Photon Distribution')

# Plot histogram for bg_list
axs[1].hist(np.array(bg_list).flatten(), bins=30, density=True, alpha=0.6, color='gray',range=[0,300],label='Sim.')
axs[1].set_xlabel('Mean background\n [photons/pixel]')
axs[1].set_ylabel('Probability')
#axs[1].set_title('Background Distribution')
fig.legend()
plt.tight_layout(pad=0.1)
plt.savefig('simulation_distributions.svg',format='svg')
plt.show()