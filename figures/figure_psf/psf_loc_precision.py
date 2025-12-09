"""
Title: Calculation of Localization Precision Using CRLB for RNA Molecules in Single-Molecule Localization Microscopy

Description:
This script calculates the localization precision of RNA molecules by computing the Cramér-Rao Lower Bound (CRLB) from microscopy data. The main steps of the script are:

1. **Data Loading**:
   - Loads pre-processed single-molecule image data (`smp_ori_v2.npy`) and corresponding background images (`bg_smp_v2_ori.npy`).
   - These data files contain regions of interest (ROIs) extracted from microscopy images, prepared for PSF fitting.

2. **Model Setup**:
   - Initializes a Gaussian 2D PSF model with a fixed sigma (`sigma = 0.9` pixels).
   - This model represents the point spread function used in the maximum likelihood estimation (MLE) fitting process.

3. **Batch Processing and MLE Fitting**:
   - Splits the data into batches to manage memory usage during computation.
   - For each batch:
     - Sets initial guesses and parameter bounds for the MLE fitting.
     - Performs MLE fitting using the Levenberg-Marquardt algorithm (`LM_MLE_with_iter`).
     - Estimates the PSF parameters (position, photons, background) for each ROI.
     - Computes the CRLB for each estimated parameter.

4. **CRLB Calculation and Visualization**:
   - Concatenates the CRLB results from all batches.
   - Calculates the localization precision by combining the CRLBs of the x and y positions.
   - Converts the localization precision from pixels to nanometers (assuming a pixel size of 128 nm).
   - Plots a histogram of the localization precision for the RNA molecules.
   - Calculates and displays the median localization precision on the plot.

Dependencies:
- numpy
- torch
- matplotlib
- tqdm
- scienceplots
- tifffile
- Custom modules:
  - `utils.psf_fit_utils` (contains `Gaussian2DFixedSigmaPSF`, `LM_MLE_with_iter`, `compute_crlb`)

Usage:
- Ensure that the data files `smp_ori_v2.npy` and `bg_smp_v2_ori.npy` are available in the current directory.
- Adjust the batch size and model parameters as needed.
- Run the script to perform the MLE fitting and compute the CRLBs.
- The script will generate and display a histogram of the localization precision and save it as `crlb_rna.svg`.

Notes:
- The CRLB provides a theoretical lower bound on the variance of unbiased estimators, representing the best possible localization precision under given noise conditions.
- The localization precision is an important metric in single-molecule localization microscopy, indicating the accuracy with which molecule positions can be determined.

"""


import os
import re
import pickle

import tifffile
import torch.cuda
import matplotlib.pyplot as plt
import tqdm
import scienceplots
import numpy as np
import os

from utils.psf_fit_utils import VectorPSF_2D, LM_MLE_with_iter,Gaussian2DFixedSigmaPSF, compute_crlb
import scienceplots


def show_napari(img):
    import napari
    viewer = napari.imshow(img)

def show_tensor(img):
    import napari
    viewer = napari.imshow(img.detach().cpu().numpy())




plt.style.use('science')
smp_ori = np.load('./smp_ori_v2.npy', )[:, :, :]
bg_smp_ori = np.load('./bg_smp_v2_ori.npy')[:, :, :]
bead_diam_list = np.linspace(0, 120, 5)
# bead_diam_list = [0]
error_list_tot = []
std_list_tot = []
mulist = []
params_list = []
sigma=0.9

#show_napari(bg_smp_ori)

model = Gaussian2DFixedSigmaPSF(8, sigma )

# Define the batch size
batch_size = 3000

# Calculate the number of batches
num_batches = smp_ori.shape[0] // batch_size
bg_smp_ori = bg_smp_ori[:,4:12,4:12]
smp_ori = smp_ori[:,4:12,4:12]
# Create batches
smp_batched = np.array_split(smp_ori, num_batches, axis=0)
bg_smp_batched = np.array_split(bg_smp_ori, num_batches, axis=0)
error_list = []
std_list = []
mu_list_temp = []
params_list_temp = []
num = 1

crlb_list = []
for smp, bg_smp in tqdm.tqdm(zip(smp_batched, bg_smp_batched)):
    dev = 'cuda'
    roisize = 8
    bounds_mle = [[1, roisize - 1],
                  [1, roisize - 1],
                  [1, 1e9],
                  [1, 1e6]
                  ]

    initial_guess_mle = [roisize / 2, roisize / 2, 200, 3]

    # bounds estimator
    param_range = torch.Tensor(bounds_mle).to(dev)

    # initial guess
    initial = np.zeros((np.size(smp, 0), 4))

    initial[:, :2] = np.array([initial_guess_mle[0], initial_guess_mle[1]])  # position
    initial[:, 2] = initial_guess_mle[2]  # photons
    initial[:, 3] = np.mean(bg_smp, axis=(-1, -2))  # bg

    initial_ = torch.Tensor(initial).to(dev)

    # mle = LM_MLE_with_iter(model, lambda_=lmlambda, iterations=iterations, param_range_min_max=param_range,
    #                        tol=torch.tensor([1e-3,1e-3,1e-2,1e-2]).to(self.dev))

    mle = LM_MLE_with_iter(model, lambda_=0.01, iterations=300,
                           param_range_min_max=param_range,
                           tol=torch.tensor([1e-3, 1e-3, 1e-2, 1e-2]).to(dev))
    # mle = torch.jit.script(mle)
    bg_smp_ = torch.tensor(bg_smp).to(dev)
    smp_ = torch.tensor(smp).to(dev)

    params_, loglik_, traces = mle.forward(smp_.type(torch.float32),
                                           initial_.type(torch.float32), bg_smp_.type(torch.float32)
                                           )
    mu, jac = model.forward(params_, bg_smp_)
    crlb = compute_crlb(mu.to(torch.float32),jac.to(torch.float32))
    crlb_list.append(crlb)

    mu = mu  # [:,4:12,4:12]
    smp_ = smp_  # [:,4:12,4:12]

    error = (abs(mu - smp_)).detach().cpu().numpy()
    #show_tensor(torch.concatenate((smp_,mu), dim=-1))
    error_list.append(error)
    mu_list_temp.append(mu)
    params_list_temp.append(params_)
    test_traces = torch.permute(traces, (1, 0, 2)).detach().cpu().numpy()
    num += 1


crlb_all = torch.concatenate(crlb_list).detach().cpu().numpy()

data = np.sqrt(crlb_all[:,0]**2 + crlb_all[:,0]**2) * 128

# Plot the histogram
plt.figure(figsize=(2.5,2.5))
plt.hist(data, bins=40, density=True, range=(0,100))
plt.xlabel(r'$\sigma_{\text{RNA}}$ [nm]')
plt.ylabel('Probability')

# Calculate and plot the median line
median_value = np.median(data)
plt.axvline(median_value, color='r', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.0f} nm')

plt.legend()
plt.tight_layout()
plt.savefig('crlb_rna.svg', format='svg')
plt.show()


# sbg = torch.concatenate(params_list_temp)[:, 2] / torch.concatenate(params_list_temp)[:, 3]
# # if iteration == 0:
# #     filter = sbg > 5
# #     print(sum(filter))
# params_list.append(torch.concatenate(params_list_temp))
# error_list_tot.append(np.concatenate(error_list))
# mulist.append(torch.concatenate(mu_list_temp))
#
# plot_list = []
# test_list = []
# rmsd_list = []
# errorbar_list = []
#
# for it, errors in enumerate(error_list_tot):
#     if it == 0:
#         paramsplot = params_list[it]
#         filter_temp = paramsplot[:, 2].detach().cpu().numpy() > 0
#
#     chi_2 = (
#         np.sum(np.sum((errors[filter_temp, ...] ** 2) / (mulist[it][filter_temp, ...]).detach().cpu().numpy(), axis=1),
#                axis=1))
#     print(torch.mean(params_list[it][:, 2]))
#     print(torch.mean(params_list[it][:, 3]))
#     errorbar_list.append(np.std(chi_2))
#     rmsd_list.append(np.median(chi_2))
#     plot_list.append(np.mean(errors))
#     test_list.append(np.mean(np.mean(errors, axis=1), axis=1))
#
# plt.figure(figsize=(2, 2))
#
# plt.plot(sigma_list, rmsd_list, marker='.', )
# # plt.errorbar(sigma_list,rmsd_list,errorbar_list)
# plt.ylabel(r'$\chi^2$ error [photons]')
# plt.xlabel(r'$\sigma$ PSF [nm]')
# # plt.ylim([334, 356])
# plt.tight_layout()
# plt.savefig('chi_error_psf.svg', format='svg')
#
# plt.show()
# show_tensor(torch.concatenate(
#     (torch.tensor(smp_ori[filter.detach().cpu().numpy()]).to(mulist[5].device), mulist[5]), dim=-1))
# show_tensor(torch.concatenate(
#     (torch.tensor(smp_ori[filter]).to(mulist[0].device), torch.tensor(bg_smp_ori).to(mulist[0].device),
#      mulist[0], mulist[1], mulist[2], mulist[3], mulist[4], mulist[5], mulist[6], mulist[7], mulist[8]), dim=-1))
# show_napari(np.concatenate((smp_ori, bg_smp_ori), axis=-1))
# import tifffile
#
# tifffile.imwrite('smp.tiff', smp_ori[0])
# tifffile.imwrite('bg.tiff', bg_smp_ori[0])
# tifffile.imwrite('mu.tiff', mulist[5][0].detach().cpu().numpy())
#
# # tifffile.imwrite('../psfstack.tiff', torch.concatenate(
# #     (torch.tensor(smp_ori).to(mulist[0].device), torch.tensor(bg_smp_ori).to(mulist[0].device),
# #     mulist[0], mulist[1], mulist[2], mulist[3], mulist[4], mulist[5], mulist[6], mulist[7], mulist[8]), dim=-1).detach().cpu().numpy())