"""
Title: PSF Fitting and Error Analysis Script

Description:
This script performs fitting of a 2D Gaussian Point Spread Function (PSF) to simulated or experimental microscopy image data. It processes batches of image data, fits the PSF model using maximum likelihood estimation (MLE), and analyzes the fitting errors and parameter estimates.

Steps:
1. **Import Libraries**: Imports necessary libraries and custom modules.
2. **Define Helper Functions**:
   - `show_napari(img)`: Function to display an image using Napari (optional).
   - `show_tensor(img)`: Function to display a tensor as an image using Napari (optional).
3. **Load Data**:
    - Use create_stack_rnp.py to create sample ROIS
   - Loads sample images (`smp_ori`) and background images (`bg_smp_ori`) from `.npy` files.
4. **Initialize Variables**:
   - Sets up lists to store errors, parameters, and other results.
   - Defines the PSF model using `Gaussian2DFixedSigmaPSF`.
5. **Batch Processing**:
   - Splits the data into batches for efficient processing.
   - Iterates over each batch:
     - Sets up initial guesses and bounds for the MLE fitting.
     - Performs MLE fitting using `LM_MLE_with_iter`.
     - Calculates errors between the fitted model and the data.
6. **Visualization**:
   - Plots a histogram of the background standard deviations.
7. **Optional Analysis**:
   - Additional code is provided (commented out) for further analysis, such as calculating chi-squared errors and visualizing fitting results.

Usage:
- Ensure all required data files (`smp_ori_v2.npy`, `bg_smp_v2_ori.npy`) are available in the specified directory.
- Adjust file paths and parameters as needed.
- Run the script to perform the PSF fitting and error analysis.
- Results are displayed and can be saved as figures.

Dependencies:
- numpy
- matplotlib
- tqdm
- torch
- scienceplots
- os
- Custom modules from `utils.psf_fit_utils`:
  - `LM_MLE_with_iter`
  - `Gaussian2DFixedSigmaPSF`

Notes:
- The script uses custom modules from `utils.psf_fit_utils`; ensure these are available and properly installed.
- The plotting style is set to 'science' for high-quality figures suitable for publications.
- Visualization functions using Napari are included but optional; they require the `napari` package.

"""


import torch.cuda
import matplotlib.pyplot as plt
import tqdm
import scienceplots
import numpy as np
import os
from utils.psf_fit_utils import LM_MLE_with_iter,Gaussian2DFixedSigmaPSF

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
    mu = mu  # [:,4:12,4:12]
    smp_ = smp_  # [:,4:12,4:12]

    error = (abs(mu - smp_)).detach().cpu().numpy()
    #show_tensor(torch.concatenate((smp_,mu), dim=-1))
    error_list.append(error)
    mu_list_temp.append(mu)
    params_list_temp.append(params_)
    test_traces = torch.permute(traces, (1, 0, 2)).detach().cpu().numpy()
    num += 1

plt.figure(figsize=(3,3))
plt.hist(np.std(bg_smp_ori[:,3:13],axis=(1,2)),bins=40, density=True, label = 'Experimental ROIs')
plt.xlabel(r'$\sigma_{\text{bg}}$ [photons]')
plt.ylabel(r'Probability')
plt.legend()
plt.tight_layout()
plt.savefig('background_experimental.svg', format='svg')
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