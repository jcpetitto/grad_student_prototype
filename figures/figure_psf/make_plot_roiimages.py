"""
Title: Visualization of PSF Fitting Results with Background Correction Using Deep Learning

Description:
This script simulates point spread function (PSF) images with varying background noise levels, performs background estimation using a deep learning model, fits the PSF to the simulated data using maximum likelihood estimation (MLE), and visualizes the results. The main goal is to demonstrate the effectiveness of background correction in PSF fitting and to illustrate the impact of background noise on localization precision.

The script performs the following steps:

1. **Imports and Initialization**:
   - Imports necessary libraries and modules.
   - Sets random seeds for reproducibility.
   - Defines helper functions for visualization.

2. **Simulation Parameters**:
   - Sets up parameters for simulating spots (PSFs) and background noise.
   - Defines the number of spots, batch size, ROI size, PSF parameters, and background noise parameters.

3. **Data Generation and Processing**:
   - Loops over batches to generate simulated ROIs with noise.
   - Uses the `generate_rois_withnoise` function to create simulated data.
   - Performs background estimation using a deep learning model (`Unet_pp_timeseries`) if specified.
   - Fits the PSF model to the simulated data using MLE, both with and without background correction.

4. **Localization Precision Calculation**:
   - Calculates the localization precision by comparing the estimated positions with the true positions.

5. **Visualization**:
   - Selects examples with specific background noise levels.
   - Concatenates the simulated images, ground truth, estimated backgrounds, and fitted models for visualization.
   - Displays and saves the final image showing the impact of background correction.

Functions:
- `show_tensor(image)`: Visualizes a tensor image using Napari.
- `show_napari(img)`: Visualizes a numpy array image using Napari.
- `normcdf(x, sigma, mu)`: Computes the cumulative distribution function of the normal distribution.

Usage:
- Ensure that all required modules and data files are available.
- Adjust the paths and parameters as needed (e.g., `modelname` for the deep learning model).
- Run the script to generate the simulated data, perform PSF fitting, and visualize the results.
- The final image will be saved as 'rois.png' and displayed using Napari.

Dependencies:
- numpy
- torch
- matplotlib
- tqdm
- scipy
- scienceplots
- Custom modules:
  - `utils.Neural_networks`: Contains neural network models and functions.
  - `utils.psf_fit_utils`: Contains PSF fitting utilities.
  - `utils_psf`: Contains the `generate_rois_withnoise` function for data simulation.

Notes:
- The script uses a fixed PSF sigma value of 0.9 pixels.
- The deep learning model (`Unet_pp_timeseries`) should be pre-trained and available at the specified path.
- The script currently runs in 'DL' mode, using the deep learning model for background estimation.
- Random seeds are set for reproducibility.

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
numspots_total = 5000  # Total number of spots you want to generate and process
batch_size = 5000  # Number of spots to process in each batch
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

mode = ['DL']

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
            delta_noise = (0, 60)
            offset_noise = (0, 40)
            offset_perlin = (0, 80)
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
            initial[:, 2] = 0  # photons
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
            mu,_ = model.forward(params_,estimated_bg_small )
            params_no_cor, _, _ = mle.forward(smp_small.type(torch.float32),
                                                    initial_.type(torch.float32),
                                                    None)
            mu_no_correction,_ = model.forward(params_,None )
            traces_ = torch.permute(traces_,(1,0,2))
            traces = traces_.detach().cpu().numpy()
            x_deviation = (initial_[:, 0] - params_[:, 0]).detach().cpu().numpy()
            y_deviation = (initial_[:, 1] - params_[:, 1]).detach().cpu().numpy()
            loc_precision = np.sqrt(x_deviation**2 + y_deviation**2)

            stdbg_list.extend(stdbg.detach().cpu().numpy())
            prec_list.extend(loc_precision)
    stdbg_list_tot.append(stdbg_list)
    prec_list_tot.append(prec_list)


tensor =target[:,0,4:12,4:12].detach().cpu().numpy()
stdbg_np = np.array(stdbg_list)

# Define the target values
target_values = np.array([0.011, 4.1, 8.1, 12, 16, 20])

# Find the indexes of the values in stdbg_np that are closest to each target value
indexes_closest = np.array([np.abs(stdbg_np - tv).argmin() for tv in target_values])

# Access the corresponding elements in tensor
selected_tensors2 = np.concatenate(np.array([tensor[idx, ...] for idx in indexes_closest]), axis=-1)
selected_tensors4 = np.concatenate(np.array([mu.detach().cpu().numpy()[idx, ...] for idx in indexes_closest]), axis=-1)
selected_tensors3 = np.concatenate(np.array([mu_no_correction.detach().cpu().numpy()[idx, ...] for idx in indexes_closest]), axis=-1)
selected_tensors = np.concatenate(np.array([smp_small.detach().cpu().numpy()[idx, ...] for idx in indexes_closest]), axis=-1)
selected_tensors1 = np.concatenate(np.array([(target[:,0,...]+target_mu*norm).detach().cpu().numpy()[idx, 4:12,4:12] for idx in indexes_closest]), axis=-1)

# Concatenate these tensors along the last axis
final_imag = np.concatenate((selected_tensors,selected_tensors1,selected_tensors2,selected_tensors3,selected_tensors4),axis=-2)
final_imag[:,0:16]+=40
# concatenate all tensors here to 1 image:


plt.figure(figsize=(final_imag.shape[1] / 100.0, final_imag.shape[0] / 100.0), dpi=100)  # Adjust figure size to image dimensions
plt.imshow(final_imag, cmap='gray')  # Display the image
plt.axis('off')  # Remove axes
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding and margins
plt.savefig('rois.png', bbox_inches='tight', pad_inches=0)  # Save the figure
plt.close()  # Close the figure to free up memory
show_napari(final_imag)

