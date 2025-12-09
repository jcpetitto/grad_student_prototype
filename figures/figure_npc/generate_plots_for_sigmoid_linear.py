"""
Title: Yeast NPC Model Comparison and Error Analysis Script

Description:
This script compares different models (Linear Gaussian, Sigmoid Gaussian, and Logistic Gaussian) for fitting yeast nuclear pore complex (NPC) microscopy data. It performs the following steps:

1. **Data Loading and Preprocessing**:
   - Iterates over specified data directories to process each cell's data.
   - Uses the `Yeast_processor` class to detect NPCs and refine NPC fits.
   - Collects fitting errors and intensity values for all cells.

2. **Model Fitting**:
   - Defines initial guesses and bounds for the Linear and Sigmoid models.
   - Fits the models to the data using maximum likelihood estimation.
   - Checks if the fitted parameters are within specified bounds.
   - Plots the model fits for comparison.

3. **Error Calculation**:
   - Calculates mean relative errors and mean squared errors (MSE) for each model.
   - Filters out invalid or extreme values.
   - Compares the MSE and mean errors between models.

4. **Visualization**:
   - Generates plots to visualize the MSE and mean ensemble errors for each model.
   - Saves the plots as SVG files in the specified directory.

Functions:
- `extract_digits(folder_name)`: Extracts numerical digits from a folder name to identify image files.
- `construct_initial_guess_linear(val_array_tensor, length_line)`: Constructs initial guesses for the linear model parameters.
- `construct_initial_guess_sigmoid(val_array_tensor, length_line)`: Constructs initial guesses for the sigmoid model parameters.
- `fit_and_calculate_error(val_array_tensor, points_tensor, ...)`: Fits the models to the data and calculates errors.

Usage:
- Ensure all required data files and models are available in the specified paths.
- Adjust the base directory `base_dir` and other configuration parameters as needed.
- Run the script to process the data and generate the plots.
- Results will be saved as SVG files in the specified directory.

Dependencies:
- numpy
- matplotlib
- tqdm
- torch
- scienceplots
- re
- os
- Custom modules: `Yeast_processor`, `LM_MLE_forspline_new`, `LinearGaussianFitClass`, `SigmoidGaussianFitClass` (should be available in the Python path)

Notes:
- The script uses custom classes from `Yeast_processor` and `ne_fit_utils` modules, which must be defined and accessible.
- File paths should be updated to reflect the correct locations of your data and models.
- The plotting style is set to 'science' for high-quality figures suitable for publications.
"""

import os
import re
import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from utils.Yeast_processor import Yeast_processor
from utils.ne_fit_utils import LM_MLE_forspline_new, LinearGaussianFitClass, SigmoidGaussianFitClass
import scienceplots
plt.style.use('science')
length_line = 18
def show_napari(img):
    import napari
    viewer = napari.imshow(img)

def show_napari_points(img, points):
    import napari
    viewer = napari.Viewer()
    viewer.add_image(img)
    viewer.add_points(points, symbol='cross', size=3, face_color='red', edge_width=0)

def extract_digits(folder_name):
    digits_match = re.search(r'\d+', folder_name)
    return digits_match.group() if digits_match else None

def construct_initial_guess_linear(val_array_tensor, length_line=length_line):
    batch_size, x = val_array_tensor.shape
    initial_guess = torch.zeros(batch_size, 6)
    initial_guess[:, 0] = (val_array_tensor[:, -1] - val_array_tensor[:, 0]) / length_line
    initial_guess[:, 1] = val_array_tensor[:, 0]
    initial_guess[:, 2] = length_line/2
    initial_guess[:, 3] = 1.5
    initial_guess[:, 4] = 150
    initial_guess[:, 5] = 0
    return initial_guess

def construct_initial_guess_sigmoid(val_array_tensor, length_line=16):
    batch_size, x = val_array_tensor.shape
    initial_guess = torch.zeros(batch_size, 7)
    initial_guess[:, 0] = (val_array_tensor[:, -1] - val_array_tensor[:, 0])
    initial_guess[:, 1] = 1
    initial_guess[:, 2] = length_line/2
    initial_guess[:, 3] = length_line/2
    initial_guess[:, 4] = 1.5
    initial_guess[:, 5] = 150
    initial_guess[:, 6] = torch.min(val_array_tensor, dim=-1).values - (val_array_tensor[:, -1] - val_array_tensor[:, 0])
    return initial_guess

bounds_linear = [
    [-8e4, 8e4],
    [-8e4, 8e4],
    [0, length_line],
    [1e-3, length_line],
    [-np.inf, np.inf],
    [-np.inf, np.inf]
]

bounds_sigmoid = [
    [-8e4, 8e4],
    [-np.inf, np.inf],
    [0, length_line],
    [0, length_line],
    [1e-3, length_line],
    [-np.inf, np.inf],
    [-np.inf, np.inf]
]


def fit_and_calculate_error(val_array_tensor, points_tensor, batch_size=2, length_line=16, offset=0, gain=1, fitvals = None):
    num_batches = val_array_tensor.shape[0] // batch_size
    mean_relative_errors_linear = []
    mean_relative_errors_sigmoid = []
    vals_lin = []
    fitvals_lin = []
    vals_sig = []
    fitvals_sig = []
    # Define bounds and 2% thresholds for both models
    bounds_linear = [
        [-8e4, 8e4],
        [-8e4, 8e4],
        [0, length_line],
        [1e-3, length_line],
        [-np.inf, np.inf],
        [-np.inf, np.inf]
    ]
    bounds_sigmoid = [
        [-8e4, 8e4],
        [-np.inf, np.inf],
        [0, length_line],
        [0, length_line],
        [1e-3, length_line],
        [-np.inf, np.inf],
        [-np.inf, np.inf]
    ]

    # Calculate 98% bound thresholds for both models
    lower_threshold_linear = [b[0] + 0.02 * (b[1] - b[0]) if np.isfinite(b[0]) else -np.inf for b in bounds_linear]
    upper_threshold_linear = [b[1] - 0.02 * (b[1] - b[0]) if np.isfinite(b[1]) else np.inf for b in bounds_linear]
    lower_threshold_sigmoid = [b[0] + 0.02 * (b[1] - b[0]) if np.isfinite(b[0]) else -np.inf for b in bounds_sigmoid]
    upper_threshold_sigmoid = [b[1] - 0.02 * (b[1] - b[0]) if np.isfinite(b[1]) else np.inf for b in bounds_sigmoid]

    # Boolean array to store validity for all instances in val_array_tensor
    valid_instances = np.ones(num_batches*batch_size, dtype=bool)
    for i in tqdm.tqdm(range(num_batches)):
        batch_data = val_array_tensor[i * batch_size:(i + 1) * batch_size]
        batch_points = points_tensor[i * batch_size:(i + 1) * batch_size]
        if fitvals is not None:
            fitvals_batched = fitvals[:,i * batch_size:(i + 1) * batch_size]
        # Linear Model
        initial_guess_linear = construct_initial_guess_linear(batch_data, length_line=length_line)
        model_linear = LinearGaussianFitClass(batch_points.cuda())
        good_arr = (torch.ones_like(initial_guess_linear).to('cuda')).type(torch.bool)[:, 0]
        mle_linear = LM_MLE_forspline_new(model_linear).cuda()
        param_range_linear = torch.tensor(bounds_linear, dtype=torch.float32).cuda()

        params_linear, _, _ = mle_linear.forward(
            initial_guess_linear.type(torch.cuda.FloatTensor),
            batch_data[..., None].type(torch.cuda.FloatTensor),
            param_range_linear, 200, 0.01
        )

        # Check if params_linear are near bounds
        params_linear_np = params_linear.detach().cpu().numpy()
        within_bounds_linear = np.all(
            (params_linear_np > lower_threshold_linear) & (params_linear_np < upper_threshold_linear), axis=1
        )

        fit_linear, _ = model_linear.forward(params_linear.to('cuda'), good_arr)
        fit_linear = fit_linear.squeeze().detach().cpu().numpy()

        # Sigmoid Model
        initial_guess_sigmoid = construct_initial_guess_sigmoid(batch_data, length_line=length_line)
        model_sigmoid = SigmoidGaussianFitClass(batch_points.cuda())
        mle_sigmoid = LM_MLE_forspline_new(model_sigmoid).cuda()
        param_range_sigmoid = torch.tensor(bounds_sigmoid, dtype=torch.float32).cuda()

        params_sigmoid, _, _ = mle_sigmoid.forward(
            initial_guess_sigmoid.type(torch.cuda.FloatTensor),
            batch_data[..., None].type(torch.cuda.FloatTensor),
            param_range_sigmoid, 200, 0.01
        )

        fit_sigmoid, _ = model_sigmoid.forward(params_sigmoid.to('cuda'), good_arr)
        fit_sigmoid = fit_sigmoid.squeeze().detach().cpu().numpy()
        # Check if params_sigmoid are near bounds
        params_sigmoid_np = params_sigmoid.detach().cpu().numpy()
        within_bounds_sigmoid = np.all(
            (params_sigmoid_np > lower_threshold_sigmoid) & (params_sigmoid_np < upper_threshold_sigmoid), axis=1
        )
        # Combine bounds-checking results for both models
        within_bounds_combined = within_bounds_linear & within_bounds_sigmoid
        valid_instances[i * batch_size: (i + 1) * batch_size] = within_bounds_combined

        x_points = batch_points[0].detach().cpu().numpy()
        num=2
        # Combined plot for Linear, Sigmoid, and Logistic Models without markers
        fig, ax = plt.subplots(figsize=(2, 2))

        # Define colorblind-friendly colors
        color_linear = "#0072B2"  # Blue
        color_sigmoid = "#009E73"  # Green
        color_logistic = "#D55E00"  # Orange

        # Data points
        ax.plot(x_points, (batch_data[num].detach().cpu().numpy() - offset) * gain, label='Data', color='black')

        # Linear model fit
        ax.plot(x_points, (fit_linear[num] - offset) * gain, label='Linear + Gaussian Model Fit', color=color_linear)

        # Sigmoid model fit
        ax.plot(x_points, (fit_sigmoid[num] - offset) * gain, label='Sigmoid + Gaussian Model Fit', color=color_sigmoid)

        # Logistic model fit
        ax.plot(x_points, (fitvals_batched.T[num] - offset) * gain, label='Logistic + Gaussian Model Fit',
                color=color_logistic)

        # Labels and legend
        ax.set_xlabel("Position [pixel]", fontsize=10)
        ax.set_ylabel("Intensity [photons]", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout(pad=0.2)

        plt.savefig(f'utils/combined_model_fit_batch_{i}.svg', format='svg')
        plt.show()
        plt.close(fig)

        # Individual plots for Linear, Sigmoid, and Logistic Models with shared x-axis, no markers
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(2.5,5), sharex=True)

        # Linear model
        ax1.plot(x_points, (batch_data[num].detach().cpu().numpy() - offset) * gain, label='Data', color='black')
        ax1.plot(x_points, (fit_linear[num] - offset) * gain, label='Linear + Gaussian Model Fit', color=color_linear)
        ax1.set_ylabel("Intensity [photons]", fontsize=10)
        ax1.legend(loc="upper right", fontsize=8)

        # Sigmoid model
        ax2.plot(x_points, (batch_data[num].detach().cpu().numpy() - offset) * gain, label='Data', color='black')
        ax2.plot(x_points, (fit_sigmoid[num] - offset) * gain, label='Sigmoid + Gaussian Model Fit', color=color_sigmoid)
        ax2.set_ylabel("Intensity [photons]", fontsize=10)
        ax2.legend(loc="upper right", fontsize=8)

        # Logistic model
        ax3.plot(x_points, (batch_data[num].detach().cpu().numpy() - offset) * gain, label='Data', color='black')
        ax3.plot(x_points, (fitvals_batched.T[num] - offset) * gain, label='Logistic + Gaussian Model Fit',
                 color=color_logistic)
        ax3.set_ylabel("Intensity [photons]", fontsize=10)
        ax3.set_xlabel("Position [pixel]", fontsize=10)
        ax3.legend(loc="upper right", fontsize=8)

        # Adjust layout
        plt.tight_layout(h_pad=0.4)  # Adds padding between plots for clarity
        plt.savefig(f'utils/shared_x_model_fit_batch_{i}.svg', format='svg')
        plt.show()
        plt.close(fig)

        # Calculate relative errors for linear and sigmoid models
        batch_data_np = batch_data.detach().cpu().numpy()
        mean_relative_error_linear = (fit_linear - batch_data_np) / fit_linear
        mean_relative_errors_linear.append(mean_relative_error_linear)

        mean_relative_error_sigmoid = (fit_sigmoid - batch_data_np) / fit_sigmoid
        mean_relative_errors_sigmoid.append(mean_relative_error_sigmoid)

        # Append data for final concatenation
        vals_lin.append(batch_data_np)
        fitvals_lin.append(fit_linear)
        vals_sig.append(batch_data_np)
        fitvals_sig.append(fit_sigmoid)

    # Concatenate all batches
    mean_relative_error_sigmoid = np.concatenate(mean_relative_errors_sigmoid)
    mean_relative_error_linear = np.concatenate(mean_relative_errors_linear)
    vals_lin = np.concatenate(vals_lin)
    fitvals_lin = np.concatenate(fitvals_lin)
    vals_sig = np.concatenate(vals_sig)
    fitvals_sig = np.concatenate(fitvals_sig)

    return mean_relative_error_linear, mean_relative_error_sigmoid, vals_lin, fitvals_lin, vals_sig, fitvals_sig, valid_instances




    # print(f'Mean relative error for linear model: {mean_error_linear}')
    # print(f'Mean relative error for sigmoid model: {mean_error_sigmoid}')


base_dir = "/media/pieter/Extreme SSD/Yeast_tracking_data2023/BMY823"
root_dirs = os.listdir(base_dir)
full_paths = [os.path.join(base_dir, item) for item in root_dirs]

error_all_cells, values_all_cells = [], []
numcells, count = 0, 0

for root_dir in full_paths:
    for folder_name in tqdm.tqdm(os.listdir(root_dir)):
        if numcells > 2:
            break
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            try:
                digits = extract_digits(os.path.basename(folder_path))
                if digits:
                    torch.cuda.empty_cache()
                    cfg = {
                        'path': folder_path,
                        'fn_reg_npc1': f'/BF1red{digits}.tiff',
                        'fn_reg_rnp1': f'/BF1green{digits}.tiff',
                        'fn_reg_npc2': f'/BF2red{digits}.tiff',
                        'fn_reg_rnp2': f'/BF2green{digits}.tiff',
                        'fn_track_rnp': f'/RNAgreen{digits}.tiff',
                        'fn_track_npc': f'/NEred{digits}.tiff',
                        'roisize': 8,
                        'sigma': 1.3,
                        'frames': [0, 2000],
                        'frames_npcfit': [0, 250],
                        'drift_bins': 4,
                        'resultdir': "/results/",
                        'gain': '/home/pieter/Data/Yeast/bright_images_20ms.tiff',
                        'offset': '/home/pieter/Data/Yeast/dark_images_20ms.tiff',
                        'model_NE': '../trained_networks/FPNresnet34_yeastvAnyee.pt',
                        'model_bg': '../trained_networks/noleakyv2.pth',
                        'pixelsize': 128
                    }
                    Yp = Yeast_processor(cfg)
                    logits = Yp.detect_npc(save_fig=False, count_good_label=40, gap_closing_distance=10,
                                           threshold=0.05)
                    all_points, errors, values = Yp.refinement_npcfit_movie_new(movie=False, registration=False,
                                                                                smoothness=10, Lambda=0.001,
                                                                                length_line=length_line, estimate_prec=False,
                                                                                save_fig=False, max_signs=np.inf,
                                                                                iterations=300)
                    error_all_cells.append(errors)
                    values_all_cells.append(values)
                    count += 1
                    numcells += len(all_points)
                    print(f'Processed {count} folders, total cells: {numcells}')
            except:
                print('Error processing folder:', folder_path)

Yp.fn_dark_image = '/media/pieter/Extreme SSD/Yeast_tracking_data2023/red_dark300.tif'
Yp.fn_bright_image = '/media/pieter/Extreme SSD/Yeast_tracking_data2023/red_gain300.tif'
gain300, offset300, _ = Yp.calibrate(savefig=False)

error_array, val_array = [], []
for qq in range(len(error_all_cells)):
    for ii in range(len(error_all_cells[qq])):
        per_cell = np.array(error_all_cells[qq][ii])[0, 0, ...]
        per_cell_val = np.array(values_all_cells[qq][ii])[0, 0, ...]

        if len(error_array) == 0:
            error_array = per_cell * 1
            val_array = per_cell_val * 1
        else:
            error_array = np.concatenate((error_array, per_cell), axis=-1)
            val_array = np.concatenate((val_array, per_cell_val), axis=-1)
fit_values = val_array*1
val_array = (error_array * val_array) + val_array
val_array = val_array.T * 1
val_array_tensor = torch.tensor(val_array, dtype=torch.float32).cuda()

num_points = val_array.shape[1]
points = np.zeros_like(val_array)
for i in range(points.shape[0]):
    points[i] = np.linspace(0, length_line, num_points)
points_tensor = torch.tensor(points, dtype=torch.float32).cuda()

mean_error_linear_raw, mean_error_sigmoid_raw,vals_lin_raw,fitvals_lin_raw,vals_sig_raw,fitvals_sig_raw, valid_indices = fit_and_calculate_error(
                                    val_array_tensor, points_tensor,length_line=length_line, offset=offset300,
                                gain=gain300, fitvals=fit_values, batch_size=3)


mean_error_linear = mean_error_linear_raw[valid_indices,:]
mean_error_sigmoid = mean_error_sigmoid_raw[valid_indices,:]
vals_lin= vals_lin_raw[valid_indices,:]
fitvals_lin = fitvals_lin_raw[valid_indices,:]
vals_sig= vals_sig_raw[valid_indices,:]
fitvals_sig=fitvals_sig_raw[valid_indices,:]




# Remove outliers using 95% quantile range
def apply_threshold(array):
    lower_quantile = np.nanquantile(array, 0.001)
    upper_quantile = np.nanquantile(array, 0.999)
    return np.where((array < lower_quantile) | (array > upper_quantile), np.nan, array)

fitvals_lin_array = np.array(fitvals_lin)
vals_lin_array =  np.array(vals_lin)
fitvals_sig_array =  np.array(fitvals_sig)
vals_sig_array =  np.array(vals_sig)



######################################################################################3
# Calculate squared differences and apply gain for MSE
squared_diff_linear = ((fitvals_lin_array - vals_lin_array) * gain300) ** 2
squared_diff_sigmoid = ((fitvals_sig_array - vals_sig_array) * gain300) ** 2

# Apply threshold for MSE values
# squared_diff_linear = apply_threshold(squared_diff_linear)
# squared_diff_sigmoid = apply_threshold(squared_diff_sigmoid)

# Calculate MSE (mean) and standard deviation for linear and sigmoid models
mse_linear = np.nanmean(squared_diff_linear, axis=0)
mse_sigmoid = np.nanmean(squared_diff_sigmoid, axis=0)
std_linear = np.nanstd(apply_threshold(squared_diff_linear), axis=0)
std_sigmoid = np.nanstd(apply_threshold(squared_diff_sigmoid), axis=0)

# Logistic function model MSE
val_array_old = val_array[0:len(valid_indices),:]  * 1
error_array_old = error_array[:,0:len(valid_indices)] * 1
val_array_old = val_array_old[valid_indices]
error_array_old=error_array_old[:,valid_indices]
chi_error_nomean = (error_array_old.T * val_array_old * gain300) ** 2
#chi_error_nomean = apply_threshold(chi_error_nomean)

chi_error = np.nanmean(chi_error_nomean, axis=0)
std_chi = np.nanstd(apply_threshold(chi_error_nomean), axis=0)


# Print summaries for each model
print("Model Summaries for Average MSE:")
print(f"Linear Model: MSE + Std Dev = {np.nanmean(squared_diff_linear):.2f}, { np.nanstd(squared_diff_linear):.2f}")
print(f"sigmoid Model: MSE + Std Dev = {np.nanmean(squared_diff_sigmoid):.2f}, { np.nanstd(squared_diff_sigmoid):.2f}")

print(f"Logistic Model: MSE + Std Dev = {np.nanmean(chi_error_nomean):.2f}, { np.nanstd(chi_error_nomean):.2f}")





# Set up MSE plots with shared x-axis and colorblind-friendly colors
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, dpi=400, figsize=(2.5, 5), sharex=True,sharey=True)

# Define colorblind-friendly colors
color_linear = "#0072B2"    # Blue for Linear Model
color_sigmoid = "#009E73"   # Green for Sigmoid Model
color_logistic = "#D55E00"  # Orange for Logistic Model

# Linear model MSE plot
x_linear_clean = np.linspace(0, length_line, mse_linear.size)
ax1.plot(x_linear_clean, mse_linear, label='MSE Linear Model', color=color_linear)
ax1.fill_between(x_linear_clean, mse_linear - std_linear, mse_linear + std_linear, alpha=0.5, color="#A6CEE3", label='±1 Std Dev')  # Light blue for fill
ax1.set_ylabel(r'MSE [Photons$^2$]')
ax1.legend(loc="upper right", fontsize=8)

# Sigmoid model MSE plot
x_sigmoid_clean = np.linspace(0, length_line, mse_sigmoid.size)
ax2.plot(x_sigmoid_clean, mse_sigmoid, label='MSE Sigmoid Model', color=color_sigmoid)
ax2.fill_between(x_sigmoid_clean, mse_sigmoid - std_sigmoid, mse_sigmoid + std_sigmoid, alpha=0.5, color="#B2DF8A", label='±1 Std Dev')  # Light green for fill
ax2.set_ylabel(r'MSE [Photons$^2$]')
ax2.legend(loc="upper right", fontsize=8)

# Logistic model MSE plot
x_chi_clean = np.linspace(0, length_line, chi_error.size)
ax3.plot(x_chi_clean, chi_error, label='MSE Logistic Model', color=color_logistic)
ax3.fill_between(x_chi_clean, chi_error - std_chi, chi_error + std_chi, alpha=0.5, color="#FDBF6F", label='±1 Std Dev')  # Light orange for fill
ax3.set_ylabel(r'MSE [Photons$^2$]')
ax3.set_xlabel(r'Distance [px]')
ax3.legend(loc="upper right", fontsize=8)

# Improve layout and spacing
plt.subplots_adjust(hspace=0.4)
plt.tight_layout(pad=0.8)
plt.savefig('utils/mse_comparison_shared_x_colorblind.svg', format='svg')
plt.show()



# Set up MSE plots with shared y-axis and colorblind-friendly colors
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=400, figsize=(5, 2.5), sharey=True, sharex=True)

# Linear model MSE plot
x_linear_clean = np.linspace(0, length_line, mse_linear.size)
ax1.plot(x_linear_clean, mse_linear, label='MSE Linear Model', color=color_linear)
ax1.fill_between(x_linear_clean, mse_linear - std_linear, mse_linear + std_linear, alpha=0.5, color="#A6CEE3", label='±1 Std Dev')
ax1.set_ylabel(r'MSE [Photons$^2$]')
ax1.legend(loc="upper right", fontsize=8)
ax1.set_xlabel(r'Distance [px]')
# Sigmoid model MSE plot
x_sigmoid_clean = np.linspace(0, length_line, mse_sigmoid.size)
ax2.plot(x_sigmoid_clean, mse_sigmoid, label='MSE Sigmoid Model', color=color_sigmoid)
ax2.fill_between(x_sigmoid_clean, mse_sigmoid - std_sigmoid, mse_sigmoid + std_sigmoid, alpha=0.5, color="#B2DF8A", label='±1 Std Dev')
ax2.set_xlabel(r'Distance [px]')
ax2.legend(loc="upper right", fontsize=8)

# Logistic model MSE plot
x_chi_clean = np.linspace(0, length_line, chi_error.size)
ax3.plot(x_chi_clean, chi_error, label='MSE Logistic Model', color=color_logistic)
ax3.fill_between(x_chi_clean, chi_error - std_chi, chi_error + std_chi, alpha=0.5, color="#FDBF6F", label='±1 Std Dev')

ax3.set_xlabel(r'Distance [px]')
ax3.legend(loc="upper right", fontsize=8)

plt.subplots_adjust(wspace=0)
plt.tight_layout(pad=0.1)
plt.savefig('utils/mse_comparison_shared_y_colorblind.svg', format='svg')
plt.show()









### ensemble##############################################################

# Calculate Mean Ensemble Error and Standard Deviation for each model
mean_mean_error_linear = np.nanmean(mean_error_linear * 100, axis=0)
std_error_linear = np.nanstd(mean_error_linear * 100, axis=0)

mean_mean_error_sigmoid = np.nanmean(mean_error_sigmoid * 100, axis=0)
std_error_sigmoid = np.nanstd(mean_error_sigmoid * 100, axis=0)

mean_mean_error_logistic = np.nanmean(-error_array * 100, axis=1)
std_error_logistic = np.nanstd(-error_array * 100, axis=1)


# Set up Mean Ensemble Error plots with shared x-axis and y-axis, arranged vertically
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, dpi=400, figsize=(2.5, 5), sharex=True, sharey=True)

# Define colorblind-friendly colors
color_linear = "#0072B2"    # Blue for Linear Model
color_sigmoid = "#009E73"   # Green for Sigmoid Model
color_logistic = "#D55E00"  # Orange for Logistic Model

# Linear model Mean Ensemble Error plot
x_linear_clean = np.linspace(0, length_line, mean_mean_error_linear.size)
ax1.plot(x_linear_clean, mean_mean_error_linear, label='Linear Model', color=color_linear)
ax1.fill_between(x_linear_clean, mean_mean_error_linear - std_error_linear, mean_mean_error_linear + std_error_linear,
                 alpha=0.5, color="#A6CEE3", label='±1 Std Dev')  # Light blue fill for Linear Model
ax1.set_ylabel(r'Mean Error $E$ [\%]')
ax1.legend(loc="upper right", fontsize=8)

# Sigmoid model Mean Ensemble Error plot
x_sigmoid_clean = np.linspace(0, length_line, mean_mean_error_sigmoid.size)
ax2.plot(x_sigmoid_clean, mean_mean_error_sigmoid, label='Sigmoid Model', color=color_sigmoid)
ax2.fill_between(x_sigmoid_clean, mean_mean_error_sigmoid - std_error_sigmoid, mean_mean_error_sigmoid + std_error_sigmoid,
                 alpha=0.5, color="#B2DF8A", label='±1 Std Dev')  # Light green fill for Sigmoid Model
ax2.set_ylabel(r'Mean Error $E$ [\%]')
ax2.legend(loc="upper right", fontsize=8)

# Logistic model Mean Ensemble Error plot
x_chi_clean = np.linspace(0, length_line, mean_mean_error_logistic.size)
ax3.plot(x_chi_clean, mean_mean_error_logistic, label='Logistic Model', color=color_logistic)
ax3.fill_between(x_chi_clean, mean_mean_error_logistic - std_error_logistic, mean_mean_error_logistic + std_error_logistic,
                 alpha=0.5, color="#FDBF6F", label='±1 Std Dev')  # Light orange fill for Logistic Model
ax3.set_ylabel(r'Mean Error $E$ [\%]')
ax3.set_xlabel(r'Distance [px]')
ax3.legend(loc="upper right", fontsize=8)

# Improve layout and spacing
plt.subplots_adjust(hspace=0.4)
plt.tight_layout(pad=0.2)
plt.savefig('utils/mean_ensemble_error_comparison_shared_xy_colorblind.svg', format='svg')
plt.show()




# Set up Mean Ensemble Error plots with shared x-axis and y-axis, arranged horizontally
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=400, figsize=(5, 2.5), sharex=True, sharey=True)

# Define colorblind-friendly colors
color_linear = "#0072B2"  # A shade of blue
color_sigmoid = "#009E73"  # A shade of green
color_logistic = "#D55E00"  # A shade of orange

# Linear model Mean Ensemble Error plot
x_linear_clean = np.linspace(0, length_line, mean_mean_error_linear.size)
ax1.plot(x_linear_clean, mean_mean_error_linear, label='Linear Model', color=color_linear)
ax1.fill_between(x_linear_clean, mean_mean_error_linear - std_error_linear, mean_mean_error_linear + std_error_linear,
                 alpha=0.5, color="#A6CEE3", label='±1 Std Dev')  # Lighter blue for fill
ax1.set_ylabel(r'Mean Error $E$ [\%]')
ax1.set_xlabel(r'Distance [px]')
##ax1.set_title("Linear Model")
ax1.legend(loc="upper right", fontsize=8)

# Sigmoid model Mean Ensemble Error plot
x_sigmoid_clean = np.linspace(0, length_line, mean_mean_error_sigmoid.size)
ax2.plot(x_sigmoid_clean, mean_mean_error_sigmoid, label='Sigmoid Model', color=color_sigmoid)
ax2.fill_between(x_sigmoid_clean, mean_mean_error_sigmoid - std_error_sigmoid, mean_mean_error_sigmoid + std_error_sigmoid,
                 alpha=0.5, color="#B2DF8A", label='±1 Std Dev')  # Lighter green for fill
ax2.set_xlabel(r'Distance [px]')
#ax2.set_title("Sigmoid Model")
ax2.legend(loc="upper right", fontsize=8)

# Logistic model Mean Ensemble Error plot
x_chi_clean = np.linspace(0, length_line, mean_mean_error_logistic.size)
ax3.plot(x_chi_clean, mean_mean_error_logistic, label='Logistic Model', color=color_logistic)
ax3.fill_between(x_chi_clean, mean_mean_error_logistic - std_error_logistic, mean_mean_error_logistic + std_error_logistic,
                 alpha=0.5, color="#FDBF6F", label='±1 Std Dev')  # Lighter orange for fill
ax3.set_xlabel(r'Distance [px]')
#ax3.set_title("Logistic Model")
ax3.legend(loc="upper right", fontsize=8)

# Adjust layout for better spacing and alignment
plt.tight_layout(w_pad=0.1)  # Increased padding between plots
plt.savefig('utils/mean_ensemble_error_comparison_shared_yx_colorblind.svg', format='svg')
plt.show()
