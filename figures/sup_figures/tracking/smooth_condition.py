"""
Title: Simulating Mixed Motion Trajectories and Evaluating Spline Smoothing Effect on Localization Error

Description:

This script simulates particle trajectories with mixed motion types (diffusive and directed segments), adds localization error to mimic measurement noise, and evaluates the effect of spline smoothing on the estimation accuracy of the particle's path. The key steps include:

1. **Trajectory Simulation**:
   - Generates synthetic 2D trajectories consisting of a mixture of diffusive and directed motion segments.
   - Each segment randomly decides between diffusive or directed motion.
   - Adds Gaussian localization error to simulate measurement noise.

2. **Spline Fitting and Error Calculation**:
   - Fits univariate splines to the noisy localization data for varying smoothing factors.
   - Calculates the Mean Squared Error (MSE) between the spline-fitted points and the ground truth trajectory.
   - Compares the MSE of the spline-fitted trajectory with the MSE of the raw noisy localizations.

3. **Visualization**:
   - Plots the average MSE as a function of the smoothing factor for both spline-smoothed and non-smoothed data.
   - Visualizes an example trajectory, including the noisy localizations, ground truth, and spline fit.
   - Uses color coding to represent the velocity along the spline-fitted trajectory.

Dependencies:

- numpy
- matplotlib
- scienceplots
- scipy
- tqdm

Usage:

- Adjust parameters such as `n_traces`, `smooth_factors`, and `loc_error` as needed.
- Run the script to generate the MSE plot and example trajectory visualization.
- The script saves the plots as SVG files (`MSE_PLOT.svg` and `example_traj.svg`).

Notes:

- The localization error (`loc_error`) is specified in nanometers and converted to pixels within the script.
- Random seeds are set for reproducibility.
- The `generate_mixed_motion` function is the core function for simulating trajectories.

"""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy.interpolate import UnivariateSpline
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


from scipy.spatial.distance import cdist
from tqdm import tqdm

plt.style.use('science')

def generate_mixed_motion(n_segments=5, max_frames_per_segment=7, loc_error=40/128):
    """
    Generates a trace with mixed motion, including both directed and diffusive segments.

    Parameters:
    - n_segments: Number of segments to generate.
    - max_frames_per_segment: Maximum number of frames in each segment.
    - loc_error: Localization error to simulate measurement imprecision.

    Returns:
    - A tuple containing the complete path, the frame numbers, and the motion type for each segment.
    """
    #np.random.seed(42)  # For reproducibility
    path = [np.array([0, 0])]  # Starting point
    motion_types = []
    frames = [0]  # Initialize frame numbers starting from 0

    for segment in range(n_segments):
        n_frames = np.random.randint(1, max_frames_per_segment + 1)
        motion_type = np.random.choice(['diffusion', 'directed'])

        if motion_type == 'diffusion':
            step_size = np.random.uniform(0.00, 1/np.sqrt(2))  # Random step size
            angles = np.random.uniform(low=0, high=2*np.pi, size=n_frames)
            steps = np.vstack((np.cos(angles), np.sin(angles))).T * step_size
        else:  # directed
            speed = np.random.uniform(0.00, 1/np.sqrt(2))  # Random speed
            direction = np.random.uniform(low=0, high=2*np.pi)  # Random direction
            dx = np.cos(direction) * speed/len(frames)
            dy = np.sin(direction) * speed/len(frames)
            steps = np.array([dx, dy]) * np.arange(1, n_frames + 1).reshape(-1, 1)

        segment_path = np.cumsum(steps, axis=0)
        segment_path += path[-1]  # Start from the last point of the previous segment
        path.extend(segment_path)
        motion_types.append((motion_type, n_frames))
        # Update frames for the current segment
        frames.extend([frames[-1] + i + 1 for i in range(n_frames)])

    # Adding localization error
    ground_truth = np.array(path)
    path = np.array(path) + np.random.normal(0, loc_error/np.sqrt(2), size=np.array(path).shape)

    frames = np.array(frames)  # Convert frame numbers to a numpy array

    return path, frames, motion_types, ground_truth

plot = True
# Define ranges and storage
smooth_factors = np.linspace(0, 7, 40)  # Adjust as needed
n_traces = 20000 # Number of traces to generate for each smooth factor
# smooth_factors = [1.6]
# n_traces = 1
mse_errors_spline = np.zeros((len(smooth_factors), n_traces))
mse_errors_locs = np.zeros((len(smooth_factors), n_traces))
loc_error = 43
np.random.seed(29)
# Loop over smooth factors and traces
for i, smoothfactor in tqdm(enumerate(smooth_factors)):
    for j in range(n_traces):
        # Generate a new trace
        localizations, framenum, motion_types, ground_truth = \
            generate_mixed_motion(n_segments=5,
                                max_frames_per_segment=10, loc_error=loc_error/128)
        # Fit splines with current smooth factor
        spline_x = UnivariateSpline(framenum, localizations[:, 0], s=smoothfactor)
        spline_y = UnivariateSpline(framenum, localizations[:, 1], s=smoothfactor)

        # Calculate spline points
        spline_points = np.vstack((spline_x(framenum), spline_y(framenum))).T
        # Calculate MSE for spline and original localizations
        differences = spline_points - ground_truth[:, :2]
        differences_nospline = localizations - ground_truth[:, :2]
        mse_errors_spline[i, j] = np.mean(np.sum(differences ** 2, axis=1))
        mse_errors_locs[i, j] = np.mean(np.sum(differences_nospline ** 2, axis=1))

# Average MSE across traces for each smooth factor
avg_mse_spline = np.mean(mse_errors_spline, axis=1)
avg_mse_locs = np.mean(mse_errors_locs, axis=1)

# Plotting
plt.figure(figsize=(2.5, 2.5))
plt.plot(smooth_factors, avg_mse_spline*128, label='Spline smoothed')
plt.plot(smooth_factors, avg_mse_locs*128, label='Non-smoothed', linestyle='--')
plt.xlabel('Smooth Factor')
plt.ylabel('MSE [nm$^2$]')
plt.title(r'$\sigma_\text{RNA}$= ' + str(loc_error) + 'nm')
plt.legend()
plt.tight_layout()
plt.savefig('MSE_PLOT.svg', format='svg')
plt.show()
if plot ==True:
    # Prepare the figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), sharey=True)  # This will share the y-axis between the two subplots

    # First subplot: Original and Ground Truth
    axs[0].plot(localizations[:,0], localizations[:,1], 'ko-'
               , markersize=3)
    axs[0].plot(ground_truth[:,0], ground_truth[:,1], 'o-', label='Ground truth', markersize=3)
    axs[0].legend()
    axs[0].set_xlabel(r'$x$ [pixels]')
    axs[0].set_ylabel(r'$y$ [pixels]')
    axs[0].axis('equal')
    # second subplot: Velocity Color Coding
    upsample_factor = 100
    framenum_up = np.linspace(min(framenum),max(framenum), len(framenum)*upsample_factor)
    v_x = spline_x.derivative()(framenum_up)
    v_y = spline_y.derivative()(framenum_up)
    velocity = np.sqrt(v_x**2 + v_y**2)

    norm = Normalize(vmin=velocity.min(), vmax=velocity.max())
    cmap = plt.get_cmap('viridis')
    sm = ScalarMappable(norm=norm, cmap=cmap)

    sc = axs[1].scatter(spline_x(framenum_up), spline_y(framenum_up), c=velocity, cmap=cmap, alpha=0.7, label='Spline',s=3)

    fig.colorbar(sm, ax=axs[1], label='Velocity (px/frame)')
    axs[1].plot(localizations[:,0], localizations[:,1], 'ko-', label=r'Including $\sigma_\text{loc}$', markersize=3)
    axs[1].set_xlabel(r'$x$ [pixels]')
    #axs[1].set_ylabel('Y [pixels]')
    #axs[1].set_title('Spline Fit with Velocity Color Coding')
    axs[1].axis('equal')
    axs[1].legend()

    # # Second subplot: Spline and Ground Truth
    # axs[2].plot(spline_points[:, 0], spline_points[:, 1], label='Spline fit')
    # axs[2].plot(ground_truth[:,0], ground_truth[:,1], 'o-', label='Ground truth')
    # axs[2].legend()
    # axs[2].set_xlabel('X [pixels]')
    # axs[2].set_ylabel('Y [pixels]')
    # axs[2].set_title('Spline Fit vs. Ground Truth')
    # axs[2].axis('equal')
    #
    #
    # # Update the title with the measure of MSE
    #
    # axs[2].set_title(f'Spline Fit vs. Ground Truth \n MSE: {mse_errors_spline[-1,-1]:.2f} $pixels^2$\n'
    #                  f'Locs vs. Ground Truth \n MSE: {mse_errors_locs[-1,-1]:.2f} $pixels^2$')


    plt.tight_layout()
    plt.savefig('example_traj.svg', format='svg')
    plt.show()

