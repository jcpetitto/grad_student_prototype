"""
Title: Yeast Tracking Data Analysis Script for Transport Events

Description:
This script analyzes yeast tracking data to study transport events, localization probabilities, and velocities of particles within yeast cells. It processes the tracking data, identifies transport events based on specified thresholds, and generates various plots to visualize the results. The script also fits models to the data to extract parameters related to binding sites and confinement effects.

Functions:
- load_and_process_tracks(cfg): Loads and processes tracking data from CSV files. It modifies specified columns by swapping values to correct the data format.
- load_other_data(cfg): Loads additional data required for the analysis, including file lookups and nuclear envelope (NE) data.
- process_for_config(cfg, make_velocity=True, succes_threshold=[120, 120]): Processes data for a given configuration, performs analysis, and returns a dictionary containing the results.

Main Workflow:
1. Sets up the base configuration.
2. Defines functions for data loading and processing.
3. Iterates over a list of digit identifiers and corresponding threshold values to process data for each configuration.
4. Processes the data using the `process_for_config` function.
5. Collects results from each configuration.
6. Generates plots for:
   - Projected velocities.
   - Localization probabilities.
   - Binding site distributions with Gaussian deconvolution.
   - Combined velocities and ratios.
   - Fitting models to velocity ratios to study confinement effects.
7. Saves the generated plots as SVG files.

Dependencies:
- numpy
- pandas
- matplotlib
- seaborn
- pickle
- scipy
- extract_results (custom module)
- scienceplots
- cycler

Usage:
- Ensure that the required data files are located in the paths specified in the configuration.
- Adjust the `digits_list` and `threshold_list` as needed for your data.
- Run the script to perform the analysis and generate the plots.

Notes:
- The script uses a custom module `yeast_extractresults` which should be available in the Python path.
- The plotting style is set to 'science' using the `scienceplots` package.
- The script is tailored to specific datasets and configurations; modifications may be required for different datasets.
- Measurement sigma and initial parameters for curve fitting are set based on assumptions; adjust as necessary.
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import stats
from matplotlib.animation import FuncAnimation
from extract_results import yeast_extractresults
import scienceplots
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import splprep, splev, UnivariateSpline
from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.signal import convolve
# Apply a scientific plotting style
plt.style.use('science')

# Base configuration
base_cfg = {
    'digits': '',
    'mainpath': '/media/pieter/Extreme SSD/Yeast_tracking_data2023/Simulation',
    'resultdir': "v3GLRT_0_95",
    'trackdata': 'tracks.data',
    'pixelsize': 128,
    'moviename': '/media/pieter/Extreme SSD/Yeast_tracking_data2023/',
    'frametime': 0.02,
    'columns_to_swap': ['may_inout', 'def_inout', 'may_inout_spline', 'def_inout_spline']
}

# Function definitions (load_and_process_tracks, load_other_data, calculate_velocity_splines, etc.)
def load_and_process_tracks(cfg):
    all_tracks = pd.read_csv(cfg['all_tracks_path'])
    for col in cfg['columns_to_swap']:
        all_tracks.loc[(all_tracks[col] == 0) & (all_tracks[col].notna()), col] = 2
        all_tracks.loc[(all_tracks[col] == 1) & (all_tracks[col].notna()), col] = 0
        all_tracks.loc[(all_tracks[col] == 2) & (all_tracks[col].notna()), col] = 1
    return all_tracks


def load_other_data(cfg):
    file_lookup = pd.read_csv(cfg['file_lookup_path'])
    ne_lookup = pd.read_csv(cfg['ne_lookup_path'])
    with open(cfg['nedata_path'], 'rb') as file:
        ne_data = pickle.load(file)
    return file_lookup, ne_lookup, ne_data

def process_for_config(cfg, make_velocity = True,succes_threshold=[120,120]):

    # Load and process data
    max_number_intersections = 3
    all_tracks = load_and_process_tracks(cfg)
    file_lookup, ne_lookup, ne_data = load_other_data(cfg)

    # Initialize yeast extract results object
    yr = yeast_extractresults(cfg)
    yr.all_tracks = all_tracks
    yr.lookup = file_lookup
    yr.nelookup = ne_lookup
    yr.nedata = ne_data

    # Perform your analysis and plotting here
    # For example:
    transport_events_gfa, success_export_df, unsuccess_export_df, success_import_df, unsuccess_import_df, retrogate_export_df, retrogate_import_df,_ \
        = yr.find_transports_v2(thressuccess_transport=succes_threshold, thres_transport=succes_threshold,
                                max_number_intersections=max_number_intersections, use_spline=True)
    print('number of transport traces: ', len(np.unique(transport_events_gfa['id'])))
    # Continue with your existing analysis and plotting...
    distance_total,num_bins,x_fit, yfit = yr.plot_diffusion_vs_distance([transport_events_gfa,], xlim=[-250, 250],
                                  num_bins=20, label_list=[cfg['digits']],ylim=[0,0.0035])
    sample_distances = np.arange(-250, 250, 4)
    if make_velocity:
        mean_velocities, mean_velocities_proj,bin_centers_velocity = yr.plot_velocity([transport_events_gfa],
                                                                  sample_distances=sample_distances, upsample_factor=100, name_list=[cfg['digits']])
    else:
        mean_velocities, mean_velocities_proj, bin_centers_velocity = [],[],[]

    results = {
        'transport_events_gfa': transport_events_gfa,
        'success_export_df': success_export_df,
        'unsuccess_export_df': unsuccess_export_df,
        'success_import_df': success_import_df,
        'unsuccess_import_df': unsuccess_import_df,
        'retrogate_export_df': retrogate_export_df,
        'retrogate_import_df': retrogate_import_df,
        'distance_total': distance_total,
        'num_bins': num_bins,
        'x_fit': x_fit,
        'y_fit': yfit,
        'mean_velocities': mean_velocities,
        'mean_velocities_proj': mean_velocities_proj,
        'bin_centers_velocity': bin_centers_velocity,
        # Add other variables as needed
    }
    return results
# List of digit sets to loop over
digits_list = [823, 820,822]
threshold_list = [[-76.24, 94.76], [-89.75,78.61], [-69.74,77.45]]
# digits_list = [823]
# threshold_list = [[-49, 49]]
all_results = []
for num, digits in enumerate(digits_list):
    # Update cfg paths for the current set of digits
    cfg = base_cfg.copy()  # Start with the base configuration
    cfg.update({
        'digits': str(digits),
        'all_tracks_path': f'../../results_tracks_gapclosing10/all_tracks_{digits}.csv',
        'ne_lookup_path': f'../../results_tracks_gapclosing10/ne_lookup{digits}.csv',
        'file_lookup_path': f'../../results_tracks_gapclosing10/file_lookup_{digits}.csv',
        'nedata_path': f'../../results_tracks_gapclosing10/nedata{digits}.pkl',
    })

    # Process data for this configuration
    result= process_for_config(cfg, make_velocity = True,succes_threshold=threshold_list[num])
    all_results.append(result)
label = ['GFA1', 'MYO2', 'TRA1']
CB_color_cycle = ['#0173b2',
 '#de8f05',
 '#029e73',
 '#d55e00',
 '#cc78bc',
 '#ca9161',
 '#fbafe4',
 '#949494',
 '#ece133',
 '#56b4e9']
# Set the color cycle for Matplotlib
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=CB_color_cycle)

from cycler import cycler
from scipy.interpolate import splrep, BSpline



custom_color_cycle = [CB_color_cycle[i] for i in [0, 0, 1, 1,  2, 2]]

fig, ax = plt.subplots(dpi=400, figsize=(3, 3))
ax.set_prop_cycle(cycler('color', custom_color_cycle))
label = ['GFA1', 'MYO2', 'TRA1']
for i, result in enumerate(all_results):
    x = result['bin_centers_velocity'] * cfg['pixelsize']
    y = result['mean_velocities_proj']
    # Plot mean velocities and their projections
    ax.plot(x,y ,linewidth=1,alpha=0.6)
    tck_s = splrep(x, y, s=0.4)
    X_ = np.linspace(x.min(), x.max(), 500)
    Y_ = BSpline(*tck_s)(X_)
    ax.plot(X_,Y_, linewidth=2, label=label[i])

ax.set_xlabel('Distance to membrane [nm]')
ax.set_ylabel(r'Projected velocity [$\mu$m/s] ')
ax.set_xlim(-250,250)
plt.legend()
plt.tight_layout()
plt.savefig('graphs/projected_velocity_transport.svg', format='svg')
plt.show()


fig, ax = plt.subplots(dpi=400, figsize=(2.3, 1.5))
fig2, ax2 = plt.subplots(dpi=400, figsize=(2.3, 1.5))

ax2.set_prop_cycle(cycler('color', custom_color_cycle))
custom_color_cycle = [CB_color_cycle[i] for i in [0, 1,  2]]
ax.set_prop_cycle(cycler('color', custom_color_cycle))
label = ['GFA', 'MYO', 'TRA']
for i, result in enumerate(all_results):


    # Plot mean velocities and their projections
    ax.plot(result['x_fit'], result['y_fit'], label=label[i], linestyle='-',linewidth=1)
    ax2.plot(result['x_fit'], result['y_fit'], label=label[i], linestyle='-',linewidth=1)
    ax2.hist(result['distance_total'], bins = result['num_bins'],range=[-250,250], density=True, alpha=0.3)
ax.set_xlabel('Distance to membrane [nm]')
ax.set_ylabel(r'Prob. of loc.')
ax2.set_xlabel('Distance to membrane [nm]')
ax2.set_ylabel(r'Prob. of loc.')
fig.legend()

fig.tight_layout(pad=0.2)
fig.savefig('graphs/localization_probabillity_transport.svg',format='svg')
fig.show()
fig2.tight_layout(pad=0.2)
fig2.legend()
fig.savefig('graphs/localization_probabillity_transport_includingbins.svg',format='svg')
fig2.show()


# Modify the double Gaussian model to include sigma directly
def double_gaussian_model(x, width1, center1, scale1, width2, center2, scale2, offset, sigma):
    return (scale1 * np.exp(-(x - center1) ** 2 / (2 * (width1 ** 2 + sigma ** 2))) +
            scale2 * np.exp(-(x - center2) ** 2 / (2 * (width2 ** 2 + sigma ** 2))) +
            offset)

# Define a wrapper function for curve_fit that includes sigma as a fixed parameter
def fit_model(x, width1, center1, scale1, width2, center2, scale2, offset):
    return double_gaussian_model(x, width1, center1, scale1, width2, center2, scale2, offset, measurement_sigma)

# Sample data and fitting
# Assuming all_results and CB_color_cycle are properly defined
labels = ['GFA1', 'MYO2', 'TRA1']
table_data = []
for i, result in enumerate(all_results):
    # Histogram data for each strain
    hist, bin_edges = np.histogram(result['distance_total'], bins=30, range=(-250, 250))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit the model using curve_fit
    measurement_sigma = 49 / np.sqrt(2)  # Your measurement sigma
    initial_params = [20, -80, 80000, 20, 80, 80000, 1000]  # Adjust these initial params as needed
    popt, pcov = curve_fit(fit_model, bin_centers, hist, p0=initial_params)
    perr = np.sqrt(np.diag(pcov))  # Standard errors of the parameters

    # Round parameter values and ceil errors
    popt_rounded = np.round(popt).astype(int)
    perr_ceiled = np.ceil(perr).astype(int)

    # Plotting for each strain
    x_plot = np.linspace(-250, 250, 1000)
    plt.figure(figsize=(3, 1.5))
    plt.bar(bin_centers, hist, width=np.diff(bin_edges)[0], alpha=0.6, color=CB_color_cycle[i], label='data ' + labels[i])
    plt.plot(x_plot, double_gaussian_model(x_plot, *popt, 0), 'k--', label='binding sites')
    plt.plot(x_plot, double_gaussian_model(x_plot, *popt, measurement_sigma), 'k:', label=r'$\sigma_\text{binding}$ + $\sigma_\text{CoLoc}$ ')
    plt.xlabel('Distance to membrane [nm]')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'graphs/binding_sites_gauss_decon_{labels[i]}.svg', format='svg')
    plt.show()

    print(f"{labels[i]} Optimized parameters and their errors:")
    print(f"Width1 (sigma) = {popt[0]:.2f} ± {perr[0]:.2f}")
    print(f"Center1 = {popt[1]:.2f} ± {perr[1]:.2f}")
    print(f"Scale1 = {popt[2]:.2f} ± {perr[2]:.2f}")
    print(f"Width2 (sigma) = {popt[3]:.2f} ± {perr[3]:.2f}")
    print(f"Center2 = {popt[4]:.2f} ± {perr[4]:.2f}")
    print(f"Scale2 = {popt[5]:.2f} ± {perr[5]:.2f}")
    print(f"Offset = {popt[6]:.2f} ± {perr[6]:.2f}")

    # Collect parameters for the table
    table_data.append({
        'Label': labels[i],
        'Mean1': f"{popt_rounded[1]} ± {perr_ceiled[1]}",
        'Width1': f"{popt_rounded[0]} ± {perr_ceiled[0]}",
        'Mean2': f"{popt_rounded[4]} ± {perr_ceiled[4]}",
        'Width2': f"{popt_rounded[3]} ± {perr_ceiled[3]}"
    })

# Create a DataFrame for the table
df_table = pd.DataFrame(table_data)

# Display the table
print("\nFitted Parameters Table:")
print(df_table)

##############
custom_color_cycle = [CB_color_cycle[i] for i in [0, 0,0,0, 1, 1, 1, 1,  2, 2,  2, 2]]

fig, ax = plt.subplots(dpi=400, figsize=(1.9, 1.8))
ax.set_prop_cycle(cycler('color', custom_color_cycle))
label = ['GFA1', 'MYO2', 'TRA1']
for i, result in enumerate(all_results):
    x = result['bin_centers_velocity'] * cfg['pixelsize']
    y = result['mean_velocities_proj']
    # Plot mean velocities and their projections
    ax.plot(x,y ,linewidth=1,alpha=0.6)
    tck_s = splrep(x, y, s=0.4)
    X_ = np.linspace(x.min(), x.max(), 500)
    Y_ = BSpline(*tck_s)(X_)
    ax.plot(X_,Y_, linewidth=2,linestyle='--')

    x = result['bin_centers_velocity'] * cfg['pixelsize']
    y = result['mean_velocities']
    # Plot mean velocities and their projections
    ax.plot(x,y ,linewidth=1,alpha=0.6)
    tck_s = splrep(x, y, s=0.4)
    X_ = np.linspace(x.min(), x.max(), 500)
    Y_ = BSpline(*tck_s)(X_)
    ax.plot(X_,Y_, linewidth=2, label=label[i])


ax.set_xlabel('Dist. to membrane [nm]')
ax.set_ylabel(r'Velocity [$\mu$m/s] ')
ax.set_xlim(-250,250)
plt.legend()
plt.tight_layout(pad=0.3)
plt.savefig('graphs/combined_velocity_transport.svg', format='svg')
plt.show()

# Assuming `all_results` and `cfg` have the necessary data structured correctly
fig, ax = plt.subplots(dpi=400, figsize=(2, 1.4))
custom_color_cycle = [CB_color_cycle[i] for i in [0, 1, 2]]  # Ensure CB_color_cycle is defined
ax.set_prop_cycle(cycler('color', custom_color_cycle))
labels = ['GFA1', 'MYO2', 'TRA1']

for i, result in enumerate(all_results):
    # Extract data directly for plotting
    x_proj = result['bin_centers_velocity'] * cfg['pixelsize']
    y_proj = result['mean_velocities_proj']
    x = result['bin_centers_velocity'] * cfg['pixelsize']
    y = result['mean_velocities']

    # Compute the ratio of velocities directly
    ratio = y / y_proj  # Ensure both y and y_proj are the same length and correspond correctly

    # Plot the ratio
    ax.plot(x_proj, ratio, linewidth=1, label=labels[i])

# Plot the horizontal line at sqrt(2)
sqrt_2 = np.sqrt(2)
ax.axhline(y=sqrt_2, color='grey', linestyle='--', label='Free motion')

ax.set_xlabel('Dist. to membrane [nm]')
ax.set_ylabel(r'Ratio $v/v_\text{proj.}$')
ax.set_xlim(-250, 250)

# Place the legend above the plot in a grid of 2 by 2
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, shadow=True)
ax.legend()
plt.tight_layout(pad = 0.1)
plt.savefig('graphs/ratio_velocities.svg', format='svg')
plt.show()


# Define the combined model using double sigmoid and Gaussian convolution
def combined_model(x, width, center, scale, baseline):
    """
    Generates a 'double sigmoid block pulse', convolves it with a Gaussian,
    and applies scaling and baseline adjustment to model an inverted signal.
    """
    sigma = 49 / 1.41  # Standard deviation of the Gaussian
    rising_edge = 1 / (1 + np.exp(-(x - (center - width / 2)) ))
    falling_edge = 1 / (1 + np.exp((x - (center + width / 2)) ))
    pulse = rising_edge + falling_edge -1
    # Gaussian function
    gauss = np.exp(-(x - center) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    # Convolution of double sigmoid pulse with Gaussian
    convolved = convolve(pulse, gauss, mode='same', method='auto') / np.sum(gauss)

    # Apply inversion and baseline adjustment
    return -scale * convolved + baseline


# Define a simplified pulse model for comparison
def pulse_model(x, width, center, scale, baseline):
    """
    Generates a double sigmoid pulse and applies scaling and baseline adjustment.
    """

    rising_edge = 1 / (1 + np.exp(-(x - (center - width / 2)) ))
    falling_edge = 1 / (1 + np.exp((x - (center + width / 2)) ))
    pulse = rising_edge + falling_edge -1

    return -scale * pulse + baseline




for i, result in enumerate(all_results):
    x_proj = result['bin_centers_velocity'] * cfg['pixelsize']
    y = result['mean_velocities'] / result['mean_velocities_proj']
    initial_params = [100, 0, 1, 0]  # width, center, scale, baseline

    # Perform curve fitting
    popt, pcov = curve_fit(combined_model, x_proj, y, p0=initial_params)
    perr = np.sqrt(np.diag(pcov))  # Calculate the standard errors from the diagonal of the covariance matrix

    # Plotting the results
    xplot = np.linspace(min(x_proj), max(x_proj), 1000)
    # Assuming `all_results` and `cfg` have the necessary data structured correctly
    fig, ax = plt.subplots(dpi=400, figsize=(3, 1.5))
    custom_color_cycle = [CB_color_cycle[i] for i in [0, 1, 2]]  # Ensure CB_color_cycle is defined
    ax.set_prop_cycle(cycler('color', custom_color_cycle))
    labels = ['GFA1', 'MYO2', 'TRA1']
    plt.plot(x_proj, y, color=CB_color_cycle[i], label=labels[i])
    plt.plot(xplot, pulse_model(xplot, *popt), 'k--', linewidth=1, label='Confined')
    plt.plot(xplot, combined_model(xplot, *popt), 'k:', linewidth=1.5, label=r'Confined + $\sigma_\text{CoLoc}$')
    plt.xlabel('Distance to membrane [nm]')
    plt.ylabel(r'Ratio $v/v_\text{proj.}$')
    plt.legend()
    plt.tight_layout(pad=0.1)
    plt.savefig('graphs/ratio_and_fit'+labels[i]+'.svg', format='svg')
    plt.show()

    print("Optimized parameters and their errors:")
    print(f"Width: {popt[0]:.2f} ± {perr[0]:.2f}")
    print(f"Center: {popt[1]:.2f} ± {perr[1]:.2f}")
    print(f"Scale: {popt[2]:.2f} ± {perr[2]:.2f}")
    print(f"Baseline: {popt[3]:.2f} ± {perr[3]:.2f}")

# Assuming `all_results` and `cfg` have the necessary data structured correctly
fig, axes = plt.subplots(len(all_results), 1, dpi=400, figsize=(2.3, 2.6), sharex=True)
custom_color_cycle = [CB_color_cycle[i] for i in [0, 1, 2]]  # Ensure CB_color_cycle is defined
labels = ['GFA1', 'MYO2', 'TRA1']

if len(all_results) == 1:  # In case there's only one result, matplotlib won't make axes an array
    axes = [axes]

for i, result in enumerate(all_results):
    ax = axes[i]
    ax.set_prop_cycle(cycler('color', custom_color_cycle))

    x_proj = result['bin_centers_velocity'] * cfg['pixelsize']
    y = result['mean_velocities'] / result['mean_velocities_proj']
    initial_params = [100, 0, 1, 0]  # width, center, scale, baseline

    # Perform curve fitting
    popt, pcov = curve_fit(combined_model, x_proj, y, p0=initial_params)
    perr = np.sqrt(np.diag(pcov))  # Calculate the standard errors from the diagonal of the covariance matrix

    # Plotting the results
    xplot = np.linspace(min(x_proj), max(x_proj), 1000)
    ax.plot(x_proj, y, color=CB_color_cycle[i], label=f'{labels[i]}')
    ax.plot(xplot, pulse_model(xplot, *popt), 'k--', linewidth=1, label='Confined')
    ax.plot(xplot, combined_model(xplot, *popt), 'k:', linewidth=1.5, label=r'Confined + $\sigma_\text{CoLoc}$')
    if i == 1:
        ax.set_ylabel(r'Ratio $v/v_\text{proj.}$')
    ax.legend()
    print(f"{labels[i]} Optimized parameters and their errors:")
    print(f"Width: {popt[0]:.2f} ± {perr[0]:.2f}")
    print(f"Center: {popt[1]:.2f} ± {perr[1]:.2f}")
    print(f"Scale: {popt[2]:.2f} ± {perr[2]:.2f}")
    print(f"Baseline: {popt[3]:.2f} ± {perr[3]:.2f}")

# Setting common labels and adjusting layout
axes[-1].set_xlabel('Distance to membrane [nm]')  # Only set x-label on the last subplot
plt.tight_layout(pad=0.1)
plt.savefig('graphs/ratio_and_fit_overall.svg', format='svg')
plt.show()

