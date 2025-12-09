#!/usr/bin/env python
"""
Script to process yeast data folders and perform analysis using the Yeast_processor class.

This script processes yeast data in the specified folder by performing calibration, registration,
NPC detection, MLE fitting, and tracking. The results are saved for further analysis.

Usage:
    python script_name.py /path/to/data_folder

The script can be run from the command line or imported as a module.
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.Yeast_processor import Yeast_processor
import tqdm
import torch
import warnings

# Suppress all warnings (optional)
warnings.filterwarnings("ignore")

# Configuration parameters (customize as needed)
cfg = {
    'path': '',  # Will be set per folder
    'roisize': 16,                      # ROI size for fitting PSFs
    'sigma': 0.92,                      # PSF width for mRNA (in pixels)
    'frames': [0, 1000],                # Frames to consider for processing
    'frames_npcfit': [0, 250],          # Frames for NE fitting
    'drift_bins': 4,                    # Number of drift bins
    'resultdir': "/results/",           # Directory to save results
    'gain': '/media/pieter/Extreme SSD/Yeast_tracking_data2023/bright_images_green_channel_20ms_300EM.tiff',  # Path to bright images (gain) for mRNA channel
    'offset': '/media/pieter/Extreme SSD/Yeast_tracking_data2023/dark_images_green_channel_20ms_300EM.tiff',  # Path to dark images (offset) for mRNA channel
    'model_NE': '/home/pieter/development/yeast_mrna_tracking/trained_networks/Modelweights_NE_segmentation.pt',  # Path to trained weights for NE segmentation NN
    'model_bg': '/home/pieter/development/yeast_mrna_tracking/trained_networks/model_wieghts_background_psf.pth',  # Path to trained weights for background estimation
    'pixelsize': 128  # Pixel size of the camera in nm
}

# Threshold for registration difference (in pixels)
threshold_regist = 0.5

# Flags to save figures and movies
save_figures = False
save_movies = False

# Bounds and initial guess for GLRT detector
bounds_glrt = [
    [0, cfg['roisize'] - 1],  # x bounds
    [0, cfg['roisize']],      # y bounds
    [0, 1e9],                 # Photons bounds
    [0, 1e6]                  # Background bounds
]

initial_guess = [
    cfg['roisize'] / 2,       # x initial guess
    cfg['roisize'] / 2,       # y initial guess
    0,                        # Photons initial guess
    60                        # Background initial guess
]

def extract_digits(folder_name):
    """
    Extract digits from a folder name using regular expressions.

    Parameters:
        folder_name (str): The name of the folder.

    Returns:
        str or None: The extracted digits as a string, or None if no digits found.
    """
    digits_match = re.search(r'\d+', folder_name)
    return digits_match.group() if digits_match else None

def process_folder(folder_path):
    """
    Process a single folder containing yeast data.

    Parameters:
        folder_path (str): The path to the folder to process.

    Returns:
        tuple: A tuple containing:
            - number_of_cells (int): Number of cells processed.
            - number_of_detections (int): Total number of detections.
            - number_of_detections_tot (list): List of detections per cell.
    """
    number_of_cells = 0
    number_of_detections = 0
    number_of_detections_tot = []

    # Extract digits from the folder name
    folder_name = os.path.basename(folder_path)
    digits = extract_digits(folder_name)

    if digits:
        # Clear GPU cache
        torch.cuda.empty_cache()

        # Update configuration for the current folder
        cfg['fn_reg_npc1'] = f'/BF1red{digits}.tiff'
        cfg['fn_reg_rnp1'] = f'/BF1green{digits}.tiff'
        cfg['fn_reg_npc2'] = f'/BF2red{digits}.tiff'
        cfg['fn_reg_rnp2'] = f'/BF2green{digits}.tiff'
        cfg['fn_track_rnp'] = f'/RNAgreen{digits}.tiff'
        cfg['fn_track_npc'] = f'/NEred{digits}.tiff'
        cfg['path'] = folder_path

        print(f'Processing folder: {folder_path}')

        # Initialize the yeast processor
        Yp = Yeast_processor(cfg)

        # Check if required files are present
        err = Yp.check_files()

        if not err:
            # Calibration and registration
            Yp.calibrate(savefig=save_figures)
            Yp.compute_registration(regmode=1, plotfig=save_figures)
            translation1 = Yp.registration['tvec']
            Yp.compute_registration(regmode=2, plotfig=save_figures)
            translation2 = Yp.registration['tvec']
            registration_diff = np.sqrt(abs(translation1[0] - translation2[0]) ** 2 + abs(
                    translation1[0] - translation2[0]) ** 2)
            # Check registration difference
            if registration_diff < threshold_regist:
                # Drift computation
                Yp.compute_drift(save_fig=save_figures)

                # NPC (Nuclear Pore Complex) detection
                Yp.detect_npc(
                    save_fig=save_figures,
                    count_good_label=40,
                    gap_closing_distance=10,
                    threshold=0.05,
                    oldmethod=False
                )

                # NPC fit refinement
                Yp.refinement_npcfit_movie_new(
                    movie=save_movies,
                    registration=True,
                    smoothness=10,
                    Lambda=0.1,
                    length_line=12,
                    estimate_prec=False,
                    save_fig=save_figures,
                    max_signs=np.inf,
                    iterations=300
                )

                # GLRT (Generalized Likelihood Ratio Test) detection
                _, _, mask, bb = Yp.GLRT_detector_DL_multichannel(
                    initial_guess=initial_guess,
                    bounds=bounds_glrt,
                    alfa=0.15,
                    lmlambda=0.0001,
                    iterations=20,
                    batch_size=80000,
                    number_channel=20
                )

                # Prepare MLE positions
                pos = Yp.prepare_mle(
                    minsize=7,
                    max_eccentricity=10,
                    use_old_mask=False
                )

                # Clear GPU cache again
                torch.cuda.empty_cache()

                # Gaussian MLE with fixed sigma
                Yp.gauss_mle_fixed_sigma(
                    pos_list=pos,
                    pfa_check=0.05,
                    vector=False
                )

                # Update counts
                number_of_cells += len(pos)
                if len(pos) > 0:
                    detections = sum(len(p) for p in pos)
                    number_of_detections += detections
                    number_of_detections_tot.extend(len(p) for p in pos)

                # Close all matplotlib figures to free memory
                plt.close('all')

                # Tracking without drift correction
                track_data = Yp.tracking(
                    movie=False,
                    drift_correct=False,
                    linking=3,
                    memory=10
                )

                # Include interpolated positions
                pos_including_inter = []
                for ne, tracks in enumerate(track_data):
                    if tracks.empty:
                        pos_including_inter.append(np.empty((0, 3)))
                    else:
                        frames = tracks['frame'].values[:, None]
                        y_coords = tracks['y'].values[:, None]
                        x_coords = tracks['x'].values[:, None]
                        pos_including_inter.append(np.hstack((frames, y_coords, x_coords)))

                # Gaussian MLE on interpolated data
                _, _, pfa, _, _ = Yp.gauss_mle_fixed_sigma(
                    pos_list=pos_including_inter,
                    interpolated=True,
                    vector=False,
                    pfa_check=0.2
                )

                # Tracking with drift correction
                Yp.tracking(
                    movie=save_movies,  # Note: movie may look weird in case of heavy drift
                    drift_correct=True,
                    linking=2,
                    memory=0,
                    pfa_arr=pfa
                )

                # Compute distances (e.g., between detected features)
                Yp.compute_distancev2()

                # Close all matplotlib figures to free memory
                plt.close('all')
            else:
                print(f"Registration difference too high ({registration_diff:.2f} pixels), skipping folder.")
        else:
            print(f"Required files missing in {folder_path}, skipping folder.")

    else:
        print(f"No digits found in folder name {folder_name}, skipping folder.")

    return number_of_cells, number_of_detections, number_of_detections_tot

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing the folder path.
    """
    parser = argparse.ArgumentParser(description='Process a folder with yeast data.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing yeast data.')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    folder_path_upper = args.folder_path

    # Initialize counters
    error_count = 0
    total_cells = 0
    total_detections = 0
    detections_per_cell = []

    # Process each subfolder in the specified folder
    for folder_name in tqdm.tqdm(os.listdir(folder_path_upper)):
        folder_path = os.path.join(folder_path_upper, folder_name)

        if os.path.isdir(folder_path):
            try:
                # Process the folder and update counts
                cells, detections, detections_tot = process_folder(folder_path)
                total_cells += cells
                total_detections += detections
                detections_per_cell.extend(detections_tot)
            except Exception as e:
                print(f"An error occurred while processing {folder_name}: {e}")
                error_count += 1

    print(f"Total errors: {error_count}")

    # Save results to files
    np.save(os.path.join(folder_path_upper, 'total_cells.npy'), total_cells)
    np.save(os.path.join(folder_path_upper, 'total_detections.npy'), total_detections)
    np.save(os.path.join(folder_path_upper, 'detections_per_cell.npy'), np.array(detections_per_cell))
