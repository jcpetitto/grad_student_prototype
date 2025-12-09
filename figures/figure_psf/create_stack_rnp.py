"""
Title: Yeast NPC Data Extraction Script for PSF Fitting

Description:
This script processes microscopy image data of yeast nuclear pore complexes (NPCs) to extract samples and background images for point spread function (PSF) fitting. It is designed to generate the data required for PSF analysis, as performed in a previous script. The script performs the following steps:

1. **Initialization**:
   - Imports necessary libraries and modules.
   - Defines helper functions for visualization (optional).
   - Sets up a configuration dictionary `cfg` with parameters for data processing.

2. **Function Definitions**:
   - `extract_digits(folder_name)`: Extracts numerical digits from folder names to identify image files.
   - `process_folder(folder_path, check_mask=False, use_old_mask=False)`: Processes individual folders to extract sample and background images.

3. **Main Processing Loop**:
   - Iterates over specified directories containing the data.
   - Limits the number of folders processed based on `max_counts`.
   - Calls `process_folder` for each valid folder to perform data extraction.
   - Collects sample and background images into lists.

4. **Data Saving**:
   - After processing, concatenates the collected sample and background images.
   - Saves the data as `.npy` files for later use in PSF fitting.

Usage:
- Update the `cfg` dictionary with the correct file paths and parameters specific to your data.
- Ensure all required data files and models are available in the specified paths.
- Run the script to process the data and generate the sample and background image arrays.
- The resulting `.npy` files can be used in subsequent PSF fitting scripts.

Dependencies:
- numpy
- matplotlib
- tqdm
- torch
- scienceplots
- re
- os
- time
- tifffile
- Custom module: `Yeast_processor` from `utils.Yeast_processor`

Notes:
- The script uses a custom `Yeast_processor` class, which must be defined in a module accessible to the script.
- File paths in the configuration should be updated to reflect the correct locations of your data and models.
- The script includes optional visualization functions that require the `napari` package.
- The plotting style is set to 'science' for high-quality figures suitable for publications.
- Adjust the `max_counts` variable to control the number of folders processed.

"""

import os
import re
import pickle

import tifffile
import torch.cuda
import matplotlib.pyplot as plt
from utils.Yeast_processor import Yeast_processor
import tqdm
import scienceplots
import numpy as np
import os

import scienceplots


def show_napari(img):
    import napari
    viewer = napari.imshow(img)

def show_tensor(img):
    import napari
    viewer = napari.imshow(img.detach().cpu().numpy())

def show_napari_points(img, points):
    import napari
    viewer = napari.Viewer()
    viewer.add_image(img)
    viewer.add_points(points, symbol='cross', size=3, face_color='red', edge_width=0)

cfg = {
    'path': '/home/pieter/Data/Yeast/july_7_exposure_20ms_emgain300/cell 21/',
    'fn_reg_npc1': 'BF1red{}.tiff',
    'fn_reg_rnp1': 'BF1green{}.tiff',
    'fn_reg_npc2': 'BF2red{}.tiff',
    'fn_reg_rnp2': 'BF2green{}.tiff',
    'fn_track_rnp': 'RNAgreen{}.tiff',
    'fn_track_npc': 'NEred{}.tiff',
    'roisize':16, # even number please
    'sigma': 0.8125,
    'frames': [0, 1000],
    'frames_npcfit': [0,250],
    'drift_bins': 4,
     'resultdir': "/results/",
    'gain': '/media/pieter/Extreme SSD/Yeast_tracking_data2023/bright_images_green_channel_20ms_300EM.tiff',
    # 'gain': '/home/pieter/Data/astigmatist_cal_100mlam_1_MMStack_Default.ome.tif',
    'offset': '/media/pieter/Extreme SSD/Yeast_tracking_data2023/dark_images_green_channel_20ms_300EM.tiff',
    'model_NE': '../trained_networks/FPNresnet34_yeastvAnyee.pt',
    'model_bg': '../trained_networks/noleakyv4_v_vectorpsf.pth',
    'pixelsize':128
}
threshold_regist = 3 # threshold for registration difference before and after (in pixels)
def extract_digits(folder_name):
    # Use regular expression to extract digits from the folder name
    digits_match = re.search(r'\d+', folder_name)
    if digits_match:
        return digits_match.group()
    return None
import time
def process_folder(folder_path, check_mask=False,use_old_mask=False):
    global cfg
    Yp = Yeast_processor(cfg)
    # Extract the digits from the folder name
    digits = extract_digits(os.path.basename(folder_path))
    number_off_cells = 0
    number_off_detections = 0
    number_off_detections_tot = []
    smp = []
    bg_concat = []
    if digits:
        torch.cuda.empty_cache()
        # Modify the cfg dictionary for the current folder
        cfg['fn_reg_npc1'] = '/BF1red' + digits + '.tiff'
        cfg['fn_reg_rnp1'] = '/BF1green' + digits + '.tiff'
        cfg['fn_reg_npc2'] = '/BF2red' + digits + '.tiff'
        cfg['fn_reg_rnp2'] = '/BF2green' + digits + '.tiff'
        cfg['fn_track_rnp'] = '/RNAgreen' + digits + '.tiff'
        cfg['fn_track_npc'] = '/NEred' + digits + '.tiff'
        # Update the path in the configuration with the current folder path
        cfg['path'] = folder_path
        # Perform the processing steps for the current folder
        Yp = Yeast_processor(cfg)
        err = Yp.check_files()
        if not err:
            Yp.calibrate(savefig=False)
            Yp.compute_registration(regmode=1)
            translation1 = Yp.registration['tvec']
            Yp.compute_registration(regmode=2)
            translation2 = Yp.registration['tvec']
            np.sqrt(abs(translation1[0] - translation2[0])**2+abs(translation1[0] - translation2[0])**2)

            if  np.sqrt(abs(translation1[0] - translation2[0])**2+abs(translation1[0] - translation2[0])**2)<threshold_regist:
                Yp.compute_drift(save_fig=False)
                logits = Yp.detect_npc(save_fig=False, count_good_label=50, gap_closing_distance=10, threshold=0.05)

                all_points, errors, values = Yp.refinement_npcfit_movie_new(movie=False, registration=True,
                                                                            smoothness=10, Lambda=1000, length_line=12,iterations=20,save_fig=False)
                bounds_glrt = [[0, cfg['roisize']-1],
                               [0, cfg['roisize']],
                               [0, 1e9],
                               [0, 1e6]]  # x, y, photons, bg
                initial_guess = [cfg['roisize']/2, cfg['roisize']/2, 0, 60]  # x, y, photons, bg

                dist_time = time.time()
                _, _, mask, bb = Yp.GLRT_detector_DL_multichannel(initial_guess, bounds_glrt, alfa=0.05,
                                                                  lmlambda=0.0001, iterations=20,
                                                                  batch_size=80000, number_channel=20,)
                print('detection takes', time.time() - dist_time)
                # traces, tracesbg = Yp.GLRT_detector_fast(initial_guess, bounds_glrt, alfa=0.05, lmlambda=0.001,
                #                                          iterations=20)
                pos = Yp.prepare_mle(minsize=4, max_eccentricity=10,use_old_mask=False)


                torch.cuda.empty_cache()


                # if check_mask:
                # check_num =0
                # mask_check = mask[check_num]
                # bb_check = bb[check_num]
                # show_napari(np.concatenate((mask_check,bb_check/np.amax(bb_check,axis=(1,2))[...,None,None]), axis=-1))
                # # traces, tracesbg = Yp.GLRT_detector_fast(initial_guess, bounds_glrt, alfa=0.3, lmlambda=0.1, iterations=20)



                muvec,_,_,smp, bg_concat = Yp.gauss_mle_fixed_sigma(pos,pfa_check=0.01,vector=True)

    return smp,bg_concat
    #return pos, params_list, roi_pos_list, smpconcat, mu, Yp.bbox_NE, Yp


if __name__ == "__main__":
    count_errors = 0
    #"/home/pieter/Data/Yeast/tracking data/BMY823_7_16_23_aqsettings1_batchA",
    root_dirs = ["/media/pieter/Extreme SSD/Yeast_tracking_data2023/BMY823"]
    max_counts = 100
    number_off_cells_all = 0
    number_off_detections_all = 0
    number_off_detections_tot_all = []
    smp_list = []
    bg_list = []
    countsss = 0
    for root_dir in root_dirs:

        if countsss > max_counts:
            break
        for folder_name_upper in tqdm.tqdm(os.listdir(root_dir)):
            if countsss > max_counts:
                break
            folder_path_upper = os.path.join(root_dir, folder_name_upper)
            if countsss> max_counts:
                break

            for folder_name in tqdm.tqdm(os.listdir(folder_path_upper)):
                if countsss > max_counts:
                    break
                folder_path = os.path.join(folder_path_upper, folder_name)

                if os.path.isdir(folder_path):
                    print(folder_path)

                    #pos,params_list, roi_pos_list, smpconcat, mu,boundingbox_coordinates, Yp = process_folder(folder_path)
                    try:

                        smp, bg = process_folder(folder_path, use_old_mask=False)
                        if len(smp)> 0:
                            smp_list.append(smp)
                            bg_list.append(bg)
                    except:
                        count_errors +=1
                        print('error!')
                        # t_nump = traces[0].detach().cpu().numpy()
                        # t_nump1 = traces[1].detach().cpu().numpy()

                    plt.close('all')

                countsss += 1
                print(countsss)
    import numpy as np

    print('amount of errors: ')
    print(count_errors)

    # smp_ori = np.concatenate(smp_list,axis=0)
    # bg_smp_ori = np.concatenate(bg_list,axis=0)
    # np.save('smp_ori_v2.npy', smp_ori)
    # np.save('bg_smp_v2_ori.npy', bg_smp_ori)

