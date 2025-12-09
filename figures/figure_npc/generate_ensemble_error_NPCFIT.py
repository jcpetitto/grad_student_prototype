"""
Title: Yeast NPC Fitting Error Analysis Script

Description:
This script processes yeast nuclear pore complex (NPC) microscopy data to analyze the fitting errors and intensities along the NPCs. It performs the following steps:

1. **Initialization**: Sets up the configuration parameters and imports necessary modules.
2. **Data Processing**:
   - Iterates over data directories to process each cell's data.
   - Uses the `Yeast_processor` class to detect NPCs and refine NPC fits.
   - Collects fitting errors and intensity values for all cells.
3. **Error Analysis**:
   - Calculates statistical measures such as mean error and mean squared error (MSE).
   - Filters out any invalid or extreme values.
4. **Visualization**:
   - Generates plots to visualize the mean error and MSE along the distance of the NPCs.
   - Saves the plots as SVG files.

Functions:
- `extract_digits(folder_name)`: Extracts numerical digits from a folder name to identify image files.

Usage:
- Ensure all required data files and models are available in the specified paths.
- Adjust the configuration dictionary `cfg` with the correct file paths and parameters.
- Run the script to process the data and generate the plots.
- Results will be saved as SVG files in the current directory.

Dependencies:
- numpy
- matplotlib
- tqdm
- torch
- scienceplots
- re
- os
- Custom module: `Yeast_processor` (should be available in the Python path)

Notes:
- The script uses a custom `Yeast_processor` class, which must be defined in a module accessible to the script.
- File paths in the configuration should be updated to reflect the correct locations of your data and models.
- The plotting style is set to 'science' for high-quality figures suitable for publications.
"""



from utils.Yeast_processor import Yeast_processor
import os
import tqdm
import re
import torch
import scienceplots
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('science')
def show_napari(img):
    import napari
    viewer = napari.imshow(img)
def show_napari_points(img, points):
    import napari
    viewer = napari.Viewer()
    viewer.add_image(img)
    viewer.add_points(points, symbol='cross', size=3, face_color='red', edge_width=0 )

folder = '/home/pieter/Data/Yeast/july_7_exposure_20ms_emgain300'

cfg = {
    'path': '/home/pieter/Data/Yeast/july_7_exposure_20ms_emgain300/',
    'fn_reg_npc': 'BFred21.tiff',
    'fn_reg_rnp': 'BFgreen19.tiff',
    'fn_track_rnp': 'NEgreen19.tiff',
    'fn_track_npc': 'NEred19.tiff',
    'roisize':8, # even number please
    'sigma': 1.3,
    'frames': [0, 2000],
    'frames_npcfit': [0,250],
    'drift_bins': 4,
     'resultdir': "/results/",
    'gain': '/home/pieter/Data/Yeast/bright_images_20ms.tiff',
    'offset': '/home/pieter/Data/Yeast/dark_images_20ms.tiff',
    'model_NE': '../trained_networks/Modelweights_NE_segmentation.pt',
    'model_bg': '../trained_networks/model_wieghts_background_psf.pth',
    'pixelsize': 128
}
def extract_digits(folder_name):
    # Use regular expression to extract digits from the folder name
    digits_match = re.search(r'\d+', folder_name)
    if digits_match:
        return digits_match.group()
    return None
error_all_cells = []
values_all_cells = []
makefig = False # true for figuresaves and to make a movie
if __name__ == "__main__":
    dif_list = []
    base_dir = "/media/pieter/Extreme SSD/Yeast_tracking_data2023/BMY823"
    root_dirs = os.listdir(base_dir)

    full_paths = [os.path.join(base_dir, item) for item in root_dirs]

    # root_dirs = ["/media/pieter/Extreme SSD/Yeast_tracking_data2023/BMY823/BMY823_7_16_23_aqsettings1_batchA",
    #             "/home/pieter/Data/Yeast/tracking data/BMY823_7_16_23_aqsettings1_batchB",
    #              "/home/pieter/Data/Yeast/tracking data/BMY823_7_20_23_aqsettings1_batchB"]
    count = 0
    numcells = 0
    for root_dir in full_paths:
        # if numcells > 1000:
        #     break
        for folder_name in tqdm.tqdm(os.listdir(root_dir)):
            # if numcells > 1000:
            #     break
            # if count>20:
            #     break
            folder_path = os.path.join(root_dir, folder_name)
            #if folder_name == 'cell 20':
            if os.path.isdir(folder_path):

                try:
                    digits = extract_digits(os.path.basename(folder_path))
                    number_off_cells = 0
                    number_off_detections = 0
                    number_off_detections_tot = []
                    if digits:
                        print(folder_path)
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

                        # Yp.calibrate(savefig=False)
                        #Yp.compute_registration()
                        #Yp.compute_drift(save_fig=True)
                        logits = Yp.detect_npc(save_fig=False, count_good_label=40, gap_closing_distance=10,
                                               threshold=0.05)

                        all_points, errors, values = Yp.refinement_npcfit_movie_new(movie=False,
                                                                                    registration=False,
                                                                                    smoothness=10,
                                                                                    Lambda=0.001,
                                                                                    length_line=12,
                                                                                    estimate_prec=False,
                                                                                    save_fig=False,
                                                                                    max_signs=np.inf,
                                                                                    iterations=300)


                        import numpy as np

                        error_all_cells.append(errors)
                        values_all_cells.append(values)
                        count += 1
                        print(count)
                        numcells = numcells + len(all_points)
                        print(' number of cells = ', numcells)
                # except
                #
                #    print('no file')
                except:
                    print('error')
    numcells_it = 0
    error_array = []
    val_array = []



    Yp.fn_dark_image = '/media/pieter/Extreme SSD/Yeast_tracking_data2023/red_dark300.tif'
    Yp.fn_bright_image = '/media/pieter/Extreme SSD/Yeast_tracking_data2023/red_gain300.tif'
    gain300, offset300,_ =Yp.calibrate(savefig=False)
    for qq in range(len(error_all_cells)):
        item = error_all_cells[qq]
        item_val = values_all_cells[qq]

        if len(item) != 0:
            for ii in range(len(item)):
                numcells_it +=1
                per_cell = np.array(item[ii])[0,0,...]
                per_cell_val = (np.array(item_val[ii])[0,0,...])


                if len(error_array) == 0:
                    error_array = per_cell*1
                    val_array = per_cell_val*1
                else:
                    error_array = np.concatenate((error_array,per_cell),axis=-1)
                    val_array = np.concatenate((val_array,per_cell_val),axis=-1)

       # test = np.array(error_all_cells)

    import matplotlib.pyplot as plt
    import numpy as np
    plt.close('all')

    # Calculate mean and standard deviation
    # Create boolean masks for NaN values
    nan_mask_error = np.isnan(error_array)
    nan_mask_val = np.isnan(val_array)

    # Combine the masks to find columns with any NaNs in either array
    combined_nan_mask = np.any(nan_mask_error | nan_mask_val, axis=0)

    # Filter out columns with NaNs from both arrays
    error_array = error_array[:, ~combined_nan_mask]
    val_array = val_array[:, ~combined_nan_mask]


    np.save('error_arr.npy', error_array)
    np.save('val_arr.npy', val_array)

    # error is defined as fit-data/fti
    chi_error_nomean =(error_array * val_array* gain300) ** 2
    columns_to_exclude = np.any(chi_error_nomean > 400, axis=0) # filter out
    chi_error_nomean = chi_error_nomean[:, ~columns_to_exclude]

    test = chi_error_nomean.T
    std_chi = np.std(chi_error_nomean, axis=1)
    chi_error = np.mean((error_array*val_array*gain300)**2, axis=1)


    meanvals = np.mean(error_array*100, axis=1)
    std = np.std(error_array*100, axis=1)

    # Plot for Mean Error
    fig, ax = plt.subplots(dpi=400)
    x = np.linspace(0, 12, 100)
    ax.plot(x, meanvals, label='Mean')
    ax.fill_between(x, meanvals - std, meanvals + std, alpha=0.5, label='Standard Deviation')
    ax.set_ylabel(r'Mean ensemble error $E$ [\%]')
    ax.set_xlabel(r'Distance [px]')
    ax.set_ylim(-5, 5)
    ax.legend()
    plt.tight_layout()
    plt.savefig('mean_error.svg', format='svg')
    plt.show()

    # Plot for Chi-Square Error
    fig, ax = plt.subplots(dpi=400)
    ax.plot(x, chi_error, label='Chi-Square Error')
    ax.fill_between(x, chi_error - std_chi, chi_error + std_chi, alpha=0.5, label='Standard Deviation')
    ax.set_ylabel(r'MSE [Photons$^2$]')
    ax.set_xlabel(r'Distance [px]')
    #ax.legend()
    plt.tight_layout()
    plt.savefig('chi_square_error.svg', format='svg')
    plt.show()

    # Stacked Plot for both Mean and Chi-Square Error
    fig, axs = plt.subplots(nrows=2, ncols=1, dpi=400, sharex=True, figsize=(2,2))

    # Mean Error Plot
    axs[0].plot(x, meanvals, label='Mean')
    axs[0].fill_between(x, meanvals - std, meanvals + std, alpha=0.5, label='Standard dev.')
    axs[0].set_ylabel(r'Mean $E$ [\%]')
    axs[0].set_ylim(-5, 5)
    axs[0].legend()

    # Chi-Square Error Plot
    axs[1].plot(x, chi_error, label='Chi-Square Error')
    axs[1].fill_between(x, chi_error - std_chi, chi_error + std_chi, alpha=0.5, label='Standard Deviation')
    #axs[1].set_ylabel(r'$\chi^2$ [Photons]')
    axs[1].set_ylabel('MSE\n[Photons$^2$]')
    axs[1].set_xlabel(r'Distance [px]')
    axs[1].set_ylim(-3, 13)
    #axs[1].legend()

    plt.tight_layout(pad=0.1)
    plt.savefig('combined_errors.svg', format='svg')
    plt.show()


    plt.figure(dpi=400)
    plt.plot(np.linspace(0,12,100), np.median(val_array,axis=1))
    plt.xlabel(r'Distance along normal $n$ [px]')
    plt.ylabel(r'Intensity $I$ [au]')
    plt.tight_layout()
    plt.show()