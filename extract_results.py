"""
Script to process yeast tracking data and extract results.

This script uses the 'yeast_extractresults' class from the 'utils.Yeast_analyzer' module
to process tracking data and extract various results, which are then saved to files.

It can be run from the command line or from the console.
"""

import os
import pickle

from utils.Yeast_analyzer import yeast_extractresults


def main():
    """
    Main function to process yeast tracking data and extract results.
    """
    # Identifier string for output files
    string = '820'

    # Configuration dictionary for yeast data extraction
    cfg = {
        'mainpath': '/media/pieter/Extreme SSD/Yeast_tracking_data2023/BMY820',  # Folder to extract the data from
        'resultdir': "results",  # Folder name where results are saved in data folders
        'trackdata': 'tracks.data',  # Name of the track data file
        'pixelsize': 128,  # Pixel size of the camera (nm)
        'moviename': '/media/pieter/Extreme SSD/Yeast_tracking_data2023/test.mp4',  # Movie name (if created)
        'frametime': 0.02,  # time of frame (seconds)
        'save_folder': './results/'  # Folder to save the merged results for further analysis
    }

    # Ensure the save folder exists
    os.makedirs(cfg['save_folder'], exist_ok=True)

    # Instantiate the yeast_extractresults class with the configuration
    yr = yeast_extractresults(cfg)

    # Gather tracks and extract results
    (
        all_tracks,
        file_lookup,
        neidlookupdf,
        nedata,
        number_off_cells_all,
        number_off_detections_all,
        number_off_detections_tot_all
    ) = yr.gather_tracks()

    # Save the number of cells data to a pickle file
    with open(os.path.join(cfg['save_folder'], f"number_off_cells_all_{string}.pkl"), 'wb') as file1:
        pickle.dump(number_off_cells_all, file1)

    # Save the number of detections data to a pickle file
    with open(os.path.join(cfg['save_folder'], f"number_off_detections_all_{string}.pkl"), 'wb') as file2:
        pickle.dump(number_off_detections_all, file2)

    # Save the total number of detections data to a pickle file
    with open(os.path.join(cfg['save_folder'], f"number_off_detections_tot_all_{string}.pkl"), 'wb') as file3:
        pickle.dump(number_off_detections_tot_all, file3)

    # Save all_tracks DataFrame to a CSV file
    all_tracks.to_csv(os.path.join(cfg['save_folder'], f"all_tracks_{string}.csv"), index=False)

    # Save file_lookup DataFrame to a CSV file
    file_lookup.to_csv(os.path.join(cfg['save_folder'], f"file_lookup_{string}.csv"), index=False)

    # Save neidlookupdf DataFrame to a CSV file
    neidlookupdf.to_csv(os.path.join(cfg['save_folder'], f"ne_lookup_{string}.csv"), index=False)

    # Save nedata to a pickle file
    with open(os.path.join(cfg['save_folder'], f"nedata_{string}.pkl"), 'wb') as file4:
        pickle.dump(nedata, file4)

    print("Data extraction and saving completed successfully.")
    return all_tracks

if __name__ == "__main__":
    tracks = main()
