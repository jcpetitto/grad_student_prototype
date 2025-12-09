# NOTE

The purpose of this repository is to provide context for refactoring of yeast
image processing and analysis pipeline refactoring. The original author wrote
this code as a proof-of-concept for a specific dataset rather than as a
production-ready pipeline. While it contains novel ideas, this code does not
reflect what the graduate student considers scalable or distribution code for
mass distribution. Effort has been made on the part of this repository's owner
to decouple this code from the identity of the original author and their own
repository.

--------------------------------------------------------------------------------

# Yeast mRNA Tracking Analysis

This repository provides scripts and tools for processing yeast tracking data and extracting results related to mRNA localization. The analysis involves calibrating data, detecting nuclear envelopes (NE), performing maximum likelihood estimation (MLE) fitting, and tracking mRNA particles within yeast cells.

## Table of Contents

- [Installation](#installation)
- [Data Format](#data-format)
- [Generating Data](#generating-data)
- [Analyzing Data](#analyzing-data)
- [Troubleshooting](#troubleshooting)
- [Miscelleanous](#miscelleanous)
---

## Installation

Before running the analysis, ensure that you have the required Python environment set up. The scripts depend on several Python packages, including numpy, torch, matplotlib, and others. It is recommended to create a virtual environment to manage dependencies.

### Steps to Install the Environment
Follow these steps to set up your environment:
```bash
# 1. Create a Virtual Environment
python3 -m venv venv

# 2. Activate the Virtual Environment
# On macOS and Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# 3. Install Required Packages from requirements.txt
pip install -r requirements.txt

# 4. Verify the Installation
pip list

# 5. Deactivate the Virtual Environment (Optional)
deactivate
```

## Data Format

In the following data repository, you can find the following folders: [Data Repository](https://surfdrive.surf.nl/files/index.php/s/eHsgzvnQcQfx7Sv)

- **Example Raw Data/Images**:  
  Located in the folder `Example raw data`.
  
- **Gain/Offset Calibration Images**:  
  Found in the folder `Calibration Data`.
  
- **Traces Used for Publication Figures**:  
  Available in CSV format within the folder `results_tracks_gapclosing10`.
  
- **Trained Models Weights**:  
  Stored in the folder `trained_networks`.
  
- **Training Data for NE Segmentation**:  
  Located in the folder `training data NE segmentation`.
  
These datasets can be used for troubleshooting the code or recreating figures without requiring access to all raw data.

**Note:** Due to their large size, the complete raw data is available upon reasonable request.

### File Structure

```plaintext
Parent_Folder/
├── cell_xxx/
│   ├── BF1Greenxxx.tiff   # Bright-field images before NPC acquisition (Green channel)
│   ├── BF1Redxxx.tiff     # Bright-field images before mRNA acquisition (Red channel)
│   ├── RNAgreenxxx.tiff   # mRNA fluorescence images (Green channel)
│   ├── NEredxxx.tiff      # NE fluorescence images (Red channel)
│   ├── BF2greenxxx.tiff   # Bright-field images after NPC acquisition (Green channel)
│   └── BF2redxxx.tiff     # Bright-field images after mRNA acquisition (Red channel)
├── cell_yyy/
│   └── ...
├── cell_zzz/
│   └── ...
```


#### File Descriptions

- **BF1Greenxxx.tiff**: Bright-field images before Nuclear Pore Complex (NPC) acquisition in the green channel.
- **BF1Redxxx.tiff**: Bright-field images before mRNA acquisition in the red channel.
- **RNAgreenxxx.tiff**: Fluorescence images of mRNA in the green channel.
- **NEredxxx.tiff**: Fluorescence images of the Nuclear Envelope (NE) in the red channel.
- **BF2greenxxx.tiff**: Bright-field images after NPC acquisition in the green channel.
- **BF2redxxx.tiff**: Bright-field images after mRNA acquisition in the red channel.

## Generating Data

To process the data and perform the analysis, use the `runfile_single_folder.py` script.

### Configuration

Before running the script, update the configuration (`cfg`) and other variables according to your setup:

```python
cfg = {
    'gain': '/path/to/bright_images.tiff',       # Path to bright images (gain) for mRNA channel
    'offset': '/path/to/dark_images.tiff',      # Path to dark images (offset) for mRNA channel
    'model_NE': '/path/to/Modelweights_NE_segmentation.pt',  # NE segmentation model weights
    'model_bg': '/path/to/model_weights_background_psf.pth',  # Background estimation model weights
    'pixelsize': 128,                            # Pixel size of the camera in nm
    # ... other configurations ...
}
```

### Running the Script

Run the script from the command line:

```bash
python runfile_single_folder.py '/path/to/Parent_Folder'
```


Replace `'/path/to/Parent_Folder'` with the actual path to your parent folder containing the cell directories.

### Saving Figures and Movies (Optional)

To save figures from the process and generate movies of the NE fit and the found tracks, adjust the following parameters in the script:

```python
save_figures = True
save_movies = True
```

*Note*: Enabling these options may slow down the analysis and consume additional memory. It is recommended to use them only for debugging purposes.

### Output Files

If NE and trajectories are successfully found, the following files will be saved in the `results` folder within each cell directory:

- `amplitude_pointsx.data`
- `npc_pointsx.data`
- `sigma_pointsx.data`
- `tracks.data`

These files contain data that will be used later for bulk analysis.


## Analyzing Data

Once the analysis has been run for all desired subfolders, you can extract all the saved data for further analysis.

### File Structure for Multiple Folders

Organize your directories as follows to facilitate data extraction (this should be done automatically in the data generation):

```plaintext
Origin_Folder/
├── Parent_Folder1/
│   ├── cell_xxx/
│   ├── cell_yyy/
│   ├── cell_zzz/
│   └── results/
│       ├── cell_xxx/
│       │   ├── amplitude_pointsx.data
│       │   ├── npc_pointsx.data
│       │   ├── sigma_pointsx.data
│       │   └── tracks.data
│       ├── cell_yyy/
│       └── ...
├── Parent_Folder2/
│   └── ...
```

### Extracting Results

Use the `extract_results.py` script to extract all tracks from all parent folders into a single DataFrame:

- **all_tracks**: Contains all the tracks combined.
- **file_lookup**: Each track is marked with a unique ID, along with the original filename and NE number.
- **neidlookup** and **nedata**: Contain coordinates of the NE.

These DataFrames are saved accordingly in your specified folder within `extract_results.py`, ready for further analysis.


### Plots 
The majority of the graphs shown in the publication are created using the scripts in the folder [figures](figures).
Each script has it's docstring explaining the purpose.

## Troubleshooting

### `imgregdft` Package Compatibility

The `imgregdft` package may use `np.bool`, which is deprecated in newer versions of NumPy. To fix this, modify the `utils.py` file within the `imgregdft` package:

- Replace all occurrences of `np.bool` with `bool`.

### Results Folder Management

Be cautious with the `results` folder. If you plan to re-run the analysis with different parameters, ensure that you delete existing `results` folders to prevent conflicts or data overwriting.

### Movie Generation Dependencies

If you encounter issues while generating movies, ensure that you have the correct versions of `imageio` and `imageio-ffmpeg` installed:

```bash 
pip install imageio==2.19.3
pip install imageio-ffmpeg==0.4.7
```

*Note*: This analysis involves significant computational resources. Ensure that your system has enough memory and processing power, especially when processing large datasets or when enabling figure and movie generation.

For any further questions or issues, please refer to the documentation or contact the maintainers.

## Miscelleanous

### Create desired data format
In `miscellaneous/make_seperate_folders.py` one could find a way to save the recorded images in a desired format for the analysis.

### Run code on cluster
To run the code on the cluster, one could use the example given in `miscellaneous/submit_cluster.sh`. This requires a singularity container.
Once the code can be excecuted locally, the local environment can be packed in the singularity container and excecuted easily on the cluster.

For packaging the container, see the manual in the miscelleanous folder.


