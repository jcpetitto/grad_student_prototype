#!/bin/bash

###############################################################################
# Yeast mRNA Tracking Job Submission Script
#
# Description:
# This script automates the submission of GPU-accelerated jobs for processing
# yeast tracking data. It iterates through specified parent directories, identifies
# subfolders containing data, and submits a job for each subfolder using the LSF
# job scheduler.
#
# Usage:
# Ensure that the necessary modules and Singularity containers are available.
# Customize the configuration variables below as needed.
###############################################################################

# ============================
# Configuration Section
# ============================

# <<< MODIFY THESE VARIABLES AS NEEDED >>>
# Array of parent directories containing data subfolders
parent_folders=(
    "/path/to/data/BMY820"
    "/path/to/data/BMY822"
    "/path/to/data/BMY823"
    # Add more parent directories here
    # "/path/to/data/BMY4449"
    # "/path/to/data/BMY4453"
    # "/path/to/data/BMY4449_32C2"
    # "/path/to/data/BMY4453_32C"
    # "/path/to/data/BMY4452"
)

# Array of corresponding log directories for each parent folder
# Ensure that the order of log_dirs matches parent_folders
log_dirs=(
    "$HOME/logs820"
    "$HOME/logs822"
    "$HOME/logs823"
    # Add more log directories here
    # "$HOME/logs4449"
    # "$HOME/logs4453"
    # "$HOME/logs4449_32C2"
    # "$HOME/logs4453_32C"
    # "$HOME/logs4452"
)

# Path to the Singularity singularity container
container_path="$HOME/container_yeast.sif"

# Path to the Python script to execute within the singularity container
python_script="$HOME/yeast_processor_v2/runfile_singlefolder.py"

# GPU job parameters
job_queue="gpu"                      # Job queue name
gpu_resource='"num=1"'                # GPU resource specification
memory_usage="20000MB"               # Memory usage per job
job_time="48:00"                      # Maximum job time (HH:MM)

# CUDA module version
cuda_module="cuda/11.8"
cuda_path="/share/pkg/cuda/11.8.0"

# ============================
# End of Configuration Section
# ============================

# Function to display an error message and exit
function error_exit {
    echo "Error: $1" >&2
    exit 1
}

# Check if the number of parent folders matches the number of log directories
if [ "${#parent_folders[@]}" -ne "${#log_dirs[@]}" ]; then
    error_exit "The number of parent folders and log directories must match."
fi

# Loop over each pair of parent folder and corresponding log directory
for index in "${!parent_folders[@]}"; do
    parent_folder="${parent_folders[index]}"
    logs_dir="${log_dirs[index]}"

    # Create the log directory if it doesn't exist
    mkdir -p "$logs_dir" || error_exit "Failed to create log directory: $logs_dir"

    echo "Processing parent folder: $parent_folder"
    echo "Logging to: $logs_dir"

    # Loop through all subdirectories in the current parent folder
    for subfolder in "$parent_folder"/*; do
        if [ -d "$subfolder" ]; then
            subfolder_name=$(basename "$subfolder")
            job_name="gpu_job_${subfolder_name}"
            subfolder_log_dir="${logs_dir}/${subfolder_name}"

            # Create a dedicated log directory for the subfolder
            mkdir -p "$subfolder_log_dir" || error_exit "Failed to create subfolder log directory: $subfolder_log_dir"

            # Create an input file containing the subfolder path
            echo "$subfolder" > "${subfolder_log_dir}/input_${subfolder_name}.txt"

            # Define the job submission script path
            job_script="${subfolder_log_dir}/submit_job_${subfolder_name}.sh"

            # Create the job submission script with necessary directives and commands
            cat > "$job_script" <<EOF
#!/bin/bash
#BSUB -J "$job_name"                          # Job name
#BSUB -q $job_queue                           # Queue name
#BSUB -gpu $gpu_resource                      # GPU resource request
#BSUB -n 1                                    # Number of CPU cores
#BSUB -R "rusage[mem=${memory_usage}]"        # Memory usage
#BSUB -W ${job_time}                          # Wall-clock time
#BSUB -o "${subfolder_log_dir}/job_output_${subfolder_name}.log"  # Standard output
#BSUB -e "${subfolder_log_dir}/job_error_${subfolder_name}.log"    # Standard error

# Load necessary modules
module load $cuda_module || exit 1

# Update environment variables for CUDA
export PATH=$cuda_path/bin:\$PATH
export LD_LIBRARY_PATH=$cuda_path/lib64:\$LD_LIBRARY_PATH

# Read the subfolder path from the input file
subfolder_path=\$(cat "${subfolder_log_dir}/input_${subfolder_name}.txt")

# Adjust permissions to ensure the script can access necessary files
chmod +rwx "\$subfolder_path"
chmod -R a+w "\$subfolder_path/results" || echo "No results directory to chmod."

# Remove existing results directory if it exists to prevent conflicts
rm -rf "\$subfolder_path/results"

# Execute the Python script within the Singularity container
singularity exec --nv "$container_path" python "$python_script" "\$subfolder_path" 0.2
EOF

            # Make the job script executable
            chmod +x "$job_script" || error_exit "Failed to make job script executable: $job_script"

            # Submit the job using bsub
            bsub < "$job_script" || echo "Failed to submit job for subfolder: $subfolder"

            # Optionally, remove old log files to save space
            rm -f "${subfolder_log_dir}/job_output_${subfolder_name}.log"*
            rm -f "${subfolder_log_dir}/job_error_${subfolder_name}.log"*
        fi
    done
done

echo "All jobs have been submitted."
