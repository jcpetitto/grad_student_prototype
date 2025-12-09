
#### Building a Singularity Container with a Conda Environment

This manual provides a step-by-step guide to building a Singularity container that incorporates a pre-packaged Conda environment. This setup ensures consistent computational environments across different systems, facilitates reproducible research, and simplifies deployment on HPC clusters.

#### Table of Contents

1. Introduction
2. Prerequisites
3. Step 1: Install Singularity
4. Step 2: Prepare the Conda Environment
5. Step 3: Export the Conda Environment
6. Step 4: Create the Singularity Definition File
7. Step 5: Build the Singularity Container
8. Step 6: Test the Container
9. Step 7: Using the Container
10. Troubleshooting
11. Conclusion

---

#### Introduction

Singularity is a container platform designed for use in high-performance computing (HPC) environments. It allows users to create portable and reproducible computational environments. By incorporating a Conda environment into a Singularity container, you can ensure that all necessary dependencies and configurations are encapsulated, making your workflows more reliable and easier to share.

---

#### Prerequisites

Before proceeding, ensure that you have the following:

- **Access to a Linux System**: Singularity primarily runs on Linux. If you are using macOS or Windows, consider using a virtual machine or Docker to emulate a Linux environment.

- **Administrative Privileges**: Building Singularity containers typically requires root access.

- **Singularity Installed**: Ensure Singularity is installed on your system. Refer to Step 1 for installation instructions.

- **An Existing Conda Environment**: This environment should contain all the necessary packages for your projects.

- **Packed Conda Environment Archive (packed_environment.tar.gz)**: This is a compressed archive of your Conda environment. Instructions to create this archive are provided in Step 3.

---

#### Step 1: Install Singularity
For the most recent installation manual, please visit the [singularity website](https://docs.sylabs.io/guides/3.0/user-guide/installation.html).

#### Step 2: Prepare the Conda Environment

#### 2.1 Activate Your Conda Environment

Activate the Conda environment you wish to package:

```bash
conda activate your_env_name
```

*Replace `your_env_name` with the name of your environment.*

#### 2.2 Clean the Environment (Optional)

To ensure that the environment is clean and free of unnecessary files:

```bash
conda clean --all
```

*Note*: This step removes unused packages and caches, reducing the size of the environment.

---

#### Step 3: Export the Conda Environment

To incorporate your Conda environment into the Singularity container, export it into a compressed archive.

#### 3.1 Export the Environment

```bash
conda env export --no-builds ``` environment.yml
```

#### 3.2 Create a Packed Conda Environment

Singularity requires a "packed" Conda environment. Use the `conda-pack` tool to create this archive.

1. **Install `conda-pack`**:

```bash
conda install -c conda-forge conda-pack
```

2. **Pack the Environment**:

```bash
conda pack -n your_env_name -o packed_environment.tar.gz
```

*Replace `your_env_name` with the name of your environment.*

3. **Verify the Archive**:

Ensure that `packed_environment.tar.gz` is created in your current directory.

---

#### Step 4: Create the Singularity Definition File

The Singularity definition file specifies how the container is built, including the base image and any additional files or commands to execute during the build process.

#### 4.1 Create `my_container.def`

Create a file named `my_container.def` with the following content:

```def
Bootstrap: docker
From: continuumio/anaconda3:2021.05

%files
    packed_environment.tar.gz /packed_environment.tar.gz

%post
    # Extract the packed Conda environment to /opt/conda
    tar xvzf /packed_environment.tar.gz -C /opt/conda
    
    # Activate the environment (assuming conda-unpack is available)
    /opt/conda/bin/conda-unpack
    
    # Remove the packed environment archive to save space
    rm /packed_environment.tar.gz

%environment
    # Set environment variables
    export PATH=/opt/conda/bin:$PATH
```

#### 4.2 Explanation of the Definition File

- **Bootstrap**: Specifies the base image source. Here, it uses a Docker image.

- **From**: Indicates the Docker image to use as the base.

- **%files**: Lists files to be copied into the container during the build.

- **%post**: Contains shell commands executed during the build process.

- **%environment**: Sets environment variables every time the container is run.

---

#### Step 5: Build the Singularity Container

With the definition file and the packed Conda environment ready, proceed to build the Singularity container.

#### 5.1 Place `packed_environment.tar.gz` and `my_container.def` in the Same Directory

Ensure both files are located in your current working directory or provide their paths accordingly.

#### 5.2 Build the Container

Execute the following command to build the Singularity Image File (`.sif`):

```bash
sudo singularity build conda.sif my_container.def
```

**Parameters**:

- `conda.sif`: The name of the output Singularity container.

- `my_container.def`: The Singularity definition file.

*Note*: Building containers typically requires root privileges.

#### 5.3 Verify the Build

Once the build process completes, verify that `conda.sif` is created:

```bash
ls -lh conda.sif
```

*Expected Output*:
`-rw-r--r-- 1 user user 2.5G Oct 10 12:34 conda.sif`


*Note*: The file size may vary based on the size of your Conda environment.

---

#### Step 6: Test the Container

Before deploying the container in a production environment, it is crucial to test its functionality.

#### 6.1 Shell Access

Launch an interactive shell within the container:

```bash
singularity shell conda.sif
```

*Expected Behavior*:

- You should be inside the container's shell.

- The Conda environment should be active, and all your packages accessible.

#### 6.2 Verify Conda Environment

Within the container shell, check the Conda environment:

```bash
conda list
```

*Expected Output*:

- A list of all packages installed in your Conda environment.

#### 6.3 Run a Python Script

Test executing a Python script to ensure everything works as expected:

1. **Create a Test Script**:

```bash
echo 'import sys; print("Python version:", sys.version)' ``` test.py
```

2. **Execute the Script**:

```bash
python test.py
```

*Expected Output*:
`Python version: 3.8.5 (default, Jul 21 2020, 10:48:26) [GCC 7.3.0]`


*Note*: The Python version should match the one in your Conda environment.

#### 6.4 Exit the Container

Exit the interactive shell:

```bash
exit
```

---

#### Step 7: Using the Container

Now that your Singularity container is built and tested, you can use it to run your applications or scripts.

#### 7.1 Executing Commands

Run a command directly within the container without entering an interactive shell:

```bash
singularity exec conda.sif python your_script.py
```

*Replace `your_script.py` with the path to your Python script.*

#### 7.2 Binding Directories

If your scripts require access to specific directories (e.g., data folders), use the `--bind` (`-B`) option to mount them:

```bash
singularity exec -B /path/to/data:/data conda.sif python /data/your_script.py
```

*This mounts `/path/to/data` on your host to `/data` inside the container.*

#### 7.3 Running as a Job on a Cluster

If you are using an HPC cluster, you can integrate Singularity with your job scheduler by incorporating Singularity commands into your job submission scripts.

*Example SLURM Job Script*:

```bash
#!/bin/bash
#SBATCH --job-name=conda_job
#SBATCH --output=conda_job.out
#SBATCH --error=conda_job.err
#SBATCH --time=01:00:00
#SBATCH --partition=compute

# Load Singularity module if required
module load singularity

# Execute your Python script within the container
singularity exec conda.sif python /path/to/your_script.py
```

*Submit the job*:

```bash
sbatch your_job_script.sh
```

---

#### Troubleshooting

#### Common Issues and Solutions

1. **Permission Denied Errors**:

   - **Cause**: Insufficient permissions to build or execute the container.

   - **Solution**: Ensure you have `sudo` privileges when building the container.

2. **Missing Conda Packages**:

   - **Cause**: The Conda environment was not correctly packed or unpacked.

   - **Solution**: Revisit Step 3 to ensure the environment was properly exported and packed.

3. **Singularity Build Failures**:

   - **Cause**: Issues within the definition file or missing dependencies.

   - **Solution**: Review the Singularity build logs for specific error messages.

4. **Container Execution Issues**:

   - **Cause**: Conflicts between host and container environments or missing dependencies.

   - **Solution**: Test the container interactively to identify and resolve issues.

---


