#!/usr/bin/env python3
"""
organize_files.py

Description:
This script organizes files within specified parent directories by moving them into
subfolders based on the trailing digits in their filenames. Each subfolder is named
"cell <number>", where <number> corresponds to the trailing digits found before the
file extension.

Usage:
    python organize_files.py /path/to/mother_directory

    If no directory is provided, it defaults to a predefined path.
"""

import os
import shutil
import re
import sys
from pathlib import Path


def get_trailing_digits(filename):
    """
    Extracts the trailing digits before the file extension from a filename.

    Args:
        filename (str): The name of the file.

    Returns:
        str or None: The trailing digits as a string if found, else None.
    """
    # Use regex to find the last group of consecutive digits before the file extension
    match = re.search(r'(\d+)(?=\.[^.]+$)', filename)
    if match:
        return match.group(1)
    return None


def organize_files(source_dir):
    """
    Organizes files in the given source directory by moving them into subfolders
    based on trailing digits in their filenames.

    Args:
        source_dir (Path): The path to the directory containing files to organize.
    """
    if not source_dir.is_dir():
        print(f"Source directory does not exist or is not a directory: {source_dir}")
        return

    # Iterate over all files in the source directory
    for file in source_dir.iterdir():
        if file.is_dir():
            # Skip subdirectories
            continue

        number = get_trailing_digits(file.name)
        if not number:
            print(f"No trailing digits found in filename: {file.name}. Skipping.")
            continue

        # Define the target directory name
        target_dir = source_dir / f"cell {number}"

        try:
            # Create the target directory if it doesn't exist
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"Ensured target directory exists: {target_dir}")
        except Exception as e:
            print(f"Failed to create directory {target_dir}: {e}")
            continue

        # Define the target file path
        target_file = target_dir / file.name

        try:
            # Move the file to the target directory
            shutil.move(str(file), str(target_file))
            print(f"Moved file {file} to {target_file}")
        except Exception as e:
            print(f"Failed to move file {file} to {target_file}: {e}")


def main():
    """
    Main function to execute the file organization process.
    """
    # Check if mother_dir is provided as a command-line argument
    if len(sys.argv) > 1:
        mother_dir = Path(sys.argv[1])
    else:
        # Default mother directory (modify as needed)
        mother_dir = Path('/home/pieter/Dropbox (UMass Medical School)/Yeast_data_2023/dual_strain_label/BMY_1408/')
        print(f"No directory provided. Using default mother directory: {mother_dir}")

    if not mother_dir.is_dir():
        print(f"The specified mother directory does not exist or is not a directory: {mother_dir}")
        sys.exit(1)

    # Iterate over each subdirectory in the mother directory
    for subfolder in mother_dir.iterdir():
        if subfolder.is_dir():
            print(f"Processing subfolder: {subfolder}")
            organize_files(subfolder)
        else:
            print(f"Skipping non-directory item in mother directory: {subfolder}")

    print("File organization process completed.")


if __name__ == "__main__":
    main()
