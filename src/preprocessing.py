import sys
sys.path.append("")

import os
from glob import glob
from shutil import copyfile
import warnings
import numpy as np
from PIL import Image
from util.general import print_result, print_warning


def generate_paired_lists(cxr_paths, mask_paths, subset_n, split_masks=False):
    """
    This function is used for a few purposes.
    Firstly, it checks whether every image has an accessory mask.
    Secondly, it generates two sorted lists (with image pairs in the proper order)
    Finally, it also generates a list of filenames under which to store the preprocessed data.
    """

    cxr_sort = []
    mask_sort = []
    subject_names = []
    missing_masks = 0

    for subject_n in range(len(cxr_paths)):
        # Find CXR filename and look for matches in the mask list
        cxr_filename = os.path.split(cxr_paths[subject_n])[-1]

        if not split_masks:
            filename_matches = [mask_path for mask_path in mask_paths if os.path.splitext(cxr_filename)[0] in mask_path]
        else:
            filename_matches0 = [mask_path for mask_path in mask_paths[0] if os.path.splitext(cxr_filename)[0] in mask_path]
            filename_matches1 = [mask_path for mask_path in mask_paths[1] if os.path.splitext(cxr_filename)[0] in mask_path]

            if len(filename_matches0) == len(filename_matches1) == 1: 
                filename_matches = [[filename_matches0[0], filename_matches1[0]]]
            else:
                warnings.warn("Missing either an R or L mask. Omitting entire mask")
                filename_matches = []

        if type(filename_matches) == list and len(filename_matches) == 1:
            cxr_sort.append(cxr_paths[subject_n])
            subject_names.append("{:d}_{:03d}".format(subset_n, subject_n))

            if not split_masks:
                mask_sort.append(filename_matches[0])
            else:
                mask_sort.append([filename_matches[0][0], filename_matches[0][1]])
        elif type(filename_matches) == list and len(filename_matches) > 1:
            warnings.warn("Multiple matches found for a single subject name!")
        elif type(filename_matches) == list and len(filename_matches) == 0:
            missing_masks += 1
        else :
            raise ValueError("Parameter 'filename_matches' should return a list") 

    return cxr_sort, mask_sort, subject_names, missing_masks


def combine_masks(mask1_path, mask2_path):
    """
    This function combines two masks into one.
    It is primarily used to combine the seperate L/R masks of the Mntg dataset.
    """

    mask1_img = Image.open(mask1_path)
    mask2_img = Image.open(mask2_path)

    mask1_array = np.asarray(mask1_img)
    mask2_array = np.asarray(mask2_img)

    if np.shape(mask1_array) == np.shape(mask2_array):
        combined_mask = np.zeros(np.shape(mask1_array))
        combined_mask[mask1_array != False] = 255
        combined_mask[mask2_array != False] = 255
    else:
        raise ValueError("Masks to be combined aren't the same size")

    return combined_mask.astype(np.uint8)


def preprocessing(rawDir = os.path.join("data", "raw"), preprocessedDir = os.path.join("data", "preprocessed"), verbose = True):
    """
    Main function for data preprocessing. 
    It handles the file structure conversion and calls on functions to perform mask manipulation.
    """

    if verbose:
        print("--- Performing data preprocessing --- ")
        print("Extracting data from:\t{}".format(os.path.abspath(rawDir)))
        print("Outputting data to:\t{}\n".format(os.path.abspath(preprocessedDir)))

    # Predefine known raw data structure and wanted preprocessed data structure
    cxr_dirs = ["CXR_ChinaSet", "CXR_Manual", "CXR_Mntg"]
    mask_dirs = ["mask_ChinaSet", "masks_Manual", ["leftMask_Mntg", "rightMask_Mntg"]]

    new_imageDir = os.path.join(preprocessedDir, "cxr")
    new_maskDir = os.path.join(preprocessedDir, "masks")

    # If required, create new directories
    if not os.path.isdir(new_imageDir) : os.mkdir(new_imageDir)
    if not os.path.isdir(new_maskDir) : os.mkdir(new_maskDir)

    # Loop over sub-datasets and perform preprocessing
    for subset_n in range(len(cxr_dirs)):
        if verbose : print("Extracting data from subset '{:12s}' ({:1d}/{:1d})... ".format(cxr_dirs[subset_n], subset_n+1, len(cxr_dirs)), end="", flush=True)

        # Extract CXR images
        cxr_paths = glob(os.path.join(rawDir, cxr_dirs[subset_n], "*.png"))
        
        # Extract masks
        if type(mask_dirs[subset_n]) == str: 
            split_masks = False
            mask_paths = glob(os.path.join(rawDir, mask_dirs[subset_n], "*.png"))
        elif type(mask_dirs[subset_n]) == list:
            split_masks = True
            mask_paths = [[], []]
            mask_paths[0] = glob(os.path.join(rawDir, mask_dirs[subset_n][0], "*.png"))
            mask_paths[1] = glob(os.path.join(rawDir, mask_dirs[subset_n][1], "*.png"))
        else:
            raise ValueError("Incorrect format for mask directory paths")
        
        # Sort lists to ensure retention of image-mask pairs
        # Also, check whether there are as many masks as images
        cxr_sort, mask_sort, subject_names, missing_mask = generate_paired_lists(cxr_paths, mask_paths, subset_n, split_masks=split_masks)

        if verbose: print(f"(found {len(subject_names)} images) ", end="", flush=True)

        # Loop over cxr images and perform file structure conversion
        for subject_n in range(len(subject_names)):
            cxr_path = cxr_sort[subject_n]
            mask_path = mask_sort[subject_n]
            subject_name = subject_names[subject_n]

            # Perform cxr file restructuring
            copyfile(cxr_path, os.path.join(new_imageDir, subject_name + ".png"))

            # Perform mask file restructuring + mask combination (optional)
            if not split_masks:
                copyfile(mask_paths[subject_n], os.path.join(new_maskDir, subject_name + ".png"))
            else:
                combined_mask = combine_masks(mask_path[0], mask_path[1])
                mask_img = Image.fromarray(combined_mask)
                mask_img.save(os.path.join(new_maskDir, subject_name + ".png"))

        if verbose: 
            if missing_mask == 0:
                print_result(True)
            else:
                warning = f"Missed {missing_mask} mask files"
                print_warning(warning)

    return


if __name__ == "__main__":
    preprocessing()