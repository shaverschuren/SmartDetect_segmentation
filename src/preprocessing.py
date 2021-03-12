import sys
sys.path.append("")

import os
from glob import glob
import warnings
import numpy as np
from PIL import Image
from util.general import *


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

    return combined_mask


def check_for_inversion(cxr_img_norm):
    """
    This function checks whether the image should be inverted based on the rim intensity.
    This is an important step of the normalization process.
    """

    # Define image rim
    rim_thickness = max(np.shape(cxr_img_norm)) // 20
    rim_array = [
        list(cxr_img_norm[:rim_thickness, :].flatten()), 
        list(cxr_img_norm[:, :rim_thickness].flatten()),
        list(cxr_img_norm[-rim_thickness:, :].flatten()),
        list(cxr_img_norm[:, -rim_thickness:].flatten())
        ]

    rim_list = [pixel for rim in rim_array for pixel in rim]

    # Compare mean of rim to mean of whole image
    img_mean = np.mean(cxr_img_norm)
    rim_mean = np.mean(np.array(rim_list))

    inversion_check = (rim_mean > img_mean) 

    return inversion_check


def intensity_normalization(img_as_np, im_type=""):
    """
    This function normalizes the intensities of images and masks.
    It also corrects for some images with 3 channels, and inverts the intensity range if necessary.
    """

    # Check for the dimensionality of the image. Some images have 3 color channels
    if np.ndim(img_as_np) == 2:
        pass
    elif np.ndim(img_as_np) == 3:
        img_as_np = np.mean(img_as_np, axis=2)
    else:
        raise ValueError("Image has an invalid number of dimensions")
    
    # Remove booleans
    new_img_as_np = np.zeros(np.shape(img_as_np))
    new_img_as_np[:, :] = img_as_np[:, :]
    new_img_as_np[img_as_np == False] = 0.

    # Rescale the image to the preferred intensity range.
    img_min = np.min(new_img_as_np)
    img_max = np.max(new_img_as_np)

    new_min = 0
    new_max = 255
    array_type = np.uint8

    img_as_np_corr = (((new_img_as_np - img_min) / img_max ) * (new_max - new_min)) + new_min

    # If applicable, invert the image
    if im_type.lower() == "cxr":
        if check_for_inversion(img_as_np_corr):
            # Perform inversion
            img_as_np_fin = new_max - img_as_np_corr + new_min
        else:
            img_as_np_fin = img_as_np_corr
    elif im_type.lower() == "mask":
        img_as_np_fin = img_as_np_corr
    else:
        raise ValueError("The 'im_type' parameter should be a string and either 'cxr' or 'mask'")

    return img_as_np_fin.astype(array_type)


def reshape_image(img_as_np, to_shape = (256, 256)):
    """
    This function pads and resamples an image. Output is in PIL Image format
    It is used for the data normalization step of the preprocessing process.
    """

    # Trim original image to have an even number of pixels (in both axes)
    if np.shape(img_as_np)[0] % 2 != 0:
        img_as_np = img_as_np[0:np.shape(img_as_np)[0]-1, :]
    if np.shape(img_as_np)[1] % 2 != 0:
        img_as_np = img_as_np[:, 0:np.shape(img_as_np)[1]-1]

    # Define old and intermediate shapes
    ori_shape = np.shape(img_as_np)
    int_shape = (max(ori_shape), max(ori_shape))
    x_pad, y_pad = (max([0, int_shape[0] - ori_shape[0]]) // 2, max([0, int_shape[1] - ori_shape[1]]) // 2)

    # Pad the image
    new_img_as_np = np.zeros(int_shape)
    new_img_as_np[x_pad:int_shape[0] - x_pad, y_pad:int_shape[0]- y_pad] = img_as_np[:, :]

    # Resample the image to the required size
    new_img = Image.fromarray(new_img_as_np.astype(np.uint8))
    new_img_res = new_img.resize(to_shape, resample=Image.BILINEAR)

    return new_img_res


def preprocess_subject(cxr_path, mask_path, split_masks=False):
    """
    This function performs several preprocessing steps on the input data.
    It returns the preprocessed images in PIL image format.
    """

    # Load CXR image
    cxr_img = Image.open(cxr_path)
    cxr_array = np.array(cxr_img)

    # Load mask image
    if split_masks:
        mask_array = combine_masks(mask_path[0], mask_path[1])
    else:
        mask_img = Image.open(mask_path)
        mask_array = np.asarray(mask_img)

    # Correct for intensities, datatypes etc. 
    cxr_array_int = intensity_normalization(cxr_array, im_type="cxr")
    mask_array_int = intensity_normalization(mask_array, im_type="mask")

    # Reshape images
    cxr_img_pr = reshape_image(cxr_array_int)
    mask_img_pr = reshape_image(mask_array_int)

    return cxr_img_pr, mask_img_pr


def preprocessing(rawDir = os.path.join("data", "raw"), preprocessedDir = os.path.join("data", "preprocessed"), verbose = True, rerun=False):
    """
    Main function for data preprocessing. 
    It handles the file structure conversion and calls on functions to perform mask/image manipulation.
    """

    # If applicable, print some info regarding the processing
    if verbose:
        print("--- Performing data preprocessing --- ")
        print("Extracting data from:\t{}".format(os.path.abspath(rawDir)))
        print("Outputting data to:\t{}".format(os.path.abspath(preprocessedDir)))

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
        if verbose : print("\nExtracting data from subset '{:12s}' ({:1d}/{:1d})... ".format(cxr_dirs[subset_n], subset_n+1, len(cxr_dirs)))
        
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

        # If applicable, initiate progress bar
        if verbose: 
            print(f"(found {len(subject_names)} images)")
            printProgressBar(0, len(subject_names), length = 50)

        # Loop over cxr images and perform file structure conversion
        for subject_n in range(len(subject_names)):
            cxr_path = cxr_sort[subject_n]
            mask_path = mask_sort[subject_n]
            subject_name = subject_names[subject_n]

            cxr_target = os.path.join(new_imageDir, subject_name + ".png")
            mask_target = os.path.join(new_maskDir, subject_name + ".png")

            if not rerun and os.path.exists(cxr_target) and os.path.exists(mask_target):
                pass
            else:
                cxr_img, mask_img = preprocess_subject(cxr_path, mask_path, split_masks=split_masks)

                cxr_img.save(cxr_target)
                mask_img.save(mask_target)

            if verbose : printProgressBar(subject_n + 1, len(subject_names), length = 50)

        # If applicable, print result
        if verbose: 
            print("\nResult: ", end="", flush=True)
            if missing_mask == 0:
                print_result(True)
            else:
                warning = f"Missed {missing_mask} mask files"
                print_warning(warning)

    return


if __name__ == "__main__":
    preprocessing(rerun=True)