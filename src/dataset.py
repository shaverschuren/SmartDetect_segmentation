import sys
if "" not in sys.path : sys.path.append("")

import os
from glob import glob
import numpy as np
import random
from PIL import Image
from util.general import *


def convert_img_format(src_img, tar_img):
    """
    This function converts the intensity range of the images from the range that's
    convenient for storing the images (8bit 0-255) to ranges that are actually helpful in the
    model training process.
    For the source image, we will standardize the images so that -std<>std = -1<>1
    For the target image, we will normalize the images so that no_lung<>lung = 0<>1 
    """

    # Convert data types to float
    src_img = src_img.astype(np.float32)
    tar_img = tar_img.astype(np.float16)

    # Standardize the source image
    src_img_sta = (src_img - np.mean(src_img)) / np.std(src_img)

    # Binarize the target image
    tar_img_bin = np.zeros(np.shape(tar_img))
    tar_img_bin[tar_img > 0.] = 1.

    return src_img_sta, tar_img_bin


def generate_dataset(dataDir, split_dataset=False, train_or_test="", split_factor=0.8, random_seed=1234, verbose=True):
    """
    This main function handles the generation of a dataset.
    It calls on several functions to generate data in the proper format.
    It then just stores everything in memory. This is very achievable since the entire 
    (processed) dataset is just over 90 MBi.
    """

    if verbose: 
        print(f"\n--- Performing data extraction ({train_or_test:s}) ---")
        print(f"\nExtracting data from {os.path.abspath(dataDir):s}")

    # Extracting images from preprocessed data directories
    img_paths = glob(os.path.join(dataDir, "cxr", "*.png"))
    mask_paths = glob(os.path.join(dataDir, "masks", "*.png"))

    # Briefly check whether preprocessing process went properly
    if len(img_paths) != len(mask_paths):
        raise ValueError("There aren't as many masks as images. Please remove the preprocessed folder and rerun preprocessing.py")
    
    # If applicable, apply shuffling to the path lists and select some for either the train or test dataset
    random.Random(random_seed).shuffle(img_paths)
    random.Random(random_seed).shuffle(mask_paths)

    if split_dataset:
        if train_or_test.lower() == "train":
            img_paths_shuffled = img_paths[:int(len(img_paths) * 0.8) + 1]
            mask_paths_shuffled = mask_paths[:int(len(img_paths) * 0.8) + 1]
        elif train_or_test.lower() == "test":
            img_paths_shuffled = img_paths[int(len(img_paths) * 0.8) + 1:]
            mask_paths_shuffled = mask_paths[int(len(img_paths) * 0.8) + 1:]
        else:
            raise ValueError("The value of parameter 'train_or_test' should be either 'train' or 'test'")
    else:
        img_paths_shuffled = img_paths
        mask_paths_shuffled = mask_paths
        
    n_subjects = len(img_paths_shuffled)

    # Predefine source and target arrays
    src_array = np.zeros((n_subjects, 256, 256))
    tar_array = np.zeros((n_subjects, 256, 256))

    # Loop over subjects and iteratively add images to the dataset arrays
    if verbose : printProgressBar(0, n_subjects, length=50)
    for subject_n in range(n_subjects):
        src_path = img_paths_shuffled[subject_n]
        tar_path = mask_paths_shuffled[subject_n]

        src_img = np.array(Image.open(src_path))
        tar_img = np.array(Image.open(tar_path))

        # We will need to transform the intensity ranges of these images for proper training
        # We will standardize the src image and create a binary 0,1 tar image
        src_img_sta, tar_img_bin = convert_img_format(src_img, tar_img)

        src_array[subject_n, :, :] = src_img_sta
        tar_array[subject_n, :, :] = tar_img_bin

        if verbose : printProgressBar(subject_n+1, n_subjects, length=50)

    if verbose: 
        dataset_size = (sys.getsizeof(src_array) + sys.getsizeof(tar_array)) / 10**9
        print(f"Completed. Dataset consists of {n_subjects} images and is {dataset_size:.2f} GB")

    return [src_array, tar_array]


if __name__ == "__main__":
    dataset = generate_dataset(os.path.join("data", "preprocessed"), True, "train")