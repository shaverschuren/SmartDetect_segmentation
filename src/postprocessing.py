import sys
if "" not in sys.path: sys.path.append("")

import numpy as np
import skimage.morphology as morph
from skimage import measure


def extract_lungs(mask_as_np):
    """
    Extract "blobs" with biggest area
    """

    labels_mask = np.array(measure.label(mask_as_np))
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[2:]:
            labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0

    labels_mask[labels_mask != 0] = 1

    return labels_mask


def postprocess_mask(mask_as_np):
    """
    This function postprocesses a single mask.
    It should be passed as a numpy array.
    Postprocessing is based on some morphological tricks:
    - An opening to seperate the lungs from false positives
    - Detection of the two largest "blobs"
    - Several small closings to fill in the gaps
    """

    # Initialize mask
    mask_ori = mask_as_np

    # Perform opening
    mask_open = morph.opening(mask_ori, morph.disk(5))

    # Detection of the two largest blobs
    mask_clean = extract_lungs(mask_open)

    # Several small closings to fill in the gaps
    mask_final = morph.closing(mask_clean, morph.disk(5))

    return mask_final
