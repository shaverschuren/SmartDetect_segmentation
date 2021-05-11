import sys
if "" not in sys.path: sys.path.append("")

import numpy as np
from PIL import Image
from preprocessing import intensity_normalization, reshape_image
from postprocessing import postprocess_mask


def predict_raw_image(in_path, out_path, model):
    """
    This function allows for the mask prediction of a raw
    (non-processed) image. It requires the (compiled) model
    and in- and out-paths of a certain subject.
    """

    # Load CXR image and extract original shape
    cxr_img = Image.open(in_path)
    cxr_array = np.array(cxr_img)

    ori_shape = np.shape(cxr_array)
    long_side = np.max(ori_shape)

    # Preprocess image
    cxr_array_int = intensity_normalization(cxr_array, im_type="cxr")
    cxr_img_pr = reshape_image(cxr_array_int)

    # Standardize image
    src_img = np.array(cxr_img_pr).astype(np.float32)
    src_img_sta = (src_img - np.mean(src_img)) / np.std(src_img)

    # Predict and reshape mask
    input = np.expand_dims(src_img_sta, axis=[0, 3])
    mask = model.predict(input)
    mask_arr = mask[0, :, :, 0]

    # Binarize mask
    mask_arr[mask_arr < 0.5] = 0
    mask_arr[mask_arr >= 0.5] = 1

    # Post-process mask
    mask_arr = postprocess_mask(mask_arr)

    # Reshape mask to original image size
    mask_img = Image.fromarray(mask_arr.astype(np.uint8))
    mask_img_res = mask_img.resize((long_side, long_side),
                                   resample=Image.BILINEAR)

    mask_arr_res = np.array(mask_img_res)

    res_shape = np.shape(mask_arr_res)

    cut_x = max(0, int(res_shape[0] - ori_shape[0]) // 2)
    cut_y = max(0, int(res_shape[1] - ori_shape[1]) // 2)

    mask_ori = mask_arr_res[cut_x:res_shape[0] - cut_x,
                            cut_y:res_shape[1] - cut_y]

    mask_img = Image.fromarray((mask_ori * 255).astype(np.uint8))

    # Save mask
    mask_img.save(out_path)
