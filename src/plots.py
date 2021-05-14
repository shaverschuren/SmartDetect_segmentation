import sys
if "" not in sys.path: sys.path.append("")

import os
from glob import glob
import warnings
import numpy as np
from PIL import Image, ImageFilter


def drawContour(m, s, RGB):
    """
    Draw edges of contour from segmented image 's' onto 'm' in colour
    'RGB'
    """
    # Fill contour "c" with white, make all else black
    thisContour = s.point(lambda p: p > 1e-5 and 255)
    # DEBUG: thisContour.save(f"interim{c}.png")

    # Find edges of this contour and make into Numpy array
    thisEdges = thisContour.filter(ImageFilter.FIND_EDGES)
    thisEdgesN = np.array(thisEdges)

    # Paint locations of found edges in color "RGB" onto "main"
    m[np.nonzero(thisEdgesN)] = RGB
    return m


def drawFill(m, s, RGB):
    """
    Draw fill from segmented image 's' onto 'm' in colour 'RGB'
    """
    # Fill contour "c" with white, make all else black
    thisContour = s.point(lambda p: p > 1e-5 and 255)
    thisContourN = np.array(thisContour)

    # Find color addition
    addition = [0, 0, 0]
    for i in range(3):
        addition[i] = 0.5 * RGB[i]

    additionT = tuple(addition)

    # Paint locations of found edges in color "RGB" onto "main"
    m[np.nonzero(thisContourN)] = m[np.nonzero(thisContourN)] * 0.5 + additionT
    return m


def create_subject_plot(src_path, pred_path, tar_path, out_path):
    """
    This function creates overlay plots used for the
    paper. It simply takes the original image, overlays
    the prediction and target masks and then stores the
    image as a png.
    """

    src_img = Image.open(src_path).convert('L').convert('RGB')
    tar_img = Image.open(tar_path).convert('L')
    pred_img = Image.open(pred_path).convert('L')

    src_img.resize((2048, 2048), resample=Image.BILINEAR)
    tar_img.resize((2048, 2048), resample=Image.BILINEAR)
    pred_img.resize((2048, 2048), resample=Image.BILINEAR)

    main = np.array(src_img)

    main = drawFill(main, pred_img, (180, 0, 0))
    main = drawContour(main, tar_img, (0, 0, 255))

    Image.fromarray(main).save(out_path)


def plot_subjects(testDir, subject_list):
    """
    This function creates plots for a list of subjects.
    """

    if not os.path.isdir(os.path.join(testDir, "plots")):
        os.mkdir(os.path.join(testDir, "plots"))

    for subject in subject_list:
        src_path = os.path.join(testDir, "src", subject + ".png")
        tar_path = os.path.join(testDir, "tar", subject + ".png")
        pred_path = os.path.join(testDir, "pred", subject + ".png")
        out_path = os.path.join(testDir, "plots", subject + ".png")

        create_subject_plot(src_path, pred_path, tar_path, out_path)


if __name__ == "__main__":
    plot_subjects(os.path.join("data", "test"),
                  ["0_005", "0_007", "2_104",   # "easy" examples (all 3 sets)
                   "0_431", "0_474", "1_003",   # "hard" pathology examples
                   "0_073", "2_059", "0_154"])  # "hard" non-pathology examples
