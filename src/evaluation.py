import sys
if "" not in sys.path: sys.path.append("")

import os
import numpy as np
from glob import glob
from model import negDSC
from PIL import Image
import matplotlib.pyplot as plt


def evaluate_masks(predDir, tarDir):
    """
    This function evaluates masks from files.
    It requires the prediction directory and target directory,
    and assumes that mask-pairs have the same file-name.
    """

    # Initialize lists
    prediction_list = glob(os.path.join(predDir, "*.png"))
    target_list = [pred.replace(predDir, tarDir) for pred in prediction_list]

    acc_list = []
    dsc_list = []
    mae_list = []
    # Loop over mask pairs
    for i in range(len(prediction_list)):
        pred_path = prediction_list[i]
        tar_path = target_list[i]

        # Read masks
        pred_arr = np.array(Image.open(pred_path))
        tar_arr = np.array(Image.open(tar_path))

        # Normalize masks
        pred_arr = pred_arr.astype('float') / np.max(pred_arr)
        tar_arr = tar_arr.astype('float') / np.max(tar_arr)

        acc = np.sum(pred_arr == tar_arr) / np.size(pred_arr)
        dsc = 1 - negDSC(pred_arr, tar_arr)
        mae = np.sum(np.absolute((pred_arr - tar_arr))) / np.size(pred_arr)

        acc_list.append(acc)
        dsc_list.append(dsc)
        mae_list.append(mae)

    acc_mu, acc_std = (np.mean(acc_list), np.std(acc_list))
    dsc_mu, dsc_std = (np.mean(dsc_list), np.std(dsc_list))
    mae_mu, mae_std = (np.mean(mae_list), np.std(mae_list))

    print(f"\n--- Evaluation metrics ---\n"
          f"\nAccuracy = {acc_mu:.4f} +/- {acc_std:.4f}"
          f"\nDSC      = {dsc_mu:.4f} +/- {dsc_std:.4f}"
          f"\nMAE      = {mae_mu:.4f} +/- {mae_std:.4f}")


def evaluate_model(dataset_test, loaded_model):
    X_test, Y_test = dataset_test
    X_test = np.expand_dims(X_test, axis=3)
    Y_test = np.expand_dims(Y_test, axis=3)
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam',
                         metrics=['accuracy', negDSC, 'mae'])

    scores = loaded_model.evaluate(X_test, Y_test, verbose=0)

    for i in range(len(scores)):
        if loaded_model.metrics_names[i] == "loss":
            pass
        elif loaded_model.metrics_names[i] == "negDSC":
            print(f"DSC       : {1 - scores[i]:.4f}")
        else:
            print(f"{loaded_model.metrics_names[i]:10s}: {scores[i]:.4f}")


def predict_dataset(dataset, model):
    X_test, _ = dataset
    X_test = np.expand_dims(X_test, axis=3)

    predicted = model.predict(X_test)
    predicted_images = predicted[:, :, :, 0]
    return predicted_images


def predict_and_plot_image(image, model):
    img = np.expand_dims(image, axis=[0, 3])
    pred = model.predict(img)
    pred_img = pred[0, :, :, 0]
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(pred_img, cmap='gray')
    plt.axis('off')
    plt.show()


def dice(input_img, test_img):
    k = 1
    dice = np.sum(input_img[test_img == k]) * 2.0 \
        / (np.sum(input_img) + np.sum(test_img))
    print('Dice similarity score is {}'.format(dice))


if __name__ == "__main__":
    predDir = os.path.join("data", "test", "pred")
    tarDir = os.path.join("data", "test", "tar")

    evaluate_masks(predDir, tarDir)
