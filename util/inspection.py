import numpy as np
import matplotlib.pyplot as plt
import random
from util.general import print_style

def inspect_dataset(dataset, dataset_name, verbose=True):
    """
    This function performs some basic dataset inspection.
    It displays some info about the size of the dataset.
    Also, it creates a figure with some image examples.
    """

    print(print_style.BOLD + "--- Performing data inspection for dataset '{:s}' ---\n".format(dataset_name) + print_style.END)

    dataset_size = np.shape(dataset)
    n_images = dataset_size[1]
    im_shape = dataset_size[2:]

    print("Amount of subjects:\t{:d}".format(n_images))
    print("Image size:\t\t{:d}, {:d}".format(im_shape[0], im_shape[1]))

    X = dataset[0]
    y = dataset[1]

    plt.figure(figsize=(20, 10))
    plt.title('Visual inspection for dataset \'{:s}\''.format(dataset_name))

    indices = list(range(len(X)))
    random.shuffle(indices)

    for i in range(1, 5):
        x_nr = i*2 - 1
        y_nr = i*2

        plt.subplot(2, 4, x_nr)
        plt.imshow(X[indices[i]], cmap='gray')
        plt.axis('off')

        plt.subplot(2, 4, y_nr)
        plt.imshow(y[indices[i]], cmap='gray')
        plt.axis('off')

    plt.show()

    return