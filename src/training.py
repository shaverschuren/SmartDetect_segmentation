import sys
if "" not in sys.path: sys.path.append("")

import tensorflow as tf
import os
import time
from tqdm import tqdm
from datetime import datetime
import numpy as np
import math
import warnings
from util.general import *
from util.inspection import *
from util.tf_session import *


class Logger(object):
    """
    This class generates a logger object for either scalars or images.
    We may use it to monitor the training process in TensorBoard.
    """
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def log_scalar(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def log_images(self, tag, images, step):
        with self.writer.as_default():
            tf.summary.image(tag, np.asarray(images), max_outputs=len(images),
                             step=step)
            self.writer.flush()


def generate_real_samples(dataset, n_samples, patch_shape, available_idx=None):
    """
    Function that selects a batch of random samples from a list of available
    ones, returns images and target.
    """
    # unpack dataset
    trainA, trainB = dataset

    # choose random instances
    if type(available_idx) in [list, np.ndarray]:
        ix = np.random.randint(0, len(available_idx), n_samples)
        img_idx = available_idx[ix]
    else:
        img_idx = np.random.randint(0, np.shape(trainA)[0], n_samples)
        ix = None

    # retrieve selected images
    X1, X2 = trainA[img_idx], trainB[img_idx]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, patch_shape, patch_shape, 1))

    return [X1, X2], y, ix


def generate_fake_samples(g_model, samples, patch_shape):
    """
    Function that generates a batch of images, returns images and targets.
    """
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


def summarize_performance(step, g_model, dataset_train, dataset_val, modelsDir,
                          logger, run, n_samples=3):
    """
    Function that logs some validation and training images, called at
    a certain interval. It logs the performance using tensorboard.
    """
    # select a sample of input images
    [X_realA_train, X_realB_train], _, _ = generate_real_samples(dataset_train,
                                                                 n_samples, 1)
    # generate a batch of fake samples
    X_fakeB_train, _ = generate_fake_samples(g_model, X_realA_train, 1)

    # select a sample of input images
    [X_realA_val, X_realB_val], _, _ = generate_real_samples(dataset_val,
                                                             n_samples, 1)
    # generate a batch of fake samples
    X_fakeB_val, _ = generate_fake_samples(g_model, X_realA_val, 1)

    # save the generator model
    filename = os.path.join(modelsDir, 'g_model_{:07d}.h5'.format((step + 1)))
    g_model.save(filename)
    print('>Saved model: {}'.format(filename))

    logger.log_images('run_{}_step{}_train'.format(run, step),
                      [X_realA_train[0], X_fakeB_train[0], X_realB_train[0]],
                      step)
    logger.log_images('run_{}_step{}_val'.format(run, step),
                      [X_realA_val[0], X_fakeB_val[0], X_realB_val[0]],
                      step)


def check_mae(g_model, dataset, n_samples=3):
    """
    Check the mae for some real and generated samples.
    """
    # select a sample of input images
    [X_realA, X_realB], _, _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)

    mae = np.sum(np.absolute((X_realB.astype(np.float32)
                              - X_fakeB.astype(np.float32)))) \
        / np.size(X_realB)

    return mae


def split_train_val(dataset, split_factor=0.8):
    """
    This function is used to split the training set into
    a separate training- and test-set.
    """

    dataset_size = np.shape(dataset)
    train_len = math.floor(dataset_size[1] * split_factor)
    val_len = math.floor(dataset_size[1] * (1 - split_factor))

    train_size = (dataset_size[0], train_len, dataset_size[2], dataset_size[3])
    val_size = (dataset_size[0], val_len, dataset_size[2], dataset_size[3])

    dataset_train = np.zeros(train_size)
    dataset_val = np.zeros(val_size)

    subject_indices = list(range(dataset_size[1]))
    random.shuffle(subject_indices)

    for i in range(train_len):
        subject_n = subject_indices[i]
        dataset_train[0][i] = dataset[0][subject_n]
        dataset_train[1][i] = dataset[1][subject_n]

    for j in range(val_len):
        subject_m = subject_indices[-(j + 1)]
        dataset_val[0][j] = dataset[0][subject_m]
        dataset_val[1][j] = dataset[1][subject_m]

    return dataset_train, dataset_val


def train(d_model, g_model, gan_model, dataset, n_epochs=1000, n_batch=4,
          early_stopping=True, patience=20):
    """
    This function performs the actual training of the GAN model.
    """
    # Extract current time for model/plot save files
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    modelsDir = os.path.join("model", f"run_{current_time}")
    os.mkdir(modelsDir)

    logsDir = os.path.join("logs", f"run_{current_time}")
    os.mkdir(logsDir)

    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]

    # Split training set in train and validation sets
    dataset_train, dataset_val = split_train_val(dataset)

    trainA_ori, trainB_ori = dataset_train
    valA, valB = dataset_val

    # Fix train- and validation-dataset dimensionality issues
    trainA = np.expand_dims(trainA_ori, axis=3)
    trainB = np.expand_dims(trainB_ori, axis=3)
    valA = np.expand_dims(valA, axis=3)
    valB = np.expand_dims(valB, axis=3)

    dataset_train = [trainA, trainB]
    dataset_val = [valA, valB]

    # calculate the number of batches per training epoch and total iterations
    bat_per_epo = int(len(trainA_ori) / n_batch)
    n_steps = bat_per_epo * n_epochs

    # Define loggers for losses, images and similarity metrics
    logger_g = Logger(os.path.join(logsDir, "gen"))
    logger_d1 = Logger(os.path.join(logsDir, "dis1"))
    logger_d2 = Logger(os.path.join(logsDir, "dis2"))
    logger_im = Logger(os.path.join(logsDir, "im"))
    logger_train = Logger(os.path.join(logsDir, "mae_train"))
    logger_val = Logger(os.path.join(logsDir, "mae_val"))
    logger_stopDelta = Logger(os.path.join(logsDir, "stop_delta"))

    # Initialize early stopping
    earlyStop_list = []

    # manually enumerate epochs
    i = 0
    for epoch in range(n_epochs):
        print("\n" + print_style.BOLD
              + f"Epoch {epoch+1}/{n_epochs}:"
              + print_style.END)

        # Create list of available indices
        # --> Images can't be used twice in the same epoch.
        available_idx = np.array(range(np.shape(dataset_train)[1]))

        # Wait for a sec (for verbose sync) and then start training
        time.sleep(1)
        for batch in tqdm(range(bat_per_epo), ascii=True):

            if len(available_idx) < n_batch:
                continue

            # select a batch of real samples
            [X_realA, X_realB], y_real, used_idx = \
                generate_real_samples(dataset_train, n_batch,
                                      n_patch, available_idx)
            np.delete(available_idx, used_idx)

            # generate a batch of fake samples
            X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)

            # Perform training on this batch (supress UserWarnings)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                # update discriminator for real samples
                d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
                # update discriminator for generated samples
                d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
                # update the generator
                g_loss, _, _, _ = \
                    gan_model.train_on_batch(X_realA,
                                             [y_real, X_realB, X_realB])

            # Store losses (tensorboard)
            if (i + 1) % (bat_per_epo // 50) == 0:
                logger_g.log_scalar('run_{}'.format(current_time), g_loss, i)
                logger_d1.log_scalar('run_{}'.format(current_time), d_loss1, i)
                logger_d2.log_scalar('run_{}'.format(current_time), d_loss2, i)

            # Store similarities (tensorboard)
            if (i + 1) % (bat_per_epo // 20) == 0:
                mae_train = check_mae(g_model, dataset_train, 3)
                mae_val = check_mae(g_model, dataset_val, 3)

                logger_train.log_scalar('run_{}'.format(current_time),
                                        mae_train, i)
                logger_val.log_scalar('run_{}'.format(current_time),
                                      mae_val, i)

            # Update step nr.
            i += 1

        # Summarize the performance of this epoch and store the model
        summarize_performance(i, g_model, dataset_train, dataset_val,
                              modelsDir, logger_im, current_time)

        # Check whether earlyStopping critereon is met (if applicable)
        if early_stopping:
            # Append val loss list
            mae_val = check_mae(g_model, dataset_val, 10)
            earlyStop_list = earlyStop_list + [mae_val]

            print(f">Validation MAE = {mae_val:.6f}")

            # Calculate moving average size
            n_avg = math.ceil(patience / 10) if patience / 10 > 3. else 3

            # # Trim list (if applicable)
            while len(earlyStop_list) > patience + 1 + n_avg // 2:
                earlyStop_list = earlyStop_list[1:]

            # Check for criterion
            if len(earlyStop_list) < patience:
                pass
            else:
                # Stop if all "new" values are worse than the previous
                # epochs. Perform a rolling average for smoothing.
                earlyStop_avgs = \
                    np.convolve(np.array(earlyStop_list),
                                np.ones(n_avg), 'valid') / n_avg

                stop_criterion = (earlyStop_avgs[0]
                                  < earlyStop_avgs[1:]).all()

                stop_delta = \
                    np.max(earlyStop_avgs[0] - earlyStop_avgs[1:])

                logger_stopDelta.log_scalar('run_{}'.format(current_time),
                                            stop_delta, i)

                if stop_criterion:
                    print(f"\n>Stopping criterion met "
                          f"(patience = {len(earlyStop_list) - 1})."
                          f"\n>Exiting training with MAE of {mae_val:.6f} "
                          f"and a 'best' avg MAE of {earlyStop_avgs[0]:.6f}")
                    break

    return current_time
