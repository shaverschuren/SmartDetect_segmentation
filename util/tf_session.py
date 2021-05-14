# Here, we define general functions that may be useful for
# setting up tensorflow sessions.

import tensorflow as tf


def setup_tf_gpu_session():
    """
    Function that fixes memory problems for gpu runs
    """

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    return len(gpus)
