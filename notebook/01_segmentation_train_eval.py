#!/usr/bin/env python
# coding: utf-8

# # 01_segmentation_train_eval.ipynb
# 
# This file performs the model training process of our Pix2Pix-based lung segmentation system.
# Additionally, it performs model evaluation based on a seperately defined test dataset.

# ### Part 1: Define directories, import required libraries and setup sesion

# In[1]:


# Change working directory to the root folder
import os, sys
if os.path.split(os.getcwd())[-1] != 'SmartDetect_segmentation':
    get_ipython().run_line_magic('cd', '..')
    sys.path.append("src")
    
    if os.path.split(os.getcwd())[-1] != 'SmartDetect_segmentation':
        raise UserError("Something went wrong in the directory reassignment!")


# In[2]:


# Perform required imports
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocessing
from dataset import generate_dataset
from model import define_discriminator, define_generator, define_gan
from training import *
from util.general import *
from util.inspection import *
from util.tf_session import *

# Setup GPU tensorflow session
n_gpus = setup_tf_gpu_session()


# In[3]:


# Define some basic directories to use in the rest of this program
dataDir = os.path.join("data", "preprocessed")
modelDir = "model"
notebookDir = "notebook"
logDir = "logs"


# ### Part 2: Perform data preprocessing
# Please note that since this part may take quite a long time, it is skipped on default if the software detects preprocessed data already.

# In[4]:


# Preprocessing the data. 
# To rerun the preprocessing, change 'rerun' to True

preprocessing(rerun=False)


# ### Part 3: Dataset generation
# The data is split into a training and test set per default.
# Validation is performed with the training set based on a later split.

# In[5]:


dataset_train = generate_dataset(dataDir, split_dataset=True, train_or_test='train')
dataset_test = generate_dataset(dataDir, split_dataset=True, train_or_test='test')


# ##### Visual data inspection
# Here, we will also briefly inspect the data we'll be training the model with.

# In[6]:


_, image_shape = inspect_dataset(dataset_train, 'train')
_, _ = inspect_dataset(dataset_test, 'test')


# ### Part 4: Model definition
# Here, we will define the GAN model we'll be using for the segmentation purposes.
# It is derived from the Pix2Pix model.

# In[7]:


image_shape = (image_shape[0], image_shape[1], 1)

g_model = define_generator(image_shape)
d_model = define_discriminator(image_shape)
gan_model = define_gan(g_model, d_model, image_shape)


# ### Part 5: Monitoring
# The training process is monitored via TensorBoard.
# The results will be displayed here by default. Note, however, that we may also monitor the process manually or after training time by opening tensorboard via the terminal as such:
# 
# `tensorboard --logdir "logs"`

# In[8]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir {logDir}')


# ### Part 6: Training
# Here, the actual training process is performed. 
# We may pass some hyperparameters in the 'train' function.

# In[9]:


train(d_model, g_model, gan_model, dataset_train) 


# ### Part 7: Evaluation
# Here, we perform the evaluation of our model, based on the previously defined test dataset.

# In[ ]:


# TODO: Implement evaluation.

