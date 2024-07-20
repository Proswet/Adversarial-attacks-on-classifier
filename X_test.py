import random

import tensorflow as tf
from tensorflow import keras


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import keras

from keras.utils import load_img, array_to_img, img_to_array
import keras.models
import keras.layers as layers
import keras.backend as backend
import keras.callbacks as callbacks

from keras import Sequential
from keras.layers import Dense
from keras.backend import clear_session

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.callbacks import Callback

from keras import optimizers

from IPython.display import clear_output, display_html, HTML
from skimage import io
import contextlib
import time
import io
import tarfile
import urllib
import os
import pathlib
from pathlib import Path
import re
import sys
import art

from Dataset import *
from train_test_split_data import *

tf.compat.v1.disable_eager_execution()

path = "model_checkpoints"

model = keras.models.load_model(path+"/last_model.hdf5")
model.load_weights(path+"/best_model.hdf5")

model.summary()

test_datagen.flow_from_dataframe(
    dataframe=X_test,

)

print(X_test, y_test)