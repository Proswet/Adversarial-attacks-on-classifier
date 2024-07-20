import random

import tensorflow as tf
from tensorflow import keras


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import keras

from keras.utils import load_img, array_to_img, img_to_array
import keras.models as models
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

import create_attack_data
import Dataset_attack
from train_test_split_data import *


path = "model_checkpoints"

model = keras.models.load_model(path+"/last_model.hdf5")
model.load_weights(path+"/best_model.hdf5")

scores_check = model.evaluate(Dataset_attack.attack_generator)
print("Accuracy attack_model:  ", scores_check[1]*100, " %")




if __name__ == "__main__":
    pass