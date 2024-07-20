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
from Dataset import *

batch_size = 32

pd.set_option('display.max_columns', None)

X = df_train["ID"]
y = df_train["Category"]

X_test = df_test["ID"]
y_test = df_test["Category"]

columns = df_train.columns


# Get list unique values
y_classes = np.unique(y)
count_classes = len(y_classes)


X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  random_state=42,
                                                  stratify=y,
                                                  train_size=0.75,
                                                  test_size=0.25,
                                                  )

train_files = pd.DataFrame(columns=columns)
train_files["ID"] = X_train
train_files["Category"] = y_train

val_files = pd.DataFrame(columns=columns)
val_files["ID"] = X_val
val_files["Category"] = y_val



train_datagen = ImageDataGenerator(rescale=1./255,)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_files,
    x_col="ID",
    y_col="Category",
    batch_size=batch_size,
    target_size=(128, 128),
    classes=dict_name_classes_one_hot_encoder
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_files,
    x_col="ID",
    y_col="Category",
    batch_size=batch_size,
    target_size=(128, 128),
    classes=dict_name_classes_one_hot_encoder
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    x_col="ID",
    y_col="Category",
    batch_size=batch_size,
    target_size=(128, 128),
    shuffle=True,
    classes=dict_name_classes_one_hot_encoder
)


if __name__ == "__main__":

    pass