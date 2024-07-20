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
from train_test_split_data import *
from base_model import *

pd.set_option('display.max_columns', None)

base_model = base_model


model = models.Sequential()

model.add(base_model)

model.add(layers.GlobalAveragePooling2D(),)


model.add(layers.Dense(102, activation="relu"))

model.add(layers.Dense(count_classes, activation="softmax"))


base_model.summary()
model.summary()

model.compile(
              optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"]
              )

#
checkpoint = keras.callbacks.ModelCheckpoint(
                                             "model_checkpoints/best_model.hdf5",
                                             monitor="val_loss",
                                             verbose=1,
                                             mode="max"
                                             )
earlystop = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)
callbacks_list = [checkpoint, earlystop]


history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples//train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples//val_generator.batch_size,
    epochs=15,
    callbacks=callbacks_list
)

# To save model
model.save("model_checkpoints/last_model.hdf5")
model.load_weights("model_checkpoints/best_model.hdf5")



scores = model.evaluate(test_generator, verbose=1)
print(f"Accuracy: {scores[1]*100} %")

if __name__ == "__main__":
    pass