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

import train_test_split_data
from Dataset import *
import create_attack_data

path_attack_name = create_attack_data.name_attack

batch_size = train_test_split_data.batch_size

pd.set_option('display.max_columns', None)


ID = []
Category = []
df_attack = df
path_attack = path_attack+path_attack_name
path_attack = Path(path_attack)

# Go through of list
for iter in path_attack.iterdir():
    # Save number category
    category = iter.name
    # Go through of list every category
    for id in iter.iterdir():
        # Adding ".jpg" on image
        #os.rename(str_id, str_id+".jpg")

        # Save adress every elements in it category
        str_id = str(id)
        # And him save
        ID.append(str_id)

    # Create two lists for dataframe
    Category = [category]*len(ID)
    list_data = list(zip(ID, Category))
    # Delete list in variable ID
    ID = []
    # Create dataframe with data of every category
    DataFrame_with_one_category = pd.DataFrame(list_data, columns=["ID", "Category"])
    # Concat dataframe with one category to general
    df_attack = pd.concat([df_attack, DataFrame_with_one_category])


attack_datagen = ImageDataGenerator(rescale=1./255)

attack_generator = attack_datagen.flow_from_dataframe(
    dataframe=df_attack,
    x_col="ID",
    y_col="Category",
    batch_size=batch_size,
    target_size=(128, 128),
    shuffle=True,
    classes=dict_name_classes_one_hot_encoder
)









if __name__ == "__main__":

    pass