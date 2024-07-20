import random

import tensorflow as tf


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import keras
from keras import utils
from keras.utils import load_img, array_to_img, img_to_array
from keras import models
from keras import utils
from keras import layers
from keras import backend
from keras import callbacks

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
from pathlib import Path
import re

import PIL
from PIL import ImageOps, ImageFilter, Image, ImageShow

pd.set_option('display.max_columns', None)

input_shape = (128, 128, 3)

#The paths to the dataset

path_attack = "/home/st/PycharmProjects/pythonProject/original_dataset/attack/"
path_train = Path("/home/st/PycharmProjects/pythonProject/original_dataset/train")
path_val = Path("/home/st/PycharmProjects/pythonProject/original_dataset/valid")


#Create the pattern of the dataframe
df = pd.DataFrame(columns=["ID", "Category"])

df_train = df
df_test = df #data from valid


ID = []
Category = []


# Go through of list
for iter in path_train.iterdir():
    # Save number category
    category = iter.name
    # Go through of list every category
    for id in iter.iterdir():
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
    df_train = pd.concat([df_train, DataFrame_with_one_category])

ID = []
Category = []


for iter in path_val.iterdir():
    # Number of the class
    category = iter.name

    for id in iter.iterdir():
        # Adress of the image in dirr
        str_id = str(id)
        # List adresses
        ID.append(str_id)
    Category = [category]*len(ID)
    list_data = list(zip(ID, Category))
    ID = []
    DataFrame_with_one_category = pd.DataFrame(list_data, columns=["ID", "Category"])
    df_test = pd.concat([df_test, DataFrame_with_one_category])
    del DataFrame_with_one_category

# Get names of classes and transform in 2D array
one_hot_encoder = OneHotEncoder()
classes = np.array([i for i in range(1, 103)]).reshape(-1, 1)
# Transform 2D array to array with strings
classes_list = map(lambda x: str(int(x)), classes)
# EnCode names of classes to one hot encode
classes_one_hot = one_hot_encoder.fit(classes)
classes_one_hot_arr = classes_one_hot.transform(classes).toarray()
# Combine names classes string and names classes encode in dictionary
dict_name_classes_one_hot_encoder = dict(zip(list(classes_list), list(classes_one_hot_arr)))

if __name__ == "__main__":
    pass