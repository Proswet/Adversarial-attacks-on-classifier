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

import art


from IPython.display import clear_output, display_html, HTML
from skimage import io
import contextlib
import time
import io
import tarfile
import urllib
import os
import shutil
import pathlib
from pathlib import Path
import time
import re

import train_test_split_data
from Dataset import *
batch_size = train_test_split_data.batch_size

pd.set_option('display.max_columns', None)

path = "model_checkpoints"

model = keras.models.load_model(path+"/last_model.hdf5")
model.load_weights(path+"/best_model.hdf5")


losses_object = keras.losses.CategoricalCrossentropy()
optimizer = keras.optimizers.Adam()
classifier = art.estimators.classification.TensorFlowV2Classifier(
    model=model,
    nb_classes=102,
    input_shape=input_shape,
    loss_object=losses_object,
    optimizer=optimizer,
    clip_values=(0.0, 0.01)
)

FGSM = art.attacks.evasion.FastGradientMethod(estimator=classifier, eps=0.001)
CW = art.attacks.evasion.CarliniLInfMethod(classifier=classifier, batch_size=32)
BIM = art.attacks.evasion.BasicIterativeMethod(estimator=classifier)
JSMA = art.attacks.evasion.SaliencyMapMethod(classifier=classifier, batch_size=4)
one_pixel_attack = art.attacks.evasion.PixelAttack(classifier=classifier, th=None, targeted=False, verbose=True, es=1)
Deepfool = art.attacks.evasion.DeepFool(classifier=classifier)

#Create dir attack
name_attack = "DeepFool"
path_attack = path_attack + f"/{name_attack}"
if not os.path.exists(path_attack):
    os.mkdir(path_attack)
#else:  # Добавлять в том случае если прходиться продолжать вычисления с n-го батча после паузы
    ID = []
    Category = []

    df_attack = df





    count_batches = len(train_test_split_data.test_generator)

    dict_name_classes = train_test_split_data.test_generator.class_indices


    for times in range(count_batches): # times worked
        attack_image, one_hot_class = train_test_split_data.test_generator[times]
        attack_image = Deepfool.generate(attack_image)

        #Preparing name classes from one_hot to natural
        natural_class = one_hot_class.argmax(axis=1)+1




        len_batch = len(attack_image)
        for times_batch in range(len_batch):

            # Create directory with name classes in dir: path_attack
            name_class = natural_class[times_batch]
            new_dir_class = path_attack+f"/{name_class}"
            if not os.path.exists(new_dir_class):
                os.mkdir(new_dir_class)
            # List names files in directory_class
            img_in_dir = next(os.walk(new_dir_class))[2]
            #Get count img in dir
            count_img_in_dir = len(img_in_dir)
            #Define serial number for new image
            name_img = new_dir_class+f"/{count_img_in_dir+1}"+".jpeg"
            # Save new image
            keras.utils.save_img(name_img, attack_image[times_batch], file_format="jpeg", scale=True)

            # Delete directory with img
            #shutil.rmtree(path_attack)
