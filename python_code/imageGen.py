"""
Created on Wed Nov 23 11:18:14 2022

@author: cfkaw
"""
import os
import datetime
import sqlite3
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from os.path import exists as file_exists
from datetime import datetime
from zipfile import ZipFile


def loadData(image_path, label_path):
    # this function loads the data from the hurricane.zip file
    # only load hurricane_image_train and hurricane_lable_train
    # as shown below
    xdata = np.load(image_path)
    ydata = np.load(label_path)
    return xdata, ydata


def imageGen(xdata, idx1, idx2, idx3):
    # this function takes the data from
    # loadData and generates the images
    x = xdata.reshape(3408, 10, 128, 257, 6)
    img_contour = plt.contourf(x[int(idx1), int(idx2), :, :, int(idx3)])
    return img_contour


def probMap(ydata, idx1, idx2):
    # this function generates the probablilty contour that
    # a hurricane is located in a specific pixel
    y = ydata.reshape(3408, 10, 128, 257, 1)
    prob_contour = plt.contourf(y[int(idx1), int(idx2), :, :, 0])
    return prob_contour


xData, yData = loadData("./hurricane_image_train.npy", "./hurricane_label_train.npy")
#%%
x = xData.reshape(3408, 10, 128, 257, 6)
y = yData.reshape(3408, 10, 128, 257, 1)
#%%
# define arrays of for each column from the x and y data
yProb = np.array(y[0, 0, 0, 0, :])
yIms = np.array(y[:, 0, 0, 0, 0])
yTimes = np.array(y[0, :, 0, 0, 0])
xParams = np.array(x[0, 0, 0, 0, :])
xIms = np.array(x[:, 0, 0, 0, 0])
xTimes = np.array(x[0, :, 0, 0, 0])
height = np.array(x[0, 0, :, 0, 0])
width = np.array(x[0, 0, 0, :, 0])
#%%

# define path to save hurricane images to
savePathh = "./Hurricane_images"
if os.path.exists(savePathh):
    print("file exists")
else:
    os.mkdir(savePathh)
for i in range(len(xIms[0:10])):
    for j in range(len(xTimes)):
        for k in range(len(xParams)):
            # date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
            imNameh = ("hurricane_image"+ "_"
                + "time"
                + "_"
                + str(j)
                + "_"
                + "image"
                + "_"
                + str(i)
                + "_"
                + "parameter"
                + "_"
                + str(k)
                + ".png"
            )
            completeNameh = os.path.join(savePathh, imNameh)
        # I = imageGen(x,xIms[i],xTimes[j],xParams[k])

#          plt.show
#           plt.savefig(completeNameh,dpi =1080)
#
#%%
# define path to save the probabilitys to
savePathp = "./Hurricane_probability"
if os.path.exists(savePathp):
    print("file exists")
else:
    os.mkdir(savePathp)
for i in range(len(yIms[0:10])):
    #  print(i)
    for j in range(len(yTimes)):

        imNamep = "hurricane_probability" + "_" + "parameter" + "_" + str(j) + "_" + "image" + "_" + str(i) + "_.png"
        completeNamep = os.path.join(savePathp, imNamep)
        I = probMap(y, yIms[i], yTimes[j])
        plt.show
        plt.savefig(completeNamep, dpi=600)
