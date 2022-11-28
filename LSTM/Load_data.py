import os
import datetime

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



def loadData(image_path,label_path): 
    xData=np.load(image_path) 
    yData=np.load(label_path)
    return xData, yData

xData, yData = load_data("./hurricane_image_train.npy","./hurricane_label_train.npy")
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.33, random_state=42)
