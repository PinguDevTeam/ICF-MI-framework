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



def load_data(image_path,label_path): 
    xdata=np.load(image_path) 
    ydata=np.load(label_path)
    return xdata, ydata

xdata, ydata = load_data("./hurricane_image_train.npy","./hurricane_label_train.npy")
X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.33, random_state=42)
