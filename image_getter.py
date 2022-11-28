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

def imageGetter(xdata,idx1,idx2,idx3):
    x = xdata.reshape(3408,10,128,257,6)
    img_contour = plt.contourf(x[int(idx1),int(idx2),:,:,int(idx3)])
    return imgContour

