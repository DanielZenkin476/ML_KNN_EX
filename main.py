#this project is based upon intro to deep learning

#lazy alg - no training req.
# good with small datasets
#parameters needed are K and distance func

#classification
#imports:

import random
import numpy as np
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Random seed
seed = 100
np.random.seed(seed)
random.seed(seed)

#data - using iris database

iris = datasets.load_iris()
data = pd.DataFrame({'sepal_length': iris.data[:,0],
                     'sepal_width': iris.data[:,1],
                     'petal_length': iris.data[:,2],
                     'petal_width': iris.data[:,3],
                     'type': iris.target})
print(data.head(5))# show first 5 entries in data


