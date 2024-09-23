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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
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

X,Y = iris['data'],iris['target']
print(f'Shapes of data: X: {X.shape},Y: {Y.shape}')
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.15, random_state=100)# split into test and train
X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train,test_size=0.15, random_state=100)# split test into test and validation

#help functions:

def most_common(lst):
    #finds and returns most common item in the list
    return max(set(lst),key = lst.count)

def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))

# KNN CLASS

class KNClassifier:
    def __init__(self, k=5, dist_metric = euclidean):
        self.k = k
        self.dist_metric = dist_metric

    def fit(self,X_train,Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self,X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x,self.X_train)# calc dist of point x to all X_train points
            y_sorted = [y for _, y in sorted(zip(distances,self.Y_train))]
            #zip(distances, self.Y_train) creates pairs where each training point’s distance from x is paired with the corresponding label from Y_train.
            neighbors.append(y_sorted[:self.k])
        return list(map(most_common,neighbors))
    def eval(self,X_test,Y_test):
        y_pred = self.predict(X_test)
        acc = sum(y_pred == Y_test) / len(y_pred)
        return acc, y_pred

# now that the model is complete , to test it out and find hyperparameter k

accs = []
preds = []
ks = range(1,30)
for k in ks:
    knn = KNClassifier(k=k)
    knn.fit(X_train,Y_train)
    accuracy ,pred = knn.eval(X_val,Y_val)# validating data first
    accs.append(accuracy)
    preds.append(pred)

#now to visualize accuracy over k's:

fig, ax = plt.subplots()
ax.scatter(ks,accs)
ax.set(xlabel='K',ylabel='Accuracy')
plt.xticks(ks[::4])
plt.show()
#we can see that 8 is the lowest k with 1.0 accuracy
k=8
knn = KNClassifier(k=k)
knn.fit(X_train,Y_train)
accuracy ,pred =knn.eval(X_test,Y_test)
print(f'model accuracy is : {accuracy}')


# visualization over whole dataset:

# Create color maps for 3-class classification problem (iris)
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

#geting data
X= iris.data[:,:2]# only first 2 features
Y = iris.target

#model
knn = KNClassifier(k=8)
knn.fit(X,Y)
# get min and max of data
x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                      np.linspace(y_min, y_max, 100))

Z = np.array(knn.predict(np.c_[xx.ravel(), yy.ravel()]))

# Put the result into a color plot- the colored area in backround - this is the "model prediction", right subplot
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
pl.pcolormesh(xx, yy, Z.reshape(xx.shape), cmap=cmap_light)

# plot the dataset itself - the bold points
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
pl.xlabel('sepal length (cm)')
pl.ylabel('sepal width (cm)')
pl.axis('tight')
pl.title('train')

# Put the result into a color plot - new subplot to show training points, left subplot
plt.subplot(1, 2, 2)
pl.pcolormesh(xx, yy, Z.reshape(xx.shape), cmap=cmap_light)

# Plot training points- the bold points
pl.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap=cmap_bold)
pl.xlabel('sepal length (cm)')
pl.ylabel('sepal width (cm)')
pl.axis('tight')
pl.title('test')
pl.show()