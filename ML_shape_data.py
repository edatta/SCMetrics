#!/usr/bin/env python

# spinal_cord_classification.py performs several types of machine learning techniques to explore the relationship between spinal cord shape and clinical outcomes in MS

import nibabel as nib
import numpy as np
import glob
import os
import sys
import csv
import scipy.io
import glob
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestCentroid
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import confusion_matrix

# Visualize data with PCA
X_deformation = np.genfromtxt('deformation_data.csv',dtype='str', delimiter=',')
X_deformation = X_deformation[:,1:]
X_deformation = X_deformation.astype(float)
X_jacobian = np.genfromtxt('jacobian_data.csv',dtype='str', delimiter=',')
X_jacobian = X_jacobian[:,1:]
X_jacobian = X_jacobian.astype(float)
X_all = np.hstack((X_deformation, X_jacobian))
X_patients = X_all[32:,:]

Y = np.loadtxt('EDSS.csv',dtype='str',delimiter=',')
Y = Y[:,1:]
Y = Y.astype(float)
Y_disease_course = Y[:,1]
Y_disease_course_patients = (Y_disease_course>1.5)[32:]
Y_edss = Y[:,0]
Y_edss_severity = (Y_edss >= 3)
Y_patient = (Y_disease_course >0)

model = PCA(n_components=5)
X_new = model.fit_transform(X_all)
plt.scatter(X_new[:,0], X_new[:,1], c=Y_edss, cmap='hot')
plt.colorbar()
plt.show()
print(model.explained_variance_ratio_)

# Try centroid method
Y = Y_disease_course_patients[:,None]
X = X_new[32:,:]
YX = np.hstack((Y,X))
np.random.shuffle(YX)
X_shuffle = YX[:,1:]
Y_shuffle = YX[:,0]
Y_shuffle = np.squeeze(Y_shuffle)

halfway_pt = 65
Xtrain = X_shuffle[0:halfway_pt,:]
Ytrain = Y_shuffle[0:halfway_pt]
Xvalidate = X_shuffle[halfway_pt:,:]
Yvalidate = Y_shuffle[halfway_pt:]

model = NearestCentroid()
model.fit(Xtrain, Ytrain)
Ypredict = model.predict(Xvalidate)

confusion_matrix(Yvalidate, Ypredict)

accuracy = sum(np.equal(Ypredict,Yvalidate))/float(len(Yvalidate))
print(accuracy)

# Try SVM
iterations = 1000
confusion_percentages = np.zeros((2,2,iterations))
for i in range(0,iterations):
    Y = Y_edss_severity[:,None]
    X = X_all
    YX = np.hstack((Y,X))
    np.random.shuffle(YX)
    X_shuffle = YX[:,1:]
    Y_shuffle = YX[:,0]
    Y_shuffle = np.squeeze(Y_shuffle)

    halfway_pt = 80
    Xtrain = X_shuffle[0:halfway_pt,:]
    Ytrain = Y_shuffle[0:halfway_pt]
    Xvalidate = X_shuffle[halfway_pt:,:]
    Yvalidate = Y_shuffle[halfway_pt:]

    model = SVC()
    model.fit(Xtrain, Ytrain)
    Ypredict = model.predict(Xvalidate)

    confusion = confusion_matrix(Yvalidate, Ypredict)
    confusion_percentage = np.divide(confusion, np.sum(confusion,0).astype(float))
    confusion_percentages[:,:,i] = confusion_percentage

avg_confusion = np.mean(confusion_percentages, axis=2)
print(avg_confusion)

# Try PLS

iterations = 1000
confusion_percentages = np.zeros((2,2,iterations))
for i in range(0,iterations):
    Y = Y_edss_severity[:,None]
    X = X_all
    YX = np.hstack((Y,X))
    np.random.shuffle(YX)
    X_shuffle = YX[:,1:]
    Y_shuffle = YX[:,0]
    Y_shuffle = np.squeeze(Y_shuffle)

    halfway_pt = 80
    Xtrain = X_shuffle[0:halfway_pt,:]
    Ytrain = Y_shuffle[0:halfway_pt]
    Xvalidate = X_shuffle[halfway_pt:,:]
    Yvalidate = Y_shuffle[halfway_pt:]

    model = PLSRegression(n_components = 5)
    model.fit(Xtrain, Ytrain)
    Ypredict = model.predict(Xvalidate)
    Ypredict = np.squeeze(Ypredict)
    Ypredict = (Ypredict >= 0.5)
    Yvalidate = Yvalidate.astype(int)
    confusion = confusion_matrix(Yvalidate, Ypredict)
    confusion_percentage = np.divide(confusion, np.sum(confusion,0).astype(float))
    confusion_percentages[:,:,i] = confusion_percentage

    accuracy = sum(Ypredict == Yvalidate)/float(len(Yvalidate))

avg_confusion = np.mean(confusion_percentages, axis=2)
print(avg_confusion)
