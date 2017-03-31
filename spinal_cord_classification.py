#!/usr/bin/env python

# spinal_cord_classification.py performs several types of machine learning techniques to explore the relationship between spinal cord metrics and clinical outcomes in MS
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
from sklearn.svm import LinearSVC, SVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = np.nan_to_num(cm)
    dim = len(cm)
    targets = np.linspace(0,dim,dim+1)
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=np.min(cm), vmax=np.max(cm))
    plt.title(title)
    tick_marks = np.arange(dim)
    plt.xticks(tick_marks, targets, rotation=45)
    plt.yticks(tick_marks, targets)
    plt.tight_layout()
    rounded = np.around(cm, 2)
    rounded = np.nan_to_num(rounded)
    for i in range(0,dim):
        for j in range(0,dim):
            plt.text(i,j,rounded[j,i], fontsize=18)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.colorbar()
    plt.show()

# Read in all data
X_gradient = np.genfromtxt('gradient_data.csv',dtype='str', delimiter=',')
X_gradient = X_gradient[:,1:]
X_gradient = X_gradient.astype(float)
X_intensity = np.genfromtxt('intensity_data.csv',dtype='str', delimiter=',')
X_intensity = X_intensity[:,1:]
X_intensity = X_intensity.astype(float)
X_image = np.hstack((X_gradient, X_intensity))
X_image_all = X_image
X_image_patients = X_image[20:,:]

X_deformation = np.genfromtxt('deformation_data.csv',dtype='str', delimiter=',')
control_ids = X_deformation[:32,0].astype(int)
X_deformation = X_deformation[:,1:]
X_deformation = X_deformation.astype(float)
X_jacobian = np.genfromtxt('jacobian_data.csv',dtype='str', delimiter=',')
X_jacobian = X_jacobian[:,1:]
X_jacobian = X_jacobian.astype(float)
X_shape = np.hstack((X_deformation, X_jacobian))

X_shape_controls = X_shape[0:32,:]
X_shape_common_controls = X_shape_controls[control_ids<=20]
X_shape_patients = X_shape[32:,:]
X_shape_all = np.vstack((X_shape_common_controls, X_shape_patients))

X_volume = np.genfromtxt('volumes.csv', dtype='str', delimiter=',')
X_volume = X_volume[:,1:]
X_volume = X_volume.astype(int)
X_volume_patients = X_volume[20:,:]

X_patients = np.hstack((X_volume_patients, X_shape_patients, X_image_patients))
X_all = np.hstack((X_volume, X_shape_all, X_image_all))

X_patients_centered = X_patients - np.mean(X_patients, axis=0)
X_patients_normalize = np.divide(X_patients_centered, np.std(X_patients_centered, axis=0))

X_all_centered = X_all - np.mean(X_all, axis=0)
X_all_normalize = np.divide(X_all_centered, np.std(X_all_centered, axis=0))

Y = np.loadtxt('EDSS_gradient.csv',dtype='str',delimiter=',')

Y = Y[:,1:]
Y = Y.astype(float)
Y_disease_course = Y[:,1]
Y_disease_course_patients = (Y_disease_course>1.5)
Y_edss = Y[:,0]
Y_edss_int = Y_edss.astype(int)
Y_edss_severity = (Y_edss >= 3.0)
Y_patient = (Y_disease_course >0)

# Visualize data with PCA
model = PCA(n_components=100)
X_new = model.fit_transform(X_all_normalize)
plt.scatter(X_new[:,0], X_new[:,1], c=Y_edss_severity, cmap='hot')
plt.colorbar()
plt.title('Deformation, Jacobian, Intensity, and Gradient PCA')
plt.xlabel('Model Explained Variance = '+str(model.explained_variance_ratio_))
plt.show()
print(model.explained_variance_ratio_)

# Linear Regression
Y = Y_edss[:,None]
X = X_new

id = np.arange(0,len(Y))
id = id[:,None]
all_Ypredictions = np.zeros((len(Y),1000))
all_Ypredictions[:,:] = np.NAN

for i in range(0, 1000):

    idYX = np.hstack((id,Y,X))
    idYX = np.squeeze(idYX)
    np.random.shuffle(idYX)
    id_shuffle = idYX[:,0].astype(int)
    X_shuffle = idYX[:,2:]
    Y_shuffle = idYX[:,1]
    Y_shuffle = np.squeeze(Y_shuffle)

    halfway_pt = len(Y)/2
    Xtrain = X_shuffle[0:halfway_pt,:]
    Ytrain = Y_shuffle[0:halfway_pt]
    Xvalidate = X_shuffle[halfway_pt:,:]
    Yvalidate = Y_shuffle[halfway_pt:]
    idvalidate = id_shuffle[halfway_pt:]    

    model = LinearRegression()
    model.fit(Xtrain, Ytrain)
    Ypredict = model.predict(Xvalidate)

    all_Ypredictions[idvalidate,i] = Ypredict

avg_Yprediction = np.nanmean(all_Ypredictions, axis=1)
plt.scatter(Y, avg_Yprediction)
plt.xlim(0,10)
plt.ylim(0,10)
plt.title('Predicted vs. Actual EDSS Scores for Linear Regression')
plt.xlabel('Actual EDSS Score')
plt.ylabel('Average Predicted EDSS Score')
plt.plot([0,10], [0,10], color='k', linestyle='-', linewidth=2)
plt.show()

# Try centroid method
iterations = 1000
confusions = np.zeros((2,2,iterations))
confusion_percentages = np.zeros((2,2,iterations))
accuracies = np.zeros((1, iterations))

for i in range(0,iterations):

    Y = Y_edss_severity[:,None]
    X = X_all_normalize
    YX = np.hstack((Y,X))
    np.random.shuffle(YX)
    X_shuffle = YX[:,1:]
    Y_shuffle = YX[:,0]
    Y_shuffle = np.squeeze(Y_shuffle)

    halfway_pt = len(Y)/2
    Xtrain = X_shuffle[0:halfway_pt,:]
    Ytrain = Y_shuffle[0:halfway_pt]
    Xvalidate = X_shuffle[halfway_pt:,:]
    Yvalidate = Y_shuffle[halfway_pt:]

    model = NearestCentroid()
    model.fit(Xtrain, Ytrain)
    Ypredict = model.predict(Xvalidate)

    Ypredict = (Ypredict >= 0.5)
    Ypredict = Ypredict.astype(int)
    Yvalidate = Yvalidate.astype(int)
    confusion = confusion_matrix(Yvalidate, Ypredict)
    confusion_percentage = np.divide(confusion, np.sum(confusion,1).astype(float))
    confusions[:,:,i] = confusion
    confusion_percentages[:,:,i] = confusion_percentage
    accuracies[0,i] = sum(Ypredict == Yvalidate)/float(len(Yvalidate))

sum_confusion = np.sum(confusions, axis=2)
avg_confusion_percentage = np.mean(confusion_percentages, axis=2)
total_sums = (np.sum(sum_confusion,1)).astype(float)
avg_confusion = np.divide(sum_confusion, total_sums[:,None])
avg_accuracy = np.mean(accuracies)
plot_confusion_matrix(avg_confusion, title='Centroid Method Confusion Matrix - Classifying EDSS >= 3')
plot_confusion_matrix(avg_confusion_percentage, title='Centroid Method Confusion Matrix - Classifying EDSS >= 3')

# Try SVM - Modify for multiclass or LinearSVM
iterations = 100
accuracies = np.zeros((1, iterations))
c_values = [1e-5, 1e-3, 1e-1, 1, 1e3, 1e5]
Y = Y_edss_int[:,None]
X = X_all_normalize
YX = np.hstack((Y,X))
cm_labels = np.unique(Y)
confusions = np.zeros((9,9,iterations))
confusion_percentages = np.zeros((9,9,iterations))
accuracies = np.zeros((1, iterations))

for i in range(0,iterations):
    print(i)
    np.random.shuffle(YX)
    X_shuffle = YX[:,1:]
    Y_shuffle = YX[:,0]
    Y_shuffle = np.squeeze(Y_shuffle)

    thirds = len(Y)/3
    Xtrain = X_shuffle[0:thirds,:]
    Ytrain = Y_shuffle[0:thirds]
    Xvalidate = X_shuffle[thirds+1:2*thirds,:]
    Yvalidate = Y_shuffle[thirds+1:2*thirds]
    Xtest = X_shuffle[(2*thirds)+1:,:]
    Ytest = Y_shuffle[(2*thirds)+1:]

    # Tune hyperparameters using Xvalidate
    confusion_scores = np.zeros((1,len(c_values)))
    j=0
    for c_value in c_values:
        model = LinearSVC(C=c_value)
        model.fit(Xtrain, Ytrain)
        Ypredict = model.predict(Xvalidate)

        Ypredict = Ypredict.astype(int)
        #Ypredict = (Ypredict >= 0.5)
        Yvalidate = Yvalidate.astype(int)
        confusion = confusion_matrix(Yvalidate, Ypredict)
        confusion_scores[0,j] = confusion[0,0] + confusion[1,1]
        j=j+1
    max_index = np.argmax(confusion_scores)
    best_c = c_values[max_index]

    # Test using Xtest
    model = LinearSVC(C=best_c)
    model.fit(Xtrain, Ytrain)
    Ypredict = model.predict(Xvalidate)

    Ypredict = Ypredict.astype(int)
    #Ypredict = (Ypredict >= 0.5)
    Yvalidate = Yvalidate.astype(int)
    confusion = confusion_matrix(Yvalidate, Ypredict,labels=cm_labels)
    confusions[:,:,i] = confusion

    accuracies[0,i] = sum(Ypredict == Yvalidate)/float(len(Yvalidate))

sum_confusion = np.sum(confusions, axis=2)
total_sums = (np.sum(sum_confusion,1)).astype(float)
avg_confusion = np.divide(sum_confusion, total_sums[:,None])
avg_accuracy = np.mean(accuracies)
plot_confusion_matrix(avg_confusion, title='Linear SVM Confusion Matrix')
print(avg_confusion)

# Try Random Forest
iterations = 1000
confusion_percentages = np.zeros((2,2,iterations))
accuracies = np.zeros((1, iterations))
num_tree_array = [10, 20, 30, 40]

for i in range(0,iterations):
    Y = Y_edss_severity[:,None]
    X = X_all_normalize
    YX = np.hstack((Y,X))
    np.random.shuffle(YX)
    X_shuffle = YX[:,1:]
    Y_shuffle = YX[:,0]
    Y_shuffle = np.squeeze(Y_shuffle)

    thirds = len(Y)/3
    Xtrain = X_shuffle[0:thirds,:]
    Ytrain = Y_shuffle[0:thirds]
    Xvalidate = X_shuffle[thirds+1:2*thirds,:]
    Yvalidate = Y_shuffle[thirds+1:2*thirds]
    Xtest = X_shuffle[(2*thirds)+1:,:]
    Ytest = Y_shuffle[(2*thirds)+1:]

    # Tune hyperparameters using Xvalidate
    confusion_scores = np.zeros((1,len(num_tree_array)))
    j=0
    for num_trees in num_tree_array:
        model = RandomForestClassifier(n_estimators=num_trees)
        model.fit(Xtrain, Ytrain)
        Ypredict = model.predict(Xvalidate)

        Ypredict = (Ypredict >= 0.5)
        Yvalidate = Yvalidate.astype(int)
        confusion = confusion_matrix(Yvalidate, Ypredict)
        confusion_scores[0,j] = confusion[0,0] + confusion[1,1]
        j=j+1
    max_index = np.argmax(confusion_scores)
    best_num_trees = num_tree_array[max_index]

    # Test using Xtest
    model = RandomForestClassifier(n_estimators=best_num_trees)
    model.fit(Xtrain, Ytrain)
    Ypredict = model.predict(Xvalidate)

    Ypredict = (Ypredict >= 0.5)
    Yvalidate = Yvalidate.astype(int)
    confusion = confusion_matrix(Yvalidate, Ypredict)
    confusion_percentages[:,:,i] = confusion

    accuracies[0,i] = sum(Ypredict == Yvalidate)/float(len(Yvalidate))

sum_confusion = np.sum(confusions, axis=2)
total_sums = (np.sum(sum_confusion,1)).astype(float)
avg_confusion = np.divide(sum_confusion, total_sums[:,None])
avg_accuracy = np.mean(accuracies)
plot_confusion_matrix(avg_confusion, title='Random Forest Method Confusion Matrix - Classifying EDSS >= 3')
print(avg_confusion)
print(avg_accuracy)


