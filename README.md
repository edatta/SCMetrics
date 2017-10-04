# SCMetrics
Exploration of Spinal Cord Metrics and Clinical Outcomes in Multiple Sclerosis

This work was done as part of CS289, UC Berkeley's Intro to Machine Learning class in Fall 2015.
(Sample data is not included since it includes patient information)

## Motivation
During the last few decades, researchers have struggled to find reliable biomarkers for multiple sclerosis (MS) that are accurately able to predict clinical outcomes. Until recently, spinal cord metrics were also poor contenders, due to the quality limitations of spinal cord imaging. With recent technological advances, we are now able to acquire better quality spinal cord images and capture these metrics more accurately. This study explores the potential of spinal cord metrics as biomarkers to predict clinical outcomes in multiple sclerosis using machine learning techniques.

## Data Set and Feature Selection
The data set includes spinal cord images from 129 patients and 20 controls.  Deformation fields, jacobian determinants, intensity, and gradient metrics at each voxel were used as input features.  This data was used to predict disease course and Expanded Disability Status Score (EDSS).

## Methods Used in Data Exploration

* Data Visualization using Principal Component Analysis
* Linear Regression
* Centroid Method
* SVM with Linear Kernel
* SVM with RBF Kernel
* Random Forest
* Multiclass Classification with Linear Support Vector Machine




More information about this work can be found in this video:
https://www.youtube.com/watch?v=hBk6F4KtDC4&f
