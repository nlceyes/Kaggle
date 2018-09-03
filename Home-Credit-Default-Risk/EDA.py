# -*- coding: utf-8 -*-
# Visualization of Features and Drop Outliers
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
train = pd.read_csv('application_train.csv')
print('Original Trainset:', train.shape)
test = pd.read_csv('application_test.csv')
print('Original Testset:', test.shape)

# Visualization
# TARGET
train['TARGET'].hist(xrot=45, bins=50)
plt.title('TARGET'+'(Trainset)')
plt.show()
# FEATURES
feature_list = [f for f in test.columns if f != 'SK_ID_CURR' ]
# print(feature_list)
for feature in feature_list:
	plt.figure(1)
	plt.subplot(121)
	train[feature].hist(xrot=45, bins=50)
	plt.title(feature+'(Trainset)')
	plt.subplot(122)
	test[feature].hist(xrot=45, bins=50)
	plt.title(feature+'(Testset)')
	plt.show()