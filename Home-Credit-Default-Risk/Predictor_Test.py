# -*- coding: utf-8 -*-
# Find the Baseline Model
# import warnings
# warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# Load Dataset
train = pd.read_csv('application_train_drop_outliers.csv')
train.drop('SK_ID_CURR', axis=1, inplace=True)
y_train= train.pop('TARGET')
print('Original Trainset (without ID and TARGET):', train.shape)

# Processing of Unstructed Features(Label(<=2) and One-Hot(>2))
le = LabelEncoder()
le_count = 0
for col in list(train.columns):
	if train[col].dtype == 'object':
		if len(list(train[col].unique())) <= 2:
			le.fit(train[col])
			train[col] = le.transform(train[col])
			le_count += 1
print('%d columns in Trainset were label encoded' % le_count)
train = pd.get_dummies(train)
print('Structured Trainset (without ID and TARGET):', train.shape)
train.to_csv('application_train_structured.csv', index=False)

# Meta-Models(None Values!)
model_list = [	('XGB', xgb.XGBClassifier()), 
				('LGB', lgb.LGBMClassifier()), 
				('CaB', cb.CatBoostClassifier(logging_level='Silent'))]

# Run and Cross Validation
for id, clf in model_list:
	t0 = time.time()
	scores = cross_val_score(clf, train, y_train, cv=6, scoring='roc_auc')
	print('%s: \t%.6f\tTime: %.1f s' %(id, scores.mean(), time.time()-t0))