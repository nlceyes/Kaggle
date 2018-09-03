# -*- coding: utf-8 -*-
# Run and Get the Predictions
import warnings
warnings.filterwarnings('ignore')
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
from sklearn.metrics import roc_auc_score

# Load Dataset
train = pd.read_csv('application_train_FE_8.csv')
train.drop('SK_ID_CURR', axis=1, inplace=True)
y_train= train.pop('TARGET')
print('Original Trainset (without ID and TARGET):', train.shape)
test = pd.read_csv('application_test_FE_8.csv')
test.drop('SK_ID_CURR', axis=1, inplace=True)
print('Original Testset (without ID and TARGET):', test.shape)
submit = pd.read_csv('sample_submission.csv')

# Processing of Unstructed Features(Label(<=2) and One-Hot(>2))
le = LabelEncoder()
le_count = 0
for col in list(train.columns):
	if train[col].dtype == 'object':
		if len(list(train[col].unique())) <= 2:
			le.fit(train[col])
			train[col] = le.transform(train[col])
			test[col] = le.transform(test[col])
			le_count += 1
print('%d columns in Dataset were label encoded' % le_count)
train = pd.get_dummies(train)
print('Structured Trainset (without ID and TARGET):', train.shape)
# train.to_csv('application_train_structured.csv', index=False)
test = pd.get_dummies(test)
print('Structured Testset (without ID and TARGET):', test.shape)
# test.to_csv('application_test_structured.csv', index=False)

# Run and Cross Validation
t0 = time.time()
# clf = lgb.LGBMClassifier(learning_rate=0.05, min_child_samples=60, min_split_gain=0.0, n_estimators=300, num_leaves=47, subsample=1.0)
# clf = lgb.LGBMClassifier()
# clf = cb.CatBoostClassifier(logging_level='Silent', random_state=0)
# clf = xgb.XGBClassifier()
clf = lgb.LGBMClassifier(learning_rate=0.1, min_child_samples=50, min_split_gain=0.5, n_estimators=5000, num_leaves=35, subsample=1.0, max_bin=300, reg_lambda=100, colsample_bytree=0.2)
print(clf)
scores = cross_val_score(clf, train, y_train, cv=6, scoring='roc_auc')
print('%.6f\tTime: %.1f s' %(scores.mean(), time.time()-t0))
t0 = time.time()
clf.fit(train, y_train)
y_train_pred = clf.predict_proba(train)[:, 1]
score = roc_auc_score(y_train, y_train_pred)
print('%.6f\tTime: %.1f s' %(score, time.time()-t0))
y_test_pred = clf.predict_proba(test)[:, 1]

# Write Submisiions
submit['TARGET'] = y_test_pred
submit.to_csv('LGB_pred.csv', index=False)
