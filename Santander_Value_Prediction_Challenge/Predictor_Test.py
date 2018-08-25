# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import cross_val_score

# 读取数据
train = pd.read_csv("train.csv")
id_train = train.pop('ID')
y_train = train.pop('target')
y_train = np.log1p(y_train)
print('Original Trainset:', train.shape)

# 处理数据
cols_with_onlyone_val = train.columns[train.nunique() == 1]
train.drop(cols_with_onlyone_val.values, axis=1, inplace=True)
print('Trainset Dropped Features with Only One Values:', train.shape)
colsToRemove = []
columns = train.columns
for i in range(len(columns)-1):
	v = train[columns[i]].values
	for j in range(i + 1,len(columns)):
		if np.array_equal(v, train[columns[j]].values):
			colsToRemove.append(columns[j])
train.drop(colsToRemove, axis=1, inplace=True)
print('Trainset Dropped Duplicated Features:', train.shape)

# 基模型
model_list = [	('RF', RandomForestRegressor(random_state=0)), 
		('GB', GradientBoostingRegressor(random_state=0)), 
		('XGB', xgb.XGBRegressor()), 
		('LGB', lgb.LGBMRegressor())]

# 建立模型并交叉验证
for id, clf in model_list:
	t0 = time.time()
	scores = cross_val_score(clf, train, y_train, cv=3, scoring='neg_mean_squared_error')
	print('%s: \t%.6f\tTime: %.1f s' %(id, scores.mean(), time.time()-t0))
