# -*- coding: utf-8 -*-
# With Features from Leakage Method
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

# 读取数据
train = pd.read_csv("train.csv")
id_train = train.pop('ID')
y_train = train.pop('target')
y_train_log = np.log1p(y_train)
print('Original Trainset:', train.shape)

# 处理数据(去除单一特征和重复特征)
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

# 处理数据(Leakage)
cols = list(train.columns)
leak_cols = []
for c in cols:
	leak1 = np.sum((train[c]==y_train).astype(int))
	leak2 = np.sum((((train[c] - y_train) / y_train) < 0.05).astype(int))
	if leak1 > 30 and leak2 > 3500:
		leak_cols.append(c)
print('Trainset with Leakage Method 1:', train[leak_cols].shape)
train['nz_mean'] = train[leak_cols].apply(lambda x: x[x!=0].mean(), axis=1)
train['nz_max'] = train[leak_cols].apply(lambda x: x[x!=0].max(), axis=1)
train['nz_min'] = train[leak_cols].apply(lambda x: x[x!=0].min(), axis=1)
train['ez'] = train[leak_cols].apply(lambda x: len(x[x==0]), axis=1)
train['mean'] = train[leak_cols].apply(lambda x: x.mean(), axis=1)
train['max'] = train[leak_cols].apply(lambda x: x.max(), axis=1)
train['min'] = train[leak_cols].apply(lambda x: x.min(), axis=1)
leak_cols += ['nz_mean', 'nz_max', 'nz_min', 'ez', 'mean', 'max', 'min']
for i in range(2, 100):
	train['index'+str(i)] = ((train.index + 2) % i == 0).astype(int)
	leak_cols.append('index'+str(i))
print('Trainset with Leakage Method 2:', train[leak_cols].shape)
train = train.replace(0, np.nan)
train = train[leak_cols]

# 建立模型并交叉验证
if __name__ == '__main__':
	t0 = time.time()
	model = lgb.LGBMRegressor()
	params = {	'boosting_type': ['gbdt'],
				'num_leaves': [7, 15, 23, 31, 39, 47],
				'learning_rate': [0.025, 0.02, 0.015, 0.01, 0.005],
				'n_estimators': [100, 150, 200],
				'min_split_gain': [0.0, 0.1, 0.2],
				'min_child_samples': [10, 20, 30],
				'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]}
	clf = GridSearchCV(model, params, cv=3, scoring='neg_mean_squared_error', n_jobs=1)
	clf.fit(train, y_train_log)
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	means_t = clf.cv_results_['mean_train_score']
	stds_t = clf.cv_results_['std_train_score']
	param_list = clf.cv_results_['params']
	for mean, std, mean_t, std_t, param in zip(means, stds, means_t, stds_t, param_list):
		print("%0.3f (+/-%0.3f) %0.3f (+/-%0.3f) %r" % (mean_t, std_t, mean, std, param))
	print(clf.best_params_)
	print(clf.best_score_)
	print('All Done in %.3f s' % (time.time() - t0))