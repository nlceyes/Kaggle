# -*- coding: utf-8 -*-
# With Features from Leakage Method
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error

# 读取数据
train = pd.read_csv("train.csv")
id_train = train.pop('ID')
y_train = train.pop('target')
y_train_log = np.log1p(y_train)
print('Original Trainset:', train.shape)
test = pd.read_csv("test.csv")
id_test = test.pop('ID')
print('Original Testset:', test.shape)
submit = pd.read_csv('sample_submission.csv')

# 处理数据(去除单一特征和重复特征)
cols_with_onlyone_val = train.columns[train.nunique() == 1]
train.drop(cols_with_onlyone_val.values, axis=1, inplace=True)
print('Trainset Dropped Features with Only One Values:', train.shape)
test.drop(cols_with_onlyone_val.values, axis=1, inplace=True)
print('Testset Dropped Features with Only One Values:', test.shape)
colsToRemove = []
columns = train.columns
for i in range(len(columns)-1):
	v = train[columns[i]].values
	for j in range(i + 1,len(columns)):
		if np.array_equal(v, train[columns[j]].values):
			colsToRemove.append(columns[j])
train.drop(colsToRemove, axis=1, inplace=True)
print('Trainset Dropped Duplicated Features:', train.shape)
test.drop(colsToRemove, axis=1, inplace=True)
print('Testset Dropped Duplicated Features:', test.shape)

# 处理数据(Leakage)
cols = list(train.columns)
leak_cols = []
for c in cols:
	leak1 = np.sum((train[c]==y_train).astype(int))
	leak2 = np.sum((((train[c] - y_train) / y_train) < 0.05).astype(int))
	if leak1 > 30 and leak2 > 3500:
		leak_cols.append(c)
print('Trainset with Leakage Method 1:', train[leak_cols].shape)
print('Testset with Leakage Method 1:', test[leak_cols].shape)
train['nz_mean'] = train[leak_cols].apply(lambda x: x[x!=0].mean(), axis=1)
train['nz_max'] = train[leak_cols].apply(lambda x: x[x!=0].max(), axis=1)
train['nz_min'] = train[leak_cols].apply(lambda x: x[x!=0].min(), axis=1)
train['ez'] = train[leak_cols].apply(lambda x: len(x[x==0]), axis=1)
train['mean'] = train[leak_cols].apply(lambda x: x.mean(), axis=1)
train['max'] = train[leak_cols].apply(lambda x: x.max(), axis=1)
train['min'] = train[leak_cols].apply(lambda x: x.min(), axis=1)
test['nz_mean'] = test[leak_cols].apply(lambda x: x[x!=0].mean(), axis=1)
test['nz_max'] = test[leak_cols].apply(lambda x: x[x!=0].max(), axis=1)
test['nz_min'] = test[leak_cols].apply(lambda x: x[x!=0].min(), axis=1)
test['ez'] = test[leak_cols].apply(lambda x: len(x[x==0]), axis=1)
test['mean'] = test[leak_cols].apply(lambda x: x.mean(), axis=1)
test['max'] = test[leak_cols].apply(lambda x: x.max(), axis=1)
test['min'] = test[leak_cols].apply(lambda x: x.min(), axis=1)
leak_cols += ['nz_mean', 'nz_max', 'nz_min', 'ez', 'mean', 'max', 'min']
for i in range(2, 100):
	train['index'+str(i)] = ((train.index + 2) % i == 0).astype(int)
	test['index'+str(i)] = ((test.index + 2) % i == 0).astype(int)
	leak_cols.append('index'+str(i))
print('Trainset with Leakage Method 2:', train[leak_cols].shape)
print('Testset with Leakage Method 2:', test[leak_cols].shape)
train = train.replace(0, np.nan)
test = test.replace(0, np.nan)
train = train[leak_cols]
test = test[leak_cols]

# 建立模型并交叉验证
clf = lgb.LGBMRegressor(learning_rate=0.02, min_child_samples=20, min_split_gain=0.2, n_estimators=150, num_leaves=15, subsample=0.6)
print(clf)
t0 = time.time()
scores = cross_val_score(clf, train, y_train_log, cv=3, scoring='neg_mean_squared_error')
print('%.3f\tTime: %.1f s' %(scores.mean(), time.time()-t0))
t0 = time.time()
clf.fit(train, y_train_log)
y_test_pred = clf.predict(test)
y_train_pred = clf.predict(train)
y_test_pred = np.expm1(y_test_pred)
y_train_pred = np.expm1(y_train_pred)
score = mean_squared_log_error(y_train, y_train_pred)
print('%.3f\tTime: %.1f s' %(score, time.time()-t0))

# 输出结果
submit['target'] = y_test_pred
submit.to_csv('LGB_pred_r5.csv', index=False)
