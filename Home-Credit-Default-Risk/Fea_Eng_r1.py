# -*- coding: utf-8 -*-
# Neptune-ml Solution 4 (Based on Solution 3)
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
import multiprocessing as mp
from scipy.stats import skew, kurtosis, iqr
from sklearn.linear_model import LinearRegression

# Load Data
t0 = time.time()
train = pd.read_csv('application_train_FE_5.csv')
print('Original Trainset:', train.shape)
test = pd.read_csv('application_test_FE_5.csv')
print('Original Testset:', test.shape)
# bureau = pd.read_csv('bureau.csv')
# print('Original Bureau_set:', bureau.shape)
# credit_card = pd.read_csv('credit_card_balance.csv')
# print('Original Credit Card Balance:', credit_card.shape)
installments = pd.read_csv('installments_payments.csv') # Dataset Featured by PA from FE_7!
print('Original Installments Payments:', installments.shape)
# pos_cash = pd.read_csv('POS_CASH_balance.csv')
# print('Original Pos Cash:', pos_cash.shape)
previous_application = pd.read_csv('previous_application.csv') # Dataset Featured by PA from FE_6!
print('Original Previous Application:', previous_application.shape)
print('Dataset Loaded in %.1fs\n' %(time.time()-t0))

# Functions to Processing Features
def fea_eng_application(X, y):
	t0 = time.time()
	# Aggregation List
	AGGREGATION_RECIPIES = [
		(['CODE_GENDER', 'NAME_EDUCATION_TYPE'], [('AMT_ANNUITY', 'max'),
												  ('AMT_CREDIT', 'max'),
												  ('EXT_SOURCE_1', 'mean'),
												  ('EXT_SOURCE_2', 'mean'),
												  ('OWN_CAR_AGE', 'max'),
												  ('OWN_CAR_AGE', 'sum')]),
		(['CODE_GENDER', 'ORGANIZATION_TYPE'], [('AMT_ANNUITY', 'mean'),
												('AMT_INCOME_TOTAL', 'mean'),
												('DAYS_REGISTRATION', 'mean'),
												('EXT_SOURCE_1', 'mean')]),
		(['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], [('AMT_ANNUITY', 'mean'),
													 ('CNT_CHILDREN', 'mean'),
													 ('DAYS_ID_PUBLISH', 'mean')]),
		(['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('EXT_SOURCE_1', 'mean'),
																							   ('EXT_SOURCE_2', 'mean')]),
		(['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], [('AMT_CREDIT', 'mean'),
													  ('AMT_REQ_CREDIT_BUREAU_YEAR', 'mean'),
													  ('APARTMENTS_AVG', 'mean'),
													  ('BASEMENTAREA_AVG', 'mean'),
													  ('EXT_SOURCE_1', 'mean'),
													  ('EXT_SOURCE_2', 'mean'),
													  ('EXT_SOURCE_3', 'mean'),
													  ('NONLIVINGAREA_AVG', 'mean'),
													  ('OWN_CAR_AGE', 'mean'),
													  ('YEARS_BUILD_AVG', 'mean')]),
		(['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('ELEVATORS_AVG', 'mean'),
																				('EXT_SOURCE_1', 'mean')]),
		(['OCCUPATION_TYPE'], [('AMT_ANNUITY', 'mean'),
							   ('CNT_CHILDREN', 'mean'),
							   ('CNT_FAM_MEMBERS', 'mean'),
							   ('DAYS_BIRTH', 'mean'),
							   ('DAYS_EMPLOYED', 'mean'),
							   ('DAYS_ID_PUBLISH', 'mean'),
							   ('DAYS_REGISTRATION', 'mean'),
							   ('EXT_SOURCE_1', 'mean'),
							   ('EXT_SOURCE_2', 'mean'),
							   ('EXT_SOURCE_3', 'mean')]),
	]
	# Difference Features
	diff_feature_names = []
	for groupby_cols, specs in AGGREGATION_RECIPIES:
		for select, agg in specs:
			groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
			diff_name = '{}_diff'.format(groupby_aggregate_name)
			abs_diff_name = '{}_abs_diff'.format(groupby_aggregate_name)
			X[diff_name] = X[select] - X[groupby_aggregate_name]
			X[abs_diff_name] = np.abs(X[select] - X[groupby_aggregate_name])
			diff_feature_names.append(diff_name)
			diff_feature_names.append(abs_diff_name)
	#
	X['long_employment'] = (X['DAYS_EMPLOYED'] > -2000).astype(int)
	X['retirement_age'] = (X['DAYS_BIRTH'] > -14000).astype(int)
	print('Dataset of %s Processed in %.1fs\n' %(y, time.time()-t0))
	# Save File
	# X.to_csv('application_'+y+'_FE_6.csv', index=False)
	return X

def fea_eng_previous_application(previous_application, application, application_test):
	t0 = time.time()
	numbers_of_applications = [1, 3, 5]
	features = pd.DataFrame({'SK_ID_CURR': previous_application['SK_ID_CURR'].unique()})
	prev_applications_sorted = previous_application.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])
	group_object = prev_applications_sorted.groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].nunique().reset_index()
	group_object.rename(index=str, columns={'SK_ID_PREV': 'previous_application_number_of_prev_application'}, inplace=True)
	features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
	prev_applications_sorted['previous_application_prev_was_approved'] = (prev_applications_sorted['NAME_CONTRACT_STATUS'] == 'Approved').astype('int')
	group_object = prev_applications_sorted.groupby(by=['SK_ID_CURR'])['previous_application_prev_was_approved'].last().reset_index()
	features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
	prev_applications_sorted['previous_application_prev_was_refused'] = (prev_applications_sorted['NAME_CONTRACT_STATUS'] == 'Refused').astype('int')
	group_object = prev_applications_sorted.groupby(by=['SK_ID_CURR'])['previous_application_prev_was_refused'].last().reset_index()
	features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
	for number in numbers_of_applications:
		prev_applications_tail = prev_applications_sorted.groupby(by=['SK_ID_CURR']).tail(number)
		group_object = prev_applications_tail.groupby(by=['SK_ID_CURR'])['CNT_PAYMENT'].mean().reset_index()
		group_object.rename(index=str, columns={'CNT_PAYMENT': 'previous_application_term_of_last_{}_credits_mean'.format(number)}, inplace=True)
		features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
		group_object = prev_applications_tail.groupby(by=['SK_ID_CURR'])['DAYS_DECISION'].mean().reset_index()
		group_object.rename(index=str, columns={'DAYS_DECISION': 'previous_application_days_decision_about_last_{}_credits_mean'.format(number)}, inplace=True)
		features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
		group_object = prev_applications_tail.groupby(by=['SK_ID_CURR'])['DAYS_FIRST_DRAWING'].mean().reset_index()
		group_object.rename(index=str, columns={'DAYS_FIRST_DRAWING': 'previous_application_days_first_drawing_last_{}_credits_mean'.format(number)}, inplace=True)
		features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
	application = application.merge(features, left_on=['SK_ID_CURR'], right_on=['SK_ID_CURR'], how='left', validate='one_to_one')
	application_test = application_test.merge(features, left_on=['SK_ID_CURR'], right_on=['SK_ID_CURR'], how='left', validate='one_to_one')
	print('Previous Application Features Processed in %.1fs\n' %(time.time()-t0))
	# Save File
	# application.to_csv('application_train_FE_7.csv', index=False)
	# application_test.to_csv('application_test_FE_7.csv', index=False)
	return application, application_test

######
def chunk_groups(groupby_object, chunk_size):
	n_groups = groupby_object.ngroups
	group_chunk, index_chunk = [],[]
	for i, (index, df) in enumerate(groupby_object):
		group_chunk.append(df)
		index_chunk.append(index)
		if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
			group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
			group_chunk, index_chunk = [],[]
			yield index_chunk_, group_chunk_

def parallel_apply(groups, func, index_name='Index', num_workers=2, chunk_size=100000):
	n_chunks = np.ceil(1.0 * groups.ngroups / chunk_size)
	indeces, features  = [],[]
	for index_chunk, groups_chunk in chunk_groups(groups, chunk_size):
		if __name__ == '__main__': # Important for Windows!!
			mp.freeze_support() # Important for Windows!!
			with mp.pool.Pool(num_workers) as executor:
				features_chunk = executor.map(func, groups_chunk)
			features.extend(features_chunk)
			indeces.extend(index_chunk)
	features = pd.DataFrame(features)
	features.index = indeces
	features.index.name = index_name
	return features

def add_features(feature_name, aggs, features, feature_names, groupby):
	feature_names.extend(['{}_{}'.format(feature_name, agg) for agg in aggs])
	for agg in aggs:
		if agg == 'kurt':
			agg_func = kurtosis
		elif agg == 'iqr':
			agg_func = iqr
		else:
			agg_func = agg
		g = groupby[feature_name].agg(agg_func).reset_index().rename(index=str, columns={feature_name: '{}_{}'.format(feature_name, agg)})
		features = features.merge(g, on='SK_ID_CURR', how='left')
	return features, feature_names

def add_features_in_group(features, gr_, feature_name, aggs, prefix):
	for agg in aggs:
		if agg == 'sum':
			features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
		elif agg == 'mean':
			features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
		elif agg == 'max':
			features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
		elif agg == 'min':
			features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
		elif agg == 'std':
			features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
		elif agg == 'count':
			features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
		elif agg == 'skew':
			features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
		elif agg == 'kurt':
			features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
		elif agg == 'iqr':
			features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
		elif agg == 'median':
			features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()
		return features

def last_k_instalment_features(gr, periods):
	gr_ = gr.copy()
	gr_.sort_values(['DAYS_INSTALMENT'],ascending=False, inplace=True)
	features = {}
	for period in periods:
		gr_period = gr_.iloc[:period]
		features = add_features_in_group(features, gr_period, 'NUM_INSTALMENT_VERSION', 
										['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'], 'last_{}_'.format(period))
		features = add_features_in_group(features,gr_period, 'instalment_paid_late_in_days', 
										['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'], 'last_{}_'.format(period))
		features = add_features_in_group(features,gr_period ,'instalment_paid_late', ['count','mean'], 'last_{}_'.format(period))
		features = add_features_in_group(features,gr_period ,'instalment_paid_over_amount', 
										['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'], 'last_{}_'.format(period))
		features = add_features_in_group(features,gr_period,'instalment_paid_over', ['count','mean'], 'last_{}_'.format(period))
	return features

def _add_trend_feature(features, gr, feature_name, prefix):
	y = gr[feature_name].values
	try:
		x = np.arange(0, len(y)).reshape(-1,1)
		lr = LinearRegression()
		lr.fit(x,y)
		trend = lr.coef_[0]
	except:
		trend = np.nan
	features['{}{}'.format(prefix, feature_name)] = trend
	return features

def trend_in_last_k_instalment_features(gr, periods):
	gr_ = gr.copy()
	gr_.sort_values(['DAYS_INSTALMENT'],ascending=False, inplace=True)
	features = {}
	for period in periods:
		gr_period = gr_.iloc[:period]
		features = _add_trend_feature(features, gr_period, 'instalment_paid_late_in_days', '{}_period_trend_'.format(period))
		features = _add_trend_feature(features, gr_period, 'instalment_paid_over_amount', '{}_period_trend_'.format(period))
	return features

def fea_eng_installments(installments, application, application_test):
	t0 = time.time()
	#
	# installments_ = installments.sample(10000)
	installments_ = installments
	installments_['instalment_paid_late_in_days'] = installments_['DAYS_ENTRY_PAYMENT'] - installments_['DAYS_INSTALMENT']
	installments_['instalment_paid_late'] = (installments_['instalment_paid_late_in_days'] > 0).astype(int)
	installments_['instalment_paid_over_amount'] = installments_['AMT_PAYMENT'] - installments_['AMT_INSTALMENT']
	installments_['instalment_paid_over'] = (installments_['instalment_paid_over_amount'] > 0).astype(int)
	features = pd.DataFrame({'SK_ID_CURR':installments_['SK_ID_CURR'].unique()})
	groupby = installments_.groupby(['SK_ID_CURR'])
	feature_names = []
	features, feature_names = add_features('NUM_INSTALMENT_VERSION', ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'], features, feature_names, groupby)
	features, feature_names = add_features('instalment_paid_late_in_days',  ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'], features, feature_names, groupby)
	features, feature_names = add_features('instalment_paid_late', ['sum','mean'], features, feature_names, groupby)
	features, feature_names = add_features('instalment_paid_over_amount',  ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'], features, feature_names, groupby)
	features, feature_names = add_features('instalment_paid_over', ['sum','mean'], features, feature_names, groupby)
	print(len(features.columns))
	print(features.head())
	#
	func = partial(last_k_instalment_features, periods=[1,5,10,20,50,100])
	g = parallel_apply(groupby, func, index_name='SK_ID_CURR', chunk_size=10000).reset_index()
	features = features.merge(g, on='SK_ID_CURR', how='left')
	print(len(g.columns))
	print(g.head())
	#
	func = partial(trend_in_last_k_instalment_features, periods=[10,50,100,500])
	g = parallel_apply(groupby, func, index_name='SK_ID_CURR', chunk_size=10000).reset_index()
	features = features.merge(g, on='SK_ID_CURR', how='left')
	print(len(g.columns))
	print(g.head())
	#
	application = application.merge(features, on='SK_ID_CURR',how='left')
	application_test = application_test.merge(features, on='SK_ID_CURR',how='left')
	print('Installments Features Processed in %.1fs\n' %(time.time()-t0))
	# Save File
	# application.to_csv('application_train_FE_8.csv', index=False)
	# application_test.to_csv('application_test_FE_8.csv', index=False)
	return application, application_test
######

# Processing Dataset
train = fea_eng_application(train, 'train')
test = fea_eng_application(test, 'test')
train, test = fea_eng_previous_application(previous_application, train, test)
train, test = fea_eng_installments(installments, train, test)
# fea_eng_bureau(bureau, train, test)
# fea_eng_credit_card(credit_card, train, test)
# fea_eng_pos_cash(pos_cash, train, test)
train.to_csv('application_train_FE_8.csv', index=False)
test.to_csv('application_test_FE_8.csv', index=False)