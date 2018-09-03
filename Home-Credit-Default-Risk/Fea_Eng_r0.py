# -*- coding: utf-8 -*-
# Neptune-ml Solution 3
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
t0 = time.time()
train = pd.read_csv('application_train_drop_outliers.csv')
print('Original Trainset:', train.shape)
test = pd.read_csv('application_test.csv')
print('Original Testset:', test.shape)
bureau = pd.read_csv('bureau.csv')
print('Original Bureau_set:', bureau.shape)
credit_card = pd.read_csv('credit_card_balance.csv')
print('Original Credit Card Balance:', credit_card.shape)
installments = pd.read_csv('installments_payments.csv')
print('Original Installments Payments:', installments.shape)
pos_cash = pd.read_csv('POS_CASH_balance.csv')
print('Original Pos Cash:', pos_cash.shape)
previous_application = pd.read_csv('previous_application.csv')
print('Original Previous Application:', previous_application.shape)
print('Dataset Loaded in %.1fs\n' %(time.time()-t0))

# Functions to Processing Features
def fea_eng_application(X, y):
	t0 = time.time()
	# Outliers
	X['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
	X['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)
	# Manual Features
	X['annuity_income_percentage'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
	X['car_to_birth_ratio'] = X['OWN_CAR_AGE'] / X['DAYS_BIRTH']
	X['car_to_employ_ratio'] = X['OWN_CAR_AGE'] / X['DAYS_EMPLOYED']
	X['children_ratio'] = X['CNT_CHILDREN'] / X['CNT_FAM_MEMBERS']
	X['credit_to_annuity_ratio'] = X['AMT_CREDIT'] / X['AMT_ANNUITY']
	X['credit_to_goods_ratio'] = X['AMT_CREDIT'] / X['AMT_GOODS_PRICE']
	X['credit_to_income_ratio'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
	X['days_employed_percentage'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
	X['income_credit_percentage'] = X['AMT_INCOME_TOTAL'] / X['AMT_CREDIT']
	X['income_per_child'] = X['AMT_INCOME_TOTAL'] / (1 + X['CNT_CHILDREN'])
	X['income_per_person'] = X['AMT_INCOME_TOTAL'] / X['CNT_FAM_MEMBERS']
	X['payment_rate'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
	X['phone_to_birth_ratio'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_BIRTH']
	X['phone_to_employ_ratio'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_EMPLOYED']
	X['external_sources_weighted'] = X.EXT_SOURCE_1 * 2 + X.EXT_SOURCE_2 * 3 + X.EXT_SOURCE_3 * 4
	for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']:
		X['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)
	# Aggregation Features
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
	groupby_aggregate_names = []
	for groupby_cols, specs in AGGREGATION_RECIPIES:
		group_object = X.groupby(groupby_cols)
		for select, agg in specs:
			groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
			X = X.merge(group_object[select].agg(agg).reset_index().rename(index=str, columns={select: groupby_aggregate_name})[groupby_cols + [groupby_aggregate_name]], 
				on=groupby_cols, how='left')
			groupby_aggregate_names.append(groupby_aggregate_name)
	print('Dataset of %s Processed in %.1fs\n' %(y, time.time()-t0))
	# Save File
	# X.to_csv('application_'+y+'_FE_0.csv', index=False)
	return X

def fea_eng_bureau(bureau, application, application_test):
	t0 = time.time()
	# Preparation of Features
	bureau['bureau_credit_active_binary'] = (bureau['CREDIT_ACTIVE'] != 'Closed').astype(int)
	bureau['bureau_credit_enddate_binary'] = (bureau['DAYS_CREDIT_ENDDATE'] > 0).astype(int)
	groupby_SK_ID_CURR = bureau.groupby(by=['SK_ID_CURR'])
	features = pd.DataFrame({'SK_ID_CURR':bureau['SK_ID_CURR'].unique()})
	group_object = groupby_SK_ID_CURR['DAYS_CREDIT'].agg('count').reset_index()
	group_object.rename(index=str, columns={'DAYS_CREDIT': 'bureau_number_of_past_loans'},inplace=True)
	features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
	group_object = groupby_SK_ID_CURR['CREDIT_TYPE'].agg('nunique').reset_index()
	group_object.rename(index=str, columns={'CREDIT_TYPE': 'bureau_number_of_loan_types'},inplace=True)
	features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
	features['bureau_average_of_past_loans_per_type'] = features['bureau_number_of_past_loans'] / features['bureau_number_of_loan_types']
	group_object = groupby_SK_ID_CURR['bureau_credit_active_binary'].agg('mean').reset_index()
	features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
	group_object = groupby_SK_ID_CURR['AMT_CREDIT_SUM_DEBT'].agg('sum').reset_index()
	group_object.rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'bureau_total_customer_debt'},inplace=True)
	features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
	group_object = groupby_SK_ID_CURR['AMT_CREDIT_SUM'].agg('sum').reset_index()
	group_object.rename(index=str, columns={'AMT_CREDIT_SUM': 'bureau_total_customer_credit'},inplace=True)
	features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
	features['bureau_debt_credit_ratio'] = features['bureau_total_customer_debt'] / features['bureau_total_customer_credit']
	group_object = groupby_SK_ID_CURR['AMT_CREDIT_SUM_OVERDUE'].agg('sum').reset_index()
	group_object.rename(index=str, columns={'AMT_CREDIT_SUM_OVERDUE': 'bureau_total_customer_overdue'},inplace=True)
	features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
	features['bureau_overdue_debt_ratio'] = features['bureau_total_customer_overdue'] / features['bureau_total_customer_debt']
	group_object = groupby_SK_ID_CURR['CNT_CREDIT_PROLONG'].agg('sum').reset_index()
	group_object.rename(index=str, columns={'CNT_CREDIT_PROLONG': 'bureau_average_creditdays_prolonged'},inplace=True)
	features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
	group_object = groupby_SK_ID_CURR['bureau_credit_enddate_binary'].agg('mean').reset_index()
	group_object.rename(index=str, columns={'bureau_credit_enddate_binary': 'bureau_credit_enddate_percentage'},inplace=True)
	features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
	application = application.merge(features, left_on=['SK_ID_CURR'], right_on=['SK_ID_CURR'], how='left', validate='one_to_one')
	application_test = application_test.merge(features, left_on=['SK_ID_CURR'], right_on=['SK_ID_CURR'], how='left', validate='one_to_one')
	# Aggregation Features
	BUREAU_AGGREGATION_RECIPIES = [('CREDIT_TYPE', 'count'), ('CREDIT_ACTIVE', 'size')]
	for agg in ['mean', 'min', 'max', 'sum', 'var']:
		for select in [ 'AMT_ANNUITY', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 
						'AMT_CREDIT_SUM_OVERDUE', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG', 
						'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE','DAYS_CREDIT_UPDATE']:
			BUREAU_AGGREGATION_RECIPIES.append((select, agg))
	BUREAU_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], BUREAU_AGGREGATION_RECIPIES)]
	groupby_aggregate_names = []
	for groupby_cols, specs in BUREAU_AGGREGATION_RECIPIES:
		group_object = bureau.groupby(groupby_cols)
		for select, agg in specs:
			groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
			application =	application.merge(group_object[select].agg(agg).reset_index().rename(index=str, columns={select: groupby_aggregate_name})
							[groupby_cols + [groupby_aggregate_name]], on=groupby_cols, how='left')
			application_test =	application_test.merge(group_object[select].agg(agg).reset_index().rename(index=str, columns={select: groupby_aggregate_name})
								[groupby_cols + [groupby_aggregate_name]], on=groupby_cols, how='left')
			groupby_aggregate_names.append(groupby_aggregate_name)
	print('Bureau Features Processed in %.1fs\n' %(time.time()-t0))
	# Save File
	# application.to_csv('application_train_FE_1.csv', index=False)
	# application_test.to_csv('application_test_FE_1.csv', index=False)
	return application, application_test

def fea_eng_credit_card(credit_card, application, application_test):
	t0 = time.time()
	# Preparation of Features
	credit_card['number_of_instalments'] = credit_card.groupby(by=['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].agg('max').reset_index()['CNT_INSTALMENT_MATURE_CUM']
	credit_card['credit_card_max_loading_of_credit_limit'] = credit_card.groupby(by=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']).apply(
															 lambda x: x.AMT_BALANCE.max() / x.AMT_CREDIT_LIMIT_ACTUAL.max()).reset_index()[0]
	features = pd.DataFrame({'SK_ID_CURR':credit_card['SK_ID_CURR'].unique()})
	group_object = credit_card.groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].agg('nunique').reset_index()
	group_object.rename(index=str, columns={'SK_ID_PREV': 'credit_card_number_of_loans'}, inplace=True)
	features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
	group_object= credit_card.groupby(by=['SK_ID_CURR'])['number_of_instalments'].sum().reset_index()
	group_object.rename(index=str, columns={'number_of_instalments': 'credit_card_total_instalments'}, inplace=True)
	features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
	features['credit_card_installments_per_loan'] = features['credit_card_total_instalments'] / features['credit_card_number_of_loans']
	group_object = credit_card.groupby(by=['SK_ID_CURR'])['credit_card_max_loading_of_credit_limit'].agg('mean').reset_index()
	group_object.rename(index=str, columns={'credit_card_max_loading_of_credit_limit': 'credit_card_avg_loading_of_credit_limit'}, inplace=True)
	features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
	group_object = credit_card.groupby(by=['SK_ID_CURR'])['SK_DPD'].agg('mean').reset_index()
	group_object.rename(index=str, columns={'SK_DPD': 'credit_card_average_of_days_past_due'}, inplace=True)
	features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
	group_object = credit_card.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_ATM_CURRENT'].agg('sum').reset_index()
	group_object.rename(index=str, columns={'AMT_DRAWINGS_ATM_CURRENT': 'credit_card_drawings_atm'}, inplace=True)
	features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
	group_object = credit_card.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].agg('sum').reset_index()
	group_object.rename(index=str, columns={'AMT_DRAWINGS_CURRENT': 'credit_card_drawings_total'}, inplace=True)
	features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
	features['credit_card_cash_card_ratio'] = features['credit_card_drawings_atm'] / features['credit_card_drawings_total']
	application = application.merge(features, left_on=['SK_ID_CURR'], right_on=['SK_ID_CURR'], how='left', validate='one_to_one')
	application_test = application_test.merge(features, left_on=['SK_ID_CURR'], right_on=['SK_ID_CURR'], how='left', validate='one_to_one')
	# Aggregation Features
	CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES = []
	for agg in ['mean', 'min', 'max', 'sum', 'var']:
		for select in [ 'AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT',
						'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT',
						'AMT_PAYMENT_CURRENT', 'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT',
						'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_INSTALMENT_MATURE_CUM', 'MONTHS_BALANCE',
						'SK_DPD', 'SK_DPD_DEF']:
			CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES.append((select, agg))
	CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES)]
	groupby_aggregate_names = []
	for groupby_cols, specs in CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES:
		group_object = credit_card.groupby(groupby_cols)
		for select, agg in specs:
			groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
			application =	application.merge(group_object[select].agg(agg).reset_index().rename(index=str, columns={select: groupby_aggregate_name})
							[groupby_cols + [groupby_aggregate_name]], on=groupby_cols, how='left')
			application_test =	application_test.merge(group_object[select].agg(agg).reset_index().rename(index=str, columns={select: groupby_aggregate_name})
								[groupby_cols + [groupby_aggregate_name]], on=groupby_cols, how='left')
			groupby_aggregate_names.append(groupby_aggregate_name)
	print('Credit Card Features Processed in %.1fs\n' %(time.time()-t0))
	# Save File
	# application.to_csv('application_train_FE_2.csv', index=False)
	# application_test.to_csv('application_test_FE_2.csv', index=False)
	return application, application_test

def fea_eng_installments(installments, application, application_test):
	t0 = time.time()
	# Aggregation Features
	INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES = []
	for agg in ['mean', 'min', 'max', 'sum', 'var']:
		for select in [ 'AMT_INSTALMENT', 'AMT_PAYMENT', 'DAYS_ENTRY_PAYMENT',
						'DAYS_INSTALMENT', 'NUM_INSTALMENT_NUMBER', 'NUM_INSTALMENT_VERSION']:
			INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES.append((select, agg))
	INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES)]
	groupby_aggregate_names = []
	for groupby_cols, specs in INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES:
		group_object = installments.groupby(groupby_cols)
		for select, agg in specs:
			groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
			application =	application.merge(group_object[select].agg(agg).reset_index().rename(index=str, columns={select: groupby_aggregate_name})
							[groupby_cols + [groupby_aggregate_name]], on=groupby_cols, how='left')
			application_test =	application_test.merge(group_object[select].agg(agg).reset_index().rename(index=str, columns={select: groupby_aggregate_name})
								[groupby_cols + [groupby_aggregate_name]], on=groupby_cols, how='left')
			groupby_aggregate_names.append(groupby_aggregate_name)
	print('Installments Features Processed in %.1fs\n' %(time.time()-t0))
	# Save File
	# application.to_csv('application_train_FE_3.csv', index=False)
	# application_test.to_csv('application_test_FE_3.csv', index=False)
	return application, application_test

def fea_eng_pos_cash(pos_cash, application, application_test):
	t0 = time.time()
	# Aggregation Features
	POS_CASH_AGGREGATION_RECIPIES = []
	for agg in ['mean', 'min', 'max', 'sum', 'var']:
		for select in ['MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF']:
			POS_CASH_AGGREGATION_RECIPIES.append((select, agg))
	POS_CASH_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], POS_CASH_AGGREGATION_RECIPIES)]
	groupby_aggregate_names = []
	for groupby_cols, specs in POS_CASH_AGGREGATION_RECIPIES:
		group_object = pos_cash.groupby(groupby_cols)
		for select, agg in specs:
			groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
			application =	application.merge(group_object[select].agg(agg).reset_index().rename(index=str, columns={select: groupby_aggregate_name})
							[groupby_cols + [groupby_aggregate_name]], on=groupby_cols, how='left')
			application_test =	application_test.merge(group_object[select].agg(agg).reset_index().rename(index=str, columns={select: groupby_aggregate_name})
								[groupby_cols + [groupby_aggregate_name]], on=groupby_cols, how='left')
			groupby_aggregate_names.append(groupby_aggregate_name)
	print('Pos Cash Features Processed in %.1fs\n' %(time.time()-t0))
	# Save File
	# application.to_csv('application_train_FE_4.csv', index=False)
	# application_test.to_csv('application_test_FE_4.csv', index=False)
	return application, application_test

def fea_eng_previous_application(previous_application, application, application_test):
	t0 = time.time()
	# Aggregation Features
	PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = []
	for agg in ['mean', 'min', 'max', 'sum', 'var']:
		for select in [ 'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE',
						'CNT_PAYMENT', 'DAYS_DECISION', 'HOUR_APPR_PROCESS_START', 'RATE_DOWN_PAYMENT']:
			PREVIOUS_APPLICATION_AGGREGATION_RECIPIES.append((select, agg))
	PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], PREVIOUS_APPLICATION_AGGREGATION_RECIPIES)]
	groupby_aggregate_names = []
	for groupby_cols, specs in PREVIOUS_APPLICATION_AGGREGATION_RECIPIES:
		group_object = previous_application.groupby(groupby_cols)
		for select, agg in specs:
			groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
			application =	application.merge(group_object[select].agg(agg).reset_index().rename(index=str, columns={select: groupby_aggregate_name})
							[groupby_cols + [groupby_aggregate_name]], on=groupby_cols, how='left')
			application_test =	application_test.merge(group_object[select].agg(agg).reset_index().rename(index=str, columns={select: groupby_aggregate_name})
								[groupby_cols + [groupby_aggregate_name]], on=groupby_cols, how='left')
			groupby_aggregate_names.append(groupby_aggregate_name)
	print('Previous Application Features Processed in %.1fs\n' %(time.time()-t0))
	# Save File
	# application.to_csv('application_train_FE_5.csv', index=False)
	# application_test.to_csv('application_test_FE_5.csv', index=False)
	return application, application_test

# Processing Dataset
train = fea_eng_application(train, 'train')
test = fea_eng_application(test, 'test')
train, test = fea_eng_bureau(bureau, train, test)
train, test = fea_eng_credit_card(credit_card, train, test)
train, test = fea_eng_installments(installments, train, test)
train, test = fea_eng_pos_cash(pos_cash, train, test)
train, test = fea_eng_previous_application(previous_application, train, test)
train.to_csv('application_train_FE_5.csv', index=False)
test.to_csv('application_test_FE_5.csv', index=False)