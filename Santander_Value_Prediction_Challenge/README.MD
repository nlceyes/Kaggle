# Santander Value Prediction Challenge

https://www.kaggle.com/c/santander-value-prediction-challenge

Rank: 294/4484 - Top 6.6% (Bronze)

Method: Combination of the leakages and LGBM predictions, where the leakages were not found

The Leakages are found, inspired by the following kernels:
  
  	https://www.kaggle.com/titericz/the-property-by-giba
    	First kernels to reveal the time series property in the dataset. Much thanks to Giba!
		
	https://www.kaggle.com/tezdhar/breaking-lb-fresh-start
		My first reference kernels
	
	https://www.kaggle.com/nulldata/jiazhen-to-armamut-via-gurchetan1000-0-56
		My final reference kernels

The method of feature engineering in LGBM model are also inspired by the following kernels:

	https://www.kaggle.com/alexpengxiao/preprocessing-model-averaging-by-xgb-lgb-1-39
	
	https://www.kaggle.com/the1owl/love-is-the-answer

Flow Chart:
	
	_Test(Get Baseline Model) --> _CV(Parameter Tuning) --> _Run(LGBM Using Best Parameters) --> _Run_L(Leakage Method and Combination of Previous LGBM Model)
