# pyarimafft library

A Python Library which efficiently combines LOESS cleaning, Fast Fourier Transform Extracted key Cyclicities and ARIMA
to produce meaningful and explainable time series forecasts.


## Installation 
''' bash

pip install pyarimafft

'''

##Usage

'''python
#endog = np.array(vector)
model_obj = pyarimafft.model(forecast_horizon=12)
model_obj.outlier_clean(endog=endog,window_size=10,outlier_threshold=0.8,peak_clean=False,trough_clean=False,both_sides_clean=True)
model_obj.extract_key_seasonalities(power_quantile=0.90,time_period=d)
model_obj.reconstruct_seasonal_features(mode='seperate')

# It is possible to add one exogenous vector at a time
model_obj.add_exog(exog1)
model_obj.add_exog(exog2)

###Call the auto_arima function 
model_obj.auto_arima(p=None,d=None,q=None,max_p=3,max_q=3,max_d=1,auto_fit=True)

####Attributes which you can extract

model_obj.endog
model_obj.trend 
model_obj.outlier_cleaned 
model_obj.seasonal_component 
model_obj.isolated_components 
model_obj.isolated_seasonality 
model_obj.forecast  
model_obj.seasonal_feature_train 
model_obj.seasonal_feature_future 
model_obj.time_train  
model_obj.time_future  
model_obj.forecast_horizon
model_obj.forecast 
model_obj.optimal_order 

'''




 
