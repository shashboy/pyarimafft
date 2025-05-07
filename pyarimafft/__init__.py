import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .constants import VERSION

__version__ = VERSION


def outlier_label(err2,threshold):
    if err2>=threshold:
        return 1
    else:
        return 0
def get_complex_conjugate(x):
    return x.conjugate()
get_complex_conjugate_vec = np.vectorize(get_complex_conjugate)
outlier_label_vec = np.vectorize(outlier_label)

class model:
    def __init__(self,forecast_horizon):
        self.endog = None
        self.trend = None
        self.outlier_cleaned = None
        self.seasonal_component = None
        self.isolated_components = None
        self.isolated_seasonality = None
        self.forecast = None 
        self.seasonal_feature_train = None
        self.seasonal_feature_future = None
        self.neg_index = None
        self.time_train = None 
        self.time_future = None 
        if forecast_horizon==0:
            forecast_horizon = None
        self.forecast_horizon = forecast_horizon
        self.forecast = None
        self.optimal_order = None

    def outlier_clean(self,endog,window_size,outlier_threshold=0.8,peak_clean=None,trough_clean=None,both_sides_clean=True):
        lowess_frac = window_size/len(endog)
        endog_bool = isinstance(endog, np.ndarray)
        if endog_bool==False:
            endog = np.array(endog)
        self.endog = np.array(endog)
        model_smooth = sm.nonparametric.lowess(endog,list(range(len(endog))),frac=lowess_frac)[:,1] ####trend on actuals.
        error_2 = (endog-model_smooth)**2 ####error squared 
        median_error_2 = np.quantile(error_2,0.5) ####median error benchmark
        error_mark = np.quantile(error_2,outlier_threshold)
        benchmark_volatility = error_mark**0.5
        seasonal_component_temp = endog - model_smooth
        label = outlier_label_vec(error_2,error_mark)
        cleaned = np.array(endog)
        if both_sides_clean==True or (peak_clean==True and trough_clean==True) :
            count = 0
            trough_clean=None
            peak_clean=None
            while count<endog.shape[0]:
                if seasonal_component_temp[count]<0 and label[count]==1:
                    cleaned[count] = model_smooth[count] - benchmark_volatility
                elif seasonal_component_temp[count]>0 and label[count]==1:
                    cleaned[count] = model_smooth[count] + benchmark_volatility
                else:
                    pass 
                count=count+1
        elif peak_clean==True and trough_clean==False and both_sides_clean==False:
            count = 0
            trough_clean=None
            both_sides_clean=None
            while count<endog.shape[0]:
                if seasonal_component_temp[count]>0 and label[count]==1:
                    cleaned[count] = model_smooth[count] + benchmark_volatility
                else:
                    pass 
                count=count+1
        elif peak_clean==False and trough_clean==True and both_sides_clean==False:
            count = 0
            trough_clean=None
            both_sides_clean=None
            while count<endog.shape[0]:
                if seasonal_component_temp[count]<0 and label[count]==1:
                    cleaned[count] = model_smooth[count] - benchmark_volatility
                else:
                    pass 
                count=count+1
        model_smooth = sm.nonparametric.lowess(cleaned,list(range(len(endog))),frac=lowess_frac)[:,1]    
        seasonal_component = cleaned - model_smooth
        self.trend = model_smooth
        plt.plot(endog,label='endog')
        plt.plot(model_smooth,label='trend')
        plt.plot(cleaned,label='outlier cleaned')
        plt.legend()
        plt.show()
        self.outlier_cleaned = cleaned
        self.seasonal_component = seasonal_component
        # return self.trend,self.outlier_cleaned,self.seasonal_component

    def extract_key_seasonalities(self,power_quantile,time_period=0.01):
        cn = np.fft.fft(self.seasonal_component)
        freq = np.fft.fftfreq(len(self.seasonal_component),d=time_period)
        neg_index = 0
        for i in freq:
            if i>=0:
                neg_index = neg_index+1
            else:
                break
        self.neg_index = neg_index
        fft_df = pd.DataFrame()
        fft_df['COEF'] = cn
        fft_df['FREQ'] = freq
        fft_df['TIME_PERIOD'] = fft_df['FREQ']**(-1)
        fft_df['POWER'] = np.abs(fft_df['COEF'])
        threshold = np.quantile(fft_df['POWER'],[power_quantile])
        fft_df['KEY_FREQ'] = outlier_label_vec(fft_df['POWER'],threshold)
        fft_df['K'] = fft_df.index
        fft_df.to_csv('fft_check.csv')
        
        self.isolated_components = fft_df.iloc[1:neg_index][['K','COEF','TIME_PERIOD','KEY_FREQ']]
        self.isolated_components = self.isolated_components[(self.isolated_components['KEY_FREQ']==1)] 
        self.isolated_seasonality = np.array(self.isolated_components['TIME_PERIOD'].tolist())
        self.isolated_components = fft_df.iloc[0:][['K','COEF','TIME_PERIOD','KEY_FREQ']]
        self.isolated_components['KEY_COEF'] = self.isolated_components['COEF'] * self.isolated_components['KEY_FREQ']
        

    def reconstruct_seasonal_features(self,mode='seperate'):
        time_train = np.arange(self.isolated_components.shape[0])
        time_future = np.arange(self.isolated_components.shape[0],self.isolated_components.shape[0]+self.forecast_horizon)
        self.time_train = time_train
        self.time_future = time_future 
        if mode=='additive':
            self.seasonal_feature_train = np.fft.ifft(self.isolated_components['KEY_COEF'])
            self.seasonal_feature_train = self.seasonal_feature_train.real
            key_coeffs = self.isolated_components.iloc[0:self.neg_index][['K','KEY_COEF','KEY_FREQ']]
            key_coeffs = key_coeffs[(key_coeffs['KEY_FREQ']==1)]
            seasonal_feature_future = []
            omega = 2*np.pi/self.isolated_components.shape[0]
            for i,j in key_coeffs.iterrows():
                complex_wave_pos = np.exp(1j*omega*key_coeffs.loc[i,'K']*self.time_future)
                signal_val_pos = key_coeffs.loc[i,'KEY_COEF'] * complex_wave_pos
                signal_val_pos = signal_val_pos.real
                complex_wave_neg = np.exp(1j*omega*key_coeffs.loc[i,'K']*self.time_future*-1)
                signal_val_neg = key_coeffs.loc[i,'KEY_COEF'].conjugate() * complex_wave_neg
                signal_val_neg = signal_val_neg.real
                signal_val = signal_val_pos+signal_val_neg
                seasonal_feature_future.append(signal_val)
            seasonal_feature_future = np.array(seasonal_feature_future)
            seasonal_feature_future = seasonal_feature_future/self.isolated_components.shape[0]
            seasonal_feature_future = np.sum(seasonal_feature_future,axis=0)
            self.seasonal_feature_future = seasonal_feature_future
            plt.plot(self.time_train,self.seasonal_feature_train,label='Train Features')
            plt.plot(self.time_future,self.seasonal_feature_future,label='Features for forecast')
            plt.legend()
            plt.show()

        if mode=='seperate':
            self.seasonal_feature_train = np.fft.ifft(self.isolated_components['KEY_COEF'])
            self.seasonal_feature_train = self.seasonal_feature_train.real
            key_coeffs = self.isolated_components.iloc[0:self.neg_index][['K','KEY_COEF','KEY_FREQ']]
            key_coeffs = key_coeffs[(key_coeffs['KEY_FREQ']==1)]
            omega = 2*np.pi/self.isolated_components.shape[0]

            seasonal_feature_train = []
            for i,j in key_coeffs.iterrows():
                complex_wave_pos = np.exp(1j*omega*key_coeffs.loc[i,'K']*self.time_train)
                signal_val_pos = key_coeffs.loc[i,'KEY_COEF'] * complex_wave_pos
                signal_val_pos = signal_val_pos.real
                complex_wave_neg = np.exp(1j*omega*key_coeffs.loc[i,'K']*self.time_train*-1)
                signal_val_neg = key_coeffs.loc[i,'KEY_COEF'].conjugate() * complex_wave_neg
                signal_val_neg = signal_val_neg.real
                signal_val = signal_val_pos+signal_val_neg
                seasonal_feature_train.append(signal_val)
            seasonal_feature_train = np.array(seasonal_feature_train)
            seasonal_feature_train = seasonal_feature_train/self.isolated_components.shape[0]
            self.seasonal_feature_train = seasonal_feature_train

            seasonal_feature_future = []
            
            for i,j in key_coeffs.iterrows():
                complex_wave_pos = np.exp(1j*omega*key_coeffs.loc[i,'K']*self.time_future)
                signal_val_pos = key_coeffs.loc[i,'KEY_COEF'] * complex_wave_pos
                signal_val_pos = signal_val_pos.real
                complex_wave_neg = np.exp(1j*omega*key_coeffs.loc[i,'K']*self.time_future*-1)
                signal_val_neg = key_coeffs.loc[i,'KEY_COEF'].conjugate() * complex_wave_neg
                signal_val_neg = signal_val_neg.real
                signal_val = signal_val_pos+signal_val_neg
                seasonal_feature_future.append(signal_val)
            seasonal_feature_future = np.array(seasonal_feature_future)
            seasonal_feature_future = seasonal_feature_future/self.isolated_components.shape[0]
            self.seasonal_feature_future = seasonal_feature_future

            

            for i in range(key_coeffs.shape[0]):  
                plt.subplot(key_coeffs.shape[0],1,i+1)
                plt.title("Time Period {}".format( round(self.isolated_seasonality[i],2)))
                plt.plot(self.time_train,self.seasonal_feature_train[i,:],label='Train Features')
                plt.plot(self.time_future,self.seasonal_feature_future[i,:],label='Features for forecast')
                plt.legend()
                plt.subplots_adjust(hspace=0.5)
            plt.show()

    def add_exog(self,exog):
        exog_bool = isinstance(exog, np.ndarray)
        if exog_bool==False:
            exog = np.array(exog)
        exog = exog.reshape(exog.shape[0],1)
        if exog.shape[0] != self.endog.shape[0]+self.forecast_horizon:
            return "Exog length does not match train + future horizon length "
        exog_train = exog[0:self.endog.shape[0]]
        exog_train = exog_train.T
        exog_future = exog[self.endog.shape[0]:self.endog.shape[0]+self.forecast_horizon]
        exog_future = exog_future.T

        try:
            if self.seasonal_feature_train==None and self.seasonal_feature_future==None:
                self.seasonal_feature_train = exog_train
                self.seasonal_feature_future = exog_future
        except:
            self.seasonal_feature_train = np.concatenate([self.seasonal_feature_train,exog_train],axis=0)
            self.seasonal_feature_future = np.concatenate([self.seasonal_feature_future,exog_future],axis=0)
            
            
        
        
        
            
            
            
        

    def auto_arima(self,p=None,d=None,q=None,max_p=3,max_q=3,max_d=1,auto_fit=True):
        if p!=None and q!=None and d!=None:
            if len(self.outlier_cleaned)==0:
                forecast_model = ARIMA(endog = self.endog, exog = self.seasonal_feature_train.T,order=(p,d,q))
            elif len(self.outlier_cleaned)>0:
                forecast_model = ARIMA(endog = self.outlier_cleaned, exog = self.seasonal_feature_train.T,order=(p,d,q))
            model_fit = forecast_model.fit()
            preds = model_fit.predict(exog = self.seasonal_feature_future.T , start = self.endog.shape[0],end = self.endog.shape[0]+self.forecast_horizon-1 )
            self.forecast = preds
            plt.plot(range(0,self.endog.shape[0]),self.endog,label='endog')
            plt.plot(range(0,self.endog.shape[0]),self.outlier_cleaned,label='outlier cleaned')
            plt.plot(range(self.endog.shape[0],self.endog.shape[0]+self.forecast_horizon),preds,label='prediction')
            plt.legend()
            plt.show()
            

        
        elif auto_fit==True:
            if len(self.outlier_cleaned)==0:
                train_series = self.endog
            elif len(self.outlier_cleaned)>0:
                train_series = self.outlier_cleaned
                
            min_aic_model = []
            
            for p_val in range(max_p+1):
                for d_val in range(max_d+1):
                    for q_val in range(max_q+1):
                        model_val = ARIMA(endog = train_series, exog = self.seasonal_feature_train.T,order=(p_val,d_val,q_val))
                        model_val_fit = model_val.fit()
                        aic_val = model_val_fit.aic
                        if len(min_aic_model)==0:
                            min_aic_model.append([aic_val,model_val_fit.model.order])
                        elif len(min_aic_model)>0:
                            aic_val = model_val_fit.aic
                            if aic_val<min_aic_model[-1][0]:
                                min_aic_model[0] = [aic_val,model_val_fit.model.order]
                        print(min_aic_model)
            self.optimal_order = min_aic_model[0][1]
            forecast_model = ARIMA(endog = train_series, exog = self.seasonal_feature_train.T,order=min_aic_model[0][1])
            model_fit = forecast_model.fit()
            preds = model_fit.predict(exog = self.seasonal_feature_future.T , start = self.endog.shape[0],end = self.endog.shape[0]+self.forecast_horizon-1 )
            self.forecast = preds
            plt.plot(range(0,self.endog.shape[0]),self.endog,label='endog')
            plt.plot(range(0,self.endog.shape[0]),self.outlier_cleaned,label='outlier cleaned')
            plt.plot(range(self.endog.shape[0],self.endog.shape[0]+self.forecast_horizon),preds,label='prediction')
            plt.legend()
            plt.show()




    def exponential_smoothing(self,trend,damped_trend,seasonal,seasonal_periods,use_boxcox):
        if len(self.outlier_cleaned)==0:
            forecast_model = ExponentialSmoothing(endog = self.endog,trend=trend,damped_trend=damped_trend,seasonal=seasonal,seasonal_periods=seasonal_periods,use_boxcox=use_boxcox)
        elif len(self.outlier_cleaned)>0:
            forecast_model = ExponentialSmoothing(endog = self.outlier_cleaned,trend=trend,damped_trend=damped_trend,seasonal=seasonal,seasonal_periods=seasonal_periods,use_boxcox=use_boxcox)
        model_fit = forecast_model.fit()
        preds = model_fit.predict(start = self.endog.shape[0],end = self.endog.shape[0]+self.forecast_horizon-1)
        self.forecast = preds
        plt.plot(range(0,self.endog.shape[0]),self.endog,label='endog')
        plt.plot(range(0,self.endog.shape[0]),self.outlier_cleaned,label='outlier cleaned')
        plt.plot(range(self.endog.shape[0],self.endog.shape[0]+self.forecast_horizon),preds,label='prediction')
        plt.legend()
        plt.show()







    
