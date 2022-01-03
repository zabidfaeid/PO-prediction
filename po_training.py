# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 09:03:21 2021

@author: M Zabid M Faeid
"""

#-- FORECASTING WITH EXTENDED PROPHET ALGORITHM --

# y(t) = T(t) + S(t) + E(t) + F(t) + A(t) + L(t)

# (1) Get the trend, seasonality and event - get the model
# (2) calculate the residual and create a model
# (3) sum (1) and (2) to get the prediction; or 
# (4) create a linear regression model with (1) and (2)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import json

df = pd.read_csv('./data/PalmOil.csv')
df = df.sort_values(by = 'YearMonth')

#-- Get date as index
df['Date'] = df['YearMonth'] + '-01'
df['Date'] = pd.to_datetime(df.Date, format ='%Y-%m-%d')
df.index = df['Date']
df = df.drop(columns = ['Date'])

#-- Decompose the time-series
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df['CPO prices'], model='additive')#, period = 1)
result.plot()
plt.show()

#-- Getting the components
resid = result.resid
seasonal = result.seasonal
trend = result.trend

plt.plot(resid)
plt.show()

df_seasonal = pd.DataFrame(seasonal)
df_seasonal['seasonaltrend'] = trend + seasonal  #--combine seasonal and trend as there is no obvious trend

df['seasonaltrend'] = seasonal #df_seasonal['seasonaltrend']
df['residual'] = df['CPO prices'] - df['seasonaltrend']
#df = df.dropna()

df['Time'] = list(range(0,48))

x1 = np.arange(0,48)
y1 = df['residual']
plt.plot(x1, y1, color = 'blue', label = "actual")
plt.title('residual')
plt.show()


#------------------------------------------------
# Modeling the seasonal factor using Fourier series transformation
#------------------------------------------------

from numpy.fft import fft, ifft
from scipy import signal as sig

# Plotting power in the frequency domain
sr = 48
ts = 1.0/sr
t = np.arange(0,1,ts)
X = fft(df['seasonaltrend'].dropna())
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T

power = np.abs(X)
mask = freq >= 0
freq = freq[mask]
power = power[mask]

plt.figure(figsize = (12, 6))
plt.subplot(121)

plt.stem(freq, np.abs(X), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, 10)
plt.ylim(0, 6500)

plt.subplot(122)
plt.plot(t, ifft(X), 'r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

#-- finding the peak signal
peaks = sig.find_peaks(power[freq >=0], prominence=1000)[0]
peak_freq =  freq[peaks]
peak_power = power[peaks]

plt.figure( figsize=(10, 4) )

ax1 = plt.subplot( 1, 2, 1 )
ax1.plot(freq, power, label='SeasonalTrend')
ax1.set_title('All Frequencies')
ax1.set_ylabel( 'Amplitude' )
ax1.set_xlabel( 'Frequency' )
plt.xticks(rotation=90)

ax2 = plt.subplot( 1, 2, 2 )
mask = (freq > 0) #& (freq <= 0.25)
ax2.plot(freq[mask], power[mask])
ax2.set_title('Frequencies in (0, 0.25]')
ax2.set_ylabel( 'Amplitude' )
ax2.set_xlabel( 'Frequency' )


plt.plot(peak_freq, peak_power, 'ro')

plt.tight_layout()
plt.xticks(rotation=90)

output = pd.DataFrame()
output['index'] = peaks
output['freq'] = peak_freq
output['amplitude'] = peak_power
output['period'] = 1 / peak_freq #/ 24
output['fft'] = X[peaks]
output = output.sort_values('amplitude', ascending=False)

#-- Get the fft model
filtered_fft_output = np.array([f if i in list(output['index']) else 0 for i, f in enumerate(X)])
filtered_residuals = ifft(filtered_fft_output)

#N = 24 * 5
plt.plot(df.index, df['seasonaltrend'], linewidth=1, label='Original seasonaltrend')
plt.plot(df.index, filtered_residuals.real, linewidth=1, label='Filtered seasonaltrend')
plt.legend(loc='upper right')
plt.suptitle('Residuals')
plt.grid()
#plt.ylim((-25, 25))
plt.xticks(rotation=90)

#how to plot fourier series as a model?
from cmath import phase
import math

fourier_terms = pd.DataFrame()
fourier_terms['fft'] = output['fft']
fourier_terms['freq'] = output['freq']
fourier_terms['amplitude'] = fourier_terms.fft.apply(lambda z: abs(z)) 
fourier_terms['phase'] = fourier_terms.fft.apply(lambda z: phase(z))
fourier_terms.sort_values(by=['amplitude'], ascending=[0])

# Create some helpful labels (FT_1..FT_N)
fourier_terms['label'] = list(map(lambda n : 'FT_{}'.format(n), range(1, len(fourier_terms) + 1)))

# Turn our dataframe into a dictionary for easy lookup
fourier_terms = fourier_terms.set_index('label')
fourier_terms_dict = fourier_terms.to_dict('index')

#fourier_terms
def seasonal_model(dict_ft, chosen_time, sr = 48, time_max = 81):
    # sr is the signal range that we used at the beginning when we develop the FT
    dt = pd.DataFrame(columns = ['Time'])
    dt['Time'] = list(range(0,time_max))
    for key in dict_ft.keys():
        a = dict_ft[key]['amplitude']
        w = 2 * math.pi * (dict_ft[key]['freq'] / sr)
        p = dict_ft[key]['phase']
        dt[key] = dt['Time'].apply(lambda t: math.cos(w*t + p))
    dt['FT_All'] = 0
    for column in list(fourier_terms.index):
        dt['FT_All'] = dt['FT_All'] + dt[column]
    
    ft_value = dt['FT_All'].loc[dt.Time == chosen_time].values
    
    return ft_value, dt

ftvalue, df_seasonal0 = seasonal_model(fourier_terms_dict, 3, time_max = 48)    

#plt.plot(df.index, df['seasonaltrend'], linewidth=1, label='Original seasonal trend')
plt.plot(df_seasonal0.index, df_seasonal0['FT_All'] , linewidth=1, label='FT_All')
plt.legend(loc='upper right')
plt.suptitle('FT_All')
plt.grid()
#plt.ylim((-25, 25))
plt.xticks(rotation=90)    


#train the seasonal model
from sklearn.linear_model import LinearRegression
X = pd.DataFrame(df_seasonal0[['Time', 'FT_All']]) 
# signal is how many calls were made in that hour
y = pd.DataFrame(df['seasonaltrend']) 

feat_seasonal = list(X)

model_seasonal = LinearRegression()
model_seasonal.fit(X, y)

y_pred = model_seasonal.predict(X)

plt.plot(X['Time'], y, linewidth=1, label='Original Signal')
plt.plot(X['Time'], y_pred, linewidth=1, label='Predicted Signal')
plt.legend(loc='upper right')
plt.suptitle('Seasonal model')
plt.grid()
plt.xticks(rotation=90)


#-- saving the seasonal model artifact
model_file = './models/seasonal_model.pkl'

joblib.dump(value=[feat_seasonal, fourier_terms_dict, model_seasonal], 
            filename=model_file)


#dict_ft, model_seasonal = joblib.load(model_file)


#------------------------------------------------
# Modeling the residual using RF regressor
#------------------------------------------------

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import pandas as pd

def mape(act, prd): 
    actual, pred = np.array(act), np.array(prd)
    return np.mean(np.abs((actual - pred) / actual)) * 100

'''
# Get the data

lag = [2,3,4]

dffin = df.copy(deep = True)
for lg in lag:
    dfx = df['Export'].reset_index()
    dfx = dfx.rename(columns = {'index':'date_num','Export':'Export_lag_' + str(lg)})
    dfx['date_num'] = dfx['date_num'] + lg
    dffin = pd.merge(dffin,dfx, on = ['date_num'], how = 'left')
#dffin = dffin.rename(columns = {'Export_x':'Export_0', 'Export_y':'Export_3'})


#-- rolling window avg
roll = [2,3,4]
dfavg = df['Export'].reset_index().rename(columns = {'index':'date_num'})
for rl in roll:
    dfavg['Export_roll_' + str(rl)] = dfavg['Export'].rolling(rl).mean().shift() #shift will not include the current

dffin = pd.merge(dffin,dfavg, on = ['date_num'], how = 'left')

'''

# Split Data - X and Y datasets are training and testing sets
columns = [
 'Time',
 'Production',
 'Stocks',
 'Export',
 'Import',
 'CPO prices']

columns_scale = [
 'Production',
 'Stocks',
 'Export',
 'Import']


dftrain = df[columns]

##scale the data
scaler_residual = preprocessing.MinMaxScaler(feature_range=(0,1))
df[columns_scale] = scaler_residual.fit_transform(df[columns_scale])

ytrain = dftrain['CPO prices'][dftrain.Time < 48]
ytest = dftrain['CPO prices'][dftrain.Time == 47]

xtrain = dftrain[columns_scale][dftrain.Time < 48]
xtest = dftrain[columns_scale][dftrain.Time == 47]

datetime_residual = dftrain['Time'].head(5)
feat_residual = list(xtrain)

# Build the Random forest Regression model
model_residual = RandomForestRegressor(n_estimators=100).fit(xtrain, ytrain)

# Predict the outcome using Test data - Score Model 
#predicted_test = model.predict(xtest)
prediction = model_residual.predict(xtrain)

#-- Get the metrics
mse = mean_squared_error(ytrain, prediction)
rmse = np.sqrt(mse)
mape = mape(ytrain, prediction)
score = 100 - mape
print(rmse)
print(mape)
print(score)

#-- saving the residual model artifact
model_file = './models/residual_model.pkl'

joblib.dump(value=[datetime_residual, feat_residual, scaler_residual, model_residual], 
            filename=model_file)

#plot the learning curve
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model_residual,
                                                                      xtrain, 
                                                                      ytrain, 
                                                                      cv=4,
                                                                      scoring="neg_mean_squared_error",
                                                                      return_times=True)
dict_learningcurve = {}
dict_learningcurve['sample_size']= train_sizes.tolist()
dict_learningcurve['train_score']= np.mean(train_scores,axis=1).tolist()
dict_learningcurve['test_score']= np.mean(test_scores,axis=1).tolist()

df_lc = pd.DataFrame(dict_learningcurve)

#-- save lc data
df_lc.to_csv('./data/lc.csv')
 

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(train_sizes,np.mean(train_scores,axis=1), label = 'train')
ax2.plot(train_sizes,np.mean(test_scores,axis=1), label = 'test')
ax2.set_title('residual model learning curve')
ax2.legend(loc=4)



#------------------------------------------------
# Combining the seasonal and residual
#------------------------------------------------

df_seasonal0['pred_seasonal'] = y_pred
df_seasonal0['pred_residual'] = prediction
df_seasonal0['pred_price'] = df_seasonal0['pred_seasonal'] + df_seasonal0['pred_residual']

plt.plot(df_seasonal0.index, df['CPO prices'], linewidth=1, label='Actual')
plt.plot(df_seasonal0.index, df_seasonal0['pred_price'] , linewidth=1, label='Prediction')
plt.legend(loc='upper right')
plt.suptitle('Actual vs prediction - RF')
plt.grid()
#plt.ylim((-25, 25))
plt.xticks(rotation=90) 

#-- Get the metrics
preds = df_seasonal0['pred_price'].to_list()
actuals = df['CPO prices'].to_list()

mse = mean_squared_error(actuals, preds)
rmse = np.sqrt(mse)
mape = mape(actuals, preds)
score = 100 - mape
print(rmse)
print(mape)
print(score)

#-- saving model metrics 
metrics_residual = {}
metrics_residual['rmse'] = round(rmse, 2)
metrics_residual['mape'] = round(mape, 2)
metrics_residual['score'] = round(score, 2)

with open('./data/metrics_model.txt', 'w') as outfile:
    json.dump(metrics_residual, outfile)  



#-- open and read json file
with open('./data/metrics_model.txt', 'r') as json_file:
    stg = json.load(json_file)

#------------------------------------------------
# testing the model for prediction


#-- load the models

seasonal_model_path = './models/seasonal_model.pkl'
residual_model_path = './models/residual_model.pkl'

dictft, model_season = joblib.load(seasonal_model_path)
scaler_resid, model_resid = joblib.load(residual_model_path)


ftvalue, df_seasonal0 = seasonal_model(dictft,18)


pred_season =  model_season.predict(pd.DataFrame({'Time':[18], 'FT_All':[ftvalue[0]]}))

new_data = {'Production': [0.103911],  'Stocks':[0.099682],  'Export':[0.036171],  'Import':[0.099978]}

pred_resid = model_resid.predict(pd.DataFrame(new_data))

pred_final = pred_season + pred_resid

print(pred_final)


#--------------------------------------------------
# plotly plot

#-- save historical and prediction data
filename = './data/prediction.csv'
df.to_csv('./data/prediction.csv')

#-- to render plot in browser
import plotly.io as pio
pio.renderers.default='browser' # 'svg' for offline

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.YearMonth, y=df['CPO prices'],
                    mode='lines',
                    name='actual'))
fig.add_trace(go.Scatter(x=df.YearMonth, y=df['pred_price'],
                    mode='lines',
                    name='prediction'))
fig.update_layout(title='CPO price actual vs prediction',
                   xaxis_title='Month',
                   yaxis_title='Prices [MYR]')
fig.show()


#-- plot the lc
from plotly.tools import make_subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=("LC1", "LC2 (placeholder)"), shared_yaxes=True )
fig.add_trace(go.Scatter(x=df_lc.sample_size, y=df_lc['train_score'],
                    mode='lines',
                    name='train'
                    ),
                     row=1, col=1)
fig.add_trace(go.Scatter(x=df_lc.sample_size, y=df_lc['test_score'],
                    mode='lines',
                    name='test'
                    ),
                    row=1, col=1)
fig.add_trace(go.Scatter(x=df_lc.sample_size, y=df_lc['train_score'],
                    mode='lines',
                    name='train'
                    ),
                    row=1, col=2)
fig.add_trace(go.Scatter(x=df_lc.sample_size, y=df_lc['test_score'],
                    mode='lines',
                    name='test'
                    ),
                    row=1, col=2)
fig.update_layout(height=600, width=800, title_text="Learning curve")

# Update xaxis properties
fig.update_xaxes(title_text="Data size", row=1, col=1)
fig.update_xaxes(title_text="data size (placeholder)", row=1, col=2)

# Update yaxis properties
fig.update_yaxes(title_text="Score", row=1, col=1)
fig.update_yaxes(title_text="score (placeholder)", row=1, col=2)

fig.show()










    
