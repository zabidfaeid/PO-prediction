# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 14:17:32 2021

@author: M Zabid M Faeid
"""
import pandas as pd
import joblib
#-- load the models

def load_models():
    seasonal_model_path = './models/seasonal_model.pkl'
    residual_model_path = './models/residual_model.pkl'

    feat_season, dictft, model_season = joblib.load(seasonal_model_path) # [feat_seasonal, fourier_terms_dict, model_seasonal]
    datetime_resid, feat_resid, scaler_resid, model_resid = joblib.load(residual_model_path)  #  [datetime_residual, feat_residual, scaler_residual, model_residual]
    
    
def seasonal_model(dict_ft, chosen_time, sr = 48, time_max = 81):
    import math
    # sr is the signal range that we used at the beginning when we develop the FT
    dt = pd.DataFrame(columns = ['Time'])
    dt['Time'] = list(range(0,time_max))
    for key in dict_ft.keys():
        a = dict_ft[key]['amplitude']
        w = 2 * math.pi * (dict_ft[key]['freq'] / sr)
        p = dict_ft[key]['phase']
        dt[key] = dt['Time'].apply(lambda t: math.cos(w*t + p))
    dt['FT_All'] = 0
    for k, v in dict_ft.items():
        dt['FT_All'] = dt['FT_All'] + dt[k]
    
    ft_value = dt['FT_All'].loc[dt.Time == chosen_time].values
    
    return ft_value, dt