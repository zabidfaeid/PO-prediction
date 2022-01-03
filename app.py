# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 14:11:27 2021

@author: M Zabid M Faeid
"""
import json
import joblib
from flask import Flask, render_template, request#, jsonify
import plotly
import plotly.graph_objs as go
import pandas as pd
#import numpy as np
#from po_predict import seasonal_model
from plotly.tools import make_subplots
#from plot_flask import create_plot


app = Flask(__name__)

#-- load the data
df = pd.read_csv('./data/prediction.csv')

#-- load the models
seasonal_model_path = './models/seasonal_model.pkl'
residual_model_path = './models/residual_model.pkl'

feat_season, dictft, model_season = joblib.load(seasonal_model_path) # [feat_seasonal, fourier_terms_dict, model_seasonal]
datetime_resid, feat_resid, scaler_resid, model_resid = joblib.load(residual_model_path)  #  [datetime_residual, feat_residual, scaler_residual, model_residual]

#-- load the model metrics
with open('./data/metrics_model.txt', 'r') as json_file:
    dict_metrics = json.load(json_file)

#-- load the lc data
df_lc = pd.read_csv('./data/lc.csv')


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



@app.route('/', methods = ['GET', 'POST'])


def index():
    if request.method == 'POST':
        
        #-- predicting the residual
        new_data = {'Production': [request.form.get('production')],  
                    'Stocks':[request.form.get('stock')],  
                    'Export':[request.form.get('export')],  
                    'Import':[request.form.get('import')]}
        
        pred_resid = model_resid.predict(pd.DataFrame(new_data))
        
        #-- predicting the seasonal
        timepoint = int(request.form.get('time'))
        ftvalue, df_s = seasonal_model(dict_ft = dictft, chosen_time = timepoint)
        pred_season =  model_season.predict(pd.DataFrame({'Time':[timepoint], 'FT_All':[ftvalue[0]]}))
        
        #-- final prediction
        pred_final = round((pred_season + pred_resid)[0][0], 2)           #-- [0][0] because it is in numpy array
        
        feature = 'Plot1'
        plot_pred = create_plot(feature)
        
        return render_template('results.html', prediction = pred_final, plot = plot_pred)
    else:
        rmse_ = dict_metrics['rmse']
        score_ = dict_metrics['score']
        plot_metr = create_plot_metrics()
        return render_template('index.html', plot = plot_metr, rmse = rmse_, score =score_ )
    
def create_plot(feature):
    
    if feature == 'Plot1':
        data = go.Figure()
        data.add_trace(go.Scatter(x=df.YearMonth, y=df['CPO prices'],
                            mode='lines',
                            name='actual'))
        data.add_trace(go.Scatter(x=df.YearMonth, y=df['pred_price'],
                            mode='lines',
                            name='prediction'))
        data.update_layout(title='CPO price actual vs prediction',
                           xaxis_title='Month',
                           yaxis_title='Prices [MYR]')
        
        
    elif feature == 'Plot2':
        
        data = go.Figure()
        data.add_trace(go.Scatter(x=df.YearMonth, y=df['CPO prices'],
                            mode='lines',
                            name='CPO prices'))
        data.add_trace(go.Scatter(x=df.YearMonth, y=df['Production'],
                            mode='lines',
                            name='CPO production',
                            yaxis='y2'))
        
        # Create axis objects
        data.update_layout(
            yaxis=dict(
                title="CPO prices",
                titlefont=dict(
                    color="#1f77b4"
                    ),
                tickfont=dict(
                    color="#1f77b4"
                    )
                ),
            yaxis2=dict(
                title="CPO production",
                titlefont=dict(
                    color="#d62728"
                    ),
                tickfont=dict(
                    color="#d62728"
                    ),
                anchor="x",
                overlaying="y",
                side="right",
                #position=0.15
                )
            )
        # Update layout properties
        data.update_layout(title='CPO price vs production',
                           xaxis_title='Month'
                           #width=1000
                           )
        
    elif feature == 'Plot3':
        
        data = go.Figure()
        data.add_trace(go.Scatter(x=df.YearMonth, y=df['Export'],
                            mode='lines',
                            name='CPO prices'))
        data.add_trace(go.Scatter(x=df.YearMonth, y=df['Import'],
                            mode='lines',
                            name='CPO production',
                            yaxis='y2'))
        
        # Create axis objects
        data.update_layout(
            yaxis=dict(
                title="CPO export",
                titlefont=dict(
                    color="#1f77b4"
                    ),
                tickfont=dict(
                    color="#1f77b4"
                    )
                ),
            yaxis2=dict(
                title="CPO import",
                titlefont=dict(
                    color="#d62728"
                    ),
                tickfont=dict(
                    color="#d62728"
                    ),
                anchor="x",
                overlaying="y",
                side="right",
                #position=0.15
                )
            )
        # Update layout properties
        data.update_layout(title='CPO export vs import',
                           xaxis_title='Month'
                           #width=1000
                           )
        
    

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def create_plot_metrics():
    
    
    metrics_plot = make_subplots(rows=1, cols=2, subplot_titles=("LC1", "LC2 (placeholder)"), shared_yaxes=True )
    metrics_plot.add_trace(go.Scatter(x=df_lc.sample_size, y=df_lc['train_score'],
                        mode='lines',
                        name='train'
                        ),
                         row=1, col=1)
    metrics_plot.add_trace(go.Scatter(x=df_lc.sample_size, y=df_lc['test_score'],
                        mode='lines',
                        name='test'
                        ),
                        row=1, col=1)
    metrics_plot.add_trace(go.Scatter(x=df_lc.sample_size, y=df_lc['train_score'],
                        mode='lines',
                        name='train'
                        ),
                        row=1, col=2)
    metrics_plot.add_trace(go.Scatter(x=df_lc.sample_size, y=df_lc['test_score'],
                        mode='lines',
                        name='test'
                        ),
                        row=1, col=2)
    metrics_plot.update_layout(height=600, width=800, title_text="Learning curve")

    # Update xaxis properties
    metrics_plot.update_xaxes(title_text="Data size", row=1, col=1)
    metrics_plot.update_xaxes(title_text="data size (placeholder)", row=1, col=2)

    # Update yaxis properties
    metrics_plot.update_yaxes(title_text="Score", row=1, col=1)
    metrics_plot.update_yaxes(title_text="score (placeholder)", row=1, col=2)
    
    graphJSON = json.dumps(metrics_plot, cls=plotly.utils.PlotlyJSONEncoder)
    
    return graphJSON


@app.route('/bar', methods=['GET', 'POST'])
def change_features():

    feature = request.args['selected']
    graphJSON= create_plot(feature)

    return graphJSON
    
if __name__ == '__main__':
    app.run(debug = True) #-- use port other than 5000