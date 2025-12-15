# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 20:14:57 2025

@author: patri

Long-Short Portfolios.

Transaction costs omitted as these are not designed for long-short portfolios.


Long-Short portfolios do not work at all on S&P 500 stocks alone, but they
work unfathomably well on the entire CRSP universe (however, no transaction
costs such as short-selling fees, slippage, portfolio turnover, etc... are
accounted for.)

Z-scores are better than rank predictions and rank predictions are better than
level predictions
"""

#%% Libraries

path = "C:/Users/patri/Desktop/ML/"

#DataFrame Libraries
import pandas as pd
import sqlite3

#Turn off pandas performance warnings
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#Plot Libraries
import matplotlib.pyplot as plt

#Scientifiy Libraries
import numpy as np

#Custom Functions
import os
os.chdir(path + "Code/")
import General_Functions as GF
settings = GF.get_settings()

#%% Functions

def long_short_portfolio(df_return_predictions, prediction_col, 
                         df_returns,
                         df_me,
                         long_cutoff = 0.9, short_cutoff = 0.1,
                         value_weighted = False):
    
    #===========================
    # Select Long & Short Stocks
    #===========================
    
    #Cross-Sectional Quantile cutoffs
    grouped = df_return_predictions.groupby('eom')[prediction_col]
    q_long = grouped.transform('quantile', long_cutoff)
    q_short= grouped.transform('quantile', short_cutoff)
    
    #---- Determine Positions ----
    #Conditions determining Long or Short
    conditions = [df_return_predictions[prediction_col] >=q_long,
                  df_return_predictions[prediction_col] <= q_short]
    
    #Numerical Position Value
    position = [1,-1]

    #Generating dataframe
    df_ls = df_return_predictions[['id','eom']]
    df_ls['position'] = np.select(conditions, position, default = 0)
    
    #Filter out zero positions
    df_ls = df_ls.loc[df_ls['position'] != 0]
    
    #===========================
    # Compute Portfolio weight
    #===========================
    
    if value_weighted:
        # Merge Market Equity
        df_ls = df_ls.merge(df_me, on = ['id','eom'], how = 'left')
        
        # Error Handling
        if df_ls['me'].isna().sum() > 0:
            print("ERROR: Missing market equity")
            
            return None, None
            
        #Compute weights
        df_ls = df_ls.assign(weight = lambda df:
                             df['me']/df.groupby(['eom','position'])['me'].transform('sum'))
    
    else: #equal weighted
        df_ls = df_ls.assign(weight = lambda df: 1/df.groupby(['eom','position'])['id'].transform('count'))

    #==============================
    #  Compute Individual Revenue
    #==============================
    
    #Merge realised Return
    df_ls = df_ls.merge(df_returns[['id','eom','tr_ld1']], on = ['id','eom'],
                        how = 'left')
    
    #Compute revenue
    df_ls['strategy_profit'] = df_ls['position']*df_ls['weight']*df_ls['tr_ld1']
    
    #==============================
    #  Compute Aggregated Revenue
    #==============================
    
    df_profit = (df_ls
                 .groupby('eom')['strategy_profit'].sum()
                 .reset_index()
                 .rename(columns = {0: 'strategy_profit'})
                 )
    df_profit['cumulative_return'] = (1 + df_profit['strategy_profit']).cumprod()

    
    return df_ls, df_profit

def long_decile_portfolio():
    """
    Only buys the top 10% of stocks with the highest predicted returns

    Returns
    -------
    None.

    """
    
#%% Data

#DataBases
JKP_Factors = sqlite3.connect(database = path + "Data/JKP_processed.db")
SP500_Constituents = sqlite3.connect(database = path + "Data/SP500_Constituents.db")
Benchmarks = sqlite3.connect(database = path + "Data/Benchmarks.db")
Models = sqlite3.connect(database = path + "Data/Predictions.db")

#============================
#       Trading Dates
#============================
#Trading dates
trading_start, trading_end = pd.to_datetime("2004-01-31"), settings['rolling_window']['trading_end']
trading_dates = pd.date_range(start=trading_start,
                              end=trading_end,
                              freq='ME'
                              )

#Start and End Date as Strings
start_date = str(trading_start - pd.offsets.MonthEnd(1))[:10]
end_date = str(trading_end)[:10]

#============================
#       Risk-free rate
#============================

# Read risk-free rate data (select only 'yyyymm' and 'RF' columns).
risk_free = (pd.read_csv(path + "Data/FF_RF_monthly.csv", usecols=["yyyymm", "RF"])
             .assign(rf = lambda df: df["RF"]/100)
             .assign(eom = lambda df: pd.to_datetime(df["yyyymm"].astype(str) + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0))
             .get(['eom','rf'])
             )

#============================
#       S&P 500 Universe
#============================

#Stocks that at date t are in the S&P 500
sp500_constituents = (pd.read_sql_query("SELECT * FROM SP500_Constituents_monthly", #" WHERE eom >= '{start_date}'",
                                       con = SP500_Constituents,
                                       parse_dates = {'eom'})
                      .rename(columns = {'PERMNO': 'id'})
                      )

#============================
#           Data 
#============================

df = pd.read_sql_query("SELECT id, eom, me, tr_ld1 FROM Factors_processed",
                       con = JKP_Factors,
                       parse_dates = {'eom'})

df_returns = df[['id','eom','tr_ld1']]
df_me = df[['id','eom','me']]
del df

#==================================================
#      Model Predictions (Requires Manual Updating)
#==================================================

# Define List of Models
models = [ "XGBClass_trmsp500DummyTarget_CRSPUniverse_RankFeatures_RollingWindow_win120_val12_test12",
            "XGBReg_LevelTrMsp500Target_CRSPUniverse_RankFeatures_RollingWindow_win120_val12_test12",
            "XGBReg_RankTrTarget_CRSPUniverse_RankFeatures_RollingWindow_win120_val12_test12",
            "XGBReg_ZscoreTrTarget_CRSPUniverse_RankFeatures_RollingWindow_win120_val12_test12",
            'XGBReg_LevelTrMsp500Target_SP500Universe_RankFeatures_RollingWindow_win120_val12_test12',
            'XGBReg_LevelTrMsp500Target_SP500UniverseFL_RankFeatures_RollingWindow_win120_val12_test12']
#XGBoost_LevelTarget_CRSPUniverse_RankFeatures_RollingWindow_win120_val12_test12

# Return Predictor column of each model
prediction_cols = ["prob_up",#
                  "ret_pred_Levelmsp500",#
                  "rank_pred",#
                  "ret_pred_Zscore",
                  "ret_pred_Levelmsp500",
                  'ret_pred_Levelmsp500'
                  ]

model = models[5]
prediction_col = prediction_cols[5]

#Load return predictions
#At 'eom', predictions are for eom+1
df_retPred = GF.load_MLpredictions(Models, [model]) 

#Truncate Date
df_retPred = df_retPred.loc[df_retPred['eom'].isin(trading_dates-pd.offsets.MonthEnd(1))]

#Truncate to S&P500 stocks
df_retPred = (df_retPred
              .merge(sp500_constituents.assign(in_sp500 = True),
                              on = ['id','eom'], how = 'left')
              .pipe(lambda df: df.loc[df['in_sp500'] == True])
              .pipe(lambda df: df.drop(columns = 'in_sp500'))
              .reset_index(drop = True)
              )

#%% Long-Short portfolio
df_strategy, df_profit = long_short_portfolio(df_retPred, prediction_col, 
                         df_returns,
                         df_me)