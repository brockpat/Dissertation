# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 10:41:17 2025

@author: pbrock
"""

#%% Libraries

path = "C:/Users/pbrock/Desktop/ML/"

#DataFrame Libraries
import pandas as pd
import sqlite3
from pandas.tseries.offsets import MonthEnd

#Turn off pandas performance warnings
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#Plot Libraries
import matplotlib.pyplot as plt

#Scientifiy Libraries
import numpy as np

import os
os.chdir(path + "Code/")
from Functions import *

import scipy.stats

#%% Functions

#Draw omega weights in RFFs
def rff_weights(signals:list, hyperparameters:dict, seed = 2025):
    """
    Draw the omega weights for the RFFs
    Parameters
    ----------
    signals : list
        list of signal names.
    hyperparameters : dict
        contains the hyperparameters for the RFF Regression.
    seed : int, optional
        seed to draw the omega weights

    Returns
    -------
    base_W : np.array
        matrix containing the omega weights.

    """
    num_signals = len(signals[0] + signals[1])
    p_max = hyperparameters['p_vec'].max()
    
    rng = np.random.default_rng(seed)
    # Base random weights for fair sigma comparison (reuse across sigmas)
    base_W = rng.normal(loc=0.0, scale=1.0, size=(num_signals, p_max // 2))
    
    return base_W

#Compute explanatory variable matrix
def rff_Ridge_ZZ(df, sigma:float, p:int, base_W):
    """
    Parameters
    ----------
    df : DataFrame
        contains the data from the signals.
    sigma : float
        variance of RFF omega weights.
    p : int
        No. of RFFs.
    base_W : TYPE
        RFF omega weights with unit variance.
    signals : list
        list of signals to be used for the prediction.

    Returns
    -------
    relevant matrices that are used in the Ridge Regression.

    """
    #Scale the variance of the omega weights
    W = (base_W / sigma)[:,:p]
    
    #Extract the values of the signals
    X = df.to_numpy()  # shape: (n, num_signals)
    
    #Compute the inner products omega'x
    proj = X @ W                    # (n, p/2)
    
    #Compute the RFFs and include a constant
    Z =  np.hstack([np.ones((proj.shape[0], 1)), #Include a constant
                    np.sqrt(1 / (p//2)) * np.cos(proj), 
                    np.sqrt(1 / (p//2)) * np.sin(proj)])
    
    #No. Obs, No. Features
    n, p = Z.shape
    
    #Output the relevant matrices for the Ridge regression estimator computation
    if n > p: #primal form
        return Z.T@Z, Z
    else:  #dual form
        return Z @ Z.T, Z

#Compute the Ridge regression estimator
def rff_Ridge_estimation(ZZ, Z, y, penalty:float):
    """
    Computes the Ridge regression estimator
    
    Parameters
    ----------
    ZZ : array
        Either Z.T@Z or Z @ Z.T depening on whether primal or dual form is lower-dimensional.
    Z : array
        RFF transformation of signals and a constant.
    y : array
        contains next period's return the forecast aims to match.
    penalty : float
        ridge penalty term.

    Returns
    -------
    beta : array
        DESCRIPTION.

    """
    #No. Obs, No. Features
    n, p = Z.shape
    
    #Compute the Ridge regression estimator
    if n > p: #primal form
        ridge_penalty = penalty * np.eye(p)
        ridge_penalty[0, 0] = 0.0  # Don't regularize the intercept
        beta = np.linalg.solve(ZZ + ridge_penalty, Z.T @ y)
    else: #dual form
        print("Here")
        ridge_penalty = penalty * np.eye(n)
        #no setting to zero in dual form
        beta = Z.T @ np.linalg.solve(ZZ + ridge_penalty, y)
    
    return beta

#Validate Ridge regression estimator
def rff_validation(Z_val, val_targets, beta):
    """
    Validates the Ridge regression estimator

    Parameters
    ----------
    Z_val : array
        RFFs of signals for the validation data set.
    val_targets : array
        actual next period's returns.
    beta : array
        Ridge regression estimator.

    Returns
    -------
    R2 : float
        R^2 metric. Evaluates predicted next period's returns vs. actual next period's returns.

    """
    
    #Compute predicted returns
    predicted_returns = Z_val@beta
    
    #Compute R^2
    residuals = predicted_returns - val_targets
    R2 = 1-np.var(residuals)/np.var(val_targets)
    
    return R2

def rff_find_hp(hyperparameters:dict, date_id:str, start_trade_date,
                window_size:int, validation_periods:int, tuning_folds:int):
    
    # Get list of dates
    unique_dates = pd.Series(df_signals[date_id].unique()).sort_values()
      
    # Dates where trading starts
    trade_dates = list(unique_dates[unique_dates >= start_trade_date])
    
    #List to store validation scores
    hp_scores = []

    #Loop over trade dates and get past data
    for trade_date in trade_dates:
        print(f"Trade Date: {trade_date}")
        
        # All past data strictly before this trading month
        past_dates = unique_dates[unique_dates < trade_date - pd.offsets.MonthEnd(1)]
        
        # To create T folds of (window_size -> validation_periods) inside a short lookback,
        # we need at least: window_size + T*validation_periods months of history.
        min_needed = window_size + tuning_folds * validation_periods
        if len(past_dates) < min_needed:
            # Not enough history to tune and fit
            continue
        
        # tuning dates
        tuning_dates = past_dates[-min_needed:]
        
        # Build folds inside this recent window (fixed-length training + next validation)
        folds = []
        for i in range(tuning_folds):
            est_start = i * validation_periods
            est_end = est_start + window_size
            val_start = est_end
            val_end = est_end + validation_periods

            est_dates = tuning_dates[est_start:est_end]
            val_dates = tuning_dates[val_start:val_end]
            folds.append((est_dates, val_dates)) #training data, validation data
        
        # --- inner rolling CV to pick hyperparameters ---
        for train_dates, val_dates in folds:
            train_data = df_signals[df_signals[date_id].isin(train_dates)][signals[0] + signals[1]]
            train_targets = df_targets[df_targets[date_id].isin(train_dates)]['tr_m_sp500_ld1']
            
            val_data = df_signals[df_signals[date_id].isin(val_dates)][signals[0] + signals[1]]
            val_targets = df_targets[df_targets[date_id].isin(val_dates)]['tr_m_sp500_ld1']
            
            
            base_W = rff_weights(signals, hyperparameters)

            for sigma in hyperparameters['sigma_vec']:
                print(f"  sigma:{sigma}")
                for p in hyperparameters['p_vec']:
                    print(f"    p:{p}")
                    ZZ,Z = rff_Ridge_ZZ(train_data, sigma, p, base_W)
                    _, Z_val = rff_Ridge_ZZ(val_data, sigma, p, base_W)

                    for penalty in hyperparameters['penalty_vec']:
                        print(f"      penalty:{penalty}")
                        beta = rff_Ridge_estimation(ZZ, Z, train_targets.to_numpy(), penalty)
                        R2 = rff_validation(Z_val, val_targets, beta)
                        hp_scores.append({'trade_date':trade_date, 'sigma':sigma, 
                                          'p':p, 'penalty':penalty, 'R2':R2}) 
         
    #Return validation metrics for each hyperparameter
    hp_scores = pd.DataFrame(hp_scores)
    return hp_scores

def rff_forecast_all_dates(df_signals, df_targets, signals, hyperparameters_opt,
                           date_id: str, start_trade_date, window_size: int):
    """
    Forecast next-period returns for each trading date using optimal hyperparameters.
    No look-ahead bias: training data strictly precedes trading date.

    Parameters
    ----------
    df_signals : DataFrame
        Signal data with 'date_id' column.
    df_targets : DataFrame
        Target data with 'date_id' column and 'tr_m_sp500_ld1' as target.
    signals : list
        List of signal names to use.
    hyperparameters_opt : dict
        Optimal hyperparameters (keys: 'sigma', 'p', 'penalty').
    date_id : str
        Column name identifying month/date.
    start_trade_date : pd.Timestamp
        First trading date.
    window_size : int
        Length of training window (in months).

    Returns
    -------
    df_forecasts : DataFrame
        Contains columns: [trade_date, actual_return, predicted_return, R2].
    """

    unique_dates = pd.Series(df_signals[date_id].unique()).sort_values()
    trade_dates = list(unique_dates[unique_dates >= start_trade_date])

    base_W = rff_weights(signals, {'p_vec': np.array([hyperparameters_opt['p']])})

    sigma = hyperparameters_opt['sigma']
    p = hyperparameters_opt['p']
    penalty = hyperparameters_opt['penalty']

    forecasts = []

    for trade_date in trade_dates:
        # Use data strictly before this trade date
        past_dates = unique_dates[unique_dates < trade_date - pd.offsets.MonthEnd(1)]
        if len(past_dates) < window_size:
            continue

        train_dates = past_dates[-window_size:]

        # Training data
        train_data = df_signals[df_signals[date_id].isin(train_dates)][signals[0] + signals[1]]
        train_targets = df_targets[df_targets[date_id].isin(train_dates)]['tr_m_sp500_ld1']

        # Forecast month data
        test_data = df_signals[df_signals[date_id] == trade_date][signals[0] + signals[1]]
        test_targets = df_targets[df_targets[date_id] == trade_date]['tr_m_sp500_ld1']

        # Build RFF matrices
        ZZ, Z = rff_Ridge_ZZ(train_data, sigma, p, base_W)
        _, Z_test = rff_Ridge_ZZ(test_data, sigma, p, base_W)

        # Fit ridge model
        beta = rff_Ridge_estimation(ZZ, Z, train_targets.to_numpy(), penalty)

        # Predict
        y_pred = Z_test @ beta
        y_true = test_targets.to_numpy()

        # Compute R^2 for that trading date
        if len(y_true) > 1:
            residuals = y_true - y_pred
            R2 = 1 - np.var(residuals) / np.var(y_true)
        else:
            R2 = np.nan

        forecasts.append({
            'trade_date': trade_date,
            'actual_return': y_true.mean(),
            'predicted_return': y_pred.mean(),
            'R2': R2
        })

    df_forecasts = pd.DataFrame(forecasts)
    return df_forecasts
    
#%% 
#Read in processed characteristics
JKP_Factors = sqlite3.connect(database=path +"Data/JKP_US_SP500.db")
df_signals = pd.read_sql_query("SELECT * FROM Signals_ZScore", 
                               con = JKP_Factors,
                               parse_dates={'eom'})

#Read in Targets
df_targets = pd.read_sql_query(("SELECT id, eom, tr_m_sp500, tr_m_sp500_ld1 FROM Factors_processed"), 
                               con = JKP_Factors,
                               parse_dates={'eom'})

#Get list of signals
signals:list = get_signals()

#Fill missing values with 0. Industry Dummys have no missing values
df_signals[signals[0]] = df_signals[signals[0]].fillna(0)

#Find Optimal Hyperparameters
rff_hp = rff_find_hp(settings['RFF'], 'eom', settings['rolling_window']['trading_month'],
                     settings['rolling_window']['window_size'], settings['rolling_window']['validation_periods'],
                     settings['rolling_window']['tuning_fold'])

rff_hp['window_size'] = settings['rolling_window']['window_size']
rff_hp['validation_periods'] = settings['rolling_window']['validation_periods']

rff_hp.to_sql(name = 'RFF_hp', con = JKP_Factors, if_exists = 'append', index = False)

rff_best_hp = rff_hp.loc[rff_hp.groupby('trade_date')['R2'].idxmax()]


"""
To test the Virtue of Complexity, I can set sigma = 0.5 fix (as in their paper) and then for a fix
p and lambda combination try to replicate their plots
"""