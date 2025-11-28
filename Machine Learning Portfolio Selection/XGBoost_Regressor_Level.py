# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 09:48:48 2025

@author: pbrock
"""

#%% Libraries

path = "C:/Users/pbrock/Desktop/ML/"

#DataFrame Libraries
import pandas as pd
import sqlite3

#Turn off pandas performance warnings
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#Scientifiy Libraries
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import optuna

import os
os.chdir(path + "Code/")
import General_Functions as GF

#%% Preliminaries & Data

#=================================================
#               Preliminaries
#=================================================

#Database
JKP_Factors = sqlite3.connect(database=path +"Data/JKP_clean.db")
db_Predictions = sqlite3.connect(database=path +"Data/Predictions.db")


#Get Settings & signals
settings = GF.get_settings()
signals = GF.get_signals()
feat_cols = signals[0] + signals[1] #Continuous and Categorical Features


#=================================================
#           Rolling Window Parameters
#=================================================

#Window Size and Validation Periods for rolling window
window_size = settings['rolling_window']['window_size']
validation_size = settings['rolling_window']['validation_periods'] 
test_size = settings['rolling_window']['test_size'] #Periods until hyperparameters are re-tuned. Fine-tuning is done monthly

trade_start = settings['rolling_window']['trading_start']
trade_end = settings['rolling_window']['trading_end']
trading_dates = pd.date_range(trade_start, trade_end, freq = 'ME')


#=================================================
#               Model Type
#=================================================
model_name = "XGBReg"
target_type = "LevelTarget"
file_end = f"CRSPUniverse_RankFeatures_RollingWindow_win{window_size}_val{validation_size}_test{test_size}"


#=================================================
#               Read in Data
#=================================================

#Read in processed signals
query = ("SELECT * FROM Signals_Rank "
         f"WHERE eom >= '{(trade_start-pd.offsets.MonthEnd(window_size+validation_size+1)).strftime("%Y-%m-%d")}' AND eom <= '{(trade_end + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")}'")
df = pd.read_sql_query(query,
                               con = JKP_Factors,
                               parse_dates = {'eom'})

#Read in Targets
query = ("SELECT id, eom, tr_m_sp500_ld1 FROM Factors_processed "
         f"WHERE eom >= '{(trade_start-pd.offsets.MonthEnd(window_size+validation_size+1)).strftime("%Y-%m-%d")}' AND eom <= '{(trade_end + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")}'")
df_targets = pd.read_sql_query(query,
                               con = JKP_Factors,
                               parse_dates = {'eom'})

    
#Merge to signals (Merging in Python was quicker than via SQL)
df = (df.merge(df_targets, on = ['id','eom'], how = 'left')
      .dropna(subset = 'tr_m_sp500_ld1')
      .sort_values(by = ['eom','id'])
      )

if df['tr_m_sp500_ld1'].isna().sum() > 0:
    print("ERROR: Missing Values in Target. Training cannot be initiated")
del df_targets


#Extract unique Months of signals
signal_months = df['eom'].sort_values().unique()

#Extract index of trading start
trade_idx = signal_months.searchsorted(trade_start)

#%% XGBoost Preparation

#=================================================
#               XGBoost Regression Tree
#=================================================

#Optuna Objective Function
def make_objective(X_train, y_train, X_val, y_val):

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": 0,  # no L1
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            
            # FIX n_estimators (M) to a high value high; early stopping decides the effective number of M
            "n_estimators": 500,
            "eval_metric": "rmse",
            'early_stopping_rounds': 50
        }

        model = XGBRegressor(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        # Save best_iteration for this trial (number of trees M)
        trial.set_user_attr("best_iteration", model.best_iteration)

        y_val_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_val_pred)
        return mse  # Optuna will minimize this
        
    return objective

#%% Rolling Window Estimation

#Empty list to store the predictions
predictions = []

# Track how many periods since last hyperparameter tuning
months_since_tune = test_size   # start so that the first iteration triggers tuning
saved_best_params = None
saved_best_n_estimators = None

#Loop over dates (Rolling Window)
for date in trading_dates:
    
    #========================================
    # Get the Data for the Rolling Regression
    #========================================
    
    #Rolling Window Dates
    start_idx = max(0, trade_idx - (window_size + validation_size+1))
    window_months = signal_months[start_idx:trade_idx-1]
    #Note: we can only use signals up to t-2 since we are predicting next period's targelt. Else-wise: Look-ahead bias (data leakage)
    
    # Split into train / val months
    train_months = window_months[:window_size]
    val_months   = window_months[window_size:]
    
    # Training Data
    X_train = df[df['eom'].isin(train_months)][feat_cols].to_numpy()
    y_train = df[df['eom'].isin(train_months)]['tr_m_sp500_ld1'].to_numpy()
    
    #Validation Data
    X_val = df[df['eom'].isin(val_months)][feat_cols].to_numpy()
    y_val = df[df['eom'].isin(val_months)]['tr_m_sp500_ld1'].to_numpy()
    
    
    #======================================================
    #     Hyperparameter tuning every `test_size` months
    #======================================================
    if months_since_tune >= test_size:
    
        print(f"Re-tuning hyperparameters at date {date}...")
    
        # Run Optuna tuning
        study = optuna.create_study(direction="minimize")
        objective = make_objective(X_train, y_train, X_val, y_val)
        study.optimize(objective, n_trials=15)
    
        # Extract optimal hyperparameters
        best_params = study.best_params.copy()
    
        # Add fixed parameters
        best_params.update({
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 2025,
            "reg_alpha": 0.0,
            "eval_metric": "rmse"
        })
    
        # Retrieve optimal number of trees from early stopping
        best_iteration = study.best_trial.user_attrs["best_iteration"]
        best_n_estimators = best_iteration + 1
        best_params["n_estimators"] = best_n_estimators
        print(f"Number of trees used for final model at date {date}: {best_n_estimators}")

    
        # Store hyperparameters for future months
        saved_best_params = best_params.copy()
        saved_best_n_estimators = best_n_estimators
    
        months_since_tune = 0  # reset counter
    
    else:
        # Reuse previously tuned hyperparameters
        best_params = saved_best_params.copy()
        best_params["n_estimators"] = saved_best_n_estimators
        print(f"Skipping hyperparameter tuning at date {date}, reusing saved params.")
    
    # Increment months since last tune
    months_since_tune += 1


    #===========================================================
    #           Refit on Train & Validation Data
    #===========================================================
    
    #Get Train + Val Data
    window_mask = df['eom'].isin(window_months)
    X_window = df.loc[window_mask, feat_cols]
    y_window = df.loc[window_mask, 'tr_m_sp500_ld1']
    
    #Train the Model given the hyperparameters
    final_model = XGBRegressor(**best_params)
    final_model.fit(
        X_window, y_window
    )    
        
    
    #===========================================================
    #           Predict next month's OOS Return
    #===========================================================
    
    # Make the next month prediction for the out-of-sample period
    test_mask = (df['eom'] == date-pd.offsets.MonthEnd(1))
    X_test = df.loc[test_mask, feat_cols]
    ids_test = df.loc[test_mask, ['id', 'eom']]
    
    y_test_pred = final_model.predict(X_test)    
    
    
    #===========================================================
    #                   Save results
    #===========================================================
    
    #At 'eom', predict return for 'eom'+1
    pred_df = ids_test.copy()
    pred_df['ret_pred'] = y_test_pred
    predictions.append(pred_df)
    
    # Save the model for this date
    model_path = os.path.join(path, "Models/XGBoost/", f"{model_name}_{target_type}_{file_end}_date_{date.strftime('%Y%m%d')}.json")
    final_model.save_model(model_path)
    
    # Dump tree structure in text format
    tree_path = os.path.join(path, "Models/XGBoost/", f"{model_name}_trees_{target_type}_{file_end}_date_{date.strftime('%Y%m%d')}.txt")
    final_model.get_booster().dump_model(tree_path)


    #Increment index due to new trading_month
    trade_idx += 1

#%% Save Predictions
df_predictions = pd.concat(predictions)

df_predictions.to_sql(name = f"{model_name}_{target_type}_{file_end}",
                   con = db_Predictions,
                   index = False,
                   if_exists = 'append')

JKP_Factors.close()
db_Predictions.close()