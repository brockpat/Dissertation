# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 10:14:16 2025

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
import optuna

import os
os.chdir(path + "Code/")
import General_Functions as GF

#%% Functions

def spearman_corr(a, b):
    """
    Spearman rank correlation without scipy.
    """
    a = pd.Series(a)
    b = pd.Series(b)
    a_rank = a.rank(method="average")
    b_rank = b.rank(method="average")
    return np.corrcoef(a_rank, b_rank)[0, 1]


def mean_spearman_ic_by_group(y_true, y_pred, group_sizes):
    """
    Compute mean Spearman IC across groups (e.g. months).

    Parameters
    ----------
    y_true : 1D array-like
        True target values, ordered by group (month).
    y_pred : 1D array-like
        Predicted values, same order as y_true.
    group_sizes : list of int
        Number of observations in each group, in the same order as y_true.

    Returns
    -------
    float
        Mean Spearman IC across groups.
    """
    ics = []
    idx = 0
    for g in group_sizes:
        y_t = y_true[idx:idx + g]
        y_p = y_pred[idx:idx + g]
        idx += g

        # Skip degenerate cases
        if len(np.unique(y_t)) < 2 or len(np.unique(y_p)) < 2:
            continue

        ic = spearman_corr(y_t, y_p)
        if not np.isnan(ic):
            ics.append(ic)

    if len(ics) == 0:
        return np.nan
    return float(np.nanmean(ics))

#%% Preliminaries & Data

#=================================================
#               Preliminaries
#=================================================

#Database
JKP_Factors = sqlite3.connect(database=path +"Data/JKP_processed.db")
db_Predictions = sqlite3.connect(database=path +"Data/Predictions.db")


#Get Settings & signals
settings = GF.get_settings()
signals = GF.get_signals()
feat_cols = signals[0] + signals[1] #Continuous and Categorical Features


#=================================================
#               Read in Data
#=================================================

#Load Data
df, signal_months, trading_month_start, feat_cols, \
    window_size, validation_size  \
        = GF.load_signals_rollingwindow(db_conn = JKP_Factors,          #Database with signals
                                        settings = settings,            #General settings
                                        target = 'tr_ld1',              #Prediction target
                                        rank_signals = True,            #Use ZScores
                                        trade_start = '2003-01-31',     #First trading date
                                        trade_end = '2024-12-31',       #Last trading date
                                        fill_missing_values = False,    #Fill missing values 
                                        )


#=================================================
#      Additional: Rank Standardise Target
#=================================================

#Note that it is irrelevant what return we learn as ranks are invariant with respect to monotonic transformations
target_col = 'tr_ld1_rank'

df[target_col] = (df.groupby('eom')['tr_ld1']
                  .transform(lambda x: x.rank(pct = True)*2-1)
                  )
df = df.drop(columns = 'tr_ld1')


#=================================================
#           Rolling Window Parameters
#=================================================

#Window Size and Validation Periods for rolling window
window_size = settings['rolling_window']['window_size']
validation_size = settings['rolling_window']['validation_periods'] 
test_size = settings['rolling_window']['test_size'] #Periods until hyperparameters are re-tuned. Fine-tuning is done monthly

#Trading Dates
trading_dates = signal_months[trading_month_start:]


#=================================================
#               Model Type
#=================================================
model_name = "XGBReg"
target_type = "RankTarget"
file_end = f"CRSPUniverse_RankFeatures_RollingWindow_win{window_size}_val{validation_size}_test{test_size}"


#%% XGBoost Preparation

def prepare_data(df, months, target_col):
    X = df[df['eom'].isin(months)].sort_values(['eom', 'id'])[feat_cols].to_numpy()
    y = df[df['eom'].isin(months)].sort_values(['eom', 'id'])[target_col].to_numpy()
    
    return X, y

#=================================================
#               XGBoost Regression Tree
#=================================================

#Optuna Objective Function
def make_objective(X_train, y_train, X_val, y_val, group_val):
    
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 2025,
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.01, 5.0), #Too few splits = Too few buckets and very coarse rank prediction
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0), #Too few splits = Too few buckets and very coarse rank prediction
            "reg_alpha": 0,  # no L1
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0001, 5.0, log=True), #too high penalty might overflatten the score
            
            # FIX n_estimators (M) to a high value high; early stopping decides the effective number of M
            "n_estimators": 500,
            #"n_estimators": trial.suggest_int("n_estimators", 10, 150),
            "eval_metric": "rmse",
            'early_stopping_rounds': 50
        }

        model = XGBRegressor(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        # Best iteration for this trial
        best_iteration = getattr(model, "best_iteration", params["n_estimators"] - 1)
        trial.set_user_attr("best_iteration", int(best_iteration))

        # ---- Our tuning metric: mean per-month Spearman IC ----
        y_val_pred = model.predict(X_val)
        ic_mean = mean_spearman_ic_by_group(
            y_true=y_val,
            y_pred=y_val_pred,
            group_sizes=group_val,
        )
        # In case of degenerate groups, penalize heavily
        if np.isnan(ic_mean):
            return 1e9

        return -ic_mean
    
    return objective 

#%% Rolling Window Estimation

#Empty list to store the predictions
predictions = []

# Track how many periods since last hyperparameter tuning
months_since_tune = test_size   # start so that the first iteration triggers tuning
saved_best_params = None
saved_best_n_estimators = None

#Loop over dates (Rolling Window)
for trade_idx, date in enumerate(trading_dates, start=trading_month_start):
    
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
    X_train, y_train = prepare_data(df, train_months, target_col)
    group_train = df[df['eom'].isin(train_months)].sort_values(['eom', 'id']).groupby('eom').size().tolist()

    #Validation Data
    X_val, y_val = prepare_data(df, val_months, target_col)
    group_val = df[df['eom'].isin(val_months)].sort_values(['eom', 'id']).groupby('eom').size().tolist()


    #======================================================
    #     Hyperparameter tuning every `test_size` months
    #======================================================
    if months_since_tune >= test_size:
    
        print(f"Re-tuning hyperparameters at date {date}...")
    
        # Run Optuna tuning
        study = optuna.create_study(direction="minimize")
        objective = make_objective(X_train, y_train, X_val, y_val, group_val)
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
    X_window, y_window = prepare_data(df, window_months, target_col)

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
    pred_df['rank_pred'] = y_test_pred
    predictions.append(pred_df)
    
    # Save the model for this date
    model_path = os.path.join(path, "Models/XGBoost/", f"{model_name}_{target_type}_{file_end}_date_{date.strftime('%Y%m%d')}.json")
    final_model.save_model(model_path)
    
    # Dump tree structure in text format
    tree_path = os.path.join(path, "Models/XGBoost/", f"{model_name}_trees_{target_type}_{file_end}_date_{date.strftime('%Y%m%d')}.txt")
    final_model.get_booster().dump_model(tree_path)

#%% Save Predictions
df_predictions = pd.concat(predictions)

df_predictions.to_sql(name = f"{model_name}_{target_type}_{file_end}",
                   con = db_Predictions,
                   index = False,
                   if_exists = 'append')

JKP_Factors.close()
db_Predictions.close()
