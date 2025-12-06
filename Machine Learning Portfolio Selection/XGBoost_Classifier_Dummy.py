# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 10:08:43 2025

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

#Scientific Libraries
import numpy as np
from xgboost import XGBClassifier          
from sklearn.metrics import log_loss       
import optuna

import os
os.chdir(path + "Code/")
import General_Functions as GF

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

#Target
target_col = 'tr_m_sp500_ld1_Dummy'

#Load Data
df, signal_months, trading_month_start, feat_cols, \
    window_size, validation_size  \
        = GF.load_signals_rollingwindow(db_conn = JKP_Factors,          #Database with signals
                                        settings = settings,            #General settings
                                        target = target_col,            #Prediction target
                                        rank_signals = True,            #Use ZScores
                                        trade_start = '2003-01-31',     #First trading date
                                        trade_end = '2024-12-31',       #Last trading date
                                        fill_missing_values = False,    #Fill missing values 
                                        )

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
#        Model Type (Requires Manual Naming)
#=================================================
model_name = "XGBClass"
target_type = "trmsp500DummyTarget"
file_end = f"CRSPUniverse_RankFeatures_RollingWindow_win{window_size}_val{validation_size}_test{test_size}"


#%% XGBoost Preparation

def prepare_data(df, months, target_col):
    X = df[df['eom'].isin(months)][feat_cols].to_numpy()
    y = df[df['eom'].isin(months)][target_col].to_numpy().astype(int)
    
    return X, y

#=================================================
#          XGBoost Binary Classification
#=================================================

#Optuna Objective Function
def make_objective(X_train, y_train, X_val, y_val, scale_pos_weight_base):

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 2025,
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            "reg_alpha": 0,  # no L1
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            
            # Class imbalance weighting
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight",
                0.5 * scale_pos_weight_base,
                2.0 * scale_pos_weight_base
            ),

            # FIX n_estimators (M) to a high value; early stopping decides effective M
            "n_estimators": 500,
            "eval_metric": "logloss",
            "early_stopping_rounds": 50
        }

        model = XGBClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Save best_iteration for this trial (number of trees M)
        trial.set_user_attr("best_iteration", model.best_iteration)

        # Predict probabilities for positive class
        y_val_proba = model.predict_proba(X_val)[:, 1]
        loss = log_loss(y_val, y_val_proba)
        return loss  # Optuna will minimize this

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
    # Get the Data for the Rolling Model
    #========================================

    #Rolling Window Dates
    start_idx = max(0, trade_idx - (window_size + validation_size + 1))
    window_months = signal_months[start_idx:trade_idx-1]
    #Note: we can only use signals up to t-2 since we are predicting next period's target

    # Split into train / val months
    train_months = window_months[:window_size]
    val_months   = window_months[window_size:]

    # Training Data
    X_train, y_train = prepare_data(df, train_months, target_col)
    
    #Validation Data
    X_val, y_val = prepare_data(df, val_months, target_col) 

    #======================================================
    #     Hyperparameter tuning every `test_size` months
    #======================================================
    if months_since_tune >= test_size:
        
        # ---------------------------------------------------
        # Class imbalance for this training window
        # ---------------------------------------------------
        pos = np.sum(y_train == 1)
        neg = np.sum(y_train == 0)
        scale_pos_weight_base = neg / max(pos, 1)  # avoid division by zero

        print(f"Re-tuning hyperparameters at date {date}...")

        # Run Optuna tuning
        study = optuna.create_study(direction="minimize")
        objective = make_objective(X_train, y_train, X_val, y_val, scale_pos_weight_base)
        study.optimize(objective, n_trials=15)

        # Extract optimal hyperparameters
        best_params = study.best_params.copy()

        # Add fixed parameters
        best_params.update({
            "objective": "binary:logistic",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 2025,
            "reg_alpha": 0.0,
            "eval_metric": "logloss"
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
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_window, y_window)

    #===========================================================
    #           Predict next month's OOS Classification
    #===========================================================

    # Make the next month prediction for the out-of-sample period
    test_mask = (df['eom'] == date - pd.offsets.MonthEnd(1))
    X_test = df.loc[test_mask, feat_cols].to_numpy()
    ids_test = df.loc[test_mask, ['id', 'eom']]

    # Predicted probability of positive return (class 1)
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    # Hard classification with 0.5 threshold
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    #===========================================================
    #                   Save results
    #===========================================================

    #At 'eom', predict dummy for 'eom'+1
    pred_df = ids_test.copy()
    pred_df['prob_up'] = y_test_proba     # probability of positive return
    #pred_df['dummy_pred'] = y_test_pred   # predicted class (0/1)
    predictions.append(pred_df)

    # Save the model for this date
    model_path = os.path.join(path, "Models/XGBoost/", f"{model_name}_{target_type}_{file_end}_date_{date.strftime('%Y%m%d')}.json")
    final_model.save_model(model_path)

    # Dump tree structure in text format
    tree_path = os.path.join(path, "Models/XGBoost/", f"{model_name}_trees_{target_type}_{file_end}_date_{date.strftime('%Y%m%d')}.txt")
    final_model.get_booster().dump_model(tree_path)

#%% Save Predictions
df_predictions = pd.concat(predictions)

df_predictions.to_sql(
    name = f"{model_name}_{target_type}_{file_end}",
    con = db_Predictions,
    index = False,
    if_exists = 'append'
)

JKP_Factors.close()
db_Predictions.close()
