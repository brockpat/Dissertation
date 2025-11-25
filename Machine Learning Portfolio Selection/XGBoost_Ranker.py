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

#Plot Libraries
import matplotlib.pyplot as plt 

#Scientifiy Libraries
import numpy as np
from xgboost import XGBRanker
from sklearn.metrics import mean_squared_error
import optuna

import os
os.chdir(path + "Code/")
import General_Functions as GF

#%% Functions

def build_group_sizes(eom_series):
    """
    Given a sorted eom Series for a dataset, return 
    a list of group sizes in the order they appear.
    """
    return eom_series.value_counts(sort=False).tolist()


def spearman_corr(a, b):
    """
    Spearman rank correlation without needing scipy.
    """
    a = pd.Series(a)
    b = pd.Series(b)
    a_rank = a.rank(method="average")
    b_rank = b.rank(method="average")
    return np.corrcoef(a_rank, b_rank)[0, 1]


def mean_spearman_ic_by_group(y_true, y_pred, group_sizes):
    """
    Compute Spearman IC for each group (month), then average across groups.
    This aligns extremely well with cross-sectional stock ranking use cases.
    """
    ics = []
    idx = 0
    for g in group_sizes:
        y_t = y_true[idx:idx+g]
        y_p = y_pred[idx:idx+g]
        idx += g
        
        # Skip degenerate cases (all equal)
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
JKP_Factors = sqlite3.connect(database=path +"Data/JKP_clean.db")
db_Predictions = sqlite3.connect(database=path +"Data/Predictions.db")


#Get Settings & signals
settings = GF.get_settings()
signals = GF.get_signals()
feat_cols = signals[0] + signals[1]

#Window Size and Validation Periods for rolling window
window_size = 120 #10 years of data
validation_size = 12 #1 year of data
test_size = 12 #Periods until hyperparameters are re-tuned. Fine-tuning is done monthly

file_end = f"Ranker_CRSPUniverse_RankFeatures_RollingWindow_win{window_size}_val{validation_size}_test{test_size}"

#Trading Dates
trading_dates = pd.date_range(settings['rolling_window']['trading_month'], "2024-12-31", freq = 'ME')

#=================================================
#               Read in Data
#=================================================

#Read in processed signals
query = ("SELECT * FROM Signals_Rank "
         f"WHERE eom >= '{(trading_dates[0]-pd.offsets.MonthEnd(window_size+validation_size+1)).strftime("%Y-%m-%d")}'")
df_signals = pd.read_sql_query(query,
                               con = JKP_Factors,
                               parse_dates = {'eom'})

#Read in Targets
query = ("SELECT id, eom, tr_m_sp500_ld1 FROM Factors_processed "
         f"WHERE eom >= '{(trading_dates[0]-pd.offsets.MonthEnd(window_size+validation_size+1)).strftime("%Y-%m-%d")}'")
df_targets = pd.read_sql_query(query,
                               con = JKP_Factors,
                               parse_dates = {'eom'})

#Create Target Ranks
df_targets['tr_m_sp500_ld1'] = (
    df_targets.groupby('eom')['tr_m_sp500_ld1']
              .transform(lambda x: x.rank(method = "dense"))
)
    

#Merge (Merging in Python was quicker than via SQL)
df_signals = (df_signals
              .merge(df_targets, on = ['id','eom'], how = 'left')
              .dropna(subset = 'tr_m_sp500_ld1')
              .assign(tr_m_sp500_ld1 = lambda df: df['tr_m_sp500_ld1'].astype(int))
              .sort_values(by = ['eom','id'])
              )

if df_signals['tr_m_sp500_ld1'].isna().sum() > 0:
    print("ERROR: Missing Values in Target. Training cannot be initiated")
del df_targets


#Extract unique Months of signals
signal_months = df_signals['eom'].sort_values().unique()

#Extract index of trading start
trade_idx = signal_months.searchsorted(trading_dates[0])


#%% XGBoost Preparation

#=================================================
#               XGBoost Regression Tree
#=================================================

#XGBoost base parameters
base_params = dict(
    objective="reg:squarederror", # L2 loss
    tree_method="hist",           # or "approx" / "gpu_hist" if GPU
    learning_rate=0.05,
    n_estimators=300,
    max_depth=3,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.0,
    reg_alpha=0,  # no L1
    reg_lambda=1.0, # L2
    n_jobs=-1,
    random_state=2025,
)

def make_objective_rank(X_train, y_train, group_train,
                        X_val, y_val, group_val):

    def objective(trial):
        params = {
            "objective": "rank:pairwise",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 2025,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": 0.0,  # can also tune if you want
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            "eval_metric":"ndcg",
            "ndcg_exp_gain": False,
            "early_stopping_rounds":50,
            "n_estimators": 1_000,
        }

        model = XGBRanker(**params)

        model.fit(
            X_train, y_train,
            group=group_train,
            eval_set=[(X_val, y_val)],
            eval_group=[group_val],
            verbose=False,
        )

        # Best iteration for this trial
        best_iteration = getattr(model, "best_iteration", None)
        if best_iteration is None:
            best_iteration = params["n_estimators"] - 1
        trial.set_user_attr("best_iteration", int(best_iteration))

        # Our *true* tuning metric: mean per-month Spearman IC
        y_val_pred = model.predict(X_val)
        ic_mean = mean_spearman_ic_by_group(
            y_true=y_val,
            y_pred=y_val_pred,
            group_sizes=group_val,
        )

        # Maximize IC → minimize negative IC
        if np.isnan(ic_mean):
            return 0.0
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
    X_train = df_signals[df_signals['eom'].isin(train_months)].sort_values(['eom', 'id'])[feat_cols].to_numpy()
    y_train = df_signals[df_signals['eom'].isin(train_months)].sort_values(['eom', 'id'])['tr_m_sp500_ld1'].to_numpy()
    group_train = df_signals[df_signals['eom'].isin(train_months)].sort_values(['eom', 'id']).groupby('eom').size().tolist()

    
    #Validation Data
    X_val = df_signals[df_signals['eom'].isin(val_months)].sort_values(['eom', 'id'])[feat_cols].to_numpy()
    y_val = df_signals[df_signals['eom'].isin(val_months)].sort_values(['eom', 'id'])['tr_m_sp500_ld1'].to_numpy()
    group_val = df_signals[df_signals['eom'].isin(val_months)].sort_values(['eom', 'id']).groupby('eom').size().tolist()

    
    
    #======================================================
    #     Hyperparameter tuning every `test_size` months
    #======================================================
    if months_since_tune >= test_size:
    
        print(f"Re-tuning hyperparameters at date {date}...")
    
        study = optuna.create_study(direction="minimize")
        objective = make_objective_rank(
            X_train, y_train, group_train,
            X_val, y_val, group_val
        )
        study.optimize(objective, n_trials=15)
    
        best_params = study.best_params.copy()
        best_params.update({
            "objective": "rank:pairwise",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 2025,
            "reg_alpha": 0.0,
        })
    
        best_iteration = study.best_trial.user_attrs["best_iteration"]
        best_n_estimators = int(best_iteration) + 1
        best_params["n_estimators"] = best_n_estimators
        print(f"Number of trees used for final model at date {date}: {best_n_estimators}")
    
        saved_best_params = best_params.copy()
        saved_best_n_estimators = best_n_estimators
        months_since_tune = 0
    
    else:
        best_params = saved_best_params.copy()
        best_params["n_estimators"] = saved_best_n_estimators
        print(f"Skipping hyperparameter tuning at date {date}, reusing saved params.")
    
    #Increment month since last tune
    months_since_tune += 1


    #===========================================================
    #           Refit on Train & Validation Data
    #===========================================================
    
    #Get Train + Val Data
    window_mask = df_signals['eom'].isin(window_months)
    X_window = df_signals.loc[window_mask].sort_values(['eom', 'id'])[feat_cols]
    y_window = df_signals.loc[window_mask].sort_values(['eom', 'id'])['tr_m_sp500_ld1']
    group_window = df_signals.loc[window_mask].sort_values(['eom', 'id']).groupby('eom').size().tolist()

    
    #Train the Model given the hyperparameters
    final_model = XGBRanker(**best_params)
    final_model.fit(
        X_window, y_window,
        group=group_window
    )  
        
    
    #===========================================================
    #           Predict next month's OOS Return
    #===========================================================
    
    # Make the next month prediction for the out-of-sample period
    test_mask = (df_signals['eom'] == date-pd.offsets.MonthEnd(1))
    X_test = df_signals.loc[test_mask, feat_cols]
    ids_test = df_signals.loc[test_mask, ['id', 'eom']]
    
    y_test_pred = final_model.predict(X_test)    
    
    
    #===========================================================
    #                   Save results
    #===========================================================
    
    #Get Scores and map them into [-1,1]
    pred_df = ids_test.copy()
    pred_df['score'] = (y_test_pred)
    pred_df["rank"] = pred_df["score"].rank(method="average")
    n = pred_df["rank"].max()
    pred_df["pct"] = pred_df["rank"] / (n + 1.0)
    pred_df["pct"] = 2 * pred_df["pct"] - 1
    
    predictions.append(pred_df)
    
    # Save the model for this date
    model_path = os.path.join(path, "Models/XGBoost/", f"xgb_{file_end}_{date.strftime('%Y%m%d')}.json")
    final_model.save_model(model_path)
    
    # Dump tree structure in text format
    tree_path = os.path.join(path, "Models/XGBoost/", f"xgb_trees_{file_end}_{date.strftime('%Y%m%d')}.txt")
    final_model.get_booster().dump_model(tree_path)


    #Increment index due to new trading_month
    trade_idx += 1

#%% Save Predictions
df_predictions = pd.concat(predictions)

df_predictions.to_sql(name = f"XGBoost_{file_end}",
                   con = db_Predictions,
                   index = False,
                   if_exists = 'append')

JKP_Factors.close()
db_Predictions.close()