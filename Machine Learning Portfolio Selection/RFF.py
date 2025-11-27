# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 10:41:17 2025

@author: pbrock
"""

#%% Libraries

path = "C:/Users/pbrock/Desktop/ML/"

# DataFrame Libraries
import pandas as pd
import sqlite3

# Turn off pandas performance warnings
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Scientific Libraries
import numpy as np

# System
import os
os.chdir(path + "Code/") 
import General_Functions as GF 

#To save the model
import json

#%% Functions

def rff_weights(num_cont_features, p_max, seed):
    """
    Sample base frequency vectors (omega) for Random Fourier Features (RFFs).

    The frequencies are drawn i.i.d. from a standard normal distribution and are
    later rescaled by the kernel bandwidth 'sigma' inside rff_feature_map().

    Parameters
    ----------
    num_cont_features : int
        Number of continuous input features.
    p_max : int
        Maximum total number of RFF features to support.
        Only half of this (`p_max // 2`) frequency vectors are sampled.
    seed : int, optional
        Random seed used for reproducible frequency sampling.

    Returns
    -------
    base_W : np.ndarray of shape (num_inputs, p_max // 2)
        Matrix of base frequency vectors (omegas) for the RFF transform.
    """
    rng = np.random.default_rng(seed)
    # Base random weights
    base_W = rng.normal(loc=0.0, scale=1.0, size=(num_cont_features, p_max // 2))
    
    return base_W


def rff_feature_map(X:list, base_W:np.array, sigma:float, p:int):
    """
    Map input features X to Random Fourier Features (RFFs) for an RBF kernel.

    This implements the standard RFF feature map for an RBF kernel by
    projecting X onto random frequencies, then applying cosine and sine
    nonlinearities and appropriate scaling.

    Specifically, for p total RFF features:
    - Use m = p / 2 random frequencies (columns of base_W / sigma).
    - Features are: sqrt(2 / p) * [cos(XW), sin(XW)].

    Parameters
    ----------
    X : List. First element is continuous signals, second is categorical signals
    base_W : np.ndarray of shape (n_features, p_max // 2)
        Base frequency matrix returned by rff_weights().
    sigma : float
        RBF kernel bandwidth. Frequencies are scaled by 1 / sigma.
    p : int
        Total number of RFF features to construct. Must be even.

    Returns
    -------
    Z : np.ndarray of shape (n_samples, p)
        Matrix of Random Fourier Features corresponding to X.
    """
    # Scale weights by sigma
    W = (base_W / sigma)[:, :p//2]
    
    # Projection: X * W
    proj = X[0] @ W
    
    # Apply non-linearity (Cos and Sin)
    Z = np.hstack([np.sqrt(1 / (p//2)) * np.cos(proj), 
                   np.sqrt(1 / (p//2)) * np.sin(proj),
                  X[1]]
                  )
    return Z

def rff_standardize_feature_map(Z):
    
    Z_mean = np.mean(Z, axis=0)
    Z_centered = Z - Z_mean
    Z_std = np.std(Z_centered, axis=0, ddof=0)
    Z_std[Z_std == 0] = 1.0  # avoid division by zero for constant columns
    Z_scaled = (Z - Z_mean) / Z_std
    
    return Z_scaled, Z_mean, Z_std

def rff_predict(Z, beta):
    """
    Generate predictions from RFF features using a fitted ridge regression model.
    
    A column of ones is prepended to Z_test internally so that the intercept
    in beta (first element) is properly applied.
    
    Parameters
    ----------
    Z : np.ndarray of shape (n_samples, n_features)
        Test design matrix (e.g., RFF features).
    beta : np.ndarray of shape (n_features + 1,)
        Coefficient vector returned by fit_ridge_regression(), with intercept
        as the first element.
    
    Returns
    -------
    y_pred : np.ndarray of shape (n_samples,)
        Predicted target values.
    """
    
    #Add constant
    Z = np.hstack([np.ones((Z.shape[0], 1)), Z])
    
    #Compute the prediction
    y_pred = Z @ beta 
    
    return y_pred

def oos_r2_score(y_true, y_pred):
    """
    Compute the out-of-sample R-squared (OOS R2) of predictions.

    The OOS R2 compares the mean squared prediction error against the variance
    of the realized values. It can be negative if the model performs worse
    than a naive predictor using the historical mean of y_true.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Realized (true) target values.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    r2_oos : float
        Out-of-sample R-squared statistic. Equals 1 for a perfect fit, 0 if
        the model is equivalent to predicting the mean, and negative if the
        model is worse than the mean.
    """
    mse = np.mean((y_true - y_pred)**2)
    var = np.var(y_true)
    if var == 0: return 0.0
    return 1 - (mse / var)


def rff_fit_beta(X, y, base_W, params):
    """
    Fit an RFF + ridge regression model for given hyperparameters.
    
    This is a convenience wrapper that:
    1. Transforms X into RFF features using params['sigma'] and params['p'].
    2. Fits a ridge regression model with penalty params['penalty'].
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features (pre-RFF).
    y : array-like of shape (n_samples,)
        Target values.
    base_W : np.ndarray of shape (n_features, p_max // 2)
        Base frequency matrix from `sample_rff_frequencies`.
    params : dict
        Hyperparameters for the RFF + ridge model. Must contain:
            - 'sigma'   : float, kernel bandwidth.
            - 'p'       : int, total number of RFF features (even).
            - 'penalty' : float, ridge penalty.
    
    Returns
    -------
    beta : np.ndarray of shape (p + 1,)
        Ridge regression coefficients including intercept as the first element.
    """
    
    #Scale Features
    Z = rff_feature_map(X, base_W, params['sigma'], params['p'])
    Z_scaled, Z_mean, Z_std = rff_standardize_feature_map(Z)
    
    #De-mean targets
    y_mean = np.mean(y)
    y_centered = y-y_mean
    
    n, p = Z_scaled.shape
    penalty = params['penalty']
    
    # 1. Compute ZZ (Primal or Dual)
    if n >= p: # Primal form (Features <= Samples)
        ZZ = Z_scaled.T @ Z_scaled
        ridge_matrix = ZZ + penalty * np.eye(p)
        beta = np.linalg.solve(ridge_matrix, Z_scaled.T @ y_centered)
    else: # Dual form (Samples < Features)
        K = Z_scaled @ Z_scaled.T
        ridge_matrix = K + penalty * np.eye(n)
        alpha = np.linalg.solve(ridge_matrix, y_centered)
        beta = Z_scaled.T @ alpha
        
    #2. Compute (unrestricted) intercept
    beta_0 = y_mean 
    #Add intercept
    beta = np.hstack([beta_0, beta])
    
    return beta, Z_mean, Z_std


def predict_rff_model(X, Z_mean, Z_std, base_W, params, beta):
    """
    Generate predictions from a fitted RFF + ridge regression model.
    
    This function applies the same RFF transformation used during training and
    then uses the supplied coefficient vector beta to compute predictions.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features (pre-RFF) for which predictions are required.
    base_W : np.ndarray of shape (n_features, p_max // 2)
        Base frequency matrix from `sample_rff_frequencies`.
    params : dict
        Hyperparameters used for training. Must contain:
            - 'sigma' : float, kernel bandwidth.
            - 'p'     : int, total number of RFF features (even).
    beta : np.ndarray of shape (p + 1,)
        Coefficient vector from `fit_rff_ridge_model`, with intercept first.
    
    Returns
    -------
    y_pred : np.ndarray of shape (n_samples,)
        Predicted target values.
    """
    Z = rff_feature_map(X, base_W, params['sigma'], params['p'])
    Z_scaled = (Z - Z_mean) / Z_std
    
    return rff_predict(Z_scaled, beta)

def save_model(filepath, date, params, beta):
    """
    Appends model date, hyperparameters, and beta coefficients to a text file.
    """
    # Prepare the data dictionary
    record = {
        "date": str(date.date()),          # Convert Pandas Timestamp to string
        "sigma": float(params['sigma']),   # Ensure native Python float
        "p": int(params['p']),             # Ensure native Python int
        "penalty": float(params['penalty']),
        "beta": beta.tolist()              # Convert Numpy array to Python list
    }
    
    # Open file in 'append' mode ('a')
    with open(filepath, 'a') as f:
        # Write the dictionary as a JSON string, followed by a newline
        f.write(json.dumps(record) + "\n")

#%% Preliminaries & Data

#=================================================
#               Preliminaries
#=================================================

#Settings
settings = GF.get_settings()
signals = GF.get_signals()

#Database
JKP_Factors = sqlite3.connect(database=path +"Data/JKP_clean.db")
db_Predictions = sqlite3.connect(database=path +"Data/Predictions.db")

#=================================================
#               Read in Data
#=================================================

#Load Data
df_signals, signal_months, trade_idx, feat_cols, \
    window_size, validation_size, trading_month_start \
        = GF.load_signals_rollingwindow(db_conn = JKP_Factors, #Database with signals
                                        settings = settings, #General settings
                                        target = 'tr_m_sp500_ld1', #Prediction target
                                        rank_signals = False, #Use ZScores
                                        trade_start = '2004-01-31', #First trading date
                                        trade_end = '2024-12-31', #Last trading date
                                        fill_missing_values = True, #Fill missing values 
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

#Seed
seed = 2025

#Naming the Estimator
file_end = f"LevelTarget_CRSPUniverse_ZScores_RollingWindow_win{window_size}_val{validation_size}_test{test_size}_seed{seed}"


#%% RFF Initialisation

# Pre-generate Base Weights (Fixed seed for consistency across time)
# We need the max p to generate a large enough weight matrix once

base_W_global = rff_weights(len(signals[0]), settings['RFF']['p_vec'].max(), seed=seed)

# Container for predictions
predictions = []

# Initialize Best Parameters
best_params = None
best_r2 = -np.inf

months_since_tune = test_size   # start so that the first iteration triggers tuning

#%% Rolling Window Estimation with Hyperparameter Tuning


print("Starting Rolling Forecast...")

#=================================================
#               Main Rolling Loop
#=================================================

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
    X_train = [df_signals[df_signals['eom'].isin(train_months)][signals[0]].to_numpy(), #continuous features
               df_signals[df_signals['eom'].isin(train_months)][signals[1]].iloc[:,:-1].to_numpy() #Categorial features (exclude last as a constant is included in the regression)
               ]
    y_train = df_signals[df_signals['eom'].isin(train_months)]['tr_m_sp500_ld1'].to_numpy()
    
    #Validation Data
    X_val = [df_signals[df_signals['eom'].isin(val_months)][signals[0]].to_numpy(), #Continuous Features
             df_signals[df_signals['eom'].isin(val_months)][signals[1]].iloc[:,:-1].to_numpy() #Categorial Features
             ]
    y_val = df_signals[df_signals['eom'].isin(val_months)]['tr_m_sp500_ld1'].to_numpy()
    
    
    # Check if we need to Tune (Once a year or first run)
    if months_since_tune >= test_size:
        
        #========================================
        #           Hyperparameter Tuning
        #========================================
        
        print(f"Re-tuning hyperparameters at date {date}...")
        
        best_r2 = -np.inf
        best_params = None
    
        # Center Y once to save time
        y_mean = np.mean(y_train)
        y_train_centered = y_train - y_mean
    
        for sigma in settings['RFF']['sigma_vec']:
            for p in settings['RFF']['p_vec']:
                
                # Compute Training Features
                Z_train = rff_feature_map(X_train, base_W_global, sigma, p)
                Z_train_scaled, Z_train_mean, Z_train_std = rff_standardize_feature_map(Z_train)
                
                #Validation Features
                Z_val   = rff_feature_map(X_val, base_W_global, sigma, p)
                Z_val_scaled = (Z_val - Z_train_mean)/Z_train_std
                
                # --- OPTIMIZATION START ---
                # Precompute Matrix Products for Ridge (Primal vs Dual)
                n, n_feat = Z_train.shape
                
                # Pre-calculate the core matrix to avoid doing it inside penalty loop
                if n >= n_feat:
                    # Primal
                    Core_Matrix = Z_train_scaled.T @ Z_train_scaled
                    XY_corr = Z_train_scaled.T @ y_train_centered
                    is_primal = True
                else:
                    # Dual
                    Core_Matrix = Z_train_scaled @ Z_train_scaled.T
                    is_primal = False
                # --- OPTIMIZATION END ---
                for penalty in settings['RFF']['penalty_vec']:
                    # Fast Solve
                    if is_primal:
                        ridge_matrix = Core_Matrix + penalty * np.eye(n_feat)
                        beta_coef = np.linalg.solve(ridge_matrix, XY_corr)
                    else:
                        ridge_matrix = Core_Matrix + penalty * np.eye(n)
                        alpha = np.linalg.solve(ridge_matrix, y_train_centered)
                        beta_coef = Z_train_scaled.T @ alpha
                    
                    # Reconstruct Intercept
                    beta_0 = y_mean 
                    beta = np.hstack([beta_0, beta_coef])
                    
                    # Predict
                    y_val_pred = rff_predict(Z_val_scaled, beta)
                    current_r2 = oos_r2_score(y_val, y_val_pred)
    
                    if current_r2 > best_r2:
                        best_r2 = current_r2
                        best_params = {'sigma': sigma, 'p': p, 'penalty': penalty}
        
        #Saved best_params
        saved_best_params = best_params.copy()
        
        # reset counter
        months_since_tune = 0  
        
        print(f"Best Params selected. sigma = {best_params['sigma']}, "
              f"p = {best_params['p']}, penalty = {best_params['penalty']}, "
              f"R2: {best_r2:.8f}")
        
    else:
        # Reuse previously tuned hyperparameters
        best_params = saved_best_params.copy()
        print(f"Skipping hyperparameter tuning at date {date}, reusing saved params.")
    
    # Increment months since last tune
    months_since_tune += 1
    
    
    #===========================================================
    #           Refit on Train & Validation Data
    #===========================================================
    
    #Get Train + Val Data
    window_mask = df_signals['eom'].isin(window_months)
    X_window = [df_signals.loc[window_mask, signals[0]],
                df_signals.loc[window_mask, signals[1]].iloc[:,:-1]
                ]
    y_window = df_signals.loc[window_mask, 'tr_m_sp500_ld1']
    
    #Compute beta
    beta, Z_mean_win, Z_std_win = rff_fit_beta(X_window, y_window, base_W_global, best_params)
        
    
    #===========================================================
    #           Predict next month's OOS Return
    #===========================================================
    
    # Make the next month prediction for the out-of-sample period 
    test_mask = (df_signals['eom'] == date - pd.offsets.MonthEnd(1))  # or == date, after you verify
    X_test = [df_signals.loc[test_mask, signals[0]], 
              df_signals.loc[test_mask, signals[1]].iloc[:,:-1]
              ]
    ids_test = df_signals.loc[test_mask, ['id', 'eom']]
    
    y_test_pred = predict_rff_model(X_test, Z_mean_win, Z_std_win, base_W_global, best_params, beta)
    
    
    #===========================================================
    #                   Save results
    #===========================================================
    
    #At 'eom', predict return for 'eom'+1
    pred_df = ids_test.copy()
    pred_df['ret_pred'] = y_test_pred
    predictions.append(pred_df)
    
    #Save Model
    save_model(path + f"Models/RFF/RFF_{file_end}_{date.strftime('%Y%m%d')}", date, best_params, beta)
    
    #Increment index due to new trading_month
    trade_idx += 1


#%% Save Predictions
df_predictions = pd.concat(predictions)

df_predictions.to_sql(name = f"RFF_{file_end}",
                   con = db_Predictions,
                   index = False,
                   if_exists = 'append')

JKP_Factors.close()
db_Predictions.close()
