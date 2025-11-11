# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 10:03:56 2025

@author: Patrick
"""

#%% Libraries
import numpy as np
import pandas as pd
import pickle

#%%

def load_data(con, start_date, sp500_ids, path, predictor):
    """
    Parameters
    ----------
    con : TYPE
        DESCRIPTION.
    start_date : TYPE
        DESCRIPTION.
    sp500_ids : TYPE
        DESCRIPTION.
    path : TYPE
        DESCRIPTION.
    predictor : TYPE
        DESCRIPTION.

    Returns
    -------
    df_pf_weights : TYPE
        DESCRIPTION.
    df_kl : TYPE
        DESCRIPTION.
    df_returns : TYPE
        DESCRIPTION.
    df_wealth : TYPE
        DESCRIPTION.
    dict_barra : TYPE
        DESCRIPTION.

    """
    #---- Data for investable universe ----
    query = ( "SELECT id, eom, in_sp500, me, lambda, tr, tr_ld1, tr_m_sp500, tr_m_sp500_ld1 "
             + "FROM Factors_processed "
             + f"WHERE eom >= '{start_date}'"
             + f"AND id IN ({sp500_ids})")

    df = (pd.read_sql_query(query,
                           parse_dates = {'eom'},
                           con=con
                           )
          .sort_values(by = ['eom', 'id'])
          .assign(in_sp500 = lambda df: df['in_sp500'].astype('boolean'))
          )

    #---- Kyle's Lambda ----
    df_kl = df.get(['id', 'eom', 'lambda'])

    #---- Evolution AUM ----
    df_wealth = pd.read_csv(path + "Data/wealth_evolution.csv", parse_dates=['eom'])
    df_wealth = df_wealth.loc[df_wealth['eom'] >= pd.to_datetime(start_date) - pd.offsets.MonthEnd(1)]

    #---- Return Forecasts ----
    #Extract individual dataframes
    if predictor == "Myopic Oracle":
        df_returns = df.get(['id','eom','tr','tr_ld1','tr_m_sp500','tr_m_sp500_ld1']) #Actual Returns


    #---- Initialise DataFrame for Portfolio Weights ----
    df_pf_weights = df.loc[df['in_sp500']].get(['id','eom','me', 'tr']) # Portfolio weights

    #Compute initial value weighted portfolio
    df_pf_weights = (
        df_pf_weights
        # 1. Filter rows where 'eom' is the target date or later
        .pipe(lambda df: df.loc[df['eom'] >= pd.to_datetime(start_date) - pd.offsets.MonthEnd(1)])
        # 2. Calculate the aggregate market cap per date
        .assign(group_sum=lambda df: df.groupby('eom')['me'].transform('sum'))
        # 3. Calculate a value-weighted initial portfolio
        .assign(pi=lambda df: df['me'] / df['group_sum'])
        # 4. Set all portfolio weights to zero if 'eom' > min_date
        .assign(pi=lambda df: np.where(df['eom'] > df['eom'].min(), 1e-16, df['pi']))
        .merge(df_wealth[['eom', 'mu']], on=['eom'], how='left')
        # 5. Calculate 'g'
        .assign(
            is_min_eom=lambda df: df['eom'] == df['eom'].min(),
            g=lambda df: np.where(
                df['is_min_eom'],
                1,
                (1 + df['tr']) / (1 + df['mu'])
            )
        )
        # 6. Clean up
        .drop(columns=['group_sum', 'is_min_eom', 'mu', 'me', 'tr'])
        .sort_values(by = ['eom','id'],ascending = [True,True])   
    )

    #---- Barra Covariance Matrix ----
    # Load Barra covariance
    with open(path + "Data/Barra_Cov.pkl", "rb") as f:
        dict_barra_all = pickle.load(f)
        
    dict_barra = {
        k: v for k, v in dict_barra_all.items() 
        if k >= pd.to_datetime(start_date)
    }

    print("Data loading complete.")
    
    return df_pf_weights, df_kl, df_returns, df_wealth, dict_barra

def get_universe_partitions(prev_date, date, df_pf_weights):
    """
    Partitions the stock universe into stayers, leavers, and newcomers.
    """
    
    #Stock universes
    prev_universe = set(df_pf_weights.loc[df_pf_weights['eom'] == prev_date]['id'])
    cur_universe = set(df_pf_weights.loc[df_pf_weights['eom'] == date]['id'])
    
    #Stocks that can no longer be in the portfolio
    leavers = list(prev_universe - cur_universe)
    #Stocks that can newly enter the portfolio
    newcomers = list(cur_universe - prev_universe)
    #Stocks that can remain the portfolio
    stayers = list(cur_universe.intersection(prev_universe))
    #Stocks which can have non-zero portfolio weights and are active choice variables
    active = sorted(list(set(newcomers + stayers)))
    
    #Stocks for which pi_t = 0 must be enforced. This affects all leavers.
    #On top of that, it can affect a subset of newcomers due to missing data 
    #on Kyle's Lambda or covariance-matrix
    zeros = leavers.copy()
    
    return stayers, leavers, newcomers, active, zeros


def portfolio_return_BFGS(logits, pig_tm1, gt, Sigma, KL, wealth, return_predictions, gamma):
    
    #Get current portfolio (in levels)
    pi_t = np.exp(logits)
    pi_t /= np.sum(pi_t)
    
    #Compute Revenue
    revenue = pi_t.T @ return_predictions 
    
    #Compute Variance penalty
    var_pen = gamma/2 * pi_t.T @ Sigma @ pi_t
    
    #Compute transaction costs
    change_pf = pi_t-pig_tm1 
    tc = 0.5* wealth * np.sum(KL * change_pf**2)
    
    return -(revenue - tc - var_pen)