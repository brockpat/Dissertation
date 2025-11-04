# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 10:03:56 2025

@author: Patrick
"""

#%% Libraries
import pandas as pd
import sqlite3
import pickle
import numpy as np

#%%
def load_data(db_path, wealth_path, barra_path, start_date):
    """
    Loads all necessary data for the backtest.
    
    df_pf_weights: Contains for each stock in the S&P 500 at date t the portfolio weight.
    
    df_kl: Contains for each stock in the CRSP universe at date t the value of Kyle's Lambda.
    
    df_returns: Contains for each stock in the S&P 500 at date t the return data.
    
    df_wealth: Contains for each date t the manager's AUM at the beginning of the period.
    """
    print("Loading data...")
    # Connect to DB
    con = sqlite3.connect(database=db_path)
    
    # Get data
    df = pd.read_sql_query(
        "SELECT id, eom, tr, tr_ld1, tr_m_sp500, tr_m_sp500_ld1, me FROM Factors_processed WHERE eom >= :dates",
        params={'dates': (start_date - pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')},
        con=con,
        parse_dates={'eom'}
    )
    
    #Extract individual dataframes
    df_returns = df.get(['id','eom','tr','tr_ld1','tr_m_sp500','tr_m_sp500_ld1']) #Actual Returns
    df_pf_weights = df.get(['id','eom','me', 'tr']) # Portfolio weights
    
    #Save RAM
    del df
    
    #Get Kyle's Lambda
    df_kl = pd.read_sql_query(
        ("SELECT id, eom, lambda FROM KL WHERE eom >= :date"),
        params = {'date': (start_date - pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")},
        parse_dates = {'eom'},
        con = con)

    
    # Load wealth
    df_wealth = pd.read_csv(wealth_path, parse_dates=['eom'])
    df_wealth = df_wealth.loc[df_wealth['eom'] >= start_date - pd.offsets.MonthEnd(1)]
    
    #Compute initial value weighted portfolio
    df_pf_weights = (
        df_pf_weights
        # 1. Filter rows where 'eom' is the target date or later
        .pipe(lambda df: df.loc[df['eom'] >= start_date - pd.offsets.MonthEnd(1)])
        # 2. Calculate the aggregate market cap per date
        .assign(group_sum=lambda df: df.groupby('eom')['me'].transform('sum'))
        # 3. Calculate a value-weighted initial portfolio
        .assign(pi=lambda df: df['me'] / df['group_sum'])
        # 4. Set all portfolio weights to zero if 'eom' > min_date
        .assign(pi=lambda df: np.where(df['eom'] > df['eom'].min(), 0, df['pi']))
        .merge(df_wealth[['eom', 'mu']], on=['eom'], how='left')
        # 6. Calculate 'g_{t+1}'
        .assign(
            is_min_eom=lambda df: df['eom'] == df['eom'].min(),
            g=lambda df: np.where(
                df['is_min_eom'],
                1,
                (1 + df['tr']) / (1 + df['mu'])
            )
        )
        # 7. Clean up
        .drop(columns=['group_sum', 'is_min_eom', 'mu', 'me', 'tr'])
    )

    # Load Barra covariance
    with open(barra_path, "rb") as f:
        dict_barra_all = pickle.load(f)
        
    dict_barra = {
        k: v for k, v in dict_barra_all.items() 
        if k >= start_date - pd.offsets.MonthEnd(1)
    }
    print("Data loading complete.")
    
    return df_kl, df_returns, df_pf_weights, df_wealth, dict_barra, con

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