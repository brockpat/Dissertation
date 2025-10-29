# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 14:13:11 2025

@author: patrick
"""

#%% Libraries

path = "C:/Users/pf122/Desktop/Uni/Frankfurt/2023-24/Machine Learning/Single Authored/"

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

from scipy.optimize import minimize

import pickle


#%% Read in Data
#Connect to DataBase
JKP_Factors = sqlite3.connect(database=path +"Data/JKP_US_SP500.db")

#Trading dates
trading_start, trading_end = settings['rolling_window']['trading_month'], settings['rolling_window']['trading_end']
trading_dates = trading_dates = pd.date_range(start=trading_start,
                                              end=trading_end,
                                              freq='M'
                                              )

#Get Data
df = pd.read_sql_query(("SELECT id, eom, lambda, tr, tr_ld1, tr_m_sp500, tr_m_sp500_ld1, me FROM Factors_processed "
                               f"WHERE eom >= :dates"),
                                 params = {'dates': 
                                           (trading_start - pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')},
                                 con = JKP_Factors,
                                 parse_dates = {'eom'}
                                 )

#Data subsets: Kyle's Lambda, returns, portfolio weights
df_kl, df_returns, df_pf_weights = (df.get(['id','eom','lambda']), 
         df.get(['id','eom','tr','tr_ld1','tr_m_sp500','tr_m_sp500_ld1']),
         df.get(['id','eom','me', 'lambda', 'tr'])
         )

#Save Memory
del df

#Read in wealth
df_wealth = (pd.read_csv(path + "Data/wealth_evolution.csv", parse_dates = ['eom'])
             .pipe(lambda df: df.loc[df['eom'] >=trading_start - pd.offsets.MonthEnd(1)])
             )

#Compute initial value weighted portfolio
df_pf_weights = (df_pf_weights
    # 1. Filter rows where 'eom' is the target date or later
    .pipe(lambda df: df.loc[df['eom'] >= trading_start - pd.offsets.MonthEnd(1)])
    # 2. Calculate the aggregate market cap per date
    .assign(group_sum=lambda df: df.groupby('eom')['me'].transform('sum'))
    # 3. Calculate a value weighted initial portfolio
    .assign(pi_stock=lambda df: df['me'] / df['group_sum'])
    # 4. Zero all portfolio weights for the trading period
    .assign(pi_stock=lambda df: np.where(df['eom'] == trading_start - pd.offsets.MonthEnd(1), df['pi_stock'], 0))
    .assign(pi_choice = 0)
    .assign(tc = 0)
    .assign(rev = 0)
    .merge(df_wealth[['eom','mu', 'wealth']], on = ['eom'], how = 'left')
    # 5. Calculate 'g'
    .assign(
        # Identify the minimum 'eom' date
        is_min_eom=lambda df: df['eom'] == df['eom'].min(),
        # Calculate g
        g=lambda df: np.where(
            df['is_min_eom'],
            1,
            (1 + df['tr']) / (1 + df['mu'])
        )
        ) # 5. Clean up
    .drop(columns=['group_sum', 'is_min_eom'])
    )

#Read in Barra Covarince matrix
with open(path + '/Data/Barra_Cov.pkl', "rb") as dict_barra:
    dict_barra = pickle.load(dict_barra)
    
#Filter out dates that are not required
dict_barra = {
    key: value 
    for key, value in dict_barra.items() 
    if key >= trading_start - pd.offsets.MonthEnd(1)  # The condition to KEEP the entry: date_key is NOT less than value
}

#%%

def portfolio_return_BFGS(logits_active, pi_stock, Sigma, KL, wealth, return_predictions, active_idx,
                          scaling_coef):
        
    logits_stable = logits_active/scaling_coef
    weights_active = np.exp(logits_stable)
    weights_active /= np.sum(weights_active)
    
    pi = np.zeros_like(pi_stock)
    pi[active_idx] = weights_active
    
    #Compute Revenue
    revenue = pi.T @ return_predictions 
    
    #Compute Variance penalty
    var_pen = 0.5*pi.T @ Sigma @ pi
    
    #Compute transaction costs
    change_pf = pi-pi_stock 
    tc = 0.5* wealth * np.sum(KL * change_pf**2)
    
    return -(revenue - tc - var_pen)

def pi_choice(logits_active, pi_stock, active_idx, scaling_coef, ids):
    logits_stable = logits_active/scaling_coef
    weights_active = np.exp(logits_stable)
    weights_active /= np.sum(weights_active)
    
    pi = np.zeros_like(pi_stock)
    pi[active_idx] = weights_active
    
    dictionary = {'id': ids, 'w':pi}
    
    return pd.DataFrame(dictionary)

for date in trading_dates:
    
    #=====================================
    #             Preliminaries
    #=====================================
    
    #Previous Date
    prev_date = date - pd.offsets.MonthEnd(1)
    
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
    
    #===========================================
    #               Kyle's Lambda
    #===========================================
    
    #Kyle's Lambda of stayers
    kyles_lambda = (df_kl.loc[(df_kl['eom'] == prev_date) & 
                              (df_kl['id'].isin(stayers))
                              ]
                    .get(['id','lambda'])
                    )
    
    #Add newcomers
    if newcomers: #If newcomers exist, fetch their Kyle's Lambda
        kyles_lambda_newcomers = pd.read_sql_query(("SELECT id, eom, lambda FROM Kyle's Lambda "
                                                 f"WHERE eom = {date} AND id IN ({ids})"
                                                 ),
                                                params = {'date': prev_date.strftime("%Y-%m-%d"),
                                                          'ids': ", ".join([f"'{x}'" for x in newcomers])
                                                          },
                                                parse_dates = {'eom'},
                                                con = JKP_Factors                                    
                                                )
        #Extend zeros with newcomers who don't have data on Kyle's Lambda
        zeros.extend([stock for stock in newcomers if stock not in set(kyles_lambda['id'])])
        #Adjust active
        active = sorted([stock for stock in active if stock not in zeros])
        
        #Add newcomers to Kyle's Lambda
        kyles_lambda = (pd.concat([kyles_lambda,kyles_lambda_newcomers]))
        
    #===========================================
    #           Barra Covariance Matrix
    #===========================================
    
    #Compute covariance matrix
    Sigma = create_cov(dict_barra[prev_date])
       
    #Only keep stocks in the active stock universe
    Sigma = Sigma.reindex(index=active, columns=active)
    
    #Remove stocks with missing variances
    zeros.extend(list(Sigma.index[pd.isna(np.diag(Sigma))]))
    active = sorted([stock for stock in active if stock not in zeros])

    #=====================================
    #        Load Return Prediction
    #=====================================
    
    #will always exist for stayers & leavers, but not newcomers
    return_predictions = (df_returns
                          .loc[df_returns['eom'] == prev_date]
                          .get(['id','tr_m_sp500_ld1'])
                          )
    zeros.extend([stock for stock in active if stock not in return_predictions['id'].values])
    active = sorted([stock for stock in active if stock not in zeros])
    
    #===========================================
    #           Only keep active stocks
    #===========================================
    
    #Covariance Matrix (sorted by 'id' as active is sorted)
    Sigma = Sigma.loc[active, active]
    
    #Kyle's Lambda
    kyles_lambda = (kyles_lambda
                    .loc[kyles_lambda['id'].isin(active)]
                    .sort_values(by = 'id')
                    .reset_index(drop = True)
                    )
    
    #Return predictions
    return_predictions = (return_predictions
                          .loc[return_predictions['id'].isin(active)]
                          .sort_values(by = 'id')
                          .reset_index(drop=True)
                          )

    
    #==========================================================
    #   Build portfolio stock (pi_t-1) and choice (pi_t) vector
    #==========================================================
    
    #---------- portfolio stock pi_t-1  ----------
    # 1. Stayers (active stocks)
    # Get portfolio stock of stayers
    pi_stock = (df_pf_weights
                #filter date to t-1
                .loc[df_pf_weights.eom == prev_date]
                #Filter out leavers
                .pipe(lambda df: df.loc[df['id'].isin(active)])
                #get relevant variables
                .get(['id','w'])
                )
    
    # 2. Newcomers (remaining active stocks)
    #Add newcomers with zero portfolio weight (no newcomers in df_pf_weights for prev_date)
    pi_stock = (pd.concat([pi_stock,
                          pd.DataFrame({'id': [stock for stock in active if stock not in pi_stock.id.values],
                                        'w': 0.0*len([stock for stock in active if stock not in pi_stock.id.values])
                                        }
                                       )
                          ]
                         )
                         .sort_values(by='id')
                         .reset_index(drop = True)
                         )
    
    #---------- portfolio choice pi_t ----------
    pi_choice = (pi_stock
              .assign(logits = lambda df: np.log(df['w']))
                  .get(['id','logits'])
                  .sort_values(by = 'id')
                  .reset_index(drop = True)
                  )
    
    #==========================================================
    #           Solve for optimal portfolio vector
    #==========================================================
    
    !!!!
    HIER MUSS ICH gt pi_{t-1} jetzt inkorporieren und prinzipiell nochmal checken,
    dass mein portfolio choice auch wirklich richtig formuliert ist
    !!!!
    logits_BFGS = minimize(portfolio_return_BFGS, logits_active.to_numpy(),
                           args = (pi_stock['w'].to_numpy(),
                                   Sigma.to_numpy(),
                                   kyles_lambda['lambda'].to_numpy(),
                                   df_wealth[df_wealth['eom'] == prev_date]['wealth'].iloc[0],
                                   return_predictions['tr_m_sp500_ld1'].to_numpy(),
                                   active_idx,
                                   np.sqrt(len(logits_active))
                                   ),
                           method='BFGS', options={'maxiter': 100, 'gtol': 1e-8}
                           )
    
    #Get optimal portfolio weights
    pi_opt = pi_choice(logits_BFGS.x, pi_stock['w'].to_numpy(), active_idx, 
                   np.sqrt(len(logits_active)), pi_stock['id'].to_numpy()
                   )
    pi_choice['w_choice'] = softmax pi_opt['logits']
    pi_choice['eom'] = prev_date
    df_pf_weights = df_pf_weights.merge(pi_choice, on = ['id', 'eom'], how = 'left')
    
    #Update Portfolio weights given their returns
    gt = (
        (df_returns.loc[(df_returns['eom'] == prev_date) 
                        & 
                        df_returns['id'].isin(leavers + stayers + newcomers)
                        ]
         .sort_values(by = 'id')
         .get(['tr_ld1'])
         )
        /
        df_wealth.loc[df_wealth['eom'] == prev_date].get(['mu_ld1']).to_numpy()
        )
    


