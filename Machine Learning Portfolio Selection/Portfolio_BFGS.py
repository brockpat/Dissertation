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
         df.get(['id','eom','me'])
         )

#Save Memory
del df

#Compute initial value weighted portfolio
df_pf_weights = (df_pf_weights
    # 1. Filter rows where 'eom' is the target date or later
    .pipe(lambda df: df.loc[df['eom'] >= trading_start - pd.offsets.MonthEnd(1)])
    # 2. Calculate the aggregate market cap per date
    .assign(group_sum=lambda df: df.groupby('eom')['me'].transform('sum'))
    # 3. Calculate a value weighted initial portfolio
    .assign(w=lambda df: df['me'] / df['group_sum'])
    # 4. Zero all portfolio weights for the trading period
    .assign(w=lambda df: np.where(df['eom'] == trading_start - pd.offsets.MonthEnd(1), df['w'], 0))
    # 5. Clean up
    .drop(columns=['group_sum'])
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

#Read in wealth
df_wealth = (pd.read_csv(path + "Data/wealth_evolution.csv", parse_dates = ['eom'])
             .pipe(lambda df: df.loc[df['eom'] >=trading_start - pd.offsets.MonthEnd(1)])
             )

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

#Initialise objects
gt = (pd.DataFrame({'id': df_returns.loc[(df_returns['eom'] == prev_date)]['id'],
                   'growth':1})
      .sort_values(by = 'id')
      .reset_index(drop = True)
      )

for date in trading_dates:
    
    #=====================================
    #             Preliminaries
    #=====================================
    
    #Previous Date
    prev_date = date - pd.offsets.MonthEnd(1)
    
    #Stock universes
    prev_universe = set(df_pf_weights.loc[df_pf_weights['eom'] == prev_date]['id'])
    cur_universe = set(df_pf_weights.loc[df_pf_weights['eom'] == date]['id'])
    
    #Stocks that are no longer in the investable universe 
    leavers = list(prev_universe - cur_universe)
    newcomers = list(cur_universe - prev_universe)
    stayers = list(cur_universe.intersection(prev_universe))
    
    #Stocks for which pi_t = 0 must be enforced due to missing data 
    #on either Kyle's Lambda or the covariance matrix
    zeros = []
    
    #===========================================
    #               Kyle's Lambda
    #===========================================
    
    #Kyle's Lambda of stayers & leavers
    kyles_lambda = (df_kl.loc[df_kl.eom == prev_date])[['id','lambda']]
    
    #Add newcomers
    if newcomers:
        kyles_lambda_newcomers = pd.read_sql_query(("SELECT id, eom, lambda FROM Kyle's Lambda "
                                                 f"WHERE eom = {date} AND id IN ({ids})"
                                                 ),
                                                params = {'date': prev_date.strftime("%Y-%m-%d"),
                                                          'ids': ", ".join([f"'{x}'" for x in newcomers])
                                                          },
                                                parse_dates = {'eom'},
                                                con = JKP_Factors                                    
                                                )
        #Add Data on Newcomers
        kyles_lambda = pd.concat([kyles_lambda,kyles_lambda_newcomers])
        
        #Fill zeros with newcomers who don't have data on Kyle's Lambda
        zeros.extend([n for n in newcomers if n not in set(kyles_lambda['id'])])
        
        #Include missing newcomers in dataframe
        kyles_lambda = (pd.concat([kyles_lambda, 
                                  pd.DataFrame({'id': zeros,
                                                'lambda': 0.0*len(zeros)
                                                })
                                  ], ignore_index=True)
                        .sort_values(by = 'id')
                        .reset_index(drop = True)
                        )
        
    else:
        kyles_lambda = (kyles_lambda
                        .sort_values(by = 'id')
                        .reset_index(drop = True)
                        )
        
    #===========================================
    #           Barra Covariance Matrix
    #===========================================
    
    #Compute covariance matrix
    Sigma = create_cov(dict_barra[prev_date])
       
    #Only keep stocks in the current stock universe
    universe = sorted(leavers + stayers  + newcomers)
    Sigma = Sigma.reindex(index=universe, columns=universe)
    
    #Append zeros with stocks who don't have a variance
    diagonal = pd.Series(np.diag(Sigma), index=Sigma.index).isna()
    diagonal = list(diagonal[diagonal].index)
    zeros.extend(diagonal)
    Sigma = Sigma.fillna(0)
    
    #=====================================
    #        Load Return Prediction
    #=====================================
    #will always exist for stayers, leavers & newcomers due to Data_Preprocessing
    return_predictions = (df_returns
                          .loc[df_returns['eom'] == prev_date]
                          .sort_values(by = 'id')
                          .reset_index(drop=True)
                          .get(['id','eom','tr_m_sp500_ld1'])
                          )
    
    #==========================================================
    #   Build portfolio stock (pi_t-1) and choice (pi_t) vector
    #==========================================================
    
    #---------- portfolio stock pi_t-1  ----------
    # Get stayers & leavers
    pi_stock = (df_pf_weights.loc[df_pf_weights.eom == prev_date][['id','w']]
            )
    
    #Add newcomers who have zero stock
    newcomer_w = {'id': newcomers,
                     'w': [0.0]*len(newcomers),
                     }
    pi_stock = (pd.concat([pi_stock,pd.DataFrame(newcomer_w)])
              .sort_values(by = 'id')
              .reset_index(drop = True)
              .assign(id=lambda df: df['id'].astype(int))
              )
    
    #---------- portfolio choice pi_t ----------
    # (leavers & newcomers = 0, stayers = w)
    pi_logits = (pi_stock
              .assign(logits = lambda df: 
                      np.where(~df.id.isin(zeros),np.log(df['w']),-np.inf)
                      )
                  .get(['id','logits'])
                  .sort_values(by = 'id')
                  .reset_index(drop = True)
                  )
    
    #Optimise only over active logits
    pi_logits.loc[~pi_logits['id'].isin(zeros)]
    active_idx = list(pi_logits.loc[~pi_logits['id'].isin(zeros)].index)
    logits_active = pi_logits['logits'].iloc[active_idx]
    
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
    


