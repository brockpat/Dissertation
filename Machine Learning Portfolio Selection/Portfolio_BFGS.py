# -*- coding: utf-8 -*- 
"""
Created on Sun Oct 26 14:13:11 2025

@author: patrick

Begin of period variables are known at the beginning of time 't'. This implies
that they are computed from information from t-1,t-2,...

The following variables are beginn of period:

'pi'
'wealth' (AUM) 

'Sigma' (Barra Cov)

'g' 

'lambda' (Kyle's Lambda)

All other variables are at the end of the period (month).
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

import torch
import torch.nn.functional as F

Gradient_Ascent = True  #Else BFGS

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
df_pf_weights = (
    df_pf_weights
    # 1. Filter rows where 'eom' is the target date or later
    .pipe(lambda df: df.loc[df['eom'] >= trading_start - pd.offsets.MonthEnd(1)])
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
    .drop(columns=['group_sum', 'is_min_eom', 'mu'])
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

#%% Compute Optimal Portfolio

df_strategy = pd.DataFrame({'eom': trading_dates[0] - pd.offsets.MonthEnd(1),
                            'revenue': 0.0,
                            'tc': 0.0},
                           index=[0])

#Loop over trading date
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
                          .get(['id', 'eom','tr_m_sp500_ld1'])
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
    #   Build portfolio vector pi_t-1 and pi_t
    #==========================================================
        
    #pi_t-1
    df_pi_tm1 = (df_pf_weights
                 .loc[df_pf_weights['eom'] == prev_date]
                 .get(['id','eom','pi', 'g'])
                 .assign(pi_g = lambda df: df['pi']*df['g'])
                 .sort_values(by = 'id')
                 .reset_index(drop=True)
                 )
    
    #pi_t
    df_pi_t = (pd.DataFrame({'id': list(stayers+newcomers+leavers),
                            'eom':date,
                            'pi':np.array(0),
                            'rev': np.array(0),
                            'tc': np.array(0)})
               .merge(df_pi_tm1[['id','pi_g']], on = 'id', how = 'left')
               .fillna(0)
               .sort_values(by = 'id')
               .reset_index(drop=True)
               )
    
    #==========================================================
    #           Solve for optimal portfolio vector
    #==========================================================
    
    #---------------- Auxiliary Objects ----------------
    #Get the portfolio weights that can be actively chosen
    df_pi_t_choice = df_pi_t.loc[df_pi_t['id'].isin(active)]
    
    #Initialise active portfolio weights with pi_{t-1}
    """
    NEED A STRATEGY TO INITIALISE PI FOR NEWCOMERS AS LOGIT OF -INF IS NOT POSSIBLE
    """
    df_pi_t_choice = (df_pi_t_choice
                      .drop(columns = 'pi')
                      .merge(df_pi_tm1[['id','pi']], on = ['id'], how = 'left',
                             suffixes = ("","")
                             )
                      #Get logits
                      .assign(logits = lambda df: np.log(df['pi']))
                      .sort_values(by = 'id')
                      )
    
    if Gradient_Ascent: #with Pytorch
        #--- Define Objects
        #Return Prediction
        r = torch.tensor(return_predictions['tr_m_sp500_ld1'])
        #Covariance Matrix of Returns
        S = torch.tensor(Sigma.to_numpy())
        #Diagonal of Kyle's Lambda matrix
        L_diag = torch.tensor(kyles_lambda['lambda'])
        #Risk Aversion
        gamma = torch.tensor(settings['gamma']) 
        #Wealth
        w = torch.tensor(df_wealth.loc[df_wealth['eom'] == date]['wealth'].iloc[0])
        #Previous portfolio (in levels)
        pi_prev = torch.tensor(df_pi_tm1[df_pi_tm1['id'].isin(active)]['pi_g'].to_numpy())
        #Initialisec current portfolio (in logits)
        pi_logits = torch.tensor(df_pi_t_choice['logits'].to_numpy(),
                                 requires_grad = True)
        
        #--- Optimizer
        optimizer = torch.optim.Adam([pi_logits], lr=1e-2)
        
        #--- Gradient Ascent
        for step in range(500):
            optimizer.zero_grad()
        
            # current portfolio (levels)
            pi = F.softmax(pi_logits, dim=0)  # shape (n,)
        
            # revenue:
            revenue = torch.dot(r, pi)
        
            # var penalty
            var_penalty = 0.5 * gamma * pi @ S @ pi
        
            # transaction costs
            diff = pi - pi_prev
            turnover_quad = (L_diag * diff.pow(2)).sum()
            tc = 0.5 * w * turnover_quad
        
            # FULL objective to MAXIMISE
            F_val = revenue - var_penalty - tc
        
            # Adam does minimization -> minimize negative for maximisation
            loss = -F_val
            loss.backward()
            optimizer.step()
        
            # (optional) print
            if step % 100 == 0:
                print(step, F_val.item())
        
        #Save Results
        df_pi_t_choice['logits'] = pi_logits.detach().cpu().numpy()
        df_pi_t_choice['pi'] = pi.detach().cpu().numpy()
        
    #BFGS
    else:
        #BFGS to maximise objective function
        logits_BFGS = minimize(portfolio_return_BFGS, df_pi_t_choice['logits'].to_numpy(),
                               args = (df_pi_tm1[df_pi_tm1['id'].isin(active)]['pi'].to_numpy(), 
                                    df_pi_tm1['g'].to_numpy(), 
                                    Sigma.to_numpy(), 
                                    kyles_lambda['lambda'].to_numpy(), 
                                    df_wealth.loc[df_wealth['eom'] == date]['wealth'].to_numpy(), 
                                    return_predictions['tr_m_sp500_ld1'].to_numpy(), 
                                    settings['gamma']),
                               method='BFGS', options={'maxiter': 300, 'gtol': 1e-8}
                               )
    
    """
    INCLUDE ZEROS HERE TOO
    """
    #Extract optimal portfolio weights
    df_pi_t_choice['logits'] = logits_BFGS.x 
    df_pi_t_choice['pi'] = np.exp(df_pi_t_choice['logits'] )/(np.exp(df_pi_t_choice['logits'] ).sum())
    
    #Save to DataFrame
    df_pf_weights = (
        df_pf_weights
        .merge(df_pi_t_choice[['id', 'eom', 'pi']], on=['id', 'eom'], how='outer', suffixes=('', '_new'))
        .assign(pi=lambda df: df['pi_new'].combine_first(df['pi']))
        .drop(columns=['pi_new'])
    )
    
    """
    COMPUTE TRANSACTION COSTS AND REVENUE
    """

    
