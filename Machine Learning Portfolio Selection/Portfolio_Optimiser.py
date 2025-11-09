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

path = "C:/Users/patri/Desktop/ML/"

#DataFrame Libraries
import pandas as pd

#Turn off pandas performance warnings
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#Plot Libraries
import matplotlib.pyplot as plt

#Scientifiy Libraries
import numpy as np

import os
os.chdir(path + "Code/")
import General_Functions as GF

settings = GF.get_settings()

import Portfolio_Optimiser_Functions as pof

from scipy.optimize import minimize


import torch
import torch.nn.functional as F

Gradient_Ascent = True  #Else BFGS

#%% Read in Data

#Trading dates
trading_start, trading_end = settings['rolling_window']['trading_month'], settings['rolling_window']['trading_end']
trading_dates = pd.date_range(start=trading_start,
                              end=trading_end,
                              freq='M'
                              )

df_kl, df_returns, df_pf_weights, df_wealth, \
    dict_barra, JKP_Factors = pof.load_data(path +"Data/JKP_SP500.db",   #Database
                                        path + "Data/wealth_evolution.csv", #Wealth Evolution
                                        path + '/Data/Barra_Cov.pkl',       #Barra Cov
                                        trading_start                       #Trading start date
                                        )

df_kl = df_kl.drop_duplicates()
    
df_pf_weights = df_pf_weights.sort_values(by = ['eom','id'],ascending = [True,True])   

#%% Compute Optimal Portfolio

df_strategy = []

#Loop over trading date
for date in trading_dates:
    print(date)
    #if date == pd.to_datetime('2008-01-31'):
    #    print("BREAK")
    #    break
    
    #=====================================
    #             Preliminaries
    #=====================================
    
    #Previous Date
    prev_date = date - pd.offsets.MonthEnd(1)
    
    #Stock univere
    stayers, leavers, newcomers, \
        active, zeros = pof.get_universe_partitions(prev_date, date, df_pf_weights)
    
    #=====================================================
    #   Shrink active universe in case of missing data
    #=====================================================
    
    #----- Kyle's Lambda -----
    #Extend zeros with newcomers who don't have data on Kyle's Lambda
    zeros.extend([stock for stock in newcomers if stock not in 
                  set(df_kl.loc[df_kl['eom'] == prev_date]['id'])])
    #Adjust active
    active = sorted([stock for stock in active if stock not in zeros])
        

    #----- Barra Covariance Matrix -----    
    #Compute covariance matrix
    Sigma = GF.create_cov(dict_barra[prev_date])
       
    #Only keep stocks in the active stock universe
    Sigma = Sigma.reindex(index=active, columns=active)
    
    #Remove stocks with missing variances
    zeros.extend(list(Sigma.index[pd.isna(np.diag(Sigma))]))
    active = sorted([stock for stock in active if stock not in zeros])


    #----- Return Predictions -----
    #Note: These will always exist for stayers & leavers, but not for newcomers
    #       due to JKP_Factors Table Factors being limited to the S&P 500 (see Data_preprocessing.py)
    return_predictions = (df_returns
                          .loc[df_returns['eom'] == prev_date]
                          .get(['id', 'eom','tr_m_sp500_ld1'])
                          )
    zeros.extend([stock for stock in active if stock not in return_predictions['id'].values])
    active = sorted([stock for stock in active if stock not in zeros])
    
    #================================================================
    #   Reduce DataFrames to active universe with non-missing data
    #================================================================
    
    #Covariance Matrix (sorted by 'id' as active is sorted)
    Sigma = Sigma.loc[active, active]
    
    #Kyle's Lambda
    kyles_lambda = (df_kl
                    .loc[(df_kl['eom'] == prev_date) 
                         & 
                         (df_kl['id'].isin(active))
                         ]
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
    #   Build DataFrame with all necessary information
    #==========================================================
    
    #---- Initialisation ----
    df_t = (pd.DataFrame({'id': list(stayers+newcomers+leavers),
                            'eom':date, #begin of month
                            'pi':np.array(1e-16)}
                         )
               .sort_values(by = 'id')
               .reset_index(drop=True)
               )
    
    #---- Get Kyle's Lambda ----
    df_t = df_t.merge(df_kl.loc[df_kl['eom'] == prev_date, 
                                ['id','lambda']], on='id', how='left')
    
    #Set it to zero for newcomers who don't have a value for KL (these newcomers are not in active)
    df_t.loc[(df_t['id'].isin(set(newcomers).intersection(set(zeros)))) 
             & 
             (df_t['lambda'].isna()), 'lambda'] = 0
        
    if df_t['lambda'].isna().sum() > 0:
        print("ERROR: A stock does not have a value for Kyle's Lambda")
        #break
        
    #---- Get actual stock's return ----
    #Merge values
    df_t = (df_t.merge(df_returns.loc[df_returns['eom'] == date, ['id', 'tr']],
                       on = ['id'], how = 'left'))
    
    #Leavers don't have an observed return. Set these to 0 (leavers are not actively held)
    df_t.loc[df_t['id'].isin(leavers), 'tr'] = 0.0
    
    if df_t['tr'].isna().sum() > 0:
        print("ERROR: A stock does not have an observed end-of-period return")
        #break
    
    #---- Compute G @ pi_{t-1} ----
    # Compute pi_g for the previous date
    pi_g_prev = (
        df_pf_weights
        .query("eom == @prev_date")[['id', 'pi', 'g']]
        .assign(pi_g=lambda df: df['pi'] * df['g'])
        [['id', 'pi_g']]
    )
    
    # Merge and update df_t
    df_t = (
        df_t
        .merge(pi_g_prev, on='id', how='left')
        .rename(columns = {'pi_g': 'pi_g_tm1'})
    )
    del pi_g_prev
    
    #Set it to 0 for newcomers (no previous position available)
    df_t.loc[df_t['id'].isin(newcomers),'pi_g_tm1'] = 0.0
    
    #---- Initialise 'pi_t' with 'pi_g_tm1' ----
    df_t.loc[df_t['id'].isin(active), 'pi'] = df_t.loc[df_t['id'].isin(active), 'pi_g_tm1']

    print(f"   MAX pi: {df_t['pi'].max()}")

    #==========================================================
    #           Solve for optimal portfolio vector
    #==========================================================
    
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
        pi_prev = torch.tensor(df_t.loc[df_t['id'].isin(active), 'pi_g_tm1'].to_numpy())
        #Initialisec current portfolio (in logits)
        pi_logits = torch.tensor(np.log(df_t.loc[df_t['id'].isin(active), 'pi'].to_numpy()),
                                 requires_grad = True)
        
        #--- Optimizer
        optimizer = torch.optim.Adam([pi_logits], lr=1e-2)
        
        #--- Define Constraint Parameters ---
        max_weight = 0.05
        # This is a new hyperparameter you may need to tune.
        # A higher value enforces the constraint more strictly.
        penalty_weight = 1000.0
        
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
        
            #--- NEW: Calculate Constraint Penalty ---
            # 1. Find the amount each weight is *over* the cap
            #    torch.relu ensures this is 0 for weights <= 0.1
            weight_violation = torch.relu(pi - max_weight)
            
            # 2. Calculate the penalty (e.g., quadratic penalty)
            #    We sum the squared violations
            constraint_penalty = (penalty_weight * weight_violation).sum()

            #--- NEW: Add penalty to the loss ---
            # Adam does minimization -> minimize negative for maximisation
            # We ADD the penalty to the loss, so the optimizer
            # minimizes it (i.e., avoids violating the constraint)
            loss = -F_val + constraint_penalty

            loss.backward()
            optimizer.step()
        
            # (optional) print
            #if step % 100 == 0:
            #    print(step, F_val.item())
        print(f"  Results: {F_val.item()}")
        print(f"  Penalty: {constraint_penalty}")
        
        #Save Results
        df_t.loc[df_t['id'].isin(active), 'pi'] = pi.detach().cpu().numpy()
        
        #Compute revenue & transaction costs
        df_t = (df_t
                .assign(rev = lambda df: df['pi']*df['tr'])
                .assign(tc = lambda df: (df['pi'] - df['pi_g_tm1'])**2 * df['lambda'] * float(w)/2 )
                )
        
        #Add to DataFrame
        df_pf_weights = df_pf_weights.set_index(['id', 'eom'])
        df_pf_weights.update(df_t.set_index(['id', 'eom'])[['pi']])
        df_pf_weights = df_pf_weights.reset_index()
        
        #Append Result
        df_strategy.append(df_t)
    
results = pd.concat(df_strategy)

df_profit = results.groupby('eom').apply(lambda df: (df['rev'] - df['tc']).sum()).reset_index().rename(columns = {0: 'strategy_profit'}) 
df_profit['cumulative_return'] = (1 + df_profit['strategy_profit']).cumprod() - 1

df_spy = pd.read_sql_query("SELECT * from SPY_Return",
                           con = JKP_Factors,
                           parse_dates = {'eom'})

df_spy = df_spy[df_spy['eom'].isin(df_profit.eom.unique())]
df_spy['cumulative_return'] = (1 + df_spy['SPY_ret']).cumprod() - 1


plt.plot(df_profit['eom'], df_profit['cumulative_return'], label ='strategy')
plt.plot(df_spy['eom'], df_spy['cumulative_return'], label ='SPY')
plt.show()


df_profit = df_profit.merge(risk_free, on = 'eom', how = 'left')
df_profit['ret_exc'] = df_profit['strategy_profit'] - df_profit['rf']

df_spy = df_spy.merge(risk_free, on = 'eom', how = 'left')
df_spy['ret_exc'] = df_spy['SPY_ret'] - df_spy['rf']


mu = df_profit['ret_exc'].mean()
sigma = df_profit['ret_exc'].std(ddof=1)  # sample std
Sharpe_Strategy = np.sqrt(12) * (mu / sigma)

mu = df_spy['ret_exc'].mean()
sigma = df_spy['ret_exc'].std(ddof=1)  # sample std
Sharpe_Spy = np.sqrt(12) * (mu / sigma)

information_ratio = np.sqrt(12) * np.mean(df_profit['ret_exc'] - df_spy['ret_exc'])/((df_profit['ret_exc'] - df_spy['ret_exc']).std(ddof=1))

"""
An IR = 0.5 means:

For every 1% of tracking error (volatility of return relative to the benchmark), your portfolio earns 0.5% of excess return on average per year.

Put differently:

Your strategy adds 0.5 units of active return per unit of active risk.
"""
        
#%%        
        
        
        
    #BFGS
    else:
        #BFGS to maximise objective function
        logits_BFGS = minimize(portfolio_return_BFGS, df_t_choice['logits'].to_numpy(),
                               args = (df_t[df_t['id'].isin(active)]['pi'].to_numpy(), 
                                    df_t['g'].to_numpy(), 
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

    
