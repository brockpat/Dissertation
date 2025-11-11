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

#Gradient Ascent
import torch
import torch.nn.functional as F

#Custom Functions
import os
os.chdir(path + "Code/")
import General_Functions as GF
import Portfolio_Optimiser_Functions as pof
settings = GF.get_settings()

#%% Read in Data

#DataBases
JKP_Factors = sqlite3.connect(database = path + "Data/JKP_clean.db")
SP500_Constituents = sqlite3.connect(database = path + "Data/SP500_Constituents.db")
Benchmarks = sqlite3.connect(database = path + "Data/Benchmarks.db")


#============================
#       Risk-free rate
#============================
"""
Used to compute Sharpe Ratios
"""
# Read risk-free rate data (select only 'yyyymm' and 'RF' columns).
risk_free = pd.read_csv(path + "Data/FF_RF_monthly.csv", usecols=["yyyymm", "RF"])

# Convert to decimal (RF is given in percentage terms)
risk_free["rf"] = risk_free["RF"] / 100

#--- Construct an end-of-month date.
risk_free["eom"] = pd.to_datetime(risk_free["yyyymm"].astype(str) + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0)

# Keep only the required columns.
risk_free = risk_free[["eom", "rf"]]


#============================
#       Trading Dates
#============================
#Trading dates
trading_start, trading_end = settings['rolling_window']['trading_month'], settings['rolling_window']['trading_end']
trading_dates = pd.date_range(start=trading_start,
                              end=trading_end,
                              freq='ME'
                              )
start_date = str(trading_start - pd.offsets.MonthEnd(1))[:10]
end_date = str(trading_end)[:10]


#===============================
#       SPY ETF Performance
#===============================
df_spy = pd.read_sql_query("SELECT * FROM SPY",
                           parse_dates = {'eom'},
                           con = Benchmarks)

#============================
#       S&P 500 Universe
#============================
#Stocks that were, are and will be in the S&P 500.
sp500_ids = list(pd.read_sql_query("SELECT * FROM SP500_Constituents_alltime",
                              con = SP500_Constituents)['id']
                 )
sp500_ids = ', '.join(str(x) for x in sp500_ids)

#Stocks that at date t are in the S&P 500
sp500_constituents = (pd.read_sql_query(f"SELECT * FROM SP500_Constituents_monthly WHERE eom >= '{start_date}'",
                                       con = SP500_Constituents,
                                       parse_dates = {'eom'})
                      .rename(columns = {'PERMNO': 'id'})
                      )

#============================
#       Data
#============================

df_pf_weights, df_kl, df_returns,\
    df_wealth, dict_barra = pof.load_data(JKP_Factors, start_date, 
                                          sp500_ids, path, 
                                          predictor = "Myopic Oracle")
#%% Compute Optimal Portfolio

#Container to store results
df_strategy = []

#Loop over trading date
for date in trading_dates:
    print(date)
    
    #=====================================
    #             Preliminaries
    #=====================================
    
    #Previous Date
    prev_date = date - pd.offsets.MonthEnd(1)
    
    #Stock universe
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
    return_predictions = (df_returns
                          .loc[df_returns['eom'] == prev_date]
                          .get(['id', 'eom','tr_m_sp500_ld1'])
                          )
    zeros.extend([stock for stock in active if stock not in return_predictions['id'].values])
    active = sorted([stock for stock in active if stock not in zeros])
    
    #========================================
    #   Reduce DataFrames to active universe 
    #========================================
    
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
    #   Build DataFrame for Portfolio Optimisation
    #==========================================================
    
    #---- Initialisation ----
    df_pf_t = (pd.DataFrame({'id': list(stayers+newcomers+leavers),
                            'eom':date, #begin of month
                            'pi':np.array(1e-16)}
                         )
               .sort_values(by = 'id')
               .reset_index(drop=True)
               )
    
    #---- Merge Kyle's Lambda ----
    df_pf_t = df_pf_t.merge(df_kl.loc[df_kl['eom'] == prev_date, 
                                ['id','lambda']], on='id', how='left')
    
    #Set it to zero for newcomers who don't have a value for KL 
    #   these newcomers are not in active, so this is irrelevant for performance.
    #   However, when computing transaction costs, even if the portfolio weight is zero,
    #   a value for Kyle's Lambda is required
    
    df_pf_t.loc[(df_pf_t['id'].isin(set(newcomers).intersection(set(zeros)))) 
             & 
             (df_pf_t['lambda'].isna()), 'lambda'] = 0
        
    if df_pf_t['lambda'].isna().sum() > 0:
        print("ERROR: A stock does not have a value for Kyle's Lambda")
        
    #---- Get stocks' return predictions ----
    #Merge values
    df_pf_t = df_pf_t.merge(df_returns.loc[df_returns['eom'] == date, ['id', 'tr']],
                            on = ['id'], how = 'left')
    
    #Set forecasts to 0 for leavers. This is irrelevant for performance, but 
    #   a missing value will cause issues when computing the revenue. 
    df_pf_t.loc[df_pf_t['id'].isin(leavers), 'tr'] = 0.0
    
    if df_pf_t['tr'].isna().sum() > 0:
        print("ERROR: A stock does not have a forecasted end-of-period return")
    
    #---- Compute G @ pi_{t-1} ----
    # Compute pi_g for pi_{t-1}
    pi_g = (df_pf_weights
            .query("eom == @prev_date")[['id', 'pi', 'g']]
            .assign(pi_g=lambda df: df['pi'] * df['g'])
            [['id', 'pi_g']]
            )
    
    # Merge to df_pf_t
    df_pf_t = (df_pf_t
               .merge(pi_g, on='id', how='left')
               .rename(columns = {'pi_g': 'pi_g_tm1'})
               )
    del pi_g
    
    #Set it to 0 for newcomers as pi_{t-1} = 0 for newcomers
    df_pf_t.loc[df_pf_t['id'].isin(newcomers),'pi_g_tm1'] = 0.0
    
    #---- Initialise 'pi_t' with 'pi_g_tm1' ----
    df_pf_t.loc[df_pf_t['id'].isin(active), 'pi'] = df_pf_t.loc[df_pf_t['id'].isin(active), 'pi_g_tm1']
    df_pf_t.loc[(df_pf_t['pi'] == 0.0) & df_pf_t['id'].isin(active), 'pi'] = 1e-16
    # Set any 'pi' for zeros to 0.0
    df_pf_t.loc[df_pf_t['id'].isin(zeros), 'pi'] = 0.0

    #==========================================================
    #           Solve for optimal portfolio
    #==========================================================
    
    #--- Define Torch Objects
    #Return Prediction for active stocks
    r = torch.tensor(return_predictions['tr_m_sp500_ld1'])
    #Covariance Matrix of Returns for active stocks
    S = torch.tensor(Sigma.to_numpy())
    #Diagonal of Kyle's Lambda matrix for active stocks
    L_diag = torch.tensor(kyles_lambda['lambda'])
    #Risk Aversion
    #gamma = torch.tensor(settings['gamma']) 
    #Wealth
    w = torch.tensor(df_wealth.loc[df_wealth['eom'] == date]['wealth'].iloc[0])
    #pi_g_{t-1}
    pi_g_tm1 = torch.tensor(df_pf_t.loc[df_pf_t['id'].isin(active), 'pi_g_tm1'].to_numpy())
    #Logits of pi_t
    pi_logits = torch.tensor(np.log(df_pf_t.loc[df_pf_t['id'].isin(active), 'pi'].to_numpy()),
                             requires_grad = True)
    
    #--- Optimizer
    optimizer = torch.optim.Adam([pi_logits], lr=1e-2) 
    
    #--- Define Constraint Parameters
    #Maximum portfolio weight
    max_pi = 0.05

    #Penalty for inequality constraints
    penalty_maxPi = 100.0
    penalty_var = 100.0
    
    #maximum allowed variance
    max_var = df_spy[df_spy['eom'] == prev_date]['variance'].iloc[0]
    
    #--- Gradient Ascent
    for _ in range(500):
        optimizer.zero_grad()
    
        # current portfolio (levels)
        pi = F.softmax(pi_logits, dim=0)
    
        # revenue:
        revenue = torch.dot(r, pi)
    
        # var penalty
        #OLD: var = 0.5 * gamma * pi @ S @ pi
        var_violation = penalty_var* F.relu(pi @ S @ pi - max_var)
    
        # transaction costs
        diff = pi - pi_g_tm1
        turnover_quad = (L_diag * diff.pow(2)).sum()
        tc = 0.5 * w * turnover_quad
    
        # Unconstrained objective to MAXIMISE
        F_val = revenue - tc
    
        #--- pi_max violation
        max_pi_violation = torch.relu(pi - max_pi)
        max_pi_violation = (penalty_maxPi * max_pi_violation).sum()

        #Full objective to MINIMISE
        loss = -F_val + max_pi_violation + var_violation

        loss.backward()
        optimizer.step()

    #Print maximised profit
    print(f"  Profit: {F_val.item()}")

    #Save portfolio weights
    df_pf_t.loc[df_pf_t['id'].isin(active), 'pi'] = pi.detach().cpu().numpy()
    #print pi_max value
    print(f"   MAX pi: {df_pf_t['pi'].max()}") 
    
    #Compute revenue & transaction costs
    df_pf_t = (df_pf_t
            .assign(rev = lambda df: df['pi']*df['tr'])
            .assign(tc = lambda df: (df['pi'] - df['pi_g_tm1'])**2 * df['lambda'] * float(w)/2 )
            )
    
    #Add to DataFrame
    df_pf_weights = df_pf_weights.set_index(['id', 'eom'])
    df_pf_weights.update(df_pf_t.set_index(['id', 'eom'])[['pi']])
    df_pf_weights = df_pf_weights.reset_index()
    
    #Append Result
    df_strategy.append(df_pf_t)

#%% Display Results

#Make DataFrame of strategy
results = pd.concat(df_strategy)

#Compute cumulative monthly profit for strategy
df_profit = (results
             .groupby('eom')
             .apply(lambda df: (df['rev'] - df['tc']).sum(), include_groups = False)
             .reset_index()
             .rename(columns = {0: 'strategy_profit'})
             )
df_profit['cumulative_return'] = (1 + df_profit['strategy_profit']).cumprod() - 1

#Compute cumulative monthly profit for benchmark
df_spy = df_spy[df_spy['eom'].isin(df_profit['eom'].unique())]
df_spy['cumulative_return'] = (1 + df_spy['ret']).cumprod() - 1

#Plot Comparison to Benchmark
plt.plot(df_profit['eom'], df_profit['cumulative_return'], label ='strategy')
plt.plot(df_profit['eom'], df_spy['cumulative_return'], label ='SPY')
plt.show()

#---- Compute Sharpe Ratio ----
#Merge risk-free Rate
df_profit = (df_profit
             .merge(risk_free, on = 'eom', how = 'left')
             .assign(ret_exc = lambda df: df['strategy_profit'] - df['rf'])
             )
df_spy = (df_spy.merge(risk_free, on = 'eom', how = 'left')
          .assign(ret_exc = lambda df: df['ret'] - df['rf'])
          )

#Sharpe Ratio Strategy
mu_s, sigma_s = df_profit['ret_exc'].mean(), df_profit['ret_exc'].std(ddof=1) 
Sharpe_s = np.sqrt(12) * (mu_s / sigma_s)

#Sharpe Ratio Benchmark
mu_b, sigma_b = df_spy['ret_exc'].mean(), df_spy['ret_exc'].std(ddof=1)
Sharpe_b = np.sqrt(12) * (mu_b / sigma_b)

#---- Compute Information Ratio ----
information_ratio = np.sqrt(12) * np.mean(df_profit['ret_exc'] - df_spy['ret_exc'])/((df_profit['ret_exc'] - df_spy['ret_exc']).std(ddof=1))

"""
An IR = 0.5 means:

For every 1% of tracking error (volatility of return relative to the benchmark), your portfolio earns 0.5% of excess return on average per year.

Put differently:

The strategy adds 0.5 units of active return per unit of active risk.
"""