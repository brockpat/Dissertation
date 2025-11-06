# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 13:21:18 2025

@author: Patrick

Computes the return of the SPDR (Standard & Poor's Depositary Receipt) ETF) of
S&P 500 as a benchmark

Can also compare against other ETFs:
    Momentum: iShares MSCI USA Momentum Factor ETF (MTUM)
    
    Quality: iShares MSCI USA Quality Factor ETF (QUAL)
    
    Minimum Volatility: iShares MSCI Min Vol USA ETF (USMV)

Other benchmark:
    Jensen's Alpha:
    
    This is the "gold standard" for proving alpha. You run a regression of your portfolio's excess returns against the market's excess returns:
    (πt′​rt+1​−rf,t+1​)=α+β(SP500TRt+1​−rf,t+1​)+ϵt​
    
    Your Goal: You want a positive and statistically significant α (alpha). This proves you generated returns after accounting for all the market risk (beta) you took.
"""
#%% Libraries
import yfinance as yf
import pandas as pd
import sqlite3


path = "C:/Users/pf122/Desktop/Uni/Frankfurt/2023-24/Machine Learning/Single Authored/"

#%% SPY S&P 500 Returns

JKP_Factors = sqlite3.connect(database=path +"Data/JKP_US_SP500.db")


#---- Download Data ----
spyticker = yf.Ticker("^SP500TR") #Total Return (Dividends reinvested)
#'Close' is adjusted close due to auto_adjust = True
df_spy = spyticker.history(period="max", interval="1d", start="2003-03-01", end="2025-12-31", 
                              auto_adjust=True, rounding=True)

# ---- Reformat Date ----
#Get rid of hours, minutes,...
df_spy.index = df_spy.index.tz_localize(None).normalize()

#Fill missing days (non-trading days) with values from last trading day
df_spy = (df_spy.reindex(pd.date_range(start=df_spy.index.min(), 
                                      end=df_spy.index.max(), 
                                      freq='D'))
          .ffill()
          .reset_index()
          .rename(columns={'index': 'Date'})
          )

#Keep end of month close to match frequency of other data
df_spy = df_spy.loc[df_spy['Date']
                    .isin(pd.date_range(start = "2003-03-31", 
                                        end = "2024-12-31", 
                                        freq = "M")
                          )
                    ]

# ---- Compute return ----
monthly_returns = df_spy['Close'].pct_change().dropna().to_frame().reset_index(drop=True)
monthly_returns['eom'] = pd.date_range(start = "2003-03-31", 
                                       end = "2024-12-31", 
                                       freq = "M")[1:]
monthly_returns = monthly_returns.rename(columns = {'Close':'SPY_ret'})

monthly_returns.to_sql(name = 'SPY_Return', 
                       con = JKP_Factors, if_exists = 'replace', index = False)

JKP_Factors.close()