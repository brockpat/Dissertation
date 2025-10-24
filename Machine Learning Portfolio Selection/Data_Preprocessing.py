# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:03:04 2025

@author: Patrick
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

#%% Read in Risk Free Rate
"""
Load Risk-Free Rate Data from Kenneth French's Website.
"""
# Read risk-free rate data (select only 'yyyymm' and 'RF' columns).
risk_free = pd.read_csv(path + "Data/FF_RF_monthly.csv", usecols=["yyyymm", "RF"])

# Convert to decimal (RF is given in percentage terms)
risk_free["rf"] = risk_free["RF"] / 100

#--- Construct an end-of-month date.
risk_free["eom"] = pd.to_datetime(risk_free["yyyymm"].astype(str) + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0)

# Keep only the required columns.
risk_free = risk_free[["eom", "rf"]]

print("Risk-free Rate Data Complete.")

#%% Market Returns 
"""
Load Market Return Data

Data Source: #https://www.dropbox.com/scl/fo/zxha6i1zcjzx8a3mb2372/AN9kkos5H5UjjXUOqW3EuDs?rlkey=i3wkvrjbadft6hld863571dol&e=1&dl=0
taken from: https://github.com/theisij/ml-and-the-implementable-efficient-frontier/tree/main?tab=readme-ov-file
"""
# Read in market returns data.
market = (pd.read_csv(path + "Data/market_returns.csv", dtype={"eom": str})
          #Only get US
          .pipe(lambda df: df[df['excntry'] == 'USA'])
          #Format Date
          .assign(eom_ret = lambda df: pd.to_datetime(df["eom"], format="%Y-%m-%d"))
          #Get relevant columns
          .get(["eom_ret", "mkt_vw_exc"])
          )
print("Market Return Loaded.")

#%% Stock Factors

#===================
# Read in Data
#===================

#Connect to Database
JKP_Factors = sqlite3.connect(database=path +"Data/JKP_US_SP500.db")

#List of Stock Features
features = get_features(exclude_poor_coverage = True)

print("Reading in Chars Data (Full Date Range)")

#Build query vector for features:
#   Set Empty String
query_features =""
#   Fill the Empty String with the features
for feature in features:
    query_features = query_features + feature + ", "
#   Delete last comma and space to avoid syntax errors
query_features = query_features[:-2]
#   Build final query vector
query = ("SELECT id, eom, sic, ff49, size_grp, me, crsp_exchcd, ret_exc, "
         +query_features 
         +" FROM Factors " 
         #+f"WHERE date BETWEEN '{start_date}' AND '{end_date}' "
         #+"AND CAST(id AS INTEGER) <= 99999" #Filter for CRSP observations (id <= 99999)
         # Not required as I only have CRSP Data in my DataSet anyway
         )

# Read in JKP characteristics data.
chars  = pd.read_sql_query(query, 
                           con=JKP_Factors,
                           parse_dates={'eom'}
                           )

#Convert features to numeric
for feature in features:
    chars[feature] = pd.to_numeric(chars[feature], errors = 'coerce')
#Convert id to int
chars['id'] = chars['id'].astype('int64')
#Convert sic to integer
chars['sic'] = pd.to_numeric(chars['sic'], errors = 'coerce', downcast='integer')
#Convert excess return to float
chars['ret_exc'] = pd.to_numeric(chars['ret_exc'], errors = 'coerce')

#Add additional columns:
#   dollar volume
chars["dolvol"] = chars["dolvol_126d"]
#   Kyle's Lambda (0.1 = price impact)
chars["lambda"] = 2 / chars["dolvol"] * 0.1  # Price impact for trading 1% of daily volume
#   Monthly version of rvol by annualizing daily volatility.
chars["rvol_m"] = chars["rvol_252d"] * (21 ** 0.5)

#Delete if Return not available
chars = chars.dropna(subset = 'ret_exc')

#==================================================
#              Data Screening
#==================================================

#Save key metrics before screening which clears out Data
n_start = len(chars)
me_start= np.sum(chars['me'].dropna())

#Require non-missing market equity 'me'.
pct_me_missing = chars["me"].isna().mean() * 100
print(f"   Non-missing me excludes {pct_me_missing:.2f}% of the observations")
chars = chars[~chars["me"].isna()]

# Require non-missing and non-zero dollar volume.
pct_dolvol_invalid = ((chars["dolvol"].isna()) | (chars["dolvol"] == 0)).mean() * 100
print(f"   Non-missing/non-zero dolvol excludes {pct_dolvol_invalid:.2f}% of the observations")
chars = chars[(~chars["dolvol"].isna()) & (chars["dolvol"] > 0)]

# Require valid SIC code.
pct_sic_invalid = (chars["sic"].isna()).mean() * 100
print(f"   Valid SIC code excludes {pct_sic_invalid:.2f}% of the observations")
chars = chars[~chars["sic"].isna()]

# Feature screens: count the number of non-missing features.
feat_available = chars[features].notna().sum(axis=1)
min_share = 0.4
min_feat = np.floor(len(features)) * min_share #50% of features with non-missings we want per stock
print(f"   At least {min_share * 100}% of feature excludes {round((feat_available < min_feat).mean() * 100, 2)}% of the observations")
chars = chars[feat_available >= min_feat] #Kick out observations with too many missing values
print(f"In total, the final dataset has {round((len(chars) / n_start) * 100, 2)}% of the observations and {round((chars['me'].sum() / me_start) * 100, 2)}% of the market cap in the data")


#==================================================
#               Compute total (raw) Return
#==================================================
#Compute total (raw) return
chars = (chars
            # Merge risk-free rate
            .merge(risk_free, on=['eom'])
            # Compute total (raw) return
            .assign(tr=lambda df: df['ret_exc'] + df['rf'])
            .drop('rf',
                  axis=1)
            )

#==================================================
#          Compute Excess Market Return
#==================================================
#Compute S&P 500 Return
sp500_rets = (chars
              .groupby('eom')[['tr','me']]
              .apply(lambda x: np.average(x.tr, weights = x.me))
              .reset_index()
              .rename(columns = {0:'sp500_ret'})
              )

#Compute Excess market return (tr_m_sp500)
chars = (chars
         .merge(sp500_rets, on = ['eom'], how = 'left')
         .assign(tr_m_sp500 = lambda df: df['tr'] - df['sp500_ret'])
         .drop(['sp500_ret'],axis=1)
         )
#==================================================
#          Lead Excess Market Return
#==================================================

chars = chars.assign(eom_lead = lambda df: df['eom'] + MonthEnd(1))
lead_ret = (chars
           .get(['id','eom','tr_m_sp500'])
           )

chars = (chars.merge(lead_ret, 
                     left_on = ['id','eom_lead'],
                     right_on = ['id','eom'],
                     how = 'left',
                     suffixes = ('','_ld1')
                     )
         .drop(['eom_ld1'],axis=1)
         )

#Save Memory
del lead_ret

#==================================================
#          Fill missing Excess Market Return
#==================================================
"""
If a stock leaves the S&P500, next month's return is no longer in the dataset.
This incurs a look-ahead bias such that at time t we'd know that in the next
month the stock is no longer in the S&P500. These leaded returns are filled
with the CRSP monthly dataset containing more than just the S&P 500.
"""

# Filter Permnos
permnos = chars.id.unique()
permnos = permnos_str = ', '.join(map(str, permnos))

query = ("SELECT PERMNO AS id, MthCalDt AS eom, MthRet AS tr " 
         "FROM Monthly_Returns "
        "WHERE "
        f"PERMNO IN ({permnos}) "
        f"AND MthCalDt BETWEEN '{chars.eom.min()}' AND '{chars.eom.max()}'"
        #US-listed stocks
        "AND ShareType = 'NS' "
        #security type equity
        "AND SecurityType = 'EQTY' "  
        #security sub type common stock
        "AND SecuritySubType = 'COM' "
        #US Incorporation Flag (Y/N)
        "AND USIncFlg = 'Y' " 
        #Issuer is a corporation
        "AND IssuerType in ('ACOR', 'CORP') " 
        #NYSE, AMEX, NASDAQ Stocks
        "AND PrimaryExch in ('N', 'A', 'Q') "
        #Stock Prices when or after issuence
        "AND ConditionalType in ('RW', 'NW') "
        #Actively Traded Stocks
        "AND TradingStatusFlg = 'A'"
        )

#Load CRSP monthly Returns
monthly_rets  = pd.read_sql_query(query, 
                           con=JKP_Factors,
                           parse_dates={'eom'}
                           )

#Drop Duplicates
monthly_rets = monthly_rets.drop_duplicates(subset = ['id','eom'])

#Make Date End of Month
monthly_rets['eom'] = monthly_rets['eom'] + pd.offsets.MonthEnd(0)

#Compute excess market return
monthly_rets = (monthly_rets
                .assign(tr = lambda x: pd.to_numeric(x['tr'], errors = 'coerce'))
                .merge(sp500_rets, on = ['eom'])
                .assign(tr_m_sp500 = lambda x: x['tr'] - x['sp500_ret'])
                )

#Merge to chars
chars = (chars
         .merge(monthly_rets[['id', 'eom', 'tr_m_sp500']],
                left_on=['id', 'eom_lead'],
                right_on=['id', 'eom'],
                how='left',
                suffixes=('', '_mr') # Add suffixes to distinguish 
                )
         .drop('eom_mr',axis=1)
         )
#Fill Missing Future Returns
chars['tr_m_sp500_ld1'] = chars['tr_m_sp500_ld1'].fillna(chars['tr_m_sp500_mr'])

#Drop Auxiliary Variable
chars = chars.drop(['tr_m_sp500_mr','eom_lead'],axis=1)

#Drop observations for which no lead return exists
chars = chars[~chars['tr_m_sp500_ld1'].isna()]

#Drop auxiliary dataset
del monthly_rets

#==================================================
#          Excess Market Return Dummys
#==================================================

chars = (chars
         .assign(tr_m_sp500_Dummy = lambda df: (df['tr_m_sp500']>0).astype(int))
         .assign(tr_m_sp500_ld1_Dummy = lambda df: (df['tr_m_sp500_ld1']>0).astype(int))
         )

#==================================================
#     Industry Classification (One-Hot Encoding)
#==================================================

#Get Industry Classification
chars["ff12"] = chars["sic"].apply(categorize_sic).astype(str)

#Get List of Industries
industries = sorted(chars["ff12"].dropna().unique())

#Enforce categorical type
chars['ff12'] = pd.Categorical(chars['ff12'], categories=industries)

#One-Hot Encoding
ff12_dummies = pd.get_dummies(chars['ff12'], prefix='ff12', drop_first=False).astype(int)

#Merge to chars
chars = pd.concat([chars, ff12_dummies], axis=1)

#Drop String Variables
chars = chars.drop('ff12',axis=1)

chars = chars.drop_duplicates(subset = ['eom','id'])

#Save Memory
del ff12_dummies


#==================================================
#               Save Preprocessing
#==================================================

#Save Chars to DataBase
chars.to_sql(name = 'Factors_processed', con = JKP_Factors, if_exists = 'replace', index = False)

print("Chars Data Complete")

JKP_Factors.close()

#%% Exogenous Wealth (AUM) Evolution
"""
For a given wealth level at the last date of the sample, compute backwards the
exogenous evolution of wealth according to the market return.

Using JKMP22 Equation (5), this wealth is used to update the portfolio weights, i.e.
g_t \pi_{t-1} is the initial portfolio weight in time period t.
"""
#Get the Wealth Evolution 
wealth = wealth_func(5e11, chars.eom.max(), market, risk_free)

wealth.to_csv(path + "Data/wealth_evolution.csv", index=False)
print("Wealth Evolution Complete.")