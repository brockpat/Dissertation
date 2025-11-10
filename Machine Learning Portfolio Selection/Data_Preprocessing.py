# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:03:04 2025

@author: Patrick
"""
#%% Libraries

path = "C:/Users/pbrock/Desktop/ML/"

#DataFrame Libraries
import pandas as pd
import sqlite3

#Turn off pandas performance warnings
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#Scientifiy Libraries
import numpy as np

import os
os.chdir(path + "Code/")
import General_Functions as GF

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

Data Source:https://www.dropbox.com/scl/fo/zxha6i1zcjzx8a3mb2372/AN9kkos5H5UjjXUOqW3EuDs?rlkey=i3wkvrjbadft6hld863571dol&e=1&dl=0
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

df_market_return = (market
                    .merge(risk_free, left_on = ['eom_ret'], right_on = ['eom'],
                           how = 'left')
                    .assign(mkt_vw = lambda df: df['mkt_vw_exc'] + df['rf'])
                    .drop(['eom'], axis = 1)
                    )
print("Market Return Loaded.")

#%% Stock Factors

#===================
# Read in Data
#===================

#Connect to Database
JKP_Factors = sqlite3.connect(database=path +"Data/JKP_clean.db")
JKP_Factors_SP500 = sqlite3.connect(database=path +"Data/JKP_SP500.db")
CRSP_monthly = sqlite3.connect(database=path +"Data/CRSP_monthly.db")
SP500_Constituents = sqlite3.connect(database = path + "Data/SP500_Constituents.db")


#List of Stock Features
features = GF.get_features(exclude_poor_coverage = True)

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
query = ("SELECT id, eom, sic, ff49, size_grp, me, crsp_exchcd, ret_exc, MthRet, "
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

#Delete if Return not available
chars = chars.dropna(subset = 'MthRet')
chars = chars.rename(columns = {'MthRet':'tr'})

#Add additional columns:
#   Kyle's Lambda (0.1 = price impact)
chars["lambda"] = 0.2 / chars["dolvol_126d"]  # Price impact for trading 1% of daily volume
#   Monthly version of rvol by annualizing daily volatility.
chars["rvol_m"] = chars["rvol_252d"] * (21 ** 0.5)

#Add S&P 500 indicator
df_sp500 = (pd.read_sql_query("SELECT * FROM SP500_Constituents",
                             con = SP500_Constituents,
                             parse_dates = {'eom'})
            .assign(in_sp500 = True)
            )

chars = (chars.merge(df_sp500, left_on = ['id', 'eom'], right_on = ['PERMNO', 'eom'],
                    how = 'left', suffixes = ("", "_delete"))
         .assign(in_sp500 = lambda df: df['in_sp500'].astype('boolean').fillna(False))
         .drop(columns = ['PERMNO'])
         )


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
pct_dolvol_invalid = ((chars["dolvol_126d"].isna()) | (chars["dolvol_126d"] == 0)).mean() * 100
print(f"   Non-missing/non-zero dolvol_126d excludes {pct_dolvol_invalid:.2f}% of the observations")
chars = chars[(~chars["dolvol_126d"].isna()) & (chars["dolvol_126d"] > 0)]

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


"""
Legacy Code: No longer required as MthRet merged from CRSP monthly is the total
return of a stock
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
"""

#==================================================
#          Compute Excess Market Return
#==================================================
#Compute value-weighted S&P 500 Return at end of month using beginning of month market equity

#Lag Market Equity to get begin of month market equity
sp500_rets = chars.get(['id','eom','me','tr'])
sp500_rets = sp500_rets.assign(eom_lead = lambda df: df['eom'] + pd.offsets.MonthEnd(1))
sp500_rets = (sp500_rets.merge(sp500_rets[['id','eom_lead','me']],
                              left_on = ['eom','id'], right_on = ['eom_lead','id'],
                              how = 'left',
                              suffixes = ("","_lag"))
                              .drop(columns = ['eom_lead', 'eom_lead_lag'])
                              .dropna()
                              )

#Compute market return over the month
sp500_rets = (sp500_rets
              .groupby('eom')[['tr','me_lag']]
              .apply(lambda x: np.average(x.tr, weights = x.me_lag))
              .reset_index()
              .rename(columns = {0:'sp500_ret'})
              )

#Save Returns
sp500_rets.to_sql(name = 'SP500_Return', con = JKP_Factors_SP500, if_exists = 'replace', index = False)

#Compute Excess market return (tr_m_sp500)
chars = (chars
         .merge(sp500_rets, on = ['eom'], how = 'left')
         .assign(tr_m_sp500 = lambda df: df['tr'] - df['sp500_ret'])
         .drop(['sp500_ret'],axis=1)
         )

#==================================================
#          Lead Excess Market Return
#==================================================

chars = chars.assign(eom_lead = lambda df: df['eom'] + pd.offsets.MonthEnd(1))
lead_ret = (chars
           .get(['id','eom', 'tr', 'tr_m_sp500'])
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

"""
LEGACY CODE: Data_Preprocessing is now done for the entire CRSP universe,
so this problem no longer exists.
#==================================================
#          Fill missing Excess Market Return
#==================================================
""
If a stock leaves the S&P500, next month's return is no longer in the dataset.
This incurs a look-ahead bias such that at time t we'd know that in the next
month the stock is no longer in the S&P500. These leaded returns are filled
with the CRSP monthly dataset containing more than just the S&P 500.

If no lead return exists, then the stock is dropped from the investable universe.
""

# Filter Permnos
permnos = chars.id.unique()
permnos = permnos_str = ', '.join(map(str, permnos))

query = ("SELECT PERMNO AS id, eom, MthRet AS tr " 
         "FROM CRSP_Monthly "
        "WHERE "
        f"PERMNO IN ({permnos}) "
        f"AND eom BETWEEN '{chars.eom.min()}' AND '{chars.eom.max()}'" 
        )

#Load CRSP monthly Returns
monthly_rets  = pd.read_sql_query(query, 
                           con=CRSP_monthly,
                           parse_dates={'eom'}
                           )

#Compute excess market return
monthly_rets = (monthly_rets
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
"""

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
chars["ff12"] = chars["sic"].apply(GF.categorize_sic).astype(str)

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

#Save Memory
del ff12_dummies


#==================================================
#               Save Preprocessing
#==================================================

#Save Chars to DataBase
chars.to_sql(name = 'Factors_processed', con = JKP_Factors, if_exists = 'replace', index = False)

print("Chars Data Complete")

#Connect to Database
JKP_Factors.close()
JKP_Factors_SP500.close()
CRSP_monthly.close()
SP500_Constituents.close()


#%% Exogenous Wealth (AUM) Evolution
"""
For a given wealth level at the last date of the sample, compute backwards the
exogenous evolution of wealth according to the market return.

Using JKMP22 Equation (5), this wealth is used to update the portfolio weights, i.e.
g_t \pi_{t-1} is the initial portfolio weight in time period t.

wealth is beginning of period, market return mu is end of period
"""
#Get the Wealth Evolution 
wealth = (GF.wealth_func(5e11, chars.eom.max(), market, risk_free)
          .assign(eom = lambda df: pd.to_datetime(df['eom']))
          )


wealth.to_csv(path + "Data/wealth_evolution.csv", index=False)
print("Wealth Evolution Complete.")