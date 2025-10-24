# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 10:31:00 2025

@author: Patrick

Given the pre-processed factors, this code computes standardisation of the signals
and saves them to the database. Cross-sectional Z-scores and rank standardisation 
for each feature are implemented.
"""

#%% Libraries

path = "C:/Users/pbrock/Desktop/ML/"

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

"""
5 years = 60 months. e.g. 1981-01-31 until 1986-12-31
"""

#%% Z-Scores of Features

def Feat_Standardisation(df, num_cols, one_hot_cols, stock_id, date, method):
    """
    Parameters
    ----------
    df : pd.DataFrame.
    num_cols : list[str]. List of columns to compute Z-scores on.
    one_hot_cols : list[str]. List of columns to remain unedited.
    stock_id : str. Stock identifier.
    date : str. Date Variable

    Returns
    -------
    DataFrame containing the stock identifier, date, the Z-scored
           num_cols, and the original unedited one_hot_cols.
           Missing values after standardization are imputed with 0.0

    """
    def standardise_col(series, method):
        """
        Compute the Z-scores of a pandas series
        """
        if method == "Z_score":
            return (series - series.mean())/series.std()
        elif method == "Rank":
            return (series.rank(method='average',pct=True)*2-1).fillna(0)
        else: return ValueError

    def standardise_df(df, num_cols:list, stock_id:str, date_id:str):
        """
        Compute the cross-sectional (date_id) Z-scores 
        of columns (num_cols) from a pandas dataframe (df).
        The stock identifier (stock_id) is kept for identification purposes.
        
        Missing values are filled with 0.
        """
        #Standardise features in the cross-section
        df_standardised = df.set_index([stock_id]).groupby(date_id)[num_cols].apply(standardise_col, method = method)
        
        #Fill missing values with 0
        #df_standardised = df_standardised.fillna(0)
        
        return df_standardised
    
    df_standardised = standardise_df(df, num_cols, stock_id, date_id)
    
    df_standardised = pd.concat([df_standardised, df.set_index([date_id,stock_id]).loc[df_standardised.index, one_hot_cols]],
                     axis = 1).reset_index()
    
    return df_standardised

#%%
#Read in processed characteristics
JKP_Factors = sqlite3.connect(database=path +"Data/JKP_US_SP500.db")
chars  = pd.read_sql_query("SELECT * FROM Factors_processed", 
                           con=JKP_Factors,
                           parse_dates={'eom'}
                           )

#Signals: First list numerical features, second list one-hot-encodings
signals = get_signals()

#Compute Z-scores
df_Z = Feat_Standardisation(chars, signals[0], signals[1], stock_id = 'id', date = 'eom', method = 'Z_score')

#Compute Rank-Standardisation
df_rank = Feat_Standardisation(chars, signals[0], signals[1], stock_id = 'id', date = 'eom', method = 'Rank')

#Save to DataBase
df_Z.to_sql(name = 'Signals_ZScore', con = JKP_Factors, if_exists = 'replace', index = False)
df_rank.to_sql(name = 'Signals_Rank', con = JKP_Factors, if_exists = 'replace', index = False)

JKP_Factors.close()