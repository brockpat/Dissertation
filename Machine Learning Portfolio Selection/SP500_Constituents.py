# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 13:18:26 2025

@author: Patrick

Create a DataFrame that for each month displays all the constituents (stock_id = PERMNO)
of the S&P 500.
"""

#%% Libraries
import pandas as pd
import sqlite3

path = "C:/Users/pf122/Desktop/Uni/Frankfurt/2023-24/Machine Learning/Single Authored/"

#%%

SP500_Constituents = sqlite3.connect(path + "Data/SP500_Constituents.db")

#Read in DataSet
df = pd.read_csv(path + "Data/SP500_Constituents.csv")

#Convert to dates
df["MbrStartDt"] = pd.to_datetime(df["MbrStartDt"])
df["MbrEndDt"] = pd.to_datetime(df["MbrEndDt"])

#Get List of months
month_ends = pd.date_range("1950-01-31", "2024-12-31", freq="M")

#For each Month, filter the stocks that are in the S&P 500
records = []
for _, row in df.iterrows():
    mask = (month_ends >= row["MbrStartDt"]) & (month_ends <= row["MbrEndDt"])
    valid_months = month_ends[mask]
    records.extend([(row["PERMNO"], d) for d in valid_months])

#Create DataFrame
df_sp500 = (pd.DataFrame(records, columns=["PERMNO", "MonthEnd"])
            .rename(columns = {'MonthEnd':'eom'})
            .reset_index(drop = True)
            )

#Save to DataBase
df_sp500.to_sql(name = "SP500_Constituents", if_exists = 'replace', 
                con = SP500_Constituents, index = False)

SP500_Constituents.close()