"""
Conduct the LASSO Variable Selection without IV
"""

#%% Libraries
import pandas as pd
import numpy as np

from sklearn.linear_model import LassoCV

import statsmodels.api as sm
#%% Read in Raw Data

path = ".../Python Replication Package"

#Load Holdings Data
Holdings = pd.read_stata(path + "/Data" + "/Data1_clean_correct_bins.dta")
Holdings['rdate'] =  pd.to_datetime(Holdings["rdate"]) #if reading in csv
# -----------------------------------------------------------------------------
#   !!! IMPORTANT !!!
#   Check Date format rdate. It must be end of Quarter Dates. If not, 
#   shift the dates to end of Quarter. Sometimes, reading in .dta files
#   can make End of Quarter Dates to Begin of Quarter Dates.
#   df['rdate'] = df['rdate'] + pd.offsets.MonthEnd(3)
# -----------------------------------------------------------------------------

#Load Baseline Stock Characteristics
StocksQ = pd.read_stata(path + "/Data" + "/StocksQ.dta")
StocksQ["date"] = StocksQ["date"] - pd.offsets.MonthBegin()+pd.offsets.MonthEnd()

#Load Additional Stock Characteristics
chars = pd.read_csv(path + "/Output" + "/Additional_Stock_Characteristics_Imputed_Winsorized.csv").drop(['Unnamed: 0'],axis = 1)
chars["rdate"] = pd.to_datetime(chars["rdate"])

#Create list of additional Stock Characteristics
Characteristics_Names = list(chars.columns.drop(['rdate','permno']))

#Construct KY19 Baseline Variable Names
Baseline_endog_Variables_Names = ['LNme','LNbe', 'profit', 'Gat', 'divA_be','beta']

#%% Loop over Quarters and do Lasso in each bin

#Extract unique Dates
Quarters = Holdings['rdate'].unique()
Quarters = Quarters[Quarters.year<2023]

#Initialise DataFrame storing Lasso Results
results = []

#Loop over quarter
for quarter in Quarters:
    print(quarter)
   
    #Slice Datasets
    Holdings_Q = Holdings[Holdings['rdate'] == quarter]
    chars_Q = chars[chars['rdate'] == quarter]
    StocksQ_Q = StocksQ[StocksQ['date'] == quarter]

    #Merge Datasets
    df_Q = (Holdings_Q
            .merge(chars_Q, 
                   left_on=["rdate", "permno"], right_on=["rdate", "permno"], 
                   how = "left", suffixes=["", "_new"])
            )
    df_Q = (df_Q
            .merge(StocksQ_Q[["permno", "date", "LNme", 'LNbe', 'profit', 'Gat', 'divA_be','beta']], 
                   left_on=["rdate", "permno"], right_on=["date", "permno"])
            )
       
    #Drop Missing Values and zeros
    df_Q = df_Q.dropna(subset=['LNrweight'] + Baseline_endog_Variables_Names + ['IVme'] + Characteristics_Names)
        
    #Standardise explanatory variables for Lasso
    df_Q[Baseline_endog_Variables_Names + Characteristics_Names] = df_Q.groupby(["rdate", "bin"], as_index=False)[Baseline_endog_Variables_Names + Characteristics_Names].transform(lambda x: (x - x.mean())/x.std())
    
    #Delete features which are constant (standardisation gives a nan then)
    df_Q[Baseline_endog_Variables_Names + Characteristics_Names] = df_Q[Baseline_endog_Variables_Names + Characteristics_Names].fillna(0)
    
    #Get dependent variable with the investor fixed effect
    df_Q["LNrweight_cons"] = df_Q["LNrweight"] - df_Q["cons"]
    
    #Assign a constant
    df_Q = df_Q.assign(constant=1)

    # Five Fold CV Lasso estimation
    reg = (df_Q
           .groupby(["rdate", "bin"])
           .apply(lambda x: LassoCV(cv=5, random_state=0,max_iter=25_000,n_jobs=-1,fit_intercept=False)#Intercept already included in x
                  .fit(x[Baseline_endog_Variables_Names + Characteristics_Names + ['constant']], x["LNrweight_cons"]).coef_)
           )
    

    #Coefficients
    coefs = pd.DataFrame([i for i in reg], index=reg.index, columns=Baseline_endog_Variables_Names + Characteristics_Names + ['constant'])
    #Store Results
    results.append(coefs)
    
#Export Results
results = pd.concat(results)
results = results.assign(constant = 0) # Set constant to zero. Will cause bugs otherwise later on. We always estimate
#GMM with a constant which is hard coded and would cause errors if in results constant is sometimes chosen

#Save Results
results.to_csv(path + "/Output" + "/Variable Selection" + "/LASSO.csv")