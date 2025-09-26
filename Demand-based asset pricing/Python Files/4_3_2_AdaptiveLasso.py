"""
Conduct adaptive LASSO
"""

#%% Libraries
import pandas as pd
import numpy as np

from sklearn.linear_model import LassoCV

#
weights = 'OLS_Standardized' #Only OLS_Standardized implemented. Code can easily be augmented to e.g. include Ridge Regression Results as weights.
gamma   = 1 #Weighting Factor of weights
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

#Load Ridge Results
df_weights = pd.read_csv(path + "/Output" + "/Variable Selection/" + weights + ".csv")
df_weights["rdate"] = pd.to_datetime(df_weights["rdate"])

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
#Baseline_exog_Variables_Names  = ['IVme','LNbe', 'profit', 'Gat', 'divA_be','beta']
#Baseline_LASSO_Variable_Names = ['IVme_stage1','LNbe', 'profit', 'Gat', 'divA_be','beta']

#%% Loop over Quarters and do Lasso in each bin

#Extract unique Dates
Quarters = Holdings['rdate'].unique()
Quarters = Quarters[Quarters.year<2023]

#Initialise DataFrame storing Lasso Results
results = pd.DataFrame()

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
       
    
    df_weights_Q = df_weights[df_weights["rdate"]==quarter]
    df_weights_Q = df_weights_Q.set_index(["rdate", "bin"])


    #Drop Missing Values
    df_Q = df_Q.dropna(subset=['LNrweight'] + Baseline_endog_Variables_Names + ['IVme'] + Characteristics_Names)
    
    #Drop categorical variables unnecessary for regression
    df_Q = df_Q.drop(['owntype','type'], axis=1)
    
    y = df_Q["LNme"]
    
    #--- Use the following with IV: in front to conduct adaptive LASSO with IV
    #   Will not change the results in any meaningful way
    #IV: X = df_Q[Baseline_exog_Variables_Names + Characteristics_Names]
    #IV: df_Q["IVme_stage1"] = sm.OLS(y, X).fit().predict(X)
    
    #Full Variable Names
    #IV: Names_Characteristics = Names_Additional_Characteristics + ['IVme_stage1','LNbe', 'profit', 'Gat', 'divA_be','beta']
    
    #Standardise explanatory variables
    #IV: df_Q[Baseline_LASSO_Variable_Names + Characteristics_Names] = df_Q.groupby(["rdate", "bin"], as_index=False)[Baseline_LASSO_Variable_Names + Characteristics_Names].transform(lambda x: (x - x.mean())/x.std())
    df_Q[Baseline_endog_Variables_Names + Characteristics_Names] = df_Q.groupby(["rdate", "bin"], as_index=False)[Baseline_endog_Variables_Names + Characteristics_Names].transform(lambda x: (x - x.mean())/x.std())

    #Fix if all values in a bin don't vary (standardisation fails & gives nan)
    df_Q = df_Q.fillna(0)
    
    #Get dependent variable
    df_Q["LNrweight_cons"] = df_Q["LNrweight"] - df_Q["cons"]
    
    #Assign a constant
    df_Q = df_Q.assign(constant=1)
    
    # Weights for adaptive LASSO
    weights= (np.abs(df_weights_Q)**(-gamma)).replace([np.inf, -np.inf], 1e12)
    
    #merge weights   
    df_Q = df_Q.set_index(["rdate", "bin"])
    #IV: new_X = (df_Q[Baseline_LASSO_Variable_Names + Characteristics_Names].div(weights)).reset_index()[Baseline_LASSO_Variable_Names + Characteristics_Names]
    new_X = (df_Q[Baseline_endog_Variables_Names + Characteristics_Names].div(weights)).reset_index()[Baseline_endog_Variables_Names + Characteristics_Names]
    df_Q = df_Q.reset_index()
    df_Q.update(new_X)

    # Five Fold CV Lasso estimation
    #IV: reg = df_Q.groupby(["rdate", "bin"]).apply(lambda x: LassoCV(
    #cv=5, random_state=0,max_iter=25_000,n_jobs=-1).fit(x[Baseline_LASSO_Variable_Names + Characteristics_Names + ['constant']], x["LNrweight_cons"]).coef_)

    reg = df_Q.groupby(["rdate", "bin"]).apply(lambda x: LassoCV(
    cv=5, random_state=0,max_iter=25_000,n_jobs=-1).fit(x[Baseline_endog_Variables_Names + Characteristics_Names + ['constant']], x["LNrweight_cons"]).coef_)

    
    #Coefficients
    #IV: coefs = pd.DataFrame([i for i in reg], index=reg.index, columns=Baseline_LASSO_Variable_Names + Characteristics_Names + ['constant'])
    coefs = pd.DataFrame([i for i in reg], index=reg.index, columns=Baseline_endog_Variables_Names + Characteristics_Names + ['constant'])


    results = pd.concat([results, coefs])
    
    #Save Results
    results.to_csv(path + "/Output" + "/Variable Selection" + "/Adaptive_LASSO_OLSWeights.csv")
