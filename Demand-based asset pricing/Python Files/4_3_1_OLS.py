"""
Conduct the unrestricted OLS Estimation.

Additionally, store the R^2 of the Regression if desired
"""

#%% Libraries
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

#%% Set Macro
Store_R2 = False #Store the R^2 from the Regression
KY19_baseline = False #Only compute the Baseline
Standardized = True #If OLS is used as weights for adaptive Lasso, the explanatory variables will be standardized which changes the beta coefficients

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

if KY19_baseline:
    Characteristics_Names = [] #Set additional characteristics empty so only baseline characteristics are picked

#Construct KY19 Baseline Variable Names
Baseline_endog_Variables_Names = ['LNme','LNbe', 'profit', 'Gat', 'divA_be','beta']

#%% Functions

# OLS Estimation that also stores the R^2
def OLS_with_R2(df):
    model = LinearRegression(fit_intercept = False) #intercept already included in df
    X = df[Baseline_endog_Variables_Names + Characteristics_Names + ['constant']]
    y = df["LNrweight_cons"]
    model.fit(X, y)
    coef = model.coef_
    r2 = model.score(X, y)  # Calculate R-squared
    return pd.Series(np.concatenate([coef, [r2]]), index=Baseline_endog_Variables_Names + Characteristics_Names + ['constant'] + ['R2'])

#%% Loop over Quarters and do Lasso in each bin

#Extract unique Dates
Quarters = Holdings['rdate'].unique()
Quarters = Quarters[Quarters.year<2023] #No characteristics post 2022

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
    
    if Standardized:
        #Standardise explanatory variables for Lasso
        df_Q[Baseline_endog_Variables_Names + Characteristics_Names] = df_Q.groupby(["rdate", "bin"], as_index=False)[Baseline_endog_Variables_Names + Characteristics_Names].transform(lambda x: (x - x.mean())/x.std())

        #Fix if all values in a bin don't vary (standardisation fails & gives nan)
        df_Q[Baseline_endog_Variables_Names + Characteristics_Names] = df_Q[Baseline_endog_Variables_Names + Characteristics_Names].fillna(0)


    #Get dependent variable with the investor fixed effect
    df_Q["LNrweight_cons"] = df_Q["LNrweight"] - df_Q["cons"]
    
    #Assign a constant
    df_Q = df_Q.assign(constant=1)

    # OLS Estimation
    if Store_R2:
        # OLS
        reg = df_Q.groupby(["rdate", "bin"]).apply(OLS_with_R2)

        # Coefficients + R2
        coefs = pd.DataFrame(reg)
    else:   
        reg = (df_Q
               .groupby(["rdate", "bin"])
               .apply(lambda x: LinearRegression(fit_intercept = False) #intercept already included in x
                      .fit(x[Baseline_endog_Variables_Names + Characteristics_Names + ['constant']], x["LNrweight_cons"]).coef_)
               )
    
        #Coefficients
        coefs = pd.DataFrame([i for i in reg], index=reg.index, columns=Baseline_endog_Variables_Names + Characteristics_Names + ['constant'])

    results.append(coefs)


results = pd.concat(results)
#Save Results
if Store_R2 and not Standardized:
    results.to_csv(path + "/Output" + "/Estimations" + "/OLS_unrestricted_R2.csv")
if not Store_R2 and Standardized:
    results.to_csv(path + "/Output" + "/Estimations" + "/OLS_Standardized.csv")
else:
    results.to_csv(path + "/Output" + "/Estimations" + "/OLS.csv")
