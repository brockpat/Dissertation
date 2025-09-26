# -*- coding: utf-8 -*-
"""
Conduct the Backward variable selection procedure using GMM. 
Default does not use the IV to greatly improve statistical power.

If IV estimation desired, set the macro No_IV to FALSE.
"""
#%% Libraries
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from scipy.stats import uniform
import pickle
import copy



#%% Classes

class loguniform_int:
    """Integer valued version of the log-uniform distribution"""

    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)


class uniform_int:
    """Integer valued version of the uniform distribution"""

    def __init__(self, a, b):
        self._distribution = uniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)


#%%
No_IV = True #Select whether to turn off IV estimation

#%% Functions


def delete_Constant_Characteristics(df_Q_bin, selected_characteristics):
    len_selected_chars_before = len(selected_characteristics)
    check_columns = copy.deepcopy(selected_characteristics)
    check_columns.remove('constant')
    check_columns.remove('cons')
    # Calculate the standard deviation for each column
    std_devs = df_Q_bin[check_columns].std()

    # Identify columns with a standard deviation of zero
    zero_std_columns = std_devs[std_devs == 0].index.tolist()

    for item in zero_std_columns:
        selected_characteristics.remove(item)

    len_selected_chars_after = len(selected_characteristics)

    boolean_var_deleted = False
    if len_selected_chars_after - len_selected_chars_before !=0:
        boolean_var_deleted = True

    return selected_characteristics, boolean_var_deleted

#%% Read in Data

#Greene Theorem THEOREM 13.2 and Newey & McFadden (1994) Theorem 3.4, Wikipedia: https://en.wikipedia.org/wiki/Generalized_method_of_moments

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

chars = pd.read_csv(path + "/Output" + "/Additional_Stock_Characteristics_Imputed_Winsorized.csv")
chars["rdate"] = pd.to_datetime(chars["rdate"])
chars.drop(['Unnamed: 0'],axis = 1,inplace=True)

#Construct variable names of BASELINE Stock Characteristics, including IVme
Characteristics_Names = list(chars.columns.drop(['rdate','permno']))
#Delete Variable different zero trade specifications
Characteristics_Names = [var for var in Characteristics_Names if var not in ['zerotradeAlt12', 'zerotradeAlt1']]
Baseline_endog_Variables_Names = ['LNme','LNbe', 'profit', 'Gat', 'divA_be','beta']
Baseline_exog_Variables_Names  = ['IVme','LNbe', 'profit', 'Gat', 'divA_be','beta']

df_select = pd.DataFrame(columns = ['rdate'] + ['bin'] +  Baseline_endog_Variables_Names + Characteristics_Names + ['constant'])
df_select.set_index(['rdate', 'bin'],inplace=True)

#%% GMM Estimation

#Extract unique dates
Quarters = Holdings['rdate'].unique()

if No_IV:
    baseline_variables = Baseline_endog_Variables_Names
else:
    baseline_variables = Baseline_exog_Variables_Names

for quarter in Quarters:
    print(quarter)

    ### --- Slice Datasets
    #Holdings Sliced
    Holdings_Q = Holdings[Holdings['rdate'] == quarter]
    
    #Baseline Stock Characteristics Sliced
    StocksQ_Q = StocksQ[StocksQ['date'] == quarter]
    
    #Additional Stock Characteristics Sliced
    chars_Q = chars[chars['rdate'] == quarter]
        
    ### --- Merge Stock Characteristics to Holdings Data 
    df_Q = Holdings_Q.merge(StocksQ_Q[["permno", "date", "LNme", 'LNbe', 'profit', 'Gat', 'divA_be','beta']], 
                            left_on=["rdate", "permno"], right_on=["date", "permno"])
    df_Q = df_Q.merge(chars_Q, left_on=["rdate", "permno"], right_on=["rdate", "permno"], how = "left", suffixes=["", "_new"])

    ### --- Mild data cleaning
    #Drop any remaining Missing Values to avoid errors
    df_Q = df_Q.dropna(subset=Characteristics_Names + baseline_variables  + ['rweight'])
    
    #Standardise explanatory variables
    df_Q[baseline_variables + Characteristics_Names] = df_Q.groupby(["rdate", "bin"], as_index=False)[baseline_variables + Characteristics_Names].transform(lambda x: (x - x.mean())/x.std())
    
    #Fix if all values in a bin don't vary (standardisation fails gives nan)
    df_Q = df_Q.fillna(0)
    
    #Get dependent variable
    df_Q["rweight_y"] = df_Q["rweight"] - df_Q["cons"]
    
    
    param_distributions = {
    "learning_rate": loguniform(0.001, 1),
    "max_iter": loguniform_int(10, 1000),
    "max_depth": uniform_int(1, 10),
    "min_samples_leaf": loguniform_int(1, 100),
    }

    # Five Fold CV Lasso estimation
    for i_bin in df_Q.bin.unique():
        print(i_bin)
        reg = df_Q[df_Q.bin==i_bin].groupby(["rdate", "bin"]).apply(
            lambda x: RandomizedSearchCV(HistGradientBoostingRegressor(),  
                param_distributions=param_distributions,
                n_iter=100,
                cv=5
            ).fit(x[baseline_variables + Characteristics_Names], x["rweight_y"])
        )

        reg = reg.reset_index()
        reg = reg.rename(columns = {0:"model"})

        for j in range(len(reg)):
            pickle.dump(reg.model[j], open(path + "/Output" + "/Variable Selection/GBR/Models/" + 
                                            str(quarter.year)  + str(quarter.month) + str(quarter.day) + "_bin_" + str(reg.bin[j]) +".sav", "wb"  ) )





