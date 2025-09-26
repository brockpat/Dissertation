# -*- coding: utf-8 -*-
"""
Conduct the Backward variable selection procedure using IV2SLS
"""
#%% Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot  as plt

import statsmodels.api as sm
from statsmodels.api import OLS

import scipy
import scipy.optimize
import scipy.stats
from scipy.stats.mstats import winsorize

from linearmodels.iv import IV2SLS

import time

import copy

#%% Functions
def remove_lowest_significance(tstat,alpha=0.05):
    """
    Removes the variable with the lowest t-stat (excluding LNme). Computes
    values under alpha = 10% which corresponds to a |t-value|>1.64 rejecting the Null.
    """
    #Smallest t-value for variable to be significant
    quantile = scipy.stats.norm.ppf(1-alpha/2)
    
    bool_remove = False
    
    #If no variable deemed significant, we need to stop the deletion manually
    if len(tstat.index) == 1:
        return tstat, False
    
    else:
        lowest_value = tstat.drop(['constant']).min()
        lowest_index = tstat.drop(['constant']).idxmin()
        
        if lowest_value < quantile:
            tstat.drop(lowest_index,inplace = True)
            bool_remove = True
            
        return tstat, bool_remove


def OLS_tstats(df_Q_bin,selected_characteristics,
                            selected_instruments):
    
    #Filter out the Zeros for linear Regression
    df_Q_bin = df_Q_bin[df_Q_bin.rweight>0]
    
    #Get vector of independent variable (subtract cons so that constant has the right level)
    y_reg = np.log(df_Q_bin.rweight)
    y_reg = y_reg - df_Q_bin['cons']
    
    #If LNme in selected_characteristics perform IV regression
    if 'LNme' in selected_characteristics:
    
        #Get X and Z matrix
        X_reg = df_Q_bin[[var for var in [selected_characteristics + ['IVme']][0] if var != 'cons']]
                    
        #Do 2SLS Regression
        tstat = IV2SLS(dependent=y_reg, exog=
                             X_reg.drop(columns=['LNme', 'IVme'],inplace=False), 
                             endog=X_reg['LNme'], instruments=X_reg['IVme']).fit().tstats
                
    else:
        X_reg = df_Q_bin[[var for var in selected_characteristics  if var != 'cons']]
        
        tstat = OLS(endog=y_reg, exog=X_reg).fit().tvalues

    tstat = tstat.abs().sort_values()
    
    return tstat
    
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

#Identify linearly independent columns of a matrix
def find_independent_columns(df, selected_characteristics):
    lin_dependent_columns = []
    lin_independent_columns = copy.deepcopy(selected_characteristics)
    #Add var for far and check rank of matrix. I rank doesn't increase, delete additional var
    for i in range(1,len(selected_characteristics[:-2])):
        dif = i - np.linalg.matrix_rank(df_Q_bin[lin_independent_columns[:i]])
        if dif > 0:
            lin_independent_columns = list(pd.Series(index=selected_characteristics).drop([selected_characteristics[i-1]]).index)
            lin_dependent_columns = lin_dependent_columns + [selected_characteristics[i-1]]
    return lin_independent_columns, lin_dependent_columns

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
#Delete Variables which are mostly constant (will cause rank issues in OLS estimation)
Characteristics_Names = [var for var in Characteristics_Names if var not in ["Spinoff", 'DivInit', 'DivOmit',
                                                                             'zerotrade','zerotradeAlt12', 'zerotradeAlt1']]
Baseline_endog_Variables_Names = ['LNme','LNbe', 'profit', 'Gat', 'divA_be','beta']
Baseline_exog_Variables_Names  = ['IVme','LNbe', 'profit', 'Gat', 'divA_be','beta']

#DataFrame storing the Variable Selection
#df_select = pd.DataFrame(columns = ['rdate'] + ['bin'] +  Baseline_endog_Variables_Names + Characteristics_Names + ['constant'])
#df_select.set_index(['rdate', 'bin'],inplace=True)
df_select = []

#%% GMM Estimation

#Extract unique dates
Quarters = Holdings['rdate'].unique()
Quarters = Quarters[Quarters.year < 2023]

for quarter in Quarters:
    print(quarter)

    ### --- Slice Datasets
    #Holdings Sliced
    Holdings_Q = Holdings[Holdings['rdate'] == quarter]
    
    #Baseline Stock Characteristics Sliced
    StocksQ_Q = StocksQ[StocksQ['date'] == quarter]
    
    #Additional Stock Characteristics Sliced
    chars_Q = chars[chars['rdate'] == quarter]
        
    ### --- Merge Stock Characteristics to Holdings Data to build X & Z Matrix for GMM
    #Merge Baseline Stock Characteristics
    df_Q = Holdings_Q.merge(StocksQ_Q[["permno", "date", "LNme", 'LNbe', 'profit', 'Gat', 'divA_be','beta']], 
                            left_on=["rdate", "permno"], right_on=["date", "permno"])
    df_Q = df_Q.merge(chars_Q, left_on=["rdate", "permno"], right_on=["rdate", "permno"], how = "left", suffixes=["", "_new"])

    ### --- Mild data cleaning
    #Drop any remaining Missing Values to avoid errors
    df_Q = df_Q.dropna(subset=Characteristics_Names + Baseline_endog_Variables_Names + ['IVme'] + ['rweight'])
    
    #Assign Constant
    df_Q = df_Q.assign(constant=1)
    
    step_size = 1
    
    ### --- Loop over each individual bin
    for i_bin in np.sort(df_Q['bin'].unique()):
        print("     Bin:" + str(i_bin))
        
        ### --- Slice DataSet on Bin
        df_Q_bin = df_Q[df_Q['bin'] == i_bin]
        
        ### --- Generate Dataframe to Save BackwardSelection Results of Bin
        df_select_bin = pd.DataFrame(columns = ['rdate'] + ['bin'] +  Baseline_endog_Variables_Names + Characteristics_Names + ['constant'])
        df_select_bin.at[0,'rdate'] = quarter
        df_select_bin.at[0,'bin'] = i_bin
        df_select_bin[Baseline_endog_Variables_Names + Characteristics_Names + ['constant']] = 0
    
        
        ### --- Generate List of selected characteristics which are updated in each iteration
        #!!!! Make sure 'cons' is always the last element in selected characteristics
        selected_characteristics = Baseline_endog_Variables_Names + Characteristics_Names +  ['constant', 'cons']
        
        #Delete constant variables as they're linearly dependent with the constant
        boolean_var_deleted = True
        while boolean_var_deleted:
            selected_characteristics, boolean_var_deleted = delete_Constant_Characteristics(df_Q_bin, selected_characteristics)
            
        #Extract linearly dependent columns
        _, lin_dependent_columns = find_independent_columns(df_Q_bin, selected_characteristics)
        
        #Only keep lin independent columns. Maintain order of selected_characteristics (important)
        selected_characteristics = [var for var in selected_characteristics if var not in lin_dependent_columns]
        
        #Selected Instruments do not get 'cons' as a variable
        selected_instruments = ['IVme' if var == 'LNme' else var for var in selected_characteristics if var !='cons']   
        
        ### --- Initialise While loop for backward selection
        bool_remove = True #As long as a variable is removed, the backward selection continues
        while bool_remove:
        
            #----------------------------------- Step 1 -----------------------------------
            #Get Initial Guesses for beta by using linear log-log regression ignoring the zeros
            tstat = OLS_tstats(df_Q_bin,selected_characteristics,
                                        selected_instruments)
            
            #----------------------------------- Step 2 -----------------------------------
            #Delete Variable with the smalles tstat
                        
            #Delete most insignificant variable
            tstat, bool_remove =  remove_lowest_significance(tstat, alpha = 0.05)
            selected_characteristics = list(tstat.index) + ['cons']
            selected_instruments = ['IVme' if var == 'LNme' else var for var in selected_characteristics if var !='cons']
            
        
        
        df_select_bin[tstat.index] = 1
        df_select.append(df_select_bin)

#Save Data
df_select = pd.concat(df_select, ignore_index=True)
df_select.to_csv(path + "/Output" + "/Variable Selection" + "/BackwardSelection_IV2SLS"  + ".csv", index = False)