# -*- coding: utf-8 -*-
"""
Treat the characteristics-based demand equation as the true data generating process
and simulate portfolio holdings data
"""

#%% Libraries
import pandas as pd

pd.options.mode.chained_assignment = None #Inapplicable warning messages are turned off

import numpy as np
import scipy.optimize


#%%
#Compute parameters mu and sigma of the log-normal distribution to enforce mean 1 and variance from the GMM Estimation
def compute_Moments(x, sigma_squared_bar):
    mu = x[0]
    sigma = x[1]
    
    error1 = mu + sigma**2/2 - 0
    error2 = np.log(np.exp(sigma**2) - 1) + 2*mu+sigma**2 - np.log(sigma_squared_bar)
    
    error = np.array([error1,error2])
    
    return error  
#%% Load Data

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

#Construct variable names of BASELINE Stock Characteristics, including IVme
Baseline_endog_Variables_Names = ['LNme','LNbe', 'profit', 'Gat', 'divA_be','beta']

GMM_Estimates = pd.read_csv(path + "/Output" + "/Estimations" + "/KY19_baseline_unrestricted.csv")
GMM_Estimates['rdate'] = pd.to_datetime(GMM_Estimates["rdate"]) #if reading in csv
GMM_Estimates = GMM_Estimates.drop_duplicates(subset = ['rdate','bin'])
GMM_Estimates = GMM_Estimates[GMM_Estimates['Error'] < 1]

Baseline_Regressor_Names = [var + "_beta_true" for var in Baseline_endog_Variables_Names]

#%%

#Extract unique dates
Quarters = GMM_Estimates['rdate'].unique()

results = pd.DataFrame()

for quarter in Quarters:
    print(quarter)
    
    ### --- Slice Datasets
    #Holdings Sliced
    Holdings_Q = Holdings[Holdings['rdate'] == quarter][['rdate', 'mgrno', 'bin', 'permno', 'aum', 'rweight', 'cons']]
    
    ### --- Slice Datasets
    #Holdings Sliced
    GMM_Estimates_Q = GMM_Estimates[GMM_Estimates['rdate'] == quarter]
    
    #Baseline Stock Characteristics Sliced
    StocksQ_Q = StocksQ[StocksQ['date'] == quarter]
    
    
        
    ### --- Merge Stock Characteristics to Holdings Data to build X & Z Matrix for GMM
    #Merge Baseline Stock Characteristics
    df_Q = Holdings_Q.merge(StocksQ_Q[["permno", "date", "LNme", 'LNbe', 'profit', 'Gat', 'divA_be','beta']], 
                            left_on=["rdate", "permno"], right_on=["date", "permno"]).drop('date',axis=1)
    
    df_Q = df_Q.assign(constant=1) 

    
    df_Q = df_Q.merge(GMM_Estimates_Q, on = ['rdate','bin'], suffixes=("","_beta_true")).drop('Error',axis=1)

    ### --- Mild data cleaning
    #Drop any remaining Missing Values to avoid errors
    df_Q = df_Q.dropna(subset=Baseline_endog_Variables_Names + ['rweight'])
    
    
    ### --- Loop over each individual bin
    unique_bins = GMM_Estimates_Q['bin'].unique()
    unique_bins = unique_bins[~np.isnan(unique_bins)]
    
    for i_bin in np.sort(unique_bins):
        print("     Bin: " + str(i_bin))
        
        ### --- Slice DataSet on Bin
        df_Q_bin = df_Q[df_Q['bin'] == i_bin]
        
        ### --- Given beta, compute epsilon
        y = df_Q_bin['rweight'].to_numpy()
        X = df_Q_bin[Baseline_endog_Variables_Names + ['constant', 'cons']].to_numpy()
        RegCoeffs = df_Q_bin[Baseline_Regressor_Names + ['constant_beta_true']].head(1).to_numpy().reshape(-1) #All RegCoeffs in a bin are the same
                
        df_Q_bin.loc[:,'epsilon'] = y* np.exp( -X[:,:-1] @ RegCoeffs - X[:,X.shape[1]-1] )
        
        
        ### --- Compute moments to simulate epsilon such that epsilon is log-normal with mean 1 and preserved variance
        epsilon_var = df_Q_bin[df_Q_bin['epsilon']>0]['epsilon'].var()
        
        moments = scipy.optimize.root(compute_Moments, [1,1], args = (epsilon_var))
        mu = moments.x[0]
        sigma = np.abs(moments.x[1])
        
        draws = np.exp(np.random.normal(mu,sigma, size = len(df_Q_bin.epsilon)))
        
        ### --- Simulate the probability for a zero
        epsilon_probabilityZero = len(df_Q_bin[df_Q_bin['epsilon']==0])/len(df_Q_bin.epsilon)
        
        draws_zeros = np.random.uniform(low=0.0, high=1.0, size=len(df_Q_bin.epsilon))
        draws_zeros[draws_zeros > epsilon_probabilityZero] = 1
        draws_zeros[draws_zeros < epsilon_probabilityZero] = 0
        
        ### --- Set the simulated epsilon
        df_Q_bin.loc[:,'epsilon_sim'] = draws
        df_Q_bin.loc[:,'zero_sim'] = draws_zeros
        
        ### --- Set the zeros
        df_Q_bin.loc[df_Q_bin['zero_sim'] == 0, 'epsilon_sim'] = 0
        
        ### --- Compute simulated rweight
        df_Q_bin.loc[:,'rweight_sim'] = np.exp( X[:,:-1] @ RegCoeffs + X[:,X.shape[1]-1] )*df_Q_bin['epsilon_sim'].to_numpy()


        results = pd.concat([results,df_Q_bin])
    
    #Save Results per Quarter
    results.to_stata(path + "/Output" + "/Simulated_Data" + "/Data_sim_" + str(quarter.strftime('%Y-%m-%d')) + ".dta")
    results = pd.DataFrame()
