# -*- coding: utf-8 -*-
"""
This File inputs the GMM Estimates and computes the latent demand under these estimates.

Furtheremore, it outputs the DataFrames that the Variance Decomposition uses

Choose your input values below.
"""
#%% Define Inputs

#Location of repository
path = ".../Python Replication Package" 

#Select Restricted Estimates
filename = 'KY19_baseline'
#(LASSO_IV, LASSO, ADAPTIVE_LASSO_OLSWEIGHTS, BackwardSelection_IV2SLS, BackwardSelection_GMM_NoIV, KY19_baseline, all, NLLS)

#Boolean:   If True, only q2 of every year is used as only these quarters
#           are used in the Variance Decomposition (saves runtime)
VarDecompOnly = True
#%% Libraries
import pandas as pd
import numpy as np

#%% Function
def compute_unpref(RegCoeffs,X,y):
    #Computes unexplained preferences, i.e. the epsilon
    return y * np.exp( -X[:,:-1] @ RegCoeffs - X[:,X.shape[1]-1] )
#%% Read in Data

#Load Holdings Data
Holdings = pd.read_stata(path + "/Data" "/Data1_clean_correct_bins.dta").drop('index',axis=1)
Holdings['rdate'] =  pd.to_datetime(Holdings["rdate"])
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
chars = pd.read_csv(path + "/Output" + "/Additional_Stock_Characteristics_Imputed_Winsorized.csv")
chars["rdate"] = pd.to_datetime(chars["rdate"])
chars.drop(['Unnamed: 0'],axis = 1,inplace=True)

#-------------------------- Coefficient Estimates -----------------------------

#Load GMM Estimates
if filename == 'NLLS':
    GMM_Estimates = (pd.read_csv(path + "/Output/Estimations" + "/" + 'NLLS_Estimates_CV_Restricted_SMAE' + ".csv")
                     .drop(['lam','best_index','lam_range','Unnamed: 0', 'index'],axis = 1)
                     )
else: 
    GMM_Estimates = pd.read_csv(path + "/Output/Estimations" + "/" + filename + "_Restricted" + ".csv")
    
GMM_Estimates["rdate"] = pd.to_datetime(GMM_Estimates["rdate"])

#---- Data Cleaning 
"""
Set non converged estimates to zero so the mean (Investor fixed effect)
is always chosen as the prediction. This will avoid numerical problems or 
errors in the Variance Decomposition.
"""
if filename != 'NLLS':
    #Extract column names of Estimates
    cols = list(GMM_Estimates.columns.drop(['rdate','bin','Error']))
    
    #If Estimator did not converge, set all estimates to 0
    GMM_Estimates.loc[np.isnan(GMM_Estimates['Error']) == True, cols] = 0
    
    #If convergence is poor, set all estimates to 0
    GMM_Estimates.loc[GMM_Estimates['Error']>0.9, cols] = 0
else:
    cols = list(GMM_Estimates.columns.drop(['rdate','bin']))

    
#If market equity coefficient too low, set all estimates to 0
GMM_Estimates.loc[GMM_Estimates['LNme'] < -20, cols] = 0

#If market equity coefficient greater than 1, set all estimates to 0
GMM_Estimates.loc[GMM_Estimates['LNme'] > 1, cols] = 0

#If any estimates take large positive or negative values, set all zero
GMM_Estimates[cols] = GMM_Estimates[cols].mask(GMM_Estimates[cols].abs() > 200, 0)

#Drop the Error of the GMM Estimates as it is no longer required
GMM_Estimates = GMM_Estimates[['rdate', 'bin']  + cols]

#---------------------------- Variable Names ----------------------------------
#Store the Estimate Column Names in a List 
GMM_Estimates_cols  = [item + "_beta" for item in cols]

#Store List of selected characteristics
Characteristics_Names = [item for item in cols if item != 'constant']

#cols no longer required
del cols
#%% Compute latent demand

#Extract unique dates
Quarters = StocksQ['date'].unique()
Quarters = Quarters[Quarters.quarter == 2] if VarDecompOnly else Quarters


#Loop over all Quarters
for quarter in Quarters:
    print(quarter)
    
    #Initialise DataFrame to store all Variables for the Variance Decomposition
    # Initialise DataFrame columns for the Variance Decomposition
    columns = (
        ['rdate', 'bin', 'mgrno', 'permno', 'aum', 'LNrweight', 'rweight', 'LNshrout', 'LNprc'] 
        + ['LNcfac']  # Return Correction Factor
        + ['unpref']  # Unexplained Preferences (latent demand)
        + Characteristics_Names
        + ['constant', 'cons']
        + GMM_Estimates_cols
    )

    ### --- Slice Datasets
    #Holdings Sliced
    Holdings_Q = Holdings[Holdings['rdate'] == quarter]

    ### --- Merge Stock Characteristics to Holdings Data to build X Matrix & y vector
    #Merge Baseline Stock Characteristics
    df_Q = (Holdings_Q
            .merge(StocksQ[['LNshrout','LNprc',"permno", "date", 'LNcfac'] 
                           + [col for col in Characteristics_Names if col in StocksQ.columns]], 
                   left_on=["rdate", "permno"], right_on=["date", "permno"])
            .drop('date_y',axis=1)
            .assign(constant=1)
            )
    
    #Merge Additional Stock Characteristics
    df_Q = (df_Q
            .merge(chars[['rdate','permno'] + [col for col in Characteristics_Names if col in chars.columns]], 
                   left_on = ['rdate','permno'], right_on = ['rdate','permno'], suffixes = ("",""))
            )
    
    #Merge Regression Coefficients
    df_Q = (df_Q
            .merge(GMM_Estimates, suffixes = ["", "_beta"], on=(['rdate','bin']), how ='left')
            )

    #Drop any remaining Missing Values to avoid errors
    df_Q = df_Q.dropna(subset=Characteristics_Names + ['constant','cons'] + ['rweight'])
    
    # Initialize a list to collect data
    all_var_decomp = []

    #---- Loop over bins and compute latent demand
    for i_bin in np.sort(df_Q['bin'].unique()):
        print("     Bin:" + str(i_bin))
        
        #Slice Dataset
        df_Q_bin = df_Q[df_Q['bin'] == i_bin]
        
        # Extract necessary variables for the Variance Decomposition
        df_vars_bin = df_Q_bin[list(set(columns) - {'unpref'})]

        #Extract only necessary variables for the Variance Decomposition
        """df_vars_bin = df_Q_bin[list(VarDecomp_Quarter.columns.drop('unpref'))] """
        
        #Extract beta coefficients as a column vector
        RegCoeffs = np.array(df_vars_bin[GMM_Estimates_cols].iloc[0]).T.reshape(-1)
        
        #Create X-Matrix
        X = np.array(df_vars_bin[Characteristics_Names + ['constant','cons']])
        
        #Create dependent variable
        y = np.array(df_vars_bin['rweight'])
        
        # Compute latent demand and append it to the DataFrame
        df_vars_bin = df_vars_bin.assign(unpref=compute_unpref(RegCoeffs, X, y))
        
        # Collect the DataFrame in a list
        all_var_decomp.append(df_vars_bin)
        
        """
        #Compute latent demand and merge it to the DataFrame
        df_vars_bin = df_vars_bin.assign(unpref = compute_unpref(RegCoeffs,X,y))
        
        #Append Results to final Dataframe
        VarDecomp_Quarter = pd.concat([VarDecomp_Quarter, df_vars_bin])
        """
        
    # Concatenate all collected DataFrames at the end
    VarDecomp_Quarter = pd.concat(all_var_decomp, ignore_index=True)
    
    #Reorder columns
    VarDecomp_Quarter = VarDecomp_Quarter.reindex(columns = columns)
    
    #Fix overflows by setting unpref to zero
    inf_count = np.isinf(VarDecomp_Quarter['unpref']).sum()
    print(f"The value 'infinity for unpref occurred {inf_count} times.")
    VarDecomp_Quarter['unpref'] = VarDecomp_Quarter['unpref'].replace(np.inf, 0)
        
    #Save Data for Variance Decomposition
    print("Saving Dataframe " + "VarDecomp_" + filename  + "_Restricted" + "_" + str(quarter)[0:10] + ".dta")
    VarDecomp_Quarter = VarDecomp_Quarter.astype({'bin': 'int32', 'mgrno': 'int32', 'constant':'int32'})
    VarDecomp_Quarter.to_stata(path +"/Output/Variance Decomposition Python/" + "VarDecomp_" + filename  + "_Restricted" + "_" + str(quarter)[0:10] + ".dta",
                               write_index=False)