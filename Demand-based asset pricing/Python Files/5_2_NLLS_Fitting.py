# -*- coding: utf-8 -*-
"""
Conduct the penalized non-linear least squares estimation

Caution: Runtime roughly 90h
"""

#%% Libraries
import pandas as pd
import numpy as np
import copy
from linearmodels.iv import IV2SLS
import scipy.optimize as opt

import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.model_selection import KFold

from scipy.optimize import minimize

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
Baseline_exog_Variables_Names  = ['IVme','LNbe', 'profit', 'Gat', 'divA_be','beta']

#%% Functions 

def get_GMM_Variables(df_Q_bin, selected_characteristics,
                            selected_instruments):
    #!!!! Make sure 'cons' is always the last element in selected characteristics
    """
    Inputs
    ----------
    df_Q_bin : Sliced Pandas Dataframe. Contains a quarter and bin slice of the overall data.
    selected_characteristics : Output of get_Bin_Characteristics_List()
    selected_instruments : Lists of Instruments based on selected_characteristics. Generated in main file.

    Returns
    -------
    X : numpy matrix of explanatory variables. Market Equity 1st column if selected by LASSO.
    Z : numpy matrix of explanatory variables. Instrument for Market Equity 1st column if selected by LASSO.
    y : numpy vector of the relative portfolio weights
    W : Weighting Matrix (redundant, see Weighting_Matrix() )
    """
    X,Z,y = df_Q_bin[selected_characteristics], df_Q_bin[selected_instruments], df_Q_bin["rweight"]

    #Convert matrices to numpy
    X = X.to_numpy()
    Z = Z.to_numpy()
    y = y.to_numpy()

    #Output Identity weighting matrix
    W = np.eye(Z.shape[1])

    return X,Z,y,W

def standardize_Values(df_Q_bin, selected_characteristics):
    """
    Squeeze every variable in selected_characteristics of df_Q_bin into [0,1]
    """
    standardize_chars = copy.deepcopy(selected_characteristics)
    standardize_chars.remove('cons')
    standardize_chars.remove('constant')
    #Avoids overwriting the input dataframe
    df = copy.deepcopy(df_Q_bin[selected_characteristics])

    # Initialize a DataFrame to store the min and max values for each column
    standardized_df = pd.DataFrame(index=['mean', 'denominator'], columns=standardize_chars)

    for column in standardize_chars:
        mean = df[column].mean()
        std =  df[column].std()
        denominator = std
        standardized_df.loc['mean', column] = mean
        standardized_df.loc['denominator', column] = denominator
        standardized_Values =  ((df[column] - mean) / (denominator)).to_numpy()
        df[column] = standardized_Values
    return df, standardized_df

def gmm_retransform_estimates(minimisers, standardized_df, selected_characteristics):
    #Re-Transform the Regression Coefficients to make them comparable to KY19

    #Take selected_characteristics (not instruments, as X (endogenous vars) are multiplied with RegCoeffs)
    #index = copy.deepcopy(selected_instruments)
    #Remove 'cons' as no RegCoeff attached to it and 'constant' as it is not transformed.
    #index.remove('constant')

    #Get Rename IVme to LNme since standardized_df uses LNme
    #index = ["LNme" if char == "IVme" else char for char in index]

    #Selected required means and denominators
    #standardized_df = standardized_df[index]
    means = standardized_df.iloc[0]
    denominator = standardized_df.iloc[1]

    #Get Slope coefficients
    gamma_slope = (minimisers[:-1]).astype(np.float64)
    beta_slope = (gamma_slope/(denominator).to_numpy()).astype(np.float64)

    #Get Constant
    coefficients = (means/denominator).to_numpy().astype(np.float64)
    beta_constant = minimisers[-1] - coefficients@gamma_slope

    #Store re-transformed Regression Coefficients in a final dataframe
    RegCoeffs = beta_slope
    RegCoeffs = np.append(RegCoeffs,beta_constant)

    df_coeffs = pd.DataFrame(RegCoeffs.reshape(1,-1), columns = list(pd.Series(index = selected_characteristics).drop('cons').index))
    return df_coeffs

def gmm_initial_guess(df_Q_bin,selected_characteristics,
                            selected_instruments):
    
    #Filter out the Zeros for linear Regression
    df_Q_bin = df_Q_bin[df_Q_bin.rweight>0]

    #Get X and Z matrix
    X_reg = df_Q_bin[[var for var in [selected_characteristics + ['IVme']][0] if var != 'cons']]
    
    #Get vector of independent variable (subtract cons so that constant has the right level)
    y_reg = np.log(df_Q_bin.rweight)
    y_reg = y_reg - df_Q_bin['cons']
        
    #Do 2SLS Regression
    beta_lin_IV = IV2SLS(dependent=y_reg, exog=
                         X_reg.drop(columns=['LNme', 'IVme'],inplace=False), 
                         endog=X_reg['LNme'], instruments=X_reg['IVme']).fit().params
    
    #Maintain order of selected_characteristics 
    beta_lin_IV = beta_lin_IV.reindex([var for var in selected_characteristics if var != 'cons'])
    
    return beta_lin_IV   

def gmm_initial_guess_outOfSample(X,y, selected_characteristics):
    
    #Create Dataset
    df = np.append(X, y.reshape(len(y),1), axis=1)
    
    #Filter out zeros
    df = df[df[:,-1]>0]
        
    #Get X and y matrix
    cons = df[:,-2]
    X_reg = df[:,:-2]
    y_reg = df[:,-1]
    
    #Get vector of independent variable (subtract cons so that constant has the right level)
    y_reg = np.log(y_reg)
    y_reg = y_reg - cons
    
    #Make Dataframes for column names
    X_reg = pd.DataFrame(X_reg, columns = [var for var in selected_characteristics if var not in ['cons']])
        
    #Do 2SLS Regression
    beta_lin = sm.OLS(y_reg, X_reg).fit().params
    
    #Maintain order of selected_characteristics 
    beta_lin = beta_lin.reindex([var for var in selected_characteristics if var not in ['cons']])
    
    return beta_lin   

def gmm_initial_guess_outOfSample_restricted(X,y, selected_characteristics):
    #Make sure 'LNme' is first in selected characteristics. Make sure 'con' is last in selected_characteristics
    
    #Create Dataset
    df = np.append(X, y.reshape(len(y),1), axis=1)
    
    #Filter out zeros
    df = df[df[:,-1]>0]
        
    #Get X and y matrix
    cons = df[:,-2]
    X_reg = df[:,:-2]
    y_reg = df[:,-1]
    
    #Get vector of independent variable (subtract cons so that constant has the right level)
    y_reg = np.log(y_reg)
    y_reg = y_reg - cons - X_reg[:,0]*bound_LNme #Subtract the Fix LNme
    
    #Make Dataframes for column names
    X_reg = pd.DataFrame(X_reg[:,1:], columns = [var for var in selected_characteristics if var not in ['LNme','cons']])
        
    #Do 2SLS Regression
    beta_lin = sm.OLS(y_reg, X_reg).fit().params
    
    #Maintain order of selected_characteristics 
    beta_lin = beta_lin.reindex([var for var in selected_characteristics if var not in ['LNme','cons']])

    return beta_lin 

def gmm_initial_guess_restricted(df_Q_bin,selected_characteristics,
                            selected_instruments):
    
    #Filter out the Zeros for linear Regression
    df_Q_bin = df_Q_bin[df_Q_bin.rweight>0]

    #Get X and Z matrix
    X_reg = df_Q_bin[[var for var in selected_characteristics if var not in ['LNme', 'cons']]]
    
    #Get vector of independent variable (subtract cons so that constant has the right level)
    y_reg = np.log(df_Q_bin.rweight)
    y_reg = y_reg - df_Q_bin['cons']
        
    #Do 2SLS Regression
    beta_lin = sm.OLS(y_reg, X_reg).fit().params
    
    #Maintain order of selected_characteristics 
    beta_lin = beta_lin.reindex([var for var in selected_characteristics if var not in ['LNme', 'cons']])
    
    return beta_lin   

def model(X, *beta):
    return np.exp(X[:,:X.shape[1]-1] @ beta + X[:,X.shape[1]-1])

def model_penalized(beta,X,y,lam):
    residual = y - np.exp(X[:,:X.shape[1]-1] @ beta + X[:,X.shape[1]-1])
    return np.sum(residual**2) + lam * np.sum(np.tanh(beta/2)*beta) 
    #https://arxiv.org/pdf/2303.09935 Smooth Mean Absolute Error approximates L1 Error (actual L1 error way too slow)
#np.mean(X,axis=0)[:-1] @ beta**2 #lam* np.sum(np.abs(beta)) #lam * np.mean(X,axis=0)[:-1] @ beta**2#np.mean(beta**2) # Model Loss + Penalty Loss
                   

def model_penalized_restricted(beta,X,y,lam):
    residual = y - np.exp( X[:,0]*bound_LNme + X[:,1:X.shape[1]-1] @ beta + X[:,X.shape[1]-1])
    return np.sum(residual**2 )  + lam * (np.tanh(bound_LNme/2)*bound_LNme + np.sum(np.tanh(beta/2)*beta))   
#Model Loss  + Penalty Loss (Loss from restricted LNme plus remaining Loss)
                   

def model_restricted(X, *beta):
    return np.exp(X[:,0]*0.99 + X[:,1:X.shape[1]-1] @ beta + X[:,X.shape[1]-1])

def loss_function(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

#%% Non-Linear Least Squares (Unrestricted)

Quarters = Holdings['rdate'].unique()
df_UnrestrictedEstimates = pd.DataFrame(columns = ['rdate'] + ['bin'] + Baseline_endog_Variables_Names + ['constant'])

for quarter in Quarters:
    print(quarter)

    ### --- Slice Datasets
    #Holdings Sliced
    Holdings_Q = Holdings[Holdings['rdate'] == quarter]
    
    #Baseline Stock Characteristics Sliced
    StocksQ_Q = StocksQ[StocksQ['date'] == quarter]
    
        
    ### --- Merge Stock Characteristics to Holdings Data to build X & Z Matrix for GMM
    #Merge Baseline Stock Characteristics
    df_Q = Holdings_Q.merge(StocksQ_Q[["permno", "date", "LNme", 'LNbe', 'profit', 'Gat', 'divA_be','beta']], 
                            left_on=["rdate", "permno"], right_on=["date", "permno"])

    ### --- Mild data cleaning
    #Drop any remaining Missing Values to avoid errors
    df_Q = df_Q.dropna(subset=Baseline_endog_Variables_Names + ['IVme'] + ['rweight'])
    
    #Assign Constant
    df_Q = df_Q.assign(constant=1) 
    
    ### --- Loop over each individual bin
    unique_bins = df_Q['bin'].unique()
    unique_bins = unique_bins[~np.isnan(unique_bins)]
    for i_bin in np.sort(unique_bins):
        print("     Bin: " + str(i_bin))
        
        ### --- Slice DataSet on Bin
        df_Q_bin = df_Q[df_Q['bin'] == i_bin]
        
        ### --- Generate Dataframe to Save BackwardSelection Results of Bin
        df_UnrestrictedEstimates_bin = pd.DataFrame(columns = ['rdate'] + ['bin'] + ['Error'] + Baseline_endog_Variables_Names + ['constant'])
        df_UnrestrictedEstimates_bin.at[0,'rdate'] = quarter
        df_UnrestrictedEstimates_bin.at[0,'bin'] = i_bin
        #df_UnrestrictedEstimates_bin.set_index(['rdate', 'bin'],inplace=True)
    
        
        ### --- Generate List of selected characteristics which are updated in each iteration
        #!!!! Make sure 'cons' is always the last element in selected characteristics
        #!!!! Make sure LNme is the first element in the list!
        selected_characteristics = Baseline_endog_Variables_Names  +  ['constant', 'cons']
               
        #Selected Instruments do not get 'cons' as a variable
        selected_instruments = ['IVme' if var == 'LNme' else var for var in selected_characteristics if var !='cons']
        
        #----------------------------------- Step 1 -----------------------------------
        #Get Initial Guesses for beta by using linear log-log regression ignoring the zeros
        beta_initial = gmm_initial_guess(df_Q_bin,selected_characteristics,
                                    selected_instruments)
        
        #----------------------------------- Step 2 -----------------------------------
        #Run GMM with self-coded Newton Method using the initial guesses
        X,Z,y,W =  get_GMM_Variables(df_Q_bin, selected_characteristics,
                                    selected_instruments)
                
        try:
            optimal_params, covariance = opt.curve_fit(model, X, y, beta_initial)
        except:
            optimal_params = pd.Series(0,list(beta_initial.index))
        
        df_UnrestrictedEstimates_bin[list(beta_initial.index)] = optimal_params
        df_UnrestrictedEstimates_bin.reset_index(inplace=True)
        
        
        df_UnrestrictedEstimates = pd.concat([df_UnrestrictedEstimates,df_UnrestrictedEstimates_bin])
        


    #df_UnrestrictedEstimates = df_UnrestrictedEstimates.reset_index()
    df_UnrestrictedEstimates.to_csv(path + "/Output" + "/Estimations" + "/NLLS_Estimates_Unrestricted.csv")
    
    df_UnrestrictedEstimates.drop(['index', 'Error'],axis = 1, inplace=True)

#%% NLLS Restricted Estimates

NLLS_Estimates = pd.read_csv(path + "/Output" + "/Estimations" + "/NLLS_Estimates_Unrestricted.csv")
NLLS_Estimates['rdate'] =  pd.to_datetime(NLLS_Estimates["rdate"]) #if reading in csv

df_RestrictedEstimates = copy.deepcopy(NLLS_Estimates)
df_RestrictedEstimates = df_RestrictedEstimates[df_RestrictedEstimates['LNme']>0.99]

df_RestrictedEstimates = df_RestrictedEstimates.reset_index()

#Extract unique dates
Quarters = df_RestrictedEstimates['rdate'].unique()

for quarter in Quarters:
    print(quarter)

    ### --- Slice Datasets
    #Holdings Sliced
    Holdings_Q = Holdings[Holdings['rdate'] == quarter]
    
    #Baseline Stock Characteristics Sliced
    StocksQ_Q = StocksQ[StocksQ['date'] == quarter]
    
    #Additional Stock Characteristics Sliced
    #chars_Q = chars[chars['rdate'] == quarter]
    
    df_RestrictedEstimates_Q = df_RestrictedEstimates[df_RestrictedEstimates['rdate']==quarter]
        
    ### --- Merge Stock Characteristics to Holdings Data to build X & Z Matrix for GMM
    #Merge Baseline Stock Characteristics
    df_Q = Holdings_Q.merge(StocksQ_Q[["permno", "date", "LNme", 'LNbe', 'profit', 'Gat', 'divA_be','beta']], 
                            left_on=["rdate", "permno"], right_on=["date", "permno"])
    #df_Q = df_Q.merge(chars_Q, left_on=["rdate", "permno"], right_on=["rdate", "permno"], how = "left", suffixes=["", "_new"])

    ### --- Mild data cleaning
    #Drop any remaining Missing Values to avoid errors
    #df_Q = df_Q.dropna(subset=Characteristics_Names + Baseline_endog_Variables_Names + ['IVme'] + ['rweight'])
    df_Q = df_Q.dropna(subset=Baseline_endog_Variables_Names + ['IVme'] + ['rweight'])

    #Assign Constant
    df_Q = df_Q.assign(constant=1)
    
    ### --- Loop over each individual bin
    unique_bins = df_RestrictedEstimates_Q['bin'].unique()
    #unique_bins = unique_bins[~np.isnan(unique_bins)]
    for i_bin in np.sort(unique_bins):
        print("     Bin: " + str(i_bin))
        
        df_Q_bin = df_Q[df_Q['bin'] == i_bin]
        
        ### --- Generate List of selected characteristics which are updated in each iteration
        #!!!! Make sure 'cons' is always the last element in selected characteristics
        selected_characteristics = Baseline_endog_Variables_Names  +  ['constant', 'cons']
               
        #Selected Instruments do not get 'cons' as a variable
        selected_instruments = ['IVme' if var == 'LNme' else var for var in selected_characteristics if var !='cons']
        
        #----------------------------------- Step 1 -----------------------------------
        #Get Initial Guesses for beta by using linear log-log regression ignoring the zeros
        beta_initial = gmm_initial_guess_restricted(df_Q_bin,selected_characteristics,
                                    selected_instruments)
        
        #----------------------------------- Step 2 -----------------------------------
        #Run GMM with self-coded Newton Method using the initial guesses
        X,Z,y,W =  get_GMM_Variables(df_Q_bin, selected_characteristics,
                                    selected_instruments)
        
        try:
            optimal_params, covariance = opt.curve_fit(model_restricted, X, y, beta_initial)
        except:
            optimal_params = pd.Series(0,list(beta_initial.index))
        
        
        #Overwrite previous estimates
        df_RestrictedEstimates.loc[(df_RestrictedEstimates['rdate'] == quarter) & (df_RestrictedEstimates['bin']==i_bin), list(beta_initial.index)] = np.array(optimal_params)
        df_RestrictedEstimates.loc[(df_RestrictedEstimates['rdate'] == quarter) & (df_RestrictedEstimates['bin']==i_bin), 'LNme'] = 0.99

#Replace the Unrestricted Estimators with the Restricted ones
df1 = NLLS_Estimates[NLLS_Estimates['LNme']<0.99]
df_merge = pd.concat([df1,df_RestrictedEstimates])

#Save the Dataframe
df_merge.to_csv(path + "/Output" + "/Estimations" + "/NLLS_Estimates_Restricted.csv")
    
#%% Compute Fitted Values and R^2 of NLLS Estimation

NLLS_Estimates = pd.read_csv(path + "/Output" + "/Estimations" + "/NLLS_Estimates_Unrestricted.csv")
NLLS_Estimates['rdate'] =  pd.to_datetime(NLLS_Estimates["rdate"]) #if reading in csv

df = pd.DataFrame()

#------- Loop over all Quarters and compute the fitted value rweight_hat
Quarters = Holdings['rdate'].unique()
Quarters = Quarters[Quarters.year<2023]
for quarter in Quarters:
    print(quarter)

    ### --- Slice Datasets
    #Holdings Sliced
    Holdings_Q = Holdings[Holdings['rdate'] == quarter][['rdate','bin','mgrno','permno','aum','rweight','cons']]

    #Baseline Stock Characteristics Sliced
    StocksQ_Q = StocksQ[StocksQ['date'] == quarter]

    ### --- Merge Stock Characteristics to Holdings Data to build X & Z Matrix for NLLS
    #Merge Baseline Stock Characteristics
    df_Q = Holdings_Q.merge(StocksQ_Q[["permno", "date", "LNme", 'LNbe', 'profit', 'Gat', 'divA_be','beta']], 
                            left_on=["rdate", "permno"], right_on=["date", "permno"])

    #Drop any remaining Missing Values to avoid errors
    df_Q = df_Q.dropna(subset=Baseline_endog_Variables_Names + ['rweight'])

    df_Q = df_Q.merge(NLLS_Estimates.add_suffix("_beta"), 
                      left_on = ['rdate','bin'], 
                      right_on = ['rdate_beta','bin_beta'],
                      how = 'left', 
                      suffixes=('', ''))
    df_Q.drop(['rdate_beta','bin_beta', 'Error_beta', 'date'],axis = 1,inplace=True)

    #Assing the constant to the dataframe
    df_Q = df_Q.assign(constant=1)
    
    #Compute fitted Values
    Baseline_endog_Variables_Names = ['LNme','LNbe', 'profit', 'Gat', 'divA_be','beta']

    Baseline_Regressors = [item + "_beta" for item in Baseline_endog_Variables_Names + ['constant']]
    
    X = df_Q[Baseline_endog_Variables_Names + ['constant']].values
    beta = df_Q[Baseline_Regressors].values.T
    
    #Get predicted values. i-th predicted value is dot product of i-th row of X with i-th column of beta + cons[i]
    df_Q['rweight_hat'] = np.exp(np.array([np.dot(X[i], beta[:, i]) for i in range(X.shape[0])]) + df_Q['cons'].values)
    
    #Save Results
    df = pd.concat([df,df_Q])
        
#------- Loop over all Bins and Compute the R^2 with the fitted values from the previous
df_level_error = pd.DataFrame()
for quarter in Quarters:
    print(quarter)
    
    df_Q = df[df['rdate'] == quarter]
    
    for i_bin in np.sort(df_Q['bin'].unique()):
        df_Q_bin = df_Q[df_Q['bin'] == i_bin]
        
        y = df_Q_bin['rweight']
        y_hat = df_Q_bin['rweight_hat']
        y_mean = np.mean(df_Q_bin['rweight'])
        
        R2 = 1- np.linalg.norm(y - y_hat)**2 /  np.linalg.norm(y - np.mean(y_mean))**2
        
        df_level_error_bin = pd.DataFrame(columns = ['rdate','bin','R_squared_NLLS'])
        df_level_error_bin.at[0,'rdate'] = quarter
        df_level_error_bin['rdate'] = pd.to_datetime(df_level_error_bin['rdate'])
        df_level_error_bin.at[0,'bin'] = i_bin
        df_level_error_bin.at[0,'R_squared_NLLS'] = R2
        
        df_level_error = pd.concat([df_level_error,df_level_error_bin])

#Save Data
df_level_error.to_csv(path + "/Output" + "/NLLS_R2LevelFits.csv")

#%% Plot R^2

"""
GMM and OLS are for inference. Perhaps we know the actual slopes (i.e. the elasticities),
but this is useless if we have the right elasticities at a completely wrong prediction points
when trying to clear the asset market
"""

df = pd.read_csv(path + "/Output" + "/NLLS_R2LevelFits.csv")
df['rdate'] =  pd.to_datetime(df["rdate"]) #if reading in csv

#Cut last two Quarters off so that the x-ticks nicely align in the plot
df = df[df['rdate'] < '2022-09-30']
df = df[df['rdate']!= '2013-03-31'] #Cut out this Quarter because Data is missing

#Filter outliers
df = df[df['R_squared_NLLS'] > df['R_squared_NLLS'].quantile(0.02)]

#Construct Household, Big & Small Investors
df_HH = df[df['bin'] == 0]

#Compute mean and Quantiles of Small and Big Investors
df_small = df[df['bin'] < 191]
df_small_grouped = df_small.groupby('rdate').agg(
    R2_mean=('R_squared_NLLS', 'mean'),
    R2_quantile_25_NLLS=('R_squared_NLLS', lambda x: x.quantile(0.25)),
    R2_quantile_75_NLLS=('R_squared_NLLS', lambda x: x.quantile(0.75)),
).reset_index()

df_big = df[df['bin'] > 190]
df_big_grouped = df_big.groupby('rdate').agg(
    R2_mean=('R_squared_NLLS', 'mean'),
    R2_quantile_25_NLLS=('R_squared_NLLS', lambda x: x.quantile(0.25)),
    R2_quantile_75_NLLS=('R_squared_NLLS', lambda x: x.quantile(0.75)),
).reset_index()

#Group all R^2 < 0 to 0 because otherwise Histogramm cannot be reasonably visualised
df_small.loc[df_small['R_squared_NLLS'] < 0, 'R_squared_NLLS'] = 0
df_big.loc[df_big['R_squared_NLLS'] < 0, 'R_squared_NLLS'] = 0

#------ Plot mean Time Series R^2 of Small Investors
plt.figure(figsize=(10,6))
plt.plot(df_small_grouped['rdate'], df_small_grouped['R2_mean'], label='Small Investors', color='red')
plt.fill_between(df_small_grouped['rdate'], df_small_grouped['R2_quantile_25_NLLS'], df_small_grouped['R2_quantile_75_NLLS'], color='red', alpha=0.3)

# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(df['rdate'].min(), df['rdate'].max())

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)
# Add labels and title
plt.ylabel('$R^2$')
plt.title('Time Series mean $R^2$ NLLS Estimates Small Investors (shade = IQR).')
plt.savefig(path + "/Output" + "/Plots" +"/R2_mean_NLLS_SmallInvestors.png", dpi=700)#, bbox_inches='tight')  # Save with 300 DPI
plt.show() 

#------ Plot mean Time Series R^2 of Big Investors
plt.figure(figsize=(10,6))
plt.plot(df_big_grouped['rdate'], df_big_grouped['R2_mean'], label='Big Investors', color='blue')
plt.fill_between(df_big_grouped['rdate'], df_big_grouped['R2_quantile_25_NLLS'], df_big_grouped['R2_quantile_75_NLLS'], color='blue', alpha=0.3)

# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(df['rdate'].min(), df['rdate'].max())

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)
# Add labels and title
plt.ylabel('$R^2$')
plt.title('Time Series mean $R^2$ NLLS Estimates Big Investors (shade = IQR).')
plt.savefig(path + "/Output" + "/Plots" +"/R2_mean_NLLS_BigInvestors.png", dpi=700)
plt.show() 

#------ Plot TS Households R^2
plt.figure(figsize=(10,6))
plt.plot(df_HH['rdate'], df_HH['R_squared_NLLS'], label='Households', color='green')

# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(df['rdate'].min(), df['rdate'].max())

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)
# Add labels and title
plt.ylabel('$R^2$')
plt.title('Time Series $R^2$ NLLS Estimates Household.')
plt.savefig(path + "/Output" + "/Plots" +"/R2_NLLS_Households.png", dpi=700)
plt.show() 


#------ Plot Histogram of R^2 for Big Investors
counts, bin_edges = np.histogram(df_big['R_squared_NLLS'], bins=20)
# Calculate relative frequencies by dividing the counts by the total number of data points
relative_frequencies = counts / len(df_big['R_squared_NLLS'])

# Plot the histogram with relative frequencies
plt.bar(bin_edges[:-1], relative_frequencies, width=np.diff(bin_edges), edgecolor='black', align='edge',
        alpha=0.7)

# Customize labels and title
ax.set_xlabel('Value', fontsize=13)
ax.set_ylabel('Relative Frequency', fontsize=13)
ax.set_title('Stylized Histogram with Background', fontsize=15, fontweight='bold')


plt.title('Histogram Relative Frequencies $R^2$ NLLS Estimates Big Investors.')
plt.xlabel('$R^2$')
plt.ylabel('Relative Frequency')
plt.savefig(path + "/Output" + "/Plots" +"/Histogramm_NLLS_R2_BigInvestors.png", dpi=700)#, bbox_inches='tight')  # Save with 300 DPI
plt.show()

#------ Plot Histogram Small Investors
counts, bin_edges = np.histogram(df_small['R_squared_NLLS'], bins=20)
# Calculate relative frequencies by dividing the counts by the total number of data points
relative_frequencies = counts / len(df_small['R_squared_NLLS'])

# Plot the histogram with relative frequencies
plt.bar(bin_edges[:-1], relative_frequencies, width=np.diff(bin_edges), edgecolor='black', align='edge',
        alpha=0.7)

# Customize labels and title
ax.set_xlabel('Value', fontsize=13)
ax.set_ylabel('Relative Frequency', fontsize=13)
ax.set_title('Stylized Histogram with Background', fontsize=15, fontweight='bold')

plt.title('Histogram Relative Frequencies $R^2$ NLLS Estimates Small Investors.')
plt.xlabel('$R^2$')
plt.ylabel('Relative Frequency')
plt.savefig(path + "/Output" + "/Plots" +"/Histogramm_NLLS_R2_SmallInvestors.png", dpi=700)#, bbox_inches='tight')  # Save with 300 DPI
plt.show()


#%% Compute NLLS with Cross Validation (Unrestricted)

Quarters = Holdings['rdate'].unique()
df_CV_Unrestricted = pd.DataFrame(columns = ['rdate'] + ['bin'] + Baseline_endog_Variables_Names + ['constant'] + ['lam'] + ['best_index'])

for quarter in Quarters:
    print(quarter)

    ### --- Slice Datasets
    #Holdings Sliced
    Holdings_Q = Holdings[Holdings['rdate'] == quarter]
    
    #Baseline Stock Characteristics Sliced
    StocksQ_Q = StocksQ[StocksQ['date'] == quarter]
    
        
    ### --- Merge Stock Characteristics to Holdings Data to build X & Z Matrix for GMM
    #Merge Baseline Stock Characteristics
    df_Q = Holdings_Q.merge(StocksQ_Q[["permno", "date", "LNme", 'LNbe', 'profit', 'Gat', 'divA_be','beta']], 
                            left_on=["rdate", "permno"], right_on=["date", "permno"])

    ### --- Mild data cleaning
    #Drop any remaining Missing Values to avoid errors
    df_Q = df_Q.dropna(subset=Baseline_endog_Variables_Names + ['IVme'] + ['rweight'])
    
    #Assign Constant
    df_Q = df_Q.assign(constant=1) 
    
    ### --- Loop over each individual bin
    unique_bins = df_Q['bin'].unique()
    unique_bins = unique_bins[~np.isnan(unique_bins)]
    for i_bin in np.sort(unique_bins):
        print("     Bin: " + str(i_bin))
        
        ### --- Slice DataSet on Bin
        df_Q_bin = df_Q[df_Q['bin'] == i_bin]
        
        ### --- Generate Dataframe to Save BackwardSelection Results of Bin
        df_CV_Unrestricted_bin = pd.DataFrame(columns = ['rdate'] + ['bin'] + Baseline_endog_Variables_Names + ['constant'])
        df_CV_Unrestricted_bin.at[0,'rdate'] = quarter
        df_CV_Unrestricted_bin.at[0,'bin'] = i_bin
        #df_CV_Unrestricted_bin.set_index(['rdate', 'bin'],inplace=True)
    
        
        ### --- Generate List of selected characteristics which are updated in each iteration
        #!!!! Make sure 'cons' is always the last element in selected characteristics
        #!!!! Make sure LNme is the first element in the list!
        selected_characteristics = Baseline_endog_Variables_Names  +  ['constant', 'cons']
               
        #Selected Instruments do not get 'cons' as a variable
        selected_instruments = ['IVme' if var == 'LNme' else var for var in selected_characteristics if var !='cons']
        
        #Shuffle DataFrame to get random test and train indexes
        df_Q_bin = df_Q_bin.sample(frac=1, random_state=48645316)
        
        #Standardize Variables
        df_Q_bin[selected_characteristics], standardized_df = standardize_Values(df_Q_bin, selected_characteristics)
                
        #----------------------------------- Step 2 -----------------------------------
        #Run GMM with self-coded Newton Method using the initial guesses
        X,Z,y,W =  get_GMM_Variables(df_Q_bin, selected_characteristics,
                                    selected_instruments)
        
        #Define Grid of penalty parameters
        lambdas = np.logspace(-1.2, 7, 100) #Max of Lambda also depends on size of Sum of Squares of residuals
        
        #Get Splits of the data (indexes are deterministic, but data has been shuffled before)
        kf = KFold(n_splits=5, shuffle=False)
        all_train_indexes = []
        all_test_indexes = []
        
        for train_index, test_index in kf.split(X):
            all_train_indexes.append(train_index)
            all_test_indexes.append(test_index)
        
        #Define vector of cross validation errors which determines the optimal lambda
        cv_errors = np.zeros(len(lambdas))
        
        #Iterate over Training and Test Data
        for train_index, test_index in zip(all_train_indexes, all_test_indexes):          
            X_train = X[train_index]
            y_train = y[train_index]
            
            X_test = X[test_index]
            y_test = y[test_index]
            
            #Get Initial Parameter Guess
            beta_initial = gmm_initial_guess_outOfSample(X_train,y_train, selected_characteristics)
            #Compute the MSE for each Lambda on the test data
            for i, lam in enumerate(lambdas):
                #Find optimal parameters
                minimiser = minimize(model_penalized, beta_initial, args=(X_train, y_train, lam))
                beta = minimiser.x  # Optimal parameters for this lambda
    
                # Evaluate test loss
                y_pred = np.exp(X_test[:,:X_test.shape[1]-1] @ beta + X_test[:,X_test.shape[1]-1])
                test_error = np.mean((y_test - y_pred) ** 2)
                
                # Accumulate test error for each lambda. Each element in cv_error is a sum of n_split components.
                cv_errors[i] += test_error 
        
        best_index = np.argmin(cv_errors)
        lam_star = lambdas[best_index]
        
        beta_star = minimize(model_penalized, beta_initial, args=(X, y, lam_star)).x
        #Re-Transform Estimates
        beta_star = gmm_retransform_estimates(beta_star, standardized_df, selected_characteristics)
        #beta_star = pd.Series(beta_star, list(beta_initial.index))


        df_CV_Unrestricted_bin.at[0,'lam'] = lam_star
        df_CV_Unrestricted_bin.at[0,'best_index'] = best_index
        df_CV_Unrestricted_bin.at[0,'lam_range'] = 'np.logspace(-1.2, 7, 100)'
        
        df_CV_Unrestricted_bin[list(beta_initial.index)] = np.array(beta_star)   

        
        df_CV_Unrestricted = pd.concat([df_CV_Unrestricted,df_CV_Unrestricted_bin])
        #If lambda takes the maximum value, because 'cons' has no beta attached to it,
        #The Cross-validation sets all parameters to zero so that the mean weight is
        #the best prediction. We saw many times that the mean far outperformed the
        #GMM estimates. So this is actually nice that we see this here too.
        #We can even contrast the bins with R^2 << 0 with the estimates from here
        #Interestingly, the big Investors are basically best estimated under lambda = 0
        #This shows that their behaviour can be somewhat well estimated, which is also clear
        #Since we know that they behave very boringly and buy the market. However, the assets
        #where they take idiosyncratic risk we predict very well.
        #Also interesting to see how often the restriction binds. It could be much less which
        #is again evident when using regularisation.
    
    df_CV_Unrestricted.to_csv(path + "/Output" + "/Estimations" + "/NLLS_Estimates_CV_unestricted_SMAE.csv")
    
#%% Compute NLLS with Cross Validation (Restricted)

df_UnrestrictedEstimates = pd.read_csv(path + "/Output" + "/Estimations" + "/NLLS_Estimates_Out_of_Sample_Unrestricted_SMAE.csv")
df_UnrestrictedEstimates['rdate'] =  pd.to_datetime(df_UnrestrictedEstimates["rdate"]) #if reading in csv

df_UnrestrictedEstimates.drop('Unnamed: 0',axis = 1, inplace = True)

df_RestrictedEstimates = copy.deepcopy(df_UnrestrictedEstimates)
df_RestrictedEstimates = df_RestrictedEstimates[df_RestrictedEstimates['LNme']>0.99]


Quarters = df_RestrictedEstimates['rdate'].unique()
Quarters = Quarters[Quarters < '2023-01-01']

for quarter in Quarters:
    print(quarter)

    ### --- Slice Datasets
    #Holdings Sliced
    Holdings_Q = Holdings[Holdings['rdate'] == quarter]
    
    #Baseline Stock Characteristics Sliced
    StocksQ_Q = StocksQ[StocksQ['date'] == quarter]
    
        
    ### --- Merge Stock Characteristics to Holdings Data to build X & Z Matrix for GMM
    #Merge Baseline Stock Characteristics
    df_Q = Holdings_Q.merge(StocksQ_Q[["permno", "date", "LNme", 'LNbe', 'profit', 'Gat', 'divA_be','beta']], 
                            left_on=["rdate", "permno"], right_on=["date", "permno"])

    ### --- Mild data cleaning
    #Drop any remaining Missing Values to avoid errors
    df_Q = df_Q.dropna(subset=Baseline_endog_Variables_Names + ['IVme'] + ['rweight'])
    
    #Assign Constant
    df_Q = df_Q.assign(constant=1) 
    
    df_RestrictedEstimates_Q = df_RestrictedEstimates[df_RestrictedEstimates['rdate']==quarter]
    
    ### --- Loop over each individual bin
    unique_bins = df_RestrictedEstimates_Q['bin'].unique()
    unique_bins = unique_bins[~np.isnan(unique_bins)]
    for i_bin in np.sort(unique_bins):
        print("     Bin: " + str(i_bin))
        
        ### --- Slice DataSet on Bin
        df_Q_bin = df_Q[df_Q['bin'] == i_bin]
        
        ### --- Generate List of selected characteristics which are updated in each iteration
        #!!!! Make sure 'cons' is always the last element in selected characteristics
        #!!!! Make sure LNme is the first element in the list!
        selected_characteristics = Baseline_endog_Variables_Names  +  ['constant', 'cons']
               
        #Selected Instruments do not get 'cons' as a variable
        selected_instruments = ['IVme' if var == 'LNme' else var for var in selected_characteristics if var !='cons']
        
        #Shuffle DataFrame to get random test and train indexes
        df_Q_bin = df_Q_bin.sample(frac=1, random_state=48645316)
        
        #Standardize Variables
        df_Q_bin[selected_characteristics], standardized_df = standardize_Values(df_Q_bin, selected_characteristics)
        
        bound_LNme = 0.99*standardized_df['LNme'].iloc[1]
                
        #----------------------------------- Step 2 -----------------------------------
        #Run GMM with self-coded Newton Method using the initial guesses
        X,Z,y,W =  get_GMM_Variables(df_Q_bin, selected_characteristics,
                                    selected_instruments)
        
        #Define Grid of penalty parameters
        """
        Adjust this grid depending on the input data. The grids must be the same
        """
        lambdas = np.logspace(-1.2, 7, 100) #Max of Lambda also depends on size of Sum of Squares of residuals
        
        #Get Splits of the data (indexes are deterministic, but data has been shuffled before)
        kf = KFold(n_splits=5, shuffle=False)
        all_train_indexes = []
        all_test_indexes = []
        
        for train_index, test_index in kf.split(X):
            all_train_indexes.append(train_index)
            all_test_indexes.append(test_index)
        
        #Define vector of cross validation errors which determines the optimal lambda
        cv_errors = np.zeros(len(lambdas))
        
        #Iterate over Training and Test Data
        for train_index, test_index in zip(all_train_indexes, all_test_indexes):          
            X_train = X[train_index]
            y_train = y[train_index]
            
            X_test = X[test_index]
            y_test = y[test_index]
                        
            #Get Initial Parameter Guess.
            beta_initial = gmm_initial_guess_outOfSample_restricted(X_train,y_train, selected_characteristics)
            #Compute the MSE for each Lambda on the test data
            for i, lam in enumerate(lambdas):
                #Find optimal parameters
                minimiser = minimize(model_penalized_restricted, beta_initial, args=(X_train, y_train, lam))
                beta = minimiser.x  # Optimal parameters for this lambda
    
                # Evaluate test loss
                y_pred = np.exp(X_test[:,0] * bound_LNme + X_test[:,1:X_test.shape[1]-1] @ beta + X_test[:,X_test.shape[1]-1])
                test_error = np.mean((y_test - y_pred) ** 2)
                
                # Accumulate test error for each lambda. Each element in cv_error is a sum of n_split components.
                cv_errors[i] += test_error 
        
        best_index = np.argmin(cv_errors)
        lam_star = lambdas[best_index]
        
        beta_star = minimize(model_penalized_restricted, beta_initial, args=(X, y, lam_star)).x
        
        #Add Coefficient for Market Equity
        beta_star = np.insert(beta_star,0,bound_LNme)
        
        #Re-Transform Estimates
        beta_star = gmm_retransform_estimates(beta_star, standardized_df, selected_characteristics)

        #Overwrite previous estimates
        df_RestrictedEstimates.loc[(df_RestrictedEstimates['rdate'] == quarter) & (df_RestrictedEstimates['bin']==i_bin), beta_star.columns] = np.array(beta_star)
        df_RestrictedEstimates.loc[(df_RestrictedEstimates['rdate'] == quarter) & (df_RestrictedEstimates['bin']==i_bin), 'lam'] = lam_star
        df_RestrictedEstimates.loc[(df_RestrictedEstimates['rdate'] == quarter) & (df_RestrictedEstimates['bin']==i_bin), 'best_index'] = best_index


#Replace the Unrestricted Estimators with the Restricted ones
df1 = df_UnrestrictedEstimates[df_UnrestrictedEstimates['LNme']<0.99]
df1.reset_index(inplace=True)
df_merge = pd.concat([df1,df_RestrictedEstimates])


df_merge.to_csv(path + "/Output" + "/Estimations" + "/NLLS_Estimates_CV_Restricted_SMAE.csv")