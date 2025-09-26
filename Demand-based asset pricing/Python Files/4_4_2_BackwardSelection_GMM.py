# -*- coding: utf-8 -*-
"""
Conduct the Backward variable selection procedure using GMM. 
Default does not use the IV to greatly improve statistical power.

If IV estimation desired, set the macro No_IV to FALSE.
"""
#%% Libraries
import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot  as plt
import statsmodels.api as sm
import scipy.optimize
from scipy.stats.mstats import winsorize

import scipy.stats


from linearmodels.iv import IV2SLS
from statsmodels.api import OLS


import time

import copy

#%%
No_IV = True #Select whether to turn off IV estimation

#%% Functions
def Epsilon(RegCoeffs,X,y):
    """
    Inputs
    ----------
    RegCoeffs : numpy vector of Regression Coefficients to be estimated (numerical solvers iterate over RegCoeffs)
    X : numpy matrix of Explanatory Variables which includes the column LNme (but not IVme!). Additionally,
        the last column must be 'cons'. 'cons' has no RegCoeff attached to it.
    y : numpy vector of relative weights (including 0 rweights)

    Returns
    -------
    numpy vector error term. Each element is the prediction error
    for an investor i holding a stock n.
    """
    return y* np.exp( -X[:,:-1] @ RegCoeffs*step_size - X[:,X.shape[1]-1] ) - 1

def G_Matrix(Z,epsilon):
    """
    Inputs
    ----------
    Z : numpy matrix of instruments which is identical to X apart from two key aspects.
    Firstly, Z includes the column IVme (but not LNme), else-wise Z wouldn't be the matrix
    of instruments. Secondly, Z does NOT contain the column 'cons'. This is because 'cons'
    has no Regression Coefficient attached to it and therefore has no moment to satisfy.

    epsilon : prediction error (see function Epsilon() ).

    Returns
    -------
    The transpose of the numpy matrix where each (i,j)-element of Z is multiplied by epsilion(i) .
    In this final matrix called G, each i-th row and j-th column contains the product of the i-th
    variable of the j-th observation in the bin with the corresponding error. Summing up the columns
    will therefore be the sample moment approximating the Expectation.
    """
    return (Z * epsilon[:,np.newaxis]).T


def gmmObjective_Vector(RegCoeffs,X,Z,y):
    """
    Only returns g_avg from gmmObjective(). Doing root-finding on g_avg minimises
    the objective function as well. However, root-finding mostly does not work.
    """
    #Compute error vector
    epsilon = Epsilon(RegCoeffs,X,y)

    #Compute matrix of g_i. Each i-th column is g_i
    G = G_Matrix(Z,epsilon)

    #Estimate the expcation E[z_i *epsilon] where z_i is the i-th variable with the average
    g_avg = G.mean(axis=1)
    
    #Return g_avg and the Jacobian
    return g_avg , jacobian_gmmObjective_Vector(RegCoeffs,X,Z,y,epsilon),G

def jacobian_gmmObjective_Vector(RegCoeffs,X,Z,y,epsilon):
    #!!!! Make sure 'cons' is always the last element in selected characteristics
    jac = -1/len(Z)*Z.T @ ( X[:,:-1]* (epsilon[:, np.newaxis]+1) )
    return jac

def Newton(RegCoeffs,X,Z,y,W,damping = 0):
    # Function Value and Jacobian at Initial iteration
    g_avg , jac,_ = gmmObjective_Vector(RegCoeffs,X,Z,y)
    
    # Update Value
    beta_new = damping*RegCoeffs + (1-damping)* (-np.linalg.inv(jac)@g_avg + RegCoeffs)
    
    return beta_new

def gmm_tstat(RegCoeffs, X, Z, y):
    """
    Computes the absolute value of the t-statistics of a two-sided test H0 is always that
    the coefficient is zero. Only tests one coefficient at a time
    
    RegCoeffs are the estimates and RegCoeffsH0 are the values of the estimates under the Null.
    """  
    
    #create empty tstat object
    tstat = pd.Series(index=RegCoeffs.index)
    
    
    for var in RegCoeffs.index:
        RegCoeffs_H0 = copy.deepcopy(RegCoeffs)
        RegCoeffs_H0[var] = 0
        
        g_avg_H0, jac_H0, G_H0 = gmmObjective_Vector(RegCoeffs_H0,X,Z,y)
        
        #Compute Omega
        Omega_H0 = G_H0 @ G_H0.T / G_H0.shape[1]
        
        #Compute inverse Jacobian
        jac_H0_inv = np.linalg.inv(jac_H0)
        
        #Compute Covariance Matrix of theta_hat - theta_H0
        VarCov = jac_H0_inv.T @ Omega_H0 @ jac_H0_inv/G_H0.shape[1]
        
        #Compute all stats
        stat = (RegCoeffs-RegCoeffs_H0)/np.sqrt(np.diag(VarCov))
        
        #Take tstat of variable in Loop
        tstat[var] = stat[var]
    
    tstat = tstat.abs()
    
    return tstat    
    
def remove_lowest_significance(tstat,alpha=0.05):
    """
    Removes the variable with the lowest t-stat (excluding LNme). Computes
    values under alpha = 10% which corresponds to a |t-value|>1.64 rejecting the Null.
    """
    
    quantile = scipy.stats.norm.ppf(1-alpha/2)
    
    bool_VarRemoved = False
    
    #If no variable deemed significant, we need to stop the deletion manually
    if len(tstat.index) == 2:
        return tstat, False
    
    else:
        lowest_value = tstat.drop(['constant']).min()
        lowest_index = tstat.drop(['constant']).idxmin()
        
        if lowest_value < quantile: ##Corresponds to alpha=10%
            tstat.drop(lowest_index,inplace = True)
            bool_VarRemoved = True
            
        return tstat, bool_VarRemoved

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

def gmm_initial_guess(df_Q_bin,selected_characteristics,
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
        beta_lin_IV = IV2SLS(dependent=y_reg, exog=
                             X_reg.drop(columns=['LNme', 'IVme'],inplace=False), 
                             endog=X_reg['LNme'], instruments=X_reg['IVme']).fit().params
        
        #Maintain order of selected_characteristics 
        beta_lin_IV = beta_lin_IV.reindex([var for var in selected_characteristics if var != 'cons'])
        
        return beta_lin_IV  
    
    else:
        X_reg = df_Q_bin[[var for var in selected_characteristics  if var != 'cons']]
        
        beta = OLS(endog=y_reg, exog=X_reg).fit().params
        
        #Maintain Order of selected_characteristics
        beta = beta.reindex([var for var in selected_characteristics if var != 'cons'])
        
        return beta

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

df_select = pd.DataFrame(columns = ['rdate'] + ['bin'] +  Baseline_endog_Variables_Names + Characteristics_Names + ['constant'])
df_select.set_index(['rdate', 'bin'],inplace=True)

#%% GMM Estimation

#Extract unique dates
Quarters = Holdings['rdate'].unique()

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
    
    ### --- Loop over each individual bin
    for i_bin in np.sort(df_Q['bin'].unique()):
        print("     Bin:" + str(i_bin))
        
        ### --- Slice DataSet on Bin
        df_Q_bin = df_Q[df_Q['bin'] == i_bin]
        
        ### --- Generate Dataframe to Save BackwardSelection Results of Bin
        df_select_bin = pd.DataFrame(columns = ['rdate'] + ['bin'] +  Baseline_endog_Variables_Names + Characteristics_Names + ['constant'])
        df_select_bin.at[0,'rdate'] = quarter
        df_select_bin.at[0,'bin'] = i_bin
        df_select_bin.set_index(['rdate', 'bin'],inplace=True)
        df_select_bin[Baseline_endog_Variables_Names + Characteristics_Names + ['constant']] = 0
    
        
        ### --- Generate List of characteristics which are updated in each iteration
        #!!!! Make sure 'cons' is always the last element in selected characteristics
        selected_characteristics = Baseline_endog_Variables_Names + Characteristics_Names +  ['constant', 'cons']
        
        #Python cannot check up to a tolerance if these variables are linearly dependent with zerotradeAlt
        #selected_characteristics = list(pd.Series(index = selected_characteristics).drop(['zerotrade','zerotradeAlt12', 'zerotradeAlt1']).index)
        
        #Delete constant variables as they're linearly dependent with the constant
        boolean_var_deleted = True
        while boolean_var_deleted:
            selected_characteristics, boolean_var_deleted = delete_Constant_Characteristics(df_Q_bin, selected_characteristics)
       
        #--------------------- Standardize
        """
        df_Q_bin[[var for var in selected_characteristics if var != 'constant' and var != 'cons']] = df_Q_bin[[var for var in selected_characteristics if var != 'constant' and var != 'cons']].apply(
            lambda x: (x - x.mean()) / x.std()
        )
        """
        
        #Extract linearly dependent columns
        _, lin_dependent_columns = find_independent_columns(df_Q_bin, selected_characteristics)
        
        #Only keep lin independent columns. Maintain order of selected_characteristics (important)
        selected_characteristics = [var for var in selected_characteristics if var not in lin_dependent_columns]
        
        #Selected Instruments do not get 'cons' as a variable
        selected_instruments = ['IVme' if var == 'LNme' else var for var in selected_characteristics if var !='cons']
        
        ### --- Initialise While loop for backward selection
        bool_VarRemoved = True #As long as a variable is removed, the backward selection continues
        while bool_VarRemoved:
        
            #----------------------------------- Step 1 -----------------------------------
            #Get Initial Guesses for beta by using linear log-log regression ignoring the zeros
            beta_initial = gmm_initial_guess(df_Q_bin,selected_characteristics,
                                        selected_instruments)
            
            
            #----------------------------------- Step 2 -----------------------------------
            #Run GMM with self-coded Newton Method using the initial guesses
            X,Z,y,W =  get_GMM_Variables(df_Q_bin, selected_characteristics,
                                        selected_instruments)
            
            #-------------- NO IV Approach
            if No_IV: #Set Z equal to X 
                Z = copy.deepcopy(X[:,:-1])
            
            step_size = 1
                                    
            iteration = 0
            error = 1
            beta = copy.deepcopy(beta_initial)
            while iteration <100 and error > 1e-14:
                beta = Newton(beta,X,Z,y,W,damping = 0)
                g_avg, _ , _ = gmmObjective_Vector(beta,X,Z,y)
                iteration = iteration +1
                error = np.linalg.norm(g_avg)
            
            #If GMM Estimation fails, terminate the loop to avoid the code from stopping
            if np.isnan(error):
                df_select_bin[beta.index] = -np.inf
                df_select = pd.concat([df_select,df_select_bin])
                bool_VarRemoved = False
                continue
            #g_avg, jac, G = gmmObjective_Vector(beta,X,Z,y)
            
            #----------------------------------- Step 3 -----------------------------------
            #Delete Variable with the smalles tstat
            
            #Compute t-stat
            tstat = gmm_tstat(beta, X, Z, y)
            
            #Delete most insignificant variable
            tstat, bool_VarRemoved =  remove_lowest_significance(tstat)
            selected_characteristics = list(tstat.index) + ['cons']
            selected_instruments = ['IVme' if var == 'LNme' else var for var in selected_characteristics if var !='cons']
            
        #Create DataFrame IF GMM Estimation succeded
        if not np.isnan(error):
            df_select_bin[tstat.index] = 1
            df_select = pd.concat([df_select,df_select_bin])
    
    #Save Data (rdate & bin are an index, so save the index as well)
    df_select.to_csv(path + "/Output" + "/Variable Selection" + "/BackwardSelection_GMM_NoIV.csv")