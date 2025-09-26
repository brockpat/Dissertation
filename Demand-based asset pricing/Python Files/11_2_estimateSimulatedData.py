# -*- coding: utf-8 -*-
"""
Estimate the Simulated Data with unrestricted GMM and compare the Price Elasticity of Demand
to the true Price Elasticity of Demand 
"""

#%% Libraries
import pandas as pd
import numpy as np
import scipy.optimize
import scipy.stats

from linearmodels.iv import IV2SLS

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import copy

from os import listdir
from os.path import isfile, join
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
    return y* np.exp( -X[:,:-1] @ RegCoeffs - X[:,X.shape[1]-1] ) - 1

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

def gmm_initial_guess(df_Q_bin,selected_characteristics):
    
    #Filter out the Zeros for linear Regression
    df_Q_bin = df_Q_bin[df_Q_bin.rweight>0]

    #Get X and Z matrix
    X_reg = df_Q_bin[[var for var in selected_characteristics if var != 'cons']]
    
    #Get vector of independent variable (subtract cons so that constant has the right level)
    y_reg = np.log(df_Q_bin.rweight)
    y_reg = y_reg - df_Q_bin['cons']
        
    #Do 2SLS Regression
    beta_lin_IV = IV2SLS(dependent=y_reg, exog=
                         X_reg.drop(columns=['LNme'],inplace=False), 
                         endog=X_reg['LNme'], instruments=X_reg['LNme']).fit().params
    
    #Maintain order of selected_characteristics 
    beta_lin_IV = beta_lin_IV.reindex([var for var in selected_characteristics if var != 'cons'])
    
    return beta_lin_IV

def ttest_df(df):
    return scipy.stats.ttest_ind(df['elasticity_true'], df['elasticity_estimated'])[1]

#%% Load Data
path = ".../Python Replication Package"
file_list = [f for f in listdir(path + "/Output" + "/Simulated_Data") if isfile(join(path + "/Output" + "/Simulated_Data", f))]

#Construct variable names of BASELINE Stock Characteristics, including IVme
Baseline_endog_Variables_Names = ['LNme','LNbe', 'profit', 'Gat', 'divA_be','beta']
Regressors_True_Names = [var + "_beta_true" for var in Baseline_endog_Variables_Names]
Regressors_Sim_Names = [var + "_beta_sim" for var in Baseline_endog_Variables_Names]

#%% Compute GMM Estimates (Unrestricted) for the simulated Data

#Generate Output DataFrame
df_simulatedEstimates = pd.DataFrame(columns = ['rdate'] + ['bin'] + ['Error'] + Baseline_endog_Variables_Names + ['constant'])
df_simulatedEstimates.set_index(['rdate', 'bin'],inplace=True)

for file_name in file_list:
    
    df_Q = pd.read_stata(path + "/Output" + "/Simulated_Data" + "/" + file_name).drop('index',axis=1)
    
    ### --- Loop over each individual bin
    unique_bins = df_Q['bin'].unique()
    unique_bins = unique_bins[~np.isnan(unique_bins)]
    
    for i_bin in np.sort(unique_bins):
        print("     Bin: " + str(i_bin))
        
        ### --- Slice DataSet on Bin
        df_Q_bin = df_Q[df_Q['bin'] == i_bin]
        
        ### --- Generate Dataframe to Save BackwardSelection Results of Bin
        df_simulatedEstimates_bin = pd.DataFrame(columns = ['rdate'] + ['bin'] + ['Error']
                                                 + Baseline_endog_Variables_Names + ['constant']
                                                 + Regressors_True_Names + ['constant_beta_true'])
        df_simulatedEstimates_bin.loc[0,Regressors_True_Names + ['constant_beta_true']] = df_Q_bin[Regressors_True_Names + ['constant_beta_true']].head(1).to_numpy()
        df_simulatedEstimates_bin.at[0,'rdate'] = df_Q_bin.rdate.unique()[0]
        df_simulatedEstimates_bin.at[0,'bin'] = i_bin
        df_simulatedEstimates_bin.set_index(['rdate', 'bin'],inplace=True)
        df_simulatedEstimates_bin[['Error'] + Baseline_endog_Variables_Names  + ['constant']] = 0
        
        ### --- Select variable names
        selected_characteristics = Baseline_endog_Variables_Names  +  ['constant', 'cons']
        selected_instruments = [ var for var in selected_characteristics if var !='cons']
        
        #----------------------------------- Step 1 -----------------------------------
        #Get Initial Guesses for beta by using linear log-log regression ignoring the zeros
        beta_initial = gmm_initial_guess(df_Q_bin,selected_characteristics)
        
        #----------------------------------- Step 2 -----------------------------------
        #Run GMM with self-coded Newton Method using the initial guesses
        X,Z,y,W =  get_GMM_Variables(df_Q_bin, selected_characteristics,
                                    selected_instruments)
        
        iteration = 0
        error = 1
        beta = copy.deepcopy(beta_initial)
        while iteration <100 and error > 1e-14:
            beta = Newton(beta,X,Z,y,W,damping = 0)
            g_avg, _ , _ = gmmObjective_Vector(beta,X,Z,y)
            iteration = iteration +1
            error = np.linalg.norm(g_avg)
            
        g_avg, _ , _ = gmmObjective_Vector(beta,X,Z,y)
        
        df_simulatedEstimates_bin.loc[:,list(beta.index)] = np.array(beta)
        df_simulatedEstimates_bin.loc[:,'Error'] = np.linalg.norm(g_avg)
        df_simulatedEstimates = pd.concat([df_simulatedEstimates,df_simulatedEstimates_bin])

df_simulatedEstimates = df_simulatedEstimates.reset_index()
#Rename Columns
for var in Baseline_endog_Variables_Names:
    df_simulatedEstimates = df_simulatedEstimates.rename(columns = {var: var + '_beta_sim'})

df_simulatedEstimates.to_csv(path + "/Output" + "/Simulated_Data" + "/Simulation_Comparison_Estimates.csv")

#%% Compute Estimated and True Demand Elasticities
df = pd.DataFrame()

df_simulatedEstimates = pd.read_csv(path + "/Output" + "/Simulated_Data" + "/Simulation_Comparison_Estimates.csv").drop('Unnamed: 0', axis=1)
df_simulatedEstimates['rdate'] =  pd.to_datetime(df_simulatedEstimates["rdate"]) #if reading in csv


file_list = [f for f in listdir(path + "/Output" + "/Simulated_Data") if isfile(join(path + "/Output" + "/Simulated_Data", f)) and f.endswith('.dta') and f.startswith('Data_sim')] 

for file in file_list:
    data = pd.read_stata(path + "/Output" + "/Simulated_Data" + "/" + file).drop(['index', 'zero_sim', 'rweight' , 'epsilon'],axis=1)
    
    data = data.merge(df_simulatedEstimates[['rdate','bin'] + 
                                            Regressors_Sim_Names], on = ['rdate','bin'], suffixes = ("",""))
    
    #Drop the zeros for the elasticity
    data = data[data['rweight_sim']>0]
    
    #Compute the level portfolio weight
    data['rweight_sim_sum'] = data.groupby('mgrno')['rweight_sim'].transform('sum')
    data['weight_sim'] =  data['rweight_sim']/(1+data['rweight_sim_sum'])
    
    data['elasticity_true'] = 1- data['LNme_beta_true']  * (1-data['weight_sim'])
    data['elasticity_estimated'] = 1- data['LNme_beta_sim']  * (1-data['weight_sim'])
    
    data = data[['rdate','mgrno','bin', 'permno', 'elasticity_true', 'elasticity_estimated', 'LNme_beta_true', 'LNme_beta_sim']]
    
    df = pd.concat([df,data], ignore_index=True)

df.to_stata(path + "/Output" + "/Simulated_Data" + "/Data_elasticities.dta")
#%% Plot Demand Elasticity Comparison

#Read in Data
df = pd.read_stata(path + "/Output" + "/Simulated_Data" + "/Data_elasticities.dta").drop('index',axis=1)
df = df[df['rdate'] != '2013-03-31']


df_grouped = df.groupby('rdate').agg(
    Elasticity_true_mean=('elasticity_true', 'mean'),
    Elasticity_true_lq=('elasticity_true', lambda x: x.quantile(0.25)),
    Elasticity_true_uq=('elasticity_true', lambda x: x.quantile(0.75)),
    
    Elasticity_estimated_mean=('elasticity_estimated', 'mean'),
    Elasticity_estimated_lq=('elasticity_estimated', lambda x: x.quantile(0.25)),
    Elasticity_estimated_uq=('elasticity_estimated', lambda x: x.quantile(0.75))
).reset_index()

# Set figure size
plt.figure(figsize=(10, 6))

#Plot Elasticities
plt.plot(df_grouped['rdate'], df_grouped['Elasticity_true_mean'], label='True', color='blue')
# Shade the area between the 10% quantile (y) and the 90% quantile (z)
plt.fill_between(df_grouped['rdate'], df_grouped['Elasticity_true_lq'], df_grouped['Elasticity_true_uq'], color='blue', alpha=0.3)
# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(df['rdate'].min(), df['rdate'].max())
plt.ylim(np.min(df_grouped.min().drop('rdate')), np.max(df_grouped.max().drop('rdate')))
# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)
# Add labels and title
plt.legend()

#Plot Elasticities
plt.plot(df_grouped['rdate'], df_grouped['Elasticity_estimated_mean'], label='Estimated', color='red')
# Shade the area between the 10% quantile (y) and the 90% quantile (z)
plt.fill_between(df_grouped['rdate'], df_grouped['Elasticity_estimated_lq'], df_grouped['Elasticity_estimated_uq'], color='red', alpha=0.3)
# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(df['rdate'].min(), df['rdate'].max())
plt.ylim(np.min(df_grouped.min().drop('rdate')), np.max(df_grouped.max().drop('rdate')))
# Rotate the x-axis labels for better readability
plt.xticks(rotation=90, fontsize = 14)
plt.yticks(fontsize=14)
# Add labels and title
#plt.title('Comparison Demand Elasticity (Shaded area IQR)')
plt.legend(fontsize = 14)

#Save Plot
plt.savefig(path + "/Output" + "/Plots" +"/TS_Elasticity_Bias.pdf", dpi=600, bbox_inches='tight')

# Display the plot
plt.show() #No need for t-tests between the different R^2 because this is the population!

#Compute p-values per quarter to check if the means are identical. t-test always rejected due to large sample size
ttest = df.groupby('rdate').apply(lambda x: ttest_df(x))

#%% Scatter Plot of coefficients

#Read in Data
df = pd.read_stata(path + "/Output" + "/Simulated_Data" + "/Data_elasticities.dta").drop('index',axis=1)
df = df[df['rdate'] != '2013-03-31']

df = df.drop_duplicates(subset = ['mgrno','rdate'])

plt.scatter(df['LNme_beta_true'], df['LNme_beta_sim'], c="blue")