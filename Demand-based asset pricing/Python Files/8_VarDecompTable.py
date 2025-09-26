"""
This file computes the Output Table of the Variance Decomposition. It further
exports a .tex table

Its input is generated in the 7_Decomp.py Files
"""
#%% Define Inputs

#Location of repository
path = ".../Python Replication Package" 

#Select Estimates with which the Stock Return Variance is Decomposed
filename = 'KY19_baseline'
#(LASSO_IV, LASSO, ADAPTIVE_LASSO_OLSWEIGHTS, BackwardSelection_IV2SLS, BackwardSelection_GMM_NoIV, KY19_baseline, all,NLLS)

#If the iterative KY19_baseline decomposition was used
iterative = False #Only implemented for Baseline Characteristics

table_caption = 'Variance Decomposition ' + filename
table_label = 'Table:VarDecomp' + filename

#%% Libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import copy
#%% Functions

#-------------------------- Computing the Covariance --------------------------
def getObjects(df, filename, iterative):
    """
    Selects the List of Counterfactual Returns
    """
    if filename in ['LASSO_IV', 'LASSO', 'ADAPTIVE_LASSO_OLSWEIGHTS', 
                    'BackwardSelection_IV2SLS', 'BackwardSelection_GMM_NoIV', 
                    'KY19_baseline', 'all', 'NLLS']:
        list_Counterfactual_Returns = ['LNret1', 'LNret2', 'LNretdA', 'LNret3', 'LNret4', 'LNret5', 'LNret6']
    
    if iterative:
        list_Counterfactual_Returns = ['LNret1', 'LNret2_LNbe', 'LNret2_profit', 
                                       'LNret2_Gat', 'LNret2_divA_be', 'LNret2_beta', 
                                       'LNretdA', 'LNret3', 'LNret4', 'LNret5', 'LNret6']
    
    list_DateDummys = ['rdate_' + str(year) + '-06-30 00:00:00' for year in range(df.rdate.dt.year.min()+1,df.rdate.dt.year.max()+1)]
    
    parameters = pd.Series(np.inf, list_Counterfactual_Returns)
    std_errors = pd.Series(np.inf, list_Counterfactual_Returns)
    n_obs = pd.Series(np.inf, list_Counterfactual_Returns)
            
    return list_Counterfactual_Returns, list_DateDummys, parameters, std_errors, n_obs

def prepareData(df, list_DateDummys):
    
    #To avoid overwriting the original dataframe
    df = copy.deepcopy(df)
    
    #Include a constant (the first Time Period has no explicit dummy, so we need a constant)
    df = df.assign(constant=1)
    
    #Get Equal and Value Weights that can be used in the Regression
    df = df.merge(StocksQ[['date','permno','_meA']], left_on = ['rdate','permno'], 
                  right_on = ['date','permno'], how = 'left',suffixes = ("","")).drop('date',axis = 1)
    
    df['eweightA'] = df.groupby('rdate')['_meA'].transform('count')
    df['eweightA'] = 1 / df['eweightA']

    df['vweightA'] = df.groupby('rdate')['_meA'].transform('count')
    df['vweightA'] = df['_meA'] / df['vweightA']
    
    df.drop('_meA',axis=1)
    
    #Generate Time Dummies (Leave out First Time Dummy as elsewise this causes rank issues with the constant)
    df = pd.get_dummies(df, columns=['rdate'], drop_first=True)
    df[list_DateDummys] = df[list_DateDummys].astype(int)
    

    return df

#-------------------------------- Output Results ------------------------------

#Get Names of the Table
def tableNames(filename, iterative):
    if filename in ['LASSO_IV', 'LASSO', 'ADAPTIVE_LASSO_OLSWEIGHTS', 
                    'BackwardSelection_IV2SLS', 'BackwardSelection_GMM_NoIV', 
                    'KY19_baseline', 'all', 'NLLS']:
        SupplyNames = ['Shares Outstanding', 'Stock Characteristics', 'Dividend Yield']
        DemandNames = ['AUM', 'Coefficients', 'Latent demand: extensive margin', 'Latent demand: intensive margin']
        
    if iterative:
        SupplyNames = ['Shares Outstanding', 'Book Equity', 'Profit', 'GAT', 'Dividends to BE', 'Market Beta', 'Dividend Yield']
        
    return SupplyNames, DemandNames

#Write Table's Preamble
def tablePreamble(caption, label):
    List = [r"\begin{table}[]",  r"\caption{" + caption + "}", r"\label{" + label + "}", 
            r"\centering", r"\begin{tabular}{lc}",  r"\hline \hline", r" & \% of Variance \\ \hline"]
    
    return List

#Write Results of Supply Factors in Table
def tableSupply(SupplyNames, parameters, std_errors):
    List = [r"\textbf{Supply}: & \\"]
    for item in SupplyNames:
        List.append(r"\hspace*{2mm} " + item + r" & " + str(parameters[item]) + r"\\")
        List.append(r" & \small (" + str(std_errors[item]) + r") \\")

    return List  

#Write Results of Demand Factors in Table
def tableDemand(DemandNames, parameters, std_errors):
    List = [r"\textbf{Demand}: & \\"]
    for item in DemandNames:
        List.append(r"\hspace*{2mm} " + item + r" & " + str(parameters[item]) + r"\\")
        List.append(r" & \small (" + str(std_errors[item]) + r") \\")

    return List      

#Assemble Table
def tableFinal(filename, iterative, caption, label, parameters, std_errors, n_obs):
    
    SupplyNames, DemandNames = tableNames(filename, iterative)
    
    #Adjust index of parameters and std_errors
    parameters.index = SupplyNames + DemandNames
    std_errors.index = SupplyNames + DemandNames
    
    preamble    = tablePreamble(caption, label)
    supply      = tableSupply(SupplyNames, parameters, std_errors)
    demand      = tableDemand(DemandNames, parameters, std_errors)
    obs         = [r"Observations & " + str(int(n_obs.min())) + r" \\ \hline\hline"]
    end         = [r"\end{tabular}", r"\end{table}"]
    
    finalTable = preamble + supply + demand + obs + end
    
    return finalTable
    
#%% Variance Decomposition Estimates

#---- Prepare the Data

#Load Data
if not iterative:
    df = pd.read_stata(path + "/Output/Variance Decomposition Python/" + filename + "_IntermediaryReturns.dta")

else:
    df = pd.read_stata(path + "/Output/Variance Decomposition Python/" + filename + "_IntermediaryReturns_Iterative.dta")


StocksQ = pd.read_stata(path + "/Data" + "/StocksQ.dta")
StocksQ["date"] = StocksQ["date"] - pd.offsets.MonthBegin()+pd.offsets.MonthEnd()

# Get Relevant Objects
(list_Counterfactual_Returns, list_DateDummys, parameters, 
 std_errors, n_obs) = getObjects(df, filename, iterative)

#Get Time Fixed Effects & Weights
df = prepareData(df,list_DateDummys)

#---- Run the Regression to compute the Covariances
# Loop through each variable, run regression with fixed effects, and store results
for cf_return in list_Counterfactual_Returns:
    
    # Get independent variable & dependent variables
    df_reg = df[ ['eweightA'] + [cf_return] + ['LNretA'] + ['constant'] + list_DateDummys]
    
    # Drop Missing Values (some cf_returns have more missing values than others)
    df_reg = df_reg.dropna()
    
    # Extract independent and dependent variables
    X = df_reg.drop([cf_return, 'eweightA'], axis = 1)
    y = df_reg[cf_return]

    
    # Run the weighted regression with robust standard errors
    model = sm.WLS(y, X, weights=df_reg['eweightA']).fit(cov_type='HC3')
    
    parameters[cf_return] = model.params['LNretA']
    std_errors[cf_return] = model.bse['LNretA']
    n_obs[cf_return] = len(X)
    
#Reformat numbers
parameters = np.round(parameters*100, decimals=2)
std_errors = np.round(std_errors*100, decimals=2)

#---- Write LaTeX Table

# Export the table as a .tex file
if iterative:
    with open(path + "/Output" + "/Tables" + "/" + filename + "_VarianceDecomposition_Iterative.tex", "w") as file:
        for item in tableFinal(filename, iterative, table_caption, table_label, parameters, std_errors, n_obs):
            file.write(item + "\n")
else:
    with open(path + "/Output" + "/Tables" + "/" + filename + "_VarianceDecomposition.tex", "w") as file:
        for item in tableFinal(filename, iterative, table_caption, table_label, parameters, std_errors, n_obs):
            file.write(item + "\n")