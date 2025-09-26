"""
Computes the unrestricted & Restricted GMM Estimates for a VariableSelection Routine. 

"""
#%% Define Input Values
#Location of repository
path = ".../Python Replication Package" 

#Set variable_selection Routine (LASSO_IV, LASSO, ADAPTIVE_LASSO_OLSWEIGHTS, BackwardSelection_IV2SLS, BackwardSelection_GMM_NoIV, KY19_baseline, all)
variable_selection = 'KY19_baseline' 

#Boolean: If True, only q2 of every year is estimated as only these quarters
#are used in the Variance Decomposition (will significantly increase the runtime)
VarDecompOnly = True

#Set the Filename under which the results are stored (do not change this as the the files will be read in by subsequent Python Scripts)
filename = variable_selection 
#%% Libraries
import pandas as pd
import scipy
import numpy as np

import scipy.optimize
from linearmodels.iv import IV2SLS

import copy

#The following packages are used to compute the Jacobian of gmmObjective() with AutoDiff.
from autograd import jacobian
import autograd.numpy as anp  # Note the alias to avoid conflict with standard numpy
#%% Read in Data

#Load Holdings Data
Holdings = pd.read_stata(path + "/Data" + "/Data1_clean_correct_bins.dta")
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
#End of Month Day and not last Business Month Day required to match with Holdings
StocksQ["date"] = StocksQ["date"] - pd.offsets.MonthBegin()+pd.offsets.MonthEnd()

#Load Additional Stock Characteristics
chars = pd.read_csv(path + "/Output" + "/Additional_Stock_Characteristics_Imputed_Winsorized.csv")
chars["rdate"] = pd.to_datetime(chars["rdate"])
chars.drop(['Unnamed: 0'],axis = 1,inplace=True)

#Construct variable names of BASELINE Stock Characteristics
Chars_Baseline_Names = ['LNme','LNbe', 'profit', 'Gat', 'divA_be','beta']
Instruments_Baseline_Names  = ['IVme','LNbe', 'profit', 'Gat', 'divA_be','beta']

#Create list of additional Stock Characteristics
Chars_Additional_Names = list(chars.columns.drop(['rdate','permno']))

#%% Functions

#------------------------------------------------------------------------------ 
#                           GMM Input Variables 
#------------------------------------------------------------------------------

def select_vars(variable_selection):
    """
    Select the desired variables for the GMM Estimation for each bin in every quarter.
    """
    if variable_selection == 'KY19_baseline':
        df_variableSelection = pd.read_csv(path + "/Output" + "/Variable Selection" + "/" + 'LASSO' + ".csv")
        df_variableSelection['rdate'] = pd.to_datetime(df_variableSelection['rdate'])
        
        df_variableSelection = (df_variableSelection
                                .assign(**{col: 1 for col in Chars_Baseline_Names[1:] if col in df_variableSelection.columns})  # Always select KY19_baseline vars
                                .assign(**{col: 0 for col in df_variableSelection.columns 
                                           if col not in Chars_Baseline_Names[1:] + ['rdate', 'bin']})  # Set all other vars to zero
                                .assign(IVme_stage1 = 1) #Always include market equity
                                )
        
    elif variable_selection == 'all':
        df_variableSelection = pd.read_csv(path + "/Output" + "/Variable Selection" + "/" + 'LASSO' + ".csv")
        df_variableSelection['rdate'] = pd.to_datetime(df_variableSelection['rdate'])
        df_variableSelection = (df_variableSelection
                                .set_index(['rdate', 'bin']) #Don't Change these values
                                .map(lambda _: 1) #Set all other variables to 1
                                .reset_index() #'rdate' and 'bin' are columns again
                                )

    else:
        df_variableSelection = pd.read_csv(path + "/Output" + "/Variable Selection" + "/" + variable_selection + ".csv")
        df_variableSelection['rdate'] = pd.to_datetime(df_variableSelection['rdate'])
        
    return df_variableSelection
    

def get_Bin_Endog_VarList(VarSelect_Q_bin):
    """
    Outputs the List of ENDOGENOUS variable names for a bin i at quarter t
    deemed relevant by the Variable Selection Procedure
    
    Inputs
    ------
    VarSelect_Q_bin: Pandas Dataframe. 
        Boolean Dataframe where a variable that is meant to be selected is labelled as True.

    Returns
    -------
    selected_characteristics : List of ENDOGENOUS variable names
    """

    #Extract the ENDOGENOUS variable names deemed relevant by the variable selection
    selected_characteristics = VarSelect_Q_bin.columns[VarSelect_Q_bin.all()].tolist()

    #Rename market equity to its original name
    selected_characteristics = ["LNme" if x == "IVme_stage1" else x for x in selected_characteristics if x != 'constant']

    #Add the constant and 'cons' to the explanatory variables. 'cons' must be the last variable
    selected_characteristics = selected_characteristics + ['constant'] + ['cons']

    #Delete duplicate names in case there are any (prevents bugs)
    res = []
    [res.append(x) for x in selected_characteristics if x not in res]
    selected_characteristics = res

    #Shift Market Equity to the 1st position
    selected_characteristics.remove('LNme')
    selected_characteristics.insert(0,"LNme")

    return selected_characteristics

def get_Bin_Exog_VarList(selected_characteristics):
    """
    Outputs the list of INSTRUMENTS
    
    Inputs
    ------
    selected_characteristics : List
        Contains list of names of ENDOGENOUS variables for the GMM Estimation

    Returns
    -------
    selected_instruments : List
        List of names of the EXOGENOUS variables for the GMM Estimation.

    """
    #Select the instruments
    selected_instruments = ["IVme" if var == "LNme" else var for var in selected_characteristics]
    
    #cons has no Regression Coefficient attached to it. Therefore it is not part of the instruments
    selected_instruments.remove('cons')

    return selected_instruments

def delete_Constant_Characteristics(df_Q_bin, selected_characteristics):
    """
    Checks whether a selected ENDOGENOUS variable for a bin i at time t is a constant.
    Will prevent rank issues and crashes. 
    """
    #Compute initial length of selected_characteristics
    len_selected_chars_before = len(selected_characteristics)
    check_columns = copy.deepcopy(selected_characteristics)
    
    #constant and 'cons' are always kept as ENDOGENOUS variables
    check_columns.remove('constant')
    check_columns.remove('cons')
    
    # Calculate the standard deviation for each column
    std_devs = df_Q_bin[check_columns].std()

    # Identify columns with a standard deviation of zero
    zero_std_columns = std_devs[std_devs == 0].index.tolist()

    #Remove endogenous variables which are constant
    for item in zero_std_columns:
        selected_characteristics.remove(item)

    #Check if a variable has been deleted
    len_selected_chars_after = len(selected_characteristics)
    boolean_var_deleted = False
    if len_selected_chars_after - len_selected_chars_before !=0:
        boolean_var_deleted = True

    return selected_characteristics, boolean_var_deleted

def find_LinIndependent_Vars(df, selected_characteristics):
    """
    Identifies ENDOGENOUS variables that are linearly dependent and removes one of them.
    Will prevent rank issues and crashes. 
    """
    
    lin_dependent_columns = []
    lin_independent_columns = copy.deepcopy(selected_characteristics)
    
    #Add var for var and check rank of matrix. If rank doesn't increase, delete additional var
    for i in range(1,len(selected_characteristics[:-2])):
        #Compute Rank
        dif = i - np.linalg.matrix_rank(df[lin_independent_columns[:i]])
        
        #If Matrix not full rank, update variable list
        if dif > 0:
            lin_independent_columns = list(pd.Series(index=selected_characteristics).drop([selected_characteristics[i-1]]).index)
            lin_dependent_columns = lin_dependent_columns + [selected_characteristics[i-1]]
            
    return lin_independent_columns, lin_dependent_columns

def get_GMM_Variables(df_Q_bin, selected_characteristics,
                            selected_instruments):
    """
    Outputs the relevant numpy Objects required to compute the GMM Objective Functions
    
    Inputs
    ----------
    df_Q_bin : Pandas Dataframe
                Contains a quarter and bin slice of the overall data.
    selected_characteristics : List
                                ENDOGENOUS Variable Names - Output of get_Bin_Endog_VarList()
    selected_instruments : Lists 
                            Instrument Variable Names - Output of get_Bin_Exog_VarList()

    Returns
    -------
    X : numpy matrix of ENDOGENOUS variables. Market Equity is always the 1st column.
    Z : numpy matrix of Instruments.
    y : numpy vector of the relative portfolio weights
    W : Weighting Matrix initialised to the Identity Matrix
    """
    #Get Dataframes
    X,Z,y = df_Q_bin[selected_characteristics], df_Q_bin[selected_instruments], df_Q_bin["rweight"]

    #Convert Objects to numpy
    X = X.to_numpy()
    Z = Z.to_numpy()
    y = y.to_numpy()

    #Initialise Weighting Matrix
    W = np.eye(Z.shape[1])

    return X,Z,y,W

def standardize_Values(df_Q_bin, selected_characteristics):
    """
    Standardize every ENDOGENOUS variable of df_Q_bin (mean 0, std 1)
    
    Inputs
    -------
    df_Q_bin : Pandas Dataframe
                Contains a quarter and bin slice of the overall data.
    selected_characteristics : List
                                ENDOGENOUS Variable Names - Output of get_Bin_Endog_VarList()
    selected_instruments : Lists 
                            ENDOGENOUS Variable Names - Output of get_Bin_Endog_VarList()
                            
    Returns
    -------
    df: Pandas Dataframe
         Store the Standardised Variables

    standardized_df: Pandas Dataframe
                        Stores the original Mean and Std of the unstandardised variables
    """
    #Never standardise the constant or 'cons'
    standardize_chars = copy.deepcopy(selected_characteristics)
    standardize_chars.remove('cons')
    standardize_chars.remove('constant')
    
    #Avoids overwriting the input dataframe
    df = copy.deepcopy(df_Q_bin[selected_characteristics])

    # Initialize a DataFrame to store the mean and std for each variable (used to re-transform GMM estimates back to normal)
    standardized_df = pd.DataFrame(index=['mean', 'denominator'], columns=standardize_chars)

    #Standardise data and store original mean and standard deviation
    for column in standardize_chars:
        mean = df[column].mean()
        std =  df[column].std()
        denominator = std
        standardized_df.loc['mean', column] = mean
        standardized_df.loc['denominator', column] = denominator
        standardized_Values =  ((df[column] - mean) / (denominator)).to_numpy()
        df[column] = standardized_Values
    
    return df, standardized_df

#------------------------------------------------------------------------------ 
#                            GMM General Functions
#------------------------------------------------------------------------------
def Epsilon(RegCoeffs,X,y):
    """
    Inputs
    ----------
    RegCoeffs : Vector of Regression Coefficients
    X : Matrix of ENDOGENOUS Variables which includes the column LNme (but not IVme!).
        Last column must be 'cons' as 'cons' has no RegCoeff attached to it.
    y : Vector of relative weights

    Returns
    -------
    Vector of error terms (latent demand).
    """
    return y* anp.exp( -X[:,:-1] @ RegCoeffs*step_size - X[:,X.shape[1]-1] ) - 1

def G_Matrix(Z,epsilon):
    """
    Inputs
    ----------
    Z :         Matrix of INSTRUMENTS. Does NOT contain the column 'cons'.
    epsilon :   prediction error (see function Epsilon() ).
    
    Returns
    -------
    Matrix G whose i-th column is the vectors g_i
    """
    return (Z * epsilon[:,np.newaxis]).T

#------------------------------------------------------------------------------ 
#                         GMM ESTIMATION (Root-Finding)
#------------------------------------------------------------------------------

def gmm_initial_guess(df_Q_bin,selected_characteristics,
                            selected_instruments):
    """
    Compute initial guess for the root-finding routine
    
    
    Inputs
    ------
     df_Q_bin : Pandas Dataframe
                 Contains a quarter and bin slice of the overall data.
     selected_characteristics : List
                                 ENDOGENOUS Variable Names - Output of get_Bin_Endog_VarList()
     selected_instruments : List
                             Instrument Variable Names - Output of get_Bin_Exog_VarList()
    Returns
    -------
    Initial Guess of RegCoeffs used for the GMM ESTIMATION by root-finding
    """
    
    #Get Initial Guess as the 2SLS Estimator
    try:
        #Filter out the Zeros for linear Regression
        df_Q_bin = df_Q_bin[df_Q_bin.rweight>0]
    
        #Get X (endogenous variables) and Z (instruments) matrix
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
    
    except:
        return pd.Series(np.zeros(len(selected_instruments)), [var for var in selected_characteristics if var != 'cons'])


def gmmObjective_Root(RegCoeffs,X,Z,y):
    """
    Computes the objective function for GMM when using root-finding if
        i) The model is exactly identified
        ii) No Restriction on the coefficients RegCoeffs is imposed
    
    Returns
    -------
    g_avg: Objective Function
    jacobian_gmmObjective_Root(): Jacobian Matrix of Objective Function
    """
    #Compute error vector
    epsilon = Epsilon(RegCoeffs,X,y)

    #Compute matrix of g_i. Each i-th column is g_i
    G = G_Matrix(Z,epsilon)

    #Estimate the expcation E[z *epsilon] with the average
    g_avg = G.mean(axis=1)
    
    #Return g_avg and the Jacobian
    return g_avg , jacobian_gmmObjective_Root(RegCoeffs,X,Z,y,epsilon)

def jacobian_gmmObjective_Root(RegCoeffs,X,Z,y,epsilon):
    """
    Computes the Jacobian Matrix of gmmObjective_Root()
    
    Inputs
    ----------
    RegCoeffs : Vector of Regression Coefficients
    X :         Matrix of ENDOGENOUS Variables which includes the column LNme (but not IVme!).
                Last column must be 'cons' as 'cons' has no RegCoeff attached to it.
    Z :         Matrix of INSTRUMENTS. Does NOT contain the column 'cons'.
    y :         Vector of relative weights
    epsilon:    vector of latent demand given by Epsilon()
    """
    #Make sure 'cons' is always the last column in X
    jac = -1/len(Z)*Z.T @ ( X[:,:-1]* (epsilon[:, np.newaxis]+1) )
    return jac

def Newton(RegCoeffs,X,Z,y,damping = 0):
    """
    Computes one Newton Step of gmmObjective_Root()
    
    
    Inputs
    ----------
    RegCoeffs : Vector of Regression Coefficients being the Running Variable
    X :         Matrix of ENDOGENOUS Variables which includes the column LNme (but not IVme!).
                Last column must be 'cons' as 'cons' has no RegCoeff attached to it.
    Z :         Matrix of INSTRUMENTS. Does NOT contain the column 'cons'.
    y :         Vector of relative weights
    damping:    Damping factor to determine magnitude of Newton Step (Full Newton Step found to be best)

    """
    
    #Try Finding the Root of the Objective function with Newton's Algorithm
    try:
        # Function Value and Jacobian at Initial iteration
        g_avg , jac = gmmObjective_Root(RegCoeffs,X,Z,y)
        
        # Update Value
        beta_new = damping*RegCoeffs + (1-damping)* (-np.linalg.inv(jac)@g_avg + RegCoeffs)
        
        return beta_new
    
    #If an overflow error occured, return values that signal failure
    except:
        return RegCoeffs.abs()*(0) #(no errors in Variance Decomposition caused despite estimator not converging)
    
#------------------------------------------------------------------------------ 
#                         GMM ESTIMATION (Minimisation)
#------------------------------------------------------------------------------

def gmmObjective(RegCoeffs,X,Z,y,W):
    """
    Computes the objective function for GMM when using minimisation.
        Overidentification and Restrictions on Coefficients are possible
        
    Inputs
    ----------
    RegCoeffs : Vector of Regression Coefficients
    X :         Matrix of ENDOGENOUS Variables which includes the column LNme (but not IVme!).
                Last column must be 'cons' as 'cons' has no RegCoeff attached to it.
    Z :         Matrix of INSTRUMENTS. Does NOT contain the column 'cons'.
    y :         Vector of relative weights
    W:          Weighting Matrix
    """
    #Compute error vector.
    epsilon= Epsilon(RegCoeffs,X,y)

    #Compute matrix G
    G = G_Matrix(Z,epsilon)

    #Estimate the expcation E[z *epsilon] with the average
    g_avg = G.mean(axis=1)

    #Compute Objective Function
    objective = g_avg.T @ W @ g_avg
    #print(objective) #print function evaluation to see the progression in the minimiser.

    return objective

def jacobian_gmmObjective(RegCoeffs, X, Z, y, W):
    """
    Computes the Jacobian of gmmObjective() using Automatic Differentiation
    """
    gmmObjective_jacobian = jacobian(gmmObjective)
    jac = gmmObjective_jacobian(RegCoeffs, X, Z, y, W)
    return jac

def Weighting_Matrix(RegCoeffs,X,Z,y):
    """
    Computes the weighting matrix. 
    """
    #Compute error vector. 
    epsilon = Epsilon(RegCoeffs,X,y)

    #Compute matrix of g_i. Each i-th column is g_i
    G = G_Matrix(Z,epsilon)

    #Compute Weighting Matrix
    W = np.linalg.inv(G @ G.T / G.shape[1])

    return W

def getBounds(X_standardized, standardized_df):
    """
    Outputs the Bound on the Regression Coefficient for Market Equity that accounts for
    the linear transformation of the data.
    
    Inputs
    ------
    X_standardized: matrix of standardised ENDOGENOUS variables
    
    standardized_df: Pandas Dataframe containing original mean and std of columns of X_standardized
    
    """
    #Compute the relevant bound on the coefficient of Market Equity under consideration of the linear transformation of the data
    bounds = bounds=[(None,0.99/step_size*standardized_df['LNme'].iloc[1])] + [ (None,None) for i in range(X_standardized.shape[1]-2)]
    return bounds

def gmm(X_standardized, Z, y, W, x0, bounds, Jacobian, tolerance= 0.001):
    """
    Minimises gmmObjective()
    
    Inputs
    ------
    X_standarised : Matrix of standardized ENDOGENOUS Variables which includes the column LNme (but not IVme!)
                        Last column must be 'cons' as 'cons' has no RegCoeff attached to it.
    Z :             Matrix of INSTRUMENTS. Does NOT contain the column 'cons'.
    y :             Vector of relative weights
    W:              Weighting Matrix
    x0:             Initial Guess for RegCoeffs (running variable for the minimisation)
    bounds:         Domain Restriction on RegCoeffs, computed from get_bounds()
    Jacobian:       Jacobian of gmmObjective() given by jacobian_gmmObjective()
    tolerance:      Measuring accuracy of minimisation (lowered iteratively)
    
    Returns
    -------
    RegCoeffs that minimise gmmObjective()
    """

    #Set initial scaling down of tolerance
    scaling = 100
    
    #Set maximum number of iterations (Trial & Error --> Depends on the setting and how quickly it converges)
    max_trials = 20 #50
    
    #Minimise gmmObjective() in each iteration
    for trials in range(max_trials):
        
        #Compute the minimising arguments of gmmObjective()
        minimiser = scipy.optimize.minimize(gmmObjective, x0, jac=Jacobian,
                                            args=(X_standardized,Z,y,W), 
                                            method="L-BFGS-B",
                                            bounds = bounds,
                                            tol = tolerance,
                                            options={#'ftol':1e-12,
                                                     'gtol': tolerance
                                                     #'maxfun':1e6,
                                                     #'maxiter':1e6}
                                                     }
                                            )
        #Compute distance of minimiser to initial guess. If these don't update, weighting
        #matrices will instantly converge (printing this is helpful for convergence checks)
        dif_minimisers = np.linalg.norm(minimiser.x-x0)
        #print("dif_minimisers = " + str(dif_minimisers))

        #Update weighting matrix for next iteration
        W_new =  Weighting_Matrix(minimiser.x,X_standardized,Z,y)

        #Compare distance to current weighting matrix
        dif_WM = np.linalg.norm(W-W_new)
        print("dif WM = " + str(dif_WM))
        #If weighting matrices converge, minimisation is finished so break the loop
        if dif_WM < 1e-8: 
            break
        
        else:
            #update weighting matrix
            W = copy.deepcopy(W_new)
    
            #Update initial guess
            x0 = minimiser.x
    
            #Decrease tolerance for next iteration so that minimisers will get more accurate
            tolerance = tolerance/scaling
            scaling = scaling*1000 

    return minimiser, W, trials

def gmm_retransform_estimates(minimisers, standardized_df, selected_characteristics):
    """
    Re-transform the GMM Estimate to undo the linear transformation
    
    Inputs
    ------
    minimisers: RegCoeffs obtained from gmm().
    
    standardized_df: Pandas Dataframe containing original mean and std of columns of X_standardized
    
    selected_characteristics : List
                                 ENDOGENOUS Variable Names - Output of get_Bin_Endog_VarList()
    
    """
    #Select required means and denominators
    means = standardized_df.iloc[0]
    denominator = standardized_df.iloc[1]

    #Get Slope coefficients
    gamma_slope = (minimisers[:-1]*step_size).astype(np.float64)
    beta_slope = (gamma_slope/(denominator).to_numpy()).astype(np.float64)

    #Get Constant
    coefficients = (means/denominator).to_numpy().astype(np.float64)
    beta_constant = minimisers[-1]*step_size - coefficients@gamma_slope

    #Store re-transformed Regression Coefficients in a final dataframe
    RegCoeffs = beta_slope
    RegCoeffs = np.append(RegCoeffs,beta_constant)

    df_coeffs = pd.DataFrame(RegCoeffs.reshape(1,-1), columns = list(pd.Series(index = selected_characteristics).drop('cons').index))
    return df_coeffs


#------------------------------------------------------------------------------ 
#                               GMM Inference
#------------------------------------------------------------------------------

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
        
        g_avg_H0, jac_H0, G_H0 = gmmObjective_Root(RegCoeffs_H0,X,Z,y)
        
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
    tstat = tstat.abs().sort_values()
    
    return tstat

def remove_lowest_significance(tstat):
    """
    Removes the variable with the lowest t-stat (excluding LNme). Computes
    values under alpha = 10% which corresponds to a |t-value|>1.64 rejecting the Null.
    """
    bool_remove = False
    
    #If no variable deemed significant, we need to stop the deletion manually
    if len(tstat.index) == 2:
        return tstat, False
    
    else:
        lowest_value = tstat.drop(['LNme','constant']).min()
        lowest_index = tstat.drop(['LNme','constant']).idxmin()
        
        if lowest_value < 1.64:
            tstat.drop(lowest_index,inplace = True)
            bool_remove = True
            
        return tstat, bool_remove
#%% GMM Estimation (Unrestricted)

#Create dataframe to store all unrestricted GMM estimates
df_UnrestrictedEstimates = []
#Read in Variable Selection Results
df_variableSelection = select_vars(variable_selection)

#Create Boolean DataFrame of the variable selection
df_variableSelection = (
    df_variableSelection
    .set_index(["rdate", "bin"])  # Keep values of rdate & bin intact
    .ne(0)  # TRUE if a value != 0 and FALSE else
    .reset_index() #Make 'rdate' and 'bin' a column again
    .rename(columns = {'IVme_stage1':'LNme'}) #Rename Market Equity variable to original name if name was changed
    .assign(LNme = True) #Always include Market Equity in the variable selection
)

#Extract unique dates
Quarters = Holdings['rdate'].unique()
Quarters = Quarters[Quarters.quarter == 2] if VarDecompOnly else Quarters
Quarters = Quarters[Quarters.year<2023] #Since coverage of additional stock characteristics for 2023 is poor

#Loop over all quarters
for quarter in Quarters:
    print(quarter)

    ### --- Slice Datasets
    #Holdings Sliced
    Holdings_Q = Holdings[Holdings['rdate'] == quarter]
    
    #Variable Selection Sliced
    df_variableSelection_Q = df_variableSelection[df_variableSelection["rdate"]==quarter]  

    ### --- Merge Stock Characteristics to Holdings Data to build X & Z Matrix for GMM
    #Merge Baseline Stock Characteristics
    df_Q = (Holdings_Q
            .merge(StocksQ[["permno", "date"] + Chars_Baseline_Names], #IVme already in Holdings_Q 
                   left_on=["rdate", "permno"], right_on=["date", "permno"])
            .assign(constant = 1) #Assign Constant
            )
    #Merge Additional stock Characteristics
    df_Q = (df_Q
            .merge(chars, 
                   left_on=["rdate", "permno"], right_on=["rdate", "permno"],
                   how = "left", suffixes=["", "_new"])
            )

    ### --- Mild data cleaning
    #Drop any remaining Missing Values to avoid errors
    df_Q = df_Q.dropna(subset=Chars_Additional_Names + Chars_Baseline_Names + ['IVme'] + ['rweight'])
    
    
    ### --- Loop over each individual bin
    for i_bin in np.sort(df_Q['bin'].unique()):
        print("     Bin: " + str(i_bin))
        
        ### --- Slice Dataset on Bin
        #Slice General Dataset
        df_Q_bin = df_Q[df_Q['bin'] == i_bin]
        
        #Slice Variable Selection
        df_variableSelection_Q_bin = df_variableSelection_Q[df_variableSelection_Q['bin'] == i_bin].drop(['rdate', 'bin'],axis=1)
        
        ### --- Initialise Dataframe to store GMM estimates of the bin 
        df_UnrestrictedEstimates_bin = pd.DataFrame(
            columns = ['rdate'] + ['bin'] + ['Error'] + Chars_Baseline_Names + Chars_Additional_Names + ['constant']
            )
        df_UnrestrictedEstimates_bin.at[0,'rdate'] = quarter
        df_UnrestrictedEstimates_bin.at[0,'bin'] = i_bin
        df_UnrestrictedEstimates_bin.set_index(['rdate', 'bin'],inplace=True)
        df_UnrestrictedEstimates_bin[['Error'] + Chars_Baseline_Names + Chars_Additional_Names + ['constant']] = 0

        #Get Endogenous Variables for GMM Estimation
        selected_characteristics = get_Bin_Endog_VarList(df_variableSelection_Q_bin)
        
        ### --- Error Checks
        #Delete constant variables as they're linearly dependent with the constant
        selected_characteristics, _ = delete_Constant_Characteristics(df_Q_bin, selected_characteristics)
            
        #Remove linearly dependent columns
        lin_independent_columns, lin_dependent_columns =  find_LinIndependent_Vars(df_Q_bin, selected_characteristics)
        selected_characteristics = [var for var in selected_characteristics if var not in lin_dependent_columns]
        
        
        #Get Exogenous Variables for GMM Estimation
        selected_instruments = get_Bin_Exog_VarList(selected_characteristics) 
        
        #--------------------------- GMM Estimation----------------------------
        ### --- Step 1 (Initial Guess)
        
        #Get Initial Guess for beta by using linear log-log regression that ignores the zeros
        beta_initial = gmm_initial_guess(df_Q_bin,selected_characteristics,
                                    selected_instruments)
        
        ### --- Step 2 (Newton)
        #Get objects for the Estimation
        X,Z,y,_ =  get_GMM_Variables(df_Q_bin, selected_characteristics,
                                    selected_instruments)
        
        #Initialise values
        step_size = 1
        iteration = 0
        error = 1
        beta = copy.deepcopy(beta_initial)
        
        #Newton Method
        while iteration <100 and error > 1e-14:
            #Full Newton Step
            beta = Newton(beta,X,Z,y,damping = 0)
            #Evaluate Objective
            g_avg, _= gmmObjective_Root(beta,X,Z,y)
            #Compute the error
            error = np.linalg.norm(g_avg)
            #Update iteration
            iteration = iteration +1
            
        #Compute the error for the estimated beta, i.e. evaluate the objective function
        g_avg, _ = gmmObjective_Root(beta,X,Z,y)          
        
        #Store Beta in overall dataframe
        df_UnrestrictedEstimates_bin.loc[:,list(beta.index)] = np.array(beta)
        df_UnrestrictedEstimates_bin.loc[:,'Error'] = np.linalg.norm(g_avg)
        df_UnrestrictedEstimates_bin.reset_index(inplace=True) #So 'rdate' and 'bin' are stored as well
        
        #Store Results
        df_UnrestrictedEstimates.append(df_UnrestrictedEstimates_bin)


#Generate Final Dataframe
df_UnrestrictedEstimates = pd.concat(df_UnrestrictedEstimates)
#If a column only consists of zeros, i.e. this variable was never selected for the estimation,
#then delete the column from the DataFrame for clarity & sparsity
df_UnrestrictedEstimates = df_UnrestrictedEstimates.fillna(0)
df_UnrestrictedEstimates = df_UnrestrictedEstimates.loc[:, ~(df_UnrestrictedEstimates == 0).all(axis=0)]

#Save Estimates to a CSV File
df_UnrestrictedEstimates.to_csv(path + "/Output" + "/Estimations" + "/" + filename + "_unrestricted.csv", index=False)
#%% GMM Estimation (Restricted)

#Read in Unrestricted Estimates
df_UnrestrictedEstimates = pd.read_csv(path + "/Output" + "/Estimations" + "/" + filename + "_unrestricted.csv")
df_UnrestrictedEstimates['rdate'] = pd.to_datetime(df_UnrestrictedEstimates["rdate"])

#Initialise Restricted Estimates (extract estimates with beta_1>0.99)
df_RestrictedEstimates = copy.deepcopy(df_UnrestrictedEstimates)
df_RestrictedEstimates = df_RestrictedEstimates[df_RestrictedEstimates['LNme']>0.99]

#Create Boolean DataFrame of the variable selection (a variable was selected if it has a non-zero unrestricted estimate)
df_variableSelection = (
    copy.deepcopy(df_RestrictedEstimates) #Only consider Bins that require re-estimation
    .drop('Error', axis=1) #Numerical Estimation Error not required
    .set_index(["rdate", "bin"]) #Do not changed these values
    .ne(0)  # TRUE if a value != 0 and FALSE else
    .reset_index() #Make 'rdate' and 'bin' a column again
    .rename(columns={'LNme': 'IVme_stage1'}) #Renaming is a consistency requirement
)

#Extract unique dates
Quarters = df_RestrictedEstimates['rdate'].unique()

#Loop over all Quarters
for quarter in Quarters:
    print(quarter)

    ### --- Slice Datasets
    #Holdings Sliced
    Holdings_Q = Holdings[Holdings['rdate'] == quarter]
        
    #Variable Selection Sliced
    df_variableSelection_Q = df_variableSelection[df_variableSelection["rdate"]==quarter]
    
    #Slice DataFrame for Restricted Estimates to update them
    df_RestrictedEstimates_Q = df_RestrictedEstimates[df_RestrictedEstimates['rdate']==quarter]
        
    ### --- Merge Stock Characteristics to Holdings Data to build X & Z Matrix for GMM
    #Merge Baseline Stock Characteristics
    df_Q = (Holdings_Q
            .merge(StocksQ[["permno", "date"] +  Chars_Baseline_Names], 
                            left_on=["rdate", "permno"], right_on=["date", "permno"])
            .assign(constant=1) #Assign Constant
            )
    df_Q = df_Q.merge(chars, left_on=["rdate", "permno"], right_on=["rdate", "permno"], how = "left", suffixes=["", "_new"])

    ### --- Mild data cleaning
    #Drop any remaining Missing Values to avoid errors
    df_Q = df_Q.dropna(subset=Chars_Additional_Names + Chars_Baseline_Names + ['IVme'] + ['rweight'])

        
    ### --- Loop over each individual bin
    unique_bins = df_RestrictedEstimates_Q['bin'].unique()
    unique_bins = unique_bins[~np.isnan(unique_bins)]
    for i_bin in np.sort(unique_bins):
        print("     Bin: " + str(i_bin))
        
        #Slice overall DataSet
        df_Q_bin = df_Q[df_Q['bin'] == i_bin]
        
        #Slice Variable Selection
        df_variableSelection_Q_bin = df_variableSelection_Q[df_variableSelection_Q['bin'] == i_bin].drop(['rdate', 'bin'],axis=1)

        #Get Endogenous Variables for GMM Estimation
        selected_characteristics = get_Bin_Endog_VarList(df_variableSelection_Q_bin)
        
        ### --- Error Checks
        #Delete constant variables as they're linearly dependent with the constant
        selected_characteristics, boolean_var_deleted = delete_Constant_Characteristics(df_Q_bin, selected_characteristics)
            
        #Remove linearly dependent columns
        lin_independent_columns, lin_dependent_columns =  find_LinIndependent_Vars(df_Q_bin, selected_characteristics)
        selected_characteristics = [var for var in selected_characteristics if var not in lin_dependent_columns]
        
        
        #Get Exogenous Variables for GMM Estimation
        selected_instruments = get_Bin_Exog_VarList(selected_characteristics) 

        #---------------------- GMM Estimation with BFGS ----------------------

        #Get the untransformed GMM Variables: !!! LNme must be first in selected_characteristics !!!
        X,Z,y,W = get_GMM_Variables(df_Q_bin, selected_characteristics, selected_instruments)
        
        #Standardised X variables apart from constant and 'cons'
        df_Q_bin_standardized, standardized_df = standardize_Values(df_Q_bin, selected_characteristics)
        X_standardized = df_Q_bin_standardized.to_numpy()

        #Set the step_size for stability
        step_size = 0.01
        
        #Get the Bound for the restriction on beta_LNme
        bounds = getBounds(X_standardized, standardized_df)
        
        #Set the initial values for the minimiser
        x0 = np.ones(len(selected_characteristics)-1)*0/step_size
        
        #Do GMM_Estimation
        minimiser, W, trials = gmm(X_standardized, Z, y, W, x0, bounds, jacobian_gmmObjective)
        
        #Re-Transform Estimates
        beta = gmm_retransform_estimates(minimiser.x, standardized_df, selected_characteristics)
        
        #Overwrite previous Unrestricted estimates
        df_RestrictedEstimates.loc[(df_RestrictedEstimates['rdate'] == quarter) & (df_RestrictedEstimates['bin']==i_bin), beta.columns] = np.array(beta)
        df_RestrictedEstimates.loc[(df_RestrictedEstimates['rdate'] == quarter) & (df_RestrictedEstimates['bin']==i_bin), 'Error'] = np.array(minimiser.fun)


#Replace the Unrestricted Estimators with the Restricted ones
df1 = df_UnrestrictedEstimates[df_UnrestrictedEstimates['LNme']<0.99]

#Put all GMM Estimates into one DataFrame
df_merge = pd.concat([df1,df_RestrictedEstimates])
df_merge = df_merge.drop_duplicates(subset = ['rdate','bin'])

#Save all GMM Estimates
df_merge.to_csv(path + "/Output" + "/Estimations" + "/" + filename + "_Restricted.csv", index = False)