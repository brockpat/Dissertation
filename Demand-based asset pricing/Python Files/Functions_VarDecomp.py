# -*- coding: utf-8 -*-
"""
Stores the Functions used to compute the Variance Decomposition
"""
import pandas as pd
import numpy as np
import scipy
import copy

#%% Functions

#Pull out year & Quarter of DataFrame
def extractTime(df,year,quarter):
    df_year = df[df['rdate'].dt.year == year] 
    df_yearQuarter = df_year[df_year['rdate'].dt.quarter == quarter]
    
    return df_yearQuarter


def getObjects(VarDecomp_df, year, quarter):
    """
    This function returns all necessary objects to compute the Variance
    Decomposition.
    
    Inputs
    ------
    VarDecomp_df :  DataFrame containing the Portfolio Holdings, Characteristics and Estimation Results10). 
    year :          Year to extract
    quarter :       Quarter to extract

    Returns
    -------
    PFRweight :  IxN matrix containing log(w_i(n)/w_i(0))
    Epsilon :    IxN matrix containing log(epsilon_i(n))
    cons :       IxN matrix containing the investor fix effect (only non-zero if investor has a positive holding for an asset)
    RegCoeffs :  IxK matrix containing the estimates
    aum :        Ix1 vector containing the assets under management of each investor.
    p :          Nx1 vector containing the log price of each asset
    s :          Nx1 vector containing the log supply (shares outstanding) of each asset
    x :          KxN matrix containing the characteristics of each asset
    LNcfac:      Nx1 vector containing Return Correction Factor
    
    Mat & Vec Extensions are simply the numpy conversions of the pandas DataFrames
    """
    #----------------------------------------------------------------------#
    #0) Extract the relevant year & quarter of the data
    #----------------------------------------------------------------------#
    VarDecomp_df = extractTime(VarDecomp_df, year, quarter)
    VarDecomp_df = VarDecomp_df.drop_duplicates(subset=["rdate","mgrno", "permno"])
    
    #----------------------------------------------------------------------#
    #1.1) Create empty Holdings template to create \mathcal{P} stock universe.
    #----------------------------------------------------------------------#
    
    #Unique Stocks in Holdings (length = N)
    #If not sorted, then iterating over n might not refer to same manager or asset for different objects
    uniquePermno = pd.DataFrame(np.sort(VarDecomp_df['permno'].unique()),columns = ['permno'])
    
    #Unique Managers in Holdings (length = I)
    #If not sorted, then iterating over i might not refer to same manager or asset for different objects
    uniqueMgrno = pd.DataFrame(np.sort(VarDecomp_df['mgrno'].unique()), columns = ['mgrno'])
    
    #Merge Cross Products to create empty Holdings template for  \mathcal{P} stock universe
    Holdings  = pd.merge(uniqueMgrno, uniquePermno, how='cross')
    
    #Include rdate
    Holdings['rdate'] = VarDecomp_df['rdate'][0]
    
    #Add log relative portfolio weights and set it to -infinity (exp(LNrweight) = 0)
    Holdings['LNrweight'] = -np.Inf
    
    #Add Error term and set it to -infinity 
    Holdings['unpref'] = -np.Inf
    
    #Add investor fix effect to 0
    Holdings['cons'] = 0
    
    #----------------------------------------------------------------------#
    # 1.2) Fill Holdings Template  
    #----------------------------------------------------------------------#
    #Merge non-zero Portfolio weights, latent demand and fixed effects from data
    Holdings = Holdings.merge(VarDecomp_df[['mgrno', 'permno', 'rdate', 'LNrweight', 'unpref', 'cons']], how = 'outer', on = ['mgrno', 'permno', 'rdate'])
    
    
    #Fill log relative portfolio weights
    Holdings['LNrweight_y'] = Holdings['LNrweight_y'].fillna(Holdings['LNrweight_x'])
    
    #Drop auxiliary variables
    Holdings = Holdings.drop(['LNrweight_x'], axis = 1)
    Holdings = Holdings.rename(columns={'LNrweight_y': 'LNrweight'})
    
    
    #Fill latent demand
    Holdings['unpref_y'] = np.log(Holdings['unpref_y']) #Transform latent demand into logs (0 latent demand will be -infinity)
    Holdings['unpref_y'] = Holdings['unpref_y'].fillna(Holdings['unpref_x'])
    
    #Drop auxiliary variables
    Holdings = Holdings.drop(['unpref_x'], axis = 1)
    Holdings = Holdings.rename(columns={'unpref_y': 'unpref'})
    
    
    #Fill investor fixed effect
    Holdings = Holdings.drop(['cons_x'], axis = 1)
    Holdings = Holdings.rename(columns={'cons_y': 'cons'})
    Holdings['cons'] = Holdings['cons'].fillna(0)
    
    #Sanity Check: If portfolio weight > -inf, then error term should always also be >-inf
    if len(Holdings[(Holdings['LNrweight'] > -np.inf) &  (Holdings['unpref'] == -np.inf)]) > 0:
        print("     !!!! WARNING !!!!" + "\n" + "Portfolio Holdings positive, but error term is zero")
        print(" Problematic Managers are: ")
        print(Holdings[(Holdings['LNrweight'] > -np.inf) &  (Holdings['unpref'] == -np.inf)].mgrno.unique())
        #print("Action taken: Deleting the Managers")
        #mgrno_exclusion = Holdings[(Holdings['LNrweight'] > -np.inf) & (Holdings['unpref'] == -np.inf)].mgrno.unique()
        #Holdings = Holdings[~Holdings['mgrno'].isin(mgrno_exclusion)]
    
    #Sanity Check: If portfolio weight = -inf, then error term should always also be = -inf
    if len(Holdings[(Holdings['LNrweight'] == -np.inf) & (Holdings['unpref'] > -np.inf)]) > 0:
        print("     !!!! WARNING !!!!" + "\n" + "Portfolio Holdings zero, but error term is positive")
        print(" Problematic Managers are: ")
        print(Holdings[(Holdings['LNrweight'] == -np.inf) & (Holdings['unpref'] > -np.inf)].mgrno.unique())
        #print("Action taken: Deleting the Managers")
        #mgrno_exclusion = Holdings[(Holdings['LNrweight'] == -np.inf) & (Holdings['unpref'] > -np.inf)].mgrno.unique()
        #Holdings = Holdings[~Holdings['mgrno'].isin(mgrno_exclusion)]

    #Sanity Check: No observations missing
    if len(Holdings) - len(uniqueMgrno) * len(uniquePermno) != 0:
        print("     !!!! WARNING !!!!" + "\n" + "Observations mismatched.")
        #print(" Action taken: None ")
        
    #----------------------------------------------------------------------#
    # 2) Create Portfolio Weights, Latent Demand & Fixed Effect
    #----------------------------------------------------------------------#
    PFRweight  = Holdings.pivot_table(index='mgrno', columns=['permno', 'rdate'], values='LNrweight')
    PFRweightMat = PFRweight.to_numpy()
    
    Epsilon  = Holdings.pivot_table(index='mgrno', columns=['permno', 'rdate'], values='unpref')
    EpsilonMat = Epsilon.to_numpy()
        
    cons = Holdings.pivot_table(index='mgrno', columns=['permno', 'rdate'], values='cons')
    consMat = cons.to_numpy()
    
    #----------------------------------------------------------------------#
    # 3) Create Coefficients 
    #----------------------------------------------------------------------#
    
    #Get Names of Coefficients
    GMM_Estimates_cols = [var for var in list(VarDecomp_df.columns) if var.endswith("_beta")]
    
    #Create the Dataframe
    RegCoeffs = VarDecomp_df[['mgrno'] + GMM_Estimates_cols].drop_duplicates(subset = ['mgrno'])
    
    #Very important to sort mgrno so corresponding elements in vectors and matrices actually match
    RegCoeffs = RegCoeffs.sort_values('mgrno')
    
    #Convert to numpy matrix
    RegCoeffs.set_index('mgrno',inplace=True)
    RegCoeffsMat = RegCoeffs.values
    
    #----------------------------------------------------------------------#
    # 4.1) Create Assets under Management Vector
    #----------------------------------------------------------------------#
    
    #Sorting is important
    aum = VarDecomp_df[['mgrno', 'aum']].drop_duplicates(subset = ['mgrno']).sort_values('mgrno')
    aumVec = aum.set_index('mgrno').values.reshape(-1)
    
    #----------------------------------------------------------------------#
    # 4.2) Create Log price and log supply Vector 
    #----------------------------------------------------------------------#
    
    #Sorting is important
    p = VarDecomp_df[["permno", 'LNprc']].sort_values(by="permno").drop_duplicates()
    s = VarDecomp_df[['permno', 'LNshrout']].sort_values(by="permno").drop_duplicates()

    #----------------------------------------------------------------------#
    # 5) Create Matrix of Characteristics
    #----------------------------------------------------------------------#
    #Get Characteristics Names
    Characteristics_Names = [var for var in list(VarDecomp_df.columns) 
                             if not var.endswith("_beta")
                             if var not in ['rdate','bin','mgrno','aum','LNrweight','rweight','LNshrout',
                                            'LNprc','LNcfac','unpref',
                                            'cons']]
    #Sorting is important
    x = VarDecomp_df[Characteristics_Names].drop_duplicates(subset="permno").sort_values(by="permno")


    #----------------------------------------------------------------------#
    # 6) Create LNcfac
    #----------------------------------------------------------------------#
    
    #Sorting is important
    LNcfac = VarDecomp_df[["permno", 'LNcfac']].drop_duplicates(subset="permno").sort_values(by="permno").set_index("permno")


    return PFRweight, PFRweightMat, Epsilon, EpsilonMat, cons, \
        consMat, RegCoeffs, RegCoeffsMat, aum, aumVec, p, s, x, uniquePermno, LNcfac
        
################# For Debugging ################
#p,sVec,x, RegCoeffsMat, EpsilonMat, consMat = prev_p["LNprc"].to_numpy(), Lead_sVec, prev_x, prev_RegCoeffsMat, prev_EpsilonMat, prev_consMat
##################################################
    
def predictLogWr(p,sVec,x, RegCoeffsMat, EpsilonMat, consMat):
    """
    Computes the Log relative portfolio weights from the  characteristics based demand equation 
    
    Inputs
    ------
    p :             Nx1 vector of log prices
    s:              Nx1 vector of log supply
    x:              NxK Pandas DataFrame of Stock Characteristics 
    RegCoeffsMat:   IxK matrix of  beta coefficients
    EpsilonMat:     IxN matrix of residuals of regression of (10)
    consMat:        IxN matrix of investor fixed effects

    Returns
    -------
    LNwR :          IxN matrix containing predicted log relative portfolio weights
    """
    
    #Overwrite Log Market equity as price is the running variable in the variance decomposition
    x['LNme'] = p + sVec
    x = x.drop(["permno"], axis = 1).to_numpy().T
        
    #Predict relative Portfolio weights through characteristics demand based equation (10)
    LNwR = RegCoeffsMat @ x + EpsilonMat + consMat
    
    #Check If LNwR has extremely high values. 
    """
    If yes, then this indicates that something went wrong in the GMM estimates
    because demand is inexplicably large. The Values of LNwR must be capped
    in this case because otherwise the numerical routine will crash as it will
    predict infinite demand.
    
    Flag = False
    if np.max(LNwR) > 30:
        print("!!!! WARNING !!!! \n LNwR has too high values. Values are capped at 30 to ensure the numerical procedure remains in tact.")
        Flag = True #"LNwR capped"
        LNwR = np.clip(LNwR, None, 30)
    """

    
    return LNwR#, Flag
"""
                        SANITY CHECK
                        ------------
Below is a sanity check that was used to check that the function predictLogWr() is correct.


#LNwR,_ = predictLogWr(p["LNprc"].to_numpy(),s['LNshrout'].to_numpy(),x, RegCoeffsMat, EpsilonMat, consMat)

#Sanity Check: Estimated weights in baseline must be equal to actual weights (since we include the error)
#However, numerical imprecisions possible. Check Supremum Distance and compare values
abs_diff = np.abs(np.exp(PFRweightMat) - np.exp(LNwR))
max_diff_index = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)

#Compare values.
print(np.exp(LNwR[max_diff_index]) - np.exp(PFRweightMat[max_diff_index]))
"""


#Extract individual Portfolio weight w_i(n) 
def portfolioWeights(LNwR):
    """
    Computes the level portfolio weights for the inside assets from the
    Log relative portfolio weights
    
    Inputs
    -------
    LNwR : IxN matrix of log relative Portfolio weights from predictLogWr()

    Returns
    -------
    PFweights : IxN matrix of portfolio weights --> Portfolio weight of Investor i (row) of asset n (column)

    """
    #Compute level relative Portfolio weights
    RPFweights = np.exp(LNwR)
    
    #If LNwR too big, then PFweights will be nan, so set this Portfolio weight to 0 (will prevent numerical problems)
    #   This Problem occurs very rarely and will be printed when evaluating np.exp(LNwR)
    #   Stata deals with the overflow problem in a similar manner by excluding these values.
    nan_indices = np.where(np.isnan(RPFweights))
    RPFweights = np.nan_to_num(RPFweights, nan=0)
    
    #Sum the rows of PFweights and format accordingly
    rowSum = np.sum(RPFweights, axis=1) #= \sum_n \delta_i(n) in (11) & (12)
    rowSum = rowSum.reshape(-1, 1)
    
    #Compute w_i(n) accordingly
    PFweights = RPFweights/(1+rowSum)
    
    #Compute w_i(0) accordingly
    PFWeightOutside = 1/(1+rowSum)
    
    #Sanity Check
    SumInsidePFWeights = np.sum(PFweights,axis=1)
    
    Check = SumInsidePFWeights + PFWeightOutside.reshape(-1)
    
    if np.abs(np.mean(Check)-1)>0.01:
        print("       !!!!! WARNING !!!!!" + "\n" + "Portfolio Weights DO NOT sum to 1")
    

    #print("Portfolio weights: nans encountered. LNwR too Big for the following")
    #print("Row indices of NaN values:", nan_indices[0])
    #print("Column indices of NaN values:", nan_indices[1])
    #print("Action taken: Nans set to 0")
        
    return PFweights

################# For Debugging ################
#aumVec = prev_aumVec
################################################
#Compute Aggregate Demand
def AssetDemand(PFweights, aumVec):
    """
    Computes the Aggregate Demand for each Asset.
    
    Inputs
    ------
    PFweights : IxN matrix consisting of w_i(n) Portfolio weights.
    aum :       Ix1 vector consisting of Assets under management A_i

    Returns
    -------
    D : Nx1 vector consisting of aggregate demand for asset n in MONETARY value
    --> This is not the q demand of stocks which gives the amount of stocks demanded

    """
    D = PFweights.T @ aumVec
    
    #If Demand is degenerate due to some update, we need to fix this to avoid divide by 0 errors
    """
    while np.min(D) <1e-323:
        #Set all Values smaller than threshold to some level
          index = np.argmin(D)
          D[index] = 1e-300
          print("Degenerate Demand")
    """
    return D


#Compute Aggregate Supply
def AssetSupply(p,sVec):
    """
    Parameters
    ----------
    p : Nx1 vector consisting of log price of asset n
    s : Nx1 vector consisting of log shares outstanding of asset n

    Returns
    -------
    S : Nx1 vector consisting of Supply of asset n in monetary value
    """
    #Compute Supply of each Asset
    S = np.exp(p+sVec)
    
    return S

#S = AssetSupply(p["LNprc"].to_numpy(),s['LNshrout'].values)

#Sanity Check: Check before computing anything whether the default price is the actual market clearing price.
#dif = np.abs(Q-S)
#dif = dif/np.exp(x['LNme'].to_numpy())
#print(np.mean(np.abs(dif)))

################# For Debugging ################
#x = prev_x
################################################

#Define Excess Demand function g(p)
def MarketClearing(p, sVec, x, aumVec, RegCoeffsMat, EpsilonMat, consMat):
    """
    Compute the MarketClearing function g

    Parameters
    ----------
    p : log market clearing price (running variable)
    
    Fix arguments
    -------------
    s : log supply of the assets (Nx1)
    aum: Assets under Manamgement (Ix1)
    RegCoeffsMat : Estimated Regression Coefficients (Ix7) 
    EpsilonMat : Error Terms (IxN)
    consMat : (IxN)
    LNbe, profit, beta, Gat, divA_be: Stock Characteristics

    Returns
    -------
    g(p) = f(p) - p (Nx1) which should be zero for the market to be cleared
    """
    
    #Predict Log Relative Portfolio Weights based on price update
    #!I can shorten this by extracting the other linear terms that don't change when the price changes
    LNwR = predictLogWr(p, sVec, x , RegCoeffsMat, EpsilonMat, consMat)
    
    #Obtain the levels of individual portfolio weights w_i(n). 
    #As MarketClearing() iterates over the price, this is w_i(n) as a function of price
    PFweights = portfolioWeights(LNwR)
    #If nan values encountered

    #Compute monetary demand per asset n
    Demand = AssetDemand(PFweights, aumVec)
    
    # Compute fp. If demand zero, replace set Demand[i] = p[i] so that gp = 0 and errors are avoided
    # Zero demand happens if PFweights are zero for one asset which can occur due to overflow errors
    fp = np.where(Demand == 0, 0 + p, np.log(Demand) - sVec)
    
    #Compute g(p) which must be zero so that Demand = Supply, i.e. markets clear
    gp = fp - p
    
    return gp  

#Sanity Check: Check if in the baseline the market is cleared
#dif = MarketClearing(p["LNprc"].to_numpy(), sVec, x, aumVec, RegCoeffsMat, EpsilonMat, consMat)
#print(np.mean(np.abs(dif)))

#Compute the intermediary Market Clearing 
def solve_MarketClearing(p, sVec, x, aumVec, RegCoeffsMat, EpsilonMat, consMat, max_iterations=3, tolerance=1e-8):
    """
    Optimize the market clearing function using the Krylov method.

    Parameters:
    - p: Numpy array of prices that serve as the initial guess for the root finding
    - s, x, aum, RegCoeffsMat, EpsilonMat, consMat: Additional arguments passed to the MarketClearing function.
    - max_iterations: Maximum number of iterations to perform (default is 3).
    - tolerance: The stopping criterion for the optimization (default is 1e-8).

    Returns:
    - root: The result of the root-finding --> Intermediary price
    """
    iteration = 0
    stopping = 10
    x0 = p

    while iteration < max_iterations and stopping > tolerance:
        root = scipy.optimize.root(MarketClearing, x0,
                                    args=(sVec, x, aumVec, RegCoeffsMat, EpsilonMat, consMat), 
                                    method='Krylov', tol=tolerance,
                                    options={'maxiter':1_000})
        stopping = np.linalg.norm(root.fun)
        iteration += 1
        x0 = root.x
        #Decrease tolerance to get more accuracy in subsequent iterations
        tolerance = tolerance/10
    
    return root
