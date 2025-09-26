"""
Computes Variance Decomposition with Endogenous Supply.

The main extension is the function LogAssetSupply(p,s0) which replaces
the exogenous supply in all other functions with the endogenous one.

This file serves as a blueprint to see how endogenous supply could be implemented.
We only tested a simple log(supply) = a+b*log(p) function.
"""
#%% Libraries
import pandas as pd
import numpy as np
import scipy
import copy

#%% Define Inputs

#Location of repository
path = ".../Python Replication Package" 

#Select Restricted Estimates
filename = 'KY19_baseline'

#Select Quarter on which the Variance Decomposition is computed (KY19 use 2nd quarter of every year)
q = 2

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
    
#Supply Function
def calibrateSupply(p, s):
    """
    Parameters
    ----------
    p : Baseline log market price from baseline
    s : Baseline log supply from baseline

    Returns
    -------
    Parameters s0 from function s = s0 + elasticity*p where s_baseline = s0+p_baseline
    """
    s0 = s-elasticity*p
    
    return s0

#Compute Supply
def LogAssetSupply(p,s0):

    s = s0+elasticity*p #Elasticity Estimated from Cross Section  
    return s
    
#Implement Characteristics-based demand equation (10) as a function of price
def predictLogWr(p,s0,x, betaMat, EpsilonMat, consMat):

    #Overwrite characteristics
    s = LogAssetSupply(p,s0)
    
    x['LNme'] = p + s
    x = x.drop(["permno"], axis = 1)
    
    x = x.to_numpy().T
    
    #Predict relative Portfolio weights through characteristics demand based equation (10)
    LNwR = betaMat @ x + EpsilonMat + consMat

    return LNwR

#Extract individual Portfolio weight w_i(n) 
def portfolioWeights(LNwR):
    #Compute level relative Portfolio weights
    PFweights = np.exp(LNwR)
    
    #Sum the rows of PFweights and format accordingly
    rowSum = np.sum(PFweights, axis=1) #= \sum_n \delta_i(n) in (11) & (12)
    rowSum = rowSum.reshape(-1, 1)
    
    #Compute w_i(n) according to (11)
    PFweights = PFweights/(1+rowSum)
    
    #Compute w_i(0) according to (12)
    PFWeightOutside = 1/(1+rowSum)
    
    #Sanity Check
    SumInsidePFWeights = np.sum(PFweights,axis=1)
    
    Check = SumInsidePFWeights + PFWeightOutside.reshape(-1)
    
    if np.abs(np.mean(Check)-1)>0.01:
        print("       !!!!! WARNING !!!!!" + "\n" + "Portfolio Weights DO NOT sum to 1")
    
    #If LNwR too big, then PFweights will be nan. This causes the code to crash
    nan_indices = np.where(np.isnan(PFweights))
    PFweights = np.nan_to_num(PFweights, nan=0)
    #print("Portfolio weights: nans encountered. LNwR too Big for the following")
    #print("Row indices of NaN values:", nan_indices[0])
    #print("Column indices of NaN values:", nan_indices[1])
    #print("Action taken: Nans set to 0")
        
    return PFweights

#Compute Aggregate Demand
def AssetDemand(PFweights, aum):

    aum = aum['aum'].to_numpy()
    D = PFweights.T @ aum

    return D

#Define Market Clearing function g(p)
def MarketClearing(p, s0, x, aum, betaMat, EpsilonMat, consMat):

    #Predict Log Relative Portfolio Weights based on price update
    #!I can shorten this by extracting the other linear terms that don't change when the price changes
    LNwR = predictLogWr(p,s0, x , betaMat, EpsilonMat, consMat)
    
    #Obtain the levels of individual portfolio weights w_i(n). 
    #As MarketClearing() iterates over the price, this is w_i(n) as a function of price
    PFweights = portfolioWeights(LNwR)
    #If nan values encountered

    #Compute monetary demand per asset n
    Demand = AssetDemand(PFweights, aum)
    
    #Compute f(p) [log Demand - log Supply]
    fp = np.log(Demand) - LogAssetSupply(p, s0)
    
    #Compute g(p) which must be zero so that Demand = Supply, i.e. markets clear
    gp = fp - p
    
    return gp  

def solve_MarketClearing(p, s0, x, aum, betaMat, EpsilonMat, consMat, max_iterations=3, tolerance=1e-8):

    iteration = 0
    stopping = 10
    x0 = p

    while iteration < max_iterations and stopping > tolerance:
        root = scipy.optimize.root(MarketClearing, x0,
                                    args=(s0, x, aum, betaMat, EpsilonMat, consMat), 
                                    method='Krylov', tol=tolerance,
                                    options={'maxiter':1_000})
        stopping = np.linalg.norm(root.fun)
        iteration += 1
        x0 = root.x
        #Decrease tolerance to get more accuracy in subsequent iterations
        tolerance = tolerance/10
    
    return root
#%% Load Data

#Load StocksQ which contains observed annual return and the dividend return
StocksQ = pd.read_stata(path + "/Data" + "/StocksQ.dta")
StocksQ["date"] = StocksQ["date"] - pd.offsets.MonthBegin()+pd.offsets.MonthEnd()
#%% Root Finding

#Define Dictionaries used in the root-finding loop to extract the correct quarter
day_dict = {1:"31", 2:"30", 3:"30", 4:"31"}
month_dict = {1:"03", 2:"06", 3:"09", 4:"12"}

#Define Dataframes to store results
df_results = pd.DataFrame()
df_numerical_errors = pd.DataFrame(columns = ['rdate','root1_error','root2_error','root3_error','root4_error','root5_error'])

#Read in Holdings Data and do root finding
for year in range(2002, 2022):
    
    print("------------------------------------------- \n" + 
          "------------------------------------------- \n \n" +
          "              YEAR = " + str(year) + "\n \n" + 
          "------------------------------------------- \n" + 
          "------------------------------------------- \n")
    #---------------------------------------------------------------------#
    # 1) Read in Datasets
    #---------------------------------------------------------------------#
    
    #---- Read in time t Data
    prev_VarDecomp_df = pd.read_stata(path + "/Output/Variance Decomposition Python/" + f"VarDecomp_{filename}_Restricted_{year}"f"-{month_dict[q]}"f"-{day_dict[q]}.dta")
    prev_VarDecomp_df['rdate'] =  pd.to_datetime(prev_VarDecomp_df["rdate"])
    
    #Drop Stocks that are not held by ANY investor 
    prev_VarDecomp_df = prev_VarDecomp_df[prev_VarDecomp_df.groupby("permno").LNrweight.transform("count")>0]
    
    #Drop Zero Holdings of each Investor (Demand is zero and needn't be explicitly computed)
    prev_VarDecomp_df = prev_VarDecomp_df.dropna(subset = 'LNrweight')
    
    #Sort dataframe (extremely important!)
    #   If not sorted, then in getObjects() while iterating over managers and stocks the indices
    #   might not refer to same manager or stock for the outputs
    prev_VarDecomp_df = prev_VarDecomp_df.sort_values(['mgrno','permno'])

    
    #----Read in time t+4 Holdings Data
    Lead_VarDecomp_df = pd.read_stata(path + "/Output/Variance Decomposition Python/" + f"VarDecomp_{filename}_Restricted_{year + 1}"f"-{month_dict[q]}"f"-{day_dict[q]}.dta")
    Lead_VarDecomp_df['rdate'] =  pd.to_datetime(Lead_VarDecomp_df["rdate"]) #if reading in csv
    
    #Drop Stocks that are not held by ANY investor (Asset Demand System not applicable to these Stocks)
    Lead_VarDecomp_df = Lead_VarDecomp_df[Lead_VarDecomp_df.groupby("permno").LNrweight.transform("count")>0]
    
    #Drop Zero Holdings of each Investor (since demand is zero it needn't be explicitly computed)
    Lead_VarDecomp_df = Lead_VarDecomp_df.dropna(subset = 'LNrweight')
    
    #Sort dataframe (extremely important!)
    #   If not sorted, then in getObjects() while iterating over managers and stocks the indices
    #   might not refer to same manager or stock for the outputs
    Lead_VarDecomp_df = Lead_VarDecomp_df.sort_values(['mgrno','permno'])
            
        
    #---------------------------------------------------------------------#
    # 2) Get Objects
    #---------------------------------------------------------------------#
    
    #These Objects are used for the root-finding process
    prev_PFRweight, prev_PFRweightMat, prev_Epsilon, prev_EpsilonMat, prev_cons, prev_consMat, \
        prev_RegCoeffs, prev_betaMat, prev_aum, _, prev_p, prev_s, prev_x, prev_uniquePermno, prev_LNcfac = getObjects(prev_VarDecomp_df, year, q)
      
    _, _, _, _, _, _, \
        _, _, _,_, Lead_p, Lead_s, _, \
            _, _ = getObjects(Lead_VarDecomp_df, year + 1, q)

    
    #---------------------------------------------------------------------#
    # 3) Lead supply, characteristics, aum, beta coefficients and compute
    #        the intermediary market clearing price
    #---------------------------------------------------------------------#
    
    #Compute the intermediary price resulting from a ceteris paribus supply change.
    print("------------------------------------------- \n" + 
          "              Leading Supply" + "\n"
          + "-------------------------------------------")
    
    elasticity = -0.14 #Obtained from Fixed Effects Regression
    
    #----- i) Lead constant of supply function
    #Create s0_prev
    df_s0 = prev_s.merge(prev_p, how = 'outer', on = 'permno').sort_values(by = 'permno')
    df_s0['s0_prev'] = df_s0['LNshrout'] - elasticity*df_s0['LNprc']
    
    #Create s0_Lead
    df_s0 = df_s0.merge(Lead_s, how = 'outer', on = 'permno', suffixes = ("", "_Lead")).sort_values(by = 'permno')
    df_s0 = df_s0.merge(Lead_p, how = 'outer', on = 'permno', suffixes = ("", "_Lead")).sort_values(by = 'permno')
    df_s0['s0_Lead'] = df_s0['LNshrout_Lead'] - elasticity*df_s0['LNprc_Lead']
    
    #Overwrite s0_Lead with s0_prev if asset in t no longer exists        
    df_s0['s0_Lead'].fillna(df_s0['s0_prev'] ,inplace=True)
    
    #Restrict investment universe to permnos held in t
    s0_Lead = df_s0[df_s0['permno'].isin(prev_uniquePermno['permno'])][['permno', 's0_Lead']].sort_values(by = 'permno')
   

    #Solve for the Market Clearing Price
    root1 = solve_MarketClearing(prev_p["LNprc"].to_numpy(), 
                                 s0_Lead['s0_Lead'].to_numpy(), prev_x, prev_aum, prev_betaMat, prev_EpsilonMat, prev_consMat)
    print("Root1 Approximation Error = " + str(np.linalg.norm(root1.fun)))
    
    #Save Results
    LNprc_1 = prev_p.copy()
    LNprc_1.loc[:, 'LNprc'] = root1.x 
    LNprc_1 = LNprc_1.rename(columns = {'LNprc':'LNprc_1'})
    

    #----- ii) Lead Stock Characteristics X
    
    #Compute the intermediary price resulting from a ceteris paribus Stock characteristics change.
    print("------------------------------------------- \n" + 
          "          Leading Characteristics" + "\n"
          + "-------------------------------------------")
    
    #-- Update Characteristics of Stocks that are still held in t+4
    #Create Dataframe with x_t and x_{t+4} for stocks at time t
    x_list = [item for item in prev_x.columns if item not in ['permno', 'constant','cons']]
    x_list_lead = [item + "_Lead" for item in prev_x.columns if item not in ['permno','constant','cons']]
    
    Lead_x = prev_x.merge(Lead_VarDecomp_df[['permno'] + x_list].drop_duplicates(subset = 'permno'), 
                          how='left', 
                           on='permno', suffixes=('', '_Lead'))
    
    #Update x_t with values from t+4 if they exist.
    Lead_x[x_list] = np.where(pd.notnull(Lead_x[x_list_lead]), Lead_x[x_list_lead], Lead_x[x_list])
    #Delete auxiliary column
    Lead_x.drop(x_list_lead,axis = 1, inplace = True)
    
    #Solve for the Market Clearing Price
    root2 = solve_MarketClearing(LNprc_1['LNprc_1'].to_numpy(), 
                                 s0_Lead['s0_Lead'].to_numpy(), Lead_x, prev_aum, prev_betaMat, prev_EpsilonMat, prev_consMat)
    print("Root2 Approximation Error = " + str(np.linalg.norm(root2.fun)))
    
    #Save Results
    LNprc_2 = prev_p.copy()
    LNprc_2.loc[:, 'LNprc'] = root2.x
    LNprc_2 = LNprc_2.rename(columns = {'LNprc':'LNprc_2'})
    
    
    #----- iii) Lead AUM 
   
    #Compute the intermediary price resulting from a ceteris paribus AUM Change.
    print("------------------------------------------- \n" + 
          "              Leading AUM" + "\n"
          + "-------------------------------------------")
    
    #-- Update AUM of Managers who still exist in t+4
    #Create Dataframe with aum_t and aum_{t+4} for stocks at time t
    Lead_aum = prev_aum.merge(Lead_VarDecomp_df[['mgrno','aum']].drop_duplicates(subset = 'mgrno'), how='left', 
                           on='mgrno', suffixes=('', '_Lead'))
    #Update aum_t with values from t+4 if they exist.
    Lead_aum['aum'] = np.where(pd.notnull(Lead_aum['aum_Lead']), Lead_aum['aum_Lead'], Lead_aum['aum'])
    #Delete auxiliary column
    Lead_aum.drop('aum_Lead',axis = 1, inplace = True)
    
    #Solve for the Market Clearing Price
    root3 = solve_MarketClearing(LNprc_2['LNprc_2'].to_numpy(), 
                                 s0_Lead['s0_Lead'].to_numpy(), Lead_x, Lead_aum, prev_betaMat, prev_EpsilonMat, prev_consMat)
    print("Root3 Approximation Error = " + str(np.linalg.norm(root3.fun)))
    
    #Save Results
    LNprc_3 = prev_p.copy()
    LNprc_3.loc[:, 'LNprc'] = root3.x
    LNprc_3 = LNprc_3.rename(columns = {'LNprc':'LNprc_3'})
    
    #----- iv) Lead  betas (coefficients) & 'cons' 
    
    #Compute the intermediary price resulting from a ceteris paribus Beta change
    print("------------------------------------------- \n" + 
          "          Leading Beta & 'cons' " + "\n"
          + "-------------------------------------------")
    
    #Extract the relevant columns
    beta_list = list(prev_RegCoeffs.columns) + ['cons']
    beta_list_Lead = [item + "_Lead" for item in beta_list]
    
    #Create a dataframe with Lead values
    Lead_RegCoeffs = prev_VarDecomp_df[['mgrno','permno'] + beta_list].merge(
        Lead_VarDecomp_df[beta_list + ['mgrno']].drop_duplicates(subset = 'mgrno'),
                                                                  on = 'mgrno', how = 'left',
                                                                  suffixes = ("", "_Lead"))
    
    #Update beta & cons with values from t+4 if they exist
    Lead_RegCoeffs[beta_list] = np.where(pd.notnull(Lead_RegCoeffs[beta_list_Lead]), Lead_RegCoeffs[beta_list_Lead], Lead_RegCoeffs[beta_list])
    #Delete auxiliary column
    Lead_RegCoeffs.drop(beta_list_Lead,axis = 1, inplace = True)
    
    #Create RegCoeffsMat
    Lead_betaMat = Lead_RegCoeffs.drop_duplicates(subset = 'mgrno')[beta_list].drop('cons',axis=1).to_numpy()
    #Create consMat
    Lead_consMat = np.nan_to_num(Lead_RegCoeffs.pivot_table(index='mgrno', columns=['permno'], values='cons').to_numpy(),0)
    
    #Solve for the Market Clearing Price
    root4 = solve_MarketClearing(LNprc_3['LNprc_3'].to_numpy(), 
                                 s0_Lead['s0_Lead'].to_numpy(), Lead_x, Lead_aum, Lead_betaMat, prev_EpsilonMat, Lead_consMat)
    print("Root4 Approximation Error = " + str(np.linalg.norm(root4.fun)))
    
    #Save Results
    LNprc_4 = prev_p.copy()
    LNprc_4.loc[:, 'LNprc'] = root4.x
    LNprc_4 = LNprc_4.rename(columns = {'LNprc':'LNprc_4'})
    
    #---------------------------------------------------------------------#
    # 4) Update Investment universe
    #---------------------------------------------------------------------#

    print("------------------------------------------- \n" + 
          "          Extensive Margin" + "\n"
          + "-------------------------------------------")

    #Merge unpref from t to unpref in t+4 if the link exists
    prev_unpref = Lead_VarDecomp_df[['mgrno','permno','unpref']].merge(
        prev_VarDecomp_df[['mgrno','permno','unpref']], how = 'left',
        on = ['mgrno', 'permno'], suffixes=('', '_prev'))
    
    #Downgrade the values in t if downgrade exists
    prev_unpref['unpref'] = np.where(pd.notnull(prev_unpref['unpref_prev']), prev_unpref['unpref_prev'], prev_unpref['unpref'])
    prev_unpref.drop('unpref_prev',axis = 1, inplace = True)
    
    #Merge Downgraded Values to the DataFrame at time t+4
    Lead_VarDecomp_df = Lead_VarDecomp_df.merge(prev_unpref, on = ['mgrno','permno'], how = 'left', 
                                      suffixes = ("","_Lag"))
            
    Lead_VarDecomp_df.drop('unpref',axis = 1, inplace = True)
    Lead_VarDecomp_df = Lead_VarDecomp_df.rename(columns = {'unpref_Lag':'unpref'})

    #Get Objects for the Root Finding
    Lead_PFRweight, Lead_PFRweightMat, prev_Epsilon, prev_EpsilonMat, Lead_cons, Lead_consMat, \
        Lead_RegCoeffs, Lead_betaMat, Lead_aum, _, Lead_p, Lead_s, Lead_x, \
            Lead_uniquePermno, Lead_LNcfac = getObjects(Lead_VarDecomp_df, year + 1, q)
    
    #Compute s0 for the t+4 investment universe
    s0_Lead = Lead_p.merge(Lead_s, on = 'permno').sort_values(by='permno')
    s0_Lead['s0_Lead'] = s0_Lead['LNshrout'] - elasticity*s0_Lead['LNprc']
    
    
    #Compute the Market Clearing Price (notice: Price has different dimension as t+4 data has different pernos than t data)
    root5 = solve_MarketClearing(Lead_p["LNprc"].to_numpy(),
                                 s0_Lead['s0_Lead'].to_numpy(), Lead_x, Lead_aum, Lead_betaMat, prev_EpsilonMat, Lead_consMat)
    print("Root5 Approximation Error = " + str(np.linalg.norm(root5.fun)))
    
    #Save Results
    LNprc_5 = Lead_p.copy()
    LNprc_5.loc[:, 'LNprc'] = root5.x
    LNprc_5 = LNprc_5.rename(columns = {'LNprc':'LNprc_5'})
    
    #---------------------------------------------------------------------#
    # 5)  Compute & Save Final Returns
    #---------------------------------------------------------------------#
    
    #------------------ Price & Return Data -------------------------------
    
    #Save LNprc1 to LNprc6 --> LNprc6 is the actual observed price at t+4
    df_LNprc = prev_p.rename(columns = {'LNprc':'LNprc_prev'}).merge(
        LNprc_1, on = "permno", how = 'outer', suffixes = ("", ""))
    df_LNprc = df_LNprc.merge(LNprc_2, on = "permno", how = 'outer', suffixes = ("", ""))
    df_LNprc = df_LNprc.merge(LNprc_3, on = "permno", how = 'outer', suffixes = ("", ""))
    df_LNprc = df_LNprc.merge(LNprc_4, on = "permno", how = 'outer', suffixes = ("", ""))
    df_LNprc = df_LNprc.merge(LNprc_5, on = "permno", how = 'outer', suffixes = ("", ""))
    df_LNprc = df_LNprc.merge(
        Lead_VarDecomp_df.rename(columns = {'LNprc':'LNprc_6'})[['permno','LNprc_6']].drop_duplicates('permno'),
                              how = 'outer', suffixes = ("","")).sort_values('permno')
    
    #Merge LNcfac & only keep Stocks that exist in both t and t+4 by inner merge
    df_LNprc = (df_LNprc
                .merge(prev_LNcfac.reset_index(), on = 'permno', how = 'inner')
                .rename(columns = {'LNcfac': 'LNcfac_prev'})
                )
    
    df_LNprc = (df_LNprc
                .merge(Lead_LNcfac.reset_index(), on = 'permno', how = 'inner')
                .rename(columns = {'LNcfac': 'LNcfac_Lead'})
                )
    
    #Compute The intermediary returns central to the Variance Decomposition
    df_LNprc['LNret1'] 	= df_LNprc['LNprc_1']+df_LNprc['LNcfac_Lead']-(df_LNprc['LNprc_prev']+ df_LNprc['LNcfac_prev'])
    for i in range(2, 7):
        df_LNprc[f'LNret{i}'] = df_LNprc[f'LNprc_{i}'] - df_LNprc[f'LNprc_{i-1}']

    #Keep the intersection of permnos, i.e. permnos existing in both t and t+4
    df_LNprc = df_LNprc[df_LNprc['permno'].isin(Lead_uniquePermno['permno'])]
    df_LNprc = df_LNprc.dropna(subset=['LNprc_prev'])
    
    #Add date column
    df_LNprc['rdate'] = Lead_VarDecomp_df['rdate'][0]
    
    #Merge observed annual and dividend return
    df_LNprc = df_LNprc.merge(StocksQ[['date','permno','LNretA','LNretdA']], left_on = ['rdate','permno'],
                              right_on = ['date','permno'], suffixes = ("","")).drop("date",axis=1)
    
    #Append results
    df_results = pd.concat([df_results,df_LNprc])

    #-------------------- Save the numerical errors -----------------------
    df_error= pd.DataFrame(columns = ['rdate', 'root1_error',
                                      'root2_error','root3_error',
                                      'root4_error','root5_error'])

    df_error.loc[0] = [
        Lead_VarDecomp_df["rdate"][0],
        np.linalg.norm(root1.fun),
        np.linalg.norm(root2.fun),
        np.linalg.norm(root3.fun),
        np.linalg.norm(root4.fun),
        np.linalg.norm(root5.fun)#, 
            ]
    df_numerical_errors = pd.concat([df_numerical_errors,df_error])
    
    #-------------------------   Export Results ---------------------------
    df_results.to_stata(path + "/Output/Variance Decomposition Python/GMMbaseline_EndgSupply_FEelasticity.dta")
    df_numerical_errors.to_csv(path + "/Output/Variance Decomposition Python/GMMbaseline_EndgSupply_FEelasticity.csv")