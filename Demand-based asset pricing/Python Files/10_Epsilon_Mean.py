# -*- coding: utf-8 -*-
"""
Computes the Median of Epsilon to show the bias
"""

#%% Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import copy

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
#%% Read in Data

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

Baseline_endog_Variables_Names = ['LNme','LNbe', 'profit', 'Gat', 'divA_be','beta']
Baseline_exog_Variables_Names  = ['IVme','LNbe', 'profit', 'Gat', 'divA_be','beta']

df_Estimates = pd.read_csv(path + "/Output/Estimations" + "/KY19_baseline_Restricted.csv")
df_Estimates['rdate'] =  pd.to_datetime(df_Estimates["rdate"]) #if reading in csv
df_Estimates = df_Estimates.drop_duplicates(subset = ['rdate','bin'])

df_UnrestrictedEstimates = pd.read_csv(path + "/Output/Estimations" + "/KY19_baseline_unrestricted.csv")
df_UnrestrictedEstimates['rdate'] =  pd.to_datetime(df_UnrestrictedEstimates["rdate"]) #if reading in csv
df_UnrestrictedEstimates = df_UnrestrictedEstimates.drop_duplicates(subset = ['rdate','bin'])

#%% Mean of Error for non-zero Holdings for Unrestricted Estimates (else-wise you'd kick out all big investors where the fit is good)

#Extract unique dates
Quarters = Holdings['rdate'].unique()
Quarters = Quarters[Quarters.year<2023]

#Prepare final results table
Results = []#pd.DataFrame(columns=['rdate', 'bin', 'Error_Mean_All', 'Error_Mean_NonZeros'])

#Loop over all Quarters
for quarter in Quarters:
    print(quarter)

    ### --- Slice Datasets
    #Holdings Sliced
    Holdings_Q = Holdings[Holdings['rdate'] == quarter]

    #Baseline Stock Characteristics Sliced
    StocksQ_Q = StocksQ[StocksQ['date'] == quarter]
    
    #Estimates Sliced
    df_UnrestrictedEstimates_Q = df_UnrestrictedEstimates[df_UnrestrictedEstimates['rdate']==quarter]

    ### --- Merge Stock Characteristics to Holdings Data to build X & Z Matrix for GMM
    #Merge Baseline Stock Characteristics
    df_Q = Holdings_Q.merge(StocksQ_Q[["permno", "date", "LNme", 'LNbe', 'profit', 'Gat', 'divA_be','beta']], left_on=["rdate", "permno"], right_on=["date", "permno"])

    #Drop any remaining Missing Values to avoid errors
    df_Q = df_Q.dropna(subset=Baseline_endog_Variables_Names + ['IVme','rweight'])

    #Assing the constant to the dataframe
    df_Q = df_Q.assign(constant=1)

    for i_bin in np.sort(df_Q['bin'].unique()):
        print("     Bin: " + str(i_bin))
        
        ### --- Slice DataSet on Bin
        df_Q_bin = df_Q[df_Q['bin'] == i_bin]
        
        #Slice Estimates on Bin
        df_UnrestrictedEstimates_bin = df_UnrestrictedEstimates_Q[df_UnrestrictedEstimates_Q['bin'] == i_bin]
        
        #Generate Results Dataframe
        Results_bin = pd.DataFrame(columns=['rdate', 'bin', 'Error_Mean_All', 'Error_Mean_NonZeros'])
        Results_bin.at[0,'rdate'] = quarter
        Results_bin.at[0,'bin'] = i_bin
        
        
        #Extract Data
        X = df_Q_bin[['rweight'] + Baseline_endog_Variables_Names + ['constant','cons']]
        
        X = X.dropna()
        
        #Extract variables from Data
        y_all = X['rweight'].values
        X_all = copy.deepcopy(X.drop('rweight',axis=1).values)
        
        #Extract exlusively non-zero variables from Data
        X_nz = X[X['rweight']>0]
        y_nz = X_nz['rweight'].values
        X_nz = X_nz.drop('rweight',axis=1).values
        
        #Extract regression coefficients
        beta = df_UnrestrictedEstimates_bin[Baseline_endog_Variables_Names + ['constant']].values.reshape(-1)
        
        #Compute vector of errors (Epsilon() = epsilon-1)
        epsilon_all = Epsilon(beta,X_all,y_all) + 1
        epsilon_nz = Epsilon(beta,X_nz,y_nz) + 1
        
        #Mean of non-zero holdings
        mean_epsilon_all = np.mean(epsilon_all)
        mean_epsilon_nz = np.mean(epsilon_nz)
                               
        Results_bin.loc[:,'Error_Mean_All'] = mean_epsilon_all
        Results_bin.loc[:,'Error_Mean_NonZeros'] = mean_epsilon_nz
        Results.append(Results_bin)

Results = pd.concat(Results)
Results.to_csv(path + "/Output" + "/Mean_Epsilon_Baseline_Unrestricted.csv", index = False)

#%% Plot Time Series

df = (pd
      .read_csv(path + "/Output" + "/Mean_Epsilon_Baseline_Unrestricted.csv")
      .assign(rdate=lambda x: pd.to_datetime(x.rdate))
      )

df_cs = (df.groupby('rdate')['Error_Mean_NonZeros']
         .agg(eps_median=('median'),
              eps_quantile_25=lambda x: x.quantile(0.25),
              eps_quantile_75=lambda x: x.quantile(0.75)
              )
         .reset_index()
         )

# Set figure size
plt.figure(figsize=(10, 6))

# Plot the mean epsilon
plt.plot(df_cs['rdate'], df_cs['eps_median'], color='black')

# Fill between the 25th and 75th percentiles
plt.fill_between(df_cs['rdate'], df_cs['eps_quantile_25'], df_cs['eps_quantile_75'], color='black', alpha=0.3)

# Set formatting for the x-axis
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks every 4 months
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))    # Format ticks as years
plt.xlim(df_cs['rdate'].min(), df_cs['rdate'].max())               # Set x-axis limits

# Set y-axis label and legend
plt.ylabel('', fontsize=14)
#plt.legend(fontsize=14)

# Rotate x-axis tick labels and adjust font size
plt.xticks(rotation=90, fontsize=14)
plt.yticks(fontsize=14)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(path + "/Output/Plots/TS_Epsilon_Median.pdf", dpi=600, bbox_inches='tight')

# Show the plot (optional)
plt.show()