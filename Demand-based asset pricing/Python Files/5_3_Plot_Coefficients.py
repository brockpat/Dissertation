# -*- coding: utf-8 -*-
"""
Plot the Coefficients of the Baseline GMM Estimation to replicate Figure 3
in KY19.
"""

#%% Define Inputs

#Location of repository
path = ".../Python Replication Package" 

#Select Estimates (Only Baseline Supported)
variable_selection = 'KY19_baseline'
#%% Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats.mstats import winsorize
#%% Data

#Load Data to obtain the manager type
df_manager = pd.read_stata(path + "/Data" + "/Manager_Summary.dta")
df_manager['rdate'] =  pd.to_datetime(df_manager["rdate"]) #if reading in csv
df_manager = df_manager[df_manager.rdate.dt.year < 2023]

#Load GMM Estimates
GMM_Estimates = pd.read_csv(path + "/Output/Estimations/" + variable_selection + "_Restricted.csv")
GMM_Estimates = GMM_Estimates.drop_duplicates(subset = ['rdate','bin'])
GMM_Estimates["rdate"] = pd.to_datetime(GMM_Estimates["rdate"])
GMM_Estimates = GMM_Estimates[GMM_Estimates['rdate'] != '2013-03-31'] #Not enough data for this date


#Extract column names of Estimates
cols = list(GMM_Estimates.columns.drop(['rdate','bin','Error']))

#If Estimator did not converge, set all estimates to 0
GMM_Estimates.loc[np.isnan(GMM_Estimates['Error']) == True, cols] = 0

#If convergence is poor, set all estimates to 0
GMM_Estimates.loc[GMM_Estimates['Error']>0.9, cols] = 0

#If market equity coefficient too low, set all estimates to 0
GMM_Estimates.loc[GMM_Estimates['LNme'] < -20, cols] = 0

#If market equity coefficient greater than 1, set all estimates to 0
GMM_Estimates.loc[GMM_Estimates['LNme'] > 1, cols] = 0

#If any estimates take large positive or negative values, set all zero
GMM_Estimates[cols] = GMM_Estimates[cols].mask(GMM_Estimates[cols].abs() > 200, 0)

GMM_Estimates.drop('Error',axis=1, inplace=True)

#%% Plot Coefficients

Baseline_endog_Variables_Names = cols[:-1]

#Merge Coefficients to type
df_coeffs = (df_manager
             .drop_duplicates(subset = ['rdate','bin'])
             .merge(GMM_Estimates, on = ['rdate','bin'], 
                    how = 'right', suffixes = ("","_beta"))
            )

df_weighted_average = (df_coeffs
                       # Make Type column a string
                       .assign(type=lambda df: df['type'].astype(str))
                       # Filter out 'Other' type
                       .pipe(lambda x: x[x['type'] != 'Other'])  
                       # Group by 'rdate' and 'type'
                       .groupby(['rdate', 'type'])
                       # Compute the weighted average of each column within group
                       .apply(lambda group: (group[Baseline_endog_Variables_Names].multiply(group['aum'], axis=0).sum() / group['aum'].sum()))
                       .reset_index()
                       # Winsorize specified columns at 1% lower and 99% upper percentiles
                       .assign(**{col: lambda df, col=col: winsorize(df[col], limits=[0.005, 0.005]) for col in Baseline_endog_Variables_Names})
                       )

#------ Create the Plot
n_rows = 3 #3
n_cols = 2 #2

colors = {'Households': 'grey', 'Banks':'black', 'Insurance companies': 'blue',
          'Investment advisors': 'cyan', 'Mutual funds':'red', 'Pension funds':'orange'}
line_style = {'Households': 'dashed', 'Banks':'solid', 'Insurance companies': 'solid',
          'Investment advisors': 'solid', 'Mutual funds':'solid', 'Pension funds':'solid'}

# Create a figure with a 3x2 grid of subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 12))  # figsize (14,12)

#titles = {'LNme': 'Log Market Equity', 'LNbe': 'Log Book Equity', 
          #'profit': 'Profit to BE', 'Gat':'Log Growth Assets', 
          #'divA_be': 'Annual Dividends to BE', 'beta':'Market Beta'}

titles = {'LNme': 'Log Market Equity', 'LNbe': 'Log Book Equity', 
          'profit': 'Profitability', 'Gat':'Investment', 
          'divA_be': 'Dividends to Book Equity', 'beta':'Market Beta'}

plt.rcParams.update({
    'font.size': 18,          # Base font size
    'axes.titlesize': 20,     # Title font size
    'axes.labelsize': 18,     # Axis labels font size
    'xtick.labelsize': 18,    # X-axis tick labels font size
    'ytick.labelsize': 18,    # Y-axis tick labels font size
    'legend.fontsize': 18,    # Legend font size
    'legend.title_fontsize': 18 # Legend title font size
})

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Iterate over each variable and its corresponding subplot
for i, var in enumerate(Baseline_endog_Variables_Names):
    ax = axes[i]  # Select the corresponding subplot
    
    # Group by 'type' and plot each type on the same subplot
    for type_name, group in df_weighted_average.groupby('type'):
        ax.plot(group['rdate'], group[var], label=type_name, color=colors[type_name], linestyle=line_style[type_name])
        
        
    # Set labels, title, and legend for each subplot
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(titles[var])

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.11), fontsize = 18, ncol=3, title="Type", title_fontsize=18)
# Adjust layout to prevent overlap
plt.tight_layout()

plt.savefig(path + "/Output/Plots" + "/TS_Coefficients.pdf",dpi = 600, bbox_inches='tight')

# Display the combined figure with all subplots
plt.show()
