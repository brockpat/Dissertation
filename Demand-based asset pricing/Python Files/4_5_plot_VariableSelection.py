"""
Plot the Results of the Variable Selection Routines
"""

#%%
path = ".../Python Replication Package"

#Choose the Variable Selection Procedure for which the plots are desired
variable_selection = "BackwardSelection_GMM_NoIV" 
#LASSO_IV, LASSO, ADAPTIVE_LASSO_OLSWEIGHTS, BackwardSelection_IV2SLS, BackwardSelection_GMM_NoIV
#%% Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot  as plt
from matplotlib.gridspec import GridSpec

import copy

#%% Load Data
df_managers = pd.read_stata(path + "/Data/Manager_Summary.dta")

df_managers['sumBinAUM'] = df_managers.groupby(['rdate', 'bin'])['aum'].transform('sum')

Chars_Baseline_Names = ['LNme','LNbe', 'profit', 'Gat', 'divA_be','beta']

#%% Function to get DataFrame
def select_vars(variable_selection):     
    if variable_selection in ['LASSO', 'LASSO_IV', 'ADAPTIVE_LASSO_OLSWEIGHTS', 
                              'BackwardSelection_IV2SLS', 'BackwardSelection_GMM_NoIV']:
        df_varSelect = (pd.read_csv(path + "/Output" + "/Variable Selection/" + variable_selection + ".csv")
                    #Reformat Date Variable
                    .assign(rdate=lambda df: pd.to_datetime(df['rdate']))
                    # Re-Name LNme if IV2SLS was used
                    .rename(columns={'IVme_stage1': 'LNme'})  
                    # Drop the constant (irrelevant for plot)
                    .drop('constant', axis=1)  
                    # Make entries binary (chosen vs. not chosen)
                    .set_index(["rdate", "bin"]).ne(0).reset_index() 
                    # Assign grouped bin indicator - Leave out Households
                    .assign(groupedBin=lambda df: np.where(df['bin'] == 0, np.nan, np.where(df['bin'] < 191, 1, 0)))  
                    # Get type of the bin
                )
        df = (df_managers.merge(df_varSelect, on = ['bin','rdate'], how='left')
                                .assign(type=lambda df: df['type'].astype(str))
                                .pipe(lambda x: x[x['rdate'].dt.year < 2023])
                    )
        char_names = [col for col in df.columns if col not in ['type', 'mgrno', 'aum','groupedBin','rdate', 'bin',
                                                               'mgrid','Ubin','typecode','owntype','mgrname','sumBinAUM']]
    else:
        raise ValueError('This variable Selection does not exist.')
    return df, char_names


# Custom function to calculate weighted average
def weighted_average(df, values, weights):
    return sum(df[weights] * df[values]) / df[weights].sum()

#%% Extract data
df, char_names = select_vars(variable_selection)

#%% Bar Plot

# Create a 4x2 grid of subplots
width = 20
height = 16


#----------------------- Plot Unweighted Average ------------------------------
# Group the DataFrame by 'type'. Each 'bin' has weight 1/N
grouped = (
    df.drop_duplicates(subset=['rdate', 'bin'])
      .pipe(lambda x: x[x['type'] != 'Other'])
      .groupby('type')
)

# Adjust figsize as needed
fig, axes = plt.subplots(3, 2, figsize=(width, height))  
# Flatten the 4x2 grid into a 1D array for easy indexing
axes = axes.flatten()  

# Iterate over each group and create a bar plot
for ax, (group_name, group_df) in zip(axes, grouped):   
    # Compute the average of each Variable (= % Chosen) 
    column_averages = group_df[char_names].mean().sort_values(ascending=False)
    
    # Highlight KY19 Baseline Vars
    bar_colors = ['orange' if col in Chars_Baseline_Names else 'blue' for col in column_averages.index]
    
    # Create the bar plot for the current group
    ax.bar(column_averages.index, column_averages.values, color=bar_colors)
    ax.set_title(f'{group_name}', fontsize=18)
    ax.set_xticks(range(len(column_averages.index)))
    ax.set_xticklabels(column_averages.index, rotation=90, fontsize=12)
    ax.set_xlim(-0.7, len(column_averages) - 0.6)
    ax.tick_params(axis='y', labelsize=12)  
    ax.set_ylabel('Average', fontsize=16)
    
    # Adjust layout to fit all subplots 
    fig.tight_layout()

# Show the multiplot
plt.savefig(path + "/Output/Variable Selection/Plots/" + variable_selection + "_Overview_BarPlot.pdf", dpi = 800, bbox_inches='tight')
plt.show()



#--------------------- Plot over all Groups Weighted Average ------------------
"""
Weighted average is tricky because AUM grows over time deterministically (inflation rate)
so later periods are overweighted
"""

df_weighted_average = (df
                       # Filter out 'Other' type
                       .pipe(lambda x: x[x['type'] != 'Other'])  
                       # Group by 'rdate' and 'type'
                       .groupby(['rdate', 'type'])
                       # Compute the weighted average of each column within group
                       .apply(lambda group: (group[char_names].multiply(group['aum'], axis=0).sum() / group['aum'].sum()))
                       #Having weighted averages at every time point for each type, 
                       #    compute overall TS average by equally weighting each time point
                       .reset_index().set_index('rdate')
                       .groupby('type')
                       .mean()
                       #Get type back as a column
                       .reset_index()
)


# Create a 4x2 grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(width, height)) 
axes = axes.flatten() 

# Iterate over each group and create a bar plot
for i, i_type in enumerate(df_weighted_average['type']):  # Use enumerate to get the index
    ax = axes[i]  # Access the correct subplot axis   
    
    group_name = i_type
    df_type = df_weighted_average.query('type == @i_type').drop('type',axis=1).iloc[0,:].sort_values(ascending = False)
        
    # Highlight KY19 Baseline Vars
    bar_colors = ['orange' if col in Chars_Baseline_Names else 'blue' for col in df_type.index]
    
    # Create the bar plot for the current group
    ax.bar(df_type.index, df_type.values, color=bar_colors)
    ax.set_title(f'{group_name}', fontsize=16)
    ax.set_xticks(range(len(df_type.index)))
    ax.set_xticklabels(df_type.index, rotation=90, fontsize=12)
    ax.set_xlim(-0.7, len(df_type) - 0.6)
    ax.tick_params(axis='y', labelsize=12)  
    ax.set_ylabel('Average', fontsize=14)
    
    # Adjust layout to fit all subplots 
    fig.tight_layout()

# Show the multiplot
plt.savefig(path + "/Output/Variable Selection/Plots/" + variable_selection + "weightedAverage_Overview_BarPlot.pdf", dpi = 800, bbox_inches='tight')
plt.show()

#%% Individual Investor Plot

#Shift Baseline chars to the end for the plot
char_names_plot = copy.deepcopy(char_names[6:] + char_names[0:6])

#Extract Managers
manager_list = [6132]
manager_names = ['AQR']

for mgrno,mgrname in zip(manager_list,manager_names):
    
    #Slice dataframe
    df_mg = df[df['mgrno'] == mgrno]
    
    #Filter Values of the columns    
    filtered_df = df_mg[char_names_plot]!=0
    matrix_data = filtered_df.to_numpy()
    
    # Get the first date of each year
    first_dates = df_mg['rdate']
    
    # Plot the data using imshow
    plt.figure(figsize=(16, 16))  # Adjust figure size if necessary
    plt.imshow(matrix_data, aspect='auto', cmap='Blues', interpolation='nearest')
    
    # Add labels for the x-axis (columns)
    plt.xticks(ticks=np.arange(len(filtered_df.columns)), labels=filtered_df.columns, rotation=90,fontsize=12)  
    
    # Convert y-axis labels to "YYYY-Qq" format
    def format_quarterly(date):
        year = date.year
        quarter = (date.month - 1) // 3 + 1
        return f"{year}-Q{quarter}"
    
    # Apply the formatting to the y-axis labels
    formatted_labels = [format_quarterly(date) for date in first_dates]
    
    # Update the y-axis ticks with formatted labels and increase spacing
    plt.yticks(ticks=np.arange(len(first_dates)), labels=formatted_labels, fontsize=12)  # Adjust fontsize if needed
    
    # Add a vertical line next to the sixth last column
    sixth_last_col_position = len(filtered_df.columns) - 6
    plt.axvline(x=sixth_last_col_position - 0.5, color='red', linestyle='--', linewidth=4)  # Add vertical line (offset by 0.5 for better visibility)
    
    # Display the plot
    plt.tight_layout()  # Adjust layout to ensure everything fits
    plt.savefig(path + "/Output/Variable Selection/Plots/"  + "Investor_" + mgrname 
                + "_" + variable_selection + "_selection.pdf", dpi = 900, bbox_inches='tight')

    plt.show()