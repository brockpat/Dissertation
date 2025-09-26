# -*- coding: utf-8 -*-
"""
Plot Summary Statistics of the 13F and Chen & Zimmermann (2022) Data.
"""

#%% Libraries
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec, cm
from sklearn.neighbors import KernelDensity
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter

import seaborn as sns


path = ".../Python Replication Package"

#%% Load Data

#Load Holdings Data
Holdings = pd.read_stata(path + "/Data" + "/Data1_clean_correct_bins.dta")
Holdings['rdate'] =  pd.to_datetime(Holdings["rdate"]) #if reading in csv

Holdings = Holdings[Holdings['rdate']!= '2013-03-31']
#IMPORTANT: Check Date format rdate. It must be end of Quarter Dates. If not, 
#           shift the dates to end of Quarter. Sometimes, reading in .dta files
#           can make End of Quarter Dates to Begin of Quarter Dates.
#           df['rdate'] = df['rdate'] + pd.offsets.MonthEnd(3)

Holdings_p = Holdings[Holdings['rweight']>0] #Only positive holdings, i.e. realized demand documented in 13F forms
Holdings_p = Holdings_p[Holdings_p['rdate'].dt.year < 2023]

#%% Heatmap Correlation of Characteristics

#Read in Additional Stock Characteristics
chars = pd.read_csv(path + "/Output" + "/Additional_Stock_Characteristics_Imputed_Winsorized.csv")
chars["rdate"] = pd.to_datetime(chars["rdate"])
chars.drop(['Unnamed: 0'],axis = 1,inplace=True)

#Read in Baseline Stock Characteristics
StocksQ = pd.read_stata(path + "/Data" + "/StocksQ.dta")
StocksQ["date"] = StocksQ["date"] - pd.offsets.MonthBegin()+pd.offsets.MonthEnd()


#Construct Additional Variable Names
Characteristics_Names = list(chars.columns.drop(['rdate','permno']))
Characteristics_Names = [var for var in Characteristics_Names if var not in ["Spinoff", 'DivInit', 'DivOmit',
                                                                             'zerotrade','zerotradeAlt12', 'zerotradeAlt1']]
#Construct Baseline Variable Names
Baseline_endog_Variables_Names = ['LNme','LNbe', 'profit', 'Gat', 'divA_be','beta']

#Construct Dataset of all Characteristics
chars = chars.merge(StocksQ[["permno", "date"] + Baseline_endog_Variables_Names], left_on=["rdate", "permno"], right_on=["date", "permno"])
chars.drop(['date'],axis = 1, inplace=True)

#Standardise Variables so Regression Coefficients are Correlation coefficients
chars[Baseline_endog_Variables_Names + Characteristics_Names] = (chars[Baseline_endog_Variables_Names + Characteristics_Names] - chars[Baseline_endog_Variables_Names + Characteristics_Names].mean()) / chars[Baseline_endog_Variables_Names + Characteristics_Names].std()

char_vals = chars[Baseline_endog_Variables_Names + Characteristics_Names].values
char_vals = char_vals[~np.isnan(char_vals).any(axis=1)]

# Initialize matrices for correlations and standard errors
correlation_df = chars[Baseline_endog_Variables_Names + Characteristics_Names].corr()

#----- Plot

# Set up the matplotlib figure
plt.figure(figsize=(14, 14))  # Increase figure size

# Create a heatmap with adjusted font size and hidden x-axis labels
sns.heatmap(
    correlation_df, 
    annot=True, 
    cmap='coolwarm', 
    center=0, 
    square=True,
    linewidths=0.5, 
    fmt=".2f", 
    cbar_kws={"shrink": 0.8},
    yticklabels=correlation_df.columns,
    xticklabels=correlation_df.columns,  
    annot_kws={"size": 0} 
)

# Set font size for the axis labels
plt.xticks(rotation=90, fontsize=12)  
plt.yticks(fontsize=12)               

# Show the plot
#plt.title("Correlation Matrix Heatmap")
plt.savefig(path + "/Output/Plots/" + "correlation_matrix_heatmap.pdf", dpi=600, bbox_inches='tight')

#%% Time Series observations per year

df = Holdings_p.groupby('rdate').size()
df = df.reset_index()

# Define a formatter function
def format_ticks(value, _):
    return f'{value:,.0f}'  # Formats the value with commas

#------ Plot Time Series of Observations
plt.figure(figsize=(10,6))
plt.plot(df['rdate'], df.iloc[:,1], color='blue')

# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(df['rdate'].min(), df['rdate'].max())

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90, fontsize=14)

# Format y-axis ticks
ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))
plt.yticks(fontsize=14)

# Add labels and title
#plt.title('Time Series Observations per Quarter.')
plt.grid(True)
#plt.legend()
plt.savefig(path + "/Output" + "/Plots/Summary_Stats" +"/TS_ObsPerQuarter.pdf", dpi=600, bbox_inches='tight')
plt.show() 

#%% Time Series Manager per Quarter
df = Holdings_p.drop_duplicates(['rdate','mgrno']).groupby(['rdate']).size()
df = df.reset_index()

#------ Plot Time Series 
plt.figure(figsize=(10,6))
plt.plot(df['rdate'], df.iloc[:,1], color='blue')

# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(df['rdate'].min(), df['rdate'].max())

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90, fontsize=14)

# Format y-axis ticks
ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))
plt.yticks(fontsize=14)

# Add labels and title
#plt.title('Time Series Manager per Quarter.')
plt.grid(True)
#plt.legend()
plt.savefig(path + "/Output" + "/Plots/Summary_Stats" +"/TS_ManagerPerQuarter.pdf", dpi=600, bbox_inches='tight')
plt.show() 

#%% Histogram over Time for Number of Holdings

df = Holdings_p.drop_duplicates(['rdate','mgrno'])
df = df[['rdate','Nholding']]
df = df[df['rdate'].dt.year < 2023]

# Create a new column for years
df['rdate'] = df['rdate'].dt.to_period('Y')

#Filter the DataFrame so one can actually see something in the plot
df = df[df['Nholding']<150]

# List of unique years and colors for each quarter
years = [str(x) for x in df['rdate'].unique()]

cmap = cm.get_cmap('plasma', len(years))
colors = [cmap(i) for i in range(len(years))]
#colors = ["FFA82E","F6A335","EE9E3B","E59942","DC9449","D4904F","CB8B56","C2865D","B98163","B17C6A","A87771","9F7277","976D7E","8E6884","85638B","7D5F92","745A98","6B559F","6250A6","5A4BAC","5146B3"]
#colors = ["#" + item for item in colors]

# Initialize grid spec and figure
gs = gridspec.GridSpec(len(years), 1)
fig = plt.figure(figsize=(16, 9))

#https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/

ax_objs = []

for i, quarter in enumerate(years):
    # Filter 'Nholding' values for the current quarter
    x = np.array(df[df['rdate'] == quarter].Nholding)
    x_d = np.linspace(min(x), max(x), 1000)

    # Kernel Density Estimation
    kde = KernelDensity(bandwidth=3, kernel='gaussian')
    kde.fit(x[:, None])
    logprob = kde.score_samples(x_d[:, None])

    # Create subplot for each quarter
    ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))

    # Plot KDE distribution
    ax_objs[-1].plot(x_d, np.exp(logprob), color="#f0f0f0", lw=1)
    ax_objs[-1].fill_between(x_d, np.exp(logprob), alpha=1, color=colors[i % len(colors)])

    # Set x and y limits
    ax_objs[-1].set_xlim(min(x), max(x))
    ax_objs[-1].set_ylim(0, np.max(np.exp(logprob)) * 1.1)

    # Transparent background
    rect = ax_objs[-1].patch
    rect.set_alpha(0)

    # Remove y-axis labels
    ax_objs[-1].set_yticklabels([])
    
    # Remove y-axis tick marks while keeping the custom year label
    ax_objs[-1].set_yticks([])
    
    ax_objs[-1].tick_params(axis='x', length=0)


    # Only set x-axis label for the last plot
    if i == len(years) - 1:
        ax_objs[-1].set_xlabel("", fontsize=16, fontweight="bold")
        # Disable vertical grid lines for x-ticks
        ax_objs[-1].grid(False, axis='x')
        ax_objs[-1].tick_params(axis='x', length=0)
    else:
        ax_objs[-1].set_xticklabels([])
        # Disable vertical grid lines for x-ticks
        ax_objs[-1].grid(False, axis='x')
        ax_objs[-1].tick_params(axis='x', length=0)

    # Remove plot borders
    spines = ["top", "right", "left", "bottom"]
    for s in spines:
        ax_objs[-1].spines[s].set_visible(False)

    # Display the quarter name on the left of each plot
    adj_quarter = quarter.replace(" ", "\n")
    ax_objs[-1].text(-0.02, 0, adj_quarter, fontweight="bold", fontsize=14, ha="right")
    
for ax in ax_objs:
    ax.tick_params(axis='x', labelsize=16)  # Adjust size as needed

# Disable vertical grid lines for x-ticks
ax_objs[-1].grid(False, axis='x')

# Adjust space between plots
gs.update(hspace=-0.7)

# Add title
#fig.text(0.07, 0.85, "Distribution of 'Nholding' by Year", fontsize=20)

#plt.tight_layout()
plt.savefig(path + "/Output" + "/Plots/Summary_Stats" +"/Hist_Nholdings.pdf", dpi=800, bbox_inches='tight')
plt.show()
#%% TS of AUM of 13F investors

df = Holdings_p.drop_duplicates(['rdate','mgrno'])[['rdate','mgrno','aum']]
df = df[df['mgrno'] > 0]

df_grouped = df.groupby('rdate')['aum'].agg(
    aum_mean=('mean'),
    aum_quantile_0=lambda x: x.quantile(0.0),
    aum_quantile_90=lambda x: x.quantile(0.9)
).reset_index()

# Define a formatter function
def format_ticks(value, _):
    return f'{value:,.0f}'  # Formats the value with commas

# Plotting
plt.figure(figsize=(10, 6))

# Plot Big Investors AUM Mean
plt.plot(df_grouped['rdate'], df_grouped['aum_mean'], color='blue')

# Shade the area between the 25% quantile and the 75% quantile
plt.fill_between(df_grouped['rdate'], df_grouped['aum_quantile_0'], df_grouped['aum_quantile_90'], color='blue', alpha=0.3)

# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks every 4 months
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(df_grouped['rdate'].min(), df_grouped['rdate'].max())  # Set x-axis limits

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90, fontsize=14)

# Format y-axis ticks
ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))
plt.yticks(fontsize=14)

# Add labels and legend
plt.ylabel('AUM in Millions', fontsize=14)
#plt.legend(fontsize=12)

# Save Plot
plt.savefig(path + "/Output/Plots/Summary_Stats/TS_AUM.pdf", dpi=600, bbox_inches='tight')

# Display the plot
plt.show()
#%% TS of Number of Stocks per Year

df = Holdings_p.drop_duplicates(['rdate','permno']).groupby('rdate').size()
df = df.reset_index()

# Plotting
plt.figure(figsize=(10,6))

#Plot TS
plt.plot(df['rdate'], df.iloc[:,1], color='blue')

# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(df['rdate'].min(), df['rdate'].max())

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90, fontsize=14)

# Format y-axis ticks
ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))
plt.yticks(fontsize=14)

# Add labels and title
plt.ylabel('')
#plt.title('Time Series of Number of Stocks per Quarter')
plt.grid(True)
#plt.legend()
#Save Plot
plt.savefig(path + "/Output" + "/Plots/Summary_Stats" +"/TS_NumberStocksperQuarter.pdf", dpi=600, bbox_inches='tight')
# Display the plot
plt.show()
#%% Plot Share of zeros per year
#Load Holdings Data

# Group by 'rdate' and count 'rweight' conditions
df = Holdings.groupby('rdate').agg(
    rweight_zero_count=('rweight', lambda x: (x == 0).sum()),
    rweight_positive_count=('rweight', lambda x: (x > 0).sum())
).reset_index()

df['share_zeros'] = df['rweight_zero_count'] / df['rweight_positive_count']

# Plotting
plt.figure(figsize=(10,6))

#Plot TS
plt.plot(df['rdate'], df['share_zeros'], color='blue')

# Set xticks for every year
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(mdates.MonthLocator(4))  # Set major ticks to April (Q2) of each year
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to only show the year
plt.xlim(df['rdate'].min(), df['rdate'].max())

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90, fontsize=14)

# Format y-axis ticks
plt.yticks(fontsize=14)

# Add labels and title
plt.ylabel('')
#plt.title('Time Series of Share Zeros')
plt.grid(True)

#plt.legend()
#Save Plot
plt.savefig(path + "/Output" + "/Plots/Summary_Stats" +"/TS_ShareZeros.pdf", dpi=600, bbox_inches='tight')#, bbox_inches='tight')  # Save with 300 DPI

plt.show()
#%% Household Sector

#Extract DataSet
df = copy.deepcopy(Holdings_p)

#Change Type Variable to Households & 13F Investors
df['type'] = df['type'].astype(str)
df.loc[df['type'] != 'Households', 'type'] = '13F'

#---------- Compute Ratio of AUM of 13F Investors to AUM of Households -------#

#Compute Aggregate AUM per Type for each date
df_aum_type = (df
               #AUM is a repeated entry for each holding, so avoid oversumming
               .drop_duplicates(subset = ['rdate','mgrno'])
               #Grouping
               .groupby(['rdate','type'])['aum']
               #Sum AUM
               .sum()
               #Reste grouping multiindex
               .reset_index()
               #So order is always the same when computing the Ratio
               .sort_values(by = ['rdate','type'])
               )
#Create DataFrame to compute the ratio of AUM
def calculate_ratio(df):
    if len(df) != 2:
        raise ValueError(f"Expected exactly 2 'aum' values for date {df.name}, but found {len(df)}")
    return df.iloc[0] / df.iloc[1]  # Ratio of first to second 'aum', i.e. 13F/Households

# Compute Ratios for each date (make sure dataframe was sorted) --> Share AUM 13F of Households
df_ratios_aum_type = df_aum_type.groupby('rdate')['aum'].apply(calculate_ratio).reset_index()

"""
# ---- Plot the histogram
data = df_ratios_aum_type['aum']

# Compute histogram counts and bin edges
counts, bin_edges = np.histogram(data, bins=10)

# Calculate relative frequencies
relative_frequencies = counts / len(data)

# Plot the relative frequency histogram
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], relative_frequencies, width=np.diff(bin_edges), edgecolor='black', align='edge', alpha=0.7)

# Customize the plot
plt.xlabel('Ratio of 13F to Households', fontsize=14)
plt.ylabel('Relative Frequency', fontsize=14)
plt.grid(True, which="both", ls="--", linewidth=0.5)

#Save Plot
plt.savefig(path + "/Output" + "/Plots/Summary_Stats" +"/Households_Ratio_Overall.pdf", dpi=600, bbox_inches='tight')#, bbox_inches='tight')  # Save with 300 DPI

# Show the plot
plt.show()
"""

#------------------------------------------------------------------------------
# Compute Ratio of AUM of 13F Investors over AUM of Households for each stock 
#       over the entire Time Series
#------------------------------------------------------------------------------

#Compute AUM invested into the stock
df['aum_permno'] = df['weight'] * df['aum']

#Compute Aggregate AUM invested per stock per type and average results over time.
df_aum_permno = (df
                 #Aggregate Aum invested in stock per type
                 .groupby(['rdate','permno','type'])['aum_permno']
                 .sum()
                 .reset_index()
                 #Average Aum invested in the stock per type
                 .groupby(['permno','type'])['aum_permno']
                 .mean()
                 .reset_index()
                 )

# ---- Compute the Ratios

#Filter Out permnos that are only held by 13F Investors (these are only very few)
print("Number of Stocks only held by 13F investors over the entire Time Series = " + str((df_aum_permno['permno'].value_counts() < 2).sum()))
df_aum_permno = df_aum_permno[df_aum_permno.groupby('permno')['permno'].transform('size') == 2]

df_ratios_aum_permno = (df_aum_permno
                        #Sort Values so Ratio always computed in the right order
                        .sort_values(by = ['permno','type'])
                        .groupby('permno')['aum_permno']
                        #Compute Ratio of AUMs
                        .apply(calculate_ratio).reset_index()
                        )
"""
# ---- Plot the Histogram

# Create the histogram
data = df_ratios_aum_permno['aum_permno']

# Generate exponential bin edges
# Start slightly above 0 to avoid log(0), and end at the maximum value
min_value = data[data > 0].min()  # Smallest non-zero value
max_value = data.max()  # Maximum value
bins = np.logspace(np.log10(min_value), np.log10(max_value), num=80)

# Compute histogram counts and bin edges
counts, bin_edges = np.histogram(data, bins=bins)

# Calculate relative frequencies
relative_frequencies = counts / len(data)

# Plot the relative frequency histogram
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], relative_frequencies, width=np.diff(bin_edges), edgecolor='black', align='edge', alpha=0.7)

# Customize the plot
plt.xscale('log')  # Use a logarithmic x-axis
plt.xlabel('Ratio of 13F to Households (log scale)', fontsize=14)
plt.ylabel('Relative Frequency', fontsize=14)
plt.grid(True, which="both", ls="--", linewidth=1)

# Manually set x-ticks with exponents increasing by one unit
tick_locations = [10**i for i in [-5,-4,-3,-2,-1,0,0.3,1,2]]  # 10^{-5}, 10^{-4}, ..., 10^{2}
tick_labels = [f"$10^{{{i}}}$" for i in [-5,-4,-3,-2,-1,0,0.3,1,2]]  # Format as LaTeX-style labels
plt.xticks(tick_locations, tick_labels,rotation = 90)  # Rotate tick labels for better readability

# Adjust x-axis limits to remove excess white space
plt.xlim(left=bin_edges[20], right=bin_edges[-12])  # Set limits to match bin edges

plt.savefig(path + "/Output" + "/Plots/Summary_Stats" +"/Households_Ratio_Permnos.pdf", dpi=600, bbox_inches='tight')
# Show the plot
plt.show()
"""


# ---- Plot the Histogram

# Create a 2x1 multiplot
fig, axes = plt.subplots(2, 1, figsize=(8, 10))  # 2 rows, 1 column
#plt.subplots_adjust(hspace=0.4)  # Adjust vertical spacing between subplots

# ---- First Subplot: Histogram for df_ratios_aum_type ----
data1 = df_ratios_aum_type['aum']

# Compute histogram counts and bin edges
counts1, bin_edges1 = np.histogram(data1, bins=10)

# Calculate relative frequencies
relative_frequencies1 = counts1 / len(data1)

# Plot the relative frequency histogram
axes[0].bar(bin_edges1[:-1], relative_frequencies1, width=np.diff(bin_edges1), edgecolor='black', align='edge', alpha=0.7)

# Customize the plot
#axes[0].set_xlabel('Ratio of 13F to Households', fontsize=14)
axes[0].set_ylabel('Relative Frequency', fontsize=12)
axes[0].grid(True, which="both", ls="--", linewidth=0.5)
axes[0].set_title('Overall', fontsize=12)

# ---- Second Subplot: Histogram for df_ratios_aum_permno ----
data2 = df_ratios_aum_permno['aum_permno']

# Generate exponential bin edges
min_value = data2[data2 > 0].min()  # Smallest non-zero value
max_value = data2.max()  # Maximum value
bins2 = np.logspace(np.log10(min_value), np.log10(max_value), num=80)

# Compute histogram counts and bin edges
counts2, bin_edges2 = np.histogram(data2, bins=bins2)

# Calculate relative frequencies
relative_frequencies2 = counts2 / len(data2)

# Plot the relative frequency histogram
axes[1].bar(bin_edges2[:-1], relative_frequencies2, width=np.diff(bin_edges2), edgecolor='black', align='edge', alpha=0.7)

# Customize the plot
axes[1].set_xscale('log')  # Use a logarithmic x-axis
axes[1].set_xlabel('Ratio of 13F AUM to Household AUM (log scale)', fontsize=12)
axes[1].set_ylabel('Relative Frequency', fontsize=12)
axes[1].grid(True, which="both", ls="--", linewidth=1)
axes[1].set_title('By Stock', fontsize=12)

# Manually set x-ticks with exponents increasing by one unit
tick_locations = [10**i for i in [-5, -4, -3, -2, -1, 0, 0.3, 1, 2]]  # 10^{-5}, 10^{-4}, ..., 10^{2}
tick_labels = [f"$10^{{{i}}}$" for i in [-5, -4, -3, -2, -1, 0, 0.3, 1, 2]]  # Format as LaTeX-style labels
axes[1].set_xticks(tick_locations)
axes[1].set_xticklabels(tick_labels, rotation=90)  # Rotate tick labels for better readability

# Adjust x-axis limits to remove excess white space
axes[1].set_xlim(left=bin_edges2[20], right=bin_edges2[-12])  # Set limits to match bin edges

# Save the multiplot
plt.savefig(path + "/Output/Plots/Summary_Stats/Households_Ratio.pdf", dpi=600, bbox_inches='tight')

# Show the plot
plt.show()