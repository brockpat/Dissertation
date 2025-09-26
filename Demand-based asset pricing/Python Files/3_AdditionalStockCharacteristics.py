"""
This file takes in the additional stock characteristics from Chen & Zimmermann (2022), 
drops all variables with minimum coverage less than 70%,
imputes missing values and winsorizes each variable with the thresholds 
that KY19 used to winsorize the data from compustat for the baseline characteristics.

The missing values are imputed in the cross-section by using the cross-sectional
mean. 
 
The output is a csv file with the additional stock characteristic grouped by
permno (stock identifier) and date.

Output File: Additional_Stock_Characteristics_Imputed_Winsorized.csv
"""
#%% Libraries
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

def winsorize_column(column, limits=(0.025, 0.025)):
    return winsorize(column, limits=limits)

#%%Load Data

path = ".../Python Replication Package"

# Load Holdings Data 
df = pd.read_stata(path + "/Data" + "/Data1_clean_correct_bins.dta")

# Keep permno and date so that only the characteristics of the relevant assets are looked at
df = df[['rdate', 'permno']].drop_duplicates()
# -----------------------------------------------------------------------------
#   !!! IMPORTANT !!!
#   Check Date format rdate. It must be end of Quarter Dates. If not, 
#   shift the dates to end of Quarter
#   df['rdate'] = df['rdate'] + pd.offsets.MonthEnd(3)
# -----------------------------------------------------------------------------

# Read in  Additional stock characteristics
characteristics = pd.read_csv(
    path + "/Data" + "/signed_predictors_dl_wide_adj_lagged_non_endog.csv")

characteristics['fdate'] = pd.to_datetime(characteristics['fdate'])

#Exclude Dummy Variables (give rank issues many times) & duplicates (also give rank issues many times)
characteristics.drop(["Spinoff", 'DivInit', 'DivOmit', 'zerotrade','zerotradeAlt12', 'zerotradeAlt1'],axis=1,inplace=True)

#%% Keep Additional Stock Characteristics with good coverage

# Extract relevant assets from additional stock characteristics
df_merge = df.merge(characteristics, how='left', left_on=[
                    'permno', 'rdate'], right_on=['permno', 'fdate'])

#Dropyear 2023 due to poor coverage
df_merge = df_merge[df_merge['rdate'] < '2023']

#Drop unnecessary variables
df_merge.drop(["fdate","Unnamed: 0"],axis=1, inplace=True)


# Compute coverage (share of non-missing values)
coverage = (df_merge
            .groupby('rdate')
            .apply(lambda x: x.count()/x.permno.count())
            )

# Drop variables if minimum coverage >=70% over entire Time Series
threshold = 0.7
min_coverage = coverage.min()
min_coverage = min_coverage[min_coverage > threshold]
df_merge = df_merge[min_coverage.index]
#%% Impute Missing Values
# Fill missing values with group means (means computed per variable per date)
df_filled = (df_merge
             .groupby('rdate')
             .apply(lambda group: group.fillna(group.mean()))
             ).reset_index(drop=True)
#%% Winsorize the remaining data

#Define Columns relevant for winsorizing
Winsor_columns = [item for item in df_filled.columns 
                  if item not in
                  ['rdate', 'permno', "ConvDebt",'DebtIssuance', 'NumEarnIncrease']
                  ]

#Winsorize the columns per year
df_filled_winsor = df_filled.copy()
df_filled_winsor[Winsor_columns] = (df_filled_winsor
                                    .groupby('rdate')[Winsor_columns]
                                    .apply(lambda group: group.apply(winsorize_column))
                                    .to_numpy()
                                    )
#%% Save data
df_filled_winsor.to_csv(path + "/Output/" + "Additional_Stock_Characteristics_Imputed_Winsorized.csv")