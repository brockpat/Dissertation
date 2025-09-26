'''
Assign mgrno as bin numbes for investors who can be estimated individually.

You do not need to run this file, as the raw data is not available
in this replication package. We loop through bins to estimate demand which is
why the adjustment in this file is necessary for us .
 
KY19 for the big managers choose a different way of estimating them separately
(see KY19 replication file), so that they do not need to consider this adjustment.
'''

#%% Libraries
import pandas as pd

#%% Read in Raw Data

path = ".../Python Replication Package"

Holdings = pd.read_stata(path + "Data/Data1_clean.dta")
Holdings["rdate"] = Holdings["rdate"] + pd.offsets.MonthEnd(3) #Make dates end of quarter

#%% Correct the bins

#Divide DataFrame into small institutions that remain pooled and large ones that get own bin
Holdings_small = Holdings[Holdings["Nholding"] <1000].copy()
Holdings_large = Holdings[Holdings["Nholding"] >=1000].copy()

#find own bin institutions
own_bins = Holdings_large[["rdate", "mgrno"]].drop_duplicates()
#own_bins["bin"] = own_bins.groupby("rdate").mgrno.cumcount()
own_bins["bin"]  = own_bins["mgrno"]

#find largest original bin per date
#max_bin = Holdings_small.groupby("rdate").bin.max()

# add largest bin to new bins to create new bins
own_bins.set_index("rdate", inplace = True)
#own_bins["bin"] = own_bins["mgrno"]# + max_bin

#merge new bin numbers
Holdings_large = Holdings_large.drop("bin", axis=1)
own_bins = own_bins.reset_index()

Holdings_large = Holdings_large.merge(own_bins, on=["rdate", "mgrno"], how = "left")

#concat both dataframes
Holdings_newBins = pd.concat([Holdings_small, Holdings_large])

#path = "C:/Users/pbrock/Desktop/Clean/Lasso"

#Holdings.to_csv(path + "/Data1_correct_bins.csv")
Holdings_newBins.to_stata(path + "/Data1_clean_correct_bins.dta")