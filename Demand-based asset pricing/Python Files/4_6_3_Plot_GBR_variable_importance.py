import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = ".../Python Replication Package"

results = pd.read_csv(path +  "/Output" + "/Variable Selection/GBR/" 
                    +"feature_importance.csv").drop("Unnamed: 0", axis=1)
results["rdate"] = pd.to_datetime(results["rdate"])

results = results.set_index(["rdate", "bin"])
row_sums = results.sum(axis=1)

# Step 2: Divide each element by its respective row sum
results_normalized = results.div(row_sums, axis=0)

# Add investor types
Holdings = pd.read_stata(path + "/Data1_clean_correct_bins.dta")
Holdings = Holdings[["rdate", "bin", "mgrno"]].drop_duplicates()

manager = pd.read_stata("/Users/maxgeilen/Documents/Uni/PhD/Demand Based Asset Pricing/Replication/Koijen Yogo Programs/Stata/Construction/Manager.dta")


results_normalized = results_normalized.reset_index().merge(Holdings, on =["rdate", "bin"], how = "left")
results_normalized = results_normalized.merge(manager[["mgrno", "type"]].drop_duplicates(subset=["mgrno"]), left_on=[ "mgrno"], right_on=["mgrno"], how="left")

results_normalized.type = results_normalized.type.astype(str)
results_normalized.loc[results_normalized.bin==0, "type"] = "Households"

KY_chars = ["LNme", "profit", "Gat", "beta", "LNbe", "divA_be"]
def bar_colors(data, indices):
    return ['orange' if i in indices else 'blue' for i in data]


# Example data
fig, ax = plt.subplots(3, 2,figsize=(20, 16))

i = "Banks"
results_plot =results_normalized[results_normalized.type==i].iloc[:,2:-2].mean().sort_values(ascending=False)
results_plot = results_plot.drop("constant")

y_pos = np.arange(len(results_plot))

ax[0][0].bar(y_pos, results_plot *100,  align='center',  color=bar_colors(results_plot.index, KY_chars)) #xerr=error,
ax[0][0].set_xticks(y_pos, labels=results_plot.index, rotation=90, fontsize=7)
ax[0][0].set_title(i, fontsize=18)
ax[0][0].set_xlim(-0.7, len(results_plot) - 0.6)


i = "Households"
results_plot =results_normalized[results_normalized.type==i].iloc[:,2:-2].mean().sort_values(ascending=False)
results_plot = results_plot.drop("constant")

y_pos = np.arange(len(results_plot))

ax[0][1].bar(y_pos, results_plot *100,  align='center',  color=bar_colors(results_plot.index, KY_chars)) #xerr=error,
ax[0][1].set_xticks(y_pos, labels=results_plot.index, rotation=90, fontsize=7)
ax[0][1].set_title(i, fontsize=18)
ax[0][1].set_xlim(-0.7, len(results_plot) - 0.6)


i = "Insurance companies"
results_plot =results_normalized[results_normalized.type==i].iloc[:,2:-2].mean().sort_values(ascending=False)
results_plot = results_plot.drop("constant")

y_pos = np.arange(len(results_plot))

ax[1][0].bar(y_pos, results_plot *100,  align='center',  color=bar_colors(results_plot.index, KY_chars)) #xerr=error,
ax[1][0].set_xticks(y_pos, labels=results_plot.index, rotation=90, fontsize=7)
ax[1][0].set_title(i, fontsize=18)
ax[1][0].set_xlim(-0.7, len(results_plot) - 0.6)


i = "Investment advisors"
results_plot =results_normalized[results_normalized.type==i].iloc[:,2:-2].mean().sort_values(ascending=False)
results_plot = results_plot.drop("constant")

y_pos = np.arange(len(results_plot))

ax[1][1].bar(y_pos, results_plot *100,  align='center',  color=bar_colors(results_plot.index, KY_chars)) #xerr=error,
ax[1][1].set_xticks(y_pos, labels=results_plot.index, rotation=90, fontsize=7)
ax[1][1].set_title(i, fontsize=18)
ax[1][1].set_xlim(-0.7, len(results_plot) - 0.6)


i = "Mutual funds"
results_plot =results_normalized[results_normalized.type==i].iloc[:,2:-2].mean().sort_values(ascending=False)
results_plot = results_plot.drop("constant")

y_pos = np.arange(len(results_plot))

ax[2][0].bar(y_pos, results_plot *100,  align='center',  color=bar_colors(results_plot.index, KY_chars)) #xerr=error,
ax[2][0].set_xticks(y_pos, labels=results_plot.index, rotation=90, fontsize=7)
ax[2][0].set_title(i, fontsize=18)
ax[2][0].set_xlim(-0.7, len(results_plot) - 0.6)


i = "Pension funds"
results_plot =results_normalized[results_normalized.type==i].iloc[:,2:-2].mean().sort_values(ascending=False)
results_plot = results_plot.drop("constant")

y_pos = np.arange(len(results_plot))

ax[2][1].bar(y_pos, results_plot *100,  align='center',  color=bar_colors(results_plot.index, KY_chars)) #xerr=error,
ax[2][1].set_xticks(y_pos, labels=results_plot.index, rotation=90, fontsize=7)
ax[2][1].set_title(i, fontsize=18)
ax[2][1].set_xlim(-0.7, len(results_plot) - 0.6)

fig.tight_layout()

plt.savefig("GB_Variable_Importance.pdf")
plt.show()

