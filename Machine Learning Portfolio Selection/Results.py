# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 16:19:23 2026

@author: patri
"""

#%% Libraries

path = "C:/Users/patri/Desktop/ML/"

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter

import pickle
import sqlite3

import statsmodels.api as sm
import statsmodels.formula.api as smf

#%% Generals

# Trading Start & End
trading_start, trading_end = pd.to_datetime("2004-01-31"), pd.to_datetime("2024-12-31")

# DataBases
JKP_Factors = sqlite3.connect(database = path + "Data/JKP_processed.db")
SP500_Constituents = sqlite3.connect(database = path + "Data/SP500_Constituents.db")
Benchmarks = sqlite3.connect(database = path + "Data/Benchmarks.db")
Models = sqlite3.connect(database = path + "Data/Predictions.db")

# Realised Returns & ME
df = pd.read_sql_query(("SELECT id, eom, me, tr_ld1 FROM Factors_processed "
                        f"WHERE eom >= '{(trading_start- pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')}' "
                        f"AND eom <= '{trading_end.strftime('%Y-%m-%d')}'"
                        ),
                       con = JKP_Factors,
                       parse_dates = {'eom'}).sort_values(by = ['eom','id'])

df_returns = df[['id','eom','tr_ld1']]
df_me = df[['id','eom','me']]
del df

# Names
df_names = pd.read_csv(path + "Data/CRSP_names.csv").drop_duplicates(subset = 'PERMNO')

# S&P500 Constituents
df_sp500_ids = (pd.read_sql_query("SELECT * FROM SP500_Constituents_monthly", #" WHERE eom >= '{start_date}'",
                                       con = SP500_Constituents,
                                       parse_dates = {'eom'})
                      .rename(columns = {'PERMNO': 'id'})
                      )

# SP500 ETF Benchmark
df_spy = pd.read_sql_query("SELECT * FROM SPY",
                           parse_dates = {'eom'},
                           con = Benchmarks)
df_spy = (df_spy[df_spy['eom'].between(trading_start, trading_end)]
              .assign(cumulative_return = lambda df: (1.0 + df['ret']).cumprod()
                      )
              ).reset_index(drop = True)

# risk-free rate
risk_free = (pd.read_csv(path + "Data/FF_RF_monthly.csv", usecols=["yyyymm", "RF"])
             .assign(rf = lambda df: df["RF"]/100)
             .assign(eom = lambda df: pd.to_datetime(df["yyyymm"].astype(str) + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0))
             .get(['eom','rf'])
             )

#%% Functions Strings

def value_to_token(v):
    """Convert Python value to a compact string token for filenames."""
    if isinstance(v, bool):
        return str(v).lower()          # True -> "true", False -> "false"
    if v is None:
        return "None"
    if isinstance(v, float):
        # Example: 0.15 -> "015", 1.0 -> "10"
        s = f"{v:.3f}".rstrip("0").rstrip(".")  # "0.15"
        return s.replace(".", "")               # "015"
    return str(v)

def settings_string(settings: dict) -> str:
    # Preserve insertion order (Python 3.7+)
    parts = [f"{k}={value_to_token(v)}" for k, v in settings.items()]
    return "_".join(parts)

def plot_string(settings: dict) -> str:
    # Preserve insertion order (Python 3.7+)
    parts = [f"{k}={value_to_token(v)}" for k, v in settings.items() 
             if k in ['flatMaxPiVal', 'volScaler', 'tcScaler']]
    return "_".join(parts)

#%% Functions Performance Metrics

def SharpeRatio(df, risk_free, return_col):
    """
    Computes the annualised Sharpe Ratio of a time series of monthly returns.
    """
    df = (df
          .merge(risk_free, on = 'eom', how = 'left')
          .assign(ret_exc = lambda df: df[return_col] - df['rf'])
          )

    #Sharpe Ratio Strategy
    mu, sigma = df['ret_exc'].mean(), df['ret_exc'].std() 
    Sharpe = np.sqrt(12) * (mu / sigma)
    
    return 12*mu, np.sqrt(12)*sigma, Sharpe

def InformationRatio(strategy, benchmark, return_col_strat, return_col_bench, risk_free):
    
    #Merge risk-free rate
    strategy = (strategy
          .merge(risk_free, on = 'eom', how = 'left')
          .assign(ret_exc = lambda df: df[return_col_strat] - df['rf'])
          )
    
    benchmark = (benchmark
          .merge(risk_free, on = 'eom', how = 'left')
          .assign(ret_exc = lambda df: df[return_col_bench] - df['rf'])
          )
    
    information_ratio = np.sqrt(12) * np.mean(strategy['ret_exc'] - benchmark['ret_exc'])/((strategy['ret_exc'] - benchmark['ret_exc']).std())
    
    return information_ratio

def MaxDrawdown(strategy, benchmark, return_col_strat, return_col_bench):
    """Calculates Maximum Drawdown from a return series."""
    # Convert returns to a cumulative wealth index
    comp_ret = (1 + strategy[return_col_strat]).cumprod()
    # Calculate the running maximum
    peaks = comp_ret.expanding(min_periods=1).max()
    # Calculate drawdown relative to the peak and return the minimum (most negative) value
    drawdown_strat = ((comp_ret / peaks) - 1).min()
    
    # Convert returns to a cumulative wealth index
    comp_ret = (1 + benchmark[return_col_bench]).cumprod()
    # Calculate the running maximum
    peaks = comp_ret.expanding(min_periods=1).max()
    # Calculate drawdown relative to the peak and return the minimum (most negative) value
    drawdown_bench = ((comp_ret / peaks) - 1).min()
    
    return drawdown_strat, drawdown_bench, (drawdown_strat - drawdown_bench)

def CaptureRatio(strategy, benchmark, return_col_strat, return_col_bench):
    """
    Calculates the Geometric Upside and Downside Capture Ratios.
    This is preferred over arithmetic mean for accuracy since going down 5% and
    back up 5% doesn't lead to being at the same level.
    
    If a fund has a downside capture ratio of 80%, then, during a period when 
    the market dropped 10%, the fund only lost 8%
    """
    
    # 1. Identify Downside Months (Benchmark < 0)
    down_mask = benchmark[return_col_bench] < 0
    strat_down = strategy[return_col_strat][down_mask]
    bench_down = benchmark[return_col_bench][down_mask]
    
    # 2. Identify Upside Months (Benchmark > 0)
    up_mask = benchmark[return_col_bench] > 0
    strat_up = strategy[return_col_strat][up_mask]
    bench_up = benchmark[return_col_bench][up_mask]

    # Helper function for Geometric Mean Return
    def geometric_mean(returns):
        # (Product of (1+r))^(1/n) - 1
        if len(returns) == 0: return np.nan
        compounded = np.prod(1 + returns)
        return compounded**(1 / len(returns)) - 1

    # 3. Calculate Geometric Means
    geo_avg_strat_down = geometric_mean(strat_down)
    geo_avg_bench_down = geometric_mean(bench_down)
    
    geo_avg_strat_up = geometric_mean(strat_up)
    geo_avg_bench_up = geometric_mean(bench_up)
    
    # 4. Calculate Means
    avg_strat_down = strat_down.mean()
    avg_bench_down = bench_down.mean()
    
    avg_strat_up = strat_up.mean()
    avg_bench_up = bench_up.mean()
    
    # 5. Calculate Ratios
    geo_downside_capture = geo_avg_strat_down / geo_avg_bench_down
    geo_upside_capture = geo_avg_strat_up / geo_avg_bench_up
    
    downside_capture = avg_strat_down / avg_bench_down
    upside_capture   = avg_strat_up / avg_bench_up
    
    return geo_downside_capture, geo_upside_capture, downside_capture, upside_capture

def calculate_alpha_beta(strategy, benchmark, return_col_strat, return_col_bench, risk_free):
    """Calculates Annualized Alpha and the Beta, setting alpha to 0 if not significant."""
    strategy = strategy.merge(risk_free, on = ['eom'], how = 'left').sort_values(by = 'eom')
    benchmark = benchmark.merge(risk_free, on = ['eom'], how = 'left').sort_values(by = 'eom')
    benchmark = benchmark.loc[benchmark['eom'].isin(strategy['eom'])]
                 
    y = strategy[return_col_strat] #- strategy['rf']
    x = benchmark[return_col_bench] #- benchmark['rf']
    
    # Linear Regression: y = alpha + beta * x
    # Note: linregress p_value is for the slope (beta). 
    # To get p_value for intercept (alpha), we use statsmodels or check significance manually.
    
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    
    alpha_monthly = model.params['const']
    beta = model.params.iloc[1] # The slope
    p_value_alpha = model.pvalues['const'] # The p-value for the intercept
    print(p_value_alpha)
    
    # Set alpha to zero if p-value is greater than 0.05
    #if p_value_alpha > 0.05:
    #    alpha_monthly = 0.0
    
    # Annualize Alpha
    alpha_annualized = alpha_monthly * 12 
    
    return alpha_annualized, beta
                                        
def Turnover(strategy, aum):
    # 1. Compute per-asset absolute weight changes
    changes = (
        strategy
        .loc[:, ['eom', 'pi', 'pi_g_tm1']]
        .assign(abs_weight_change=lambda df: (df['pi'] - df['pi_g_tm1']).abs())
    )
    
    # 2. Aggregate to monthly gross turnover
    monthly_gross_turnover = (
        changes
        .groupby('eom', as_index=False)['abs_weight_change']
        .sum()
        .rename(columns={'abs_weight_change': 'gross_turnover'})
    )
    
    # 3. Compute AUM weights across time
    aum_weights = (
        aum
        .loc[:, ['eom', 'wealth']]
        .assign(aum_weight=lambda df: df['wealth'] / df['wealth'].sum())
    )
    
    # 4. AUM-weighted average turnover
    aum_weighted_turnover = (
        monthly_gross_turnover
        .merge(aum_weights, on='eom', how='left')
        .assign(weighted_turnover=lambda df: df['gross_turnover'] * df['aum_weight'])
        ['weighted_turnover']
        .sum()
    )      

    return monthly_gross_turnover['gross_turnover'].mean(), aum_weighted_turnover

#%% Long-Short Type Portfolios

def long_short_portfolio(df_return_predictions, prediction_col,  # Predicted Returns
                         df_returns, # Realised Returns
                         df_me, # Market Equity for Value Weighting
                         long_cutoff = 0.9, short_cutoff = 0.1,
                         value_weighted = False,
                         long_only = False,
                         ):
    
    #===========================
    # Select Long & Short Stocks
    #===========================
    
    #Cross-Sectional Quantile cutoffs
    grouped = df_return_predictions.groupby('eom')[prediction_col]
    q_long = grouped.transform('quantile', long_cutoff)
    q_short= grouped.transform('quantile', short_cutoff)
    
    #---- Determine Positions ----
    #Conditions determining Long or Short
    conditions = [df_return_predictions[prediction_col] >=q_long,
                  df_return_predictions[prediction_col] <= q_short]
    
    #Numerical Position Value
    position = [1,-1]

    #Generating dataframe
    df_ls = df_return_predictions[['id','eom']]
    df_ls['position'] = np.select(conditions, position, default = 0)
    
    #Filter out zero positions
    df_ls = df_ls.loc[df_ls['position'] != 0]
    
    #===========================
    # Compute Portfolio weight
    #===========================
    
    if value_weighted:
        # Merge Market Equity
        df_ls = df_ls.merge(df_me, on = ['id','eom'], how = 'left')
        
        # Error Handling
        if df_ls['me'].isna().sum() > 0:
            print("ERROR: Missing market equity")
            
            return None, None
            
        #Compute weights
        df_ls = df_ls.assign(weight = lambda df:
                             df['me']/df.groupby(['eom','position'])['me'].transform('sum'))
    
    else: #equal weighted
        df_ls = df_ls.assign(weight = lambda df: 1/df.groupby(['eom','position'])['id'].transform('count'))

    # If Long only
    if long_only:
        df_ls = df_ls.loc[df_ls['position']>0]

    #==============================
    #  Compute Individual Revenue
    #==============================
    
    #Merge realised Return
    df_ls = df_ls.merge(df_returns[['id','eom','tr_ld1']], on = ['id','eom'],
                        how = 'left')
    
    #Compute revenue
    df_ls['ret'] = df_ls['position']*df_ls['weight']*df_ls['tr_ld1']
    
    #==============================
    #  Compute Aggregated Revenue
    #==============================
    
    df_profit = (df_ls
                 .groupby('eom')['ret'].sum()
                 .reset_index()
                 .rename(columns = {0: 'ret'})
                 )
    df_profit['cumret'] = (1 + df_profit['ret']).cumprod()

    
    return df_ls, df_profit

# ---- Predictor ----
predictors = ['XGBReg',
             'TransformerSet_Dropout010',
             'IPCA',
             'RFF']

# ---- Label ----
labels = ['XGBoost', 
          'Transformer',
          'IPCA',
          'RFF']

# ---- Target Type ----
target_col = ['LevelTrMsp500Target', 
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              'LevelTrMsp500Target']

# ---- Est. Universe ----
est_univs = ['SP500UniverseFL',
             'SP500UniverseFL',
             'CRSPUniverse',
             'SP500UniverseFL']

# ---- Input Features ----
input_feat = ['RankFeatures',
              'RankFeatures',
              'ZscoreFeatures',
              'ZscoreFeatures']

# ---- HP Tuning ----
hp_tuning = "RollingWindow_win120_val12_test12"

# ---- Load Return Predictions ----
predictions = []
for i, predictor in enumerate(predictors):
    # Recall that at date 'eom' the prediction is for 'eom' + 1 
    df = pd.read_sql_query(("SELECT * "
                            f"FROM {predictor}_{target_col[i]}_{est_univs[i]}_{input_feat[i]}_{hp_tuning} "
                            f"WHERE eom >= '{(trading_start - pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')}' AND eom <= '{trading_end.strftime('%Y-%m-%d')}'"
                           ),
                           con= Models,
                           parse_dates = {'eom'}
                           )
    
    # Select SP500 Subset
    df = df.merge(df_sp500_ids.assign(in_sp500 = True), on = ['id', 'eom'], how = 'left')
    df = df.dropna().drop(columns = 'in_sp500')
    
    predictions.append([df, labels[i]])
    
# ---- Compute Long-Short Portfolios ----
all_strategy_profits = []
sharpe_results = []

for df_pred, label in predictions:
    # 1. Identify the prediction column dynamically
    # It's the column that isn't 'id' or 'eom'
    pred_col = [col for col in df_pred.columns if col not in ['id', 'eom']][0]
    
    print(f"Processing {label} using column: {pred_col}")
    
    # Long-Short Portfolio
    df_ls, df_profit = long_short_portfolio(
        df_return_predictions=df_pred,
        prediction_col=pred_col,
        df_returns=df_returns, 
        df_me=df_me,
        long_cutoff=0.9, 
        short_cutoff=0.1,
        value_weighted=False, 
        long_only = False
    )
    
    # Add the label to the profit dataframe for identification
    df_profit['model'] = label
    all_strategy_profits.append([df_ls,df_profit])
    
    # Sharpe Ratio
    mu, sigma, sharpe = SharpeRatio(df_profit, risk_free, 'ret')
    
    sharpe_results.append({
        'Model': label,
        'mu': mu,
        'sigma': sigma,
        'Sharpe': sharpe
    })
    
df_metrics = (pd.DataFrame(sharpe_results)
              .rename(columns = {'mu':'$\mu$', 'sigma':'$\sigma$'})
              )

print(df_metrics.to_latex(index=False, escape=False, float_format="%.3f"))

#%% DolVol Portfolio

# =========================================
# Long-Short & Top 10 Long DolVol Portfolio
# =========================================

# Load Kyles Lambda
df_dolvol = pd.read_sql_query(("SELECT id, eom, lambda, dolvol_126d, in_sp500 FROM Factors_processed "
                            f"WHERE eom >= '{(trading_start - pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')}' AND eom <= '{trading_end.strftime('%Y-%m-%d')}'"
                           ),
                          con = JKP_Factors,
                          parse_dates = {'eom'})
# Restrict to S&P 500 Universe
df_dolvol = (df_dolvol
             .loc[df_dolvol['in_sp500'] == 1]
             .sort_values(by = ['eom','id'])
             .drop(columns = ['in_sp500'])
             )

# Get Next period's return
df_ls_dolvol, df_profit_dolvol = long_short_portfolio(df_dolvol, 'dolvol_126d',  # Predicted dollar volume
                         df_returns, # Realised Returns
                         df_me, # Market Equity for Value Weighting
                         long_cutoff = 0.9, short_cutoff = 0.1,
                         value_weighted = False,
                         long_only = False,
                         )

mu, sigma, sharpe = SharpeRatio(df_profit_dolvol, risk_free, 'ret')
sharpe_results = {
    'Model': "DolVol",
    'mu': mu,
    'sigma': sigma,
    'Sharpe': sharpe
}
    
df_metrics = (pd.DataFrame(sharpe_results, index=[0])
              .rename(columns = {'mu':'$\mu$', 'sigma':'$\sigma$'})
              )

print(df_metrics.to_latex(index=False, escape=False, float_format="%.3f"))

# ===============================================
# Correlation of DolVol Top 10 & Strategy Returns
# ===============================================

final_results = {}
for tc_val in [1.0, 0.5, 0.1, 0.01]:
    
    # container for this tc
    col_results = {}

    # Settings 
    run_settings = dict(includeRF    = False,
                        flatMaxPi    = True,
                        flatMaxPiVal = 0.15,
                        Wmax         = None,
                        Wmin         = None,
                        volScaler    = 1.0, 
                        tcScaler     = tc_val, 
                        )
    
    # ---- Predictor ----
    predictor = ['XGBRegHPlenient',
                 'TransformerSet_Dropout010',
                 'IPCA',
                 'RFF']
    
    # ---- Label ----
    labels = ['XGBoost', 
              'Transformer',
              'IPCA',
              'RFF']
    
    # ---- Target Type ----
    target_col = ['LevelTrMsp500Target', 
                  'LevelTrMsp500Target',
                  'LevelTrMsp500Target',
                  'LevelTrMsp500Target']
    
    # ---- Est. Universe ----
    est_univs = ['SP500UniverseFL',
                 'SP500UniverseFL',
                 'CRSPUniverse',
                 'SP500UniverseFL']
    
    # ---- Input Features ----
    input_feat = ['RankFeatures',
                  'RankFeatures',
                  'ZscoreFeatures',
                  'ZscoreFeatures']
    
    # ---- HP Tuning ----
    hp_tuning = "RollingWindow_win120_val12_test12"
    
    # Container to Store strats
    strats = []
    
    for i in range(len(predictor)):
        if predictor[i] != 'MarketOracle':
            load_string = (f"{settings_string(run_settings)}_" 
                           f"{predictor[i]}_" 
                           f"{target_col[i]}_" 
                           f"{est_univs[i]}_" 
                           f"{input_feat[i]}_"
                           f"{hp_tuning}.pkl")
        else:
            load_string = (f"{settings_string(run_settings)}_" 
                           f"{predictor[i]}"
                           ".pkl")
        
        with open(path + f"Portfolios/{load_string}", "rb") as f:
            strats.append([pickle.load(f)['Profit'], labels[i]])
    
    # Compute Spearman Rank
    for item in strats:
        df_profit       = item[0]
        label           = item[1]
        col_results[label]    = df_profit['ret_net'].corr(df_profit_dolvol['ret'], method='pearson')
        
    # Column Name            
    if tc_val == 1.0:
        tc_status = "TC: High"
    elif tc_val == 0.5:
        tc_status = "TC: Med"
    elif tc_val == 0.1:
        tc_status = "TC: Low"
    else:
        tc_status = "TC: Tiny"
    
    # store column
    final_results[tc_status] = col_results
       
final_df = pd.DataFrame(final_results)

print(final_df.to_latex(index=True, escape=False, float_format="%.3f"))


# =======================
#     Capture Ratio
# =======================

final_results = {}
for tc_val in [1.0, 0.5, 0.1, 0.01]:
    
    # container for this tc
    col_results = []

    # Settings 
    run_settings = dict(includeRF    = False,
                        flatMaxPi    = True,
                        flatMaxPiVal = 0.15,
                        Wmax         = None,
                        Wmin         = None,
                        volScaler    = 1.0, 
                        tcScaler     = tc_val, 
                        )
    
    # ---- Predictor ----
    predictor = ['XGBRegHPlenient',
                 'TransformerSet_Dropout010',
                 'IPCA',
                 'RFF']
    
    # ---- Label ----
    labels = ['XGBoost', 
              'Transformer',
              'IPCA',
              'RFF']
    
    # ---- Target Type ----
    target_col = ['LevelTrMsp500Target', 
                  'LevelTrMsp500Target',
                  'LevelTrMsp500Target',
                  'LevelTrMsp500Target']
    
    # ---- Est. Universe ----
    est_univs = ['SP500UniverseFL',
                 'SP500UniverseFL',
                 'CRSPUniverse',
                 'SP500UniverseFL']
    
    # ---- Input Features ----
    input_feat = ['RankFeatures',
                  'RankFeatures',
                  'ZscoreFeatures',
                  'ZscoreFeatures']
    
    # ---- HP Tuning ----
    hp_tuning = "RollingWindow_win120_val12_test12"
    
    # Container to Store strats
    strats = []
    
    for i in range(len(predictor)):
        if predictor[i] != 'MarketOracle':
            load_string = (f"{settings_string(run_settings)}_" 
                           f"{predictor[i]}_" 
                           f"{target_col[i]}_" 
                           f"{est_univs[i]}_" 
                           f"{input_feat[i]}_"
                           f"{hp_tuning}.pkl")
        else:
            load_string = (f"{settings_string(run_settings)}_" 
                           f"{predictor[i]}"
                           ".pkl")
        
        with open(path + f"Portfolios/{load_string}", "rb") as f:
            strats.append([pickle.load(f)['Profit'], labels[i]])
    
    # Compute Spearman Rank
    for item in strats:
        df_profit       = item[0]
        label           = item[1]
        # Compute the capture ratios
        # Assuming df_profit_dolvol is your benchmark dataframe available in the environment
        geo_downside_capture, geo_upside_capture, _ , _  = \
            CaptureRatio(df_profit, df_profit_dolvol, 'ret_net', 'ret')
        
        # Append results for this specific model
        col_results.append({
            'Model': label,
            'Upside Capture': geo_upside_capture,
            'Downside Capture': geo_downside_capture
        })
    
    # Store the results for this TC value as a DataFrame
    final_results[tc_val] = pd.DataFrame(col_results)

# =======================
#    LaTeX Table Output
# =======================
print("\\begin{tabular}{lrrr}")
print("    \\toprule")
print("    \\textbf{Predictor} & $\\mu$ & $\\sigma$ & \\textbf{Sharpe} \\\\")

for tc_val in [1.0, 0.5, 0.1, 0.01]:
    print("    \\midrule")
    print(f"    \\multicolumn{{3}}{{c}}{{\\textbf{{TC: {tc_val}}}}} \\\\")
    print("    \\midrule")
    
    df = final_results[tc_val]
    for _, row in df.iterrows():
        print(f"        {row['Model']} & {row['Upside Capture']:.3f} & {row['Downside Capture']:.3f} \\\\")

print("    \\bottomrule")
print("\\end{tabular}")

#%% Plot Cumulative Strategy Returns

# ======================
# Load Portfolio Strats
# ======================

# ---- Settings ----
run_settings = dict(includeRF    = False,
                    flatMaxPi    = True,
                    flatMaxPiVal = 0.15,
                    Wmax         = None,
                    Wmin         = None,
                    volScaler    = 1.0, 
                    tcScaler     = 0.1, 
                    )
tick_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]
fontsize = 18
Save_Figure = False
Net = True

# ---- Predictor ----
predictor = ['XGBRegHPlenient',
             'TransformerSet_Dropout010',
             'IPCA',
             'RFF', 
             'MarketOracle']

# ---- Label ----
labels = ['XGBoost', 
          'Transformer',
          'IPCA',
          'RFF',
          'MarketOracle']

# ---- Target Type ----
target_col = ['LevelTrMsp500Target', 
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              '']

# ---- Est. Universe ----
est_univs = ['SP500UniverseFL',
             'SP500UniverseFL',
             'CRSPUniverse',
             'SP500UniverseFL',
             '']

# ---- Input Features ----
input_feat = ['RankFeatures',
              'RankFeatures',
              'ZscoreFeatures',
              'ZscoreFeatures',
              '']

# ---- HP Tuning ----
hp_tuning = "RollingWindow_win120_val12_test12"

# Container to Store strats
strats = []

for i in range(len(predictor)):
    if predictor[i] != 'MarketOracle':
        load_string = (f"{settings_string(run_settings)}_" 
                       f"{predictor[i]}_" 
                       f"{target_col[i]}_" 
                       f"{est_univs[i]}_" 
                       f"{input_feat[i]}_"
                       f"{hp_tuning}.pkl")
    else:
        load_string = (f"{settings_string(run_settings)}_" 
                       f"{predictor[i]}"
                       ".pkl")
    
    with open(path + f"Portfolios/{load_string}", "rb") as f:
        strats.append([pickle.load(f)['Profit'], labels[i]])


# ======================
#     Plot CumRet TS
# ======================

# --- Generate SaveName of Plot ----

flatMaxPiVal = run_settings["flatMaxPiVal"]
volScaler    = run_settings["volScaler"]
tcScaler     = run_settings["tcScaler"]

if Net:
    plot_filename = (path + "Plots/" + 
                     f"CumRetNet_AllPredictors_LevelTarget_{plot_string(run_settings)}.pdf")
else:
    plot_filename = (path + "Plots/" + 
                     f"CumRetGross_AllPredictors_LevelTarget_{plot_string(run_settings)}.pdf")


# ---- Set the color ----
colors = [
    "#1f77b4",  # XGBoost
    "#658A0B",  # Transformer
    "#967969",  # IPCA
    "#d62728",  # RFF
    "#AD9721",   # Market Oracle
]

# ---- Figure Dimensions ----
fig, ax = plt.subplots(figsize=(10, 6))

# ---- Load CumRets ----
for i, strat in enumerate(strats):
    ycol = 'cumret_net' if Net else 'cumret_gross'   # adjust names as needed
    ax.plot(
        strat[0]['eom'],
        strat[0][ycol],
        label=strat[1],
        color = colors[i % len(colors)], 
        alpha=0.9,
        linewidth=1.5,
        zorder=1
    )
ax.plot(df_spy['eom'], 
         df_spy['cumulative_return'], 
         linewidth=2,
         zorder=3,
         color = 'black',
         label = "Benchmark")

# --- Labels ---
if Net:
    ax.set_ylabel("Cumulative Return (Net)", fontsize=fontsize)
else:
    ax.set_ylabel("Cumulative Return (Gross)", fontsize=fontsize)

# --- Log Scale ---
ax.set_yscale("log")
# Force the y-axis to use decimal labels instead of scientific notation
ax.yaxis.set_major_formatter(ScalarFormatter())

#Set Ticks 
ax.set_yticks(tick_range) 

# Ensure the formatter doesn't revert to scientific notation for small/large numbers
ax.ticklabel_format(axis='y', style='plain', useOffset=False)

# ---- Grid Lines ----
ax.grid(visible=True, which='major', color='gray', linestyle='-', alpha=0.4, zorder=0)
ax.grid(visible=True, which='minor', color='gray', linestyle=':', alpha=0.2, zorder=0)

# --- Year ticks: every year, vertical labels ---
all_dates = df_spy['eom']
ax.set_xlim(all_dates.min(), all_dates.max())
ax.xaxis.set_major_locator(mdates.YearLocator(1))    # tick every year
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))    # show as 2020, 2021, ...
plt.setp(ax.get_xticklabels(), rotation=90, ha="center")    # vertical

# --- Tick font sizes ---
ax.tick_params(axis="both", which="major", labelsize=fontsize)

# --- Legend font size ---
ax.legend(fontsize=fontsize)

# --- Save Plot ---
plt.tight_layout()
if Save_Figure:
    plt.savefig(plot_filename, dpi=300)

# ---- Display Plot ----
plt.title(f"MaxPi: {flatMaxPiVal}, VolScale: {volScaler}, TcScale: {tcScaler}")
plt.show()

#%% Table Summary Statistics

# ======================
# Load Portfolio Strats
# ======================

# ---- Settings ----
run_settings = dict(includeRF    = False,
                    flatMaxPi    = True,
                    flatMaxPiVal = 0.15,
                    Wmax         = None,
                    Wmin         = None,
                    volScaler    = 1.0, 
                    tcScaler     = 0.01, 
                    )

Net = False

# ---- Predictor ----
predictor = ['XGBRegHPlenient',
             'TransformerSet_Dropout010',
             'IPCA',
             'RFF']

# ---- Label ----
labels = ['XGBoost', 
          'Transformer',
          'IPCA',
          'RFF']

# ---- Target Type ----
target_col = ['LevelTrMsp500Target', 
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              'LevelTrMsp500Target']

# ---- Est. Universe ----
est_univs = ['SP500UniverseFL',
             'SP500UniverseFL',
             'CRSPUniverse',
             'SP500UniverseFL']

# ---- Input Features ----
input_feat = ['RankFeatures',
              'RankFeatures',
              'ZscoreFeatures',
              'ZscoreFeatures']

# ---- HP Tuning ----
hp_tuning = "RollingWindow_win120_val12_test12"

# Container to Store strats
strats = []

for i in range(len(predictor)):
    load_string = (f"{settings_string(run_settings)}_" 
                   f"{predictor[i]}_" 
                   f"{target_col[i]}_" 
                   f"{est_univs[i]}_" 
                   f"{input_feat[i]}_"
                   f"{hp_tuning}.pkl")
    
    with open(path + f"Portfolios/{load_string}", "rb") as f:
        strats.append([pickle.load(f), labels[i]])
        
# ====================
#       Load AUM
# ====================
df_wealth = pd.read_csv(path + "Data/wealth_evolution.csv", parse_dates=['eom'])
df_wealth = df_wealth.loc[df_wealth['eom'] >= trading_start]

# =============================
# Compute Performance Measures
# =============================

# Initialise Dictionary to Store results
perform_dict = {}
for strat in strats:
    # item[1] is the label (e.g., 'XGBoost', 'Transformer')
    label = strat[1]

    # 1. Initialize the nested dictionary for this label
    perform_dict[label] = {}
perform_dict['benchmark'] = {}
    
# ---- Compute Performance Measures for Benchmark ----
# Sharpe Ratio
mu_bench, sigma_bench, Sharpe_bench \
    = SharpeRatio(df_spy, risk_free, 
                  return_col = 'ret')
    
# Max Drawdown
drawdown_bench, _, _ = MaxDrawdown(df_spy, 
                                   df_spy, 
                                   'ret', 
                                   'ret')

perform_dict['benchmark']['mu_strategy']              = mu_bench
perform_dict['benchmark']['sigma_strategy']           = sigma_bench
perform_dict['benchmark']['Sharpe_strategy']          = Sharpe_bench
perform_dict['benchmark']['information_ratio']        = 0.0
perform_dict['benchmark']['drawdown_strat']           = drawdown_bench
perform_dict['benchmark']['geo_downside_capture']     = 1.0
# perform_dict['benchmark']['alpha_annualized']         = 0.0
# perform_dict['benchmark']['beta']                     = 1.0


# ---- Compute Performance Measures for Strategies ----
for strat in strats:
    
    # Sharpe Ratio
    mu_strategy, sigma_strategy, Sharpe_strategy \
        = SharpeRatio(strat[0]['Profit'], risk_free, 
                      return_col = 'ret_net' if Net else 'ret_gross')
        
    # Information Ratio
    information_ratio = InformationRatio(strat[0]['Profit'], 
                                         df_spy, 
                                         'ret_net' if Net else 'ret_gross', 
                                         'ret', 
                                         risk_free)
    
    # turnover
    turnover, _ = Turnover(strat[0]['Strategy'], df_wealth)

    # Max Drawdown
    drawdown_strat, _, _ = MaxDrawdown(strat[0]['Profit'], 
                                       df_spy, 
                                       'ret_net' if Net else 'ret_gross', 
                                       'ret')

    # Capture Ratio
    geo_downside_capture, _, _, _ = CaptureRatio(strat[0]['Profit'], 
                                                        df_spy, 
                                                        'ret_net' if Net else 'ret_gross', 
                                                        'ret')
    
    # Alpha and Beta
    alpha_annualized, beta = calculate_alpha_beta(strat[0]['Profit'], 
                                                  df_spy, 
                                                  'ret_net' if Net else 'ret_gross', 
                                                  'ret', 
                                                  risk_free)

    # Store Values
    label = strat[1]
    perform_dict[label]['mu_strategy']              = mu_strategy
    perform_dict[label]['sigma_strategy']           = sigma_strategy
    perform_dict[label]['Sharpe_strategy']          = Sharpe_strategy
    perform_dict[label]['Turnover']                 = turnover
    perform_dict[label]['information_ratio']        = information_ratio
    perform_dict[label]['drawdown_strat']           = drawdown_strat
    perform_dict[label]['geo_downside_capture']     = geo_downside_capture
    # perform_dict[label]['alpha_annualized']         = alpha_annualized
    # perform_dict[label]['beta']                     = beta


# ---- Generate Table ----
# 1. Define the descriptive strings based on your logic
if run_settings['volScaler'] == 1.0:
    vol_status = "Volatility: Anchored" 
if run_settings['volScaler'] == 1000000:
    vol_status = "Volatility: Detached"
if run_settings['volScaler'] == 0.01:
    vol_status = "Volatility: Mean-Variance"

tc_val = run_settings['tcScaler']
if tc_val == 1.0:
    tc_status = "TC: High"
elif tc_val == 0.5:
    tc_status = "TC: Med"
elif tc_val == 0.1:
    tc_status = "TC: Low"
else:
    tc_status = "TC: Tiny"

net_status = "Returns: Net" if Net else "Returns: Gross"

# 2. Prepare the DataFrame
df_perform = pd.DataFrame(perform_dict).T
cols = ['mu_strategy', 'sigma_strategy', 'Sharpe_strategy', 'Turnover', 'information_ratio', 
        'drawdown_strat', 'geo_downside_capture'] # , 'alpha_annualized', 'beta']
df_perform = df_perform[cols]
df_perform.columns = ['$\mu$', '$\sigma$', 'Sharpe', 'TurnOv', 'IR', 'MaxDD', 'DownCap'] # , 'Alpha', 'Beta']

# 3. Create the Custom Header Row
# We have 1 index column + 8 data columns = 9 columns total. 
# Your example used \multicolumn{10}, adjust to 9 if using the standard layout.
header_row = (f"\\multicolumn{{8}}{{c}}{{$\pi_{{max}}: {run_settings['flatMaxPiVal']}$. "
              f"{vol_status}. {tc_status}. {net_status}}} \\\\ \n")

# 4. Convert to LaTeX (dropping the standard caption to use our custom header)
latex_body = df_perform.to_latex(
    index=True, 
    escape=False, 
    float_format="%.3f",
    column_format="l" + "c" * len(df_perform.columns)
)

# 5. Inject the custom row after \toprule
final_table = latex_body.replace("\\toprule", "\\toprule\n" + header_row + "\\midrule")
print(final_table)

lines = final_table.splitlines()
trimmed_lines = lines[1:-1]
final_table_trimmed = "\n".join(trimmed_lines)

if Net:
    table_filename = (path + "Tables/" + 
                     f"CumRetNet_AllPredictors_LevelTarget_{plot_string(run_settings)}.tex")
else:
    table_filename = (path + "Tables/" + 
                      f"CumRetGross_AllPredictors_LevelTarget_{plot_string(run_settings)}.tex")

# 3. Save the trimmed table
with open(table_filename, "w", encoding="utf-8") as f:
    f.write(final_table_trimmed)

"""
An IR = 0.5 means:

For every 1% of tracking error (volatility of return relative to the benchmark), your portfolio earns 0.5% of excess return on average per year.

Put differently:

The strategy adds 0.5 units of active return per unit of active risk.
"""

#%% Effective dollar volume

# =====================
#  Load Kyle's Lambda
# =====================
df_kl = pd.read_sql_query(("SELECT id, eom, lambda "
                           "FROM Factors_processed "
                           f"WHERE eom >= '{trading_start.strftime('%Y-%m-%d')}' "),
                          con = JKP_Factors,
                          parse_dates = {'eom'})
df_kl['lambda'] = df_kl['lambda'] * 0.5 #JKMP22 legacy implementation as TC multiplied by 1/2

# ======================
# Load Portfolio Strats
# ======================

# ---- Settings ----
run_settings = dict(includeRF    = False,
                    flatMaxPi    = True,
                    flatMaxPiVal = 0.15,
                    Wmax         = None,
                    Wmin         = None,
                    volScaler    = 1.0, 
                    tcScaler     = 0.1, 
                    )
Net = True

# ---- Predictor ----
predictor = ['XGBRegHPlenient',
             'TransformerSet_Dropout010',
             'IPCA',
             'RFF']

# ---- Label ----
labels = ['XGBoost', 
          'Transformer',
          'IPCA',
          'RFF']

# ---- Target Type ----
target_col = ['LevelTrMsp500Target', 
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              'LevelTrMsp500Target']

# ---- Est. Universe ----
est_univs = ['SP500UniverseFL',
             'SP500UniverseFL',
             'CRSPUniverse',
             'SP500UniverseFL']

# ---- Input Features ----
input_feat = ['RankFeatures',
              'RankFeatures',
              'ZscoreFeatures',
              'ZscoreFeatures']

# ---- HP Tuning ----
hp_tuning = "RollingWindow_win120_val12_test12"

# Container to Store strats
strats = []

for i in range(len(predictor)):
    load_string = (f"{settings_string(run_settings)}_" 
                   f"{predictor[i]}_" 
                   f"{target_col[i]}_" 
                   f"{est_univs[i]}_" 
                   f"{input_feat[i]}_"
                   f"{hp_tuning}.pkl")
    
    with open(path + f"Portfolios/{load_string}", "rb") as f:
        strats.append([pickle.load(f)['Strategy'], labels[i]])
        

all_data = []

for df_strat, label in strats:
    # 1. Prepare variables for the regression
    # (Using a small constant for logs to avoid -inf if values are 0)
    df_temp = df_strat.copy()
    
    # Omit stocks that leave the trading universe (mechanical relationship with lambda)
    df_temp = df_temp.loc[df_temp['pi'] > 0]
    
    df_temp['dep_var'] = np.log((df_temp['pi'] - df_temp['pi_g_tm1'])**2)
    df_temp['log_lambda'] = np.log(df_temp['lambda'])
    df_temp['strategy'] = label  # To identify the group in the stacked model
    

    
    all_data.append(df_temp[['eom', 'dep_var', 'log_lambda', 'strategy']])

# 2. Combine all strategies into one "stacked" DataFrame
df_stacked = pd.concat(all_data)

# 3. Run the regression
# We include 'strategy' as a categorical fixed effect (alpha_t/label)
# and interact log_lambda with strategy to get a beta for each label.
# The "- 1" in the formula removes the global intercept to show individual strategy alphas.

model = smf.ols(
    'dep_var ~ C(strategy):log_lambda + C(eom)',
    data=df_stacked
).fit(cov_type='HC1')

print(model.summary())

"""
Conditional on trading and controlling for time effects, XGBoost and RFF 
rebalance positions more aggressively in high–dollar-volume stocks than IPCA 
and the Transformer. This liquidity-sensitive trading behavior leads to 
significantly lower realized transaction costs. This is even causal. Time FE 
required as dolvol has a trend (decreasing) so that later time periods would
have a higher weight.

a linear-linear regression was numerically ill-conditioned. This is the case
as e.g. a one-unit increase in λ corresponds to destroying market liquidity entirely. 
That’s nonsense economically. The log helps to spread the tiny values more out
so that no tiny nudges in either dependent or independent variable have
large effects on the regression.

I use the square because this gives more (relative) weight to large deviations
and therefore pronounces aggressive trades more. Moreover, the TC are quadradic
in this way

Why I don't use a model by model regression:

XGB/RFF:
    Trade a small set of liquid winners (NVDA, AAPL)
    Often hit constraints (15% cap, volatility)
    Once they trade those names, they don’t need to scale trade size much with liquidity

IPCA:
    Trades a much broader cross-section
    Must adapt trade size within each month depending on liquidity
    Therefore shows a steeper conditional elasticity inside its own traded set

In Panel regressions with interactions:
→ XGB/RFF show stronger overall liquidity sorting. Each strategy has the
same time fixed effect

In strategy-by-strategy intensive regressions:
→ IPCA shows stronger within-strategy liquidity elasticity. Each strategy
has its own time fixed effects.

Individual Models: The model asks, "Relative to the average trading intensity of this specific strategy in month t, how does λ affect trading?"

Stacked Model: The model asks, "Relative to the average trading intensity of all strategies combined in month t, how does λ affect trading?"
"""


"""
Next period's return of the top 10 lowest lambda stocks compared to the other
stocks (equal-weighted 'portfolio')
"""

#%% Transaction Costs

# ====================
# TS Transaction Costs
# ====================

"""
Check the top 10 highest return assets, check their rank of Kyle's Lambda and check
how high each of the strategies loads onto these stocks
"""

#%% Return Predictions
"""
Cross-sectional Variance of Return Predictions for each predictor.

Low Variance = low scale = TC more important in objective
"""
# ---- Predictor ----
predictors = ['XGBRegHPlenient',
             'TransformerSet_Dropout010',
             'IPCA',
             'RFF']

# ---- Label ----
labels = ['XGBoost', 
          'Transformer',
          'IPCA',
          'RFF']

# ---- Target Type ----
target_col = ['LevelTrMsp500Target', 
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              'LevelTrMsp500Target']

# ---- Est. Universe ----
est_univs = ['SP500UniverseFL',
             'SP500UniverseFL',
             'CRSPUniverse',
             'SP500UniverseFL']

# ---- Input Features ----
input_feat = ['RankFeatures',
              'RankFeatures',
              'ZscoreFeatures',
              'ZscoreFeatures']

# ---- HP Tuning ----
hp_tuning = "RollingWindow_win120_val12_test12"

predictions = []

for i, predictor in enumerate(predictors):
    df = pd.read_sql_query(("SELECT * "
                            f"FROM {predictor}_{target_col[i]}_{est_univs[i]}_{input_feat[i]}_{hp_tuning} "
                            f"WHERE eom >= '{trading_start.strftime('%Y-%m-%d')}' AND eom <= '{trading_end.strftime('%Y-%m-%d')}'"
                           ),
                           con= Models,
                           parse_dates = {'eom'}
                           )
    predictions.append([df, labels[i]])
    
variances   = []
for i, item in enumerate(predictions):
    # Unpack
    df      = item[0]
    label   = item[1]
    
    # Get Name of Prediction column
    ret_col = [col for col in df.columns if col not in ['eom', 'id']][0]
    # Compute cross-sectional variance for each date
    df_var = df.groupby('eom')[ret_col].var().rename(f"var_{labels[i]}")

    variances.append(df_var)
    
# Drop Transformer as it always has the highest variance and distorts results
df_variances = pd.concat(variances, axis = 1).drop(columns = "var_Transformer")
row_sum = df_variances.sum(axis=1)
df_variances = df_variances.div(row_sum, axis=0)

# ---- Plot ----

# Rename columns to remove 'var_' prefix for the legend
df_variances.columns = [col.replace('var_', '') for col in df_variances.columns]

# Color map keys to match the new names
color_map = {
    "XGBoost": "#1f77b4",
    "Transformer": "#658A0B",
    "IPCA": "#967969",
    "RFF": "#d62728",
    "Market Oracle": "#AD9721"
}

current_colors = [color_map.get(col, "#333333") for col in df_variances.columns]

# --- FORCE datetime index ---
dfv = df_variances.copy()
dfv.index = pd.to_datetime(dfv.index)
dfv = dfv.sort_index()

# --- Build x as actual datetimes ---
x = dfv.index.to_pydatetime()
cols = dfv.columns

fig, ax = plt.subplots(figsize=(12, 7))

# stackplot needs one array per series
ys = [dfv[c].to_numpy() for c in cols]

ax.stackplot(
    x,
    ys,
    labels=cols,
    colors=current_colors,
    alpha=0.85
)

# --- Force yearly ticks (use YearLocator + DateFormatter) ---
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# If you REALLY want every single year label even if crowded:
# years = pd.date_range(f"{dfv.index.min().year}-01-01", f"{dfv.index.max().year}-01-01", freq="YS")
# ax.set_xticks(years.to_pydatetime())
# ax.set_xticklabels([d.strftime("%Y") for d in years], rotation=90, ha="center")

plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

ax.set_xlabel(None)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_ylabel("Share of Total Variance", fontsize=18)

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=len(cols),
    frameon=False,
    fontsize=18
)

ax.set_xlim(dfv.index.min(), dfv.index.max())
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(path + "Plots/Variances_RetPreds.pdf", dpi=300, bbox_inches='tight')
plt.show()


#%% Portfolio Analysis

# ===================
# Unique Predictions
# ===================


# ===================
#   Weight Difference
# ===================
"""
Check the difference of the portfolio weights to the benchmark. See whether
the portfolios in which the algorithm goes very long in are the high-dolvol stocks
"""

# =====================
# Tech Frenzy 2023-2024
# =====================
"""
Check if the algorithm managed to buy the top performing (tech) firms in
these years
"""




