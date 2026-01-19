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
from linearmodels.panel import PanelOLS

import os
os.chdir(path + "Code/")
import General_Functions as GF

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

# Kyle's Lambda (load one month before trading_start as lambda can be required at the BEGINNING of a month)
df_kl = (
    pd.read_sql_query(
        ("SELECT id, eom, lambda FROM Factors_processed "
         f"WHERE eom >= '{(trading_start - pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')}'"),
        con=JKP_Factors,
        parse_dates={'eom'}
        )
    .sort_values(['id', 'eom'])
    .assign(eom_lead = lambda df: df['eom'] + pd.offsets.MonthEnd(1))
    )

# Wealth (AUM). load one month before trading_start as wealth can be required at the BEGINNING of a month)
df_wealth = pd.read_csv(path + "Data/wealth_evolution.csv", parse_dates=['eom'])
# df_wealth['wealth_lagged'] = df_wealth['wealth'].shift(1)
df_wealth = df_wealth.loc[(df_wealth['eom'] >= trading_start - pd.offsets.MonthEnd(1)) 
                          & 
                          (df_wealth['eom'] <= trading_end)
                          ].sort_values(by = 'eom')

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

def meanRet_varRet(df, return_col):
    mu = df[return_col].mean()
    sigma = df[return_col].std(ddof=0)
    
    # Annualised
    return 12*mu, np.sqrt(12)*sigma

def SharpeRatio(df, risk_free, return_col):
    """
    Computes the annualised Sharpe Ratio of a time series of monthly returns.
    """
    df = (df
          .merge(risk_free, on = 'eom', how = 'left')
          .assign(ret_exc = lambda df: df[return_col] - df['rf'])
          )

    #Sharpe Ratio
    mu_Sharpe, sigma_Sharpe = df['ret_exc'].mean(), df['ret_exc'].std() 
    Sharpe = np.sqrt(12) * (mu_Sharpe / sigma_Sharpe)
    
    return Sharpe

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

#%% Function: Other

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

def hypothetical_TC(predictor, labels, target_col, est_univs, input_feat, hp_tuning, 
                    pi_max, volScaler, tcReguliser, 
                    df_wealth, df_kl):
    """
    Computes hypothetical transaction costs and net returns for multiple strategies.
    
    Returns:
        dict_dfs: Dictionary of detailed DataFrames keyed by label.
        dict_profits: Dictionary of aggregated profit DataFrames keyed by label.
    """
    dict_dfs = {}
    dict_profits = {}
    
    # ---- Settings ----
    run_settings = dict(includeRF    = False,
                        flatMaxPi    = True,
                        flatMaxPiVal = pi_max,
                        Wmax         = None,
                        Wmin         = None,
                        volScaler    = volScaler, 
                        tcScaler     = tcReguliser, 
                        )
    
    # Scale factors for transaction costs
    tc_scalers = [1.0, 0.5, 0.1, 0.01]

    for i in range(len(predictor)):
        label = labels[i]
        
        # 1. Generate path and load data
        load_string = (f"{settings_string(run_settings)}_" 
                       f"{predictor[i]}_" 
                       f"{target_col[i]}_" 
                       f"{est_univs[i]}_" 
                       f"{input_feat[i]}_"
                       f"{hp_tuning}.pkl")
        
        try:
            with open(f"{path}Portfolios/{load_string}", "rb") as f:
                df = (pickle.load(f)['Strategy']
                      .drop(columns=['lambda', 'tc'], errors='ignore'))
        except FileNotFoundError:
            print(f"Warning: File not found for {label}")
            continue
        
        # Drop ghost obsevations (will cause tiny inconsistencies when recomputing TC)
        # df = df.dropna()

        # 2. Merge auxiliary data
        # Baseline Wealth at BEGINNING of month
        df = df.merge(df_wealth.assign(eom_lead = lambda df: df['eom'] + pd.offsets.MonthEnd(1))
                      [['eom_lead','wealth']],
                      left_on = ['eom'], right_on=['eom_lead'], how='left').drop(columns='eom_lead')
        # Baseline Lambda at BEGINNING of month
        df = df.merge(df_kl.assign(eom_lead = lambda df: df['eom'] + pd.offsets.MonthEnd(1))
                      [['id','eom_lead','lambda']], left_on=['id', 'eom'],
                      right_on = ['id','eom_lead'], 
                      how = 'left').drop(columns='eom_lead')


        # 3. Compute Hypothetical Transaction Costs (Loop through scalers)
        for tc_scaler in tc_scalers:
            tc_col = f"tc_{tc_scaler}"
            ret_net_col = f"ret_net_{tc_scaler}"
            
            df = df.assign(**{
                tc_col: lambda x, s=tc_scaler: s * x['wealth'] * 0.5 * x['lambda'] * (x['pi'] - x['pi_g_tm1'])**2, # Caution: 0.5 due to JKMP22 legacy code.
                ret_net_col: lambda x, tc=tc_col: x['rev'] - x[tc]
            })

        # Store the granular dataframe
        dict_dfs[label] = df

        # 4. Aggregate results for df_profit
        df_profit = None
        for tc_scaler in tc_scalers:
            ret_net_col = f"ret_net_{tc_scaler}"
            cum_ret_col = f"cum_net_{tc_scaler}"
            
            df_add = (df
                      .groupby('eom', as_index=False)
                      .agg(**{ret_net_col: (ret_net_col, 'sum')})
                      .sort_values('eom')
                      .assign(**{cum_ret_col: lambda x: (1.0 + x[ret_net_col]).cumprod()}))

            # --- Logic to handle missing initial df_profit ---
            if df_profit is None:
                df_profit = df_add.copy()
            else:
                # For subsequent scalers, merge into the base
                df_profit = df_profit.merge(
                    df_add[['eom', ret_net_col, cum_ret_col]], 
                    on='eom', 
                    how='left'
                )
            
        dict_profits[label] = df_profit

    return dict_dfs, dict_profits

#%% Long-Short Type Portfolios


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
est_univs = ['CRSPUniverse',
             'CRSPUniverse',
             'CRSPUniverse',
             'CRSPUniverse']

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
    
    # If operating on SP500 Universe
    # df = df.merge(df_sp500_ids.assign(in_sp500 = True), on = ['id', 'eom'], how = 'left')
    # df = df.dropna().drop(columns = 'in_sp500')
    
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
    mu, sigma = meanRet_varRet(df_profit, 'ret')
    sharpe = SharpeRatio(df_profit, risk_free, 'ret')
    
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

# Get Next period's return (For Top 10: long_cutoff 0.98 & long_only = True)
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
#       Portfolio Share of DolVol Top 10 
# ===============================================

#Dol Vol Top 10
# Get Next period's return (For Top 10: long_cutoff 0.98 & long_only = True)
df_dolvolTop10, df_profit_dolvolTop10 = long_short_portfolio(df_dolvol, 'dolvol_126d',  # Predicted dollar volume
                         df_returns, # Realised Returns
                         df_me, # Market Equity for Value Weighting
                         long_cutoff = 0.98, short_cutoff = 0.1,
                         value_weighted = False,
                         long_only = True,
                         )

df_dolvolTop10 = df_dolvolTop10.assign(in_Top10 = True)

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
            strats.append([pickle.load(f)['Strategy'], labels[i]])
    
    # Compute Spearman Rank
    for item in strats:
        df_pf       = item[0]
        label       = item[1]
        # Subset of DolVol Top10 Stocks
        df_pf = df_pf.merge(df_dolvolTop10[['id','eom','in_Top10']], on = ['id', 'eom'], how = 'left')
        df_pf = df_pf.dropna() #Drop all non DolVol Top10 Stocks
        
        """
        I have to check how much of the revenue is due to the Top10 DolVol stocks.
        Because XGB basically only buys AAPL and NVIDIA from the Top 10 DolVol stocks.
        
        Whereas RFF does not buy AAPL, but it buys NVIDIA.
        """
        # check = df_pf.groupby('eom')['pi'].sum()
        
        
        
        
        
        
        
        
        
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
#%% DolVol Stocks Share

# Top 10 DolVol Stocks
# Load DolVol
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

# Get Next period's return (For Top 10: long_cutoff 0.98 & long_only = True)
df_ls_dolvol, df_profit_dolvol = long_short_portfolio(df_dolvol, 'dolvol_126d',  # Predicted dollar volume
                         df_returns, # Realised Returns
                         df_me, # Market Equity for Value Weighting
                         long_cutoff = 0.98, short_cutoff = 0.1,
                         value_weighted = False,
                         long_only = True,
                         )

# ====================
#   Load Portfolios
# ====================

# ---- Settings ----
run_settings = dict(includeRF    = False,
                    flatMaxPi    = True,
                    flatMaxPiVal = 0.15,
                    Wmax         = None,
                    Wmin         = None,
                    volScaler    = 1.0, 
                    tcScaler     = 1.0, 
                    )
tick_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 17]
fontsize = 18
Save_Figure = True
Ridge = True

# ---- Predictor ----
predictor = ['XGBRegHPlenient',
             'TransformerSet_Dropout010',
             'IPCA',
             'RFF', 
             ]

# ---- Label ----
labels = ['XGB', 
          'Transformer',
          'IPCA',
          'RFF',
          ]

# ---- Target Type ----
target_col = ['LevelTrMsp500Target', 
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              ]

# ---- Est. Universe ----
est_univs = ['SP500UniverseFL',
             'SP500UniverseFL',
             'CRSPUniverse',
             'SP500UniverseFL',
             ]

# ---- Input Features ----
input_feat = ['RankFeatures',
              'RankFeatures',
              'ZscoreFeatures',
              'ZscoreFeatures',
              ]

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
    if Ridge:
        load_string = "Ridge_" + load_string

    
    with open(path + f"Portfolios/{load_string}", "rb") as f:
        df_pf = pickle.load(f)['Strategy']
        # Get Top 10 DolVol Stocks
        df_pf = df_pf.merge(df_ls_dolvol.assign(in_Top10 = True)[['id','eom','in_Top10']],
                            how = 'left', on = ['id','eom']).dropna()
        df_pf = df_pf.groupby('eom')['pi'].sum().reset_index()
        
        strats.append([df_pf, labels[i]]) # 'Strategy' for Portfolio Weights

#%% DolVol Effect on Stock Return

df = pd.read_sql_query(("SELECT id, eom, tr, dolvol_126d FROM Factors_processed "
                        f"WHERE eom >='{trading_start.strftime('%Y-%m-%d')}' AND eom <='{trading_end.strftime('%Y-%m-%d')}'"
                        ),
                       con = JKP_Factors,
                       parse_dates = {'eom'}
                       )

# 1. Pre-process the data
# Log-transform the independent variable
df['log_dolvol'] = np.log(df['dolvol_126d'])

df['weights'] = df.groupby('eom')['dolvol_126d'].transform(lambda x: x / x.sum())

# 2. Set the index to (Entity, Time) - This is required for PanelOLS
df_reg = df.set_index(['id', 'eom'])

# 3. Define the regression
# tr = dependent variable
# log_dolvol = independent variable
# TimeEffects=True adds the time fixed effects (intercept is usually excluded or handled automatically)
model = PanelOLS(
    dependent=df_reg['tr'], 
    exog=df_reg['log_dolvol'], 
    weights=df_reg['weights'],
    time_effects=True
)

# 4. Fit the model
# Using 'clustered' covariance is common in finance to handle autocorrelation
results = model.fit(cov_type='clustered', cluster_entity=True)

print(results)
        
        
        

#%% Plot Cumulative Strategy Returns

# ======================
# Load Portfolio Strats
# ======================

# Broadcom: 93002
# NVDA: 86580 
# AAPL: 14593

# ---- Settings ----
run_settings = dict(includeRF    = False,
                    flatMaxPi    = True,
                    flatMaxPiVal = 0.15,
                    Wmax         = None,
                    Wmin         = None,
                    volScaler    = 1.0, 
                    tcScaler     = 1.0, 
                    )
tick_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 17]
fontsize = 18
Save_Figure = False
Net = True

# ---- Predictor ----
predictor = ['XGBRegHPlenient',
             'TransformerSet_Dropout010',
             'IPCA',
             'RFF', 
             #'MarketOracle',
             ]

# ---- Label ----
labels = ['XGB', 
          'Transformer',
          'IPCA',
          'RFF',
          #'MarketOracle',
          ]

# ---- Target Type ----
target_col = ['LevelTrMsp500Target', 
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              #'',
              ]

# ---- Est. Universe ----
est_univs = ['SP500UniverseFL',
             'SP500UniverseFL',
             'CRSPUniverse',
             'SP500UniverseFL',
             #'',
             ]

# ---- Input Features ----
input_feat = ['RankFeatures',
              'RankFeatures',
              'ZscoreFeatures',
              'ZscoreFeatures',
              #'',
              ]

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
        strats.append([pickle.load(f)['Profit'], labels[i]]) # 'Strategy' for Portfolio Weights


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

# ==========================================
# 1. Settings & Setup
# ==========================================
tc_values       = [1.0, 0.5, 0.1, 0.01]
tc_map          = {1.0: "High", 0.5: "Med", 0.1: "Low", 0.01: "Tiny"}
flatMaxPiVal    = 0.15
volScaler       = 1.0

# Ridge
Ridge = False

#Net Returns
Net = True

# ---- Predictors ----
predictor = ['XGBRegHPlenient',
             'TransformerSet_Dropout010',
             'IPCA',
             'RFF']

# ---- Labels ----
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

# Master container: {Model_Label: {TC_Status: {Metrics}}}
all_results = {label: {} for label in labels}

# ========================
# 2. Main Processing Loop
# ========================

# Calculate Benchmark (only needs to run once)
mu_bench, sigma_bench, = meanRet_varRet(df_spy, 'ret')
Sharpe_bench = SharpeRatio(df_spy, risk_free, return_col='ret')
drawdown_bench, _, _ = MaxDrawdown(df_spy, df_spy, 'ret', 'ret')

# Loop over Strats
for tc_val in tc_values:
    
    # ---- Settings ----
    run_settings = dict(includeRF    = False,
                        flatMaxPi    = True,
                        flatMaxPiVal = flatMaxPiVal,
                        Wmax         = None,
                        Wmin         = None,
                        volScaler    = volScaler, 
                        tcScaler     = tc_val, 
                        )

    # ---- Load Strats ----
    
    # Container to Store strats
    strats = []
    
    for i in range(len(predictor)):
        load_string = (f"{settings_string(run_settings)}_" 
                       f"{predictor[i]}_" 
                       f"{target_col[i]}_" 
                       f"{est_univs[i]}_" 
                       f"{input_feat[i]}_"
                       f"{hp_tuning}.pkl")
        
        if Ridge:
            load_string = "Ridge_" + load_string
        
        with open(path + f"Portfolios/{load_string}", "rb") as f:
            strats.append([pickle.load(f), labels[i]])
        
    # ---- Compute Performance Measures ----    
    # Initialise Dictionary to Store results
    perform_dict = {}
    for strat in strats:
        data = strat[0]
        label = strat[1]
        ret_col = 'ret_net' if Net else 'ret_gross'
        
        mu_s, sigma_s = meanRet_varRet(data['Profit'], ret_col)
        sharpe_s = SharpeRatio(data['Profit'], risk_free, return_col=ret_col)
        ir = InformationRatio(data['Profit'], df_spy, ret_col, 'ret', risk_free)
        to, _ = Turnover(data['Strategy'], df_wealth)
        dd, _, _ = MaxDrawdown(data['Profit'], df_spy, ret_col, 'ret')
        down_cap, _, _, _ = CaptureRatio(data['Profit'], df_spy, ret_col, 'ret')
        
        # Store in master dictionary
        all_results[label][tc_map[tc_val]] = {
            'mu': mu_s, 'sigma': sigma_s, 'sharpe': sharpe_s,
            'turnover': to, 'ir': ir, 'dd': dd, 'down_cap': down_cap
        }

# ==========================================
# 3. Manual LaTeX Table Construction
# ==========================================
status_str = "Net" if Net else "Gross"

latex_str = f"""\\begin{{table}}[htpb]
\\centering
\\caption{{Summary Statistics {status_str} Returns}}
\\label{{Table:SummaryStats_{status_str}Returns}}
\\begin{{threeparttable}}
\\begin{{tabular}}{{lccccccc}}
\\toprule
 & $\\mu$ & $\\sigma$ & Sharpe & TurnOv & IR  &  MaxDD & DownCap \\\\
 \\midrule 
"""

# Benchmark Row
latex_str += f"S\&P 500 & {mu_bench:.3f} & {sigma_bench:.3f} & {Sharpe_bench:.3f} & $\\bullet$ & 0.000 & {drawdown_bench:.3f} & 1.000 \\\\\n"
latex_str += "\\bottomrule\n"

# Model Blocks
for label in labels:
    latex_str += "%" + "="*50 + "\n"
    latex_str += "\\toprule\n"
    latex_str += f"\\multicolumn{{8}}{{c}}{{\\textbf{{{label}}}}} \\\\\n"
    
    # Iterate through TC statuses in specific order
    for tc_status in ["High", "Med", "Low", "Tiny"]:
        if tc_status in all_results[label]:
            res = all_results[label][tc_status]
            row = (f"{tc_status} & {res['mu']:.3f} & {res['sigma']:.3f} & "
                   f"{res['sharpe']:.3f} & {res['turnover']:.3f} & {res['ir']:.3f} & "
                   f"{res['dd']:.3f} & {res['down_cap']:.3f} \\\\\n")
            latex_str += row
        
    latex_str += "\\bottomrule\n"

latex_str += r"""\end{tabular}
\end{threeparttable}
\end{table}"""

# ==========================================
# 4. Save to File
# ==========================================
file_suffix = "Net" if Net else "Gross"
table_filename = path + f"Tables/SummaryStats_{file_suffix}.tex"

with open(table_filename, "w", encoding="utf-8") as f:
    f.write(latex_str)

print(f"Table saved to {table_filename}")
    

"""
An IR = 0.5 means:

For every 1% of tracking error (volatility of return relative to the benchmark), your portfolio earns 0.5% of excess return on average per year.

Put differently:

The strategy adds 0.5 units of active return per unit of active risk.
"""

#%% Panel Regression Lambda

"""
The higher the lambda, the higher the trade.
"""

results_list = []

for tc_val in [1.0, 0.5, 0.1, 0.01]:

    # ---- Settings ----
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
    
    # Extract coefficients and confidence intervals
    conf_int = model.conf_int()
    params = model.params
    
    # Filter for the interaction terms (C(strategy)[...]:log_lambda)
    # This ignores the eom fixed effects
    for label in labels:
        coeff_name = f'C(strategy)[{label}]:log_lambda'
        if coeff_name in params.index:
            results_list.append({
                'tc_val': tc_val,
                'strategy': label,
                'beta': params[coeff_name],
                'lower': conf_int.loc[coeff_name, 0],
                'upper': conf_int.loc[coeff_name, 1]
            })

# Final DataFrame for plotting
df_ci = pd.DataFrame(results_list)
df_ci['strategy'] = df_ci['strategy'].replace({
    'XGBoost': 'XGB',
    'Transformer': 'TF'
})

# 1. Setup a single plot
fig, ax = plt.subplots(figsize=(10, 6))

tc_values = [1.0, 0.5, 0.1, 0.01]
limits_map = {
    1.0: (-1.6, -1.0),
    0.5: (-1.1, -0.4),
    0.1: (-0.2, 0.66),
    0.01: (0.88, 1.9)
}
desired_order = ['XGB', 'TF', 'IPCA', 'RFF']

# Colors for Plots
color_map = {
    'XGB': "#1f77b4",
    'TF': "#658A0B",
    'IPCA': "#967969",
    'RFF': "#d62728",
    'Market Oracle': "#AD9721"
}

# Iterate through each TC value and create a standalone figure for each
for tc in tc_values:
    # 1. Setup individual figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    subset = df_ci[df_ci['tc_val'] == tc].copy()
    
    # Ensure categorical order for the y-axis
    subset['strategy'] = pd.Categorical(subset['strategy'], categories=desired_order[::-1], ordered=True)
    subset = subset.sort_values('strategy')
    
    # 2. Plotting
    for _, row in subset.iterrows():
        strat_label = row['strategy']
        strat_color = color_map.get(strat_label, "black")
        
        error_low = row['beta'] - row['lower']
        error_high = row['upper'] - row['beta']
        
        ax.errorbar(
            row['beta'], strat_label, 
            xerr=[[error_low], [error_high]], 
            fmt='o', capsize=6, 
            color=strat_color, 
            markersize=9, 
            linewidth=2.5
        )
    
    # Tick Size
    ax.tick_params(axis='both', which='major', labelsize=18)

    # 3. Individual Formatting
    if tc in limits_map:
        ax.set_xlim(limits_map[tc])
    
    # ax.set_title(f'Sensitivity of Trading Variance: TC Scale {tc}', fontweight='bold', fontsize=14)
    ax.set_xlabel('$\log (\lambda)$', fontsize=18)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(path + f"Plots/Log_Lambda_tc_{str(tc).replace(".", "")}.pdf")
    # This will display the current plot before moving to the next iteration
    plt.show()

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

#%% TC AVerages

# ---- Predictor ----
predictor = ['XGBRegHPlenient',
             'TransformerSet_Dropout010',
             'IPCA',
             'RFF', 
             ]

# ---- Label ----
labels = ['XGB', 
          'Transformer',
          'IPCA',
          'RFF',
          ]

# ---- Target Type ----
target_col = ['LevelTrMsp500Target', 
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              'LevelTrMsp500Target',
              ]

# ---- Est. Universe ----
est_univs = ['SP500UniverseFL',
             'SP500UniverseFL',
             'CRSPUniverse',
             'SP500UniverseFL',
             ]

# ---- Input Features ----
input_feat = ['RankFeatures',
              'RankFeatures',
              'ZscoreFeatures',
              'ZscoreFeatures',
              ]

# ---- HP Tuning ----
hp_tuning = "RollingWindow_win120_val12_test12"

# ---- Settings ----

tc_values       = [1.0, 0.5, 0.1, 0.01]
flatMaxPiVal    = 0.15
volScaler       = 1.0

# --- Objects for Storing ---
all_results = {label: {} for label in labels}
tc_map          = {1.0: "TC: High", 0.5: "TC: Med", 0.1: "TC: Low", 0.01: "TC: Tiny"}

for tc_scaler in tc_values:
    
    # ---- Settings ----
    run_settings = dict(includeRF    = False,
                        flatMaxPi    = True,
                        flatMaxPiVal = flatMaxPiVal,
                        Wmax         = None,
                        Wmin         = None,
                        volScaler    = volScaler, 
                        tcScaler     = tc_scaler, 
                        )
    

    
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
            strats.append([pickle.load(f)['Strategy'], labels[i]]) # 'Strategy' for Portfolio Weights


    
    for item in strats:
        df_strat    = item[0]
        label       = item[1]
        
        all_results[label][tc_map[tc_scaler]] = df_strat['tc'].sum()


all_values = [val for model_dict in all_results.values()
                  for val in model_dict.values()]

global_max = max(all_values)


all_results = {key: {tc: val/global_max for tc, val in inner_dict.items()}
               for key, inner_dict in all_results.items()
               }

df = pd.DataFrame(all_results)
latex_table = df.to_latex(
    index=True, 
    caption="Normalized Transaction Costs across Regimes",
    label="Table:TC_Summary",
    float_format="%.3f",
    column_format="lcccc"
)

print(latex_table)







#%% TC Hypothetical

# ================
# Hypothetical TC
# ================

# ---- Predictor ----
predictor = ['XGBRegHPlenient',
             #'TransformerSet_Dropout010',
             #'IPCA',
             #'RFF',
             ]

# ---- Label ----
labels = ['XGB', 
          #'Transformer',
          #'IPCA',
          #'RFF',
          ]

# ---- Target Type ----
target_col = ['LevelTrMsp500Target', 
              #'LevelTrMsp500Target',
              #'LevelTrMsp500Target',
              #'LevelTrMsp500Target',
              ]

# ---- Est. Universe ----
est_univs = ['SP500UniverseFL',
             #'SP500UniverseFL',
             #'CRSPUniverse',
             #'SP500UniverseFL',
             ]

# ---- Input Features ----
input_feat = ['RankFeatures',
              #'RankFeatures',
              #'ZscoreFeatures',
              #'ZscoreFeatures',
              ]

# ---- HP Tuning ----
hp_tuning = "RollingWindow_win120_val12_test12"


# =======================
# Compute Hypothetical TC
# =======================


# Load Strategy
dict_strats, dict_profits = hypothetical_TC(
    predictor=predictor,
    labels=labels,
    target_col=target_col,
    est_univs=est_univs,
    input_feat=input_feat,
    hp_tuning=hp_tuning,
    pi_max=0.15,
    volScaler=1.0,
    tcReguliser=1.0, # Regularisation according to TC
    df_wealth=df_wealth,
    df_kl=df_kl
)

# ============================================
# Compute Performance Metrics & Generate Table
# ============================================

tc_map = {1.0: 'High', 0.5: 'Med', 0.1: 'Low', 0.01: 'Tiny'}
table_results = {label: {} for label in labels}
for label in labels:
    
    # Retrieve the specific dataframes for this label
    df_current_strat = dict_strats[label]
    df_current_profit = dict_profits[label]
    
    for tc_scaler in [1.0, 0.5, 0.1, 0.01]:
        ret_net_col = f"ret_net_{tc_scaler}"
        
        # Calculate Statistics
        mu_s, sigma_s = meanRet_varRet(df_current_profit, ret_net_col)
        sharpe_s = SharpeRatio(df_current_profit, risk_free, ret_net_col)
        ir = InformationRatio(df_current_profit, df_spy, ret_net_col, 'ret', risk_free)
        
        # Note: Turnover calculation might need the specific scaler columns from df_current_strat
        to, _ = Turnover(df_current_strat, df_wealth) 
        
        dd, _, _ = MaxDrawdown(df_current_profit, df_spy, ret_net_col, 'ret')
        down_cap, _, _, _ = CaptureRatio(df_current_profit, df_spy, ret_net_col, 'ret')
        
        # Store results
        row_label = tc_map[tc_scaler]
        table_results[label][row_label] = {
            'mu': mu_s, 'sigma': sigma_s, 'Sharpe': sharpe_s,
            'TurnOv': to, 'IR': ir, 'MaxDD': dd, 'DownCap': down_cap
        }
        
# ================
# EXPORT TO LATEX
# ================

for label in labels:

    # Convert results for the first label into a DataFrame
    df_table = pd.DataFrame.from_dict(table_results[label], orient='index')
    
    # Formatting and Export
    cols = ['mu', 'sigma', 'Sharpe', 'TurnOv', 'IR', 'MaxDD', 'DownCap']
    df_table = df_table[cols]
    
    latex_output = df_table.to_latex(
        index=True,
        float_format="%.3f",
        column_format="lccccccc",
        caption=f"Performance Analysis: {label}",
        label=f"tab:{label.lower()}_results",
        escape=False
    )
    
    # Replace placeholders with LaTeX math symbols
    latex_output = latex_output.replace('mu', r'$\mu$').replace('sigma', r'$\sigma$')
    
    print(latex_output)


#%% Variance Return Predictions
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
#%% Portfolio Visualisation
"""
Visualise the holdings of AAPL for XGBoost in tc = 1.0.

Probably best to display the percentile of tr and of retpred to
get the cross-sectional comparison
"""
# ---- Settings ----
run_settings = dict(includeRF    = False,
                    flatMaxPi    = True,
                    flatMaxPiVal = 0.15,
                    Wmax         = None,
                    Wmin         = None,
                    volScaler    = 1.0, 
                    tcScaler     = 1.0, 
                    )
#tick_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]
fontsize = 18
Save_Figure = False
Net = True
stock_id = 86580 #AAPL: 14593, NVDA: 86580

# ---- Predictor ----
predictor = ['XGBRegHPlenient']

# ---- Label ----
labels = ['XGB']

# ---- Target Type ----
target_col = ['LevelTrMsp500Target']

# ---- Est. Universe ----
est_univs = ['SP500UniverseFL']

# ---- Input Features ----
input_feat = ['RankFeatures']

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
        strats.append([pickle.load(f)['Strategy'], labels[i]]) # 'Strategy' for Portfolio Weights
        
        
for item in strats:
    df = item[0]
    label = item[1]
    df = df[['id','eom', 'pi', 'pi_g_tm1', 'tr', df.columns[-1]]] # Last column is the return prediction
    df = df.rename(columns = {'tr':'Real. Ret', df.columns[-1]:'Pred. Ret'})
    df = df[df['id'] == stock_id]
   
    df = df.sort_values('eom')
    
    # --- Plot 1: pi and pi_g_tm1 (Single Axis) ---
    plt.figure(figsize=(10, 5))
    plt.plot(df['eom'], df['pi'], label='pi', marker='o')
    plt.plot(df['eom'], df['pi_g_tm1'], label='pi_g_tm1', linestyle='--', marker='x')
    
    plt.title('Comparison of PI and PI_G_TM1')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # --- Plot 2: tr and ret_pred_Levelmsp500 (Dual Axis) ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Primary Y-Axis (Left) for 'Real. Ret'
    color_tr = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('TR (Total Return)', color=color_tr)
    ax1.plot(df['eom'], df['Real. Ret'], color=color_tr, label='Real. Ret', marker='s')
    ax1.tick_params(axis='y', labelcolor=color_tr)
    
    # Secondary Y-Axis (Right) for 'Pred. Ret'
    ax2 = ax1.twinx()  
    color_ret = 'tab:red'
    ax2.set_ylabel('Ret Pred Level MSP500', color=color_ret)
    ax2.plot(df['eom'], df['Pred. Ret'], color=color_ret, label='ret_pred', marker='d')
    ax2.tick_params(axis='y', labelcolor=color_ret)
    
    plt.title('TR vs Ret Pred Level MSP500 (Dual Axis)')
    fig.tight_layout()
    plt.show()
#%% Mega Cap

"""
Not controlling for quality definitely makes it such that there isn't a premium.

ML algorithms give quality stocks a positive signal and combined with low lambda
they will buy the quality mega cap stocks.

It is true that more DolVol means stock has higher Market Cap !!!!!
"""

df = pd.read_sql_query(("SELECT id, eom, me, dolvol_126d, tr_ld1, in_sp500 FROM Factors_processed "
                        f"WHERE eom >= '{trading_start.strftime('%Y-%m-%d')}' "
                        f"AND eom <= '{trading_end.strftime('%Y-%m-%d')}' "
                        ),
                       con = JKP_Factors,
                       parse_dates = {'eom'}).sort_values(by = ['eom','id'])

df = df.loc[df['in_sp500'] == 1].drop('in_sp500', axis = 1)

# 1. Create the log variables
df['log_me'] = np.log(df['me'])
df['log_dolvol'] = np.log(df['dolvol_126d'])

# 2. Set the MultiIndex (Required: [Entity, Time])
df = df.set_index(['id', 'eom'])
from statsmodels.api import add_constant
X = add_constant(df['log_dolvol'])
y = df['log_me']

# Initialize the model
# time_effects=True handles the 'eom' fixed effects automatically
model = PanelOLS(y, X, time_effects=True)

# Fit the model with clustered standard errors
results = model.fit(cov_type='clustered', cluster_entity=True)

print(results)


# 1. Define your cut-off (e.g., top 10% and bottom 10%)
upper_q = 0.98
lower_q = 0.97
var = 'me'

# 2. Go LONG in stocks ABOVE a cut-off (Big Stocks)
long_high_me = df.groupby('eom').apply(
    lambda x: x[x[var] > x[var].quantile(upper_q)]
).reset_index(drop=True)
long_high_me['weight'] = 1 / long_high_me.groupby('eom')['id'].transform('count')
long_high_me['ret'] = long_high_me['weight']*long_high_me['tr_ld1']
long_high_me = long_high_me.groupby('eom')['ret'].sum().reset_index()
long_high_me['cumret'] = (1+long_high_me['ret']).cumprod()

# 3. Go LONG in stocks BELOW a cut-off (Small Stocks)
long_low_me = df.groupby('eom').apply(
    lambda x: x[x[var] < x[var].quantile(lower_q)]
).reset_index(drop=True)
long_low_me['weight'] = 1 / long_low_me.groupby('eom')['id'].transform('count')
long_low_me['ret'] = long_low_me['weight']*long_low_me['tr_ld1']
long_low_me = long_low_me.groupby('eom')['ret'].sum().reset_index()
long_low_me['cumret'] = (1+long_low_me['ret']).cumprod()

ls = long_high_me[['eom','ret']].merge(long_low_me[['eom','ret']], on = 'eom', how = 'left')
ls['ret'] = ls['ret_x'] - ls['ret_y']
ls['cumret'] = (1+ls['ret']).cumprod()

#%% Portfolio Analysis

# ================================
# Predictability by DolVol
# ================================
"""
High DolVol stocks have higher predictability.
"""

estimators = [
          'XGBRegHPlenient_LevelTrMsp500Target_SP500UniverseFL_RankFeatures_RollingWindow_win120_val12_test12',
          'TransformerSet_Dropout010_LevelTrMSp500Target_SP500UniverseFL_RankFeatures_RollingWindow_win120_val12_test12',
          'RFF_LevelTrMsp500Target_SP500UniverseFL_ZscoreFeatures_RollingWindow_win120_val12_test12',
          'IPCA_LevelTrMsp500Target_CRSPUniverse_ZscoreFeatures_RollingWindow_win120_val12_test12',
          ]

#Load return predictions
#At 'eom', predictions are for eom+1
df_retPred = GF.load_MLpredictions(Models, estimators) 

prediction_cols = list(df_retPred.columns.drop(['id','eom']))

sp500_constituents = (pd.read_sql_query("SELECT * FROM SP500_Constituents_monthly", #" WHERE eom >= '{start_date}'",
                                       con = SP500_Constituents,
                                       parse_dates = {'eom'})
                      .rename(columns = {'PERMNO': 'id'})
                      ).assign(in_sp500 = True)

df_dolvol = pd.read_sql_query(("SELECT id, eom, dolvol_126d FROM Factors_processed "
                               f"WHERE eom >= '{(trading_start- pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')}' "
                               f"AND eom <= '{trading_end.strftime('%Y-%m-%d')}'"
                               ),
                              con = JKP_Factors,
                              parse_dates = {'eom'})

df_retPred = (df_retPred
              .merge(sp500_constituents, on = ['id','eom'], how = 'left')
              .pipe(lambda df: df.loc[df['in_sp500'] == True])
              .drop('in_sp500', axis = 1)
              .pipe(lambda df: df[(df['eom'] >= trading_start - pd.offsets.MonthEnd(1)) 
                                  & 
                                  (df['eom'] <= trading_end)]
                    )
              .pipe(lambda df: df.merge(df_returns[['id','eom','tr_ld1']], on = ['id','eom'], how = 'left'))
              .pipe(lambda df: df.merge(df_dolvol, how = 'left', on = ['id','eom']))
              .sort_values(by = ['eom','id'])
              .reset_index(drop = True)
              )

df_retPred['quintile'] = (
    df_retPred.groupby('eom')['dolvol_126d']
    .transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))
)

dummies = pd.get_dummies(df_retPred['quintile'] + 1, prefix='dolvol')

# 3. Join them back to the original dataframe
df_retPred = pd.concat([df_retPred, dummies], axis=1).drop('quintile', axis=1)

results = {}
for predictor in prediction_cols:
    
    results[predictor] = {}

    for quintile in list(dummies.columns):
                
        data = df_retPred.loc[df_retPred[quintile] == 1][['id','eom',predictor, 'tr_ld1']]
        
        data = data.set_index(['id', 'eom'])
        
        mod = PanelOLS(data['tr_ld1'], data[predictor], time_effects=True)
        
        res = mod.fit(cov_type='clustered', cluster_time=True)

        # Store the coefficient and p-value in the sub-dictionary
        results[predictor][quintile] = {
            'slope': res.params[predictor],
            'p_value': res.pvalues[predictor]
        }
        
#%% Portfolios have Large Cap Stocks

"""
Compare 
ratio of value-weighted MarketCap of strategy
                vs. 
value-weighted MarketCap of S&P500 

Higher tc strats have higher market cap than index
"""

df_me = pd.read_sql_query("SELECT id, eom, me From Factors_processed",
                          con = JKP_Factors,
                          parse_dates = {'eom'})

df_me['sp500_weight'] = df_me.groupby('eom')['me'].transform(lambda x: x/x.sum())

df_sp500_me = df_me.assign(me_sp500 = lambda df: df['me'] * df['sp500_weight'])
df_sp500_me = df_sp500_me.groupby('eom')['me_sp500'].sum().reset_index()


tc_values       = [1.0, 0.5, 0.1, 0.01]
tc_map          = {1.0: "High", 0.5: "Med", 0.1: "Low", 0.01: "Tiny"}
flatMaxPiVal    = 0.15
volScaler       = 1.0

# ---- Predictors ----
predictor = ['XGBRegHPlenient',
             'TransformerSet_Dropout010',
             'IPCA',
             'RFF']

# ---- Labels ----
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

# Master container: {Model_Label: {TC_Status: {Metrics}}}
all_results = {label: {} for label in labels}

# Loop over Strats
for tc_val in tc_values:
    
    # ---- Settings ----
    run_settings = dict(includeRF    = False,
                        flatMaxPi    = True,
                        flatMaxPiVal = flatMaxPiVal,
                        Wmax         = None,
                        Wmin         = None,
                        volScaler    = volScaler, 
                        tcScaler     = tc_val, 
                        )

    # ---- Load Strats ----
    
    # Container to Store strats
    strats = []
    
    for i in range(len(predictor)):
        load_string = (f"{settings_string(run_settings)}_" 
                       f"{predictor[i]}_" 
                       f"{target_col[i]}_" 
                       f"{est_univs[i]}_" 
                       f"{input_feat[i]}_"
                       f"{hp_tuning}.pkl")
        
        if Ridge:
            load_string = "Ridge_" + load_string
        
        with open(path + f"Portfolios/{load_string}", "rb") as f:
            strats.append([pickle.load(f), labels[i]])
    
    for item in strats:
        df_strat    = item[0]['Strategy']
        label       = item[1]
            

        df_strat['eom_lag'] = df_strat['eom'] - pd.offsets.MonthEnd(1)


        df_strat = df_strat.merge(df_me[['id','eom', 'me', 'sp500_weight']], how = 'left', 
                          left_on = ['id','eom_lag'],
                          right_on = ['id','eom'], 
                          suffixes = ('','_y')).drop('eom_y', axis=1)
        
        df_strat = df_strat.assign(me_w = lambda df: df['me']*df['pi'])
        
        df_sum = df_strat.groupby('eom_lag')['me_w'].sum().reset_index()
        
        df_sum = df_sum.merge(df_sp500_me, left_on = ['eom_lag'],
                              right_on = ['eom'], 
                              how = 'left')
        
        df_sum = df_sum.assign(ratio = lambda df: df['me_w'] / df['me_sp500'])

        """
        df_strat['pi_diff'] = df_strat['pi'] - df_strat['sp500_weight']
        df_strat['log_me'] = np.log(df_strat['me'])
                
        # 1. Clean data for regression (remove NaNs which occur due to lags/missing ME)
        reg_data = df_strat[['id', 'eom', 'pi_diff', 'log_me']].dropna()
        
        # 2. Set MultiIndex for Panel data
        reg_data = reg_data.set_index(['id', 'eom'])
        
        # 3. Define and Fit Model: Active Weight ~ Log(Market Equity) + Time Fixed Effects
        # We use time_effects=True to control for month-specific shifts in average active weights
        mod = PanelOLS(reg_data['pi_diff'], reg_data['log_me'], time_effects=True)
        
        # Fit with clustered errors (by time) to account for cross-sectional correlation
        res = mod.fit(cov_type='clustered', cluster_time=True)
        
        # 4. Store in the master container
        # tc_val is the current loop variable; label is the model name
        all_results[label][tc_val] = {
            'slope': res.params['log_me'],
            'p_value': res.pvalues['log_me']
        }
        
        # Optional: Print progress
        print(f"Processed {label} at TC {tc_val}: Slope={res.params['log_me']:.6f}")
        """

#%% Fama-French Regressions




# ================================
# Regress on Fama French 5 Factor
# ================================



# ================================
#       Factor Extraction
# ================================
# Compare with FF Factors by regressing factor on FF factors

# Regress Fama-French Portfolios on my additional factor to see if it is relevant (by increasing R^2)



"""
Do portfolio optimiser with tc that solely reflect regularisation on (pi - pi_g_tm1) without the 
scaling by lambda
"""

