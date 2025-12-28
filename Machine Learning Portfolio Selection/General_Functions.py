# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 15:04:35 2025

@author: Patrick
"""

import pandas as pd
import numpy as np
import sqlite3
import pickle
from scipy import stats

#%% Hyperparameter

def get_settings():

    settings = {'rolling_window':{'validation_periods':12, #12 months
                                  'window_size':10*12,  #10 years
                                  'test_size': 12, #1 year
                                  'trading_start': pd.to_datetime('2003-01-31'), #Dates onwards must be fully out of sample
                                  'tuning_fold': 1, # = 1 implies only the last window_size + validation_periods are used for cross-validation of the hyperparameters
                                  'trading_end': pd.to_datetime('2024-12-31') #Last trading period
                                  },
                   'RFF': {'p_vec':np.array([2**i for i in range(6,11)]),
                           'sigma_vec': np.array([0.5,1,10,50]),
                           'penalty_vec':np.array([1e-3, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7])
                           },
                   'gamma': 5.0
                   }
    
    return settings

#%%

#Get the List of Features
# Features ---------------------
def get_features(exclude_poor_coverage = True):
    """
    Original Features of the Jensen, Kelly & Pedersen (2023) Global factor dataset
    """
    features = [
      "age",                 "aliq_at",             "aliq_mat",            "ami_126d",           
      "at_be",               "at_gr1",              "at_me",               "at_turnover",        
      "be_gr1a",             "be_me",               "beta_60m",            "beta_dimson_21d",    
      "betabab_1260d",       "betadown_252d",       "bev_mev",             "bidaskhl_21d",       
      "capex_abn",           "capx_gr1",            "capx_gr2",            "capx_gr3",           
      "cash_at",             "chcsho_12m",          "coa_gr1a",            "col_gr1a",           
      "cop_at",              "cop_atl1",            "corr_1260d",          "coskew_21d",         
      "cowc_gr1a",           "dbnetis_at",          "debt_gr3",            "debt_me",            
      "dgp_dsale",           "div12m_me",           "dolvol_126d",         "dolvol_var_126d",    
      "dsale_dinv",          "dsale_drec",          "dsale_dsga",          "earnings_variability",
      "ebit_bev",            "ebit_sale",           "ebitda_mev",          "emp_gr1",            
      "eq_dur",              "eqnetis_at",          "eqnpo_12m",           "eqnpo_me",           
      "eqpo_me",             "f_score",             "fcf_me",              "fnl_gr1a",           
      "gp_at",               "gp_atl1",             "ival_me",             "inv_gr1",            
      "inv_gr1a",            "iskew_capm_21d",      "iskew_ff3_21d",       "iskew_hxz4_21d",     
      "ivol_capm_21d",       "ivol_capm_252d",      "ivol_ff3_21d",        "ivol_hxz4_21d",      
      "kz_index",            "lnoa_gr1a",           "lti_gr1a",            "market_equity",      
      "mispricing_mgmt",     "mispricing_perf",     "ncoa_gr1a",           "ncol_gr1a",          
      "netdebt_me",          "netis_at",            "nfna_gr1a",           "ni_ar1",             
      "ni_be",               "ni_inc8q",            "ni_ivol",             "ni_me",              
      "niq_at",              "niq_at_chg1",         "niq_be",              "niq_be_chg1",        
      "niq_su",              "nncoa_gr1a",          "noa_at",              "noa_gr1a",           
      "o_score",             "oaccruals_at",        "oaccruals_ni",        "ocf_at",             
      "ocf_at_chg1",         "ocf_me",              "ocfq_saleq_std",      "op_at",              
      "op_atl1",             "ope_be",              "ope_bel1",            "opex_at",            
      "pi_nix",              "ppeinv_gr1a",         "prc",                 "prc_highprc_252d",   
      "qmj",                 "qmj_growth",          "qmj_prof",            "qmj_safety",         
      "rd_me",               "rd_sale",             "rd5_at",              "resff3_12_1",        
      "resff3_6_1",          "ret_1_0",             "ret_12_1",            "ret_12_7",           
      "ret_3_1",             "ret_6_1",             "ret_60_12",           "ret_9_1",            
      "rmax1_21d",           "rmax5_21d",           "rmax5_rvol_21d",      "rskew_21d",          
      "rvol_21d",            "sale_bev",            "sale_emp_gr1",        "sale_gr1",           
      "sale_gr3",            "sale_me",             "saleq_gr1",           "saleq_su",           
      "seas_1_1an",          "seas_1_1na",          "seas_11_15an",        "seas_11_15na",       
      "seas_16_20an",        "seas_16_20na",        "seas_2_5an",          "seas_2_5na",         
      "seas_6_10an",         "seas_6_10na",         "sti_gr1a",            "taccruals_at",       
      "taccruals_ni",        "tangibility",         "tax_gr1a",            "turnover_126d",      
      "turnover_var_126d",   "z_score",             "zero_trades_126d",    "zero_trades_21d",    
      "zero_trades_252d",
      "rvol_252d"
    ]
    
    # Exclude features without sufficient coverage
    feat_excl = ["capex_abn", "capx_gr2", "capx_gr3", "debt_gr3", "dgp_dsale",           
                   "dsale_dinv", "dsale_drec", "dsale_dsga", "earnings_variability", "eqnetis_at",          
                   "eqnpo_me", "eqpo_me", "f_score", "iskew_hxz4_21d", "ivol_hxz4_21d",       
                   "netis_at", "ni_ar1", "ni_inc8q", "ni_ivol", "niq_at", "niq_at_chg1", "niq_be", 
                   "niq_be_chg1", "niq_su", "ocfq_saleq_std", "qmj", "qmj_growth", "rd_me", 
                   "rd_sale", "rd5_at", "resff3_12_1", "resff3_6_1", "sale_gr3", "saleq_gr1", 
                   "saleq_su", "seas_16_20an", "seas_16_20na", "sti_gr1a", "z_score"
    ]
    if exclude_poor_coverage:
        # Filter out the excluded features
        features = [feature for feature in features if feature not in feat_excl]
    
    return features


def get_signals(exclude_poor_coverage = True):
    industry_codes = ['ff12_BusEq',
                      'ff12_Chems',
                      'ff12_Durbl',
                      'ff12_Enrgy',
                      'ff12_Hlth',
                      'ff12_Manuf',
                      'ff12_Money',
                      'ff12_NoDur',
                      'ff12_Other',
                      'ff12_Shops',
                      'ff12_Telcm',
                      'ff12_Utils'
                      ]
    signals = [get_features(exclude_poor_coverage),industry_codes]
    
    return signals
    

#%%
def long_horizon_ret(data, h=1, impute="zero"):
    """
    Compute long-horizon returns.
    
    Parameters:
      data: pandas DataFrame with at least columns 'id', 'eom', 'ret_exc'
      h: int, horizon (number of future periods)
      impute: str, imputation method in {"zero", "mean", "median"}
      
    Returns:
      DataFrame with future return columns ret_ld1, ..., ret_ld{h}.
    """
    # Ensure that the input data is not modified
    data = data.copy()
    
    #Ensure 'eom' is datetime
    data['eom'] = pd.to_datetime(data['eom'])
    
    # Only consider rows where ret_exc is not NA.
    valid_data = data.dropna(subset=["ret_exc"])
    
    # Get unique dates (for merging) present in valid_data.
    dates = valid_data[['eom']].drop_duplicates()
    dates = dates.rename(columns={'eom': 'merge_date'})
    
    # For each id, find the start and end dates where ret_exc is not missing.
    ids = valid_data.groupby('id')['eom'].agg(start='min', end='max').reset_index()
    
    # Create a cross between each security’s valid date range and the unique dates.
    # For each id, select the dates between start and end.
    id_dates = ids.merge(dates, how="cross")
    id_dates = id_dates[(id_dates['merge_date'] >= id_dates['start']) & 
                        (id_dates['merge_date'] <= id_dates['end'])]
    
    # Remove the extra start/end columns.
    id_dates = id_dates.drop(columns=["start", "end"]).rename(columns={"merge_date": "eom"})
    
    # Merge the full panel with the original return data.
    full_ret = pd.merge(id_dates, data[['id', 'eom', 'ret_exc']], on=["id", "eom"], how="left")
    full_ret = full_ret.sort_values(by=["id", "eom"]).reset_index(drop=True)
    
    # Create the lead return columns ret_ld1, ..., ret_ldh within each id group.
    for l in range(1, h+1):
        full_ret[f"ret_ld{l}"] = full_ret.groupby("id")["ret_exc"].shift(-l)
    
    # Drop the original ret_exc column.
    full_ret = full_ret.drop(columns=["ret_exc"])
    
    # Identify rows where all lead returns are missing.
    lead_cols = [f"ret_ld{l}" for l in range(1, h+1)]
    all_missing = full_ret[lead_cols].isna().sum(axis=1) == h
    perc_all_missing = all_missing.mean() * 100
    print(f"All missing excludes {perc_all_missing:.2f}% of the observations")
    
    # Keep only rows where not all lead returns are missing.
    full_ret = full_ret[~all_missing].reset_index(drop=True)
    
    # Impute missing values in the lead columns based on the specified method.
    if impute == "zero":
        full_ret[lead_cols] = full_ret[lead_cols].fillna(0)
    elif impute == "mean":
        # Replace NA with the mean of the column within each month (eom group).
        full_ret[lead_cols] = full_ret.groupby("eom")[lead_cols].transform(lambda x: x.fillna(x.mean()))
    elif impute == "median":
        full_ret[lead_cols] = full_ret.groupby("eom")[lead_cols].transform(lambda x: x.fillna(x.median()))
    
    return full_ret


#%%

# Wealth Calculation Function
def wealth_func(wealth_end, end, market, risk_free):
    """
    Compute the wealth trajectory backwards from a given initial wealth
    level. Since backtesting works backwards, for a given initial wealth level
    we need to know how our strategy would have performed in the past to evaluate
    it.
    
    Wealth is assumed to grow exogenously at the market return.
    
    Parameters:
      wealth_end: final wealth value
      end: final date (as a pd.Timestamp)
      market: DataFrame with market returns, containing 'eom_ret' and 'mkt_vw_exc'
      risk_free: DataFrame with risk-free data, containing 'eom' and 'rf'
      
    Returns:
      A DataFrame with columns: 'eom' (end-of-month), 'wealth', and 'mu_ld1' (total return for the period)
    """
    # Rename and merge
    risk_free = risk_free.rename(columns={"eom": "eom_ret"})
    wealth = pd.merge(risk_free[["eom_ret", "rf"]], market, on="eom_ret", how="left")

    # Compute total return
    wealth["tret"] = wealth["mkt_vw_exc"] + wealth["rf"]

    # Filter for dates up to 'end'
    wealth = wealth[wealth["eom_ret"] <= end]

    # Sort descending by date and compute cumulative wealth backward
    wealth = wealth.sort_values("eom_ret", ascending=False)
    wealth["wealth"] = (1 - wealth["tret"]).cumprod() * wealth_end

    # Final output formatting
    wealth["eom"] = (wealth["eom_ret"].dt.to_period("M").dt.to_timestamp("M") - pd.offsets.MonthEnd(1))
    wealth["mu"] = wealth["tret"]

    result = wealth[["eom", "wealth", "mu"]].copy()

    # Add one row for the exact 'end' date
    row = pd.DataFrame([{"eom": end, "wealth": wealth_end, "mu": np.nan}])
    result = pd.concat([result, row], ignore_index=True)

    # Sort chronologically
    result = result.sort_values("eom").reset_index(drop=True)

    return result

def create_cov(x, ids=None):
    """
    Create the Barra Covariance Matrix 
    
    Parameters:
    x (dict): Dictionary containing 'fct_load', 'ivol_vec', and 'fct_cov'
    ids (array-like, optional): List of Stock IDs to subset the data
    
    Returns:
    numpy.ndarray: The computed covariance matrix
    """
    ################## Compute the Covariance Matrix ##########################
    # Extract the relevant loadings and ivol
    if ids is None:
        load = x['fct_load']
        ivol = x['ivol_vec']
    else:
        # Convert ids to strings to match R's behavior with as.character()
        load = x['fct_load'].loc[[i for i in ids]]
        ivol = x['ivol_vec'].loc[[i for i in ids]]
    
    # Create the covariance matrix
    sigma = load @ x['fct_cov'] @ load.T + np.diag(ivol) 
    
    ################## Error Correction ##########################
    """
    In case a stock variance is negative. Can happen as (37) is not exactly an
    equality. Due to the EWMA the residuals are not uncorrelated with the fitted values.
    """
    if min(np.diag(sigma)) < 0:
        # Get diagonal indices
        diag_indices = np.arange(len(sigma))
        print("Warning: Negative Variances:", diag_indices)
        
        """
        # Extract the diagonal
        diag_values = sigma.values[diag_indices, diag_indices]
        
        # Find the minimum of the positive diagonal values and use them to replac ethe negatives
        positive_diag = diag_values[diag_values > 0]
        if len(positive_diag) > 0:
            min_positive = positive_diag.min()
        
            # Create a boolean mask where diagonal elements are zero
            mask = diag_values < 0
        
            # Replace zero diagonal elements with min_positive
            sigma.values[diag_indices[mask], diag_indices[mask]] = min_positive
        """
            
    return sigma

#%%

#Categories Industries According to Fama-French 12 Industries
#Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_12_ind_port.html
#Categories can also be found in the data folder
def categorize_sic(sic):
    """
    Assigns Fama-French 12 industry classification based on SIC codes.

    Parameters:
    -----------
    chars : pd.DataFrame
        DataFrame containing a 'sic' column with Standard Industrial Classification codes
        
    Returns:
    --------
    pd.Series
        Series containing industry classifications according to Fama-French 12 categories:
        - NoDur: Non-durable Consumer Goods
        - Durbl: Durable Consumer Goods
        - Manuf: Manufacturing
        - Enrgy: Energy
        - Chems: Chemicals
        - BusEq: Business Equipment
        - Telcm: Telecommunications
        - Utils: Utilities
        - Shops: Shops (Wholesale/Retail)
        - Hlth: Healthcare
        - Money: Financial Services
        - Other: All other industries
        
    Note:
    -----
    Follows the standard Fama-French 12 industry classification scheme based on SIC code ranges.
    """
    # Non-Durables
    if ((100 <= sic <= 999) or
        (2000 <= sic <= 2399) or
        (2700 <= sic <= 2749) or
        (2770 <= sic <= 2799) or
        (3100 <= sic <= 3199) or
        (3940 <= sic <= 3989)):
        return "NoDur"
    
    # Durables
    elif ((2500 <= sic <= 2519) or
          (3630 <= sic <= 3659) or
          (sic in [3710, 3711, 3714, 3716, 3750, 3751, 3792]) or
          (3900 <= sic <= 3939) or
          (3990 <= sic <= 3999)):
        return "Durbl"
    
    # Manufacturing
    elif ((2520 <= sic <= 2589) or
          (2600 <= sic <= 2699) or
          (2750 <= sic <= 2769) or
          (3000 <= sic <= 3099) or
          (3200 <= sic <= 3569) or
          (3580 <= sic <= 3629) or
          (3700 <= sic <= 3709) or
          (3712 <= sic <= 3713) or
          (sic in [3715]) or
          (3717 <= sic <= 3749) or
          (3752 <= sic <= 3791) or
          (3793 <= sic <= 3799) or
          (3830 <= sic <= 3839) or
          (3860 <= sic <= 3899)):
        return "Manuf"
    
    # Energy
    elif ((1200 <= sic <= 1399) or
          (2900 <= sic <= 2999)):
        return "Enrgy"
    
    # Chemicals
    elif ((2800 <= sic <= 2829) or
          (2840 <= sic <= 2899)):
        return "Chems"
    
    # Business Equipment
    elif ((3570 <= sic <= 3579) or
          (3660 <= sic <= 3692) or
          (3694 <= sic <= 3699) or
          (3810 <= sic <= 3829) or
          (7370 <= sic <= 7379)):
        return "BusEq"
    
    # Telecommunications
    elif (4800 <= sic <= 4899):
        return "Telcm"
    
    # Utilities
    elif (4900 <= sic <= 4949):
        return "Utils"
    
    # Shops
    elif ((5000 <= sic <= 5999) or
          (7200 <= sic <= 7299) or
          (7600 <= sic <= 7699)):
        return "Shops"
    
    # Health
    elif ((2830 <= sic <= 2839) or
          (sic == 3693) or
          (3840 <= sic <= 3859) or
          (8000 <= sic <= 8099)):
        return "Hlth"
    
    # Money
    elif (6000 <= sic <= 6999):
        return "Money"
    
    # Other
    else:
        return "Other"
    
#%%
def portfolio_return_BFGS(logits, pi_tm1, gt, Sigma, KL, wealth, return_predictions, gamma):
    
    #Get current portfolio (in levels)
    pi_t = np.exp(logits)
    pi_t /= np.sum(pi_t)
    
    #Compute Revenue
    revenue = pi_t.T @ return_predictions 
    
    #Compute Variance penalty
    var_pen = gamma/2 * pi_t.T @ Sigma @ pi_t
    
    #Compute transaction costs
    change_pf = pi_t-pi_tm1 
    tc = 0.5* wealth * np.sum(KL * change_pf**2)
    
    return -(revenue - tc - var_pen)

#%% Return Prediction

def load_signals_rollingwindow(
    db_conn: sqlite3.Connection,
    settings: dict,
    target: str,
    rank_signals: bool,
    trade_start: str,
    trade_end: str, 
    fill_missing_values: bool
) -> tuple[pd.DataFrame | None, np.ndarray | None, int | None]:
    """
    Reads, merges, and preprocesses signal and target data for a rolling window analysis.

    Args:
        db_conn: An active SQLite database connection object (e.g., to JKP_clean.db).
        settings: A dictionary containing 'rolling_window' configuration.
        target: Prediction Target
        rank_signals: If True, queries the 'Signals_Rank' table; otherwise,
                          queries the 'Signals_ZScore' table.
        fill_missing_values: If True, fills NaNs of features with zero (industry classifiers have no missing values).

    Returns:
        A tuple containing:
        - df_signals: The merged and cleaned DataFrame (or None on failure).
        - signal_months: A numpy array of unique 'eom' dates in the signals data (or None on failure).
        - trade_idx: The index in signal_months where the rolling window process starts (or None on failure).
    """
    # 1. Configuration Extraction
    window_size = settings['rolling_window']['window_size']
    validation_size = settings['rolling_window']['validation_periods']

    # Determine which signals table to query
    signal_table = "Signals_Rank" if rank_signals else "Signals_ZScore"

    # Trading Dates (Hardcoded end date from original snippet)
    trading_dates = pd.date_range(trade_start, trade_end, freq='ME')

    # Calculate the required data before 1st trading date
    required_lag = window_size + validation_size + 1
    query_start_date = trading_dates[0] - pd.offsets.MonthEnd(required_lag)
    query_start_date_str = query_start_date.strftime("%Y-%m-%d")
    
    #Feature columns
    signals = get_signals()
    feat_cols = signals[0] + signals[1]

    # 2. Read Signals Data
    query_signals = (f"SELECT * FROM {signal_table} "
                     f"WHERE eom >= '{query_start_date_str}' "
                     f"AND eom <= '{(pd.to_datetime(trade_end)+pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")}'"
                     )

    df_signals = pd.read_sql_query(
        query_signals,
        con=db_conn,
        parse_dates={'eom'}
    )

    # 3. Read Targets Data
    query_targets = (f"SELECT id, eom, {target} FROM Factors_processed "
                     f"WHERE eom >= '{query_start_date_str}' "
                     f"AND eom <= '{(pd.to_datetime(trade_end)+pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")}'"
                     )

    df_targets = pd.read_sql_query(
        query_targets,
        con=db_conn,
        parse_dates={'eom'}
    )

    # 4. Merge and Pre-process
    df_signals = (df_signals
                  .merge(df_targets, on=['id', 'eom'], how='left')
                  #!!!! BUG !!!! Will drop the last date of the sample.
                  # Drop rows where target (tr_m_sp500_ld1) is missing.
                  .dropna(subset=[target])
                  .sort_values(by=['eom', 'id'])
                 )

    # Sanity check from original code (should be 0 after dropna)
    if df_signals[target].isna().sum() > 0:
        print("ERROR: Missing Values in Target after merge and dropna. This is unexpected.")

    # 5. Fill Missing Values in Signals
    if fill_missing_values:
        df_signals[signals[0]] = df_signals[signals[0]].fillna(0)

    # 6. Calculate outputs
    signal_months = df_signals['eom'].sort_values().unique()

    # Find the index of the first trading date in the signal months array
    trade_idx = np.searchsorted(signal_months, trading_dates[0])

    return df_signals, signal_months, trade_idx, feat_cols, window_size, validation_size

#%% Portfolio Choices

def load_portfolio_backtest_data(con, start_date, sp500_ids, path, predictor):
    """
    Load and assemble all inputs required for the portfolio backtest.

    This function pulls data for the investable universe (S&P 500 subset),
    Kyle's lambda, market equity, realised returns, the exogenous AUM
    evolution, and the Barra-style covariance matrices. It also constructs
    the initial portfolio weights (value-weighted) 
    and the exogenous AUM growth factor `g_t`
    used in the portfolio optimisation.

    Parameters
    ----------
    con : sqlite3.Connection
        Open SQLite connection to the JKP/Factors database that contains
        the table ``Factors_processed``.
    start_date : str or pandas.Timestamp
        First end-of-month date (inclusive) for which the backtest should
        be run. Data for the portfolio initialisation is pulled from
        ``start_date - 1 month`` as well.
    sp500_ids : str
        String of comma-separated stock identifiers (e.g. PERMNOs) used in
        the SQL ``IN (...)`` clause to restrict the universe to S&P 500
        constituents over time.
    path : str or pathlib.Path
        Root path to the project directory that contains the
        ``Data/`` subdirectory. Used to locate
        ``wealth_evolution.csv`` and ``Barra_Cov.pkl``.
    predictor : str
        Name of the return-forecasting method. Currently, the special
        value ``"Myopic Oracle"`` selects realised returns from the JKP
        dataset as the forecast target (perfect foresight benchmark).

    Returns
    -------
    df_pf_weights : pandas.DataFrame
        DataFrame of portfolio weights and AUM growth factors with columns:

        - ``id`` : stock identifier
        - ``eom`` : end-of-month timestamp
        - ``pi`` : portfolio weight at the beginning of the month;
          initial month is value-weighted by market equity, subsequent
          months are initialised with a small positive value (``1e-16``)
        - ``g`` : exogenous AUM growth factor g_t^w used to construct
          G_t π_{t-1}. For the first month, ``g = 1``; thereafter,
          ``g = (1 + tr) / (1 + mu)``, where ``tr`` is the stock return
          and ``mu`` is the benchmark/market return.

    df_kl : pandas.DataFrame
        Kyle's lambda (price impact) data with columns:

        - ``id``
        - ``eom``
        - ``lambda`` : Kyle's lambda at the beginning of each month.

    df_me : pandas.DataFrame
        Market equity (size) data with columns:

        - ``id``
        - ``eom``
        - ``me`` : market equity at the beginning of each month.

    df_returns : pandas.DataFrame
        Realised return data used for forecasts and evaluation.
        For ``predictor == "Myopic Oracle"`` this contains:

        - ``id``
        - ``eom``
        - ``tr`` : total stock return over month t
        - ``tr_ld1`` : lagged one-month return
        - ``tr_m_sp500`` : excess return over the SP500 in month t
        - ``tr_m_sp500_ld1`` : lagged one-month excess return over SP500.

    df_wealth : pandas.DataFrame
        Exogenous AUM evolution with at least columns:

        - ``eom`` : end-of-month timestamp
        - ``wealth`` : assets under management at the beginning of month t
        - ``mu`` : market / benchmark return used to compute g_t^w

        Only rows with ``eom >= start_date - 1 month`` are kept.

    dict_barra : dict
        Dictionary mapping end-of-month dates (keys) to Barra-style
        covariance model objects (values). Only entries with date
        ``>= start_date`` are retained. Each value is expected to be
        consumable by ``GF.create_cov(dict_barra[date])`` to produce
        the stock-level covariance matrix Σ_t.

    """
    #---- Data for the investable universe ----
    query = ( "SELECT id, eom, in_sp500, me, lambda, tr, tr_ld1, tr_m_sp500, tr_m_sp500_ld1 "
             + "FROM Factors_processed "
             + f"WHERE eom >= '{start_date}' "
             + f"AND id IN ({sp500_ids})")

    df = (pd.read_sql_query(query,
                           parse_dates = {'eom'},
                           con=con
                           )
          .sort_values(by = ['eom', 'id'])
          .assign(in_sp500 = lambda df: df['in_sp500'].astype('boolean'))
          )

    #---- Subset Kyle's Lambda ----
    df_kl = df.get(['id', 'eom', 'lambda'])
    
    #---- Subset Market Equity ----
    df_me = df.get(['id', 'eom', 'me'])
    
    #---- Subset Realised Returns ----
    if predictor == "Myopic Oracle":
        df_returns = df.get(['id','eom','tr','tr_ld1','tr_m_sp500','tr_m_sp500_ld1'])

    #---- Evolution AUM ----
    df_wealth = pd.read_csv(path + "Data/wealth_evolution.csv", parse_dates=['eom'])
    df_wealth = df_wealth.loc[df_wealth['eom'] >= pd.to_datetime(start_date) - pd.offsets.MonthEnd(1)]

    #---- Initialise DataFrame for Portfolio Weights ----
    df_pf_weights = df.loc[df['in_sp500']].get(['id','eom','me', 'tr']) 

    #Compute initial value weighted portfolio
    df_pf_weights = (
        df_pf_weights
        # 1. Filter rows where 'eom' is the target date or later
        .pipe(lambda df: df.loc[df['eom'] >= pd.to_datetime(start_date) - pd.offsets.MonthEnd(1)])
        # 2. Calculate the aggregate market cap per date
        .assign(group_sum=lambda df: df.groupby('eom')['me'].transform('sum'))
        # 3. Calculate a value-weighted initial portfolio
        .assign(pi=lambda df: df['me'] / df['group_sum'])
        # 4. Set all portfolio weights to zero if 'eom' > min_date
        .assign(pi=lambda df: np.where(df['eom'] > df['eom'].min(), 1e-16, df['pi']))
        .merge(df_wealth[['eom', 'mu']], on=['eom'], how='left')
        # 5. Calculate 'g'
        .assign(
            is_min_eom=lambda df: df['eom'] == df['eom'].min(),
            g=lambda df: np.where(
                df['is_min_eom'],
                1,
                (1 + df['tr']) / (1 + df['mu'])
            )
        )
        # 6. Clean up
        .drop(columns=['group_sum', 'is_min_eom', 'mu', 'me', 'tr'])
        .sort_values(by = ['eom','id'],ascending = [True,True])   
    )

    #---- Barra Covariance Matrix ----
    # Load Barra covariance
    with open(path + "Data/Barra_Cov.pkl", "rb") as f:
        dict_barra_all = pickle.load(f)
        
    dict_barra = {
        k: v for k, v in dict_barra_all.items() 
        if k >= pd.to_datetime(start_date)
    }

    print("Data loading complete.")
    
    return df_pf_weights, df_kl, df_me, df_returns, df_wealth, dict_barra

def load_MLpredictions(DataBase, ensemble:list):
    """
    Load and merge ML model return predictions.

    All ML models from ``ensemble``, are merged into one DataFrame.

    Note: At date ``eom``, the predictions are for the return at ``eom`` +1

    Only rows with non-missing values across all ensemble members
    are retained (rows with any NaNs are dropped).
    """
    
    #Initialise Dataframe 
    df = None
    
    #Loop over models in ensemble
    for model in ensemble:
        #Load Predictions
        df_next = pd.read_sql_query(f"SELECT * FROM {model}", con=DataBase, parse_dates={"eom"})
        if df is None:
            df = df_next
        else:
            #Merge to existing dataframe
            df = df.merge(df_next, on=['id','eom'], how='outer')
    
    return df

#%% Performance Ratios

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
    """Calculates Annualized Alpha and the Beta."""
    y = strategy[return_col_strat] - risk_free[risk_free['eom'].isin(strategy['eom'].unique())]['rf']
    x = benchmark[return_col_bench] - risk_free[risk_free['eom'].isin(strategy['eom'].unique())]['rf']
    
    # Linear Regression: y = alpha + beta * x
    beta, alpha_monthly, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Annualize Alpha
    alpha_annualized = alpha_monthly*12 
    return alpha_annualized, beta
                                        
def TurnoverAUMweighted(strategy, df_wealth):
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
        df_wealth
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

    return aum_weighted_turnover
