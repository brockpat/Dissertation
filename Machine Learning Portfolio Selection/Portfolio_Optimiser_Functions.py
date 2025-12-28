# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 10:03:56 2025

@author: Patrick
"""

#%% Libraries
import numpy as np
import pandas as pd
import pickle

#%%

def load_portfolio_backtest_data(con, start_date, sp500_ids, path, predictor):
    """
    Load and assemble all inputs required for the portfolio backtest.

    This function pulls data for the investable universe (S&P 500 subset),
    Kyle's lambda, market equity, realised returns, the exogenous AUM
    evolution, and the Barra-style covariance matrices. It also constructs
    the initial portfolio weights and the exogenous AUM growth factor `g_t`
    used in the optimisation.

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

    Notes
    -----
    The function assumes that:

    * The table ``Factors_processed`` contains the columns
      ``['id', 'eom', 'in_sp500', 'me', 'lambda', 'tr', 'tr_ld1',
      'tr_m_sp500', 'tr_m_sp500_ld1']``.
    * ``wealth_evolution.csv`` contains at least ``['eom', 'mu']`` and
      typically also ``'wealth'``.
    * ``Barra_Cov.pkl`` is a pickled dictionary keyed by dates.
    """
    #---- Data for investable universe ----
    query = ( "SELECT id, eom, in_sp500, me, lambda, tr, tr_ld1, tr_m_sp500, tr_m_sp500_ld1 "
             + "FROM Factors_processed "
             + f"WHERE eom >= '{start_date}'"
             + f"AND id IN ({sp500_ids})")

    df = (pd.read_sql_query(query,
                           parse_dates = {'eom'},
                           con=con
                           )
          .sort_values(by = ['eom', 'id'])
          .assign(in_sp500 = lambda df: df['in_sp500'].astype('boolean'))
          )

    #---- Kyle's Lambda ----
    df_kl = df.get(['id', 'eom', 'lambda'])
    
    #---- Market Equity ----
    df_me = df.get(['id', 'eom', 'me'])

    #---- Evolution AUM ----
    df_wealth = pd.read_csv(path + "Data/wealth_evolution.csv", parse_dates=['eom'])
    df_wealth = df_wealth.loc[df_wealth['eom'] >= pd.to_datetime(start_date) - pd.offsets.MonthEnd(1)]

    #---- Return Forecasts ----
    #Extract individual dataframes
    if predictor == "Myopic Oracle":
        df_returns = df.get(['id','eom','tr','tr_ld1','tr_m_sp500','tr_m_sp500_ld1']) #Actual Returns


    #---- Initialise DataFrame for Portfolio Weights ----
    df_pf_weights = df.loc[df['in_sp500']].get(['id','eom','me', 'tr']) # Portfolio weights

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

def get_universe_partitions(prev_date, date, df_pf_weights):
    """
    Partitions the stock universe into stayers, leavers, and newcomers.
    """
    
    #Stock universes
    prev_universe = set(df_pf_weights.loc[df_pf_weights['eom'] == prev_date]['id'])
    cur_universe = set(df_pf_weights.loc[df_pf_weights['eom'] == date]['id'])
    
    #Stocks that can no longer be in the portfolio
    leavers = list(prev_universe - cur_universe)
    #Stocks that can newly enter the portfolio
    newcomers = list(cur_universe - prev_universe)
    #Stocks that can remain the portfolio
    stayers = list(cur_universe.intersection(prev_universe))
    #Stocks which can have non-zero portfolio weights and are active choice variables
    active = sorted(list(set(newcomers + stayers)))
    
    #Stocks for which pi_t = 0 must be enforced. This affects all leavers.
    #On top of that, it can affect a subset of newcomers due to missing data 
    #on Kyle's Lambda or covariance-matrix
    zeros = leavers.copy()
    
    return stayers, leavers, newcomers, active, zeros


def portfolio_return_BFGS(logits, pig_tm1, gt, Sigma, KL, wealth, return_predictions, gamma):
    
    #Get current portfolio (in levels)
    pi_t = np.exp(logits)
    pi_t /= np.sum(pi_t)
    
    #Compute Revenue
    revenue = pi_t.T @ return_predictions 
    
    #Compute Variance penalty
    var_pen = gamma/2 * pi_t.T @ Sigma @ pi_t
    
    #Compute transaction costs
    change_pf = pi_t-pig_tm1 
    tc = 0.5* wealth * np.sum(KL * change_pf**2)
    
    return -(revenue - tc - var_pen)