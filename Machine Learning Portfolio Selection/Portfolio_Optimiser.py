# -*- coding: utf-8 -*- 
"""
Created on Sun Oct 26 14:13:11 2025

@author: patrick

Begin of period variables are known at the beginning of time 't'. This implies
that they are computed from information from t-1,t-2,...

The following variables are beginn of period:

'pi'
'wealth' (AUM) 

'Sigma' (Barra Cov)

'g' 

'lambda' (Kyle's Lambda)

All other variables are at the end of the period (month).
"""

#%% Libraries

path = "C:/Users/patri/Desktop/ML/"

#DataFrame Libraries
import pandas as pd
import sqlite3
import pickle

#Turn off pandas performance warnings
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

#Plot Libraries
import matplotlib.pyplot as plt

#Scientifiy Libraries
import numpy as np

#Gradient Ascent
import torch
import torch.nn.functional as F

import statsmodels.api as sm

#Saving results
from copy import deepcopy
import json
import re

#Custom Functions
import os
os.chdir(path + "Code/")
import General_Functions as GF
settings = GF.get_settings()

#%% Macros
""" 
Being able to invest in the riskless asset changes nothing as from the value-weighted
position shifting to the risky asset, which offers an extremely low return, is not
worth it in light of the transaction costs
"""
include_rf_asset = False
if include_rf_asset:
    new_permno = 100_000 #Fictional permno of risk-free asset. Must not overlap with any existing permno in the investable universe

#%% Functions: Portfolio Selection

def get_universe_partitions(prev_date, date, df_pf_weights):
    """
    Partition the stock universe at trading date t
    into stayers, leavers, newcomers and helpers.

    * ``stayers``  : stocks that remain in the S&P500.
    * ``leavers``  : stocks that leave the S&P500 (must be fully liquidated --> π_t = 0).
    * ``newcomers``: stocks newly added to the S&P500 (no previous portfolio weight available)
    * ``active``   : sorted list of stocks that are candidates for non-zero
                     portfolio weights at the trading date
    * ``zeros``    : Stocks that must have π_t = 0; starts as
                     the list of leavers and may be extended later (e.g. for
                     missing Kyle's lambda or missing covariance/return data).

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

def optimise_portfolio(
    df_pf_weights: pd.DataFrame,
    df_kl: pd.DataFrame,
    df_me: pd.DataFrame,
    dict_barra: dict,
    df_returns: pd.DataFrame,
    df_wealth: pd.DataFrame,
    df_spy: pd.DataFrame,
    df_retPred: pd.DataFrame,
    trading_dates,
    prediction_col: str ,
    include_rf_asset: bool, 
    flat_MaxPi: bool,
    flat_MaxPi_limit: float,
    w_upperLimit: float,
    w_lowerLimit: float,
    vol_scaler: float,
    tc_scaler: float,
    Ridge = False
) -> pd.DataFrame:
    
    """
    
    Solves the one-period myopic portfolio problem for each trading date.

    For each month t in ``trading_dates``, this function:

    1. Determines the investable universe (stayers, leavers, newcomers).
        Stock with with missing values for any of Kyle's lambda, covariance, or predictions
        must be completely sold off.
    2. Builds the Barra-style covariance matrix Σ_t for the active universe.
    3. Assembles all required inputs (estimated returns, Kyle's lambda,
       lagged portfolio weights, benchmark variance, AUM).
    4. Solves the myopic portfolio optimisation via
       gradient ascent in PyTorch.

    5. Stores the resulting weights, realised revenue and transaction costs.

    Args:
        df_pf_weights: DataFrame of current portfolio weights π_t for each
            stock and date. Must contain ``['id', 'eom', 'pi', 'g']`` where
            ``g`` is the growth factor g_t^w used to compute the
            drifted portfolio weights ( G_t π_{t-1} ).
        df_kl: Kyle's lambda data. Requires columns ``['id', 'eom', 'lambda']``.
        df_me: Market equity / size data. Used to construct upper (and lower)
            bounds on portfolio weights relative to benchmark weights.
        dict_barra: Dictionary mapping dates to Barra covariance matrix.
        df_returns: DataFrame containing realised stock returns, used to
            compute realised revenue; columns ``['id', 'eom', 'tr']``.
        df_wealth: DataFrame of AUM (wealth) per date; columns
            ``['eom', 'wealth']``.
        df_spy: Benchmark (SPY) data with at least columns
            ``['eom', 'variance']`` giving benchmark return variance per month.
        df_retPred: DataFrame with model-based expected returns. Must include
            ``['id', 'eom', prediction_col]`` for at least the trading dates.
            Note that prediction for trading date t is made at date t-1.
        trading_dates: Iterable of end-of-month timestamps indicating the
            portfolio rebalancing dates.
        prediction_col: Column name in ``df_retPred`` containing the expected
            (excess) returns used in the objective.
        w_upperLimit: Scalar multiple applied to benchmark weights to obtain
            the maximum allowed portfolio weight for each stock.
        w_lowerLimit: Scalar multiple applied to benchmark weights to obtain
            the minimum allowed portfolio weight for each stock.
        vol_scaler: Scaling factor applied to the benchmark variance constraint
            σ_B^2; >1 relaxes, <1 tightens the variance limit.
        tc_scaler: Scaling factor applied to Kyle's lambda to run
            sensitivity analyses on transaction costs.

    Returns:
        pd.DataFrame: Long dataframe with one row per (id, eom) containing:

            - optimal portfolio weight π_t
            - Kyle's lambda
            - lagged G_t π_{t-1} (pi_g_tm1)
            - realised end-of-period return ``tr``
            - Realised revenue ``rev = π_t * tr``
            - transaction cost ``tc``

        The returned dataframe aggregates the strategy for all trading dates.
    """

    # Ensure for each trading date there is a prediction
    trading_dates = list(sorted(set(trading_dates).intersection(set(df_retPred.eom)).copy()))
    #Add one more date as for trading date t, the predictions are based on date t-1
    trading_dates = trading_dates + [trading_dates[-1] + pd.offsets.MonthEnd(1)]

    #Container to store results
    df_strategy = []    
    
    # Scale Transaction costs
    df_kl = df_kl.copy() #avoid overwriting the function input
    df_kl['lambda'] = df_kl['lambda']*tc_scaler

    #Loop over trading date
    for date in trading_dates:
        print(date)

        #=====================================
        #             Preliminaries
        #=====================================
        
        #Previous Date (the information we possess)
        prev_date = date - pd.offsets.MonthEnd(1)
        
        #Stock universe
        stayers, leavers, newcomers, \
            active, zeros = get_universe_partitions(prev_date, date, df_pf_weights)
            
        #Compute Barra covariance matrix
        Sigma = GF.create_cov(dict_barra[prev_date])
        
        if include_rf_asset: 
        # This is easiest to include inside this function. 
        # The remaining objects are appended outside of this function.
          
            # Extend Labels
            new_labels = Sigma.index.union([new_permno])

            # Reindex the dataframe on BOTH axes (index and columns) 
            # fill_value=0 ensures the new row/col are all zeros
            Sigma = Sigma.reindex(index=new_labels, columns=new_labels, fill_value=0)
        
        #Get return predictions
        df_return_predictions = (df_retPred
                              .loc[df_retPred['eom'] == prev_date]
                              .get(['id', 'eom', prediction_col])
                              )
        
        #=====================================================
        #   Shrink active universe in case of missing data
        #=====================================================
        
        active, zeros = shrink_universe(prev_date, active, newcomers, zeros,
                            df_kl, Sigma, df_return_predictions)
        
        #========================================
        #   Reduce DataFrames to active universe 
        #========================================
        
        Sigma_active, df_kl_active, df_ret_pred_active = reduce_to_active(prev_date, active, Sigma, df_kl, df_return_predictions)
        
        # Save Workspace
        del Sigma
        
        #==========================================================
        #   Build DataFrame for Portfolio Optimisation
        #==========================================================
        
        df_pf_t = build_portfolio_dataframe(date, prev_date,
                                      active, stayers, newcomers, leavers, zeros,
                                      df_pf_weights, df_kl, df_returns,
                                      df_me)
        
        #==========================================================
        #           Solve for optimal portfolio
        #==========================================================
        
        # Solve for optimal portfolio weight for stocks in the active universe
        pi_opt = \
            solve_pf_optimisation(prev_date, date, 
                                  active, 
                                  df_pf_t, df_kl_active, df_ret_pred_active, Sigma_active, 
                                  df_me, df_spy, df_wealth, prediction_col, include_rf_asset,
                                  flat_MaxPi, flat_MaxPi_limit,
                                  w_upperLimit, w_lowerLimit, vol_scaler, tc_scaler, Ridge)
        
        # Convert to Series
        pi_opt = pd.Series(pi_opt, index=active, name='pi_opt')
        
        # Merge to Dataframe
        df_pf_t = df_pf_t.merge(pi_opt, left_on='id', right_index=True, how='left')
        
        # Overwrite previous values
        df_pf_t.loc[df_pf_t['id'].isin(active), 'pi'] = df_pf_t['pi_opt']
        
        # Drop auxiliary column
        df_pf_t = df_pf_t.drop(columns='pi_opt')
        
        # print pi_max value
        print(f"   MAX pi: {df_pf_t['pi'].max()}") 
        
        #==========================================================
        #           Compute Revenue & TC
        #==========================================================
        
        # Get Baseline wealth level
        w = df_wealth[df_wealth['eom'] == date]['wealth'].iloc[0]
        
        # Compute revenue & transaction costs
        df_pf_t = (df_pf_t
                .assign(rev = lambda df: df['pi']*df['tr'])
                # Important: Transaction cost scale (tc_scaler) is absorbed into df['lambda']
                .assign(tc = lambda df: (df['pi'] - df['pi_g_tm1'])**2 * df['lambda'] * float(w)/2 )
                )
        
        # Add to main DataFrame
        df_pf_weights = df_pf_weights.set_index(['id', 'eom'])
        df_pf_weights.update(df_pf_t.set_index(['id', 'eom'])[['pi']])
        df_pf_weights = df_pf_weights.reset_index()
        
        #Append Result
        df_strategy.append(df_pf_t)
        
    #Make DataFrame of strategy
    df_strategy = pd.concat(df_strategy)
    
    return df_strategy, df_pf_weights


def shrink_universe(prev_date, active, newcomers, zeros,
                    df_kl, Sigma, return_predictions):
    """
    Shrink the active stock universe based on data availability.

    This function enforces that every stock in the active universe has:
    - Kyle's lambda at ``prev_date``
    - a finite variance entry in the Barra covariance matrix ``Sigma``
    - a return prediction for ``prev_date``.

    Any stock that fails one of these checks is added to ``zeros``
    (i.e. forced to have π_t = 0), and removed from the ``active`` set.

    Parameters
    ----------
    prev_date : pandas.Timestamp
        Previous end-of-month date (t-1), i.e. the information date.
    active : list of hashable
        List of stock ids that are candidates for non-zero portfolio
        weights at date t.
    newcomers : list of hashable
        Stocks that newly enter the investable universe at date t.
    zeros : list of hashable
        Stocks that must have π_t = 0. Initially contains leavers and
        is extended in this function.
    df_kl : pandas.DataFrame
        Kyle's lambda data with columns ``['id', 'eom', 'lambda']``.
    Sigma : pandas.DataFrame
        Full Barra covariance matrix for ``prev_date`` with stock ids
        as both index and columns.
    return_predictions : pandas.DataFrame
        DataFrame with at least columns ``['id', 'eom', prediction_col]``
        for ``eom == prev_date``.

    Returns
    -------
    active : list
        Sorted list of stock ids that remain eligible for non-zero
        weights after all data-availability checks.
    zeros : list
        Updated list of stock ids that must have π_t = 0.
    """
    
    #--- Kyle's lambda ---
    kl_prev = df_kl.loc[df_kl['eom'] == prev_date, 'id']
    zeros.extend([s for s in newcomers if s not in set(kl_prev)])
    active = sorted([s for s in active if s not in zeros])

    #--- Barra Covariance ---
    Sigma = Sigma.reindex(index=active, columns=active).copy()

    zeros.extend(list(Sigma.index[pd.isna(np.diag(Sigma))]))
    active = sorted([s for s in active if s not in zeros])
    
    # --- Return predictions ---
    zeros.extend([s for s in active if s not in return_predictions['id'].values])
    active = sorted([s for s in active if s not in zeros])

    return active, zeros

def reduce_to_active(prev_date, active, Sigma, df_kl, return_predictions):
    """
    Restrict inputs to the active stock universe and sort by id.
    
    This function:
    1. Slices the Barra covariance matrix ``Sigma`` to the active ids.
    2. Extracts Kyle's lambda for the active ids at ``prev_date``.
    3. Restricts return predictions to the active ids.
    
    All returned objects are sorted by ``id`` so that they are aligned
    for vectorised operations in PyTorch.
    
    Parameters
    ----------
    prev_date : pandas.Timestamp
        Previous end-of-month date (t-1).
    active : list of hashable
        List of stock ids that are eligible for non-zero portfolio
        weights at date t.
    Sigma : pandas.DataFrame
        Full Barra covariance matrix (index and columns are stock ids).
    df_kl : pandas.DataFrame
        Kyle's lambda data with columns ``['id', 'eom', 'lambda']``.
    return_predictions : pandas.DataFrame
        DataFrame with columns ``['id', 'eom', prediction_col]`` for
        at least ``eom == prev_date``.
    
    Returns
    -------
    Sigma_active : pandas.DataFrame
        Covariance matrix restricted to the active universe, with both
        index and columns equal to ``active`` (sorted).
    df_kl_active : pandas.DataFrame
        Kyle's lambda for active stocks at ``prev_date``, sorted by id.
    return_predictions_active : pandas.DataFrame
        Return predictions for active stocks at ``prev_date``,
        sorted by id.
    """
    
    #Covariance Matrix (sorted by 'id' as active is sorted)
    Sigma = Sigma.loc[active, active]

    kyles_lambda = (df_kl
                    .loc[(df_kl['eom'] == prev_date) & (df_kl['id'].isin(active))]
                    .sort_values('id')
                    .reset_index(drop=True))

    return_predictions = (return_predictions
                          .loc[return_predictions['id'].isin(active)]
                          .sort_values('id')
                          .reset_index(drop=True))

    return Sigma, kyles_lambda, return_predictions

def build_portfolio_dataframe(date, prev_date,
                              active, stayers, newcomers, leavers, zeros,
                              df_pf_weights, df_kl, df_returns,
                              df_me):
    """
    Build the per-period portfolio dataframe used in optimisation.
    
    Constructs a DataFrame with one row per stock in the union of
    stayers, newcomers and leavers at trading date ``date``. It
    initialises portfolio weights, merges Kyle's lambda, realised
    returns, and computes the drifted portfolio weights G_t π_{t-1}.
    
    Parameters
    ----------
    date : pandas.Timestamp
        Current trading date (end-of-month t).
    prev_date : pandas.Timestamp
        Previous trading date (t-1).
    stayers, newcomers, leavers : list
        Universe partitions returned by ``get_universe_partitions``.
    zeros : list
        Stocks that are known to have π_t = 0 (e.g. leavers and stocks
        with missing data).
    df_pf_weights : pandas.DataFrame
        Historical portfolio weights with columns
        ``['id', 'eom', 'pi', 'g']`` at least.
    df_kl : pandas.DataFrame
        Kyle's lambda data with columns ``['id', 'eom', 'lambda']``.
    df_returns : pandas.DataFrame
        Realised returns with columns ``['id', 'eom', 'tr']``.
    
    Returns
    -------
    df_portfolio_t : pandas.DataFrame
        Portfolio state at date t with columns including
        ``['id', 'eom', 'pi', 'lambda', 'tr', 'pi_g_tm1']``.
        The column ``pi`` is initialised to a small positive value
        (1e-16) and will be overwritten by the optimiser for active
        stocks.
    """
    
    #---- Initialisation ----
    df_pf_t = (pd.DataFrame({ #df_portfolio_t
        'id': list(stayers + newcomers + leavers),
        'eom': date,
        'pi': np.array(1e-4)
    }).sort_values(by = 'id').reset_index(drop=True))

    # ---- Merge Kyle's lambda  ----
    #   (Note, need KL also for leavers and not just for active)
    df_pf_t = df_pf_t.merge(df_kl[df_kl['eom'] == prev_date][['id', 'lambda']],
                            on='id', how='left')
    
    #Set 'lambda' to zero for newcomers for which 'lambda' is NA. 
    #   Reason: When computing transaction costs, valid values for 'lambda' are
    #           required. If newcomers have missing 'lambda', they will not be
    #           be traded (i.e. they are not in active). Thus, their portfolio 
    #           weight is and will remain zero, so no transaction costs will be 
    #           incured anyway.
    df_pf_t.loc[(df_pf_t['id'].isin(set(newcomers).intersection(set(zeros)))) 
                &
                (df_pf_t['lambda'].isna()), 'lambda'] = 0.0
    
    if df_pf_t['lambda'].isna().sum() > 0:
        print("ERROR: A stock does not have a value for Kyle's Lambda")

    # ---- Realised Returns (to compute profit later) ---
    # Merge realised return
    df_pf_t = df_pf_t.merge(df_returns[df_returns['eom'] == date][['id', 'tr']],
                            on='id', how='left')
    
    #Set return for leavers to zero to avoid NaNs spreading, i.e. pi * tr = 0*NaN = NaN.
    df_pf_t.loc[df_pf_t['id'].isin(leavers), 'tr'] = 0

    # ---- Drifted Weights G @ pi_{t-1} ----
    # Compute
    pi_g = (df_pf_weights.query("eom == @prev_date")[['id', 'pi', 'g']]
            .assign(pi_g=lambda df: df['pi'] * df['g'])
            [['id', 'pi_g']])
    
    # Merge
    df_pf_t = df_pf_t.merge(pi_g, on='id', how='left').rename(columns={'pi_g': 'pi_g_tm1'})
    
    #Set value for newcomers to zero
    df_pf_t.loc[df_pf_t['id'].isin(newcomers), 'pi_g_tm1'] = 0
    
    # ---- Initialise 'pi_t' with 'pi_g_tm1' ----
    
    # Initialise pi_t with G @ pi_{t-1}
    df_pf_t.loc[df_pf_t['id'].isin(active), 'pi'] = df_pf_t.loc[df_pf_t['id'].isin(active), 'pi_g_tm1']
    
    # For newcomers that are actively traded --> g pi_{t-1} = 0.
    #   So, set pi_t to some epsilon (else-wise log(pi_t) undefined)
    df_pf_t.loc[(df_pf_t['pi'] == 0.0) & df_pf_t['id'].isin(active), 'pi'] = 1e-4
    
    # Set any 'pi' for zeros to 0.0
    df_pf_t.loc[df_pf_t['id'].isin(zeros), 'pi'] = 0.0

    return df_pf_t

def solve_pf_optimisation(prev_date, date, 
                          active, 
                          df_pf_t, df_kl_active, df_ret_pred_active, Sigma_active, 
                          df_me, df_spy, df_wealth, prediction_col, include_rf_asset,
                          flat_MaxPi, flat_MaxPi_limit,
                          w_upperLimit, w_lowerLimit, vol_scaler, tc_scaler, Ridge):
    
    """
    Solve the one-period myopic portfolio optimisation for date t.
    
    The optimiser chooses portfolio weights π_t over the active universe
    to maximise expected revenue minus transaction costs, subject to:
    
    - portfolio weights summing to one (via softmax parametrisation),
    - upper and lower bounds on each π_t,i defined as multiples of
      benchmark (value) weights from ``df_me``,
    - a maximum allowed portfolio variance scaled by ``vol_scaler``.
    
    Transaction costs are modelled as quadratic in turnover using
    Kyle's lambda
    
    Parameters
    ----------
    prev_date : pandas.Timestamp
        Previous trading date (t-1).
    date : pandas.Timestamp
        Current trading date (t).
    active : list
        List of stock ids in the active universe at date t.
    df_pf_t : pandas.DataFrame
        Portfolio dataframe at date t as returned by
        ``build_portfolio_dataframe``. Must contain columns
        ``['id', 'pi', 'pi_g_tm1']`` at least.
    df_kl_active : pandas.DataFrame
        Kyle's lambda for active stocks at ``prev_date`` with columns
        ``['id', 'lambda']``.
    df_return_predictions_active : pandas.DataFrame
        Expected returns for active stocks at ``prev_date`` with
        columns ``['id', prediction_col]``.
    Sigma : pandas.DataFrame
        Covariance matrix for active stocks at ``prev_date``.
    df_me : pandas.DataFrame
        Market equity data with columns ``['id', 'eom', 'me']`` used to
        derive weight bounds.
    df_spy : pandas.DataFrame
        Benchmark variance data with columns ``['eom', 'variance']``.
    df_wealth : pandas.DataFrame
        Wealth (AUM) data with columns ``['eom', 'wealth']``.
    prediction_col : str
        Name of the expected return column in
        ``df_return_predictions_active``.
    w_upperLimit, w_lowerLimit : float
        Multipliers applied to value weights to derive upper and lower
        bounds on π_t,i.
    vol_scaler : float
        Multiplier applied to benchmark variance constraint.
    
    Returns
    -------
    numpy.ndarray
        Array of optimised portfolio weights π_t for the active stocks,
        ordered consistently with the rows of ``df_pf_t`` restricted to
        ``active`` and with ``Sigma`` / ``df_return_predictions_active``.
    """

    # ---- Define Torch Objects ----
    # Return predictions
    r = torch.tensor(df_ret_pred_active[prediction_col], dtype=torch.float64)
    
    # Covariance Matrix 
    S = torch.tensor(Sigma_active.to_numpy(), dtype=torch.float64)
    
    # Kyle's Lambda (diagonal) Matrix
    L_diag = torch.tensor(df_kl_active['lambda'], dtype=torch.float64)

    # Wealth (AUM)
    w = torch.tensor(df_wealth[df_wealth['eom'] == date]['wealth'].iloc[0], dtype=torch.float64)
    
    # Drifted portfolio weights G @ pi_{t-1}
    pi_g_tm1 = torch.tensor(df_pf_t[df_pf_t['id'].isin(active)]['pi_g_tm1'].to_numpy(),
                            dtype=torch.float64)
    
    # Logits of pi_t
    pi_logits = torch.tensor(
        np.log(df_pf_t[df_pf_t['id'].isin(active)]['pi'].to_numpy()),
        requires_grad=True,
        dtype=torch.float64
    )
    
    # ---- Bound on pi_t ----
    
    if flat_MaxPi:
        max_pi = torch.tensor(flat_MaxPi_limit, dtype=torch.float64)
        min_pi = torch.tensor(0.0, dtype=torch.float64)
        
    else:
        weights_df = (df_me
                      .loc[(df_me['eom'] == prev_date) & (df_me['id'].isin(active))]
                      .sort_values(by = 'id')
                      .assign(w_max=lambda df: df['me'] / df['me'].sum() * w_upperLimit)
                      .assign(w_min=lambda df: df['me'] / df['me'].sum() * w_lowerLimit))
        
        if include_rf_asset:
            #Include upper bound of 100% and lower bound of 0% for rf (since rf not in df_me)
            weights_df = pd.concat([weights_df, pd.DataFrame({'id': [new_permno],   # ID risk-free asset
                                      'w_min': [0.0],
                                      'w_max': [1.0],
                                      'eom': [prev_date]})
                                    ], 
                                   ignore_index=True).sort_values(by = 'id')

    # ---- Bounds on portfolio variance ----
    max_var = df_spy[df_spy['eom'] == prev_date]['variance'].iloc[0]*vol_scaler
    
    # --- Scalers for Inequality Constraints ----
    penalty_maxPi   = 1.0
    penalty_minPi   = 1.0
    penalty_var     = 1.0

    # ---- Optimizer ----
    optimizer = torch.optim.Adam([pi_logits], lr=1e-2)

    # ---- Gradient Ascent ----
    for _ in range(500):
        # Clear Gradient
        optimizer.zero_grad()
        
        # pi_t (in levels)
        pi = F.softmax(pi_logits, dim=0)
        
        # Predicted revenue
        revenue = torch.dot(r, pi)
        
        # Transaction costs
        diff = pi - pi_g_tm1
        if Ridge:
            tc = diff.pow(2).sum() * torch.tensor(tc_scaler) #tc_scaler acts as ​\gamma.
        else:
            tc = 0.5 * w * (L_diag * diff.pow(2)).sum()
        
        # pi_t bounds violation
        max_pi_violation = (penalty_maxPi * F.relu(pi - max_pi)).sum()
        min_pi_violation = (penalty_minPi * F.relu(min_pi - pi)).sum()
        
        # Variance Violation  
        var_violation = penalty_var * F.relu(pi @ S @ pi - max_var)

        # Loss Function
        F_val = revenue - tc
        loss = -F_val + max_pi_violation + min_pi_violation + var_violation
        
        # One step of gradient ascent
        loss.backward()
        optimizer.step()

    # Print predicted profit
    print(f"  Predicted Profit: {F_val.item()}")
    
    return pi.detach().cpu().numpy()


def inclue_rf(df_pf_weights, df_kl, df_retPred, df_returns,
              risk_free, df_wealth, return_predictor_col,
              new_permno):
    
    #---- Portfolio Vector ----    

    # Expand Rows
    df = (pd.concat([df_pf_weights,
                               pd.DataFrame({'eom': df_pf_weights.eom.unique(),
                                             'id': new_permno})],
                              ignore_index = True)
                     .sort_values(by=['eom', 'id'])
                     .reset_index(drop=True)
                     )
    # Fill 'pi'
    df.loc[df['id'] == new_permno, 'pi'] = 1e-16
    
    # Compute DataFrame storing 'g'
    df_g = risk_free.merge(df_wealth[['eom','mu']], on = ['eom'], how = 'left')
    df_g = df_g[df_g['eom'].isin(df_pf_weights['eom'].unique())].sort_values(by = 'eom')
    df_g['g'] = (1+df_g['rf'])/(1+df_g['mu'])
    df_g.loc[df_g['eom'] == df['eom'].min(),'g'] = 1
    
    # Fill 'g'
    mask = (df['id'] == new_permno)
    df.loc[mask, 'g'] = df.loc[mask, 'eom'].map(df_g.set_index('eom')['g']) 
    
    #---- Kyle's Lambda ----
    df_kl = df_kl.copy() 
    
    # Compute minima
    kl_mins = (df_kl.groupby('eom')['lambda']
                  .min()
                  .reset_index()
                  )
    kl_mins['lambda'] = 0.0
    
    # Add rows
    df_kl = (pd.concat([df_kl, pd.DataFrame({'eom': df_kl.eom.unique(),
                                             'id': new_permno})
                        ], ignore_index = True)
                     .sort_values(by=['eom', 'id'])
                     .reset_index(drop=True)
                     )
    
    # Fill Lambda
    mask = (df_kl['id'] == new_permno)
    df_kl.loc[mask, 'lambda'] = df_kl.loc[mask, 'eom'].map(kl_mins.set_index('eom')['lambda']) 
    
    
    #---- Return Predictions ----
    
    df_retPred = df_retPred.copy()
    
    # Add rows
    df_retPred = (pd.concat([df_retPred, pd.DataFrame({'eom': df_retPred.eom.unique(),
                                             'id': new_permno})
                        ], ignore_index = True)
                     .sort_values(by=['eom', 'id'])
                     .reset_index(drop=True)
                     )
    
    # Fill return predictions
    mask = (df_retPred['id'] == new_permno)
    df_retPred.loc[mask, return_predictor_col] = df_retPred.loc[mask, 'eom'].map(risk_free.set_index('eom')['rf']) 
    
    #---- Realised Returns ----
    
    df_returns = df_returns.copy()
    
    # Add rows
    df_returns = (pd.concat([df_returns, pd.DataFrame({'eom': df_returns.eom.unique(),
                                             'id': new_permno})
                        ], ignore_index = True)
                     .sort_values(by=['eom', 'id'])
                     .reset_index(drop=True)
                     )
    
    # Fill return predictions
    mask = (df_returns['id'] == new_permno)
    df_returns.loc[mask, 'tr'] = df_returns.loc[mask, 'eom'].map(risk_free.set_index('eom')['rf']) 
    
    return df, df_kl, df_retPred, df_returns

#%% Function: Return Predictions
def scale_return_predictions(df_predictions, sp500_constituents, 
                  trading_start:str, trading_end:str,
                  predictor_col,
                  target_col):
    """
    Linearly combine model predictions to match realised returns.

    This function:

    1. Merges realised (excess) returns into the prediction dataframe.
    2. Restricts the universe to S&P 500 constituents at each month.
    3. Rescales rank predictions cross-sectionally into
       the interval ``[-1, 1]`` via percentile ranks.
    4. Runs a rolling cross-sectional OLS to map the (scaled) predictions
       into actual target excess returns, and uses the estimated
       coefficients to generate out-of-sample scaled predictions.

    The regression is re-estimated annually using a one-year rolling window,
    and the resulting fitted values are stored in the column
    ``'ret_pred_scaled'``.

    Args:
        df_predictions: DataFrame with at least columns ``['id', 'eom']`` and
            one or more prediction columns.
        sp500_constituents: DataFrame with S&P 500 membership information.
            Must contain ``['eom', 'id']`` indicating which stocks are in the
            index at each month.
        trading_start: Start date for the
            trading / backtest period. Used to define the first regression
            re-estimation date.
        trading_end: End date (string parsable by pandas) for the
            trading / backtest period.
        rank_cols: List of column names in ``df_predictions`` whose values
            must be rescaled to ``[-1, 1]`` as they are rank predictions.
        target_col: Name of the realised return column in the JKP database
            to be used as the regression target, e.g. excess return over
            the SP500 next month (default ``'tr_m_sp500_ld1'``).

    Returns:
        pd.DataFrame: DataFrame with the same identifier structure as
        ``df_predictions``, plus:

            - the realised target column ``target_col``
            - the column ``'ret_pred_scaled'`` containing the OOS
              predicted excess returns from the rolling OLS.

        Only rows with non-missing scaled predictions and targets are kept.
    """
    
    #------------------
    #   Data Assembly
    #------------------
    # Get target data
    df_target = pd.read_sql_query(f"SELECT id, eom, {target_col} FROM Factors_processed WHERE eom >='{df_predictions.eom.min().strftime('%Y-%m-%d')}'",
                               con = JKP_Factors,
                               parse_dates = {'eom'})
    
    # Merge actual returns to predictions
    df = df_predictions.merge(df_target[['id','eom', target_col]],
                              on = ['id','eom'], how = 'left')
    
    # Save Workspace
    del df_target
    
    # Only keep S&P500 stocks (active universe)
    df = (df
          .merge(sp500_constituents.assign(in_sp500 = True), 
                 on = ['eom', 'id'], how = 'left')
          .dropna(subset = ('in_sp500'))
          .drop(columns = 'in_sp500')
          .get(['id','eom', predictor_col, target_col])
          )
    
    #------------------
    #   Regression
    #------------------
    #Container to store regression results
    reg_results = []
    
    #Dates at which the coefficients are re-trained
    reg_dates = pd.date_range(start=trading_start,
                                  end=trading_end,
                                  freq='YE'
                                  )-pd.offsets.MonthEnd(11)
    
    #Loop over re-training dates
    for date in reg_dates:
        
        # Dates on which the the coefficients are trained on
        train_start = max(date - pd.offsets.MonthEnd(2+12),df['eom'].min())
        train_end = date-pd.offsets.MonthEnd(2)
        
        # Dates of OOS fitted values
        pred_start = date-pd.offsets.MonthEnd(1)
        pred_end = date + pd.offsets.MonthEnd(10)
        
        # Get Train and Prediction Data
        train_data = df.loc[df['eom'].between(train_start, train_end, inclusive='both')]
        pred_data  = df.loc[df['eom'].between(pred_start, pred_end, inclusive='both')]
        
        # Prepare Training Data (Add Constant)
        X_train = sm.add_constant(train_data[train_data.columns.drop(['id','eom', target_col])])
        y_train = train_data[target_col]
        
        # Fit the Model (OLS)
        model = sm.OLS(y_train, X_train).fit()
        
        # Make predictions (add constant)
        X_pred = sm.add_constant(pred_data[pred_data.columns.drop(['id','eom', target_col])], has_constant='add')
        pred_data[predictor_col + "_rescaled"] = model.predict(X_pred)
        
        # Store results
        reg_results.append(pred_data)
    
    # Concatenate dataframes
    reg_df = pd.concat(reg_results)
    
    #Drop missing predictions
    reg_df = reg_df.dropna().drop_duplicates(subset = ['id','eom']).drop(columns = target_col)
    
    #Drop unscaled variable
    reg_df = reg_df.drop(columns = predictor_col)
    
    return reg_df
#%% Function: Saving Output
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

def settings_to_id(settings: dict) -> str:
    """
    File-system friendly ID based on settings (used in filenames).
    """
    s = json.dumps(settings, sort_keys=True)
    # Replace anything non-alphanumeric with underscores
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_")
    return s

#%% Read in Data

#DataBases
JKP_Factors = sqlite3.connect(database = path + "Data/JKP_processed.db")
SP500_Constituents = sqlite3.connect(database = path + "Data/SP500_Constituents.db")
Benchmarks = sqlite3.connect(database = path + "Data/Benchmarks.db")
Models = sqlite3.connect(database = path + "Data/Predictions.db")

#============================
#       Trading Dates
#============================
#Trading dates
trading_start, trading_end = pd.to_datetime("2004-01-31"), settings['rolling_window']['trading_end']
trading_dates = pd.date_range(start=trading_start,
                              end=trading_end,
                              freq='ME'
                              )

#Start and End Date as Strings
start_date = str(trading_start - pd.offsets.MonthEnd(1))[:10]
end_date = str(trading_end)[:10]

#============================
#       Risk-free rate
#============================

# Read risk-free rate data (select only 'yyyymm' and 'RF' columns).
risk_free = (pd.read_csv(path + "Data/FF_RF_monthly.csv", usecols=["yyyymm", "RF"])
             .assign(rf = lambda df: df["RF"]/100)
             .assign(eom = lambda df: pd.to_datetime(df["yyyymm"].astype(str) + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0))
             .get(['eom','rf'])
             )

#===============================
#       SPY ETF Performance
#===============================
df_spy = pd.read_sql_query("SELECT * FROM SPY",
                           parse_dates = {'eom'},
                           con = Benchmarks)

#============================
#       S&P 500 Universe
#============================
#Stocks that were, are and will be in the S&P 500.
sp500_ids = list(pd.read_sql_query("SELECT * FROM SP500_Constituents_alltime",
                              con = SP500_Constituents)['id']
                 )
sp500_ids = ', '.join(str(x) for x in sp500_ids)

#Stocks that at date t are in the S&P 500
sp500_constituents = (pd.read_sql_query("SELECT * FROM SP500_Constituents_monthly", #" WHERE eom >= '{start_date}'",
                                       con = SP500_Constituents,
                                       parse_dates = {'eom'})
                      .rename(columns = {'PERMNO': 'id'})
                      )


#============================
#       Data
#============================

df_pf_weights, df_kl, df_me, df_returns,\
    df_wealth, dict_barra = GF.load_portfolio_backtest_data(JKP_Factors, start_date, 
                                          sp500_ids, path)
    

    
#==================================================
#      Model Predictions (Requires Manual Updating)
#==================================================

estimators = [
          # Baseline
          #'XGBRegHPlenient_LevelTrMsp500Target_SP500UniverseFL_RankFeatures_RollingWindow_win120_val12_test12',
          # Robustness Estimation Universe (CRSP vs. SP500)
          #'IPCA_LevelTrMsp500Target_SP500UniverseFL_ZscoreFeatures_RollingWindow_win120_val12_test12',
          # Robustness variable
          #'TransformerSet_Dropout010_LevelTrMSp500Target_SP500UniverseFL_RankFeatures_RollingWindow_win120_val12_test12',
          # Robustness variable & Est. Univ
          'RFF_LevelTrMsp500Target_SP500UniverseFL_ZscoreFeatures_RollingWindow_win120_val12_test12',
          #'IPCA_LevelTrMsp500Target_CRSPUniverse_ZscoreFeatures_RollingWindow_win120_val12_test12'
          #'MarketOracle'
          ]


#Load return predictions
#At 'eom', predictions are for eom+1
df_retPred = GF.load_MLpredictions(Models, estimators) 
"""
CAUTION: I need a safeguard that I df_retPred only contains the S&P 500 stocks when feeding
them into optimise_portfolio().
"""

#Get names of return predictors
prediction_cols = list(df_retPred.columns.drop(['id','eom']).astype('string'))

#--- Rescale Zscores ----
for predictor_col in [s for s in prediction_cols if "Zscore" in s]:
    
    reg_df = scale_return_predictions(df_retPred, sp500_constituents, 
                      trading_start, trading_end,
                      predictor_col,
                      target_col = 'tr_ld1')  
    
    df_retPred = df_retPred.merge(reg_df[['id','eom',predictor_col+"_rescaled"]],
                                  on = ['id','eom'], how = 'left').drop(columns = predictor_col)

df_retPred = df_retPred.dropna().reset_index(drop = True)

#Update prediction_cols name
prediction_cols = list(df_retPred.columns.drop(['id','eom']).astype('string'))


#=======================================================
#   Extend universe with risk-free asset (if desired)
#=======================================================
if include_rf_asset:
    df_pf_weights, df_kl, df_retPred, df_returns = inclue_rf(df_pf_weights, df_kl, df_retPred, df_returns,
                  risk_free, df_wealth, prediction_col,
                  new_permno)

#%% Compute Optimal Portfolio

# If True: Computes the transaction cost term as a Ridge regularisation on portfolio changes
Ridge = False # Use e.g. for tc_scale in [25.0] for the regularisation

for tc_scale in [1.0, 0.5, 0.1, 0.01]:
    
    #---- Common Settings for This Run (same for all models) ----
    run_settings = dict(includeRF    = include_rf_asset,
                        flatMaxPi    = True,
                        flatMaxPiVal = 0.15,
                        Wmax         = None,
                        Wmin         = None,
                        volScaler    = 1.0, 
                        tcScaler     = tc_scale,
                        )
    tc_to_run = [1.0, 0.5, 0.1, 0.01] if Ridge else [run_settings['tcScaler']]

    
    print("Run ID:", settings_to_id(run_settings))
    print("Settings:", settings_string(run_settings))
    
    #---- Save empty portfolio weights ----
    df_pf_weights_base = df_pf_weights.copy()
    
    #---- Containers to collect results across models ----
    per_model_returns   = {}   # model_name -> df_ret_strat
    per_model_strategy  = {}   # model_name -> df_strategy
    per_model_metrics   = []   # list of dicts with metrics
    
    #---- Benchmark cumulative returns ----
    df_spy_run = (df_spy[df_spy['eom'].between(trading_start, trading_end)]
                  .assign(cumulative_return = lambda df: (1.0 + df['ret']).cumprod()
                          )
                  )
        
    for model_name, prediction_col in zip(estimators, prediction_cols):
        print("\n==============================")
        print("Running model:", model_name)
        print("    Prediction column:", prediction_col)
        print("==============================")
        
        # Use a fresh copy of the initial weights for each model
        df_pf_weights = df_pf_weights_base.copy()
        
        #---- Compute Optimal Portfolio ----
        df_strategy, df_pf_weights \
            = optimise_portfolio(df_pf_weights, df_kl, df_me, dict_barra, df_returns, df_wealth, df_spy, 
                               df_retPred, 
                               trading_dates, 
                               prediction_col, 
                               run_settings['includeRF'],
                               run_settings['flatMaxPi'], 
                               run_settings['flatMaxPiVal'], #portfolio weight bound  [0,flat_MaxPi_limit] for every stock
                               run_settings['Wmax'], 
                               run_settings['Wmin'], #Benchmark dependent portfolio bound for every stock
                               run_settings['volScaler'],
                               run_settings['tcScaler'],
                               Ridge)
            
        # Merge predicted returns
        df_strategy = df_strategy.merge(df_retPred.get(['id','eom',prediction_col])
                                        .assign(eom = lambda df: df['eom'] + pd.offsets.MonthEnd(1)),
                                        on = ['id', 'eom'], how = 'left')  

        for tc_scaler in tc_to_run:
            if Ridge: # !! Must compute transaction costs ex-post !!
            
                # Merge Wealth to recompute Transaction Costs
                df_strategy = df_strategy.merge(df_wealth[['eom','wealth']], on = ['eom'], how = 'left')
            
                # Merge Baseline Lambda
                df_strategy = df_strategy.drop(columns = 'lambda')
                df_strategy = df_strategy.merge(df_kl, on = ['id','eom'])
                # Compute Transaction Costs
                df_strategy = (df_strategy
                               .assign(tc = lambda df: 
                                       tc_scaler * df['wealth'] * 0.5 * df['lambda']
                                                 * (df['pi']-df['pi_g_tm1'])**2)
                                   .drop(columns = 'wealth')
                                   )
            
            # ---- Compute strategy returns (net & gross) per month ----
            df_ret_strat = (
                df_strategy
                .assign(ret_net_row = lambda df: df['rev'] - df['tc'])
                .groupby('eom', as_index=False)
                .agg(
                    ret_net   = ('ret_net_row', 'sum'),
                    ret_gross = ('rev', 'sum')
                )
                .assign(
                    cumret_net   = lambda df: (1.0 + df['ret_net']).cumprod(),
                    cumret_gross = lambda df: (1.0 + df['ret_gross']).cumprod()
                )
            )
            
            # Compute cumulative monthly profit for benchmark
            df_ret_bench = (df_spy[df_spy['eom'].isin(df_ret_strat['eom'].unique())]
                                   .assign(cumulative_return = lambda df: (1+df['ret']).cumprod())
                                   .reset_index(drop = True)
                                   )
            
            
            # ---- Save per-model pickle (with settings attached) ----
            result_dict = {
            'run_settings' : run_settings,
            'model_name'   : model_name,
            'prediction_col': prediction_col,
            'Strategy'     : df_strategy,
            'Profit'       : df_ret_strat,
            }
        
            safe_model_name = re.sub(r"[^0-9a-zA-Z]+", "_", model_name).strip("_")
            
            if Ridge: # Reformat tcScaler to reflect ex-post transaction costs in the file name and not the Ridge Regularisation parameter \gamma
                run_settings['tcScaler'] = tc_scaler
            
            filename = f"{settings_string(run_settings)}_{safe_model_name}.pkl"
            
            if Ridge:
                filename = "Ridge_" + filename
            
            with open(os.path.join(path, "Portfolios", filename), "wb") as f:
                pickle.dump(result_dict, f)
            
            print(f"Saved model results to: {filename}")