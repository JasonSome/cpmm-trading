import numpy as np
from EWMA_Vol import EWMA_Vol

def CPMM_Price(X, Y, D):
    """
    Computes the CPMM price based on current dollar reserves (X),
    asset reserves (Y), and trade volume (D).
    """
    return X / (Y - D)

def CPMM_Marginal_Price(X, Y):
    """
    Returns the marginal price of the CPMM given the current reserves.
    This is simply the ratio of X to Y.
    """
    return X / Y

def CPMM_Update(X, Y, D):
    """
    Updates the pool reserves after a trade of volume D.
    The price at which the trade is executed is calculated using CPMM_Price.
    The new dollar reserve increases by P * D while the asset reserve decreases by D.
    """
    P = CPMM_Price(X, Y, D)
    return X + P * D, Y - D

def generate_trade_filters(N, M, seed=42):
    """
    Generate trade filters for systematic trading orders.

    This function creates two boolean filter arrays of shape (N+1, M):
    - filtr_bfs: True when a buyer trades first.
    - filtr_sfs: True when a seller trades first.

    Parameters:
        N : Number of time steps (the filters are generated for N+1 periods).
        M : Number of market instances.
        seed : Random seed for reproducibility (default is 42).

    Returns:
        filtr_bfs : Boolean array of shape (N+1, M) for buyer‑trade‑first filters.
        filtr_sfs : Boolean array of shape (N+1, M) for seller‑trade‑first filters.
    """
    # Set the random seed.
    np.random.seed(seed)
    
    # Generate a uniform random array with values in [0, 1) for each time step and market instance.
    uni = np.random.uniform(0, 1, size=(N+1, M))
    
    # Create the buyer-trade-first filter:
    filtr_bfs = uni < 0.5
    
    # Create the seller-trade-first filter:
    filtr_sfs = uni >= 0.5
    
    # Return the two filter arrays.
    return filtr_bfs, filtr_sfs

def CEX_Price(S, mu, sigma, dt, N, M=1, seed=42):
    """
    Simulates the CEX price using a geometric Brownian motion model.
    
    Parameters:
        S   : Initial price.
        mu  : Drift coefficient.
        sigma: Volatility.
        dt  : Time step.
        N : Number of time steps.
        M   : Number of market instances (default is 1).
        seed : Random seed for reproducibility (default is 42).
    
    Returns:
        CEX Price series for each market instance.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Pre-allocate array for CEX price
    S0 = np.zeros((N+1, M)) 

    # Initial CEX price
    S0[0] = S

    # Simulate Brownian Motion increments
    dWs = np.sqrt(dt) * np.random.normal(0., 1., size=(N, M))

    # Update prices according to the geometric Brownian motion model
    S0[1:] = S * np.exp( np.cumsum((mu - 0.5*sigma**2)*dt + sigma * dWs, axis=0) )
    
    # Return price series for each market instance
    return S0
    
def CEX_Price_Antithetic(S, mu, sigma, dt, N, M=1, seed=42):
    """
    Simulates the CEX price using a geometric Brownian motion model. Uses antithetic variates for variance reduction.
    
    Parameters:
        S   : Initial price.
        mu  : Drift coefficient.
        sigma: Volatility.
        dt  : Time step.
        N : Number of time steps.
        M   : Number of market instances (default is 1).
        seed : Random seed for reproducibility (default is 42).
    
    Returns:
        Two CEX Price series for each market instance. The second output corresponds to the antithetic variates.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Pre-allocate array for CEX price
    S0 = np.zeros((N+1, M)) 
    S0_ = np.zeros((N+1, M)) #antithetic variates

    # Initial CEX price
    S0[0] = S
    S0_[0] = S

    # Simulate Brownian Motion increments
    dWs = np.sqrt(dt) * np.random.normal(0., 1., size=(N, M))

    # Update prices according to the geometric Brownian motion model
    S0[1:] = S * np.exp( np.cumsum((mu - 0.5*sigma**2)*dt + sigma * dWs, axis=0) )
    S0_[1:] = S * np.exp( np.cumsum((mu - 0.5*sigma**2)*dt - sigma * dWs, axis=0) )
    
    # Return price series for each market instance
    return S0, S0_

def Arb_Trade_CPMM(S, X, Y, eta0, eta1):
    """
    Calculates the arbitrage trade volume for the CPMM.
    
    Parameters:
        S   : Current CEX price (array with shape (M,)).
        X   : Current CPMM dollar reserves.
        Y   : Current CPMM asset reserves.
        eta0: CEX proportional cost.
        eta1: CPMM proportional cost.
    
    Returns:
        Arbitrage trade volume for each market instance.
    """
    # Arbitrage volume computation
    P = CPMM_Marginal_Price(X, Y)
    q1 = np.minimum(Y * (1 - np.sqrt((P * (1 - eta1)) / (S * (1 + eta0)))), 0.)
    q2 = np.maximum(Y * (1 - np.sqrt((P * (1 + eta1)) / (S * (1 - eta0)))), 0.)
    return q1 + q2

def Buy_Trade_CPMM(buy, S, X, Y, eta0, eta1):
    """
    Computes the trade volume for a systematic buy order on the CPMM.
    
    Parameters:
        buy : The constant trade size for buyers.
        S   : Current CEX price for the trade.
        X   : Current CPMM dollar reserves.
        Y   : Current CPMM asset reserves.
        eta0: CEX proportional cost.
        eta1: CPMM proportional cost.
    
    Returns:
        The buy trade volume, ensuring it is non-negative and within a specified limit.
    """
    P = CPMM_Marginal_Price(X, Y)
    # v: Maximum allowable trade volume based on reserve levels and costs
    v = Y * (1 - np.sqrt((P * (1 + eta1)) / (S * (1 + eta0))))
    # Limit the trade volume between 0 and v
    return np.maximum(np.minimum(buy, v), 0.)

def Sell_Trade_CPMM(sell, S, X, Y, eta0, eta1):
    """
    Computes the trade volume for a systematic sell order on the CPMM.
    
    Parameters:
        sell: The constant trade size for sellers.
        S   : Current CEX price for the trade.
        X   : Current CPMM dollar reserves.
        Y   : Current CPMM asset reserves.
        eta0: CEX proportional cost.
        eta1: CPMM proportional cost.
    
    Returns:
        The sell trade volume, ensuring it is non-positive and within a specified limit.
    """
    P = CPMM_Marginal_Price(X, Y)
    # v: Maximum allowable trade volume based on reserve levels and costs
    v = Y * (1 - np.sqrt((P * (1 - eta1)) / (S * (1 - eta0))))
    # Limit the trade volume between v and 0 (sell orders are negative)
    return np.minimum(np.maximum(sell, v), 0.)

def process_trade(trade_func, trade_amt, S, X, Y, filt, eta0, eta1):
    """
    Processes a trade (buy or sell) on a subset of indices defined by a boolean filter.
    
    Parameters:
        trade_func: The trading function to use (either Buy_Trade_CPMM or Sell_Trade_CPMM).
        trade_amt : The constant trade size.
        S         : The CEX price array at the current time step.
        X         : The CPMM dollar reserves array.
        Y         : The CPMM asset reserves array.
        filt      : Boolean filter array to select indices for the trade.
        eta0      : CEX proportional cost.
        eta1      : CPMM proportional cost.
    
    Returns:
        X_new     : Updated dollar reserves for the selected indices.
        Y_new     : Updated asset reserves for the selected indices.
        new_price : New marginal price after the trade.
        D         : Trade volume executed.
        revenue   : Trade revenue calculated as D * price * eta1 
                    (positive for buy trades, negative for sell trades).
    """
    # Select sub-arrays based on the provided filter
    S_sub = S[filt]
    X_sub = X[filt]
    Y_sub = Y[filt]
    
    # Compute trade volume using the designated trade function
    D = trade_func(trade_amt, S_sub, X_sub, Y_sub, eta0, eta1)
    # Determine the price at which the trade is executed
    P = CPMM_Price(X_sub, Y_sub, D)
    
    # Update the reserves based on the trade volume
    X_new, Y_new = CPMM_Update(X_sub, Y_sub, D)
    # Calculate the new marginal price after the trade
    new_price = CPMM_Marginal_Price(X_new, Y_new)
    
    # Calculate trade revenue: positive for buys, negative for sells
    if trade_func.__name__ == "Buy_Trade_CPMM":
        revenue = D * P * eta1
    else:
        revenue = -D * P * eta1
        
    return X_new, Y_new, new_price, D, revenue

def simulation(M_, N_, T_, dt_, buy_, sell_, eta0_, eta1_, S0_, X_, Y_, filtr_bfs, filtr_sfs):
    """
    Runs a market simulation for a CPMM with systematic and arbitrage trading.
    
    Parameters:
        M_       : Number of market indices/instances.
        N_       : Number of time periods.
        T_       : Time horizon for the simulation.
        dt_      : Time step increment.
        buy_     : Constant trade size for systematic buyers.
        sell_    : Constant trade size for systematic sellers.
        eta0_    : CEX proportional cost.
        eta1_    : CPMM proportional cost.
        S0_      : An (N+1) x M array representing the CEX price time series.
        X_       : Initial CPMM dollar reserve (scalar or array).
        Y_       : Initial CPMM asset reserve (scalar or array).
        filtr_bfs: Array of boolean filters for systematic trades (buy-first order).
        filtr_sfs: Array of boolean filters for systematic trades (sell-first order).
    
    Returns:
        A tuple of numpy arrays containing the simulation results:
            - Pool_X          : CPMM dollar reserves over time (shape: (N+1, M)).
            - Pool_Y          : CPMM asset reserves over time (shape: (N+1, M)).
            - S0_             : CEX price time series (possibly updated; shape: (N+1, M)).
            - S1              : CPMM marginal price time series (shape: (N+1, M)).
            - CPMM_buy_revenue: Revenue from buy trades over time (shape: (N+1, M)).
            - CPMM_sell_revenue: Revenue from sell trades over time (shape: (N+1, M)).
            - CPMM_arb_revenue: Revenue from arbitrage trades over time (shape: (N+1, M)).
            - Hedging_port_val: Hedging portfolio value over time (shape: (N+1, M)).
    """
    
    M = M_  # Number of market indices
    N = N_  # Number of time periods
    
    # Pre-allocate arrays to store time-series data for each metric.
    Pool_X = np.zeros((N + 1, M))         # CPMM dollar reserves over time.
    Pool_Y = np.zeros((N + 1, M))         # CPMM asset reserves over time.
    S1 = np.zeros((N + 1, M))             # CPMM marginal price over time.
    Arb = np.zeros((N + 1, M))            # Arbitrage trade volumes over time.
    CPMM_buy = np.zeros((N + 1, M))       # Systematic buy trade volumes over time.
    CPMM_sell = np.zeros((N + 1, M))      # Systematic sell trade volumes over time.
    CPMM_buy_revenue = np.zeros((N + 1, M))   # Revenue from buy trades over time.
    CPMM_sell_revenue = np.zeros((N + 1, M))  # Revenue from sell trades over time.
    CPMM_arb_revenue = np.zeros((N + 1, M))   # Revenue from arbitrage trades over time.
    Hedging_port_val = np.zeros((N + 1, M)) # Time series for tracking hedging portfolio value.
    
    # Initialize simulation parameters.
    T = T_       # Total simulation time horizon.
    dt = dt_     # Time increment per period.
    buy = buy_   # Trade size for systematic buyers.
    sell = sell_ # Trade size for systematic sellers.
    eta0 = eta0_ # CEX proportional cost.
    eta1 = eta1_ # CPMM proportional cost.
    
    # Initialize pool reserves using starting values (broadcast across M instances).
    X = X_ * np.ones(M)
    Y = Y_ * np.ones(M)
    
    # Initialize the marginal price at time 0, store initial reserves,
    # and set the initial hedging portfolio value based on the initial CEX price.
    S1[0] = CPMM_Marginal_Price(X, Y)
    Pool_X[0] = X
    Pool_Y[0] = Y
    Hedging_port_val[0] = Pool_X[0] + Pool_Y[0] * S0_[0]

    # Main simulation loop over each time period.
    for i in range(1, N + 1):
        # Obtain current CEX price for all market indices at time i.
        S = S0_[i]  # S is an array of shape (M,)
        
        # Update the hedging portfolio value based on price changes.
        Hedging_port_val[i] = Hedging_port_val[i-1] + Pool_Y[i-1] * (S - S0_[i-1])
        
        # --- Arbitrage Trade (applied to all indices) ---
        # Calculate arbitrage trade volume based on current price and reserves
        D_A = Arb_Trade_CPMM(S, X, Y, eta0, eta1)
        # Determine price at which arbitrage trade is executed
        P_A = CPMM_Price(X, Y, D_A)
        # Update reserves after the arbitrage trade
        X, Y = CPMM_Update(X, Y, D_A)
        # Update the marginal price post-arbitrage trade
        S1[i] = CPMM_Marginal_Price(X, Y)
        # Record updated reserves and arbitrage trade volume
        Pool_X[i] = X
        Pool_Y[i] = Y
        Arb[i] = D_A
        # Record arbitrage revenue (absolute value is used)
        CPMM_arb_revenue[i] = np.abs(D_A * P_A * eta1)
        
        # Retrieve boolean filters for systematic trades at this time step
        filtr_bf = filtr_bfs[i]
        filtr_sf = filtr_sfs[i]
        
        # --- Process systematic trades on the "bf" (buy-first) filter ---
        # First, process BUY trades for indices where filtr_bf is True
        X_bf, Y_bf, new_price_bf, D_buy_bf, rev_buy_bf = process_trade(
            Buy_Trade_CPMM, buy, S, X, Y, filtr_bf, eta0, eta1
        )
        # Update marginal price and reserves for the filtered indices after buy trade
        S1[i, filtr_bf] = new_price_bf
        X[filtr_bf] = X_bf
        Y[filtr_bf] = Y_bf
        Pool_X[i, filtr_bf] = X_bf
        Pool_Y[i, filtr_bf] = Y_bf
        # Record buy trade volume and revenue
        CPMM_buy[i, filtr_bf] = D_buy_bf
        CPMM_buy_revenue[i, filtr_bf] = rev_buy_bf

        # Then, process SELL trades for the same "bf" indices
        X_bf, Y_bf, new_price_bf, D_sell_bf, rev_sell_bf = process_trade(
            Sell_Trade_CPMM, sell, S, X, Y, filtr_bf, eta0, eta1
        )
        # Update marginal price and reserves for the filtered indices after sell trade
        S1[i, filtr_bf] = new_price_bf
        X[filtr_bf] = X_bf
        Y[filtr_bf] = Y_bf
        Pool_X[i, filtr_bf] = X_bf
        Pool_Y[i, filtr_bf] = Y_bf
        # Record sell trade volume and revenue
        CPMM_sell[i, filtr_bf] = D_sell_bf
        CPMM_sell_revenue[i, filtr_bf] = rev_sell_bf

        # --- Process systematic trades on the "sf" (sell-first) filter ---
        # First, process SELL trades for indices where filtr_sf is True
        X_sf, Y_sf, new_price_sf, D_sell_sf, rev_sell_sf = process_trade(
            Sell_Trade_CPMM, sell, S, X, Y, filtr_sf, eta0, eta1
        )
        # Update marginal price and reserves for the filtered indices after sell trade
        S1[i, filtr_sf] = new_price_sf
        X[filtr_sf] = X_sf
        Y[filtr_sf] = Y_sf
        Pool_X[i, filtr_sf] = X_sf
        Pool_Y[i, filtr_sf] = Y_sf
        # Record sell trade volume and revenue
        CPMM_sell[i, filtr_sf] = D_sell_sf
        CPMM_sell_revenue[i, filtr_sf] = rev_sell_sf

        # Then, process BUY trades for the same "sf" indices
        X_sf, Y_sf, new_price_sf, D_buy_sf, rev_buy_sf = process_trade(
            Buy_Trade_CPMM, buy, S, X, Y, filtr_sf, eta0, eta1
        )
        # Update marginal price and reserves for the filtered indices after buy trade
        S1[i, filtr_sf] = new_price_sf
        X[filtr_sf] = X_sf
        Y[filtr_sf] = Y_sf
        Pool_X[i, filtr_sf] = X_sf
        Pool_Y[i, filtr_sf] = Y_sf
        # Record buy trade volume and revenue
        CPMM_buy[i, filtr_sf] = D_buy_sf
        CPMM_buy_revenue[i, filtr_sf] = rev_buy_sf

    # Return the complete simulation results as a tuple of arrays.
    return Pool_X, Pool_Y, S0_, S1, CPMM_buy_revenue, CPMM_sell_revenue, CPMM_arb_revenue, Hedging_port_val


def adaptive_fee_simulation(M_, N_, T_, dt_, buy_, sell_, eta0_, eta1_func, S0_, X_, Y_, filtr_bfs, filtr_sfs, sigma_0, lambda_=0.97):
    """
    Runs a market simulation for a CPMM with systematic and arbitrage trading.
    
    Parameters:
        M_          : Number of market indices/instances.
        N_          : Number of time periods.
        T_          : Time horizon for the simulation.
        dt_          : Time step increment.
        buy_         : Constant trade size for systematic buyers.
        sell_        : Constant trade size for systematic sellers.
        eta0_        : CEX proportional cost.
        eta1_func    : CPMM proportional cost, function of the estimated volatility.
        S0_         : An (N+1) x M array representing the CEX price time series.
        X_           : Initial CPMM dollar reserve (scalar or array).
        Y_           : Initial CPMM asset reserve (scalar or array).
        filtr_bfs   : Array of boolean filters for systematic trades (buy-first order).
        filtr_sfs   : Array of boolean filters for systematic trades (sell-first order).
        sigma_0     : initial volatility.
    
    Returns:
        A tuple of numpy arrays containing the simulation results:
            - Pool_X          : CPMM dollar reserves over time (shape: (N+1, M)).
            - Pool_Y          : CPMM asset reserves over time (shape: (N+1, M)).
            - S0_             : CEX price time series (possibly updated; shape: (N+1, M)).
            - S1              : CPMM marginal price time series (shape: (N+1, M)).
            - CPMM_buy_revenue: Revenue from buy trades over time (shape: (N+1, M)).
            - CPMM_sell_revenue: Revenue from sell trades over time (shape: (N+1, M)).
            - CPMM_arb_revenue: Revenue from arbitrage trades over time (shape: (N+1, M)).
            - Hedging_port_val: Hedging portfolio value over time (shape: (N+1, M)).
    """
    
    M = M_  # Number of market indices
    N = N_  # Number of time periods
    
    # Pre-allocate arrays to store time-series data for each metric.
    Pool_X = np.zeros((N + 1, M))         # CPMM dollar reserves over time.
    Pool_Y = np.zeros((N + 1, M))         # CPMM asset reserves over time.
    S1 = np.zeros((N + 1, M))             # CPMM marginal price over time.
    Arb = np.zeros((N + 1, M))            # Arbitrage trade volumes over time.
    CPMM_buy = np.zeros((N + 1, M))       # Systematic buy trade volumes over time.
    CPMM_sell = np.zeros((N + 1, M))      # Systematic sell trade volumes over time.
    CPMM_buy_revenue = np.zeros((N + 1, M))   # Revenue from buy trades over time.
    CPMM_sell_revenue = np.zeros((N + 1, M))  # Revenue from sell trades over time.
    CPMM_arb_revenue = np.zeros((N + 1, M))   # Revenue from arbitrage trades over time.
    Hedging_port_val = np.zeros((N + 1, M)) # Time series for tracking hedging portfolio value.
    
    # Initialize simulation parameters.
    T = T_       # Total simulation time horizon.
    dt = dt_     # Time increment per period.
    buy = buy_   # Trade size for systematic buyers.
    sell = sell_ # Trade size for systematic sellers.
    eta0 = eta0_ # CEX proportional cost.
    eta1 = eta1_func(sigma_0) # CPMM proportional cost.
    eta1s = [eta1]

    # Initialize volatility estimator

    ewma_vol = EWMA_Vol(sigma_0, lambda_=lambda_)
    sigma_hat = sigma_0
    sigma_hats = [sigma_0]
    
    # Initialize pool reserves using starting values (broadcast across M instances).
    X = X_ * np.ones(M)
    Y = Y_ * np.ones(M)
    
    # Initialize the marginal price at time 0, store initial reserves,
    # and set the initial hedging portfolio value based on the initial CEX price.
    S1[0] = CPMM_Marginal_Price(X, Y)
    Pool_X[0] = X
    Pool_Y[0] = Y
    Hedging_port_val[0] = Pool_X[0] + Pool_Y[0] * S0_[0]

    # Main simulation loop over each time period.
    for i in range(1, N + 1):
        # Obtain current CEX price for all market indices at time i.
        S = S0_[i]  # S is an array of shape (M,)

        # Update estimate of instantaneous volatility and fee
        rt = S0_[i,0]/S0_[i-1,0] - 1.
        sigma_hat = ewma_vol.update(rt/np.sqrt(dt))
        sigma_hats.append(sigma_hat)
        eta1 = eta1_func(sigma_hat)
        eta1s.append(eta1)

        
        # Update the hedging portfolio value based on price changes.
        Hedging_port_val[i] = Hedging_port_val[i-1] + Pool_Y[i-1] * (S - S0_[i-1])
        
        # --- Arbitrage Trade (applied to all indices) ---
        # Calculate arbitrage trade volume based on current price and reserves
        D_A = Arb_Trade_CPMM(S, X, Y, eta0, eta1)
        # Determine price at which arbitrage trade is executed
        P_A = CPMM_Price(X, Y, D_A)
        # Update reserves after the arbitrage trade
        X, Y = CPMM_Update(X, Y, D_A)
        # Update the marginal price post-arbitrage trade
        S1[i] = CPMM_Marginal_Price(X, Y)
        # Record updated reserves and arbitrage trade volume
        Pool_X[i] = X
        Pool_Y[i] = Y
        Arb[i] = D_A
        # Record arbitrage revenue (absolute value is used)
        CPMM_arb_revenue[i] = np.abs(D_A * P_A * eta1)
        
        # Retrieve boolean filters for systematic trades at this time step
        filtr_bf = filtr_bfs[i]
        filtr_sf = filtr_sfs[i]
        
        # --- Process systematic trades on the "bf" (buy-first) filter ---
        # First, process BUY trades for indices where filtr_bf is True
        X_bf, Y_bf, new_price_bf, D_buy_bf, rev_buy_bf = process_trade(
            Buy_Trade_CPMM, buy, S, X, Y, filtr_bf, eta0, eta1
        )
        # Update marginal price and reserves for the filtered indices after buy trade
        S1[i, filtr_bf] = new_price_bf
        X[filtr_bf] = X_bf
        Y[filtr_bf] = Y_bf
        Pool_X[i, filtr_bf] = X_bf
        Pool_Y[i, filtr_bf] = Y_bf
        # Record buy trade volume and revenue
        CPMM_buy[i, filtr_bf] = D_buy_bf
        CPMM_buy_revenue[i, filtr_bf] = rev_buy_bf

        # Then, process SELL trades for the same "bf" indices
        X_bf, Y_bf, new_price_bf, D_sell_bf, rev_sell_bf = process_trade(
            Sell_Trade_CPMM, sell, S, X, Y, filtr_bf, eta0, eta1
        )
        # Update marginal price and reserves for the filtered indices after sell trade
        S1[i, filtr_bf] = new_price_bf
        X[filtr_bf] = X_bf
        Y[filtr_bf] = Y_bf
        Pool_X[i, filtr_bf] = X_bf
        Pool_Y[i, filtr_bf] = Y_bf
        # Record sell trade volume and revenue
        CPMM_sell[i, filtr_bf] = D_sell_bf
        CPMM_sell_revenue[i, filtr_bf] = rev_sell_bf

        # --- Process systematic trades on the "sf" (sell-first) filter ---
        # First, process SELL trades for indices where filtr_sf is True
        X_sf, Y_sf, new_price_sf, D_sell_sf, rev_sell_sf = process_trade(
            Sell_Trade_CPMM, sell, S, X, Y, filtr_sf, eta0, eta1
        )
        # Update marginal price and reserves for the filtered indices after sell trade
        S1[i, filtr_sf] = new_price_sf
        X[filtr_sf] = X_sf
        Y[filtr_sf] = Y_sf
        Pool_X[i, filtr_sf] = X_sf
        Pool_Y[i, filtr_sf] = Y_sf
        # Record sell trade volume and revenue
        CPMM_sell[i, filtr_sf] = D_sell_sf
        CPMM_sell_revenue[i, filtr_sf] = rev_sell_sf

        # Then, process BUY trades for the same "sf" indices
        X_sf, Y_sf, new_price_sf, D_buy_sf, rev_buy_sf = process_trade(
            Buy_Trade_CPMM, buy, S, X, Y, filtr_sf, eta0, eta1
        )
        # Update marginal price and reserves for the filtered indices after buy trade
        S1[i, filtr_sf] = new_price_sf
        X[filtr_sf] = X_sf
        Y[filtr_sf] = Y_sf
        Pool_X[i, filtr_sf] = X_sf
        Pool_Y[i, filtr_sf] = Y_sf
        # Record buy trade volume and revenue
        CPMM_buy[i, filtr_sf] = D_buy_sf
        CPMM_buy_revenue[i, filtr_sf] = rev_buy_sf

    # Return the complete simulation results as a tuple of arrays.
    sigma_hats = np.array(sigma_hats)
    return Pool_X, Pool_Y, S0_, S1, CPMM_buy_revenue, CPMM_sell_revenue, CPMM_arb_revenue, Hedging_port_val, sigma_hats, eta1s


def simulation_summary(M_, N_, T_, dt_, buy_, sell_, eta0_, eta1_, S0_, X_, Y_, filtr_bfs, filtr_sfs):
    """
    Runs the simulation and computes summary metrics across all market instances.
    
    Parameters:
        M_       : Number of market indices/instances.
        N_       : Number of time periods.
        T_       : Time horizon for the simulation.
        dt_      : Time step increment.
        buy_     : Constant trade size for systematic buyers.
        sell_    : Constant trade size for systematic sellers.
        eta0_    : CEX proportional cost.
        eta1_    : CPMM proportional cost.
        S0_      : An (N+1) x M array representing the CEX price time series.
        X_       : Initial CPMM dollar reserve.
        Y_       : Initial CPMM asset reserve.
        filtr_bfs: Boolean filter array for buy-first systematic trades.
        filtr_sfs: Boolean filter array for sell-first systematic trades.
    
    Returns:
        A numpy array with key performance metrics:
            [avg_buy_fee_rev, avg_sell_fee_rev, avg_arb_fee_rev, avg_imp_loss, avg_hedge_val]
            where:
                - avg_buy_fee_rev  : Average revenue from buy trades.
                - avg_sell_fee_rev : Average revenue from sell trades.
                - avg_arb_fee_rev  : Average revenue from arbitrage trades.
                - avg_imp_loss     : Average impermanent loss over the simulation horizon.
                - avg_hedge_val    : Average hedging portfolio value at the final time step.
    """
    # Run the full simulation and collect time-series data.
    Pool_X, Pool_Y, S0_, S1, CPMM_buy_revenue, CPMM_sell_revenue, CPMM_arb_revenue, Hedging_port_val = \
        simulation(M_, N_, T_, dt_, buy_, sell_, eta0_, eta1_, S0_, X_, Y_, filtr_bfs, filtr_sfs)
    
    # Compute the average revenue from buy trades across all market instances.
    avg_buy_fee_rev = np.mean(np.sum(CPMM_buy_revenue, axis=0))
    # Compute the average revenue from sell trades.
    avg_sell_fee_rev = np.mean(np.sum(CPMM_sell_revenue, axis=0))
    # Compute the average revenue from arbitrage trades.
    avg_arb_fee_rev = np.mean(np.sum(CPMM_arb_revenue, axis=0))
    # Compute the average impermanent loss:
    # Difference between final and initial portfolio values (dollar reserve + asset reserve * CEX price).
    avg_imp_loss = np.mean(Pool_X[-1] + Pool_Y[-1] * S0_[-1] - Pool_X[0] - Pool_Y[0] * S0_[-1])
    # Compute the average hedging portfolio value at the final time step.
    avg_hedge_val = np.mean(Hedging_port_val[-1])

    # Return a summary array containing key performance metrics.
    return np.array([avg_buy_fee_rev, avg_sell_fee_rev, avg_arb_fee_rev, avg_imp_loss, avg_hedge_val])
    
    
def simulation_summary_antithetic(M_, N_, T_, dt_, buy_, sell_, eta0_, eta1_, S0_, S00_, X_, Y_, filtr_bfs, filtr_sfs):
    """
    Computes the simulation summary using antithetic variates for variance reduction.
    
    Parameters:
        M_, N_, T_, dt_, buy_, sell_, eta0_, eta1_, X_, Y_, filtr_bfs, filtr_sfs:
            Same as for simulation_summary.
        S0_   : Primary CEX price series for the simulation.
        S00_  : Antithetic CEX price series (a second CEX price series with opposite characteristics).
    
    Returns:
        A numpy array containing the averaged summary metrics based on the antithetic simulation runs.
    """
    # Run the simulation with the primary CEX price series.
    out0 = simulation_summary(M_, N_, T_, dt_, buy_, sell_, eta0_, eta1_, S0_, X_, Y_, filtr_bfs, filtr_sfs)
    # Run the simulation with the antithetic CEX price series.
    out1 = simulation_summary(M_, N_, T_, dt_, buy_, sell_, eta0_, eta1_, S00_, X_, Y_, filtr_bfs, filtr_sfs)

    # If the CPMM proportional cost is lower than the CEX proportional cost,
    # perform additional simulations with inverted trade filters to further reduce variance.
    if eta1_ < eta0_:
        # Invert the buyer and seller filters.
        out2 = simulation_summary(M_, N_, T_, dt_, buy_, sell_, eta0_, eta1_, S0_, X_, Y_, ~filtr_bfs, ~filtr_sfs)
        out3 = simulation_summary(M_, N_, T_, dt_, buy_, sell_, eta0_, eta1_, S00_, X_, Y_, ~filtr_bfs, ~filtr_sfs)
        # Average the outputs from all four simulation runs.
        output = (out0 + out1 + out2 + out3) / 4
    else:
        # Otherwise, average the outputs from the two simulation runs.
        output = (out0 + out1) / 2

    return output
