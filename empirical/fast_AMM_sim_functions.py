import numpy as np
from numba import njit

# -----------------------------------------------------------------------------
# CEX Price Simulation
# -----------------------------------------------------------------------------

def CEX_Price(S, mu, sigma, dt, N, M=1, seed=42):
    """
    Simulates the CEX price using a geometric Brownian motion model.
    
    Parameters:
        S    : Initial price.
        mu   : Drift coefficient.
        sigma: Volatility.
        dt   : Time step.
        N    : Number of time steps.
        M    : Number of market instances (default is 1).
        seed : Random seed for reproducibility (default is 42).
    
    Returns:
        S0   : A (N+1) x M array containing the CEX price series.
    """
    # Set random seed for reproducibility.
    np.random.seed(seed)

    # Pre-allocate array for the CEX price series.
    S0 = np.zeros((N+1, M))
    
    # Set initial CEX price for all market instances.
    S0[0] = S

    # Generate Brownian motion increments.
    dWs = np.sqrt(dt) * np.random.normal(0., 1., size=(N, M))

    # Update prices using the geometric Brownian motion formula.
    S0[1:] = S * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * dWs, axis=0))
    
    # Return the simulated CEX price series.
    return S0

# -----------------------------------------------------------------------------
# Trade Filter Generation
# -----------------------------------------------------------------------------

def generate_trade_filters(N, M, seed=42):
    """
    Generate trade filters for systematic trading orders.
    
    This function creates two boolean filter arrays of shape (N+1, M):
      - filtr_bfs: Indicates when a buyer trades first.
      - filtr_sfs: Indicates when a seller trades first.
    
    Parameters:
        N    : Number of time steps (filters generated for N+1 periods).
        M    : Number of market instances.
        seed : Random seed for reproducibility (default is 42).
    
    Returns:
        filtr_bfs : Boolean array for buyer‑trade‑first filters.
        filtr_sfs : Boolean array for seller‑trade‑first filters.
    """
    # Set random seed.
    np.random.seed(seed)
    
    # Generate a uniform random array for each time step and market instance.
    uni = np.random.uniform(0, 1, size=(N+1, M))
    
    # Buyer-trade-first filter: True if random value < 0.5.
    filtr_bfs = uni < 0.5
    
    # Seller-trade-first filter: Complement of filtr_bfs.
    filtr_sfs = uni >= 0.5
    
    return filtr_bfs, filtr_sfs

# -----------------------------------------------------------------------------
# Arbitrage and Systematic Trade Processing (Single Market Instance)
# -----------------------------------------------------------------------------

@njit
def arb_trade_single(S, X, Y, eta0, eta1):
    """
    Compute the arbitrage trade volume for a single market instance.
    
    Parameters:
        S    : Current CEX price.
        X    : Current dollar reserves.
        Y    : Current asset reserves.
        eta0 : CEX proportional cost.
        eta1 : CPMM proportional cost.
    
    Returns:
        Arbitrage trade volume.
    """
    # Compute marginal price.
    P = X / Y

    # Perform volume computation
    val1 = Y * (1 - np.sqrt((P * (1 - eta1)) / (S * (1 + eta0))))
    if val1 > 0:
        q1 = 0.0
    else:
        q1 = val1
    
    val2 = Y * (1 - np.sqrt((P * (1 + eta1)) / (S * (1 - eta0))))
    if val2 < 0:
        q2 = 0.0
    else:
        q2 = val2
        
    # Return arbitrage volume
    return q1 + q2

@njit
def process_buy_trade_single(trade_amt, S, X, Y, eta0, eta1):
    """
    Process a buy trade for one market instance.
    
    Parameters:
        trade_amt : Desired trade amount.
        S         : Current CEX price.
        X         : Current dollar reserves.
        Y         : Current asset reserves.
        eta0      : CEX proportional cost.
        eta1      : CPMM proportional cost.
    
    Returns:
        Updated dollar reserves, updated asset reserves, new marginal price,
        trade volume executed (D), and fee revenue from the trade.
    """
    # Compute current marginal price.
    P = X / Y
    
    # Determine maximum trade volume to be directed to the AMM.
    sqrt_term = np.sqrt((P * (1 + eta1)) / (S * (1 + eta0)))
    v = Y * (1 - sqrt_term)
    
    # Use the minimum of the desired trade amount and the maximum volume.
    if trade_amt > v:
        D = v
    else:
        D = trade_amt

    # Ensure trade volume is non-negative.
    if D < 0:
        D = 0
    
    # Calculate execution price based on updated reserves.
    P_trade = X / (Y - D)
    # Update reserves after the trade.
    X_new = X + P_trade * D
    Y_new = Y - D
    # Compute fee revenue (positive for buy trades).
    revenue = D * P_trade * eta1
    # Determine new marginal price.
    new_price = X_new / Y_new
    return X_new, Y_new, new_price, D, revenue

@njit
def process_sell_trade_single(trade_amt, S, X, Y, eta0, eta1):
    """
    Process a sell trade for one market instance.
    
    Parameters:
        trade_amt : Desired trade amount.
        S         : Current CEX price.
        X         : Current dollar reserves.
        Y         : Current asset reserves.
        eta0      : CEX proportional cost.
        eta1      : CPMM proportional cost.
    
    Returns:
        Updated dollar reserves, updated asset reserves, new marginal price,
        trade volume executed (D), and fee revenue from the trade.
    """
    # Compute current marginal price.
    P = X / Y
    
    # Determine maximum trade volume to be directed to the AMM for a sell trade.
    sqrt_term = np.sqrt((P * (1 - eta1)) / (S * (1 - eta0)))
    v = Y * (1 - sqrt_term)

    # For sell trades, if the desired trade amount is less than the limit, use the limit.
    if trade_amt < v:
        D = v
    else:
        D = trade_amt

    # Sell trades should yield a negative trade volume; ensure D is non-positive.
    if D > 0:
        D = 0
    
    # Calculate the execution price.
    P_trade = X / (Y - D)
    # Update reserves after the sell trade.
    X_new = X + P_trade * D
    Y_new = Y - D
    # Fee revenue is negative for sell trades.
    revenue = -D * P_trade * eta1
    # Determine new marginal price.
    new_price = X_new / Y_new
    return X_new, Y_new, new_price, D, revenue

# -----------------------------------------------------------------------------
# Fast Simulation Functions (Vectorized over Markets & Time)
# -----------------------------------------------------------------------------

@njit
def fast_simulation(M, N, T, dt, buy_amt, sell_amt, eta0, eta1, S0, X0, Y0, filtr_bfs, filtr_sfs):
    """
    Run a market simulation with arbitrage and systematic trades.
    
    For each time step:
      - Process arbitrage trades for every market instance.
      - Then, based on the boolean filters, process systematic trades:
          * If filtr_bfs[i,j] is True, process a buy trade followed by a sell trade.
          * If filtr_sfs[i,j] is True, process a sell trade followed by a buy trade.
    
    Parameters:
        M         : Number of market instances.
        N         : Number of time steps.
        T         : Total simulation time (not directly used in the loop).
        dt        : Time increment per step.
        buy_amt   : Trade amount for systematic buy orders.
        sell_amt  : Trade amount for systematic sell orders.
        eta0      : CEX proportional cost.
        eta1      : CPMM proportional cost.
        S0        : (N+1) x M array of CEX prices.
        X0, Y0    : Initial CPMM dollar and asset reserves.
        filtr_bfs : Boolean filter array for buyer‑trade‑first orders.
        filtr_sfs : Boolean filter array for seller‑trade‑first orders.
    
    Returns:
        Tuple containing:
          - Pool_X: Dollar reserves over time.
          - Pool_Y: Asset reserves over time.
          - S0: Original CEX prices.
          - S1: Updated CPMM marginal prices over time.
          - CPMM_buy_revenue: Fee revenue from buy trades.
          - CPMM_sell_revenue: Fee revenue from sell trades.
          - CPMM_arb_revenue: Fee revenue from arbitrage trades.
          - Hedging_port_val: Hedging portfolio value over time.
    """
    # Pre-allocate arrays for simulation metrics.
    Pool_X = np.empty((N+1, M))
    Pool_Y = np.empty((N+1, M))
    S1 = np.empty((N+1, M))
    Arb = np.empty((N+1, M))
    CPMM_buy = np.empty((N+1, M))
    CPMM_sell = np.empty((N+1, M))
    CPMM_buy_revenue = np.empty((N+1, M))
    CPMM_sell_revenue = np.empty((N+1, M))
    CPMM_arb_revenue = np.empty((N+1, M))
    Hedging_port_val = np.empty((N+1, M))
    
    # Initialize values at time 0 for each market instance.
    for j in range(M):
        Pool_X[0, j] = X0
        Pool_Y[0, j] = Y0
        S1[0, j] = X0 / Y0
        Arb[0, j] = 0.0
        CPMM_buy[0, j] = 0.0
        CPMM_sell[0, j] = 0.0
        CPMM_buy_revenue[0, j] = 0.0
        CPMM_sell_revenue[0, j] = 0.0
        CPMM_arb_revenue[0, j] = 0.0
        # Hedging portfolio is initialized as the sum of dollar reserves plus asset value.
        Hedging_port_val[0, j] = X0 + Y0 * S0[0, j]
    
    # Main simulation loop over time steps.
    for i in range(1, N+1):
        # Process each market instance separately.
        for j in range(M):
            # Current CEX price at time i for market instance j.
            S = S0[i, j]

            # Update hedging portfolio value:
            # Increase from previous hedging value plus the change in asset value.
            Hedging_port_val[i, j] = Hedging_port_val[i-1, j] + Pool_Y[i-1, j] * (S - S0[i-1, j])
            
            # Retrieve previous reserves.
            X_prev = Pool_X[i-1, j]
            Y_prev = Pool_Y[i-1, j]
            # Process arbitrage trade.
            D_A = arb_trade_single(S, X_prev, Y_prev, eta0, eta1)
            # Compute execution price for arbitrage trade.
            P_A = X_prev / (Y_prev - D_A)
            # Update reserves based on arbitrage trade.
            X_new = X_prev + P_A * D_A
            Y_new = Y_prev - D_A
            Pool_X[i, j] = X_new
            Pool_Y[i, j] = Y_new
            Arb[i, j] = D_A
            # Record arbitrage fee revenue (absolute value).
            CPMM_arb_revenue[i, j] = np.abs(D_A * P_A * eta1)
            # Update marginal price after arbitrage.
            S1[i, j] = X_new / Y_new
        
            # Process systematic trades based on trade filter.
            if filtr_bfs[i, j]:
                # Buyer-first: Process a buy trade first.
                X_old = Pool_X[i, j]
                Y_old = Pool_Y[i, j]
                S = S0[i, j]
                X_new, Y_new, new_price, D_buy, rev_buy = process_buy_trade_single(buy_amt, S, X_old, Y_old, eta0, eta1)
                Pool_X[i, j] = X_new
                Pool_Y[i, j] = Y_new
                S1[i, j] = new_price
                CPMM_buy[i, j] = D_buy
                CPMM_buy_revenue[i, j] = rev_buy

                # Then process a sell trade immediately.
                X_old = Pool_X[i, j]
                Y_old = Pool_Y[i, j]
                X_new, Y_new, new_price, D_sell, rev_sell = process_sell_trade_single(sell_amt, S, X_old, Y_old, eta0, eta1)
                Pool_X[i, j] = X_new
                Pool_Y[i, j] = Y_new
                S1[i, j] = new_price
                CPMM_sell[i, j] = D_sell
                CPMM_sell_revenue[i, j] = rev_sell
        
            if filtr_sfs[i, j]:
                # Seller-first: Process a sell trade first.
                X_old = Pool_X[i, j]
                Y_old = Pool_Y[i, j]
                S = S0[i, j]
                X_new, Y_new, new_price, D_sell, rev_sell = process_sell_trade_single(sell_amt, S, X_old, Y_old, eta0, eta1)
                Pool_X[i, j] = X_new
                Pool_Y[i, j] = Y_new
                S1[i, j] = new_price
                CPMM_sell[i, j] = D_sell
                CPMM_sell_revenue[i, j] = rev_sell

                # Then process a buy trade immediately.
                X_old = Pool_X[i, j]
                Y_old = Pool_Y[i, j]
                X_new, Y_new, new_price, D_buy, rev_buy = process_buy_trade_single(buy_amt, S, X_old, Y_old, eta0, eta1)
                Pool_X[i, j] = X_new
                Pool_Y[i, j] = Y_new
                S1[i, j] = new_price
                CPMM_buy[i, j] = D_buy
                CPMM_buy_revenue[i, j] = rev_buy

    # Return simulation arrays.
    return Pool_X, Pool_Y, S0, S1, CPMM_buy, CPMM_sell, Arb, CPMM_buy_revenue, CPMM_sell_revenue, CPMM_arb_revenue, Hedging_port_val

# -----------------------------------------------------------------------------
# Fast Simulation Summary
# -----------------------------------------------------------------------------

@njit
def fast_simulation_summary(M, N, T, dt, buy_amt, sell_amt, eta0, eta1, S0, X0, Y0, filtr_bfs, filtr_sfs):
    """
    Run the market simulation and return a summary of key performance metrics.
    
    This function runs the same simulation as fast_simulation and then computes:
      - Average fee revenue from buy trades.
      - Average fee revenue from sell trades.
      - Average fee revenue from arbitrage trades.
      - Average pool value.
      - Average hedging portfolio value.
    
    Parameters:
        M, N, T, dt, buy_amt, sell_amt, eta0, eta1, S0, X0, Y0, filtr_bfs, filtr_sfs:
            Same as in fast_simulation.
    
    Returns:
        result : A 1D array containing the summarized metrics.
                 [avg_buy_fee_rev, avg_sell_fee_rev, avg_arb_fee_rev, avg_pool_val, avg_hedge_val]
    """
    # Pre-allocate arrays for simulation metrics.
    Pool_X = np.empty((N+1, M))
    Pool_Y = np.empty((N+1, M))
    S1 = np.empty((N+1, M))
    Arb = np.empty((N+1, M))
    CPMM_buy = np.empty((N+1, M))
    CPMM_sell = np.empty((N+1, M))
    CPMM_buy_revenue = np.empty((N+1, M))
    CPMM_sell_revenue = np.empty((N+1, M))
    CPMM_arb_revenue = np.empty((N+1, M))
    Hedging_port_val = np.empty((N+1, M))
    
    # Initialize time 0 values for each market instance.
    for j in range(M):
        Pool_X[0, j] = X0
        Pool_Y[0, j] = Y0
        S1[0, j] = X0 / Y0
        Arb[0, j] = 0.0
        CPMM_buy[0, j] = 0.0
        CPMM_sell[0, j] = 0.0
        CPMM_buy_revenue[0, j] = 0.0
        CPMM_sell_revenue[0, j] = 0.0
        CPMM_arb_revenue[0, j] = 0.0
        Hedging_port_val[0, j] = X0 + Y0 * S0[0, j]
    
    # Main simulation loop.
    for i in range(1, N+1):
        # Process each market instance separately.
        for j in range(M):
            S = S0[i, j]
            # Update hedging portfolio value based on the change in the CEX price.
            Hedging_port_val[i, j] = Hedging_port_val[i-1, j] + Pool_Y[i-1, j] * (S - S0[i-1, j])
            
            X_prev = Pool_X[i-1, j]
            Y_prev = Pool_Y[i-1, j]
            # Process arbitrage trade.
            D_A = arb_trade_single(S, X_prev, Y_prev, eta0, eta1)
            P_A = X_prev / (Y_prev - D_A)
            X_new = X_prev + P_A * D_A
            Y_new = Y_prev - D_A
            Pool_X[i, j] = X_new
            Pool_Y[i, j] = Y_new
            Arb[i, j] = D_A
            CPMM_arb_revenue[i, j] = np.abs(D_A * P_A * eta1)
            S1[i, j] = X_new / Y_new
        
            if filtr_bfs[i, j]:
                # For buyer-first: process buy trade then sell trade.
                X_old = Pool_X[i, j]
                Y_old = Pool_Y[i, j]
                S = S0[i, j]
                X_new, Y_new, new_price, D_buy, rev_buy = process_buy_trade_single(buy_amt, S, X_old, Y_old, eta0, eta1)
                Pool_X[i, j] = X_new
                Pool_Y[i, j] = Y_new
                S1[i, j] = new_price
                CPMM_buy[i, j] = D_buy
                CPMM_buy_revenue[i, j] = rev_buy

                X_old = Pool_X[i, j]
                Y_old = Pool_Y[i, j]
                X_new, Y_new, new_price, D_sell, rev_sell = process_sell_trade_single(sell_amt, S, X_old, Y_old, eta0, eta1)
                Pool_X[i, j] = X_new
                Pool_Y[i, j] = Y_new
                S1[i, j] = new_price
                CPMM_sell[i, j] = D_sell
                CPMM_sell_revenue[i, j] = rev_sell
        
            if filtr_sfs[i, j]:
                # For seller-first: process sell trade then buy trade.
                X_old = Pool_X[i, j]
                Y_old = Pool_Y[i, j]
                S = S0[i, j]
                X_new, Y_new, new_price, D_sell, rev_sell = process_sell_trade_single(sell_amt, S, X_old, Y_old, eta0, eta1)
                Pool_X[i, j] = X_new
                Pool_Y[i, j] = Y_new
                S1[i, j] = new_price
                CPMM_sell[i, j] = D_sell
                CPMM_sell_revenue[i, j] = rev_sell

                X_old = Pool_X[i, j]
                Y_old = Pool_Y[i, j]
                X_new, Y_new, new_price, D_buy, rev_buy = process_buy_trade_single(buy_amt, S, X_old, Y_old, eta0, eta1)
                Pool_X[i, j] = X_new
                Pool_Y[i, j] = Y_new
                S1[i, j] = new_price
                CPMM_buy[i, j] = D_buy
                CPMM_buy_revenue[i, j] = rev_buy

    # Compute average fee revenues across all market instances.
    avg_buy_fee_rev = 0.0
    avg_sell_fee_rev = 0.0
    avg_arb_fee_rev = 0.0
    for j in range(M):
        for i in range(1, N+1):
            avg_buy_fee_rev += CPMM_buy_revenue[i, j]
            avg_sell_fee_rev += CPMM_sell_revenue[i, j]
            avg_arb_fee_rev += CPMM_arb_revenue[i, j]
    avg_buy_fee_rev /= M
    avg_sell_fee_rev /= M
    avg_arb_fee_rev /= M
    
    # Compute average values for the hedging portfolio and the CPMM pool.
    avg_hedge_val = 0.0
    avg_pool_val = 0.0
    for j in range(M):
        hedge_val = Hedging_port_val[N, j]
        pool_val = Pool_X[N, j] + Pool_Y[N, j] * S0[N, j]
        avg_hedge_val += hedge_val
        avg_pool_val += pool_val
    avg_hedge_val /= M
    avg_pool_val /= M

    # Pack the summarized results into a single array.
    result = np.empty(5)
    result[0] = avg_buy_fee_rev
    result[1] = avg_sell_fee_rev
    result[2] = avg_arb_fee_rev
    result[3] = avg_pool_val
    result[4] = avg_hedge_val
    
    return result
