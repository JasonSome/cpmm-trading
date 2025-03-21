#!/usr/bin/env python3
import os
import argparse
import numpy as np
import multiprocessing
import pickle
from joblib import Parallel, delayed
import AMM_sim_functions as sim
import fast_AMM_sim_functions as fsim

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description='Run AMM simulations with different parameters.')
parser.add_argument('--eta0', type=float, default=0.005, help='CEX proportional cost')
parser.add_argument('--mu', type=float, default=0.0, help='Asset drift')
parser.add_argument('--br', type=float, default=2500.0, help='Buy rate')
parser.add_argument('--sr', type=float, default=2500.0, help='Sell rate')
parser.add_argument('--M', type=int, default=10000, help='Number of sims')
args = parser.parse_args()

# Input Parameters (from command line or environment)
buy_rate = args.br # Purchase rate per unit time for systematic buyers
sell_rate = args.sr # Sale rate per unit time for systematic sellers
eta0 = args.eta0 # CEX proportional cost
mu = args.mu  # Mean of CEX price shock
M = args.M # Total number of sims
thds = int(os.environ.get('SLURM_CPUS_PER_TASK', 1)) # Get number of threads from job environment

# Fixed Market Parameters
T = 1 # Time horizon
N = 1440 # Number of periods
dt = T/N # Time increment
buy = buy_rate * dt # Constant trade size for systematic buyers
sell = -sell_rate * dt # Constant trade size for systematic sellers
X = 30000000 # Initial CPMM Dollar Reserves
Y = 10000 # Initial CPMM Asset Reserves
S = X/Y # Initial CEX Price
time = np.array([i * dt for i in range(N+1)]) # Time grid

# Generate buyer-first/seller-first trade filter
filtr_bfs , filtr_sfs =  sim.generate_trade_filters(N, M) 

# Set of AMM Fee values to test
eta1_vals = np.array([i*0.00001 for i in range(0,1001)])

# Set of sigma values for simulation
sigma_vals = np.array([i*0.001 for i in range(1,101)])

# Collect data
all_outputs = {}  # Dictionary to store outputs keyed by sigma.

for i in range(len(sigma_vals)):
    sigma_value = sigma_vals[i]
    
    # Set the sigma for this iteration and simulate the CEX price series.
    sigma = sigma_value
    S0 = sim.CEX_Price(S, mu, sigma, dt, N, M)
    
    # Run the parallel simulation for each eta1 value.
    output_vals = Parallel(n_jobs=thds)(
        delayed(fsim.fast_simulation_summary)(
            M, N, T, dt, buy, sell, eta0, eta1_vals[j], S0, X, Y, filtr_bfs, filtr_sfs
        )
        for j in range(len(eta1_vals))
    )
    
    # Store the output in the dictionary.
    all_outputs[sigma_value] = np.array(output_vals)

# Save the dictionary to a file
filename = "all_outputs_eta0_" + str(eta0) + "_mu_" + str(mu) \
            + "_buy_" + str(round(buy_rate)) + "_sell_" + str(round(sell_rate))+ ".pkl"

with open(filename, "wb") as f:
    pickle.dump(all_outputs, f)

print("Simulation successfully completed.")
