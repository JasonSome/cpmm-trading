#!/bin/sh
#
# SLURM job script for running the parallelized AMM simulation.
#
#SBATCH --account=stats           # Replace with your group account name
#SBATCH --job-name=AMM_sim        # Job name
#SBATCH -c 12                     # The number of cpu cores to use (the script these cores in the simulation)
#SBATCH -t 0-10:00                # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=5gb         # Memory per CPU core

# Load the Anaconda module
module load anaconda

# Run the Python simulation script with command-line parameters.
# eta0: CEX fee, mu: Asset drift, br: buy rate per unit time, sr: sell rate per unit time.
python3 run_AMM_sim.py --eta0 0.0025 --mu 0.0 --br 2500.0 --sr 2500.0
