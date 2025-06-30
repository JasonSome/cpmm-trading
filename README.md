# CPMM Trading

Repository for code that runs and analyzes simulations for CPMM performance.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Directory Structure](#directory-structure)

## Overview

Simulation code corresponding to the paper ``Optimal Fees for Liquidity Provision in Automated Market Makers" by Steven Campbell, Philippe Bergault, Jason Milionis and Marcel Nutz. The code base can be used to study the key determinants of AMM LP profitability in markets where an AMM operates in parallel with a centralized exchange (CEX), and both arbitrageurs and fundamental traders are afforded the option to interact with either venue.

## Installation

To run the simulations and notebooks locally clone the repository, set up a Python environment and install the necessary dependencies (see the code files and jupyter notebooks for details).

## Usage

The files are organized using the below directory structure. To run the notebooks you will need to include the src/ and data/ files in the same directory as the notebook.

## Directory Structure

The main python code files are located in the src/ directory.
The Jupyter notebooks are located in the notebooks/ directory. 
The files needed to run the simulation on an HPC cluster are located in the hpc_files/ directory.
The simulation output files are stored in the data/ directory.
