Portfolio Optimization Algorithms Simulation
This document outlines the implementation plan for simulating the algorithms presented in the paper "Large-scale portfolio optimization on a trapped-ion quantum computer".

The paper simulates a 250-asset universe, with clusters up to 64 qubits. Simulating more than ~20 qubits classically on a standard machine is extremely resource-intensive and often impossible without a supercomputer. To run the simulation in a reasonable amount of time, I propose scaling down the problem:

Asset Universe: 20-30 assets (instead of 250).
Hardware Qubit Limit ($Q_{max}$): 8-10 qubits (instead of 36/64). This will allow the full pipeline (including the BF-DCQO quantum simulation via Qiskit) to execute locally within minutes, while still demonstrating all the algorithms and producing the required graphs. Please let me know if this scaled-down approach is acceptable.
Proposed Changes
We will create a set of Python scripts to implement the entire pipeline. The codebase will be organized into logical modules.

Data Collection
[NEW] data_collection.py
Uses yfinance to fetch daily closing prices for a subset of S&P 500 stocks over a specified period.
Computes daily log-returns, expected returns vector ($\mu$), covariance matrix ($C$), and correlation matrix ($\mathscr{C}$).
Algorithm 1: Hardware-Aware Clustering
[NEW] clustering.py
Implements RMT denoising (Marchenko-Pastur spectral separation) to extract the structured correlation matrix ($\mathscr{C}_{\star}$).
Uses the Louvain method for initial community detection.
Implements the correlation-guided greedy splitting rule to enforce the hardware qubit limit ($Q_{max}$).
Algorithm 2 & 3: BF-DCQO Quantum Optimizer
[NEW] quantum_optimizer.py
Implements the QUBO/Ising mapping for the cardinality-constrained portfolio optimization problem.
Implements Algorithm 2 (DCQO): Constructs the Trotterized quantum circuit for digitized counterdiabatic evolution using qiskit.
Implements Algorithm 3 (BF-DCQO): Iteratively executes DCQO circuits, post-selects low-energy samples, and updates the bias fields.
Uses qiskit_aer for classical statevector/aer simulation of the quantum circuits.
Algorithm 4: Post-processing Local Search
[NEW] local_search.py
Implements Phase 1: Gradient-based Hamming weight recovery (cardinality repair).
Implements Phase 2: First-improvement swap-based local search preserving the cardinality constraint.
Main Execution and Visualization
[NEW] main.py
Orchestrates the full pipeline: Data Collection -> Clustering -> BF-DCQO on subproblems -> Recombination -> Local Search.
Generates performance distributions and risk-return trade-off graphs mimicking Figure 2 and Figure 3 from the paper.
Verification Plan
Automated Verification
Run individual tests for clustering sizes to ensure they do not exceed $Q_{max}$.
Verify that the local search effectively preserves the exact cardinality constraint $K$.
Verify that the BF-DCQO circuit correctly maps the asset Ising formulation.
Manual Verification
Execute main.py and inspect the generated plots for the energy distributions and risk-return scatter plots.
