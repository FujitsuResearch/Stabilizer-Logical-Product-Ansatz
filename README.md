# Stabilizer-Logical-Product-Ansatz
This repository contains the source codes for the stabilizer-logical product ansatz (SLPA) for efficient gradient estimation in variational quantum algorithms, which has been introduced in the paper titled "Trade-off between Gradient Measurement Efficiency and Expressivity in Deep Quantum Neural Networks."

Links: https://www.nature.com/articles/s41534-025-01036-7, https://arxiv.org/abs/2406.18316

# Usage
This repository contains three folders:
- `/commutator` for Fig.3 in the paper
- `/unitary_learning` for Fig.4 in the paper
- `/barren_plateau` for Fig.10 in the paper

## `/commutator`
The main script is `/commutator/commutator.ipynb`. 
This code calculates the commutator between gradient operators, evaluating the gradienet measurement efficiency.
One can reproduce Fig.3 by executing the cells step by step.

## `/unitary_learning`

## `/barren_plateau`
The main scripts are `/barren_plateau/barren_plateau_sc2.ipynb` and `/barren_plateau/barren_plateau_cbc2.ipynb`. 
These codes calculate the variance of the cost function with varying the number of qubits and the depth of the circuit.
Fig.10 can be plotted in the latter part of `/barren_plateau/barren_plateau_cbc2.ipynb` based on the output files from these two codes.
