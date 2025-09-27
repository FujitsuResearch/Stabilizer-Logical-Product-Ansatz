# Stabilizer-Logical-Product-Ansatz (SLPA)
This repository contains the source codes for the stabilizer-logical product ansatz (SLPA), a novel model for efficient gradient estimation in variational quantum algorithms.
It has been introduced in the paper titled "Trade-off between Gradient Measurement Efficiency and Expressivity in Deep Quantum Neural Networks."

Links: https://www.nature.com/articles/s41534-025-01036-7, https://arxiv.org/abs/2406.18316

# Installation

1. **Clone the repository**
```bash
git clone https://github.com/FujitsuResearch/Stabilizer-Logical-Product-Ansatz.git
cd Stabilizer-Logical-Product-Ansatz
```

2. **Install the required packages for Python (tested with v3.10.12)**
```bash
# Create an environment
python -m venv .venv

# Activate the environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

# Usage
This repository is organized into four main folders, each corresponding to specific figures in our paper or providing a general implementation of SLPA. 
Below is a guide to help you navigate the codebase and reproduce our results:
- `/commutator` for Fig.3 in the paper
- `/unitary_learning` for Fig.4 in the paper
- `/barren_plateau` for Fig.10 in the paper
- `/SLPA_sample` for general usage of SLPA

## `/commutator`
The main script is `/commutator/commutator.ipynb`. 
This code calculates the commutator between gradient operators, evaluating the gradienet measurement efficiency.
One can reproduce Fig.3 by executing the cells step by step.

## `/unitary_learning`
There are three main scripts: `main_finite_cbc.py`, `main_finite_sc.py`, and `main_finite_nsc.py`, which solve the unitary learning problem (symmetric function learning problem) with SLPA, a symmetric circuit, and a non-symmetric circuit, respectively.
`job_cbc.sh`, `job_sc.sh`, `job_nsc.sh` are job scripts.
`output_cbc_ns=1000`, `output_sc_ns=1000`, and `output_nsc_ns=1000` are directories containing calculation output files recording the changes in training and test loss.
`results.ipynb` is a script for plotting the results from these output files.

## `/barren_plateau`
The main scripts are `/barren_plateau/barren_plateau_sc2.ipynb` and `/barren_plateau/barren_plateau_cbc2.ipynb`. 
These codes calculate the variance of the cost function with varying the number of qubits and the depth of the circuit.
Fig.10 can be plotted in the latter part of `/barren_plateau/barren_plateau_cbc2.ipynb` based on the output files from these two codes.

## `/SLPA_sample`
The codes in `/unitary_learning` are specifically tailored for the problem addressed in the paper. 
For a more general and flexible implementation of SLPA suitable for various problems, please refer to `sample.ipynb` in this folder.
