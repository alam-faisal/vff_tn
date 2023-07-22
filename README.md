# Variational fast forwarding using tensor networks

Finds fast forwarding circuits for Hamiltonian evolution by representing the ansatz circuit as layers of MPOs. 

Written using NumPy (and a little bit of JAX) 

mpo.py contains the MPO class

models.py contains the Hamiltonians used (so far only the XY model) 

ansatz.py contains Ansatz class built from dressed single qubit gates in brickwall pattern (soon to include Givens rotation gates) 

environments.py computes the various environment tensors necessary to efficiently optimize the ansatz

nodewise_train contains the optimization methods

ipynb_tests includes some Jupyter notebooks testing the codebase and showing how to call the different functions
