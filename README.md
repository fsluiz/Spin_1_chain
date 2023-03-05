# Spin-1 Chain Evolution
This code simulates the evolution of spin-1 chains with different Hamiltonians. The module is developed in Python using several scientific computing libraries such as NumPy, SciPy, and h5py.

The class spin_chain_evolution contains methods to define and construct operators for the spin chain. The operators include Sx, Sy, Sz for each site in the chain, as well as tensor product operators for all sites. The class also includes methods to construct Hamiltonians for the model being simulated, which can be either a Spin-1 Bilinear Biquadratic Chain, Spin-1 XXZ Chains with Uniaxial Single-Ion-Type Anisotropy or Spins-1 Bond-Alternating XXZ Chains. At each step of the evolution of the Hamiltonian, it saves the correlations in hdf5 files, at the end it joins all the files in a single dataframe.

### Requirements
+ Python 3.x
+ NumPy
+ SciPy
+ h5py
