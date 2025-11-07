# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 14:02:20 2025

@author: Abhi
"""

import numpy as np
import matplotlib.pyplot as plt
from TightBindingModelDraft import TightBindingModel

# -----------------------------
# Haldane model parameters
# -----------------------------
t  = 1.0        # nearest-neighbour hopping (real)
t2 = 0.3     # next-nearest-neighbour hopping magnitude
phi = np.pi/2   # phase for NNN (Haldane flux)
M  = 0.0      # sublattice staggered mass (onsite: +M on A, -M on B)

# Use the same lattice convention as in your file (a1, a2 rows)
a1 = np.array([1.0, 0.0])
a2 = np.array([0.5, np.sqrt(3)/2.0])
lattice = np.array([a1, a2])

# fractional orbital positions within unit cell (A and B)
orbitals = [[0.0, 0.0], [1/3, 1/3]]

tb = TightBindingModel(lattice_vec=lattice, lattice_orb=orbitals)


# nearest neighbours interactions from A -> B:
nn_Rs = [(0, 0), (-1, 0), (0, -1)]
for R in nn_Rs:
    tb.update_hopping_matrix(-t, 0, 1, R)   # convention: -t for hopping

# next-nearest neighbour interactions in terms of displacement vectors:
nnn_Rs = [(1, 0), (0, 1), (1, -1)]

# updating the hopping matrix with the flux included, the Hermitian conjugates are added automatically.
for R in nnn_Rs:
    tb.update_hopping_matrix(t2 * np.exp(1j * phi), 0, 0, R)   # A->A with +phi
    tb.update_hopping_matrix(t2 * np.exp(-1j * phi), 1, 1, R)  # B->B with -phi

#staggered onsite potentials giving us a band gap if M is non-zero
tb.update_onsite_energies([ M, -M ])   # +M on A, -M on B


pts = {"Γ":[0.0, 0.0], "K":[1/3, 2/3], "M":[-1/2, 1/3], "Γ2":[0.0,0.0]}
path = ["Γ", "K", "M", "Γ2"]
Nk = 300
tb.plot_bandstructure(pts, path, Nk_per_segment=Nk)

#computing the Chern number at occupied band filling of 1
chern = tb.chern_number_FHZ_method(Nkx=100, Nky=100, occ_bands=1)
print("Chern number (occupied band) =", chern)


'''

#---------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------#
# --- Since TightBinding class is already defined --- #
#---------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------#

a1 = np.array([1, 0])
a2 = np.array([1/2, np.sqrt(3)/2])
lattice = np.array([a1, a2])            # rows = a1, a2  (NO transpose)

orbital_positions = [
    [0.0, 0.0],   # A
    [1/3, 1/3]    # B
]

tb = TightBindingModel(lattice_vec=lattice, lattice_orb=orbital_positions)
t = 1
tb.update_hopping_matrix(t, 0, 1, (0,0))
tb.update_hopping_matrix(t, 0, 1, (1,0))
tb.update_hopping_matrix(t, 0, 1, (0,1))

pts = {"Γ": [0, 0], "K1": [1/3, 2/3], "K2": [-1/3, -2/3], "M": [0.5, 0.0]}
path = ["Γ", "K1", "K2", "Γ"]

for name, kred in [("K1",[1/3,2/3]), ("K2",[-1/3,-2/3]), ("Gamma",[0,0]), ("M",[0,1/3])]:
    Hk = tb.H_k(kred)
    print(name, " f=", np.round(Hk[0,1],9), " eigs=", np.round(np.linalg.eigvalsh(Hk),9))

tb.plot_bandstructure(pts, path, 200)
c_n = tb.chern_number_FHZ_method(80, 80, 1)
print("Chern number is, ", c_n)
'''