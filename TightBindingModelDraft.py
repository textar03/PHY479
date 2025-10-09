# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 23:22:57 2025

@author: Abhi
"""
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


class TightBindingModel:
    
    """
    This class provides the band structure of different lattice structures based on the tight binding model approach. We may have
    both open and periodic boundary conditions. The parameters passed are
    
    norbs: number of orbitals per cell
    nsites: number of cells
    lattice_vec: The Bravais lattice vectors a1, a2, a3,...
    boundary_condition: specifying whether we are working in a periodic bulk or a finite structure with open boundary conditions
    
    """
    
    def __init__(self, lattice_vec, lattice_orb):
        """
        lattice_vec : (d, d) array
            Each row is a real-space primitive vector a1, a2, ...
        lattice_orb : list of orbital positions (fractional coordinates)
        """
        self.lattice_vec = np.array(lattice_vec, dtype=float)
        self.reciprocal_lattice_vec = 2 * np.pi * np.linalg.inv(self.lattice_vec).T
        self.lattice_orb = np.array(lattice_orb, dtype=float)
        self.norbs = len(lattice_orb)

        # store hopping elements as dict: (i, j, R) → t_ij(R)
        self.hopping_element = {}
        # precompute orbital positions in Cartesian coords
        self.pos_orbitals = self.lattice_orb @ self.lattice_vec

    def update_hopping_matrix(self, T_ij, i, j, R):
        """
        Add hopping from orbital i in cell 0 to orbital j in cell R.
        Also adds the Hermitian conjugate term automatically.
        """
        R_tuple = tuple(R)
        self.hopping_element[(i, j, R_tuple)] = T_ij
        # add Hermitian partner
        self.hopping_element[(j, i, tuple(-np.array(R_tuple)))] = np.conjugate(T_ij)

    def H_k(self, k_val):
        """
        Construct Bloch Hamiltonian H(k) for a given k-point in reciprocal vector coordinates.
        k_val : list or array of fractional coordinates (k1, k2, ...)
        """
        H_ij = np.zeros((self.norbs, self.norbs), dtype=complex)

        # Convert reduced k to Cartesian k
        k_cart = np.dot(k_val, self.reciprocal_lattice_vec)

        # Loop over hoppings
        for (i, j, R), T_ij in self.hopping_element.items():
            # Cartesian orbital displacement
            delta_vec = self.pos_orbitals[j] - self.pos_orbitals[i]

            # Integer lattice displacement R -> real vector 
            R_vec = np.dot(np.array(R, dtype=float), self.lattice_vec)

            # Total displacement = lattice + orbital offset
            R_n = R_vec + delta_vec

            # Bloch phase
            geo_factor = np.exp(1j * np.dot(k_cart, R_n))
            H_ij[i, j] += T_ij * geo_factor

        # Enforce Hermiticity numerically
        H_ij = 0.5 * (H_ij + H_ij.conj().T)
        return H_ij

    def H_energies(self, k_val):
        """
        Compute eigenvalues of H(k)
        """
        Hk = self.H_k(k_val)
        eigs = np.linalg.eigvalsh(Hk)
        return np.real_if_close(eigs)
    
    
    '''
    The N1, N2, N3 parameters are the number of grid points you would like to input. By using ratios of the reciprocal lattice vectors we can then create a momentum
    space grid. 
    '''
    def define_k_spacegrid(self, N1, N2, N3):
        k_list = []
        for n1 in range(N1):
            for n2 in range(N2):
                for n3 in range(N3):
                    k_point = np.dot([(n1/N1), (n2/N2), (n3/N3)], self.reciprocal_lattice_vec)
                    k_list.append(k_point)
        return np.array(k_list)
    
    def plot_band_structure(self, k_list):
        return 0


# --- Once TightBinding class is already defined ---


a1 = np.array([1, 0])
a2 = np.array([1/2, np.sqrt(3)/2])
lattice = np.array([a1, a2])            # rows = a1, a2  (DO NOT transpose)

orbital_positions = [
    [0.0, 0.0],   # A
    [1/3, 1/3]    # B
]

tb = TightBindingModel(lattice_vec=lattice, lattice_orb=orbital_positions)
t = -2.7
tb.update_hopping_matrix(t, 0, 1, (0,0))
tb.update_hopping_matrix(t, 0, 1, (-1,0))
tb.update_hopping_matrix(t, 0, 1, (0,-1))

for name, kred in [("K1",[1/3,1/3]), ("K2",[-1/3,2/3]), ("Gamma",[0,0]), ("M",[0.5,0])]:
    Hk = tb.H_k(kred)
    print(name, " f=", np.round(Hk[0,1],9), " eigs=", np.round(np.linalg.eigvalsh(Hk),9))

"""
     def energies_from_klist(self, klist):

    def energies_from_afew_k(self, klist):

    def plot_bandstructure(self, hamiltonian, k_val):
        return 0
    

    
    #---------------------------------------------------------------------------------------------------------------#
    #If we have an open system for our lattice we will use real-space Hamiltonians
    def H_ij(self, [nsites]):
        Htot = zeros((product(nsites)*self.norb, product(nsites)*self.norb), dtype=complex)
        return 0

    def edge_states(self):
        return 0
    
    
class TB_diaginfullBZ(TightBindingModel)

    def calculate_density_of_state(self):
        return 0
"""