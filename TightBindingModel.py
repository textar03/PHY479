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
        self.lattice_dimension = np.array(lattice_vec).shape[0]
        self.lattice_vec = np.array(lattice_vec)
        self.hopping_element = {}
        self.norbs = len(lattice_orb)
        
        self.reciprocal_lattice_vec = 2 * np.pi * np.linalg.inv(self.lattice_vec).T
       
        if lattice_orb is None:
            lattice_orb = [[0.0] * self.lattice_dimension for i in range(self.norbs)]
        self.pos_orbitals = np.array(lattice_orb, dtype=float) @ self.lattice_vec
       
        
    def update_hopping_matrix(self, T_ij, i, j, R):
        R_tuple = tuple(R) #making sure that the tuple of cell displacement
        R_ctuple = tuple(-np.array(R))
        hop_key = (i, j, R_tuple)
        self.hopping_element[hop_key] = complex(T_ij)
        self.hopping_element[j, i, R_ctuple] = np.conjugate(T_ij) #ensures have that T_ij* = T_ij

    def update_onsite_energies(self, energy, i):
        return 0
   
   
    """
    Function that inputs hopping elements to the Hamiltonian, which we will then multiply by the geometric factor of the lattice/material.
    """
    def H_k(self, k_val):
        #We must construct the Hamiltonian in the momentum space in this
        H_ij = np.zeros((self.norbs, self.norbs), dtype=complex)
       
        k_momentum = np.dot(k_val, self.reciprocal_lattice_vec) #turning (k1,k2) into actual vector using dot product with b1,b2,...
       
        #We now iterate over key-value pairs in the dictionary containing the hopping matrix
        for (i, j, R), T_ij in self.hopping_element.items():
            delta_vec = self.pos_orbitals[j] - self.pos_orbitals[i]
            R_cartesian = np.dot(self.lattice_vec.T, (np.array(R) + delta_vec))
            geo_factor = np.exp(1j * np.dot(k_momentum, R_cartesian))
            H_ij[i, j] += T_ij * geo_factor
           
        return (1/2)*(np.conjugate(H_ij).T + H_ij) #creates a full Hermitian matrix by adding the conjugate transpose
 
    def H_energies(self, k_values):
        H_k = self.H_k(k_values)
        return np.linalg.eigvalsh(H_k)
    
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
    

    """
    This function will output a real-space Hamiltonian based on the hopping parameters and orbitals= indices. 
    """
    def H_ij(self, nsites):
        Htot = np.zeros((nsites*self.norbs, nsites*self.norbs), dtype=complex)
        
        
        
        
        return 0



# --- Assuming TightBindingModel class is already defined we test our code with the following snippet ---

# Graphene lattice
a1 = np.array([1.0, 0.0])
a2 = np.array([0.5, np.sqrt(3)/2])
lattice = np.array([a1, a2]).T  # shape (2,2)

# Orbital positions (fractions of a1, a2)
orbital_positions = [
    [0.0, 0.0],   # A sublattice
    [1/3, 1/3]    # B sublattice
]

graphene = TightBindingModel(lattice_vec=lattice, lattice_orb=orbital_positions)

# Nearest-neighbor hopping
t = -2.7
graphene.update_hopping_matrix(t, 0, 1, [0,0])   # A->B inside cell
graphene.update_hopping_matrix(t, 0, 1, [1,0])   # A->B in +a1
graphene.update_hopping_matrix(t, 0, 1, [0,1])   # A->B in +a2

# --- High-symmetry points in fractional coordinates ---
high_symmetry = {
    "Gamma": [0.0, 0.0],
    "K": [-1/3, 2/3],
    "M": [0.5, 0.5]
}

print("Graphene band energies at high-symmetry points:")
for name, kred in high_symmetry.items():
    eigs = graphene.H_energies(kred)
    print(f"{name:6s} (kred={kred}): {np.round(eigs, 6)}")


"""
     def energies_from_klist(self, klist):

    def energies_from_afew_k(self, klist):
   

    def edge_states(self):
        return 0
   
   
class TB_diaginfullBZ(TightBindingModel)

    def calculate_density_of_state(self):
        return 0
"""