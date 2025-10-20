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
        
        Hk = self.H_k(k_val)
        eigs = np.linalg.eigvalsh(Hk)
        return np.real_if_close(eigs)
    
    
    def high_symmetry_path(self, high_symmetry_points, path_values, Nk):
        
        k_list, tick_positions = [] , [0]
        for i in range(len(path_values) - 1):
            k_start = np.array(high_symmetry_points[path_values[i]])
            k_end = np.array(high_symmetry_points[path_values[i + 1]])
            segment = [
                k_start + (k_end - k_start) * t
                for t in np.linspace(0, 1, Nk, endpoint=False)
            ]
            k_list.extend(segment)
            tick_positions.append(len(k_list))
        return np.array(k_list), tick_positions


    def plot_bandstructure(self, sym_points, path_val, Nk_per_segment):
       
        klist, ticks = self.high_symmetry_path(sym_points, path_val, Nk_per_segment)

        energies = np.array([self.H_energies(k) for k in klist])

        plt.figure(figsize=(6,4))
        for n in range(energies.shape[1]):
            plt.plot(energies[:,n], 'k-', lw=1)
        for t in ticks:
            plt.axvline(t, color='gray', lw=0.5)
        plt.xticks(ticks, path_val)
        plt.ylabel("Energy (t units)")
        plt.title("Band structure")
        plt.tight_layout()
        plt.show()
        
        
        
#-------------------------------------- Open Boundary Conditions ----------------------------------------- #

def build_realspace_H(self, Ncells):
    
        ndim = len(Ncells)
        norb = self.norbs
        total_cells = np.prod(Ncells)
        Ntot = total_cells * norb
        H = np.zeros((Ntot, Ntot), dtype=complex)

        # --- Loop over all real-space cells ---
        if ndim == 1:
            Nx = Ncells[0]
            for x in range(Nx):
                for (i, j, R), T in self.hopping_element.items():
                    tx = x + R[0]
                    if 0 <= tx < Nx:
                        a = (x * norb) + i
                        b = (tx * norb) + j
                        H[a, b] += T

        elif ndim == 2:
            Nx, Ny = Ncells
            for x in range(Nx):
                for y in range(Ny):
                    for (i, j, R), T in self.hopping_element.items():
                        tx, ty = x + R[0], y + R[1]
                        if 0 <= tx < Nx and 0 <= ty < Ny:
                            a = (x * Ny + y) * norb + i
                            b = (tx * Ny + ty) * norb + j
                            H[a, b] += T

        elif ndim == 3:
            Nx, Ny, Nz = Ncells
            for x in range(Nx):
                for y in range(Ny):
                    for z in range(Nz):
                        for (i, j, R), T in self.hopping_element.items():
                            tx, ty, tz = x + R[0], y + R[1], z + R[2]
                            if 0 <= tx < Nx and 0 <= ty < Ny and 0 <= tz < Nz:
                                a = ((x * Ny + y) * Nz + z) * norb + i
                                b = ((tx * Ny + ty) * Nz + tz) * norb + j
                                H[a, b] += T

        else:
            raise ValueError("Only up to 3D supported for open BCs.")

        # --- Add onsite energies ---
        for c in range(total_cells):
            base = c * norb
            for i in range(norb):
                H[base + i, base + i] += self.onsite_energies[i]

        # Hermitize
        H = 0.5 * (H + H.conj().T)
        return H

# --- Since TightBinding class is already defined ---


a1 = np.array([1, 0])
a2 = np.array([1/2, np.sqrt(3)/2])
lattice = np.array([a1, a2])            # rows = a1, a2  (DO NOT transpose)

orbital_positions = [
    [0.0, 0.0],   # A
    [1/3, 1/3]    # B
]

tb = TightBindingModel(lattice_vec=lattice, lattice_orb=orbital_positions)
t = -1
tb.update_hopping_matrix(t, 0, 1, (0,0))
tb.update_hopping_matrix(t, 0, 1, (1,0))
tb.update_hopping_matrix(t, 0, 1, (0,1))

pts = {"Γ": [0, 0], "K1": [1/3, 2/3], "K2": [-1/3, -2/3], "M": [0.5, 0.0]}
path = ["Γ", "K1", "K2", "Γ"]

for name, kred in [("K1",[1/3,2/3]), ("K2",[-1/3,-2/3]), ("Gamma",[0,0]), ("M",[0,1/3])]:
    Hk = tb.H_k(kred)
    print(name, " f=", np.round(Hk[0,1],9), " eigs=", np.round(np.linalg.eigvalsh(Hk),9))


tb.plot_bandstructure(pts, path, 50)