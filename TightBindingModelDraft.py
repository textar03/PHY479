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
    
    lattice_vec: The Bravais lattice vectors a1, a2, a3,...
    lattice_orb: The position of the orbitals in fractional coordinates
    """
    
    def __init__(self, lattice_vec, lattice_orb):
        
        self.lattice_vec = np.array(lattice_vec, dtype=float)
        self.reciprocal_lattice_vec = 2 * np.pi * np.linalg.inv(self.lattice_vec).T
        self.lattice_orb = np.array(lattice_orb, dtype=float)
        self.norbs = len(lattice_orb)

        self.hopping_element = {}
        self.pos_orbitals = self.lattice_orb @ self.lattice_vec
        self.onsite_energies = np.zeros(self.norbs)

    def update_hopping_matrix(self, T_ij, i, j, R):
        """
        Add hopping from orbital i in cell 0 to orbital j in cell R.
        Also adds the Hermitian conjugate term automatically.
        """
        R_tuple = tuple(R)
        self.hopping_element[(i, j, R_tuple)] = T_ij
        # add Hermitian partner
        self.hopping_element[(j, i, tuple(-np.array(R_tuple)))] = np.conjugate(T_ij)

    def update_onsite_energies(self, energies):
        """Set onsite energies."""
        self.onsite_energies = np.array(energies, dtype=float)
    
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

        for i in range(self.norbs):
            H_ij[i, i] += self.onsite_energies[i]
            
        # Enforce Hermiticity numerically
        H_ij = 0.5 * (H_ij + H_ij.conj().T)
        return H_ij

    def H_energies(self, k_val):
        
        Hk = self.H_k(k_val)
        eigs = np.linalg.eigvalsh(Hk)
        return np.real_if_close(eigs)
    
    
    def high_symmetry_path(self, high_symmetry_points, path_values, Nk_per_segment):
    
        # build list of reduced coords segments (including endpoints)
        segments = []
        for i in range(len(path_values) - 1):
            k_start = np.array(high_symmetry_points[path_values[i]])
            k_end   = np.array(high_symmetry_points[path_values[i + 1]])
            seg = np.linspace(k_start, k_end, Nk_per_segment, endpoint=True)
            # drop last point to avoid duplication at internal nodes (keep final segment's endpoint)
            if i < len(path_values) - 2:
                seg = seg[:-1]
            segments.append(seg)
    
        # concatenate segments into full k-path
        k_list = np.vstack(segments)
        Npts = len(k_list)
    
        # compute Cartesian k (k_cart = k_reduced @ b_vectors)
        k_cart = np.dot(k_list, self.reciprocal_lattice_vec)  # shape (Npts, ndim)
    
        # compute cumulative distance along path for x-axis
        if Npts > 1:
            deltas = np.linalg.norm(np.diff(k_cart, axis=0), axis=1)
            k_dist = np.concatenate(([0.0], np.cumsum(deltas)))
        else:
            k_dist = np.array([0.0])
    
        # compute tick positions (indices) for each high-symmetry point:
        # start indices of segments, plus final index = Npts-1
        tick_positions = []
        cum = 0
        for seg in segments:
            tick_positions.append(cum)   # start of this segment = index of a high-symmetry point
            cum += len(seg)
        # append final index (last high-symmetry point)
        tick_positions.append(max(0, Npts - 1))
    
        return k_list, k_dist, tick_positions


    def plot_bandstructure(self, sym_points, path_val, Nk_per_segment=200):
        """
        Smooth band-structure plot along high-symmetry path using path-length x-axis.
        """
        klist, kdist, ticks = self.high_symmetry_path(sym_points, path_val, Nk_per_segment)
    
        # Evaluate energies along klist
        energies = np.array([self.H_energies(k) for k in klist])  # shape (Npts, nbands)
    
        fig, ax = plt.subplots(figsize=(7,4))
        nbands = energies.shape[1]
        for n in range(nbands):
            ax.plot(kdist, energies[:, n], color='k', lw=1)
    
        # vertical lines and xticks at high-symmetry points (use kdist[ticks[i]])
        for t_idx in ticks:
            ax.axvline(kdist[t_idx], color='gray', lw=0.6)
    
        ax.set_xticks([kdist[t] for t in ticks])
        ax.set_xticklabels(path_val)
        ax.set_xlim(kdist[0], kdist[-1])
        ax.set_ylabel(r"$E/t$")
        ax.set_title("Band structure")
        fig.tight_layout()
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


    def chern_number_FHZ_method(self, Nkx, Nky, occ_bands, return_flux_grid=False):
        """
        Compute the total Chern number of the lowest occupied bands using the
        Fukui-Hatsugai-Suzuki discrete method.
    
        Nkx: discrete k-points in x-direction
        Nky: discrete k-points in y-direction
        occ_bands: the number of occupied bands beyond the Fermi energy
        return_flux_grid: is a bool that returns flux per plaquette
        """
    
        kx_vals = np.linspace(0.0, 1.0, Nkx, endpoint=False)
        ky_vals = np.linspace(0.0, 1.0, Nky, endpoint=False)
    
        # storage allocation
        norb = self.norbs
        U = np.zeros((Nkx, Nky, norb, occ_bands), dtype=complex)
    
        #first we compute all occupied eigenvectors at each k
        for ix, kx in enumerate(kx_vals):
            for iy, ky in enumerate(ky_vals):
                kval = np.array([kx, ky])
                Hk = self.H_k(kval)
                eigvals, eigvecs = np.linalg.eigh(Hk)   # eigvecs columns are eigenvectors
                # take the lowest `occ_bands` eigenvectors
                U[ix, iy, :, :] = eigvecs[:, :occ_bands]
    
        # we make a smaller function to help us determine the determinant overlap which should be gauge invariant (phases cancel out)
        def compute_overlap(A, B):
            # A, B: shape (norb, occ_bands)
            # overlap matrix M = A^\dagger B (shape occ_bands x occ_bands)
            M = np.conjugate(A.T) @ B
            return np.linalg.det(M)
    
        #Compute link variables Ux and Uy (as determinants to cancel out phases of the wavefunctions) for each k
        Ux = np.zeros((Nkx, Nky), dtype=complex)
        Uy = np.zeros((Nkx, Nky), dtype=complex)
    
        for ix in range(Nkx):
            ix1 = (ix + 1) % Nkx
            for iy in range(Nky):
                iy1 = (iy + 1) % Nky
                Ux[ix, iy] = compute_overlap(U[ix, iy], U[ix1, iy])
                Uy[ix, iy] = compute_overlap(U[ix, iy], U[ix, iy1])
    
        # computing plaquette flux F at each index (ix, iy)
        F = np.zeros((Nkx, Nky), dtype=float)
        for ix in range(Nkx):
            ix1 = (ix + 1) % Nkx
            for iy in range(Nky):
                iy1 = (iy + 1) % Nky
                # we take the product all around the plaquette
                prod = Ux[ix, iy] * Uy[ix1, iy] / (Ux[ix, iy1] * Uy[ix, iy])
                #we take the log and then the imaginary to find the principal branch
                F[ix, iy] = np.imag(np.log(prod))
    
        # 4) sum fluxes and divide by 2*pi
        total_flux = np.sum(F)
        chern_n = total_flux / (2.0 * np.pi)
    
        chern_number = int(np.rint(chern_n))   # round to nearest integer
    
        if return_flux_grid:
            return chern_number, F
        
        return chern_number

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