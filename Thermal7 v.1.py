# -*- coding: utf-8 -*-
"""
2-Layer 1D Thermal Conduction Model Using a Semi-Implicit (Backward Euler) Scheme

This model simulates conduction through a two-layer system:
  - Layer 1 (0 ≤ x < L1): Silicon solar cells, with incoming solar flux at x=0.
  - Layer 2 (L1 ≤ x ≤ L_total): An aluminum radiator that radiates to deep space at x=L_total.

The conduction and storage terms are treated implicitly (backward Euler), while the nonlinear
radiative boundary fluxes are evaluated at the previous time step (semi-implicit treatment).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# =============================================================================
# 1) Simulation Setup (Time and Spatial Discretization Parameters)
# =============================================================================
# Define layer thicknesses (in meters)
L1 = 0.02         # Thickness of layer 1 (silicon solar cells)
L2 = 0.05         # Thickness of layer 2 (radiator)
L_total = L1 + L2 # Total thickness (0.07 m)

# Time discretization
t_total = 60000.0      # Total simulation time [s]
dt = 0.1              # Time step [s] (with implicit scheme dt can be larger than explicit)
n_steps = int(t_total / dt)

# Spatial discretization
N = 31                          # Number of spatial grid points
dx = L_total / (N - 1)          # Uniform grid spacing [m]
x = np.linspace(0, L_total, N)  # Spatial grid positions

# Effective surface areas (in m^2)
A_top = 1.0   # Effective surface area for solar absorption at the top (layer 1)
A_bot = 1.0   # Effective surface area for radiation at the bottom (layer 2)

# =============================================================================
# 2) Material Properties (Assigned to Each Spatial Cell)
# =============================================================================
# Create arrays for density (rho in kg/m³), specific heat (cp in J/(kg·K)),
# thermal conductivity (k in W/(m·K)), and volumetric internal heating (Q in W/m³).
rho = np.zeros(N)
cp  = np.zeros(N)
k   = np.zeros(N)
Q   = np.zeros(N)  # No internal heating in this example

# Here we assume each cell's effective thickness is dx
cell_length = np.full(N, dx)

# Assign properties based on the layer.
# For simplicity, grid points with x < L1 belong to layer 1; those with x >= L1 belong to layer 2.
for i in range(N):
    if x[i] < L1:
        # Layer 1: Silicon solar cells
        rho[i] = 2320.0    # kg/m³
        cp[i]  = 800.0     # J/(kg·K)
        k[i]   = 150.0     # W/(m·K)
        Q[i]   = 0.0
    else:
        # Layer 2: Aluminum radiator
        rho[i] = 2700.0    # kg/m³
        cp[i]  = 877.0     # J/(kg·K)
        k[i]   = 205.0     # W/(m·K)
        Q[i]   = 0.0

# =============================================================================
# 3) Boundary Conditions and Radiative Parameters
# =============================================================================
sigma = 5.670374419e-8  # Stefan-Boltzmann constant [W/(m²·K⁴)]
solar_flux = 1361.0     # Solar flux [W/m²]
T_env = 2.7             # Environment temperature [K] (deep space)

# Surface properties
alpha_top = 0.9  # Solar absorptivity at the top surface
eps_top   = 0.9  # Emissivity at the top surface
eps_bot   = 0.92 # Emissivity at the bottom surface

# =============================================================================
# 4) Initial Conditions and Storage for Time History
# =============================================================================
T = np.ones(N) * 290.0   # Initial temperature [K] everywhere
T_hist = np.zeros((n_steps, N))
T_hist[0, :] = T

# =============================================================================
# 5) Helper Function: Harmonic Mean for Conductivity at Interfaces
# =============================================================================
def k_interface(k1, k2):
    """Return the harmonic mean of two thermal conductivities."""
    if (k1 + k2) == 0:
        return 0.0
    return 2.0 * k1 * k2 / (k1 + k2)

# Precompute interface conductivities for i = 0,...,N-2.
# (These will be used in the matrix coefficients.)
k_half = np.zeros(N-1)
for i in range(N-1):
    k_half[i] = k_interface(k[i], k[i+1])

# =============================================================================
# 6) Time Integration Using a Semi-Implicit (Backward Euler) Method
# =============================================================================
# We assemble and solve a tridiagonal system at each time step.
# The radiative fluxes at the boundaries are evaluated using the previous time step.
for n in range(0, n_steps-1):
    # Create the coefficient matrix A and right-hand side vector r.
    A = np.zeros((N, N))
    r = np.zeros(N)
    
    # --- Top Boundary (i = 0) ---
    # Radiative flux at top evaluated explicitly from previous time step:
    q_top = alpha_top * A_top * solar_flux - eps_top * A_top * sigma * ((T[0])**4 - T_env**4)
    # Energy balance: 
    #   - (k_half[0]/dx^2) * T_1^(n+1) + (rho[0]*cp[0]/dt + k_half[0]/dx^2)*T_0^(n+1)
    #       = rho[0]*cp[0]*T_0^n/dt + q_top/dx - Q[0]
    A[0, 0] = rho[0]*cp[0]/dt + k_half[0]/(dx**2)
    A[0, 1] = - k_half[0]/(dx**2)
    r[0] = (rho[0]*cp[0]/dt)*T[0] + q_top/dx - Q[0]
    
    # --- Interior Nodes (i = 1 to N-2) ---
    for i in range(1, N-1):
        # Coefficients:
        # a_i = - k_half[i-1]/dx^2, c_i = - k_half[i]/dx^2,
        # b_i = rho[i]*cp[i]/dt + (k_half[i-1] + k_half[i])/dx^2.
        A[i, i-1] = - k_half[i-1] / (dx**2)
        A[i, i]   = rho[i]*cp[i]/dt + (k_half[i-1] + k_half[i])/(dx**2)
        A[i, i+1] = - k_half[i] / (dx**2)
        r[i] = (rho[i]*cp[i]/dt)*T[i] + Q[i]
    
    # --- Bottom Boundary (i = N-1) ---
    # Radiative flux at bottom evaluated explicitly:
    q_bot = eps_bot * A_bot * sigma * ((T[N-1])**4 - T_env**4)
    # Energy balance:
    #   - (k_half[N-2]/dx^2)*T_{N-2}^(n+1) + (rho[N-1]*cp[N-1]/dt + k_half[N-2]/dx^2)*T_{N-1}^(n+1)
    #       = rho[N-1]*cp[N-1]*T_{N-1}^n/dt - q_bot/dx - Q[N-1]
    A[N-1, N-2] = - k_half[N-2] / (dx**2)
    A[N-1, N-1] = rho[N-1]*cp[N-1]/dt + k_half[N-2]/(dx**2)
    r[N-1] = (rho[N-1]*cp[N-1]/dt)*T[N-1] - q_bot/dx - Q[N-1]
    
    # Solve the linear system A * T_new = r for T_new.
    T_new = np.linalg.solve(A, r)
    
    # Update T for next time step
    T = T_new.copy()
    T_hist[n+1, :] = T

# =============================================================================
# 7) 3D Plot of Temperature vs. Position and Time
# =============================================================================
time_array = np.linspace(0, t_total, n_steps)  # Time vector [s]
X, TimeGrid = np.meshgrid(x, time_array)          # Meshgrid for plotting

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, TimeGrid, T_hist, cmap='viridis', linewidth=0, antialiased=True)
ax.set_xlabel("Position (m)")
ax.set_ylabel("Time (s)")
ax.set_zlabel("Temperature (K)")
ax.set_title("2-Layer Temperature Evolution (Semi-Implicit Scheme)")
fig.colorbar(surf, shrink=0.5, aspect=10, label="Temperature (K)")
plt.show()

# =============================================================================
# 8) Text Output of Temperature Profiles at Selected Times
# =============================================================================
mid_index = n_steps // 2
print("Spatial positions (m):")
print(x)
print("\nTemperature profile at t = 0 s (K):")
print(T_hist[0, :])
print("\nTemperature profile at t = 0.5*t_total s (K):")
print(T_hist[mid_index, :])
print("\nTemperature profile at t = t_total s (K):")
print(T_hist[-1, :])
