# -*- coding: utf-8 -*-
"""
3-Layer 1D Thermal Conduction Model Using a Semi-Implicit (Backward Euler) Scheme

This model simulates conduction through a three-layer system:
  - Layer 1 (0 ≤ x < L1): Silicon solar cells, with solar flux at x = 0.
  - Layer 2 (L1 ≤ x < L1+L2): A circuit board (FR4) generating 200 W of heat.
  - Layer 3 (L1+L2 ≤ x ≤ L_total): An aluminum radiator that radiates to deep space at x = L_total.

Conduction and storage terms are treated implicitly (backward Euler), while the nonlinear
radiative boundary fluxes are evaluated at the previous time step.

Note:
  - L3 has been changed to 0.005 m and A_bot is set to 1.
  - The simulation time remains in seconds, but the 3D plot displays time in hours.
  - The printed average temperatures are converted to Celsius.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# =============================================================================
# 1) Simulation Setup (Time and Spatial Discretization)
# =============================================================================
# Define layer thicknesses (in meters)
L1 = 0.002         # Thickness of Layer 1 (silicon solar cells)
L2 = 0.003         # Thickness of Layer 2 (FR4 circuit board)
L3 = 0.003         # Thickness of Layer 3 (aluminum radiator) changed to 0.005 m
L_total = L1 + L2 + L3  # Total thickness (0.002 + 0.015 + 0.005 = 0.022 m)

# Time discretization
t_total = 6000.0    # Total simulation time [s]
dt = 0.1            # Time step [s]
n_steps = int(t_total / dt)

# Spatial discretization
N = 31                           # Number of spatial grid points
dx = L_total / (N - 1)           # Uniform grid spacing [m]
x = np.linspace(0, L_total, N)   # Spatial grid positions

# Effective surface areas (in m^2)
A_top = 1.0   # Effective area for solar absorption at the top (layer 1)
A_bot = 1.5   # Effective area for radiation at the bottom (layer 3), changed to 1

# =============================================================================
# 2) Material Properties and Internal Heating
# =============================================================================
# Create arrays for:
#  - Density (rho) in kg/m³,
#  - Specific heat (cp) in J/(kg·K),
#  - Thermal conductivity (k) in W/(m·K),
#  - Volumetric internal heating (Q) in W/m³.
rho = np.zeros(N)
cp  = np.zeros(N)
k   = np.zeros(N)
Q   = np.zeros(N)
# Assume each cell's effective thickness is dx
cell_length = np.full(N, dx)

# Layer boundaries:
#   - Layer 1: 0 <= x < L1
#   - Layer 2: L1 <= x < L1+L2
#   - Layer 3: L1+L2 <= x <= L_total
for i in range(N):
    if x[i] < L1:
        # Layer 1: Silicon solar cells
        rho[i] = 2320.0      # kg/m³
        cp[i]  = 800.0       # J/(kg·K)
        k[i]   = 150.0       # W/(m·K)
        Q[i]   = 0.0         # No internal heating
    elif x[i] < L1 + L2:
        # Layer 2: FR4 circuit board (producing 200 W)
        rho[i] = 1850.0      # kg/m³
        cp[i]  = 820.0       # J/(kg·K)
        k[i]   = 0.3         # W/(m·K)
        # Distribute 200 W over the thickness L2:
        Q[i]   = 200.0 / L2   # ~13333.33 W/m³ when L2 = 0.015 m
    else:
        # Layer 3: Aluminum radiator
        rho[i] = 2700.0      # kg/m³
        cp[i]  = 877.0       # J/(kg·K)
        k[i]   = 205.0       # W/(m·K)
        Q[i]   = 0.0         # No internal heating

# =============================================================================
# 3) Boundary Conditions and Radiative Parameters
# =============================================================================
sigma = 5.670374419e-8  # Stefan-Boltzmann constant [W/(m²·K⁴)]
solar_flux = 1361.0     # Solar flux [W/m²]
T_env = 2.7             # Deep space temperature [K]

# Surface properties at boundaries:
alpha_top = 0.9  # Solar absorptivity at top
eps_top   = 0.9  # Emissivity at top
eps_bot   = 0.92 # Emissivity at bottom

# =============================================================================
# 4) Initial Conditions and Storage for Time History
# =============================================================================
T = np.ones(N) * 290.0   # Initial temperature [K] everywhere
T_hist = np.zeros((n_steps, N))
T_hist[0, :] = T

# =============================================================================
# 5) Helper Function: Harmonic Mean for Interface Conductivity
# =============================================================================
def k_interface(k1, k2):
    """Return the harmonic mean of two thermal conductivities."""
    if (k1 + k2) == 0:
        return 0.0
    return 2.0 * k1 * k2 / (k1 + k2)

# Precompute interface conductivities between grid cells:
k_half = np.zeros(N-1)
for i in range(N-1):
    k_half[i] = k_interface(k[i], k[i+1])

# =============================================================================
# 6) Time Integration Using a Semi-Implicit (Backward Euler) Method
# =============================================================================
# At each time step we assemble and solve a tridiagonal system.
# The radiative fluxes at the boundaries are treated explicitly (using previous T).
for n in range(0, n_steps-1):
    # Assemble coefficient matrix A and right-hand side vector r.
    A = np.zeros((N, N))
    r = np.zeros(N)
    
    # --- Top Boundary (i = 0) ---
    q_top = alpha_top * A_top * solar_flux - eps_top * A_top * sigma * (T[0]**4 - T_env**4)
    A[0, 0] = rho[0]*cp[0]/dt + k_half[0]/(dx**2)
    A[0, 1] = - k_half[0]/(dx**2)
    r[0] = (rho[0]*cp[0]/dt)*T[0] + q_top/dx - Q[0]
    
    # --- Interior Nodes (i = 1 to N-2) ---
    for i in range(1, N-1):
        A[i, i-1] = - k_half[i-1] / (dx**2)
        A[i, i]   = rho[i]*cp[i]/dt + (k_half[i-1] + k_half[i])/(dx**2)
        A[i, i+1] = - k_half[i] / (dx**2)
        r[i] = (rho[i]*cp[i]/dt)*T[i] + Q[i]
    
    # --- Bottom Boundary (i = N-1) ---
    q_bot = eps_bot * A_bot * sigma * (T[N-1]**4 - T_env**4)
    A[N-1, N-2] = - k_half[N-2] / (dx**2)
    A[N-1, N-1] = rho[N-1]*cp[N-1]/dt + k_half[N-2]/(dx**2)
    r[N-1] = (rho[N-1]*cp[N-1]/dt)*T[N-1] - q_bot/dx - Q[N-1]
    
    # Solve the linear system A * T_new = r
    T_new = np.linalg.solve(A, r)
    
    # Update T and store the new temperature profile
    T = T_new.copy()
    T_hist[n+1, :] = T

# =============================================================================
# 7) 3D Plot of Temperature vs. Position and Time
# =============================================================================
# Create a time array (still in seconds) and convert it to hours for plotting.
time_array = np.linspace(0, t_total, n_steps)
time_array_hours = time_array / 3600.0

X, TimeGrid = np.meshgrid(x, time_array_hours)  # Use time in hours for the y-axis

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, TimeGrid, T_hist, cmap='viridis', linewidth=0, antialiased=True)
ax.set_xlabel("Position (m)")
ax.set_ylabel("Time (hours)")
ax.set_zlabel("Temperature (K)")
ax.set_title("3-Layer Temperature Evolution (Semi-Implicit Scheme)")
fig.colorbar(surf, shrink=0.5, aspect=10, label="Temperature (K)")
plt.show()

# =============================================================================
# 8) Print Average Temperature of Each Layer at Selected Times (in Celsius)
# =============================================================================
# Define layer masks:
layer1_mask = x < L1
layer2_mask = (x >= L1) & (x < L1 + L2)
layer3_mask = x >= (L1 + L2)

# Indices for selected times:
mid_index = n_steps // 2
times_to_print = {"t = 0 s": 0, 
                  "t = 0.5 * t_total": mid_index, 
                  "t = t_total": n_steps - 1}

print("Average Temperatures (in °C) for Each Layer:")
for label, idx in times_to_print.items():
    T_current = T_hist[idx, :]
    avg_layer1 = np.mean(T_current[layer1_mask])
    avg_layer2 = np.mean(T_current[layer2_mask])
    avg_layer3 = np.mean(T_current[layer3_mask])
    
    # Convert from Kelvin to Celsius
    avg_layer1_C = avg_layer1 - 273.15
    avg_layer2_C = avg_layer2 - 273.15
    avg_layer3_C = avg_layer3 - 273.15
    
    print(f"\nAt {label}:")
    print(f"  Layer 1 (Silicon solar cells): {avg_layer1_C:.2f} °C")
    print(f"  Layer 2 (FR4 circuit board):   {avg_layer2_C:.2f} °C")
    print(f"  Layer 3 (Aluminum radiator):     {avg_layer3_C:.2f} °C")
