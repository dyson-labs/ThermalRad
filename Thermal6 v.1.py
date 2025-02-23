# -*- coding: utf-8 -*-
"""
Spatially Resolved 1D Thermal Conduction Model with 3D Plot,
Effective Surface Areas, and Improved k Interface Reporting

This simulation models conduction through a layered system with total thickness L_total.
Material properties (density, specific heat, thermal conductivity, and internal heating)
are assigned to each spatial cell (with an associated cell_length). The top boundary (layer 1)
receives solar flux (with radiative losses), and the bottom boundary (layer 5) radiates to the environment.
Effective surface areas A_top and A_bot can be changed.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# =============================================================================
# 1) Simulation Setup (Time and Spatial Discretization Parameters)
# =============================================================================
L_total = 0.064       # Total thickness of the system [m]
t_total = 1000.0      # Total simulation time [s] (shorter here for stability)
dt = 0.1              # Time step [s]
n_steps = int(t_total / dt)

N = 11                        # Number of spatial grid points
dx = L_total / (N - 1)        # Uniform grid spacing [m]
x = np.linspace(0, L_total, N)  # Spatial grid positions

# Effective surface areas for the top and bottom boundaries (in m²)
A_top = 1.0   # For the silicon solar cell (layer 1)
A_bot = 2.0   # For the aluminum radiator (layer 5)

# =============================================================================
# 2) Material Properties Array (Assigned to Each Spatial Cell)
# =============================================================================
# Create arrays for density (rho, kg/m³), specific heat (cp, J/(kg·K)),
# thermal conductivity (k, W/(m·K)), and volumetric internal heating (Q, W/m³).
rho = np.zeros(N)
cp = np.zeros(N)
k = np.zeros(N)
Q = np.zeros(N)
# Also, define cell_length for each grid cell (here uniform, equal to dx)
cell_length = np.full(N, dx)

# Define layer boundaries (as in the original 5-layer system):
boundaries = np.array([0.0, 0.002, 0.017, 0.047, 0.062, 0.064])
# Assign properties based on which layer a grid point falls into.
for i in range(N):
    xi = x[i]
    if xi < boundaries[1]:
        # Layer 1: Silicon solar cell
        rho[i] = 2320.0       # kg/m³
        cp[i]  = 800.0        # J/(kg·K)
        k[i]   = 150.0        # W/(m·K)
        Q[i]   = 0.0          # No internal heating
    elif xi < boundaries[2]:
        # Layer 2: FR4 control board
        rho[i] = 1850.0
        cp[i]  = 820.0
        k[i]   = 1.8          # (You may experiment by increasing this above 1)
        Q[i]   = 50.0 / (boundaries[2] - boundaries[1])
    elif xi < boundaries[3]:
        # Layer 3: Partial-aluminum heatsink
        rho[i] = 2700.0 * 0.6
        cp[i]  = 877.0
        k[i]   = 120.0
        Q[i]   = -50.0 / (boundaries[3] - boundaries[2])
    elif xi < boundaries[4]:
        # Layer 4: FR4 hash board
        rho[i] = 1850.0
        cp[i]  = 820.0
        k[i]   = 1.8          # (You may experiment by increasing this above 1)
        Q[i]   = 500.0 / (boundaries[4] - boundaries[3])
    else:
        # Layer 5: Aluminum radiator
        rho[i] = 2700.0
        cp[i]  = 877.0
        k[i]   = 205.0
        Q[i]   = 0.0

# =============================================================================
# 3) Boundary Conditions and Radiative Parameters
# =============================================================================
sigma = 5.670374419e-8   # Stefan-Boltzmann constant [W/(m²·K⁴)]
solar_flux = 1361.0      # Solar flux [W/m²]
T_env = 2.7              # Environment temperature [K]

# Surface properties at boundaries:
alpha_top = 0.9  # Solar absorptivity at top surface
eps_top   = 0.9  # Emissivity at top surface
eps_bot   = 0.92 # Emissivity at bottom surface

# =============================================================================
# 4) Initial Conditions and Storage for Time History
# =============================================================================
T = np.ones(N) * 290.0  # Initial temperature profile [K]
T_initial = T.copy()    # Store initial profile

# Array to store temperature history for plotting
T_hist = np.zeros((n_steps, N))
T_hist[0, :] = T

# =============================================================================
# 5) Helper Function: Harmonic Mean for Effective Conductivity at Interfaces
# =============================================================================
def k_interface(k1, k2):
    """Return the harmonic mean of two thermal conductivities.
       Print a warning if the computed value is negative (which should not occur
       if both inputs are positive)."""
    if (k1 + k2) == 0:
        return 0.0
    result = 2.0 * k1 * k2 / (k1 + k2)
    if result < 0:
        print("Warning: Negative effective conductivity encountered: k1 = {}, k2 = {}, result = {}".format(k1, k2, result))
    return result

# =============================================================================
# 6) Time Integration (Explicit Finite Difference Scheme)
# =============================================================================
# We update T at each grid point using an explicit Euler method.
# Interior nodes use central differences; boundaries use flux (Neumann) conditions.
for step in range(1, n_steps):
    T_new = np.copy(T)
    
    # -- Top Boundary (i = 0) --
    k_half = k_interface(k[0], k[1])
    # Net flux at top (W): solar absorption minus radiative emission, scaled by effective area A_top.
    q_top = alpha_top * A_top * solar_flux - eps_top * A_top * sigma * (T[0]**4 - T_env**4)
    # Convert the surface flux (W) to a volumetric source by dividing by cell volume (cell_length[0] * 1 m²)
    T_new[0] = T[0] + dt/(rho[0]*cp[0]) * ( (q_top - k_half*(T[1]-T[0])/dx) / cell_length[0] + Q[0] )
    
    # -- Interior Nodes (i = 1 to N-2) --
    for i in range(1, N-1):
        k_ip = k_interface(k[i], k[i+1])
        k_im = k_interface(k[i], k[i-1])
        T_new[i] = T[i] + dt/(rho[i]*cp[i]) * ( ( k_ip*(T[i+1]-T[i]) - k_im*(T[i]-T[i-1]) )/(dx*dx) + Q[i] )
    
    # -- Bottom Boundary (i = N-1) --
    k_bottom = k_interface(k[N-1], k[N-2])
    # Radiative loss from bottom surface (W), scaled by effective area A_bot.
    q_bot = eps_bot * A_bot * sigma * (T[N-1]**4 - T_env**4)
    # (Note: The flux is taken as a loss, so it enters with a minus sign.)
    T_new[N-1] = T[N-1] + dt/(rho[N-1]*cp[N-1]) * ( (-q_bot + k_bottom*(T[N-2]-T[N-1])/dx) / cell_length[N-1] + Q[N-1] )
    
    T = T_new.copy()
    T_hist[step, :] = T

# =============================================================================
# 7) 3D Plot of Temperature vs. Position and Time
# =============================================================================
time_array = np.linspace(0, t_total, n_steps)  # time vector [s]
X, T_grid = np.meshgrid(x, time_array)          # X: spatial positions, T_grid: time

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, T_grid, T_hist, cmap='viridis', linewidth=0, antialiased=True)

ax.set_xlabel("Position (m)")
ax.set_ylabel("Time (s)")
ax.set_zlabel("Temperature (K)")
ax.set_title("Temperature Evolution vs. Position and Time")
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
print("\nTemperature profile at t = 0.5 * t_total s (K):")
print(T_hist[mid_index, :])
print("\nTemperature profile at t = t_total s (K):")
print(T_hist[-1, :])
