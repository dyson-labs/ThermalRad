# -*- coding: utf-8 -*-
"""
5-Node Thermal Model (with 2.7 K Background, spatial plotting)
This version computes conduction strictly using:
    Qdot = k * A * delta(T) / L
with no additional thermal resistance.
It prints the temperatures at t = 0, t = 0.5*t_total, and t = t_total.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------------------
# 1) Physical Constants & Setup
# ---------------------------------------------------------------------
sigma       = 5.670374419e-8   # Stefan-Boltzmann constant [W/(m^2 K^4)]
solar_flux  = 1361.0           # Solar flux at 1 AU [W/m^2]
T_env       = 2.7              # Background temperature [K]

# Node thicknesses [m]
thicknesses = [0.002, 0.015, 0.030, 0.015, 0.005]

# Define boundaries and node centers for spatial plotting
boundaries = np.concatenate(([0.0], np.cumsum(thicknesses)))
x_nodes    = 0.5 * (boundaries[:-1] + boundaries[1:])

# Simulation time parameters
t_total = 60000.0   # total simulation time [s]
dt      = 0.1       # time step [s]
n_steps = int(t_total / dt)

# ---------------------------------------------------------------------
# 2) Node Thermal Properties
# ---------------------------------------------------------------------
M       = np.zeros(5)  # Mass of each node [kg] (assuming 1 m^2 cross-sectional area)
rho     = np.zeros(5)  # Density [kg/m^3]
cp      = np.zeros(5)  # Specific heat capacity [J/(kg·K)]
alpha   = np.zeros(5)  # Solar absorptivity
epsilon = np.zeros(5)  # Emissivity for radiation

# Example densities [kg/m^3]
rho[0] = 2320         # Node 1: silicon-based (solar cell)
rho[1] = 1850         # Node 2: FR4 board
rho[2] = 2700 * 0.6   # Node 3: partial aluminum (heatsink)
rho[3] = 1850         # Node 4: FR4 board
rho[4] = 2700         # Node 5: full aluminum (radiator)

# Example specific heats [J/(kg·K)]
cp[0] = 800.0         # Node 1: solar cell
cp[1] = 820.0         # Node 2: FR4
cp[2] = 877.0         # Node 3: aluminum heatsink
cp[3] = 820.0         # Node 4: FR4
cp[4] = 877.0         # Node 5: aluminum radiator

# Compute masses (assuming unit area)
for i in range(5):
    M[i] = rho[i] * thicknesses[i]

# Radiative properties
alpha[0]   = 0.9   # Node 1 absorptivity (solar cell)
epsilon[0] = 0.9   # Node 1 emissivity
epsilon[4] = 0.92  # Node 5 emissivity (radiator)

# Thermal capacitances [J/K]
C = M * cp

# ---------------------------------------------------------------------
# 3) Internal Power Inputs [W]
# ---------------------------------------------------------------------
Q_int = np.zeros(5)
Q_int[1] = 50.0    # Node 2: Control board
Q_int[2] = -50.0   # Node 3: Heatsink (cooling effect)
Q_int[3] = 500.0   # Node 4: Hash board (heat generating)
# Node 5 has no internal power input

# ---------------------------------------------------------------------
# 4) Conduction Between Nodes using Thermal Conductivity
# ---------------------------------------------------------------------
# Using:
#    Qdot = k * A * (T_hot - T_cold) / L
# where L = 0.5*thickness[i] + 0.5*thickness[i+1]
A = 1.0  # Cross-sectional area [m^2]

# Effective conduction path lengths between nodes:
L_12 = 0.5 * thicknesses[0] + 0.5 * thicknesses[1]
L_23 = 0.5 * thicknesses[1] + 0.5 * thicknesses[2]
L_34 = 0.5 * thicknesses[2] + 0.5 * thicknesses[3]
L_45 = 0.5 * thicknesses[3] + 0.5 * thicknesses[4]

# Bulk thermal conductivities for each interface [W/(m·K)]
k_12 = 70.0   # between Nodes 1 and 2
k_23 = 100.0  # between Nodes 2 and 3
k_34 = 100.0  # between Nodes 3 and 4
k_45 = 70.0   # between Nodes 4 and 5

# ---------------------------------------------------------------------
# 5) Initial Conditions
# ---------------------------------------------------------------------
T0 = 290.0  # Initial temperature for all nodes [K]
T  = np.ones(5) * T0

# Allocate an array to store temperatures over time for plotting
T_hist       = np.zeros((n_steps, 5))
T_hist[0, :] = T

# Time array for plotting [hours]
time_array = np.linspace(0, t_total, n_steps) / 3600.0

# ---------------------------------------------------------------------
# 6) Main Time Loop (Explicit Euler Integration)
# ---------------------------------------------------------------------
for step_i in range(1, n_steps):
    T_old = T.copy()
    
    # Compute conduction between nodes:
    Q_12 = k_12 * A * (T_old[0] - T_old[1]) / L_12
    Q_23 = k_23 * A * (T_old[1] - T_old[2]) / L_23
    Q_34 = k_34 * A * (T_old[2] - T_old[3]) / L_34
    Q_45 = k_45 * A * (T_old[3] - T_old[4]) / L_45

    # Energy balance for each node:
    # Node 1 (Solar Cell)
    A1_sun    = 1.0  # Area exposed to the sun [m^2]
    Q_solar_1 = alpha[0] * A1_sun * solar_flux   # Absorbed solar power [W]
    Q_rad_1   = epsilon[0] * sigma * A1_sun * (T_old[0]**4 - T_env**4)  # Radiative loss [W]
    dT1_dt = (Q_solar_1 - Q_rad_1 - Q_12 + Q_int[0]) / C[0]
    T[0] = T_old[0] + dT1_dt * dt

    # Node 2 (Control Board)
    dT2_dt = (Q_12 - Q_23 + Q_int[1]) / C[1]
    T[1] = T_old[1] + dT2_dt * dt

    # Node 3 (Heatsink)
    dT3_dt = (Q_23 - Q_34 + Q_int[2]) / C[2]
    T[2] = T_old[2] + dT3_dt * dt

    # Node 4 (Hash Board)
    dT4_dt = (Q_34 - Q_45 + Q_int[3]) / C[3]
    T[3] = T_old[3] + dT4_dt * dt

    # Node 5 (Radiator)
    A5_rad  = 2.0  # Radiator area [m^2]
    Q_rad_5 = epsilon[4] * sigma * A5_rad * (T_old[4]**4 - T_env**4)  # Radiative loss [W]
    dT5_dt  = (Q_45 - Q_rad_5 + Q_int[4]) / C[4]
    T[4]    = T_old[4] + dT5_dt * dt

    T_hist[step_i, :] = T

# ---------------------------------------------------------------------
# 7) Print Temperature Outputs at Selected Times
# ---------------------------------------------------------------------
mid_index = n_steps // 2
print("Temperature at t = 0 s:           ", T_hist[0, :])
print("Temperature at t = 0.5 * t_total s: ", T_hist[mid_index, :])
print("Temperature at t = t_total s:       ", T_hist[-1, :])

# ---------------------------------------------------------------------
# 8) Plot Results (Optional)
# ---------------------------------------------------------------------
X_2D, Y_2D = np.meshgrid(x_nodes, time_array, indexing='xy')
Z_2D = T_hist

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_2D, Y_2D, Z_2D,
                       cmap='viridis', linewidth=1, antialiased=False)

ax.set_xlabel("Distance [m]")
ax.set_ylabel("Time [hr]")
ax.set_zlabel("Temperature [K]")
ax.set_title("5-Node Model: Temperature vs. Distance & Time\n(Using Pure Thermal Conductivity)")

fig.colorbar(surf, shrink=0.5, aspect=10, label="Temperature [K]")
plt.show()
