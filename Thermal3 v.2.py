# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:17:44 2025

@author: elder
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------------------
# 1) Physical Constants & Setup
# ---------------------------------------------------------------------
sigma       = 5.670374419e-8   # Stefan-Boltzmann constant [W/(m^2 K^4)]
solar_flux  = 1361.0           # [W/m^2] approx. solar flux at 1 AU

# Node thicknesses:
thicknesses = [0.002, 0.015, 0.010, 0.015, 0.018]

# Compute cumulative positions so that Node 1 is at x_positions[0],
# Node 2 is at x_positions[1], etc.:
x_positions = np.cumsum(thicknesses)  # e.g., [0.002, 0.017, 0.027, 0.042, 0.060]

t_total     = 6000.0           # [s]  (example: 100 minutes)
dt          = 1.0              # [s]  time step
n_steps     = int(t_total / dt)

# ---------------------------------------------------------------------
# 2) Node Thermal Properties
# ---------------------------------------------------------------------
M      = np.zeros(5)
cp     = np.zeros(5)
alpha  = np.zeros(5)  # absorptivity (used for node 1 if it sees sun)
epsilon= np.zeros(5)  # emissivity (used for node 5 or others if they radiate)

# Assign masses (example values)
M[0] = 1.0   # Node 1: solar cell
M[1] = 2.0   # Node 2: control board
M[2] = 5.0   # Node 3: heat sink
M[3] = 2.0   # Node 4: hash board
M[4] = 1.0   # Node 5: radiator

# Assign specific heats (example values)
cp[0] = 800.0
cp[1] = 900.0
cp[2] = 900.0
cp[3] = 900.0
cp[4] = 900.0

# Suppose Node 1 has high absorptivity for solar
alpha[0]   = 0.9
epsilon[0] = 0.9
# For Node 5
epsilon[4] = 0.92

# Thermal capacitances
C = M * cp

# ---------------------------------------------------------------------
# 3) Internal Power
# ---------------------------------------------------------------------
Q_int = np.zeros(5)
Q_int[1] = 50.0  # Node 2
Q_int[3] = 250.0 # Node 4

# ---------------------------------------------------------------------
# 4) Conduction Between Nodes
# ---------------------------------------------------------------------
G_12 = 20.0
G_23 = 30.0
G_34 = 30.0
G_45 = 25.0

# ---------------------------------------------------------------------
# 5) Initial Temperatures
# ---------------------------------------------------------------------
T0 = 290.0  # [K]
T = np.ones(5) * T0

T_hist = np.zeros((n_steps, 5))
T_hist[0, :] = T

time_array = np.linspace(0, t_total, n_steps)

# ---------------------------------------------------------------------
# 6) Main Time Loop (Explicit Euler)
# ---------------------------------------------------------------------
for step_i in range(1, n_steps):
    T_old = T.copy()
    
    # Node 1 (Solar Cell)
    A1_sun = 1.0
    Q_solar_1 = alpha[0] * A1_sun * solar_flux
    Q_rad_1   = epsilon[0] * sigma * A1_sun * (T_old[0]**4)
    Q_12 = G_12 * (T_old[0] - T_old[1])

    dT1_dt = (Q_solar_1 - Q_rad_1 - Q_12 + Q_int[0]) / C[0]
    T[0] = T_old[0] + dT1_dt * dt

    # Node 2 (Control Board)
    Q_21 = -Q_12
    Q_23_2 = G_23 * (T_old[1] - T_old[2])
    dT2_dt = (Q_int[1] + Q_21 - Q_23_2) / C[1]
    T[1] = T_old[1] + dT2_dt * dt

    # Node 3 (Heat Sink)
    Q_32 = -Q_23_2
    Q_34_3 = G_34 * (T_old[2] - T_old[3])
    dT3_dt = (Q_32 - Q_34_3 + Q_int[2]) / C[2]
    T[2] = T_old[2] + dT3_dt * dt

    # Node 4 (Hash Board)
    Q_43 = -Q_34_3
    Q_45_4 = G_45 * (T_old[3] - T_old[4])
    dT4_dt = (Q_int[3] + Q_43 - Q_45_4) / C[3]
    T[3] = T_old[3] + dT4_dt * dt

    # Node 5 (Radiator)
    Q_54 = -Q_45_4
    A5_rad = 1.0
    Q_rad_5 = epsilon[4] * sigma * A5_rad * (T_old[4]**4)
    dT5_dt = (Q_54 - Q_rad_5 + Q_int[4]) / C[4]
    T[4] = T_old[4] + dT5_dt * dt

    T_hist[step_i, :] = T

# ---------------------------------------------------------------------
# 7) Plot Results
# ---------------------------------------------------------------------
# We want:
#   x-axis -> distance
#   y-axis -> time
#   z-axis -> temperature

# Create the 2D mesh:
#   X_2D shape: (n_steps, 5)
#   Y_2D shape: (n_steps, 5)
X_2D, Y_2D = np.meshgrid(x_positions, time_array, indexing='xy')

# T_hist is (n_steps, 5) matching shape for (time, node).
Z_2D = T_hist

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot with x=distance, y=time, z=temperature
surf = ax.plot_surface(X_2D, Y_2D, Z_2D,
                       cmap='viridis', linewidth=0, antialiased=False)

ax.set_xlabel("Distance [m]")
ax.set_ylabel("Time [s]")
ax.set_zlabel("Temperature [K]")
ax.set_title("Temperature vs. Distance & Time (5-Node)")

fig.colorbar(surf, shrink=0.5, aspect=10, label="Temperature [K]")
plt.show()
