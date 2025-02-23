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
# We'll store node data in arrays for convenience.
# For each node i, define:
#   M[i] = mass [kg]
#   cp[i] = specific heat [J/(kg*K)]
#   alpha[i], epsilon[i], area[i], etc. if needed

# Example: 
# - Node 1 (solar cell): small mass, picks up solar flux, can also radiate if you choose
# - Node 2 (control board): 50 W
# - Node 3 (heat sink): bigger mass to soak up heat
# - Node 4 (hash board): 250 W
# - Node 5 (radiator): radiates to space

M      = np.zeros(5)
cp     = np.zeros(5)
alpha  = np.zeros(5)  # absorptivity (used for node 1 if it sees sun)
epsilon= np.zeros(5)  # emissivity (used for node 5 or others if they radiate)

# Assign masses (guesses for illustration)
M[0] = 1.0   # Node 1: solar cell (kg)
M[1] = 2.0   # Node 2: control board
M[2] = 5.0   # Node 3: heat sink
M[3] = 2.0   # Node 4: hash board
M[4] = 1.0   # Node 5: radiator

# Assign specific heats (just examples)
cp[0] = 800.0
cp[1] = 900.0
cp[2] = 900.0
cp[3] = 900.0
cp[4] = 900.0

# Suppose Node 1 has high absorptivity for solar
alpha[0]   = 0.9    # Node 1 solar cell
epsilon[0] = 0.9    # If Node 1 radiates, can set this
# For other nodes, set emissivity if you want them to radiate
epsilon[4] = 0.92   # Node 5 (radiator)

# Thermal capacitances
C = M * cp  # [J/K] for each node

# ---------------------------------------------------------------------
# 3) Internal Power
# ---------------------------------------------------------------------
Q_int = np.zeros(5)
Q_int[1] = 50.0    # Node 2 (control board)
Q_int[3] = 250.0   # Node 4 (hash board)

# ---------------------------------------------------------------------
# 4) Conduction Between Nodes
# ---------------------------------------------------------------------
# We define G_12, G_23, G_34, G_45 as conduction [W/K].
# For example:
G_12 = 20.0
G_23 = 30.0
G_34 = 30.0
G_45 = 25.0

# ---------------------------------------------------------------------
# 5) Initial Temperatures
# ---------------------------------------------------------------------
T0 = 290.0  # [K]
T = np.ones(5) * T0

# For storing results over time
T_hist = np.zeros((n_steps, 5))
T_hist[0,:] = T

# For plotting time
time_array = np.linspace(0, t_total, n_steps)

# ---------------------------------------------------------------------
# 6) Main Time Loop (Explicit Euler)
# ---------------------------------------------------------------------
for step_i in range(1, n_steps):
    T_old = T.copy()
    
    # -- Node 1 (Solar Cell) --
    # Absorbs solar, possibly radiates to space, conduction to Node 2
    # Let's define area for Node 1 if it sees the sun:
    A1_sun = 1.0  # [m^2], example
    Q_solar_1 = alpha[0] * A1_sun * solar_flux
    
    # Radiate to space? 
    # For a simple approach: Q_rad_1 = epsilon[0] * sigma * A1_sun * T_old[0]^4
    # We'll assume Node 1 does radiate if you want. Or set epsilon[0]=0 to skip it.
    Q_rad_1 = epsilon[0] * sigma * A1_sun * (T_old[0]**4)
    
    # Conduction to Node 2
    Q_12 = G_12 * (T_old[0] - T_old[1])  # positive if Node 0 > Node 1
    
    # Net flow for Node 1
    dT1_dt = (Q_solar_1 - Q_rad_1 - Q_12 + Q_int[0]) / C[0]
    T[0] = T_old[0] + dT1_dt * dt
    
    # -- Node 2 (Control Board) --
    # Gains internal 50 W, conduction from Node 1, conduction to Node 3
    Q_21 = -Q_12  # from Node 1’s perspective
    Q_23_2 = G_23 * (T_old[1] - T_old[2])
    
    dT2_dt = (Q_int[1] + Q_21 - Q_23_2) / C[1]
    T[1] = T_old[1] + dT2_dt * dt
    
    # -- Node 3 (Heat Sink) --
    # conduction from Node 2, conduction to Node 4
    Q_32 = -Q_23_2
    Q_34_3 = G_34 * (T_old[2] - T_old[3])
    
    dT3_dt = (Q_32 - Q_34_3 + Q_int[2]) / C[2]
    T[2] = T_old[2] + dT3_dt * dt
    
    # -- Node 4 (Hash Board) --
    # Gains internal 250 W, conduction from Node 3, conduction to Node 5
    Q_43 = -Q_34_3
    Q_45_4 = G_45 * (T_old[3] - T_old[4])
    
    dT4_dt = (Q_int[3] + Q_43 - Q_45_4) / C[3]
    T[3] = T_old[3] + dT4_dt * dt
    
    # -- Node 5 (Radiator) --
    # conduction from Node 4, plus radiative losses to space
    Q_54 = -Q_45_4
    # Radiate to space:
    # Suppose area for Node 5 is 1.0 m^2
    A5_rad = 1.0
    Q_rad_5 = epsilon[4] * sigma * A5_rad * (T_old[4]**4)
    
    dT5_dt = (Q_54 - Q_rad_5 + Q_int[4]) / C[4]
    T[4] = T_old[4] + dT5_dt * dt
    
    # Save results
    T_hist[step_i,:] = T

# ---------------------------------------------------------------------
# 7) Plot Results
# ---------------------------------------------------------------------
# 2D grids for X (distance) and Z (time)
Z_2D, X_2D = np.meshgrid(time_array, x_positions, indexing='xy')
# Now X_2D.shape => (5, n_steps), Z_2D.shape => (5, n_steps)

# But T_hist is shape (n_steps, 5).
# We want a 2D array that matches (5, n_steps). Let's transpose T_hist:
Y_2D = T_hist.T  # shape => (5, n_steps)


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X_2D, Y_2D, Z_2D,
                       cmap='viridis', linewidth=0, antialiased=False)

ax.set_xlabel("Distance [m]")
ax.set_ylabel("Temperature [K]")
ax.set_zlabel("Time [s]")
ax.set_title("Temperature vs. Distance & Time (5-Node)")

fig.colorbar(surf, shrink=0.5, aspect=10, label="Temperature [K]")
plt.show()


