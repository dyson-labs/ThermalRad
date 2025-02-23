# -*- coding: utf-8 -*-
"""
5-Node Thermal Model (with 2.7 K Background, clearer spatial plotting, and interface thermal resistance)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------------------
# 1) Physical Constants & Setup
# ---------------------------------------------------------------------
sigma       = 5.670374419e-8   # Stefan-Boltzmann constant [W/(m^2 K^4)]
solar_flux  = 1361.0           # [W/m^2] ~ solar flux at 1 AU
T_env       = 2.7              # background temperature (e.g. cosmic microwave)

# Node thicknesses
thicknesses = [0.002, 0.015, 0.030, 0.015, 0.005]

# For better spatial distinction, define boundary array and node centers:
boundaries = np.concatenate(([0.0], np.cumsum(thicknesses)))  # e.g. [0, 0.002, 0.017, 0.047, 0.062, 0.067]
x_nodes    = 0.5*(boundaries[:-1] + boundaries[1:])           # midpoints, shape (5,)

# Simulation time
t_total = 60000.0   # [s]
dt      = 0.1       # [s]
n_steps = int(t_total / dt)

# ---------------------------------------------------------------------
# 2) Node Thermal Properties
# ---------------------------------------------------------------------
M       = np.zeros(5)  # nodal mass
rho     = np.zeros(5)  # nodal density
cp      = np.zeros(5)  # nodal specific heat capacity
alpha   = np.zeros(5)  # absorptivity for solar
epsilon = np.zeros(5)  # emissivity for radiating

# Example densities
rho[0] = 2320    # silicon-based (solar cell)
rho[1] = 1850    # FR4 board
rho[2] = 2700*0.6  # partial aluminum for a heatsink
rho[3] = 1850    # FR4 again
rho[4] = 2700    # full aluminum radiator

# Example specific heats
cp[0] = 800.0    # solar cell
cp[1] = 820.0    # FR4
cp[2] = 877.0    # al heat sink
cp[3] = 820.0    # FR4
cp[4] = 877.0    # al radiator

# Calculate masses (density * thickness for each node),
# ignoring cross-sectional area for simplicity or assume it's 1 m^2
for i in range(5):
    # If each node is 1 m^2 cross-section:
    M[i] = rho[i] * thicknesses[i]

# Radiative properties
alpha[0]   = 0.9   # absorptivity for Node 1 (solar cell)
epsilon[0] = 0.9   # if Node 1 radiates
epsilon[4] = 0.92  # radiator node

# Thermal capacitances
C = M * cp

# ---------------------------------------------------------------------
# 3) Internal Power
# ---------------------------------------------------------------------
Q_int = np.zeros(5)
Q_int[1] = 50.0    # Node 2: Control board
Q_int[2] = -50.0   # Node 3: maybe pumping out heat?
Q_int[3] = 500.0   # Node 4: Hash board (very hot)
# Q_int[4] = ...   # Usually 0 or small background

# ---------------------------------------------------------------------
# 4) Conduction Between Nodes
# ---------------------------------------------------------------------
# Base conduction values (bulk conduction)
G_12_base = 70.0   # [W/K]
G_23_base = 100.0
G_34_base = 100.0
G_45_base = 70.0

# Interface thermal resistances [K/W]
# (These can represent contact resistances or interface materials)
R_int12 = 0.02
R_int23 = 0.01
R_int34 = 0.05
R_int45 = 0.01

# Compute effective conduction G_total = 1 / (1/G_base + R_int)
# This lumps the bulk conduction in series with interface R_int.
# e.g. R_total = (1 / G_base) + R_int, so G_total = 1 / R_total.
G_12 = 1.0 / ((1.0 / G_12_base) + R_int12)
G_23 = 1.0 / ((1.0 / G_23_base) + R_int23)
G_34 = 1.0 / ((1.0 / G_34_base) + R_int34)
G_45 = 1.0 / ((1.0 / G_45_base) + R_int45)

# ---------------------------------------------------------------------
# 5) Initial Conditions
# ---------------------------------------------------------------------
T0 = 290.0
T  = np.ones(5)*T0

T_hist       = np.zeros((n_steps, 5))
T_hist[0, :] = T

# We'll store time in hours for the plot
time_array = np.linspace(0, t_total, n_steps)/3600.0  # in hours

# ---------------------------------------------------------------------
# 6) Main Time Loop (Explicit Euler)
# ---------------------------------------------------------------------
for step_i in range(1, n_steps):
    T_old = T.copy()

    # -- Conduction flows (i->i+1) --
    Q_12 = G_12*(T_old[0] - T_old[1])
    Q_23 = G_23*(T_old[1] - T_old[2])
    Q_34 = G_34*(T_old[2] - T_old[3])
    Q_45 = G_45*(T_old[3] - T_old[4])

    # Node 1 (Solar Cell)
    A1_sun   = 1.0  # assume 1 m^2
    Q_solar_1= alpha[0]*A1_sun*solar_flux
    Q_rad_1  = epsilon[0]*sigma*A1_sun*(T_old[0]**4 - T_env**4)
    dT1_dt = (
        Q_solar_1    # absorbed solar
      - Q_rad_1       # radiative to background
      - Q_12          # conduction out
      + Q_int[0]
    ) / C[0]
    T[0] = T_old[0] + dT1_dt*dt

    # Node 2 (Control Board)
    dT2_dt = (
        Q_12    # conduction in from Node1
      - Q_23    # conduction out to Node3
      + Q_int[1]
    ) / C[1]
    T[1] = T_old[1] + dT2_dt*dt

    # Node 3 (Heatsink)
    dT3_dt = (
        Q_23
      - Q_34
      + Q_int[2]
    ) / C[2]
    T[2] = T_old[2] + dT3_dt*dt

    # Node 4 (Hash Board)
    dT4_dt = (
        Q_34
      - Q_45
      + Q_int[3]
    ) / C[3]
    T[3] = T_old[3] + dT4_dt*dt

    # Node 5 (Radiator)
    A5_rad  = 2.0  # 2 m^2
    Q_rad_5 = epsilon[4]*sigma*A5_rad*(T_old[4]**4 - T_env**4)
    dT5_dt  = (
        Q_45
      - Q_rad_5
      + Q_int[4]
    ) / C[4]
    T[4]    = T_old[4] + dT5_dt*dt

    T_hist[step_i, :] = T

# ---------------------------------------------------------------------
# 7) Plot Results
# ---------------------------------------------------------------------
# Distance => node midpoints in x_nodes
# Time     => time_array (in hours)
# Temp     => T_hist

X_2D, Y_2D = np.meshgrid(x_nodes, time_array, indexing='xy')
Z_2D       = T_hist  # shape (n_steps, 5)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X_2D, Y_2D, Z_2D,
                       cmap='viridis', linewidth=1, antialiased=False)

ax.set_xlabel("Distance [m]")
ax.set_ylabel("Time [hr]")
ax.set_zlabel("Temperature [K]")
ax.set_title("5-Node Model: Temperature vs. Distance & Time (with Interface Resistance)")

fig.colorbar(surf, shrink=0.5, aspect=10, label="Temperature [K]")
plt.show()
