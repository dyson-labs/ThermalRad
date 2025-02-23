
# -*- coding: utf-8 -*-
"""
5-Node Thermal Model with 2.7 K Background
Corrected conduction sign
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------------------
# 1) Physical Constants & Setup
# ---------------------------------------------------------------------
sigma       = 5.670374419e-8  # Stefan-Boltzmann constant [W/(m^2 K^4)]
solar_flux  = 1361.0          # [W/m^2] ~ solar flux at 1 AU
T_env       = 2.7             # background temperature (e.g. cosmic microwave)

# Node thicknesses & positions
thicknesses = [0.002, 0.015, 0.030, 0.015, 0.005]
x_positions = np.cumsum(thicknesses)

t_total = 60000.0    # [s]
dt      = 0.1     # [s]
n_steps = int(t_total / dt)

# ---------------------------------------------------------------------
# 2) Node Thermal Properties
# ---------------------------------------------------------------------
M       = np.zeros(5)  # nodal mass
rho     = np.zeros(5)  # nodal density
cp      = np.zeros(5)  # nodal specific heat capacity
alpha   = np.zeros(5)  # absorptivity for solar
epsilon = np.zeros(5)  # emissivity for radiating

rho_al_sheet = 2700               #[kg/m^3]
rho_al_heatsink = 2700 * (3/5)
rho_si_solar = 2320
rho_fr4 = 1850

rho[0] = 2320 
rho[1] = 1850
rho[2] = 2700 * (3/5)
rho[3] = 1850
rho[4] = 2700 

# Example masses and cp
M[0] = rho[0] *thicknesses[0]   # Node 1
M[1] = rho[1] *thicknesses[1]   # Node 2
M[2] = rho[2] *thicknesses[2]   # Node 3
M[3] = rho[3] *thicknesses[3]   # Node 4
M[4] = rho[4] *thicknesses[4]   # Node 5

cp[:] = 900.0
cp[0] = 800.0  # solar cell
cp[1] = 820
cp[2] = 900
cp[3] = 820
cp[4] = 900

# Solar absorptivity / emissivity
alpha[0]   = 0.9
epsilon[0] = 0.9
epsilon[4] = 0.92  # radiator node

# Thermal capacitances
C = M * cp

# ---------------------------------------------------------------------
# 3) Internal Power
# ---------------------------------------------------------------------
Q_int = np.zeros(5)
Q_int[1] = 50.0   # Node 2, Power/control board
Q_int[2] = -50      # Heat pumped out?  
Q_int[3] = 500.0  # Node 4, Hash board

# ---------------------------------------------------------------------
# 4) Conduction Between Nodes
# ---------------------------------------------------------------------
# We'll define conduction flows from i -> i+1:
G_12 = 500.0        # [W/K]
G_23 = 500.0
G_34 = 500.0
G_45 = 500.0

# ---------------------------------------------------------------------
# 5) Initial Temperatures
# ---------------------------------------------------------------------
T0 = 290.0
T  = np.ones(5)*T0

T_hist          = np.zeros((n_steps, 5))
T_hist[0, :]    = T
time_array      = np.linspace(0, t_total, n_steps)

# ---------------------------------------------------------------------
# 6) Main Time Loop (Explicit Euler)
# ---------------------------------------------------------------------
for step_i in range(1, n_steps):
    T_old = T.copy()

    # ------------------- Conduction Flows (One Direction) -------------------
    # Each Q_i,i+1 is from node i to i+1 (positive if T_i > T_i+1)
    Q_12 = G_12 * (T_old[0] - T_old[1])  # Node1 -> Node2
    Q_23 = G_23 * (T_old[1] - T_old[2])  # Node2 -> Node3
    Q_34 = G_34 * (T_old[2] - T_old[3])  # Node3 -> Node4
    Q_45 = G_45 * (T_old[3] - T_old[4])  # Node4 -> Node5

    # ------------------- Node 1 (Solar Cell) -------------------
    A1_sun = 1.0
    # Solar absorption
    Q_solar_1 = alpha[0]*sigma*A1_sun*(solar_flux / sigma)  # or alpha[0]*A1_sun*solar_flux
    # Radiative to 2.7K
    Q_rad_1 = epsilon[0]*sigma*A1_sun*((T_old[0])**4 - (T_env)**4)

    # Node 1 net
    dT1_dt = (
          Q_solar_1                 # absorbed solar
        - Q_rad_1                   # radiative to background
        - Q_12                      # conduction out to Node2
        + Q_int[0]                  # internal (0 here)
    ) / C[0]
    T[0] = T_old[0] + dT1_dt*dt

    # ------------------- Node 2 (Control Board) -------------------
    # Gains conduction from Node1 (Q_12), loses conduction to Node3 (Q_23)
    # Also has 50 W internally
    dT2_dt = (
          Q_12                       # conduction in from Node1
        - Q_23                       # conduction out to Node3
        + Q_int[1]
    ) / C[1]
    T[1] = T_old[1] + dT2_dt*dt

    # ------------------- Node 3 (Heat Sink) -------------------
    # Gains conduction from Node2 (Q_23), loses conduction to Node4 (Q_34)
    dT3_dt = (
          Q_23
        - Q_34
        + Q_int[2]
    ) / C[2]
    T[2] = T_old[2] + dT3_dt*dt

    # ------------------- Node 4 (Hash Board) -------------------
    # Gains conduction from Node3 (Q_34), loses conduction to Node5 (Q_45)
    # Also has 250 W internally
    dT4_dt = (
          Q_34
        - Q_45
        + Q_int[3]
    ) / C[3]
    T[3] = T_old[3] + dT4_dt*dt

    # ------------------- Node 5 (Radiator) -------------------
    # Gains conduction from Node4 (Q_45), radiates to 2.7K
    A5_rad = 2
    Q_rad_5 = epsilon[4]*sigma*A5_rad*((T_old[4])**4 - (T_env)**4)

    dT5_dt = (
          Q_45
        - Q_rad_5
        + Q_int[4]  # typically 0
    ) / C[4]
    T[4] = T_old[4] + dT5_dt*dt

    # Store
    T_hist[step_i, :] = T


# ---------------------------------------------------------------------
# 7) Plot Results
# ---------------------------------------------------------------------
# x-axis -> distance
# y-axis -> time
# z-axis -> temperature
X_2D, Y_2D = np.meshgrid(x_positions, time_array/3600, indexing='xy')
Z_2D       = T_hist

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X_2D, Y_2D, Z_2D,
                       cmap='viridis', linewidth=0, antialiased=False)

ax.set_xlabel("Distance [m]")
ax.set_ylabel("Time [hr]")
ax.set_zlabel("Temperature [K]")
ax.set_title("Temperature vs. Distance & Time (5-Node with 2.7K Background)")

fig.colorbar(surf, shrink=0.5, aspect=10, label="Temperature [K]")
plt.show()
