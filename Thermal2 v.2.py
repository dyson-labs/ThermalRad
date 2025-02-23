# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 17:12:46 2025

@author: elder
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. Physical Constants and Geometry
# -------------------------------------------------------------------------
sigma = 5.670374419e-8  # Stefan-Boltzmann [W/(m^2*K^4)]
solar_flux = 1361.0     # Approx. solar flux at 1 AU [W/m^2]

# Lengths for each region (meters)
L1 = 0.002   # Solar cell section
L2 = 0.030   # Middle section
L3 = 0.018   # Radiator section
L_total = L1 + L2 + L3

# Cross-sectional area (assume 1 m^2 for simplicity)
A = 1.0  

# Number of discrete cells
Nx = 51
dx = L_total / (Nx - 1)  # spacing along the rod

# -------------------------------------------------------------------------
# 2. Piecewise Density and Specific Heat
# -------------------------------------------------------------------------
cp1 = 800.0  # [J/(kg*K)] for solar cell region
cp2 = 900.0  # [J/(kg*K)] for middle region
cp3 = 900.0  # [J/(kg*K)] for radiator region

# Example densities (just to match some masses)
rho1 = 2 / (A * L1)
rho2 =  4 / (A * L2)
rho3 = 6 / (A * L3)

rho_array = np.zeros(Nx)
cp_array  = np.zeros(Nx)

x_values = np.linspace(0, L_total, Nx)
for i, x in enumerate(x_values):
    if x <= L1:
        rho_array[i] = rho1
        cp_array[i]  = cp1
    elif x <= (L1 + L2):
        rho_array[i] = rho2
        cp_array[i]  = cp2
    else:
        rho_array[i] = rho3
        cp_array[i]  = cp3

# -------------------------------------------------------------------------
# 3. Piecewise Thermal Conductivity
# -------------------------------------------------------------------------
k1 = 180.0  # solar cell region [W/(m*K)]
k2 = 200.0  # middle region
k3 = 150.0  # radiator region

k_array = np.zeros(Nx)
for i, x in enumerate(x_values):
    if x <= L1:
        k_array[i] = k1
    elif x <= (L1 + L2):
        k_array[i] = k2
    else:
        k_array[i] = k3

# -------------------------------------------------------------------------
# 4. Other Thermal Properties and Setup
# -------------------------------------------------------------------------
epsilon_radiator = 0.92      # Radiator emissivity
alpha_solar      = 0.92      # Solar absorptivity
epsilon_solar    = 0.92      # Solar emissivity
internal_power_total = 300.0 # [W]

# Distribute internal power uniformly in the middle segment (region 2).
idx_region2 = np.where((x_values > L1) & (x_values <= L1 + L2))[0]
power_per_cell = internal_power_total / len(idx_region2)

# Time stepping
t_total = 6000.0  # [s] (~100 minutes)
dt = 0.01         # [s]
n_steps = int(t_total // dt)

# -------------------------------------------------------------------------
# 5. Initialize Temperature
# -------------------------------------------------------------------------
T = np.ones(Nx) * 290.0  # 290 K everywhere initially

# 5a. Allocate a 2D array to store T at all steps: T_store[step, i]
T_store = np.zeros((n_steps, Nx))
T_store[0, :] = T

# Identify some "node-like" indices for a simple line plot
i1 = np.argmin(np.abs(x_values - (0.5*L1)))      
i2 = np.argmin(np.abs(x_values - (L1 + 0.5*L2)))
i3 = Nx - 1                                     

T1_history = np.zeros(n_steps)
T2_history = np.zeros(n_steps)
T3_history = np.zeros(n_steps)

T1_history[0] = T[i1]
T2_history[0] = T[i2]
T3_history[0] = T[i3]

# -------------------------------------------------------------------------
# 6. Main Time Loop
# -------------------------------------------------------------------------
for step in range(1, n_steps):
    T_old = T.copy()
    
    # --- Left boundary (i=0) ---
    Q_solar = alpha_solar * solar_flux  # [W]
    vol_0   = dx * A
    mCp_0   = rho_array[0] * cp_array[0] * vol_0
    
    d2T_dx2_0 = (T_old[1] - T_old[0]) * 2.0 / (dx*dx)
    conduction_0 = k_array[0] * d2T_dx2_0
    
    Q_rad_left = epsilon_solar * sigma * A * T_old[0]**4
    T[0] = T_old[0] + dt/mCp_0 * (conduction_0 + Q_solar - Q_rad_left)

    
    # --- Interior cells (i=1..Nx-2) ---
    for i in range(1, Nx-1):
        vol_i = dx * A
        mCp_i = rho_array[i] * cp_array[i] * vol_i
        
        d2T_dx2 = (T_old[i+1] - 2*T_old[i] + T_old[i-1]) / (dx*dx)
        conduction = k_array[i] * d2T_dx2
        
        if i in idx_region2:
            Q_int = power_per_cell
        else:
            Q_int = 0.0
        
        T[i] = T_old[i] + dt/mCp_i * (conduction + Q_int)
    
    # --- Right boundary (i=Nx-1) ---
    iR = Nx - 1
    vol_R = dx * A
    mCp_R = rho_array[iR] * cp_array[iR] * vol_R
    
    d2T_dx2_R = (T_old[iR-1] - T_old[iR]) * 2.0 / (dx*dx)
    conduction_R = k_array[iR] * d2T_dx2_R
    
    T4 = T_old[iR]**4
    Q_rad = epsilon_radiator * sigma * A * T4
    
    T[iR] = T_old[iR] + dt/mCp_R * (conduction_R - Q_rad)
    
    # Store T in T_store
    T_store[step, :] = T
    
    # Save "node-like" temps for line plot
    T1_history[step] = T[i1]
    T2_history[step] = T[i2]
    T3_history[step] = T[iR]

# -------------------------------------------------------------------------
# 7a. Plot Time Histories (Line Plots)
# -------------------------------------------------------------------------
time_array_minutes = np.linspace(0, t_total, n_steps)/60.0

plt.figure(figsize=(8,5))
plt.plot(time_array_minutes, T1_history, label="Region 1 (Solar Cell)")
plt.plot(time_array_minutes, T2_history, label="Region 2 (Middle)")
plt.plot(time_array_minutes, T3_history, label="Region 3 (Radiator)")
plt.xlabel("Time [min]")
plt.ylabel("Temperature [K]")
plt.title("1D Thermal Model: Node-Like Points vs. Time")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------------------------
# 7b. Create a 3D Surface Plot of T(x, t)
# -------------------------------------------------------------------------
# We'll use time in seconds for the 3D plot.
time_array_sec = np.linspace(0, t_total, n_steps)

# Make a meshgrid so that X_grid[i, j] = x_values[j], T_grid[i, j] = time_array_sec[i].
T_grid, X_grid = np.meshgrid(time_array_sec, x_values, indexing='xy')

# But note: T_store has shape (n_steps, Nx). We want Z = T_store^T if we want
# Z[i, j] to correspond to X_grid[i, j] and T_grid[i, j].
Z = T_store.T  # shape is (Nx, n_steps)

# Now make the 3D plot
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X_grid, T_grid, Z, cmap='hot', linewidth=0, antialiased=False)

ax.set_xlabel("Position, x [m]")
ax.set_ylabel("Time [s]")
ax.set_zlabel("Temperature [K]")
ax.set_title("Temperature Distribution: 3D Surface")

fig.colorbar(surf, shrink=0.5, aspect=10, label="Temperature [K]")
plt.show()

