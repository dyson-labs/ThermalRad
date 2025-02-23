# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 18:42:49 2025

@author: elder
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Define constants and parameters
# -----------------------------------------------------------------------------
# Physical constants
sigma = 5.670374419e-8     # Stefan-Boltzmann constant [W/(m^2·K^4)]

# Panel properties
area = 1.0                 # [m^2] area of the panel
alpha = 0.92               # Absorptivity (fraction of solar irradiance absorbed)
epsilon = 0.92             # Emissivity (fraction of blackbody emission)

# Orbit / Environment
solar_flux = 1361.0        # [W/m^2] approximate solar constant at 1 AU
internal_power = 10.0      # [W] internal heat dissipation from electronics

# Thermal mass (lumped)
mass = 5.0                 # [kg] mass of the 1 m^2 section
specific_heat = 900.0      # [J/(kg·K)] approximate (aluminum honeycomb or composite)
                           # This can vary widely based on real materials
# Initial condition
T0 = 290.0                 # [K] initial temperature guess

# Simulation time
t_total = 6000.0           # [s] total simulation time (e.g., ~100 minutes)
dt = 1.0                   # [s] time step
num_steps = int(t_total/dt)

# -----------------------------------------------------------------------------
# 2. Define helper function for net heat flux
# -----------------------------------------------------------------------------
def net_heat_flux(T):
    """
    Compute net heat flux into the panel node:
    Q_net = Q_solar_absorbed + Q_internal - Q_radiated
    """
    # Solar power absorbed by the solar-cell side
    Q_solar = alpha * solar_flux * area
    
    # Internal heat (e.g., computer) dissipated into the panel
    Q_int = internal_power
    
    # Radiative heat loss on the radiator side
    Q_rad = epsilon * sigma * area * T**4
    
    return Q_solar + Q_int - Q_rad

# -----------------------------------------------------------------------------
# 3. Time integration (simple explicit Euler approach)
# -----------------------------------------------------------------------------
times = np.linspace(0, t_total, num_steps)
temperatures = np.zeros(num_steps)
temperatures[0] = T0

for i in range(1, num_steps):
    T_current = temperatures[i-1]
    
    # Calculate net heat flux at current temperature
    Q_net = net_heat_flux(T_current)
    
    # Lumped thermal capacitance
    C = mass * specific_heat  # [J/K]
    
    # dT/dt = Q_net / (m * cp)
    dTdt = Q_net / C
    
    # Explicit Euler update
    temperatures[i] = T_current + dTdt * dt

# -----------------------------------------------------------------------------
# 4. Plot the results
# -----------------------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(times/60, temperatures, label="Panel Temperature")
plt.xlabel("Time [min]")
plt.ylabel("Temperature [K]")
plt.title("Simple 1-Node Satellite Panel Thermal Model")
plt.grid(True)
plt.legend()
plt.show()
