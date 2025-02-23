# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 18:58:07 2025

@author: elder
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Define constants and parameters
# -----------------------------------------------------------------------------
# Physical constants
sigma = 5.670374419e-8     # Stefan-Boltzmann constant [W/(m^2·K^4)]

# Areas
A_solar = 1              # [m^2] area of the panel exposed to the Sun
A_radiator = 3.5         # [m^2] area of the radiator facing deep space
                           # These can be any values you want; total area might exceed 1 m²
                           
# Optical / Thermal properties
alpha_solar = 0.92         # Absorptivity of the solar panel surface
epsilon_radiator = 0.92    # Emissivity of the radiator surface

# Orbit / Environment
solar_flux = 1361.0        # [W/m^2] approximate solar constant at 1 AU
internal_power = 100.0      # [W] internal heat dissipation from electronics

# Thermal mass (lumped)
mass = 5.0                 # [kg] total mass associated with this panel assembly
specific_heat = 900.0      # [J/(kg·K)] approximate (depends on actual materials)

# Initial condition
T0 = 290.0                 # [K] initial temperature

# Simulation time
t_total = 6000.0           # [s] total simulation time (100 minutes)
dt = 1.0                   # [s] time step
num_steps = int(t_total/dt)

# -----------------------------------------------------------------------------
# 2. Define helper function for net heat flux
# -----------------------------------------------------------------------------
def net_heat_flux(T):
    """
    Compute net heat flux into the single thermal node:
      Q_net = Q_solar_absorbed + Q_internal - Q_radiated
    """
    # Solar power absorbed by the sun-facing panel area
    Q_solar = alpha_solar * solar_flux * A_solar
    
    # Internal heat dissipation (e.g., from computer/electronics)
    Q_int = internal_power
    
    # Radiative heat loss from the radiator area
    Q_rad = epsilon_radiator * sigma * A_radiator * T**4
    
    return Q_solar + Q_int - Q_rad

# -----------------------------------------------------------------------------
# 3. Time integration (simple explicit Euler approach)
# -----------------------------------------------------------------------------
times = np.linspace(0, t_total, num_steps)
temperatures = np.zeros(num_steps)
temperatures[0] = T0

C = mass * specific_heat  # Lumped thermal capacitance [J/K]

for i in range(1, num_steps):
    T_current = temperatures[i-1]
    
    # Calculate net heat flux at the current temperature
    Q_net = net_heat_flux(T_current)
    
    # dT/dt = Q_net / (m * cp)
    dTdt = Q_net / C
    
    # Update temperature (explicit Euler)
    temperatures[i] = T_current + dTdt * dt

# -----------------------------------------------------------------------------
# 4. Plot the results
# -----------------------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(times/60, temperatures, label="Panel Temperature")
plt.xlabel("Time [min]")
plt.ylabel("Temperature [K]")
plt.title("Single-Node Model: Separate Solar and Radiator Areas")
plt.grid(True)
plt.legend()
plt.show()
