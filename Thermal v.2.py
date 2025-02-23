# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:13:37 2025

As conductance (G) increases, temperatures converge towards 335 K, slightly too warm

@author: elder
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Define Constants and Parameters
# -----------------------------------------------------------------------------
# Stefan-Boltzmann constant
sigma = 5.670374419e-8  # [W/(m^2*K^4)]

# Solar flux at 1 AU
solar_flux = 1361.0     # [W/m^2]

# Time-stepping parameters
t_total = 6000.0        # [s] total time for simulation (~100 minutes)
dt = 1.0                # [s] time step
n_steps = int(t_total / dt)

# -----------------------------------------------------------------------------
# 2. Node Properties
# -----------------------------------------------------------------------------
# -- Node 1: Solar Cells --
m1 = 1.0                # [kg] mass of the solar cell assembly
cp1 = 800.0             # [J/(kg*K)] specific heat
C1 = m1 * cp1           # [J/K] thermal capacitance

alpha1 = 0.92           # absorptivity of solar cell surface
epsilon1 = alpha1       # absorptivity = emissivity if gray
A1_sun = 1            # [m^2] area receiving sunlight
# If Node 1 also radiates to space, you can define an emissivity and area:
# e1 = 0.80
# A1_rad = 0.8

# -- Node 2: Computer/Heat-Pipe Middle --
m2 = 3.0                # [kg] mass of middle section
cp2 = 900.0             # [J/(kg*K)]
C2 = m2 * cp2

internal_power = 300.0   # [W] constant heat dissipation

# -- Node 3: Radiator --
m3 = 1.0                # [kg] mass of radiator
cp3 = 900.0             # [J/(kg*K)]
C3 = m3 * cp3
epsilon3 = 0.92         # emissivity of radiator
A3_rad = 1            # [m^2] area radiating to space

# -----------------------------------------------------------------------------
# 3. Thermal Conduction (or Heat Pipe Conductance) Between Nodes
# -----------------------------------------------------------------------------

# G_12: conductance (W/K) between Node 1 and Node 2
# G_23: conductance (W/K) between Node 2 and Node 3
# Need way to approximate this, G = k (thermal conductivity) * A (cross sectional area) / L (length along conduction path)
#k_al_nh3 = 1e4 # 1e3 to 1e5?? 
#k_cu_h2o = 1e5 # 1e4 to 1e6??
#A_12 = 1   # Area of the interface from node 1 to node 2[m]
#A_23 = 1   # Area of the interface from node 1 to node 2[m]
#L_12 = 0.0254 # Thickness of the interface from node 1 to node 2[m]
#L_23 = 0.003 # Thickness of the interface from node 1 to node 2[m]
#G_12 = k_al_nh3 * A_12 / L_12  # [W/K]
#G_23 = k_cu_h2o * A_23 / L_23  # [W/K]

G_12 = 100 # [W/K]
G_23 = 100 # [W/K]


# -----------------------------------------------------------------------------
# 4. Set Initial Temperatures
# -----------------------------------------------------------------------------
T1_0 = 290.0            # [K]
T2_0 = 290.0            # [K]
T3_0 = 290.0            # [K]

# Arrays to store temperatures over time
T1 = np.zeros(n_steps)
T2 = np.zeros(n_steps)
T3 = np.zeros(n_steps)

T1[0] = T1_0
T2[0] = T2_0
T3[0] = T3_0

# -----------------------------------------------------------------------------
# 5. Time Integration (Explicit Euler)
# -----------------------------------------------------------------------------
for i in range(1, n_steps):
    # Current values
    t1 = T1[i-1]
    t2 = T2[i-1]
    t3 = T3[i-1]
    
    # -------------------------------------------------------------------------
    # 5a. Calculate Heat Flows
    # -------------------------------------------------------------------------
    # Solar absorbed by Node 1
    Q_solar_1 = alpha1 * A1_sun * solar_flux  # [W]
    Q_rad_1 = epsilon1 * sigma * A1_sun * (t1**4) 
    
    # Conduction (or heat pipe) between Node 1 and Node 2
    # Positive Q_12 means heat flows from Node 1 to Node 2
    Q_12 = G_12 * (t1 - t2)  
    
    # Conduction between Node 2 and Node 3
    Q_23 = G_23 * (t2 - t3)
    
    # Radiative cooling from Node 3 to deep space
    Q_rad_3 = epsilon3 * sigma * A3_rad * (t3**4)
    
    # Internal heat in Node 2
    Q_int_2 = internal_power
    
    # -------------------------------------------------------------------------
    # 5b. Write the Energy Balance for Each Node
    # -------------------------------------------------------------------------
    # Node 1 (solar cells) energy balance:
    #   dT1/dt = (Q_solar_1 - Q_12 +/- any other terms) / C1
    # (If Node 1 also radiates to space, subtract Q_rad_1, etc.)
    dT1_dt = (Q_solar_1 - Q_rad_1 - Q_12) / C1
    
    # Node 2 (computer/middle) energy balance:
    #   dT2/dt = (Q_int_2 + Q_12 - Q_23) / C2
    dT2_dt = (Q_int_2 + Q_12 - Q_23) / C2
    
    # Node 3 (radiator) energy balance:
    #   dT3/dt = (Q_23 - Q_rad_3) / C3
    dT3_dt = (Q_23 - Q_rad_3) / C3
    
    # -------------------------------------------------------------------------
    # 5c. Update Temperatures
    # -------------------------------------------------------------------------
    T1[i] = t1 + dT1_dt * dt
    T2[i] = t2 + dT2_dt * dt
    T3[i] = t3 + dT3_dt * dt

# -----------------------------------------------------------------------------
# 6. Plot Results
# -----------------------------------------------------------------------------
time_array = np.linspace(0, t_total, n_steps) / 60.0  # convert seconds to minutes

plt.figure(figsize=(8,5))
plt.plot(time_array, T1, label="Node 1 (Solar Cells)")
plt.plot(time_array, T2, label="Node 2 (Computer/Middle)")
plt.plot(time_array, T3, label="Node 3 (Radiator)")
plt.xlabel("Time [min]")
plt.ylabel("Temperature [K]")
plt.title("3-Node Thermal Model (Solar -> Middle -> Radiator)")
plt.grid(True)
plt.legend()
plt.show()
