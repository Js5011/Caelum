import numpy as np

# Column parameters
H = float(input("Example Column Height (m): "))
n_slices = int(input("Number of slices: "))
dz = H / n_slices

# Time parameters
dt = 0.1          # time step (s)
t_final = 10.0    # total simulation time (s)
n_steps = int(t_final / dt)

# Mass transfer parameters
kLa = 0.08        # 1/s (example)
C_star = 1.2      # mol/m^3 (equilibrium CO2 concentration)

# Initial concentrations
C_CO2 = np.zeros(n_slices)  # CO2 starts at zero everywhere

C_NaOH_feed = float(input("NaOH feed concentration (mol/m^3): "))
C_NaOH = np.full(n_slices, C_NaOH_feed)

print("Initial CO2:", C_CO2)
print("Initial NaOH:", C_NaOH)

# Two-film theory function
def N_CO2(C, C_star, kLa):
    return kLa * (C_star - C)

# Time integration loop
for step in range(n_steps):
    # CO2 absorption rate in each slice
    N = N_CO2(C_CO2, C_star, kLa)

    # Update CO2 concentration (explicit Euler)
    C_CO2 = C_CO2 + N * dt

# Results
print("Final CO2 concentration profile (mol/m^3):")
print(C_CO2)
