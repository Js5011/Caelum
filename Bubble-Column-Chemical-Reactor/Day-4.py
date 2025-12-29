import numpy as np

# Column parameters
H = float(input("Column height (m): "))
n_slices = int(input("Number of slices: "))
dz = H / n_slices

# Time parameters
dt = 0.01          # small timestep (s)
t_final = 5.0
n_steps = int(t_final / dt)

# Mass transfer parameters
kLa = 0.08         # 1/s
C_star = 1.2       # mol/m^3, equilibrium CO2 concentration

# Reaction parameters
k_rxn = 0.5        # m^3/(mol·s)  (example value)

# Initial concentrations
C_CO2 = np.zeros(n_slices)  # dissolved CO2

C_NaOH_feed = float(input("NaOH feed concentration (mol/m^3): "))
C_NaOH = np.full(n_slices, C_NaOH_feed)

# Track initial NaOH for carbonate calculation
C_NaOH_initial = C_NaOH.copy()

# Initialize Na2CO3 concentration
C_Na2CO3 = np.zeros(n_slices)

print("Initial CO2:", C_CO2)
print("Initial NaOH:", C_NaOH)

# Functions
def N_CO2(C, C_star, kLa):
    """Mass transfer flux of CO2 into liquid"""
    return kLa * (C_star - C)

def reaction_rate(C_CO2, C_NaOH, k):
    """Reaction rate of CO2 with NaOH"""
    return k * C_CO2 * C_NaOH

# Time loop
for step in range(n_steps):
    # Mass transfer into liquid
    N = N_CO2(C_CO2, C_star, kLa)

    # Reaction rate in each slice
    r = reaction_rate(C_CO2, C_NaOH, k_rxn)

    # Update concentrations (explicit Euler)
    C_CO2 = C_CO2 + (N - r) * dt
    C_NaOH = C_NaOH - r * dt

    # Prevent negative concentrations
    C_CO2 = np.maximum(C_CO2, 0.0)
    C_NaOH = np.maximum(C_NaOH, 0.0)

# Calculate Na2CO3 from NaOH consumed
C_Na2CO3 = (C_NaOH_initial - C_NaOH) / 2  # stoichiometry: 2 NaOH → 1 Na2CO3

# Results
print("\nFinal CO2 concentration (mol/m^3):")
print(C_CO2)

print("\nFinal NaOH concentration (mol/m^3):")
print(C_NaOH)

print("\nFinal Na2CO3 concentration (mol/m^3):")
print(C_Na2CO3)
