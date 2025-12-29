import numpy as np
# Column parameters
H = float(input("Example Column Height"))  # meters, example column height
n_slices = int(input("Number of slices in example"))  # number of slices
dz = H / n_slices  # height of each slice

# Initial concentrations
C_CO2 = np.zeros(n_slices)  # CO₂ concentration in mol/m³
C_NaOH_feed = float(input("What is the feed concentration of NaOH in mol/m^3"))  # mol/m³, example feed concentration of NaOH
C_NaOH = np.full(n_slices, C_NaOH_feed)  # Initialize NaOH to feed value
print(f"This is CO2 concentration in mol/m^3: {C_CO2}")
print(f"This is feed concentration of NaOH: {C_NaOH_feed}")
#This all has no effect, since we are assuming that there is no CO2 currently in the solution, so it outputs zeroes.