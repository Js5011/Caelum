import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# 1. COLUMN PARAMETERS
# ======================================================

H = float(input("Column height (m): "))
N = int(input("Number of slices: "))
dz = H / N
z = np.linspace(0, H, N)

A = float(input("Column cross-sectional area (m^2): "))

# Liquid
L = 1e-3                     # m^3/s
vL = L / A                   # m/s

# Gas
G = 0.5                      # m^3/s (typical pilot-scale)
vG = G / A                   # m/s

# ======================================================
# 2. MASS TRANSFER
# ======================================================

kLa = 0.08                   # 1/s (physical)
E_max = 10                   # max enhancement due to NaOH
C_NaOH_ref = 1000             # mol/m^3 (for enhancement scaling)

# ======================================================
# 3. GAS–LIQUID EQUILIBRIUM
# ======================================================

H_CO2 = 3.4e4                # Pa·m^3/mol
P_total = 101325             # Pa

# ======================================================
# 4. INLET CONDITIONS
# ======================================================

# Gas inlet (bottom)
y_CO2_in = 0.15
C_CO2g_in = y_CO2_in * P_total / (8.314 * 298)  # mol/m^3

# Liquid inlet (top)
C_CO2l_in = 0.0
C_NaOH_in = 2000.0            # mol/m^3
C_Na2CO3_in = 0.0

# ======================================================
# 5. INITIAL PROFILES
# ======================================================

C_CO2g = np.ones(N) * C_CO2g_in
C_CO2l = np.zeros(N)
C_NaOH = np.ones(N) * C_NaOH_in
C_Na2CO3 = np.zeros(N)

# ======================================================
# 6. STEADY-STATE SOLVER
# ======================================================

tol = 1e-6
max_iter = 10000

for _ in range(max_iter):

    old = np.vstack([C_CO2g, C_CO2l, C_NaOH, C_Na2CO3]).copy()

    for i in range(N):

        # Upstream values
        CO2g_up = C_CO2g[i-1] if i > 0 else C_CO2g_in
        CO2l_up = C_CO2l[i-1] if i > 0 else C_CO2l_in
        NaOH_up = C_NaOH[i-1] if i > 0 else C_NaOH_in
        Na2CO3_up = C_Na2CO3[i-1] if i > 0 else C_Na2CO3_in

        # Local equilibrium
        P_CO2 = CO2g_up * 8.314 * 298
        C_star = P_CO2 / H_CO2

        # Reaction enhancement (bounded & stable)
        E = 1 + (E_max - 1) * (NaOH_up / (NaOH_up + C_NaOH_ref))
        kLa_eff = kLa * E

        # CO2 flux
        N_CO2 = kLa_eff * (C_star - C_CO2l[i])

        # Gas-phase balance
        C_CO2g[i] = CO2g_up - (N_CO2 * dz / vG)

        # Liquid CO2 balance
        C_CO2l[i] = CO2l_up + (N_CO2 * dz / vL)

        # Stoichiometric reaction
        dCO2 = max(C_CO2l[i] - CO2l_up, 0)
        dNaOH = 2 * dCO2

        C_NaOH[i] = max(NaOH_up - dNaOH, 0)
        C_Na2CO3[i] = Na2CO3_up + dCO2

    if np.max(np.abs(old - np.vstack([C_CO2g, C_CO2l, C_NaOH, C_Na2CO3]))) < tol:
        break

# ======================================================
# 7. RESULTS
# ======================================================

capture = 1 - C_CO2g[-1] / C_CO2g_in

print("\n===== DESIGN RESULTS =====")
print(f"CO₂ capture efficiency: {capture:.1%}")
print(f"NaOH remaining:         {C_NaOH[-1]:.1f} mol/m³")
print(f"Na₂CO₃ produced:        {C_Na2CO3[-1]:.1f} mol/m³")

# ======================================================
# 8. PLOTS
# ======================================================

plt.figure(figsize=(7,8))
plt.plot(C_CO2g, z, label="Gas CO₂")
plt.plot(C_CO2l, z, label="Liquid CO₂")
plt.plot(C_NaOH, z, label="NaOH")
plt.plot(C_Na2CO3, z, label="Na₂CO₃")
plt.xlabel("Concentration (mol/m³)")
plt.ylabel("Height (m)")
plt.title("Bubble Column CO₂ Absorption with NaOH")
plt.legend()
plt.grid()
plt.show()
