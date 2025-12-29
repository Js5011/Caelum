import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ===================== USER INPUTS =====================
V_total = float(input("Total reactor volume V_total (m^3): "))
N = int(input("Number of CSTRs in series: "))
F = float(input("Liquid flow rate F (m^3/s): "))

C_Na2CO3_in = float(input("Inlet Na2CO3 (mol/m^3): "))
C_NaOH_in   = float(input("Inlet NaOH (mol/m^3): "))
CaOH2_0 = float(input("Initial Ca(OH)2 solids (mol/m^3 slurry): "))

k = float(input("Surface rate constant k (m/s): "))
a_s0 = float(input("Initial specific surface area a_s0 (m^2/m^3): "))

eta_eq = float(input("Equilibrium causticizing efficiency (0–1): "))
t_final = float(input("Simulation time (s): "))

# Divide volume equally
V = V_total / N

# ===================== MODEL =====================
def multi_cstr(t, y):
    dydt = np.zeros_like(y)
    for i in range(N):
        idx = i*4
        # Extract species
        C_Na2CO3 = y[idx]
        C_NaOH   = y[idx+1]
        CaOH2_s  = y[idx+2]
        CaCO3_s  = y[idx+3]

        # Previous reactor outlet or inlet
        if i == 0:
            C_Na2CO3_in_i = C_Na2CO3_in
            C_NaOH_in_i = C_NaOH_in
        else:
            C_Na2CO3_in_i = y[(i-1)*4]
            C_NaOH_in_i = y[(i-1)*4 + 1]

        # Shrinking-core surface area
        a_s = a_s0 * (CaOH2_s / CaOH2_0)**(2/3) if CaOH2_s>0 else 0
        r = k * a_s * C_Na2CO3
        eta = C_NaOH / (C_NaOH + C_Na2CO3 + 1e-12)
        r_eff = r * max(0, 1 - eta / eta_eq)

        # CSTR balances
        dydt[idx]   = (F/V)*(C_Na2CO3_in_i - C_Na2CO3) - r_eff
        dydt[idx+1] = (F/V)*(C_NaOH_in_i - C_NaOH) + 2*r_eff
        dydt[idx+2] = -r_eff
        dydt[idx+3] = r_eff

    return dydt

# ===================== INITIAL CONDITIONS =====================
y0 = []
for _ in range(N):
    y0.extend([C_Na2CO3_in, C_NaOH_in, CaOH2_0, 0.0])

t_eval = np.linspace(0, t_final, 600)

sol = solve_ivp(multi_cstr, [0, t_final], y0, t_eval=t_eval, method="BDF")

# ===================== PLOTS =====================
plt.figure(figsize=(14, 10))

species_labels = ["Na₂CO₃ (aq)", "NaOH (aq)", "Ca(OH)₂ (solid)", "CaCO₃ (solid)"]
colors = ["blue", "orange", "red", "green"]

# Plot each species for each reactor
for s in range(4):
    plt.subplot(2,2,s+1)
    for i in range(N):
        plt.plot(sol.t, sol.y[i*4+s], label=f"Reactor {i+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Concentration (mol/m³)")
    plt.title(f"{species_labels[s]} in each reactor")
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()

# ===================== CARBON CAPTURE EFFECTIVENESS =====================
plt.figure(figsize=(10,6))
for i in range(N):
    CaCO3_i = sol.y[i*4+3]
    eta_capture_i = CaCO3_i / (C_Na2CO3_in * V_total) * 100
    plt.plot(sol.t, eta_capture_i, label=f"Reactor {i+1}")
plt.xlabel("Time (s)")
plt.ylabel("Carbon Capture (%)")
plt.title("Carbon Capture Effectiveness in Each Reactor")
plt.legend()
plt.grid()
plt.show()
