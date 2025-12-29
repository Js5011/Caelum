import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ===================== USER INPUTS =====================

V = float(input("CSTR volume V (m^3): "))
F = float(input("Volumetric flow rate F (m^3/s): "))

C_Na2CO3_in = float(input("Inlet Na2CO3 concentration (mol/m^3): "))
C_CaOH2_in = float(input("Inlet Ca(OH)2 concentration (mol/m^3): "))

k = float(input("Reaction rate constant k (m^3/mol/s): "))

t_final = float(input("Simulation time (s): "))

# ===================== MODEL =====================

def cstr_odes(t, y):
    C_Na2CO3, C_CaOH2, C_NaOH, C_CaCO3 = y

    # Reaction rate (mol/m^3/s)
    r = k * C_Na2CO3 * C_CaOH2

    dC_Na2CO3 = (F / V) * (C_Na2CO3_in - C_Na2CO3) - r
    dC_CaOH2  = (F / V) * (C_CaOH2_in  - C_CaOH2)  - r
    dC_NaOH   = (F / V) * (0 - C_NaOH) + 2 * r
    dC_CaCO3  = r  # solid accumulation

    return [dC_Na2CO3, dC_CaOH2, dC_NaOH, dC_CaCO3]

# ===================== INITIAL CONDITIONS =====================

y0 = [
    C_Na2CO3_in,   # start at inlet concentration
    C_CaOH2_in,
    0.0,           # no NaOH initially
    0.0            # no CaCO3 initially
]

# ===================== SOLVE =====================

t_eval = np.linspace(0, t_final, 400)

sol = solve_ivp(
    cstr_odes,
    [0, t_final],
    y0,
    t_eval=t_eval,
    method="RK45"
)

# ===================== PLOTS =====================

plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label="Na₂CO₃")
plt.plot(sol.t, sol.y[1], label="Ca(OH)₂")
plt.plot(sol.t, sol.y[2], label="NaOH")
plt.plot(sol.t, sol.y[3], label="CaCO₃ (solid)")
plt.xlabel("Time (s)")
plt.ylabel("Concentration (mol/m³)")
plt.title("CSTR Causticizing Reaction")
plt.legend()
plt.grid()
plt.show()