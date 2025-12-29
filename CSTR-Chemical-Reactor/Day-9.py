import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ===================== USER INPUTS =====================

V = float(input("Reactor volume V (m^3): "))
F = float(input("Liquid flow rate F (m^3/s): "))

C_Na2CO3_in = float(input("Inlet Na2CO3 (mol/m^3): "))
C_NaOH_in   = float(input("Inlet NaOH (mol/m^3): "))

CaOH2_solid_in = float(input("Inlet Ca(OH)2 solids (mol/m^3 slurry): "))

k = float(input("Surface reaction constant k (1/s): "))
a_s0 = float(input("Initial Ca(OH)2 surface area a_s0 (m^2/m^3): "))

t_final = float(input("Simulation time (s): "))

# ===================== MODEL =====================

def cstr_causticizing(t, y):
    C_Na2CO3, C_NaOH, CaOH2_s, CaCO3_s = y

    # Prevent non-physical values
    C_Na2CO3 = max(C_Na2CO3, 0)
    CaOH2_s  = max(CaOH2_s, 0)

    # Surface area decays as Ca(OH)2 is consumed
    a_s = a_s0 * (CaOH2_s / CaOH2_solid_in)

    # Reaction rate (mol/m^3/s)
    r = k * a_s * C_Na2CO3

    # CSTR balances
    dC_Na2CO3 = (F/V)*(C_Na2CO3_in - C_Na2CO3) - r
    dC_NaOH   = (F/V)*(C_NaOH_in   - C_NaOH) + 2*r
    dCaOH2_s  = -r
    dCaCO3_s  = r

    return [dC_Na2CO3, dC_NaOH, dCaOH2_s, dCaCO3_s]

# ===================== INITIAL CONDITIONS =====================

y0 = [
    C_Na2CO3_in,
    C_NaOH_in,
    CaOH2_solid_in,
    0.0
]

# ===================== SOLVER =====================

t_eval = np.linspace(0, t_final, 500)

sol = solve_ivp(
    cstr_causticizing,
    [0, t_final],
    y0,
    t_eval=t_eval,
    method="BDF"  # stiff, efficient for reactors
)

# ===================== PLOTS =====================

plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label="Na₂CO₃ (aq)")
plt.plot(sol.t, sol.y[2], label="Ca(OH)₂ (solid)")
plt.plot(sol.t, sol.y[3], label="CaCO₃ (solid)")
plt.xlabel("Time (s)")
plt.ylabel("Concentration (mol/m³)")
plt.title("Industrial-Grade Causticizing CSTR")
plt.legend()
plt.grid()
plt.show()
