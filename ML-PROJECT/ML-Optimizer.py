import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import matplotlib.pyplot as plt

# ==========================================================
# CONSTANTS
# ==========================================================
R = 8.314
T = 298
P = 101325
H_CO2 = 3.4e4
SEC_PER_YEAR = 365*24*3600
capacity_factor = 0.85

# ==========================================================
# FIXED INPUTS (can tweak these later)
# ==========================================================
yCO2 = 0.12           # mole fraction of CO2 in feed
N = 3                 # number of CSTRs
eta_eq = 0.95         # equilibrium conversion
lime_price = 300      # $/t Ca(OH)2
elec_price = 0.12     # $/kWh

# ==========================================================
# CORE MODEL
# ==========================================================
def run_full_model(D, H, G, L, C_NaOH0,
                   V_total, k_caus):

    # Geometry
    A = np.pi * (D / 2) ** 2
    vG = G / A
    vL = L / A

    # Absorber parameters
    k_rxn = 8000
    kLa = 0.28 * (vG / 0.1) ** 0.7
    eta_effective = 0.90

    Cg0 = yCO2 * P / (R * T)
    Cl0 = 0.0

    def absorber(z, y):
        Cg, Cl, NaOH = y
        P_CO2 = Cg * R * T
        C_star = P_CO2 / H_CO2
        N_mt = eta_effective * kLa * (C_star - Cl)
        r_rxn = k_rxn * Cl * (NaOH / (NaOH + 1000))
        return [-N_mt / vG, (N_mt - r_rxn) / vL, -2 * r_rxn / vL]

    z = np.linspace(0, H, 300)
    sol = solve_ivp(absorber, [0, H], [Cg0, Cl0, C_NaOH0], t_eval=z)

    Cg = sol.y[0]
    CO2_abs_mol_s = max(G * (Cg0 - Cg[-1]), 0.0)
    efficiency = min(100 * CO2_abs_mol_s / (G * Cg0), 100)

    # CSTR train
    V = V_total / N
    tau = V / L
    t = np.linspace(0, 4*tau, 200)

    Na2CO3 = CO2_abs_mol_s
    NaOH = 0.0
    CaOH2 = 1.05 * Na2CO3

    for _ in range(N):
        def cstr(t, y):
            Na2CO3, NaOH, CaOH2 = y
            r = min(k_caus * Na2CO3, eta_eq * Na2CO3 / tau)
            return [-r, 2*r, -r]

        sol_cstr = solve_ivp(cstr, [0, t[-1]], [Na2CO3, NaOH, CaOH2], t_eval=t)
        Na2CO3, NaOH, CaOH2 = sol_cstr.y[:, -1]

    # Annual CO2
    CO2_tpy = CO2_abs_mol_s * 44.01/1000 * SEC_PER_YEAR * capacity_factor

    # Economics
    absorber_cost = 18000 * (A*H)**0.62
    causticizer_cost = 22000 * V_total**0.6
    CAPEX = (absorber_cost + causticizer_cost) * 3.2 * 1.15

    pump_power = 1.5e5 * L / 0.7
    pump_cost = pump_power * SEC_PER_YEAR / 3.6e6 * elec_price

    CaOH2_tpy = CO2_tpy * (74.09/44.01)
    lime_cost = CaOH2_tpy * lime_price

    fixed_OM = 0.05 * CAPEX
    compression_cost = 35 * CO2_tpy

    annual_cost = 0.10*CAPEX + pump_cost + lime_cost + fixed_OM + compression_cost
    cost_per_t = annual_cost / CO2_tpy

    return CO2_tpy, cost_per_t, efficiency

# ==========================================================
# BAYESIAN OPTIMIZATION
# ==========================================================
# Define search space
space = [
    Real(2.0, 8.0, name='D'),          # absorber diameter [m]
    Real(10.0, 40.0, name='H'),        # absorber height [m]
    Real(0.5, 5.0, name='G'),          # gas flow rate [m3/s]
    Real(0.5, 5.0, name='L'),          # liquid flow rate [m3/s]
    Real(500.0, 5000.0, name='C_NaOH0'), # NaOH conc [mol/m3]
    Real(50.0, 500.0, name='V_total'),   # causticizer volume [m3]
    Real(0.001, 0.1, name='k_caus')     # causticizing rate [1/s]
]

# Objective: minimize cost while targeting 91-92% efficiency
@use_named_args(space)
def objective(**params):
    _, cost, eff = run_full_model(**params)
    penalty = 0
    # Penalize if outside target efficiency
    if eff < 91:
        penalty += (91-eff)**2 * 1e4
    elif eff > 92:
        penalty += (eff-92)**2 * 1e4
    return cost + penalty

# Run optimization
res = gp_minimize(objective, space, n_calls=50, random_state=42)

# ==========================================================
# SHOW RESULTS
# ==========================================================
print("\nOptimal Settings:")
for dim, val in zip(space, res.x):
    print(f"{dim.name}: {val:.3f}")

CO2_tpy_opt, cost_opt, eff_opt = run_full_model(*res.x)
print(f"\nPredicted CO2 capture: {CO2_tpy_opt:,.0f} t/year")
print(f"Predicted cost: ${cost_opt:,.0f}/tCO2")
print(f"Predicted efficiency: {eff_opt:.2f}%")

# Optional: Plot convergence
plt.figure(figsize=(6,4))
plt.plot(res.func_vals, 'o-')
plt.xlabel('Iteration')
plt.ylabel('Objective (Cost + Penalty)')
plt.title('Bayesian Optimization Convergence')
plt.grid(True)
plt.show()
