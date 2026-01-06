import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# ==========================================================
# CONSTANTS (MATCH STREAMLIT)
# ==========================================================
R = 8.314
T = 298
P = 101325
H_CO2 = 3.4e4
SEC_PER_YEAR = 365 * 24 * 3600
capacity_factor = 0.85

# ==========================================================
# USER INPUTS (CLI VERSION OF SIDEBAR)
# ==========================================================
print("===== CO₂ Capture System Inputs =====")
D = float(input("Absorber diameter (m): "))
H = float(input("Absorber height (m): "))
G = float(input("Gas flow rate (m³/s): "))
yCO2 = float(input("CO₂ mole fraction (0–1): "))
L = float(input("Liquid flow rate (m³/s): "))
C_NaOH0 = float(input("NaOH concentration (mol/m³): "))

V_total = float(input("Total causticizer volume (m³): "))
N = int(input("Number of CSTRs: "))
k_caus = float(input("Causticizing rate constant (1/s): "))
eta_eq = float(input("Equilibrium conversion (0–1): "))

elec_price = float(input("Electricity price ($/kWh): "))
lime_price = float(input("Lime price ($/ton Ca(OH)₂): "))

# ==========================================================
# CORE MODEL (IDENTICAL TO app.py)
# ==========================================================
def run_full_model(D, H, G, L, yCO2, C_NaOH0,
                   V_total, N, k_caus, eta_eq,
                   elec_price, lime_price):

    # ---------- Geometry ----------
    A = np.pi * (D / 2) ** 2
    vG = G / A
    vL = L / A

    # ---------- Absorber ----------
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

        return [
            -N_mt / vG,
            (N_mt - r_rxn) / vL,
            -2 * r_rxn / vL
        ]

    z = np.linspace(0, H, 300)
    sol = solve_ivp(absorber, [0, H], [Cg0, Cl0, C_NaOH0], t_eval=z)

    Cg = sol.y[0]
    CO2_abs_mol_s = max(G * (Cg0 - Cg[-1]), 0.0)
    efficiency = min(100 * CO2_abs_mol_s / (G * Cg0), 100)

    # ---------- CSTR Train ----------
    V = V_total / N
    tau = V / L
    t = np.linspace(0, 4 * tau, 200)

    Na2CO3 = CO2_abs_mol_s
    NaOH = 0.0
    CaOH2 = 1.05 * Na2CO3

    for _ in range(N):
        def cstr(t, y):
            Na2CO3, NaOH, CaOH2 = y
            r = min(k_caus * Na2CO3, eta_eq * Na2CO3 / tau)
            return [-r, 2 * r, -r]

        sol_cstr = solve_ivp(cstr, [0, t[-1]],
                             [Na2CO3, NaOH, CaOH2], t_eval=t)
        Na2CO3, NaOH, CaOH2 = sol_cstr.y[:, -1]

    # ---------- Annual CO₂ ----------
    CO2_tpy = (
        CO2_abs_mol_s
        * 44.01 / 1000
        * SEC_PER_YEAR
        * capacity_factor
    )

    # ---------- Economics (MATCH app.py) ----------
    absorber_cost = 18000 * (A * H) ** 0.62
    causticizer_cost = 22000 * V_total ** 0.6
    CAPEX = (absorber_cost + causticizer_cost) * 3.2 * 1.15

    pump_power = 1.5e5 * L / 0.7
    pump_cost = pump_power * SEC_PER_YEAR / 3.6e6 * elec_price

    CaOH2_tpy = CO2_tpy * (74.09 / 44.01)
    lime_cost = CaOH2_tpy * lime_price

    fixed_OM = 0.05 * CAPEX
    compression_cost = 35 * CO2_tpy

    annual_cost = (
        0.10 * CAPEX +
        pump_cost +
        lime_cost +
        fixed_OM +
        compression_cost
    )

    cost_per_t = annual_cost / CO2_tpy

    return CO2_tpy, cost_per_t, efficiency

# ==========================================================
# BASELINE RUN
# ==========================================================
CO2_tpy, cost_per_t, efficiency = run_full_model(
    D, H, G, L, yCO2, C_NaOH0,
    V_total, N, k_caus, eta_eq,
    elec_price, lime_price
)

print(f"\nCO₂ captured: {CO2_tpy:,.0f} t/year")
print(f"Cost of capture: ${cost_per_t:,.0f}/tCO₂")
print(f"Capture efficiency: {efficiency:.1f}%")

# ==========================================================
# MONTE CARLO (FIXED)
# ==========================================================
N_MC = 2000
cost_MC = []
eff_MC = []

for _ in range(N_MC):
    CO2_tpy_i, cost_i, eff_i = run_full_model(
        D * np.random.uniform(0.7, 1.3),
        H * np.random.uniform(0.7, 1.3),
        G * np.random.uniform(0.7, 1.3),
        L * np.random.uniform(0.7, 1.3),
        yCO2,
        C_NaOH0 * np.random.uniform(0.7, 1.3),
        V_total,
        N,
        k_caus,
        eta_eq,
        elec_price * np.random.uniform(0.7, 1.3),
        lime_price
    )
    cost_MC.append(cost_i)
    eff_MC.append(eff_i)

df_MC = pd.DataFrame({
    "Cost ($/t)": cost_MC,
    "Efficiency (%)": eff_MC
})

print("\nMonte Carlo Summary")
print(df_MC.describe())
