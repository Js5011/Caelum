import numpy as np
from scipy.integrate import solve_ivp

SEC_PER_YEAR = 365*24*3600

def run_full_model(D=2.0, H=12.0, G=0.4, L=0.6, yCO2=0.12, C_NaOH0=1000,
                   V_total=20, N=3, k_caus=0.5, eta_eq=0.85,
                   elec_price=0.1, lime_price=150):
    """
    Runs the CO₂ capture + CSTR simulator.
    Returns:
        CO2_tpy: annual CO₂ captured (t/year)
        cost_per_t: cost of capture ($/t CO₂)
        efficiency: capture efficiency (%)
    """
    # ---------- Geometry ----------
    A = np.pi * (D/2)**2
    vG = G / A
    vL = L / A

    # ---------- Bubble column ----------
    kLa = 0.28 * (vG / 0.1)**0.7
    k_rxn = 8000
    H_CO2 = 3.4e4

    Cg0 = yCO2 * 101325 / (8.314 * 298)
    Cl0 = 0.0

    def absorber(z, y):
        Cg, Cl, NaOH = y
        eta_effective = 0.90
        P_CO2 = Cg * 8.314 * 298
        C_star = P_CO2 / H_CO2
        N_mt = eta_effective * kLa * (C_star - Cl)
        r_rxn = k_rxn * Cl * (NaOH / (NaOH + 1000))
        dCgdz = -N_mt / vG
        dCldz = (N_mt - r_rxn) / vL
        dNaOHdz = -2 * r_rxn / vL
        return [dCgdz, dCldz, dNaOHdz]

    z_eval = np.linspace(0, H, 300)
    sol_abs = solve_ivp(absorber, [0, H], [Cg0, Cl0, C_NaOH0], t_eval=z_eval)
    Cg_abs, Cl_abs, NaOH_abs = sol_abs.y
    Cg_out = max(Cg_abs[-1], 0.0)

    CO2_abs_mol_s = max(G * (Cg0 - Cg_out), 0.0)
    efficiency = min(100 * CO2_abs_mol_s / (G * Cg0), 100)

    # ---------- CSTR Train ----------
    V = V_total / N
    tau = V / L
    tspan = np.linspace(0, 4*tau, 200)

    Na2CO3_in = CO2_abs_mol_s
    NaOH_in = 0.0
    CaOH2_in = 1.05 * Na2CO3_in

    for i in range(N):
        def cstr(t, y):
            Na2CO3, NaOH, CaOH2 = y
            r = min(k_caus * Na2CO3, eta_eq * Na2CO3 / tau)
            return [-r, 2*r, -r]

        sol_cstr = solve_ivp(cstr, [0, tspan[-1]],
                             [Na2CO3_in, NaOH_in, CaOH2_in],
                             t_eval=tspan)
        Na2CO3_in, NaOH_in, CaOH2_in = sol_cstr.y[:, -1]

    # ---------- Economics ----------
    absorber_cost = 18000 * (A * H)**0.62
    causticizer_cost = 22000 * V_total**0.6
    CAPEX = (absorber_cost + causticizer_cost) * 3.2 * 1.15

    pump_power = 1.5e5 * L / 0.7
    pump_cost = pump_power * SEC_PER_YEAR / 3.6e6 * elec_price

    CaOH2_tpy = CO2_abs_mol_s * 44.01/1000 * SEC_PER_YEAR
    lime_cost = CaOH2_tpy * lime_price

    fixed_OM = 0.05 * CAPEX
    compression_cost = 35 * CO2_abs_mol_s * SEC_PER_YEAR * 44.01/1000

    annual_cost = 0.10*CAPEX + pump_cost + lime_cost + fixed_OM + compression_cost
    CO2_tpy = CO2_abs_mol_s * 44.01/1000 * SEC_PER_YEAR
    cost_per_t = annual_cost / CO2_tpy

    return CO2_tpy, cost_per_t, efficiency
