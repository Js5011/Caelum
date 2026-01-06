# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="CO₂ Capture & Causticizing Simulator", layout="wide")
st.title("🏭 Industrial CO₂ Capture & Causticizing Simulator")

# ==========================================================
# SIDEBAR INPUTS (EXPANDED RANGES)
# ==========================================================
st.sidebar.header("Bubble Column Inputs")

D = st.sidebar.slider("Absorber diameter (m)", 0.5, 10.0, 2.0, 0.1)
H = st.sidebar.slider("Absorber height (m)", 5.0, 40.0, 15.0, 0.5)
G = st.sidebar.slider("Gas flow rate (m³/s)", 0.05, 5.0, 0.8, 0.05)
L = st.sidebar.slider("Liquid flow rate (m³/s)", 0.05, 5.0, 1.2, 0.05)
yCO2 = st.sidebar.slider("CO₂ mole fraction", 0.02, 0.30, 0.12, 0.01)
C_NaOH0 = st.sidebar.slider("NaOH concentration (mol/m³)", 300, 3000, 1200, 50)

st.sidebar.header("Causticizing Reactor Inputs")
V_total = st.sidebar.slider("Total reactor volume (m³)", 5.0, 80.0, 30.0, 1.0)
N = st.sidebar.slider("Number of CSTRs", 1, 8, 4)
k_caus = st.sidebar.slider("Causticizing rate constant (1/s)", 0.05, 1.5, 0.6, 0.05)
eta_eq = st.sidebar.slider("Equilibrium conversion limit", 0.5, 0.98, 0.85, 0.01)

st.sidebar.header("Economic Inputs")
elec_price = st.sidebar.number_input("Electricity price ($/kWh)", 0.05, 0.5, 0.1)
lime_price = st.sidebar.number_input("Lime price ($/ton Ca(OH)₂)", 50, 500, 150)

# ==========================================================
# GEOMETRY
# ==========================================================
A = np.pi * (D / 2) ** 2
vG = G / A
vL = L / A

# ==========================================================
# CORE MODEL
# ==========================================================
@st.cache_data(show_spinner=False)
def run_model(
    D, H, G, L, yCO2, C_NaOH0,
    V_total, N, k_caus, eta_eq,
    vG, vL
):
    # ---------- Constants ----------
    R = 8.314
    T = 298
    P = 101325
    H_CO2 = 3.4e4
    k_rxn = 8000

    # Realistic kLa (bubble column correlation)
    kLa = 0.12 * (vG ** 0.6) * (vL ** 0.2)

    # ---------- Inlet ----------
    Cg0 = yCO2 * P / (R * T)
    Cl0 = 0.0

    # ---------- Bubble Column ----------
    def absorber(z, y):
        Cg, Cl, NaOH = y

        # Dimensionless groups
        Da = k_rxn * H / max(vL, 1e-6)
        NTU = kLa * H / max(vG, 1e-6)

        # Realistic effectiveness (hydrodynamics + kinetics)
        eta_eff = 1 - np.exp(-0.6 * NTU * (Da / (1 + Da)))
        eta_eff = np.clip(eta_eff, 0.1, 0.95)

        # Mass transfer
        P_CO2 = Cg * R * T
        C_star = P_CO2 / H_CO2
        N_mt = eta_eff * kLa * (C_star - Cl)

        # Reaction
        r_rxn = k_rxn * Cl * (NaOH / (NaOH + 1000))

        dCgdz = -N_mt / vG
        dCldz = (N_mt - r_rxn) / vL
        dNaOHdz = -2 * r_rxn / vL

        return [dCgdz, dCldz, dNaOHdz]

    z = np.linspace(0, H, 300)
    sol = solve_ivp(absorber, [0, H], [Cg0, Cl0, C_NaOH0], t_eval=z)

    Cg_abs, Cl_abs, NaOH_abs = sol.y
    Cg_out = max(Cg_abs[-1], 1e-6)

    CO2_abs_mol_s = max(G * (Cg0 - Cg_out), 0.0)
    efficiency = min(100 * CO2_abs_mol_s / (G * Cg0), 95.0)

    # Diagnostics
    driving_force = (Cg_abs * R * T / H_CO2) - Cl_abs
    reaction_rate = k_rxn * Cl_abs * (NaOH_abs / (NaOH_abs + 1000))

    # ---------- CSTR TRAIN ----------
    V = V_total / N
    tau = V / L
    t = np.linspace(0, 4 * tau, 200)

    Na2CO3_in = CO2_abs_mol_s
    NaOH_in = 0.0
    CaOH2_in = 1.05 * Na2CO3_in

    Na2CO3_hist, conv_hist = [], []

    for _ in range(N):
        def cstr(t, y):
            Na2CO3, NaOH, CaOH2 = y
            r = min(k_caus * Na2CO3, eta_eq * Na2CO3 / tau)
            return [-r, 2 * r, -r]

        sol_cstr = solve_ivp(
            cstr, [0, t[-1]],
            [Na2CO3_in, NaOH_in, CaOH2_in],
            t_eval=t
        )

        Na2CO3 = sol_cstr.y[0]
        Na2CO3_hist.append(Na2CO3)
        conv_hist.append((1 - Na2CO3 / Na2CO3[0]) * 100)

        Na2CO3_in = Na2CO3[-1]

    return (
        z, Cg_abs, Cl_abs, NaOH_abs,
        CO2_abs_mol_s, efficiency,
        driving_force, reaction_rate,
        t, Na2CO3_hist, conv_hist
    )

# ==========================================================
# RUN
# ==========================================================
if st.button("▶ Run Simulation"):

    (
        z, Cg_abs, Cl_abs, NaOH_abs,
        CO2_abs_mol_s, efficiency,
        driving_force, reaction_rate,
        t, Na2CO3_hist, conv_hist
    ) = run_model(
        D, H, G, L, yCO2, C_NaOH0,
        V_total, N, k_caus, eta_eq,
        vG, vL
    )

    # ==========================================================
    # ANNUAL CO₂
    # ==========================================================
    SEC_PER_YEAR = 365 * 24 * 3600
    capacity_factor = 0.85

    CO2_tpy = CO2_abs_mol_s * 44.01 / 1000 * SEC_PER_YEAR * capacity_factor

    # ==========================================================
    # ECONOMICS
    # ==========================================================
    absorber_cost = 18000 * (A * H) ** 0.62
    causticizer_cost = 22000 * V_total ** 0.6
    CAPEX = (absorber_cost + causticizer_cost) * 3.2 * 1.15

    pump_power = 1.5e5 * L / 0.7
    pump_cost = pump_power * SEC_PER_YEAR / 3.6e6 * elec_price

    CaOH2_tpy = CO2_tpy * (74.09 / 44.01)
    lime_cost = CaOH2_tpy * lime_price

    annual_cost = (
        0.10 * CAPEX +
        0.05 * CAPEX +
        pump_cost +
        lime_cost +
        35 * CO2_tpy
    )

    cost_per_t = annual_cost / max(CO2_tpy, 1)

    # ==========================================================
    # RESULTS
    # ==========================================================
    st.subheader("📌 Key Results")
    st.metric("CO₂ Capture Efficiency (%)", f"{efficiency:.1f}")
    st.metric("CO₂ Captured (t/year)", f"{CO2_tpy:,.0f}")
    st.metric("Cost of Capture ($/t)", f"{cost_per_t:,.0f}")

    # ==========================================================
    # PLOTS – ABSORBER
    # ==========================================================
    st.subheader("📊 Bubble Column Profiles")

    fig1, ax1 = plt.subplots()
    ax1.plot(Cg_abs / Cg_abs[0], z, label="Gas CO₂")
    ax1.plot(NaOH_abs / NaOH_abs[0], z, label="NaOH")
    ax1.set_xlabel("Normalized Concentration")
    ax1.set_ylabel("Height (m)")
    ax1.legend()
    ax1.grid()
    st.pyplot(fig1)

    # Driving force
    fig2, ax2 = plt.subplots()
    ax2.plot(driving_force, z)
    ax2.set_xlabel("CO₂ Driving Force (mol/m³)")
    ax2.set_ylabel("Height (m)")
    ax2.grid()
    st.pyplot(fig2)

    # NaOH utilization
    fig3, ax3 = plt.subplots()
    ax3.plot((C_NaOH0 - NaOH_abs) / C_NaOH0 * 100, z)
    ax3.set_xlabel("NaOH Utilization (%)")
    ax3.set_ylabel("Height (m)")
    ax3.grid()
    st.pyplot(fig3)

    # Limiting regime
    fig4, ax4 = plt.subplots()
    ax4.plot(driving_force / np.max(driving_force), z, label="Mass Transfer")
    ax4.plot(reaction_rate / np.max(reaction_rate), z, label="Reaction")
    ax4.set_xlabel("Normalized Limitation")
    ax4.set_ylabel("Height (m)")
    ax4.legend()
    ax4.grid()
    st.pyplot(fig4)

    # ==========================================================
    # CSTR CONVERSION
    # ==========================================================
    st.subheader("📊 Causticizing Conversion")

    fig5, ax5 = plt.subplots()
    for i in range(N):
        ax5.plot(t / 60, conv_hist[i], label=f"CSTR {i+1}")
    ax5.set_xlabel("Time (min)")
    ax5.set_ylabel("Conversion (%)")
    ax5.legend()
    ax5.grid()
    st.pyplot(fig5)
