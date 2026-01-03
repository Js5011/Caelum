# streamlit_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.set_page_config(page_title="CO₂ Capture Simulator", layout="wide")
st.title("🏭 Industrial CO₂ Capture + Causticizing Simulator")

# ===================== SIDEBAR INPUTS =====================
st.sidebar.header("Bubble Column Inputs")
D = st.sidebar.slider("Absorber diameter (m)", 0.5, 5.0, 1.5, 0.1)
H = st.sidebar.slider("Absorber height (m)", 5.0, 20.0, 10.0, 0.5)
G = st.sidebar.slider("Gas flow rate (m³/s)", 0.05, 1.0, 0.25, 0.01)
yCO2 = st.sidebar.slider("CO₂ mole fraction", 0.02, 0.15, 0.12, 0.01)
L = st.sidebar.slider("Liquid flow rate (m³/s)", 0.1, 1.0, 0.6, 0.01)
C_NaOH0 = st.sidebar.slider("NaOH concentration (mol/m³)", 500, 1500, 1000, 50)

st.sidebar.header("CSTR Train Inputs")
V_total = st.sidebar.slider("Total causticizer volume (m³)", 5.0, 50.0, 20.0, 1.0)
N = st.sidebar.slider("Number of CSTRs", 1, 6, 3, 1)
eta_eq = st.sidebar.slider("Max causticizing conversion", 0.6, 0.9, 0.8, 0.05)

st.sidebar.header("Economic Inputs")
elec_price = st.sidebar.number_input("Electricity price ($/kWh)", 0.05, 0.25, 0.1, 0.01)
lime_price = st.sidebar.number_input("Lime price ($/ton Ca(OH)₂)", 80, 250, 140, 10)

# ===================== CONSTANTS =====================
R = 8.314
T = 298
P = 101325
SEC_PER_YEAR = 365 * 24 * 3600

A = np.pi * (D / 2)**2
vG = G / A
vL = L / A

# ===================== MODEL =====================
@st.cache_data
def run_models():
    # ---- Bubble column ----
    kLa = 0.12  # realistic industrial range (1/s)
    H_CO2 = 3.4e4

    Cg0 = yCO2 * P / (R * T)
    Cl0 = 0.0

    def absorber(z, y):
        Cg, Cl, NaOH = y
        C_star = (Cg * R * T) / H_CO2
        N_mt = kLa * max(C_star - Cl, 0)

        # reaction limited by NaOH availability
        r_rxn = 0.8 * N_mt

        return [
            -N_mt / vG,
            (N_mt - r_rxn) / vL,
            -2 * r_rxn / vL
        ]

    z = np.linspace(0, H, 250)
    sol = solve_ivp(absorber, [0, H], [Cg0, Cl0, C_NaOH0], t_eval=z)

    Cg, Cl, NaOH = sol.y

    # ---- CO₂ balance ----
    n_gas = G * P / (R * T)
    CO2_in = yCO2 * n_gas
    CO2_out = (Cg[-1] / Cg0) * CO2_in
    CO2_abs = CO2_in - CO2_out
    efficiency = 100 * CO2_abs / CO2_in

    # ---- Causticizer ----
    tau = (V_total / N) / L
    Na2CO3 = CO2_abs

    conv = eta_eq * (1 - np.exp(-N * tau / 600))
    NaOH_regen = 2 * Na2CO3 * conv

    return z, Cg, Cl, NaOH, efficiency, CO2_abs, conv, NaOH_regen

# ===================== RUN =====================
if st.button("Run Simulation"):
    z, Cg, Cl, NaOH, eff, CO2_abs, conv, NaOH_regen = run_models()

    # ===================== ECONOMICS =====================
    CO2_tpy = CO2_abs * 44.01 / 1000 * SEC_PER_YEAR * 0.85

    CAPEX = (
        15000 * (A * H)**0.6 +
        20000 * V_total**0.55
    ) * 2.8

    pump_power = 1.2e5 * L / 0.7
    pump_cost = pump_power * SEC_PER_YEAR / 3.6e6 * elec_price

    CaOH2_tpy = CO2_tpy * (74.09 / 44.01)
    lime_cost = CaOH2_tpy * lime_price

    compression_cost = 30 * CO2_tpy
    fixed_OM = 0.04 * CAPEX

    OPEX = pump_cost + lime_cost + compression_cost + fixed_OM
    annual_cost = 0.1 * CAPEX + OPEX
    cost_per_t = annual_cost / CO2_tpy

    # ===================== METRICS =====================
    st.subheader("📌 Key Results")
    st.metric("CO₂ Capture Efficiency (%)", f"{eff:.1f}")
    st.metric("CO₂ Captured (t/year)", f"{CO2_tpy:,.0f}")
    st.metric("Cost of Capture ($/ton)", f"${cost_per_t:,.0f}")

    # ===================== PLOTS =====================
    st.subheader("📊 Absorber Profiles")
    fig, ax = plt.subplots()
    ax.plot(Cg / Cg[0] * 100, z, label="Gas CO₂ (%)")
    ax.plot(NaOH / NaOH[0] * 100, z, label="NaOH (%)")
    ax.set_xlabel("Relative concentration (%)")
    ax.set_ylabel("Height (m)")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    st.subheader("📊 CO₂ Driving Force")
    fig2, ax2 = plt.subplots()
    ax2.plot(z, (Cg * R * T) / 3.4e4 - Cl)
    ax2.set_xlabel("Height (m)")
    ax2.set_ylabel("Driving force (mol/m³)")
    ax2.grid()
    st.pyplot(fig2)

    st.subheader("📊 Cost Breakdown")
    fig3, ax3 = plt.subplots()
    ax3.bar(
        ["Pumping", "Lime", "Compression", "Fixed O&M"],
        [pump_cost, lime_cost, compression_cost, fixed_OM]
    )
    ax3.set_ylabel("$/year")
    ax3.grid(axis="y")
    st.pyplot(fig3)

