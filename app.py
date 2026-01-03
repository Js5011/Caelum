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
yCO2 = st.sidebar.slider("CO₂ mole fraction (0–1)", 0.01, 0.2, 0.12, 0.01)
L = st.sidebar.slider("Liquid flow rate (m³/s)", 0.1, 1.0, 0.6, 0.01)
C_NaOH0 = st.sidebar.slider("NaOH concentration (mol/m³)", 500, 2000, 1000, 50)

st.sidebar.header("CSTR Train Inputs")
V_total = st.sidebar.slider("Total Causticizer volume (m³)", 5.0, 50.0, 20.0, 1.0)
N = st.sidebar.slider("Number of CSTRs", 1, 6, 3, 1)
k_caus = st.sidebar.slider("Causticizing rate constant (1/s)", 0.1, 1.0, 0.5, 0.05)
eta_eq = st.sidebar.slider("Equilibrium conversion (0–1)", 0.5, 1.0, 0.85, 0.01)

st.sidebar.header("Economic Inputs")
elec_price = st.sidebar.number_input("Electricity price ($/kWh)", 0.05, 0.5, 0.1, 0.01)
lime_price = st.sidebar.number_input("Lime price ($/ton Ca(OH)₂)", 50, 500, 150, 10)

# ===================== ANIMATION OPTION =====================
run_animation = st.checkbox("▶ Run animations (slower)")

# ===================== GEOMETRY =====================
A = np.pi*(D/2)**2
vG = G/A
vL = L/A

# ===================== CACHE HEAVY COMPUTATIONS =====================
@st.cache_data
def run_models(D,H,G,L,C_NaOH0,V_total,N,k_caus,eta_eq):
    # --- Bubble column parameters ---
    kLa = 0.28 * (vG / 0.1)**0.7
    k_rxn = 8000
    H_CO2 = 3.4e4
    R = 8.314
    T = 298
    P = 101325

    Cg0 = yCO2*P/(R*T)
    Cl0 = 0.0

    def absorber(z, y):
        Cg, Cl, NaOH = y
        P_CO2 = Cg*R*T
        C_star = P_CO2/H_CO2
        N_mt = kLa*(C_star - Cl)
        r_rxn = k_rxn*Cl*(NaOH/(NaOH+1000))
        dCgdz = -N_mt/vG
        dCldz = (N_mt - r_rxn)/vL
        dNaOHdz = -2*r_rxn/vL
        return [dCgdz, dCldz, dNaOHdz]

    z_eval = np.linspace(0,H,300)
    sol_abs = solve_ivp(absorber,[0,H],[Cg0,Cl0,C_NaOH0], t_eval=z_eval)
    Cg, Cl, NaOH = sol_abs.y

    CO2_abs = max(G*(Cg0 - Cg[-1]), 1e-4)
    efficiency = 100*(1 - (G*Cg[-1])/(G*Cg0))

    # --- CSTR Train ---
    V = V_total/N
    F = L
    tau = V/F
    Na2CO3_in = CO2_abs
    NaOH_in = 0.0
    CaOH2_in = Na2CO3_in*1.05

    tspan = np.linspace(0, 4*tau, 200)
    Na2CO3_hist = []
    NaOH_hist = []
    CaOH2_hist = []
    conv_hist = []

    for i in range(N):
        def cstr(t,y):
            Na2CO3, NaOH, CaOH2 = y
            r = k_caus*Na2CO3*(1 - NaOH/(NaOH+Na2CO3+1))
            r = min(r, eta_eq*Na2CO3/tau)
            return [-r, 2*r, -r]
        sol = solve_ivp(cstr,[0,tspan[-1]],[Na2CO3_in,NaOH_in,CaOH2_in], t_eval=tspan)
        Na2CO3_in, NaOH_in, CaOH2_in = sol.y[:,-1]
        Na2CO3_hist.append(sol.y[0])
        NaOH_hist.append(sol.y[1])
        CaOH2_hist.append(sol.y[2])
        conv_hist.append((1 - sol.y[0]/Na2CO3_hist[0][0])*100)

    return z_eval, Cg, Cl, NaOH, CO2_abs, efficiency, tspan, Na2CO3_hist, NaOH_hist, CaOH2_hist, conv_hist

# Run cached computation
(z_eval, Cg, Cl, NaOH, CO2_abs, efficiency,
 tspan, Na2CO3_hist, NaOH_hist, CaOH2_hist, conv_hist) = run_models(D,H,G,L,C_NaOH0,V_total,N,k_caus,eta_eq)

# ===================== ECONOMICS =====================
SEC_PER_YEAR = 365 * 24 * 3600
capacity_factor = 0.85

# Annual CO2 captured
CO2_mol_s = CO2_abs * capacity_factor
CO2_tpy = CO2_mol_s * 44.01 / 1000 * SEC_PER_YEAR

# Installed CAPEX
absorber_cost = 18000 * (A * H)**0.62
causticizer_cost = 22000 * V_total**0.6
bare_CAPEX = absorber_cost + causticizer_cost
CAPEX = bare_CAPEX * 3.2 * 1.15

# Pump electricity
pump_eff = 0.7
deltaP = 1.5e5
pump_power = (deltaP * L) / pump_eff
pump_cost = pump_power * SEC_PER_YEAR / 3.6e6 * elec_price

# Lime cost
CaOH2_mol_s = CO2_mol_s
CaOH2_tpy = CaOH2_mol_s * 74.1 / 1000 * SEC_PER_YEAR / 1000
lime_cost = CaOH2_tpy * lime_price

# Fixed O&M
fixed_OM = 0.045 * CAPEX

# Compression + MRV
compression_cost = 25 * CO2_tpy

# Total costs
OPEX = pump_cost + lime_cost + fixed_OM + compression_cost
annual_CAPEX = 0.10 * CAPEX
annual_cost = annual_CAPEX + OPEX
cost_per_t = annual_cost / CO2_tpy

# ===================== METRICS =====================
st.subheader("💨 Bubble Column Performance")
st.metric("CO₂ Capture Efficiency (%)", f"{efficiency:.1f}")
st.metric("CO₂ Captured Annually (t/year)", f"{CO2_tpy:,.0f}")
st.metric("Total CAPEX ($)", f"{CAPEX:,.0f}")
st.metric("Annual OPEX ($/year)", f"{OPEX:,.0f}")
st.metric("Cost of CO₂ Capture ($/ton)", f"${cost_per_t:,.0f}")

# ===================== OPTIONAL ANIMATIONS =====================
if run_animation:
    st.subheader("🎬 Bubble Column Animation")
    placeholder = st.empty()
    for frame in range(1, len(z_eval), 3):
        fig, ax = plt.subplots(figsize=(7,5))
        ax.plot(Cg[:frame]/Cg[0]*100, z_eval[:frame], 'b-', lw=3, label='Gas CO₂ (%)')
        ax.plot(Cl[:frame]/Cl.max()*100, z_eval[:frame], 'r--', lw=2, label='Liquid CO₂ (%)')
        ax.plot(NaOH[:frame]/NaOH[0]*100, z_eval[:frame], 'g-.', lw=2, label='NaOH (%)')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, H)
        ax.set_xlabel("Relative concentration (%)")
        ax.set_ylabel("Column height (m)")
        ax.set_title("Bubble Column Dynamics")
        ax.legend()
        ax.grid(True)
        placeholder.pyplot(fig)
        plt.close(fig)

    st.subheader("🎬 CSTR Train Conversion Animation")
    placeholder_cstr = st.empty()
    colors = ['b','r','g','m','c']
    for frame in range(1, len(tspan), 4):
        fig2, ax2 = plt.subplots(figsize=(8,5))
        for i in range(N):
            ax2.plot(tspan[:frame]/60, Na2CO3_hist[i][:frame], color=colors[i%5], lw=2, label=f'Na₂CO₃ CSTR {i+1}')
            ax2.plot(tspan[:frame]/60, NaOH_hist[i][:frame], color=colors[i%5], lw=2, ls='--', label=f'NaOH CSTR {i+1}')
        ax2.set_xlim(0, tspan[-1]/60)
        ax2.set_ylim(0, max(max(Na2CO3_hist[0]), max(NaOH_hist[0]))*1.1)
        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel("Molar flow (mol/s)")
        ax2.set_title("CSTR Concentrations Over Time")
        ax2.legend(ncol=2, fontsize=9)
        ax2.grid(True)
        placeholder_cstr.pyplot(fig2)
        plt.close(fig2)

# ===================== FINAL PLOTS =====================
st.subheader("📊 Bubble Column Profiles")
fig3, ax3 = plt.subplots(figsize=(7,5))
ax3.plot(Cg/Cg[0]*100, z_eval, 'b-', lw=3, label='Gas CO₂ (%)')
ax3.plot(Cl/Cl.max()*100, z_eval, 'r--', lw=2, label='Liquid CO₂ (%)')
ax3.plot(NaOH/NaOH[0]*100, z_eval, 'g-.', lw=2, label='NaOH (%)')
ax3.set_xlabel("Relative concentration (%)")
ax3.set_ylabel("Column height (m)")
ax3.set_title("Final Bubble Column Profiles")
ax3.legend()
ax3.grid(True)
st.pyplot(fig3)

cumulative_CO2 = G*(Cg[0] - Cg)
fig4, ax4 = plt.subplots(figsize=(7,5))
ax4.plot(cumulative_CO2, z_eval, 'm-', lw=3)
ax4.set_xlabel("Cumulative CO₂ captured (mol/s)")
ax4.set_ylabel("Column height (m)")
ax4.set_title("Cumulative CO₂ Capture Along Column Height")
ax4.grid(True)
st.pyplot(fig4)

st.subheader("📊 CSTR Conversion and Product Formation")
fig5, ax5 = plt.subplots(figsize=(8,5))
for i in range(N):
    ax5.plot(tspan/60, conv_hist[i], lw=2, label=f'CSTR {i+1} Conversion (%)')
ax5.set_xlabel("Time (min)")
ax5.set_ylabel("Conversion (%)")
ax5.set_title("CSTR Stage Conversions Over Time")
ax5.legend()
ax5.grid(True)
st.pyplot(fig5)

fig6, ax6 = plt.subplots(figsize=(8,5))
for i in range(N):
    ax6.plot(tspan/60, Na2CO3_hist[i], lw=2, label=f'Na₂CO₃ CSTR {i+1}')
    ax6.plot(tspan/60, NaOH_hist[i], lw=2, ls='--', label=f'NaOH CSTR {i+1}')
ax6.set_xlabel("Time (min)")
ax6.set_ylabel("Molar flow (mol/s)")
ax6.set_title("Na₂CO₃ and NaOH Production in CSTR Stages")
ax6.legend()
ax6.grid(True)
st.pyplot(fig6)
