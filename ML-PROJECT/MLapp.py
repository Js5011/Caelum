import streamlit as st
from simulation import run_full_model

st.set_page_config(page_title="CO₂ Capture Simulator", layout="wide")
st.title("🏭 CO₂ Capture & Causticizing Simulator")

st.sidebar.header("Bubble Column")
D = st.sidebar.slider("Absorber Diameter (m)", 1,5,2,0.1)
H = st.sidebar.slider("Absorber Height (m)", 6,20,12,0.5)
G = st.sidebar.slider("Gas Flow Rate (m³/s)", 0.1,1.0,0.4,0.01)
L = st.sidebar.slider("Liquid Flow Rate (m³/s)", 0.2,1.2,0.6,0.01)
C_NaOH0 = st.sidebar.slider("NaOH Concentration (mol/m³)",700,1800,1000,10)

st.sidebar.header("CSTR Reactor")
V_total = st.sidebar.slider("Total Volume (m³)", 5,50,20,1)
N = st.sidebar.slider("Number of CSTRs",1,6,3)
k_caus = st.sidebar.slider("Causticizing Rate (1/s)",0.1,1.0,0.5,0.01)
eta_eq = st.sidebar.slider("Equilibrium Conversion",0.5,1.0,0.85,0.01)

st.sidebar.header("Economics")
elec_price = st.sidebar.number_input("Electricity ($/kWh)",0.05,0.5,0.1)
lime_price = st.sidebar.number_input("Lime Price ($/ton)",50,500,150)

if st.button("▶ Run Simulation"):
    CO2_tpy, cost_per_t, efficiency = run_full_model(D,H,G,L,0.12,C_NaOH0,
                                                    V_total,N,k_caus,eta_eq,
                                                    elec_price,lime_price)
    st.metric("CO₂ Capture Efficiency (%)", f"{efficiency:.1f}")
    st.metric("CO₂ Captured (t/year)", f"{CO2_tpy:,.0f}")
    st.metric("Cost of Capture ($/tCO₂)", f"{cost_per_t:,.0f}")
