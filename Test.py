import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ===================== CONSTANTS =====================
R = 8.314
T = 298
P = 101325
SEC_PER_YEAR = 365*24*3600

# ===================== USER INPUTS =====================
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

# ===================== GEOMETRY =====================
A = np.pi*(D/2)**2
vG = G / A
vL = L / A

if not (0.02 <= vG <= 0.4):
    raise ValueError(f"Gas velocity {vG:.2f} m/s outside bubble-column limits (0.02–0.4 m/s). Reduce G or increase D.")

# ===================== BUBBLE COLUMN MODEL =====================
kLa = 0.12 * (vG / 0.1)**0.6
k_rxn = 3e3
H_CO2 = 3.4e4

Cg0 = yCO2*P/(R*T)
Cl0 = 0.0

def absorber(z, y):
    Cg, Cl, NaOH = y
    P_CO2 = Cg*R*T
    C_star = P_CO2/H_CO2
    N_mt = kLa*(C_star - Cl)
    r_rxn = k_rxn*Cl*(NaOH/(NaOH+500)) # saturation
    dCgdz = -N_mt/vG
    dCldz = (N_mt - r_rxn)/vL
    dNaOHdz = -2*r_rxn/vL
    return [dCgdz, dCldz, dNaOHdz]

z_eval = np.linspace(0, H, 300)
sol_abs = solve_ivp(absorber, [0,H],[Cg0,Cl0,C_NaOH0], t_eval=z_eval)
Cg, Cl, NaOH = sol_abs.y
CO2_abs = G*(Cg0 - Cg[-1])
CO2_abs = max(CO2_abs, 1e-6)

# ===================== CSTR TRAIN =====================
V = V_total/N
F = L
tau = V/F
Na2CO3_in = CO2_abs
NaOH_in = 0.0
CaOH2_in = CO2_abs*1.1

tspan = np.linspace(0, 4*tau, 300)
Na2CO3_hist = []
NaOH_hist = []
CaOH2_hist = []

for _ in range(N):
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

# ===================== COSTS =====================
absorber_cost = 12000*(A*H)**0.6
causticizer_cost = 15000*V_total**0.6
CAPEX = absorber_cost+causticizer_cost
pump_power = (1.5e5*L)/0.7
pump_cost = pump_power*SEC_PER_YEAR/3.6e6*elec_price
lime_ton = CaOH2_in*74.1/1e6*SEC_PER_YEAR
lime_cost = lime_ton*lime_price
OPEX = max(pump_cost+lime_cost,0.07*CAPEX)
annual_cost = 0.1*CAPEX + OPEX
CO2_tpy = CO2_abs*44.01/1000*SEC_PER_YEAR
cost_per_t = annual_cost/CO2_tpy

# ===================== OUTPUT =====================
print(f"\nCO₂ capture efficiency: {100*(1-Cg[-1]/Cg0):.1f}%")
print(f"CO₂ captured: {CO2_tpy:,.0f} t/year")
print(f"Cost of capture: ${cost_per_t:,.0f}/tCO₂")

# ===================== VISUALS =====================

# Bubble column profiles
plt.figure(figsize=(8,6))
plt.plot(Cg/Cg0*100, z_eval, label="Gas CO₂ (%)", linewidth=2)
plt.plot(Cl/Cl.max()*100, z_eval, label="Liquid CO₂ (%)", linewidth=2)
plt.plot(NaOH/C_NaOH0*100, z_eval, label="NaOH remaining (%)", linewidth=2)
plt.xlabel("Relative concentration (%)")
plt.ylabel("Height (m)")
plt.title("Bubble Column Axial Profiles")
plt.legend()
plt.grid()
plt.show()

# CSTR concentration dynamics over time
plt.figure(figsize=(8,5))
for i in range(N):
    plt.plot(tspan/60, Na2CO3_hist[i], label=f"Na2CO3 CSTR {i+1}")
    plt.plot(tspan/60, NaOH_hist[i], '--', label=f"NaOH CSTR {i+1}")
    plt.plot(tspan/60, CaOH2_hist[i], ':', label=f"Ca(OH)₂ CSTR {i+1}")
plt.xlabel("Time (min)")
plt.ylabel("Concentration (mol/s)")
plt.title("CSTR Concentration Dynamics Over Time")
plt.legend()
plt.grid()
plt.show()

# Mass balance bar chart
plt.figure(figsize=(6,5))
plt.bar(["Inlet","Captured","Outlet"], [G*Cg0, CO2_abs, G*Cg[-1]])
plt.ylabel("CO₂ molar flow (mol/s)")
plt.title("CO₂ Mass Balance")
plt.grid(axis='y')
plt.show()

# Cost breakdown
plt.figure(figsize=(6,6))
plt.pie([0.1*CAPEX,pump_cost,lime_cost], labels=["Annualized CAPEX","Pumping","Lime"], autopct="%1.1f%%")
plt.title("Annual Cost Breakdown")
plt.show()

# Additional: bubble column CO₂ removal fraction along height
plt.figure(figsize=(7,5))
plt.plot((Cg0-Cg)/Cg0*100, z_eval, color='green', linewidth=2)
plt.xlabel("CO₂ removed (%)")
plt.ylabel("Column height (m)")
plt.title("Cumulative CO₂ Removal Along Column")
plt.grid()
plt.show()

# Additional: CSTR NaOH vs CaOH2 evolution per stage at final time
plt.figure(figsize=(7,5))
plt.plot(range(1,N+1), [h[-1] for h in NaOH_hist], marker='o', label='NaOH')
plt.plot(range(1,N+1), [h[-1] for h in CaOH2_hist], marker='s', label='Ca(OH)₂')
plt.xlabel("CSTR Stage")
plt.ylabel("Concentration (mol/s)")
plt.title("Final CSTR Stage Concentrations")
plt.legend()
plt.grid()
plt.show()
