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
vG = G/A
vL = L/A

# ===================== BUBBLE COLUMN MODEL =====================
# ===================== BUBBLE COLUMN MODEL FIXED =====================
kLa = 0.2 * (vG / 0.1)**0.7     # increase kLa to typical industrial values
k_rxn = 5000                    # reaction coefficient (mol/m3/s)
H_CO2 = 3.4e4                   # Henry constant (Pa·m³/mol)

Cg0 = yCO2*P/(R*T)
Cl0 = 0.0

def absorber(z, y):
    Cg, Cl, NaOH = y
    P_CO2 = Cg*R*T
    C_star = P_CO2/H_CO2
    N_mt = kLa*(C_star - Cl)
    
    # reaction limited by NaOH, remove artificial cap
    r_rxn = k_rxn * Cl * (NaOH / (NaOH + 1000))
    
    dCgdz = -N_mt/vG
    dCldz = (N_mt - r_rxn)/vL
    dNaOHdz = -2*r_rxn/vL
    return [dCgdz, dCldz, dNaOHdz]

z_eval = np.linspace(0,H,300)
sol_abs = solve_ivp(absorber,[0,H],[Cg0,Cl0,C_NaOH0], t_eval=z_eval)
Cg, Cl, NaOH = sol_abs.y

CO2_abs = max(G*(Cg0 - Cg[-1]), 1e-4)
CO2_out = G*Cg[-1]
efficiency = 100*(1 - CO2_out/(G*Cg0))
print(f"Bubble column CO₂ capture efficiency: {efficiency:.1f}%")


# ===================== CSTR TRAIN =====================
V = V_total/N
F = L
tau = V/F
Na2CO3_in = CO2_abs
NaOH_in = 0.0
CaOH2_in = CO2_abs*1.05

tspan = np.linspace(0,4*tau,300)
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
    conv_hist.append((1 - sol.y[0]/CO2_abs)*100)

# ===================== COSTS =====================
absorber_cost = 12000*(A*H)**0.6
causticizer_cost = 15000*V_total**0.6
CAPEX = absorber_cost + causticizer_cost
pump_power = (1.5e5*L)/0.7
pump_cost = pump_power*SEC_PER_YEAR/3.6e6*elec_price
lime_ton = CaOH2_in*74.1/1e6*SEC_PER_YEAR
lime_cost = lime_ton*lime_price
OPEX = max(pump_cost + lime_cost, 0.07*CAPEX)
annual_cost = 0.1*CAPEX + OPEX
CO2_tpy = CO2_abs*44.01/1000*SEC_PER_YEAR
cost_per_t = annual_cost/CO2_tpy

# ===================== OUTPUT =====================
print(f"\nCO₂ capture efficiency: {100*(1 - CO2_out/(G*Cg0)):.1f}%")
print(f"CO₂ captured: {CO2_tpy:,.0f} t/year")
print(f"Cost of capture: ${cost_per_t:,.0f}/tCO₂")

# ===================== VISUALS =====================
plt.style.use('ggplot')   # Safe built-in style

# Bubble column profiles
plt.figure(figsize=(10,6))
plt.plot(Cg/Cg0*100, z_eval, 'b-', lw=3, label='Gas CO₂ (%)')
plt.plot(Cl/Cl.max()*100, z_eval, 'r--', lw=2, label='Liquid CO₂ (%)')
plt.plot(NaOH/C_NaOH0*100, z_eval, 'g-.', lw=2, label='NaOH remaining (%)')
plt.xlabel("Relative concentration (%)")
plt.ylabel("Height (m)")
plt.title("Bubble Column Concentration Profiles")
plt.legend(loc='center right')
plt.grid(True)
plt.show()

# CO₂ removal fraction along column
plt.figure(figsize=(8,5))
plt.plot((Cg0-Cg)/Cg0*100, z_eval, color='darkorange', lw=3)
plt.xlabel("Cumulative CO₂ removed (%)")
plt.ylabel("Column height (m)")
plt.title("CO₂ Removal Along Bubble Column")
plt.grid(True)
plt.show()

# CSTR dynamics over time
plt.figure(figsize=(10,6))
colors = ['b','r','g','m','c']
for i in range(N):
    plt.plot(tspan/60, Na2CO3_hist[i], color=colors[i%5], lw=2, label=f'Na₂CO₃ CSTR {i+1}')
    plt.plot(tspan/60, NaOH_hist[i], color=colors[i%5], lw=2, ls='--', label=f'NaOH CSTR {i+1}')
    plt.plot(tspan/60, CaOH2_hist[i], color=colors[i%5], lw=2, ls=':', label=f'Ca(OH)₂ CSTR {i+1}')
plt.xlabel("Time (min)")
plt.ylabel("Molar flow (mol/s)")
plt.title("CSTR Concentrations Over Time")
plt.legend(ncol=2, fontsize=9)
plt.grid(True)
plt.show()

# CSTR conversion fraction
plt.figure(figsize=(8,5))
for i in range(N):
    plt.plot(tspan/60, conv_hist[i], lw=2, label=f'CSTR {i+1}')
plt.xlabel("Time (min)")
plt.ylabel("Conversion (%)")
plt.title("CSTR Na₂CO₃ Conversion Over Time")
plt.legend()
plt.grid(True)
plt.show()

# CO₂ mass balance
plt.figure(figsize=(6,5))
plt.bar(["Inlet","Captured","Outlet"], [G*Cg0, CO2_abs, CO2_out], color=['blue','green','red'])
plt.ylabel("Molar flow (mol/s)")
plt.title("CO₂ Mass Balance")
plt.grid(axis='y')
plt.show()

# CAPEX/OPEX breakdown
plt.figure(figsize=(6,6))
plt.pie([0.1*CAPEX,pump_cost,lime_cost], labels=["Annualized CAPEX","Pumping","Lime"], autopct="%1.1f%%", colors=['gold','skyblue','lightgreen'])
plt.title("Annual Cost Breakdown")
plt.show()

# Final CSTR stage concentrations
plt.figure(figsize=(8,5))
plt.plot(range(1,N+1), [h[-1] for h in NaOH_hist], 'b-o', label='NaOH')
plt.plot(range(1,N+1), [h[-1] for h in CaOH2_hist], 'r-s', label='Ca(OH)₂')
plt.plot(range(1,N+1), [h[-1] for h in Na2CO3_hist], 'g-^', label='Na₂CO₃')
plt.xlabel("CSTR Stage")
plt.ylabel("Final Concentration (mol/s)")
plt.title("Final Concentrations per CSTR Stage")
plt.legend()
plt.grid(True)
plt.show()
