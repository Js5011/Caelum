import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

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
kLa = 0.2 * (vG / 0.1)**0.7
k_rxn = 5000
H_CO2 = 3.4e4

Cg0 = yCO2*P/(R*T)
Cl0 = 0.0

def absorber(z, y):
    Cg, Cl, NaOH = y
    P_CO2 = Cg*R*T
    C_star = P_CO2/H_CO2
    N_mt = kLa*(C_star - Cl)
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
print(f"\nBubble column CO₂ capture efficiency: {efficiency:.1f}%")

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

print(f"\nCO₂ captured: {CO2_tpy:,.0f} t/year")
print(f"Cost of capture: ${cost_per_t:,.0f}/tCO₂")

# ===================== SENSITIVITY ANALYSIS =====================
baseline = {
    "D": D, "H": H, "G": G, "yCO2": yCO2, "L": L, "C_NaOH0": C_NaOH0,
    "V_total": V_total, "N": N, "k_caus": k_caus, "eta_eq": eta_eq,
    "elec_price": elec_price, "lime_price": lime_price
}

# Define variable ranges for sensitivity
sensitivity_ranges = {
    "H": np.linspace(0.5*H, 1.5*H, 5),
    "G": np.linspace(0.7*G, 1.3*G, 5),
    "C_NaOH0": np.linspace(0.5*C_NaOH0, 2*C_NaOH0, 5),
    "elec_price": np.linspace(0.5*elec_price, 1.5*elec_price, 5)
}

def run_full_model(H_val, G_val, C_NaOH_val, elec_val):
    # Update parameters
    vG_val = G_val/A
    vL_val = L/A
    kLa_val = 0.2 * (vG_val / 0.1)**0.7

    # Bubble column
    def absorber_local(z, y):
        Cg, Cl, NaOH = y
        P_CO2 = Cg*R*T
        C_star = P_CO2/H_CO2
        N_mt = kLa_val*(C_star - Cl)
        r_rxn = k_rxn * Cl * (NaOH / (NaOH + 1000))
        dCgdz = -N_mt/vG_val
        dCldz = (N_mt - r_rxn)/vL_val
        dNaOHdz = -2*r_rxn/vL_val
        return [dCgdz, dCldz, dNaOHdz]

    sol_abs = solve_ivp(absorber_local,[0,H_val],[Cg0,0,C_NaOH_val], t_eval=z_eval)
    Cg_end = sol_abs.y[0,-1]
    CO2_abs_val = max(G_val*(Cg0 - Cg_end),1e-4)

    # CSTRs (simplified)
    Na2CO3_in = CO2_abs_val
    NaOH_in = 0.0
    CaOH2_in = CO2_abs_val*1.05
    for i in range(N):
        def cstr_local(t,y):
            Na2CO3, NaOH, CaOH2 = y
            r = k_caus*Na2CO3*(1 - NaOH/(NaOH+Na2CO3+1))
            r = min(r, eta_eq*Na2CO3/tau)
            return [-r, 2*r, -r]
        sol = solve_ivp(cstr_local,[0,tspan[-1]],[Na2CO3_in,NaOH_in,CaOH2_in], t_eval=[tspan[-1]])
        Na2CO3_in, NaOH_in, CaOH2_in = sol.y[:,-1]

    # Costs
    absorber_cost_val = 12000*(A*H_val)**0.6
    causticizer_cost_val = 15000*V_total**0.6
    CAPEX_val = absorber_cost_val + causticizer_cost_val
    pump_power_val = (1.5e5*L)/0.7
    pump_cost_val = pump_power_val*SEC_PER_YEAR/3.6e6*elec_val
    lime_ton_val = CaOH2_in*74.1/1e6*SEC_PER_YEAR
    lime_cost_val = lime_ton_val*lime_price
    OPEX_val = max(pump_cost_val + lime_cost_val, 0.07*CAPEX_val)
    annual_cost_val = 0.1*CAPEX_val + OPEX_val
    CO2_tpy_val = CO2_abs_val*44.01/1000*SEC_PER_YEAR
    cost_per_t_val = annual_cost_val/CO2_tpy_val

    return CO2_tpy_val, cost_per_t_val

# Run sensitivity analysis
results = []
for var, values in sensitivity_ranges.items():
    for v in values:
        H_val = H if var!="H" else v
        G_val = G if var!="G" else v
        C_val = C_NaOH0 if var!="C_NaOH0" else v
        elec_val = elec_price if var!="elec_price" else v
        CO2_tpy_val, cost_val = run_full_model(H_val, G_val, C_val, elec_val)
        results.append({"Variable": var, "Value": v, "CO2_tpy": CO2_tpy_val, "Cost_per_t": cost_val})

df_sens = pd.DataFrame(results)

# ===================== PLOTS =====================
# Spider plot
plt.figure(figsize=(8,6))
for var in sensitivity_ranges.keys():
    subset = df_sens[df_sens["Variable"]==var]
    plt.plot(subset["Value"], subset["Cost_per_t"], lw=2, marker='o', label=var)
plt.xlabel("Variable value")
plt.ylabel("Cost ($/t CO₂)")
plt.title("Sensitivity Analysis - Cost vs Variable")
plt.legend()
plt.grid(True)
plt.show()

# Tornado plot
tornado = []
for var in sensitivity_ranges.keys():
    subset = df_sens[df_sens["Variable"]==var]
    delta = subset["Cost_per_t"].max() - subset["Cost_per_t"].min()
    tornado.append((var, delta))

tornado_df = pd.DataFrame(tornado, columns=["Variable","Cost Impact"]).sort_values("Cost Impact")
plt.figure(figsize=(6,4))
plt.barh(tornado_df["Variable"], tornado_df["Cost Impact"], color='skyblue')
plt.xlabel("Δ Cost ($/t CO₂)")
plt.title("Tornado Plot - Cost Sensitivity")
plt.grid(axis='x')
plt.show()
