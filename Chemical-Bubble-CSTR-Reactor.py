import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# ===================== CONSTANTS =====================
R = 8.314             # J/(mol·K)
T = 298               # K
P = 101325            # Pa
SEC_PER_YEAR = 365*24*3600

# ===================== USER INPUTS =====================
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

def absorber(z, y, vG_val, vL_val, kLa_val, C_NaOH):
    Cg, Cl, NaOH = y
    P_CO2 = Cg*R*T
    C_star = P_CO2/H_CO2
    N_mt = kLa_val*(C_star - Cl)
    r_rxn = k_rxn * Cl * (NaOH / (NaOH + 1000))
    dCgdz = -N_mt/vG_val
    dCldz = (N_mt - r_rxn)/vL_val
    dNaOHdz = -2*r_rxn/vL_val
    return [dCgdz, dCldz, dNaOHdz]

z_eval = np.linspace(0,H,300)

# ===================== CSTR MODEL =====================
def run_cstr(CO2_abs_val):
    V = V_total/N
    F = L
    tau = V/F
    Na2CO3_in = CO2_abs_val
    NaOH_in = 0.0
    CaOH2_in = CO2_abs_val*1.05
    Na2CO3_hist = []
    NaOH_hist = []
    CaOH2_hist = []
    conv_hist = []

    tspan = np.linspace(0,4*tau,300)
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
        conv_hist.append((1 - sol.y[0]/CO2_abs_val)*100)
    return Na2CO3_hist, NaOH_hist, CaOH2_hist, conv_hist, tspan

# ===================== COST FUNCTION =====================
def compute_cost(CO2_abs_val, CaOH2_final, H_val, G_val, elec_val):
    absorber_cost_val = 12000*(A*H_val)**0.6
    causticizer_cost_val = 15000*V_total**0.6
    CAPEX_val = absorber_cost_val + causticizer_cost_val
    pump_power_val = (1.5e5*L)/0.7
    pump_cost_val = pump_power_val*SEC_PER_YEAR/3.6e6*elec_val
    lime_ton_val = CaOH2_final*74.1/1e6*SEC_PER_YEAR
    lime_cost_val = lime_ton_val*lime_price
    OPEX_val = max(pump_cost_val + lime_cost_val, 0.07*CAPEX_val)
    annual_cost_val = 0.1*CAPEX_val + OPEX_val
    CO2_tpy_val = CO2_abs_val*44.01/1000*SEC_PER_YEAR
    cost_per_t_val = annual_cost_val/CO2_tpy_val
    return CO2_tpy_val, cost_per_t_val

# ===================== FULL MODEL FUNCTION =====================
def run_full_model(H_val, G_val, C_NaOH_val, elec_val):
    vG_val = G_val/A
    vL_val = L/A
    kLa_val = 0.2 * (vG_val / 0.1)**0.7
    sol_abs = solve_ivp(lambda z, y: absorber(z, y, vG_val, vL_val, kLa_val, C_NaOH_val),
                        [0,H_val],[Cg0,0,C_NaOH_val], t_eval=z_eval)
    Cg_end = sol_abs.y[0,-1]
    CO2_abs_val = max(G_val*(Cg0 - Cg_end),1e-4)
    Na2CO3_hist, NaOH_hist, CaOH2_hist, conv_hist, tspan = run_cstr(CO2_abs_val)
    CO2_tpy_val, cost_val = compute_cost(CO2_abs_val, CaOH2_hist[-1], H_val, G_val, elec_val)
    return CO2_tpy_val, cost_val

# ===================== RUN BASELINE =====================
CO2_tpy, cost_per_t = run_full_model(H, G, C_NaOH0, elec_price)
print(f"\nBaseline CO₂ captured: {CO2_tpy:,.0f} t/year")
print(f"Baseline cost of capture: ${cost_per_t:,.0f}/tCO₂")

# ===================== SENSITIVITY ANALYSIS =====================
sensitivity_ranges = {
    "H": np.linspace(0.7*H,1.3*H,5),
    "G": np.linspace(0.7*G,1.3*G,5),
    "C_NaOH0": np.linspace(0.7*C_NaOH0,1.3*C_NaOH0,5),
    "elec_price": np.linspace(0.7*elec_price,1.3*elec_price,5)
}

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

# ===================== MONTE CARLO UNCERTAINTY =====================
N_MC = 3000
H_vals = np.random.uniform(0.7*H,1.3*H,N_MC)
G_vals = np.random.uniform(0.7*G,1.3*G,N_MC)
C_NaOH_vals = np.random.uniform(0.7*C_NaOH0,1.3*C_NaOH0,N_MC)
elec_vals = np.random.uniform(0.7*elec_price,1.3*elec_price,N_MC)

cost_MC = []
CO2_MC = []

for i in range(N_MC):
    CO2_tpy_val, cost_val = run_full_model(H_vals[i], G_vals[i], C_NaOH_vals[i], elec_vals[i])
    CO2_MC.append(CO2_tpy_val)
    cost_MC.append(cost_val)

df_MC = pd.DataFrame({"Cost_per_t": cost_MC, "CO2_tpy": CO2_MC})

# Histogram of cost
plt.figure(figsize=(8,6))
plt.hist(df_MC["Cost_per_t"], bins=50, color='skyblue', edgecolor='black')
plt.xlabel("Cost ($/t CO₂)")
plt.ylabel("Frequency")
plt.title("Monte Carlo Simulation - Cost per ton CO₂")
plt.grid(True)
plt.show()

# Monte Carlo statistics
mean_cost = df_MC["Cost_per_t"].mean()
median_cost = df_MC["Cost_per_t"].median()
min_cost = df_MC["Cost_per_t"].min()
max_cost = df_MC["Cost_per_t"].max()
prob_under_100 = np.mean(df_MC["Cost_per_t"] < 100)*100

print(f"\nMonte Carlo Results:")
print(f"Mean cost: ${mean_cost:.1f}/tCO₂")
print(f"Median cost: ${median_cost:.1f}/tCO₂")
print(f"Min cost: ${min_cost:.1f}/tCO₂")
print(f"Max cost: ${max_cost:.1f}/tCO₂")
print(f"Probability cost < $100/t: {prob_under_100:.1f}%")
