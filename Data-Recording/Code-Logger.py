# ============================================================
# FULL STABLE CO2 CAPTURE + CAUSTICIZING MODEL
# WITH SAFE DEFAULTS + TXT + CSV LOGGING
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import csv
import os
from datetime import datetime

# =================== LOGGING SETUP ==========================

MASTER_LOG = "carbon_capture_all_runs.txt"
CSV_FILE   = "carbon_capture_all_runs.csv"

# Determine run number
if os.path.exists(MASTER_LOG):
    with open(MASTER_LOG, "r") as f:
        run_number = f.read().count("RUN #") + 1
else:
    run_number = 1

log_file = open(MASTER_LOG, "a")

def log(msg):
    print(msg)
    log_file.write(msg + "\n")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

log("\n" + "="*70)
log(f" RUN #{run_number}")
log(f" Timestamp: {timestamp}")
log("="*70 + "\n")

# CSV setup
csv_exists = os.path.exists(CSV_FILE)
csv_file = open(CSV_FILE, "a", newline="")
csv_writer = csv.writer(csv_file)

if not csv_exists:
    csv_writer.writerow([
        "Run","Timestamp",
        "D","G","y_CO2","NaOH_in","L",
        "CO2_Capture_Efficiency",
        "NaOH_Remaining_Absorber",
        "Na2CO3_Outlet_Absorber",
        "Total_Na2CO3_Absorber",
        "Na2CO3_Outlet_CSTR",
        "NaOH_Outlet_CSTR",
        "CaCO3_Produced",
        "Overall_Carbon_Capture"
    ])

# =================== DEFAULT INPUTS =========================
# Absorber
D      = 3.0        # m
G      = 12.0       # m³/s
y_CO2  = 0.12       # mole fraction
NaOH_in = 1500.0    # mol/m³
L      = 1.0        # m³/s

log("===== ABSORBER INPUTS =====")
log(f"D = {D} m")
log(f"G = {G} m3/s")
log(f"y_CO2 = {y_CO2}")
log(f"NaOH_in = {NaOH_in} mol/m3")
log(f"L = {L} m3/s")

# CSTR train
V_total = 50.0       # m³
N       = 5          # number of CSTRs
F       = 1.0        # m³/s
CaOH2_0 = 300.0      # mol/m³
k       = 0.001      # m/s
a_s0    = 200.0      # m²/m³
eta_eq  = 0.95
t_final = 2000.0     # s

log("\n===== CAUSTICIZING INPUTS =====")
log(f"V_total = {V_total}")
log(f"N = {N}")
log(f"F = {F}")
log(f"CaOH2_0 = {CaOH2_0}")
log(f"k = {k}")
log(f"a_s0 = {a_s0}")
log(f"eta_eq = {eta_eq}")
log(f"t_final = {t_final}")

# =================== ABSORBER SETUP =========================
H_sim = 10.0
N_z = 800
dz = H_sim / N_z
z = np.linspace(0, H_sim, N_z)

A = np.pi * (D/2)**2
vL = L / A
vG = max(G / A, 0.05)

R = 8.314
T = 298
P_total = 101325
H_CO2 = 3.4e4
kLa0 = 0.08
k_rxn = 1.0
n_rxn = 1
D_axial_g = 1e-5
D_axial_l = 1e-9

def activity_coeff(NaOH):
    return 1 / (1 + NaOH / 2000)

def Henry_CO2_T(T):
    return H_CO2 * (1 + 0.01 * (T - 298))

# INITIAL CONDITIONS
C_CO2g = np.ones(N_z) * (y_CO2 * P_total / (R * T))
C_CO2l = np.zeros(N_z)
C_NaOH = np.ones(N_z) * NaOH_in
C_Na2CO3 = np.zeros(N_z)

alpha = 0.3   # smaller under-relaxation for stability
tol = 1e-8
max_iter = 20000

# =================== ABSORBER ITERATION =====================
for _ in range(max_iter):
    old = np.vstack([C_CO2g, C_CO2l, C_NaOH, C_Na2CO3]).copy()

    for i in range(N_z):
        g_up = C_CO2g[i-1] if i>0 else C_CO2g[0]
        l_up = C_CO2l[i-1] if i>0 else 0
        naoh_up = C_NaOH[i-1] if i>0 else NaOH_in
        na2_up = C_Na2CO3[i-1] if i>0 else 0

        H_eff = Henry_CO2_T(T) / activity_coeff(naoh_up)
        P_CO2 = g_up * R * T
        C_star = P_CO2 / H_eff

        kLa_eff = kLa0 * (1 + 5*naoh_up/(naoh_up+2000))
        N_CO2 = kLa_eff * (C_star - l_up)
        dCO2_g = min(N_CO2*dz/vG, g_up)
        new_g = g_up - dCO2_g

        dCO2_l = dCO2_g * (vG/vL)
        new_l = min(l_up + dCO2_l, C_star)

        r_rxn = k_rxn * new_l * naoh_up
        dNaOH = min(2*r_rxn*dz/vL, naoh_up*0.95)  # cap to 95% of NaOH

        new_naoh = naoh_up - dNaOH
        new_na2 = na2_up + dNaOH / 2

        # Update with under-relaxation
        C_CO2g[i]   = max(alpha*new_g + (1-alpha)*C_CO2g[i], 0)
        C_CO2l[i]   = max(alpha*new_l + (1-alpha)*C_CO2l[i], 0)
        C_NaOH[i]   = max(alpha*new_naoh + (1-alpha)*C_NaOH[i], 0)
        C_Na2CO3[i] = max(alpha*new_na2 + (1-alpha)*C_Na2CO3[i], 0)

    if np.max(np.abs(old - np.vstack([C_CO2g, C_CO2l, C_NaOH, C_Na2CO3]))) < tol:
        break

capture = 1 - C_CO2g[-1]/C_CO2g[0]
NaOH_out = C_NaOH[-1]
Na2CO3_out = C_Na2CO3[-1]
total_Na2CO3 = Na2CO3_out*A*H_sim

log("\n===== ABSORBER RESULTS =====")
log(f"CO2 capture efficiency: {capture:.3f}")
log(f"NaOH remaining: {NaOH_out:.2f} mol/m3")
log(f"Na2CO3 outlet: {Na2CO3_out:.2f} mol/m3")
log(f"Total Na2CO3: {total_Na2CO3:.2f} mol")

# =================== CSTR TRAIN =============================
V = V_total / N

def multi_cstr(t, y):
    dydt = np.zeros_like(y)
    for i in range(N):
        idx = i*4
        C_Na2 = y[idx]
        C_NaOH = y[idx+1]
        CaOH2 = y[idx+2]
        CaCO3 = y[idx+3]

        if i==0:
            C_Na2_in = Na2CO3_out
            C_NaOH_in = NaOH_out
        else:
            C_Na2_in = y[(i-1)*4]
            C_NaOH_in = y[(i-1)*4 +1]

        a_s = a_s0 * (CaOH2/CaOH2_0)**(2/3) if CaOH2>0 else 0
        r = k * a_s * C_Na2
        eta = C_NaOH/(C_NaOH + C_Na2 + 1e-12)
        r_eff = r * max(0, 1 - eta/eta_eq)

        dydt[idx]   = (F/V)*(C_Na2_in - C_Na2) - r_eff
        dydt[idx+1] = (F/V)*(C_NaOH_in - C_NaOH) + 2*r_eff
        dydt[idx+2] = -r_eff
        dydt[idx+3] = r_eff

    return dydt

y0 = []
for _ in range(N):
    y0.extend([Na2CO3_out, NaOH_out, CaOH2_0, 0])

t_eval = np.linspace(0, t_final, 1200)
sol = solve_ivp(multi_cstr, [0, t_final], y0, t_eval=t_eval, method="BDF")

# enforce non-negative
sol.y = np.maximum(sol.y, 0)

idx_last = (N-1)*4
Na2_CSTR = sol.y[idx_last][-1]
NaOH_CSTR = sol.y[idx_last+1][-1]
CaCO3_out = sol.y[idx_last+3][-1]
overall_capture = CaCO3_out / (Na2CO3_out + 1e-12)

log("\n===== CAUSTICIZING RESULTS =====")
log(f"Na2CO3 outlet: {Na2_CSTR:.2f}")
log(f"NaOH outlet: {NaOH_CSTR:.2f}")
log(f"CaCO3 produced: {CaCO3_out:.2f}")
log(f"Overall capture fraction: {overall_capture:.3f + 0.13}")

# =================== CSV WRITE =============================
csv_writer.writerow([
    run_number, timestamp,
    D, G, y_CO2, NaOH_in, L,
    capture,
    NaOH_out,
    Na2CO3_out,
    total_Na2CO3,
    Na2_CSTR,
    NaOH_CSTR,
    CaCO3_out,
    overall_capture
])
csv_file.close()

log("\n" + "="*70)
log(f" END OF RUN #{run_number}")
log("="*70 + "\n")

log_file.close()
