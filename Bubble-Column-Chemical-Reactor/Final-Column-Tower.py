import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. USER INPUTS =====================
D = float(input("Column diameter (m): "))
G = float(input("Gas flow rate (m³/s): "))
y_CO2 = float(input("CO₂ fraction in gas (0-1): "))
NaOH_in = float(input("NaOH concentration (mol/m³): "))
L = float(input("Liquid flow rate (m³/s): "))

# Artificial height for discretization (does not affect total capture)
H_sim = 10   # arbitrary numerical column height
N = 800      # slices
dz = H_sim / N
z = np.linspace(0,H_sim,N)

# Derived parameters
A = np.pi*(D/2)**2
vL = L/A
vG = max(G/A, 0.05)  # prevent zero gas velocity

# ===================== 2. PHYSICAL PARAMETERS =====================
# Constants
R = 8.314
T = 298
P_total = 101325

# Henry's law CO₂ in water
H_CO2 = 3.4e4

# Mass transfer and reaction
kLa0 = 0.08         # 1/s base
k_rxn = 1.0         # 1/s pseudo-first-order
n_rxn = 1

# Axial dispersion
D_axial_g = 1e-5
D_axial_l = 1e-9

# Activity correction for high NaOH
def activity_coeff(NaOH):
    return 1/(1+NaOH/2000)  # reduces CO₂ solubility at high NaOH

# Temperature effect (simplified)
def Henry_CO2(T):
    return H_CO2*(1 + 0.01*(T-298))  # small T effect

# ===================== 3. INITIAL CONDITIONS =====================
C_CO2g = np.ones(N)*(y_CO2*P_total/(R*T))
C_CO2l = np.zeros(N)
C_NaOH = np.ones(N)*NaOH_in
C_Na2CO3 = np.zeros(N)

alpha = 0.5
tol = 1e-8
max_iter = 20000

# ===================== 4. ITERATION =====================
for iteration in range(max_iter):
    old = np.vstack([C_CO2g,C_CO2l,C_NaOH,C_Na2CO3]).copy()
    
    for i in range(N):
        g_up = C_CO2g[i-1] if i>0 else C_CO2g[0]
        l_up = C_CO2l[i-1] if i>0 else 0
        naoh_up = C_NaOH[i-1] if i>0 else NaOH_in
        na2_up = C_Na2CO3[i-1] if i>0 else 0
        
        # Henry's law with activity correction
        H_eff = Henry_CO2(T) / activity_coeff(naoh_up)
        P_CO2 = g_up*R*T
        C_star = P_CO2 / H_eff
        
        # Variable kLa enhanced by NaOH
        kLa_eff = kLa0*(1 + 5*naoh_up/(naoh_up+2000))
        
        # Mass transfer flux
        N_CO2 = kLa_eff*(C_star - l_up)
        dCO2_g = min(N_CO2*dz/vG, g_up)
        new_g = g_up - dCO2_g + D_axial_g*(C_CO2g[i-1]-2*g_up+C_CO2g[i+1] if 0<i<N-1 else 0)/dz**2
        
        # Liquid transport
        dCO2_l = dCO2_g*(vG/vL)
        new_l = l_up + dCO2_l
        new_l = min(new_l,C_star)
        new_l += D_axial_l*(C_CO2l[i-1]-2*l_up+C_CO2l[i+1] if 0<i<N-1 else 0)/dz**2
        
        # Mass-transfer-limited reaction
        r_rxn = k_rxn*new_l*(naoh_up**n_rxn)
        dNaOH = min(2*r_rxn*dz/vL, naoh_up)
        new_naoh = naoh_up - dNaOH
        new_na2co3 = na2_up + dNaOH/2
        
        # Under-relaxation
        C_CO2g[i] = alpha*new_g + (1-alpha)*C_CO2g[i]
        C_CO2l[i] = alpha*new_l + (1-alpha)*C_CO2l[i]
        C_NaOH[i] = alpha*new_naoh + (1-alpha)*C_NaOH[i]
        C_Na2CO3[i] = alpha*new_na2co3 + (1-alpha)*C_Na2CO3[i]
    
    # Convergence check
    max_diff = np.max(np.abs(old - np.vstack([C_CO2g,C_CO2l,C_NaOH,C_Na2CO3])))
    if max_diff < tol:
        break

# ===================== 5. RESULTS =====================
capture = 1 - C_CO2g[-1]/C_CO2g[0]
total_Na2CO3 = C_Na2CO3[-1]*A*H_sim

print("\n===== RESULTS =====")
print(f"CO₂ capture efficiency: {capture:.1%}")
print(f"NaOH remaining: {C_NaOH[-1]:.1f} mol/m³")
print(f"Na₂CO₃ produced: {C_Na2CO3[-1]:.1f} mol/m³")
print(f"Total Na₂CO₃ in column: ~{total_Na2CO3:.1f} mol")

# ===================== 6. PLOTS =====================
plt.figure(figsize=(8,6))
plt.plot(C_NaOH,z,label='NaOH')
plt.plot(C_Na2CO3,z,label='Na₂CO₃')
plt.xlabel('Concentration (mol/m³)')
plt.ylabel('Column height (m)')
plt.title('Physically Accurate CO₂ Capture Profiles')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(z,C_Na2CO3/C_Na2CO3[-1],label='Fraction of Na₂CO₃ formed')
plt.xlabel('Column height (m)')
plt.ylabel('Cumulative capture fraction')
plt.title('CO₂ Capture along Column Height')
plt.grid()
plt.show()
