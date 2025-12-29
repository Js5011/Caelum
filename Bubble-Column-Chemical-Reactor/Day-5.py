import numpy as np
import matplotlib.pyplot as plt

# 1. COLUMN PARAMETERS

H = float(input("What is the column height in meters?    "))           # Column height (m)
N = int(input("What is the number of slices     "))            # Number of slices
dz = H / N
z = np.linspace(0, H, N)

L = 1e-3            # Liquid flow rate (m^3/s)
A = 0.05            # Column area (m^2)
vL = L / A          # Liquid velocity (m/s)

kLa = 0.08          # Mass transfer coefficient (1/s)

# 2. GAS–LIQUID EQUILIBRIUM

P_CO2 = 0.15 * 101325   # Pa
H_CO2 = 3.4e4           # Pa·m^3/mol
C_star = P_CO2 / H_CO2  # mol/m^3

# 3. BOUNDARY CONDITION

C_in = 0.0              # Lean solvent entering top

# 4. INITIAL GUESS

C = np.zeros(N)

# 5. STEADY-STATE SOLVER

tolerance = 1e-6
max_iter = 10000

for iteration in range(max_iter):
    C_old = C.copy()

    for i in range(N):
        C_up = C_in if i == 0 else C[i - 1]

        C[i] = (
            (vL / dz) * C_up + kLa * C_star
        ) / (
            (vL / dz) + kLa
        )

    if np.max(np.abs(C - C_old)) < tolerance:
        break

# 6. CLEAR, HUMAN-READABLE OUTPUT

C_out = C[-1]
absorption = C_out - C_in
fraction_equilibrium = C_out / C_star

print("\n===== CO₂ ABSORPTION COLUMN RESULTS =====")
print(f"Column height:           {H:.1f} m")
print(f"Number of slices:        {N}")
print(f"Slice height (dz):       {dz:.2f} m\n")

print(f"Equilibrium CO₂ (C*):    {C_star:.3f} mol/m³")
print(f"Inlet CO₂ (top):         {C_in:.3f} mol/m³")
print(f"Outlet CO₂ (bottom):    {C_out:.3f} mol/m³\n")

print(f"CO₂ absorbed:            {absorption:.3f} mol/m³")
print(f"Fraction of equilibrium: {fraction_equilibrium:.2%}")

if fraction_equilibrium > 1:
    print("✔ Column is near equilibrium (possibly oversized).")
elif fraction_equilibrium > 0.5:
    print("✔ Column has good absorption efficiency.")
else:
    print("⚠ Column may be too short or kLa too low.")

# 7. SIMPLE TABLE (SELECTED HEIGHTS)

print("\nHeight (m) | CO₂ Concentration (mol/m³)")
print("--------------------------------------")

for i in range(0, N, N // 5):
    print(f"{z[i]:8.2f} | {C[i]:.3f}")

print(f"{H:8.2f} | {C_out:.3f}")

# 8. GRAPH

plt.figure(figsize=(6, 8))
plt.plot(C, z, linewidth=2, label="Liquid CO₂ concentration")
plt.axvline(C_star, linestyle="--", color="gray", label="Equilibrium $C^*$")

plt.xlabel("CO₂ concentration in liquid (mol/m³)")
plt.ylabel("Column height (m)")
plt.title("Steady-State CO₂ Absorption Profile")
plt.legend()
plt.grid()
plt.show()
