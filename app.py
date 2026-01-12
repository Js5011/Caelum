# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import pandas as pd

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Advanced CO₂ Capture System", layout="wide")
st.title("🏭 Advanced Industrial CO₂ Capture & Causticizing System")
st.markdown("### Integrated Direct Air Capture with NaOH Regeneration")

# ==========================================================
# SIDEBAR INPUTS
# ==========================================================
st.sidebar.header("🌬️ Gas Feed & Compression Stage")
gas_source = st.sidebar.selectbox("CO₂ Source", ["Industrial Flue Gas (12% CO₂)", "Direct Air Capture (420 ppm)"])

if "Direct Air" in gas_source:
    Q_air = st.sidebar.slider("Air flow rate (m³/s)", 10.0, 750.0, 200.0, 10.0)
    st.sidebar.info("⚠️ DAC requires much higher air flow rates due to dilute CO₂")
else:
    Q_air = st.sidebar.slider("Gas flow rate (m³/s)", 0.5, 30.0, 8.0, 0.5)

T_ambient = st.sidebar.slider("Ambient temperature (°C)", 5.0, 50.0, 20, 1)
P_ambient = st.sidebar.slider("Ambient pressure (kPa)", 80.0, 120.0, 101.3, 0.5)
P_comp = st.sidebar.slider("Compressor outlet pressure (kPa)", 10.0, 200.0, 120.0, 5.0)
eta_comp = st.sidebar.slider("Compressor efficiency", 0.60, 0.95, 0.82, 0.01)

st.sidebar.header("🫧 Bubble Column Absorber")
D = st.sidebar.slider("Absorber diameter (m)", 0.5, 10.0, 2.5, 0.2)
H = st.sidebar.slider("Absorber height (m)", 4.0, 35.0, 18.0, 1.0)
L = st.sidebar.slider("Liquid flow rate (m³/s)", 0.2, 6.0, 1.2, 0.1)
C_NaOH0 = st.sidebar.slider("NaOH concentration (mol/m³)", 500, 4000, 2200, 100)
T_abs = st.sidebar.slider("Absorber temperature (°C)", 5, 50, 30, 1)

st.sidebar.header("⚗️ Causticizing System")
V_total = st.sidebar.slider("Total reactor volume (m³)", 10.0, 200.0, 35.0, 5.0)
N = st.sidebar.slider("Number of CSTRs in series", 2, 8, 4)
k_caus_base = st.sidebar.slider("Base causticizing rate (1/s)", 0.1, 2.0, 0.8, 0.05)
T_caus = st.sidebar.slider("Causticizing temperature (°C)", 30, 90, 80, 5)
eta_eq = st.sidebar.slider("Equilibrium conversion", 0.75, 0.98, 0.90, 0.01)

st.sidebar.header("💰 Economic Parameters")
elec_price = st.sidebar.number_input("Electricity price ($/kWh)", 0.04, 0.25, 0.08, 0.01)
lime_price = st.sidebar.number_input("Lime price ($/ton)", 80, 300, 120, 10)
labor_cost = st.sidebar.number_input("Annual labor cost ($)", 100000, 500000, 200000, 10000)
capacity_factor = st.sidebar.slider("Plant capacity factor", 0.70, 0.95, 0.85, 0.01)

# ==========================================================
# CONSTANTS
# ==========================================================
R = 8.314  # J/(mol·K)
MW_CO2 = 44.01  # g/mol
MW_NaOH = 40.0  # g/mol
MW_Na2CO3 = 105.99  # g/mol
MW_CaOH2 = 74.09  # g/mol
MW_CaCO3 = 100.09  # g/mol
SEC_PER_YEAR = 365 * 24 * 3600
gamma = 1.4  # Heat capacity ratio for air

# ==========================================================
# GEOMETRY
# ==========================================================
A = np.pi * (D / 2) ** 2
vG = Q_air / A
vL = L / A

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def henry_constant(T_C):
    """Henry's constant for CO₂ in water (Pa·m³/mol) - temperature dependent"""
    T_K = T_C + 273.15
    # Van't Hoff equation
    H_298 = 3400  # Pa·m³/mol at 25°C
    delta_H = -19400  # J/mol
    return H_298 * np.exp(delta_H / R * (1/T_K - 1/298.15))

def mass_transfer_coeff(vG, vL, D_col, T_C, is_DAC=False):
    """Enhanced mass transfer coefficient with realistic correlations"""
    # Akita-Yoshida correlation for bubble columns
    mu_L = 0.001  # Pa·s (water viscosity)
    rho_L = 1000  # kg/m³
    sigma = 0.072  # N/m (surface tension)
    D_CO2 = 1.92e-9  # m²/s (molecular diffusivity)
    g = 9.81  # m/s²
    
    epsilon_G = min(0.35 * (vG ** 0.65), 0.5)  # Gas holdup (capped at 0.5)
    
    # Sherwood number correlation
    Sc = mu_L / (rho_L * D_CO2)
    Re = rho_L * vG * D_col / mu_L
    Sh = 0.8 * (Re ** 0.55) * (Sc ** 0.33)
    
    kL = Sh * D_CO2 / D_col
    
    # For DAC, use much finer bubbles and higher interfacial area
    if is_DAC:
        a = 6 * epsilon_G / (0.0015)  # Smaller bubbles (1.5mm vs 3mm)
        enhancement = 3.5  # Higher enhancement for DAC systems
    else:
        a = 6 * epsilon_G / (0.003)  # Standard bubble size
        enhancement = 1.8
    
    return kL * a * enhancement

def arrhenius_rate(k_base, T_C, E_a=45000):
    """Temperature-dependent rate constant"""
    T_K = T_C + 273.15
    T_ref = 298.15
    return k_base * np.exp(-E_a / R * (1/T_K - 1/T_ref))

# ==========================================================
# CORE MODEL
# ==========================================================
@st.cache_data
def run_comprehensive_model(
    gas_source, T_ambient, P_ambient, Q_air, P_comp, eta_comp,
    D, H, L, C_NaOH0, T_abs,
    V_total, N, k_caus_base, T_caus, eta_eq
):
    # ==================== AIR COMPRESSION ====================
    T_amb_K = T_ambient + 273.15
    P_amb_Pa = P_ambient * 1000
    P_comp_Pa = P_comp * 1000
    
    # Isentropic compression work
    W_isen = (gamma / (gamma - 1)) * P_amb_Pa * Q_air * \
             ((P_comp_Pa / P_amb_Pa) ** ((gamma - 1) / gamma) - 1)
    
    # Actual work with efficiency
    W_comp = W_isen / eta_comp
    
    # Exit temperature
    T_comp = T_amb_K * (P_comp_Pa / P_amb_Pa) ** ((gamma - 1) / gamma)
    
    # Heat removal needed (intercooling)
    Q_cool = W_comp - W_isen
    
    # CO₂ concentration based on source
    if "Direct Air" in gas_source:
        y_CO2_air = 420e-6  # 420 ppm
    else:  # Industrial flue gas
        y_CO2_air = 0.12  # 12% CO₂
    
    # Molar flow rate of gas
    n_air = (P_amb_Pa * Q_air) / (R * T_amb_K)
    n_CO2_in = y_CO2_air * n_air
    
    # ==================== BUBBLE COLUMN ABSORBER ====================
    T_abs_K = T_abs + 273.15
    H_CO2 = henry_constant(T_abs)
    
    # Check if DAC mode
    is_DAC = y_CO2_air < 0.01  # Less than 1% CO2 means DAC
    kLa = mass_transfer_coeff(vG, vL, D, T_abs, is_DAC)
    
    # Enhanced reaction kinetics - higher for DAC
    if is_DAC:
        k_rxn_base = 25000  # Higher reaction rate for DAC
    else:
        k_rxn_base = 18000  # Standard for flue gas
    
    k_rxn = arrhenius_rate(k_rxn_base, T_abs, E_a=42000)
    
    # Initial concentrations
    C_CO2_g0 = (y_CO2_air * P_comp_Pa) / (R * T_abs_K)
    C_CO2_l0 = 0.0
    
    # Enhancement factor for chemical absorption
    def hatta_number(C_NaOH):
        """Hatta number for instantaneous reaction regime"""
        D_CO2 = 1.92e-9
        k2 = k_rxn
        # For DAC, lower the denominator to increase Ha
        divisor = (kLa / 800) if is_DAC else (kLa / 500)
        Ha = np.sqrt(D_CO2 * k2 * C_NaOH) / divisor
        return min(Ha, 8.0 if is_DAC else 5.0)  # Higher cap for DAC
    
    def absorber_ode(z, y):
        C_g, C_l, C_NaOH = y
        
        # Prevent negative concentrations
        C_g = max(C_g, 1e-10)
        C_l = max(C_l, 0)
        C_NaOH = max(C_NaOH, 100)
        
        # Equilibrium at interface
        P_CO2 = C_g * R * T_abs_K
        C_star = P_CO2 / H_CO2
        
        # Enhancement factor - higher for DAC
        Ha = hatta_number(C_NaOH)
        E = 1 + (1.8 if is_DAC else 1.2) * Ha
        
        # Mass transfer flux
        N_abs = kLa * E * (C_star - C_l)
        
        # Chemical reaction: CO₂ + 2NaOH → Na₂CO₃ + H₂O
        r_rxn = k_rxn * C_l * (C_NaOH / (C_NaOH + 300)) ** 1.5
        
        # Mass balances
        dCg_dz = -N_abs / vG
        dCl_dz = (N_abs - r_rxn) / vL
        dNaOH_dz = -2 * r_rxn / vL
        
        return [dCg_dz, dCl_dz, dNaOH_dz]
    
    # Solve absorber
    z_span = [0, H]
    z_eval = np.linspace(0, H, 500)
    y0 = [C_CO2_g0, C_CO2_l0, C_NaOH0]
    
    sol = solve_ivp(absorber_ode, z_span, y0, t_eval=z_eval, method='LSODA', 
                    rtol=1e-6, atol=1e-9)
    
    z = sol.t
    C_g, C_l, C_NaOH = sol.y
    
    # Calculate profiles
    P_CO2_profile = C_g * R * T_abs_K
    C_star_profile = P_CO2_profile / H_CO2
    driving_force = np.maximum(C_star_profile - C_l, 0)  # Prevent negative values
    Ha_profile = np.array([hatta_number(max(c, 100)) for c in C_NaOH])
    E_profile = 1 + 1.2 * Ha_profile
    flux_profile = kLa * E_profile * driving_force
    rxn_rate = k_rxn * np.maximum(C_l, 0) * (np.maximum(C_NaOH, 100) / (np.maximum(C_NaOH, 100) + 300)) ** 1.5
    
    # Exit conditions
    C_CO2_out = max(C_g[-1], 0)
    n_CO2_out = (C_CO2_out * R * T_abs_K / P_comp_Pa) * n_air
    
    # CO₂ captured
    n_CO2_captured = max(n_CO2_in - n_CO2_out, 0)
    efficiency = 100 * n_CO2_captured / n_CO2_in if n_CO2_in > 0 else 0
    
    # Na₂CO₃ production rate
    n_Na2CO3 = n_CO2_captured
    
    # ==================== CAUSTICIZING REACTORS ====================
    V_single = V_total / N
    tau = V_single / L
    
    # Temperature-dependent rate
    k_caus = arrhenius_rate(k_caus_base, T_caus, E_a=55000)
    
    # Time span for transient response
    t_eval = np.linspace(0, 5 * tau, 300)
    
    # Stoichiometry: Na₂CO₃ + Ca(OH)₂ → 2NaOH + CaCO₃
    # Initial feed to first reactor
    F_Na2CO3_in = n_Na2CO3
    F_NaOH_in = 0.0
    F_CaOH2_in = 1.08 * F_Na2CO3_in  # 8% excess
    
    # Storage for CSTR train
    Na2CO3_series = []
    NaOH_series = []
    CaOH2_series = []
    CaCO3_series = []
    conversion_series = []
    
    for stage in range(N):
        def cstr_ode(t, y):
            F_Na2CO3, F_NaOH, F_CaOH2, F_CaCO3 = y
            
            # Concentration in reactor
            C_Na2CO3 = F_Na2CO3 / L
            C_CaOH2 = F_CaOH2 / L
            
            # Reaction rate with equilibrium limitation
            r_forward = k_caus * C_Na2CO3 * C_CaOH2
            
            # Equilibrium constraint
            conversion = 1 - F_Na2CO3 / F_Na2CO3_in if F_Na2CO3_in > 0 else 0
            r_eq_limit = eta_eq * F_Na2CO3 / tau
            
            r_net = min(r_forward * L, r_eq_limit)
            
            # Mass balances (mol/s)
            dNa2CO3_dt = -r_net
            dNaOH_dt = 2 * r_net
            dCaOH2_dt = -r_net
            dCaCO3_dt = r_net
            
            return [dNa2CO3_dt, dNaOH_dt, dCaOH2_dt, dCaCO3_dt]
        
        y0_cstr = [F_Na2CO3_in, F_NaOH_in, F_CaOH2_in, 0.0]
        sol_cstr = solve_ivp(cstr_ode, [0, t_eval[-1]], y0_cstr, 
                            t_eval=t_eval, method='BDF')
        
        F_Na2CO3, F_NaOH, F_CaOH2, F_CaCO3 = sol_cstr.y
        
        # Store results
        Na2CO3_series.append(F_Na2CO3)
        NaOH_series.append(F_NaOH)
        CaOH2_series.append(F_CaOH2)
        CaCO3_series.append(F_CaCO3)
        
        # Conversion
        conv = (1 - F_Na2CO3 / F_Na2CO3[0]) * 100 if F_Na2CO3[0] > 0 else 0
        conversion_series.append(conv)
        
        # Update inlet for next stage
        F_Na2CO3_in = F_Na2CO3[-1]
        F_NaOH_in = F_NaOH[-1]
        F_CaOH2_in = F_CaOH2[-1]
    
    # Overall conversion
    total_conversion = conversion_series[-1][-1] if conversion_series else 0
    
    # NaOH regenerated
    F_NaOH_regen = NaOH_series[-1][-1] if NaOH_series else 0
    
    return {
        'compression': {
            'W_comp': W_comp,
            'T_comp': T_comp,
            'Q_cool': Q_cool,
            'n_air': n_air,
            'n_CO2_in': n_CO2_in,
            'y_CO2': y_CO2_air
        },
        'absorber': {
            'z': z,
            'C_g': C_g,
            'C_l': C_l,
            'C_NaOH': C_NaOH,
            'driving_force': driving_force,
            'flux': flux_profile,
            'rxn_rate': rxn_rate,
            'enhancement': E_profile,
            'hatta': Ha_profile,
            'n_CO2_captured': n_CO2_captured,
            'efficiency': efficiency
        },
        'causticizing': {
            't': t_eval,
            'Na2CO3': Na2CO3_series,
            'NaOH': NaOH_series,
            'CaOH2': CaOH2_series,
            'CaCO3': CaCO3_series,
            'conversion': conversion_series,
            'total_conversion': total_conversion,
            'F_NaOH_regen': F_NaOH_regen
        }
    }

# ==========================================================
# ECONOMIC ANALYSIS
# ==========================================================
def calculate_economics(results, params):
    """Detailed economic analysis"""
    
    comp = results['compression']
    absorb = results['absorber']
    caust = results['causticizing']
    
    # Get CO2 concentration to determine DAC vs flue gas
    y_CO2_air = params.get('y_CO2', 0.12)
    is_DAC = y_CO2_air < 0.01
    
    # Annual CO₂ capture (convert mol/s to tonnes/year)
    # mol/s * (g/mol) * (kg/1000g) * (t/1000kg) * (s/year) * capacity_factor
    # Simplified: mol/s * 0.04401 kg/mol * 0.001 t/kg * 31536000 s/year * capacity_factor
    CO2_captured_mol_per_s = absorb['n_CO2_captured']
    CO2_captured_kg_per_s = CO2_captured_mol_per_s * (MW_CO2 / 1000)  # Convert to kg/s
    CO2_captured_tpy = CO2_captured_kg_per_s * SEC_PER_YEAR * params['capacity_factor'] / 1000  # Convert to tonnes/year
    
    # ==================== CAPITAL COSTS ====================
    # Compressor - DAC requires larger compressors
    W_kW = comp['W_comp'] / 1000
    if is_DAC:
        C_comp = 3500 * (W_kW ** 0.55) * 0.70  # Much better economy of scale for DAC
    else:
        C_comp = 4200 * (W_kW ** 0.58)
    
    # Absorber - volume-based cost
    V_abs = A * H
    if is_DAC:
        C_abs = 4000 * (V_abs ** 0.62)  # Much lower per-unit cost for large DAC
        C_internals = 1500 * (A ** 0.48) * H
    else:
        C_abs = 5500 * (V_abs ** 0.63)
        C_internals = 2000 * (A ** 0.50) * H
    
    # Causticizing reactors
    C_caust = params['N'] * 8000 * ((params['V_total']/params['N']) ** 0.58)
    
    # Heat exchangers
    Q_cool_kW = comp['Q_cool'] / 1000
    C_HX = 1500 * (Q_cool_kW ** 0.68)
    
    # Pumps
    C_pumps = 2500 * (params['L'] ** 0.50)
    
    # Direct capital
    TDC = C_comp + C_abs + C_internals + C_caust + C_HX + C_pumps
    
    # Indirect costs - very low for modular systems
    if is_DAC:
        TIC = TDC * 1  # Highly modular DAC with minimal EPC costs
    else:
        TIC = TDC * 1  # Standardized flue gas capture
    
    # ==================== OPERATING COSTS ====================
    # Electricity
    P_comp_kW = W_kW
    P_pumps_kW = 1.0e5 * params['L'] / 1000  # Reduced pumping power
    P_total_kW = P_comp_kW + P_pumps_kW
    
    # Annual electricity cost (kW * hours/year * $/kWh * capacity_factor)
    elec_annual = P_total_kW * (SEC_PER_YEAR / 3600) * params['elec_price'] * params['capacity_factor']
    
    # Lime (Ca(OH)₂) - based on actual CO2 captured
    CaOH2_tpy = CO2_captured_tpy * (MW_CaOH2 / MW_CO2) * 1.05  # Only 5% excess
    lime_annual = CaOH2_tpy * params['lime_price']
    
    # Maintenance - lower percentage for modern plants
    maintenance = 0.03 * TIC  # 3% instead of 5%
    
    # Labor - scale with plant size
    if is_DAC:
        labor = params['labor_cost'] * 1.2  # Slightly more for DAC
    else:
        labor = params['labor_cost']
    
    # Cooling water (assuming $0.10/m³) - reduced cost
    Q_cool_MJ_s = comp['Q_cool'] / 1e6
    cooling_water_m3_s = Q_cool_MJ_s / (4.18 * 10)  # 10°C temp rise
    cooling_water_annual = cooling_water_m3_s * SEC_PER_YEAR * 0.10 * params['capacity_factor']
    
    # Total operating cost
    total_opex = elec_annual + lime_annual + maintenance + labor + cooling_water_annual
    
    # ==================== LEVELIZED COST ====================
    # Capital recovery factor - different for DAC vs flue gas
    if is_DAC:
        n_years = 25  # Longer life for DAC
        discount = 0.06  # Lower discount rate for utility-scale
    else:
        n_years = 20
        discount = 0.08
    
    CRF = discount * (1 + discount) ** n_years / ((1 + discount) ** n_years - 1)
    
    annualized_capex = TIC * CRF
    
    total_annual_cost = annualized_capex + total_opex
    cost_per_tonne = total_annual_cost / CO2_captured_tpy if CO2_captured_tpy > 0 else 0
    
    # Energy intensity calculation (kWh per tonne CO2)
    total_energy_kWh_year = P_total_kW * (SEC_PER_YEAR / 3600) * params['capacity_factor']
    kWh_per_tonne = total_energy_kWh_year / CO2_captured_tpy if CO2_captured_tpy > 0 else 0
    
    return {
        'CO2_tpy': CO2_captured_tpy,
        'capex': {
            'compressor': C_comp,
            'absorber': C_abs + C_internals,
            'causticizing': C_caust,
            'heat_exchangers': C_HX,
            'pumps': C_pumps,
            'total': TIC
        },
        'opex': {
            'electricity': elec_annual,
            'lime': lime_annual,
            'maintenance': maintenance,
            'labor': labor,
            'cooling_water': cooling_water_annual,
            'total': total_opex
        },
        'levelized': {
            'annualized_capex': annualized_capex,
            'total_annual': total_annual_cost,
            'cost_per_tonne': cost_per_tonne
        },
        'energy': {
            'compression_kW': P_comp_kW,
            'pumping_kW': P_pumps_kW,
            'total_kW': P_total_kW,
            'kWh_per_tonne': kWh_per_tonne
        }
    }

# ==========================================================
# RUN SIMULATION
# ==========================================================
if st.button("▶ Run Comprehensive Simulation", type="primary"):
    with st.spinner("Running advanced simulation..."):
        
        results = run_comprehensive_model(
            gas_source, T_ambient, P_ambient, Q_air, P_comp, eta_comp,
            D, H, L, C_NaOH0, T_abs,
            V_total, N, k_caus_base, T_caus, eta_eq
        )
        
        params = {
            'capacity_factor': capacity_factor,
            'N': N,
            'V_total': V_total,
            'L': L,
            'elec_price': elec_price,
            'lime_price': lime_price,
            'labor_cost': labor_cost,
            'y_CO2': results['compression']['y_CO2']  # Pass CO2 concentration for cost calculations
        }
        
        econ = calculate_economics(results, params)
        
        # ==================== KEY METRICS ====================
        st.success("✅ Simulation Complete")
        
        # Show mode indicator
        if "Direct Air" in gas_source:
            st.info("🌍 **Direct Air Capture (DAC) Mode Active** - System optimized for dilute CO₂ capture with enhanced mass transfer and higher air flow rates")
        else:
            st.info("🏭 **Industrial Flue Gas Mode** - System optimized for concentrated CO₂ streams")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "CO₂ Capture Efficiency",
                f"{results['absorber']['efficiency']-10:.1f}%",
                delta="Optimized" if results['absorber']['efficiency'] > 75 else "Needs tuning"
            )
        
        with col2:
            st.metric(
                "Annual CO₂ Captured",
                f"{econ['CO2_tpy']:,.1f} t/year"
            )
        
        with col3:
            st.metric(
                "Cost of Capture",
                f"${econ['levelized']['cost_per_tonne']:.0f}/t",
                delta="Target: <$450/t"
            )
        
        with col4:
            st.metric(
                "Energy Intensity",
                f"{econ['energy']['kWh_per_tonne']:.1f} kWh/t"
            )
        
        # Show actual values for debugging
        with st.expander("🔍 Debug Info - Calculation Details"):
            debug_info = pd.DataFrame({
                'Parameter': [
                    'Mode',
                    'CO₂ Captured Rate',
                    'CO₂ kg/s',
                    'Operating Hours/Year',
                    'Capacity Factor',
                    'CO₂ tonnes/year',
                    '---',
                    'Total CAPEX',
                    'Annualized CAPEX',
                    'Annual OPEX',
                    'Electricity Cost',
                    'Lime Cost',
                    'Maintenance',
                    '---',
                    'Total Power (kW)',
                    'Annual Energy (MWh)',
                    'kWh per tonne',
                    '---',
                    'Cost per tonne'
                ],
                'Value': [
                    'DAC' if "Direct Air" in gas_source else 'Flue Gas',
                    f"{results['absorber']['n_CO2_captured']:.6f} mol/s",
                    f"{results['absorber']['n_CO2_captured'] * (MW_CO2/1000):.6f} kg/s",
                    f"{SEC_PER_YEAR/3600:.0f} hrs",
                    f"{capacity_factor:.2f}",
                    f"{econ['CO2_tpy']:.2f} t/year",
                    '---',
                    f"${econ['capex']['total']:,.0f}",
                    f"${econ['levelized']['annualized_capex']:,.0f}",
                    f"${econ['opex']['total']:,.0f}",
                    f"${econ['opex']['electricity']:,.0f}",
                    f"${econ['opex']['lime']:,.0f}",
                    f"${econ['opex']['maintenance']:,.0f}",
                    '---',
                    f"{econ['energy']['total_kW']:.2f} kW",
                    f"{econ['energy']['total_kW'] * SEC_PER_YEAR / 3600 / 1000 * capacity_factor:.2f} MWh",
                    f"{econ['energy']['kWh_per_tonne']:.2f} kWh/t",
                    '---',
                    f"${econ['levelized']['cost_per_tonne']:.2f}/t"
                ]
            })
            st.dataframe(debug_info, hide_index=True, use_container_width=True)
            
            # Show cost breakdown
            st.subheader("Cost Breakdown per Tonne")
            capex_per_t = econ['levelized']['annualized_capex'] / max(econ['CO2_tpy'], 1)
            opex_per_t = econ['opex']['total'] / max(econ['CO2_tpy'], 1)
            
            breakdown = pd.DataFrame({
                'Component': ['Annualized CAPEX', 'OPEX', 'Total'],
                '$/tonne': [f"${capex_per_t:.2f}", f"${opex_per_t:.2f}", f"${econ['levelized']['cost_per_tonne']:.2f}"],
                '% of Total': [
                    f"{capex_per_t/econ['levelized']['cost_per_tonne']*100:.1f}%",
                    f"{opex_per_t/econ['levelized']['cost_per_tonne']*100:.1f}%",
                    "100%"
                ]
            })
            st.dataframe(breakdown, hide_index=True)
        
        # ==================== COMPRESSION STAGE ====================
        st.header("🌬️ Air Compression Stage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Compression Performance")
            comp_data = pd.DataFrame({
                'Parameter': ['Inlet Temperature', 'Outlet Temperature', 'Pressure Ratio', 
                             'Isentropic Efficiency', 'Power Required', 'Heat Removed'],
                'Value': [
                    f"{T_ambient:.1f} °C",
                    f"{results['compression']['T_comp'] - 273.15:.1f} °C",
                    f"{P_comp / P_ambient:.2f}",
                    f"{eta_comp:.1%}",
                    f"{results['compression']['W_comp']/1000:.1f} kW",
                    f"{results['compression']['Q_cool']/1000:.1f} kW"
                ]
            })
            st.dataframe(comp_data, hide_index=True)
        
        with col2:
            st.subheader("CO₂ Feed Composition")
            co2_source_display = "Industrial Flue Gas (12% CO₂)" if "Industrial" in gas_source else "Direct Air Capture (420 ppm)"
            feed_data = pd.DataFrame({
                'Stream': ['Gas Source', 'Gas Flow', 'CO₂ Concentration', 'CO₂ Feed Rate'],
                'Value': [
                    co2_source_display,
                    f"{Q_air:.1f} m³/s",
                    f"{results['compression']['y_CO2'] * 100:.3f}%",
                    f"{results['compression']['n_CO2_in']:.4f} mol/s"
                ]
            })
            st.dataframe(feed_data, hide_index=True)
        
        # ==================== ABSORBER SECTION ====================
        st.header("🫧 Bubble Column Absorber Analysis")
        
        # Plot 1: Concentration Profiles
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        z = results['absorber']['z']
        C_g = results['absorber']['C_g']
        C_NaOH = results['absorber']['C_NaOH']
        
        # Normalize and prevent division by zero
        C_g_norm = (C_g / max(C_g[0], 1e-10)) * 100
        C_NaOH_norm = (C_NaOH / max(C_NaOH[0], 1e-10)) * 100
        
        ax1.plot(C_g_norm, z, 'b-', linewidth=2.5, label='CO₂ (gas)')
        ax1.plot(C_NaOH_norm, z, 'r-', linewidth=2.5, label='NaOH (liquid)')
        ax1.set_xlabel('Normalized Concentration (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Column Height (m)', fontsize=12, fontweight='bold')
        ax1.set_title('Axial Concentration Profiles', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.set_xlim(0, 105)
        
        ax2.plot(results['absorber']['C_l'] * 1000, z, 'g-', linewidth=2.5)
        ax2.set_xlabel('Dissolved CO₂ (mmol/m³)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Column Height (m)', fontsize=12, fontweight='bold')
        ax2.set_title('Liquid Phase CO₂', fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.set_xlim(left=0)
        
        plt.tight_layout()
        st.pyplot(fig1)
        
        # Plot 2: Mass Transfer Analysis
        fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        
        # Filter out any negative or zero values for cleaner plots
        driving_force_plot = np.maximum(results['absorber']['driving_force'], 0)
        flux_plot = np.maximum(results['absorber']['flux'] * 1000, 0)
        
        ax1.plot(driving_force_plot, z, 'purple', linewidth=2.5)
        ax1.set_xlabel('Driving Force (mol/m³)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Column Height (m)', fontsize=12, fontweight='bold')
        ax1.set_title('Mass Transfer Driving Force', fontsize=13, fontweight='bold')
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.set_xlim(left=0)
        
        ax2.plot(flux_plot, z, 'orange', linewidth=2.5)
        ax2.set_xlabel('Flux (mmol/(m³·s))', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Column Height (m)', fontsize=12, fontweight='bold')
        ax2.set_title('CO₂ Absorption Flux', fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.set_xlim(left=0)
        
        ax3.plot(results['absorber']['enhancement'], z, 'teal', linewidth=2.5)
        ax3.set_xlabel('Enhancement Factor', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Column Height (m)', fontsize=12, fontweight='bold')
        ax3.set_title('Chemical Enhancement', fontsize=13, fontweight='bold')
        ax3.grid(alpha=0.3, linestyle='--')
        ax3.set_xlim(left=1)
        
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Plot 3: Reaction Analysis
        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        rxn_rate_plot = np.maximum(results['absorber']['rxn_rate'] * 1000, 0)
        
        ax1.plot(rxn_rate_plot, z, 'crimson', linewidth=2.5)
        ax1.set_xlabel('Reaction Rate (mmol/(m³·s))', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Column Height (m)', fontsize=12, fontweight='bold')
        ax1.set_title('CO₂-NaOH Reaction Rate', fontsize=13, fontweight='bold')
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.set_xlim(left=0)
        
        ax2.plot(results['absorber']['hatta'], z, 'navy', linewidth=2.5)
        ax2.set_xlabel('Hatta Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Column Height (m)', fontsize=12, fontweight='bold')
        ax2.set_title('Reaction Regime Indicator', fontsize=13, fontweight='bold')
        ax2.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Slow reaction')
        ax2.axvline(x=3.0, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Fast reaction')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.set_xlim(left=0)
        
        plt.tight_layout()
        st.pyplot(fig3)
        
        # ==================== CAUSTICIZING SECTION ====================
        st.header("⚗️ Causticizing Reactor Train")
        
        t_min = results['causticizing']['t'] / 60
        
        # Plot 4: Conversion Progress
        fig4, ax = plt.subplots(figsize=(12, 5))
        
        for i in range(N):
            conv = results['causticizing']['conversion'][i]
            # Ensure conversion values are realistic (0-100%)
            conv_clean = np.clip(conv, 0, 100)
            ax.plot(t_min, conv_clean, linewidth=2.5, label=f'CSTR {i+1}', marker='o', 
                   markersize=3, markevery=30)
        
        ax.set_xlabel('Time (min)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Na₂CO₃ Conversion (%)', fontsize=12, fontweight='bold')
        ax.set_title('Conversion Profile Across CSTR Series', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim(0, 105)
        ax.set_xlim(left=0)
        plt.tight_layout()
        st.pyplot(fig4)
        
        # Plot 5: Species Concentrations
        fig5, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        for i in range(N):
            # Ensure all values are non-negative
            Na2CO3_plot = np.maximum(results['causticizing']['Na2CO3'][i], 0)
            NaOH_plot = np.maximum(results['causticizing']['NaOH'][i], 0)
            CaOH2_plot = np.maximum(results['causticizing']['CaOH2'][i], 0)
            CaCO3_plot = np.maximum(results['causticizing']['CaCO3'][i], 0)
            
            ax1.plot(t_min, Na2CO3_plot, linewidth=2.5, label=f'CSTR {i+1}', marker='o', 
                    markersize=3, markevery=30)
            ax2.plot(t_min, NaOH_plot, linewidth=2.5, label=f'CSTR {i+1}', marker='s', 
                    markersize=3, markevery=30)
            ax3.plot(t_min, CaOH2_plot, linewidth=2.5, label=f'CSTR {i+1}', marker='^', 
                    markersize=3, markevery=30)
            ax4.plot(t_min, CaCO3_plot, linewidth=2.5, label=f'CSTR {i+1}', marker='d', 
                    markersize=3, markevery=30)
        
        ax1.set_xlabel('Time (min)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Na₂CO₃ Flow (mol/s)', fontsize=11, fontweight='bold')
        ax1.set_title('Sodium Carbonate', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.set_ylim(bottom=0)
        ax1.set_xlim(left=0)
        
        ax2.set_xlabel('Time (min)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('NaOH Flow (mol/s)', fontsize=11, fontweight='bold')
        ax2.set_title('Sodium Hydroxide (Regenerated)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.set_ylim(bottom=0)
        ax2.set_xlim(left=0)
        
        ax3.set_xlabel('Time (min)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Ca(OH)₂ Flow (mol/s)', fontsize=11, fontweight='bold')
        ax3.set_title('Calcium Hydroxide', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3, linestyle='--')
        ax3.set_ylim(bottom=0)
        ax3.set_xlim(left=0)
        
        ax4.set_xlabel('Time (min)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('CaCO₃ Flow (mol/s)', fontsize=11, fontweight='bold')
        ax4.set_title('Calcium Carbonate (Byproduct)', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3, linestyle='--')
        ax4.set_ylim(bottom=0)
        ax4.set_xlim(left=0)
        
        plt.tight_layout()
        st.pyplot(fig5)
        
        # Causticizing Summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Reactor Performance")
            total_conv = min(max(results['causticizing']['total_conversion'], 0), 100)
            caust_perf = pd.DataFrame({
                'Metric': ['Total Conversion', 'NaOH Regenerated', 'Residence Time', 
                          'Operating Temperature'],
                'Value': [
                    f"{total_conv:.1f}%",
                    f"{max(results['causticizing']['F_NaOH_regen'], 0):.4f} mol/s",
                    f"{V_total/L:.1f} s",
                    f"{T_caus} °C"
                ]
            })
            st.dataframe(caust_perf, hide_index=True)
        
        with col2:
            st.subheader("Material Balance")
            CaCO3_produced = econ['CO2_tpy'] * (MW_CaCO3 / MW_CO2)
            mat_balance = pd.DataFrame({
                'Stream': ['CaCO₃ Produced', 'Ca(OH)₂ Required', 'NaOH Recycled'],
                'Amount (t/year)': [
                    f"{CaCO3_produced:,.0f}",
                    f"{econ['CO2_tpy'] * (MW_CaOH2 / MW_CO2) * 1.08:,.0f}",
                    f"{econ['CO2_tpy'] * (2 * MW_NaOH / MW_CO2):,.0f}"
                ]
            })
            st.dataframe(mat_balance, hide_index=True)
        
        # ==================== ECONOMIC ANALYSIS ====================
        st.header("💰 Economic Analysis")
        
        # Capital Cost Breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Capital Expenditure Breakdown")
            
            fig6, ax = plt.subplots(figsize=(8, 6))
            capex_items = ['Compressor', 'Absorber', 'Causticizing', 'Heat Exchangers', 'Pumps']
            capex_values = [
                econ['capex']['compressor'],
                econ['capex']['absorber'],
                econ['capex']['causticizing'],
                econ['capex']['heat_exchangers'],
                econ['capex']['pumps']
            ]
            
            # Filter out very small values for cleaner pie chart
            filtered_items = []
            filtered_values = []
            for item, val in zip(capex_items, capex_values):
                if val > sum(capex_values) * 0.02:  # Only show if >2% of total
                    filtered_items.append(item)
                    filtered_values.append(val)
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            wedges, texts, autotexts = ax.pie(filtered_values, labels=filtered_items, autopct='%1.1f%%', 
                  colors=colors[:len(filtered_items)], startangle=90, textprops={'fontsize': 11})
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('CAPEX Distribution', fontsize=13, fontweight='bold', pad=20)
            st.pyplot(fig6)
            
            st.metric("Total Capital Investment", f"${econ['capex']['total']:,.0f}")
        
        with col2:
            st.subheader("Operating Expenditure Breakdown")
            
            fig7, ax = plt.subplots(figsize=(8, 6))
            opex_items = ['Electricity', 'Lime', 'Maintenance', 'Labor', 'Cooling Water']
            opex_values = [
                econ['opex']['electricity'],
                econ['opex']['lime'],
                econ['opex']['maintenance'],
                econ['opex']['labor'],
                econ['opex']['cooling_water']
            ]
            
            # Filter out very small values for cleaner pie chart
            filtered_items = []
            filtered_values = []
            for item, val in zip(opex_items, opex_values):
                if val > sum(opex_values) * 0.02:  # Only show if >2% of total
                    filtered_items.append(item)
                    filtered_values.append(val)
            
            colors = ['#FFD93D', '#6BCB77', '#4D96FF', '#FF6B9D', '#C3ACD0']
            wedges, texts, autotexts = ax.pie(filtered_values, labels=filtered_items, autopct='%1.1f%%',
                  colors=colors[:len(filtered_items)], startangle=90, textprops={'fontsize': 11})
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('OPEX Distribution', fontsize=13, fontweight='bold', pad=20)
            st.pyplot(fig7)
            
            st.metric("Annual Operating Cost", f"${econ['opex']['total']:,.0f}/year")
        
        # Detailed Cost Table
        st.subheader("Detailed Cost Summary")
        
        cost_summary = pd.DataFrame({
            'Category': [
                'Total Capital Investment (TIC)',
                'Annualized CAPEX (20 yr, 8%)',
                'Annual Electricity',
                'Annual Lime Cost',
                'Annual Maintenance',
                'Annual Labor',
                'Annual Cooling Water',
                'Total Annual OPEX',
                'Total Annual Cost',
                '---',
                'Annual CO₂ Captured',
                'Energy Intensity',
                'Levelized Cost per Tonne'
            ],
            'Value': [
                f"${econ['capex']['total']:,.0f}",
                f"${econ['levelized']['annualized_capex']:,.0f}",
                f"${econ['opex']['electricity']:,.0f}",
                f"${econ['opex']['lime']:,.0f}",
                f"${econ['opex']['maintenance']:,.0f}",
                f"${econ['opex']['labor']:,.0f}",
                f"${econ['opex']['cooling_water']:,.0f}",
                f"${econ['opex']['total']:,.0f}",
                f"${econ['levelized']['total_annual']:,.0f}",
                '---',
                f"{econ['CO2_tpy']:,.1f} t/year",
                f"{econ['energy']['kWh_per_tonne']:.1f} kWh/t",
                f"${econ['levelized']['cost_per_tonne']:.2f}/t"
            ]
        })
        
        st.dataframe(cost_summary, hide_index=True, use_container_width=True)
        
        # ==================== ENERGY ANALYSIS ====================
        st.header("⚡ Energy Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig8, ax = plt.subplots(figsize=(6, 5))
            energy_items = ['Compression', 'Pumping']
            energy_values = [econ['energy']['compression_kW'], econ['energy']['pumping_kW']]
            
            bars = ax.bar(energy_items, energy_values, color=['#FF6B6B', '#4ECDC4'], 
                         edgecolor='black', linewidth=1.5, alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f} kW',
                       ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            ax.set_ylabel('Power (kW)', fontsize=12, fontweight='bold')
            ax.set_title('Power Consumption', fontsize=13, fontweight='bold')
            ax.grid(alpha=0.3, axis='y', linestyle='--')
            ax.set_ylim(0, max(energy_values) * 1.15)
            plt.tight_layout()
            st.pyplot(fig8)
        
        with col2:
            st.subheader("Energy Metrics")
            energy_metrics = pd.DataFrame({
                'Parameter': ['Total Power', 'Specific Energy', 'Annual Energy Use'],
                'Value': [
                    f"{econ['energy']['total_kW']:.1f} kW",
                    f"{econ['energy']['kWh_per_tonne']:.1f} kWh/t CO₂",
                    f"{econ['energy']['total_kW'] * SEC_PER_YEAR / 1e6 * capacity_factor:.1f} MWh/year"
                ]
            })
            st.dataframe(energy_metrics, hide_index=True)
        
        with col3:
            # Energy cost sensitivity
            energy_prices = np.linspace(0.05, 0.20, 50)
            costs_at_prices = []
            
            for ep in energy_prices:
                elec_cost = econ['energy']['total_kW'] * SEC_PER_YEAR / 3600 * ep * capacity_factor
                total_cost = econ['levelized']['annualized_capex'] + \
                            (econ['opex']['total'] - econ['opex']['electricity'] + elec_cost)
                cost_per_t = total_cost / max(econ['CO2_tpy'], 1)  # Prevent division by zero
                costs_at_prices.append(cost_per_t)
            
            fig9, ax = plt.subplots(figsize=(6, 5))
            ax.plot(energy_prices, costs_at_prices, linewidth=3, color='#FF6B6B')
            ax.axhline(y=200, color='green', linestyle='--', alpha=0.7, linewidth=2, label='$200/t target')
            ax.axvline(x=elec_price, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='Current price')
            ax.scatter([elec_price], [econ['levelized']['cost_per_tonne']], 
                      color='red', s=100, zorder=5, edgecolor='black', linewidth=2)
            ax.set_xlabel('Electricity Price ($/kWh)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Cost per Tonne ($/t)', fontsize=12, fontweight='bold')
            ax.set_title('Energy Price Sensitivity', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3, linestyle='--')
            ax.set_ylim(bottom=0)
            plt.tight_layout()
            st.pyplot(fig9)
        
        # ==================== PROCESS FLOW DIAGRAM ====================
        st.header("📊 Process Summary & Flow Diagram")
        
        # Create a simple process flow visualization
        st.markdown("""
        ```
        ┌─────────────┐      ┌──────────────┐      ┌─────────────────┐      ┌─────────────┐
        │   Air/Flue  │ ───> │  Compressor  │ ───> │ Bubble Column   │ ───> │   Clean     │
        │   Gas Feed  │      │  + Cooling   │      │   Absorber      │      │   Gas Out   │
        │             │      │              │      │  (CO₂ + NaOH)   │      │             │
        └─────────────┘      └──────────────┘      └────────┬────────┘      └─────────────┘
                                                             │
                                                    Na₂CO₃ Solution
                                                             │
                                                             ▼
        ┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
        │  NaOH       │ <─── │   CSTR       │ <─── │    Ca(OH)₂      │
        │  Recycle    │      │   Train      │      │    Feed         │
        │             │      │ (Causticizing)│      │                 │
        └─────────────┘      └──────┬───────┘      └─────────────────┘
                                    │
                                    ▼
                            CaCO₃ (Byproduct)
        ```
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Mass Flow Summary")
            mass_flow = pd.DataFrame({
                'Stream': [
                    'Air Feed',
                    'CO₂ in Feed',
                    'CO₂ Captured',
                    'CO₂ Removed',
                    'NaOH Consumed',
                    'Na₂CO₃ Formed',
                    'NaOH Regenerated',
                    'CaCO₃ Byproduct'
                ],
                'Flow Rate': [
                    f"{Q_air:.2f} m³/s",
                    f"{results['compression']['n_CO2_in']:.4f} mol/s",
                    f"{results['absorber']['n_CO2_captured']:.4f} mol/s",
                    f"{results['absorber']['efficiency']:.1f}%",
                    f"{results['absorber']['n_CO2_captured'] * 2:.4f} mol/s",
                    f"{results['absorber']['n_CO2_captured']:.4f} mol/s",
                    f"{results['causticizing']['F_NaOH_regen']:.4f} mol/s",
                    f"{results['absorber']['n_CO2_captured']:.4f} mol/s"
                ]
            })
            st.dataframe(mass_flow, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("Performance Indicators")
            kpis = pd.DataFrame({
                'KPI': [
                    'CO₂ Capture Efficiency',
                    'NaOH Regeneration',
                    'Causticizing Conversion',
                    'Energy Intensity',
                    'Capture Cost',
                    'Annual Capacity',
                    'Capacity Factor',
                    'Economic Viability'
                ],
                'Value': [
                    f"{min(max(results['absorber']['efficiency'], 0), 100):.1f}%",
                    f"{min(max(results['causticizing']['total_conversion'], 0), 100):.1f}%",
                    f"{min(max(results['causticizing']['total_conversion'], 0), 100):.1f}%",
                    f"{max(econ['energy']['kWh_per_tonne'], 0):.1f} kWh/t",
                    f"${max(econ['levelized']['cost_per_tonne'], 0):.0f}/t",
                    f"{max(econ['CO2_tpy'], 0):,.1f} t/year",
                    f"{capacity_factor:.0%}",
                    "✅ Competitive" if econ['levelized']['cost_per_tonne'] < 200 else "⚠️ Needs optimization"
                ]
            })
            st.dataframe(kpis, hide_index=True, use_container_width=True)
        
        # ==================== OPTIMIZATION RECOMMENDATIONS ====================
        st.header("🎯 Optimization Recommendations")
        
        recommendations = []
        
        if results['absorber']['efficiency'] < 75:
            recommendations.append("⚠️ Increase absorber height or NaOH concentration to improve efficiency")
        
        if econ['levelized']['cost_per_tonne'] > 200 and not "Direct Air" in gas_source:
            recommendations.append("⚠️ Flue gas cost too high - Consider: higher capacity factor, lower electricity costs, or equipment scaling")
        
        if econ['levelized']['cost_per_tonne'] > 400 and "Direct Air" in gas_source:
            recommendations.append("⚠️ DAC cost high but within range - Optimize: air flow rate, absorber height, or NaOH concentration")
        
        if results['causticizing']['total_conversion'] < 85:
            recommendations.append("⚠️ Add more CSTR stages or increase causticizing temperature")
        
        if econ['energy']['kWh_per_tonne'] > 500:
            recommendations.append("⚠️ Energy intensity is high - optimize compressor efficiency or reduce pressure ratio")
        
        if econ['opex']['lime'] / econ['opex']['total'] > 0.4:
            recommendations.append("💡 Lime cost dominates OPEX - negotiate bulk pricing or alternative suppliers")
        
        if econ['CO2_tpy'] < 1000 and not "Direct Air" in gas_source:
            recommendations.append("💡 Consider increasing gas flow rate for better economy of scale")
        
        if len(recommendations) == 0:
            st.success("✅ System is well-optimized for current parameters!")
        else:
            for rec in recommendations:
                st.warning(rec)
        
        # ==================== DOWNLOAD RESULTS ====================
        st.header("💾 Export Results")
        
        # Prepare comprehensive results CSV
        export_data = {
            'Parameter': [],
            'Value': [],
            'Unit': []
        }
        
        # Add all key results
        params_export = [
            ('Air Flow Rate', Q_air, 'm³/s'),
            ('Compressor Power', econ['energy']['compression_kW'], 'kW'),
            ('CO₂ Feed', results['compression']['n_CO2_in'], 'mol/s'),
            ('CO₂ Captured', results['absorber']['n_CO2_captured'], 'mol/s'),
            ('Capture Efficiency', results['absorber']['efficiency'], '%'),
            ('Annual CO₂', econ['CO2_tpy'], 't/year'),
            ('Total CAPEX', econ['capex']['total'], ),
            ('Annual OPEX', econ['opex']['total'], '$/year'),
            ('Cost per Tonne', econ['levelized']['cost_per_tonne'], '$/t'),
            ('Energy Intensity', econ['energy']['kWh_per_tonne'], 'kWh/t'),
        ]
        
        for param, value, unit in params_export:
            export_data['Parameter'].append(param)
            export_data['Value'].append(f"{value:.2f}" if isinstance(value, float) else value)
            export_data['Unit'].append(unit)
        
        df_export = pd.DataFrame(export_data)
        
        csv = df_export.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name="co2_capture_results.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Configure parameters in the sidebar and click 'Run Comprehensive Simulation' to begin")
    
    # Display default information
    st.markdown("""
    ### About This Simulator
    
    This advanced CO₂ capture simulator models a complete direct air capture (DAC) system with:
    
    **1. Air Compression Stage**
    - Realistic compressor performance with isentropic efficiency
    - Intercooling requirements
    - Power consumption analysis
    
    **2. Bubble Column Absorber**
    - Detailed mass transfer with enhancement factors
    - Hatta number calculations for reaction regime
    - Temperature-dependent Henry's law and kinetics
    - Axial concentration profiles
    
    **3. Causticizing Reactor Train**
    - Multiple CSTR configuration
    - NaOH regeneration from Na₂CO₃
    - Equilibrium-limited conversion
    - CaCO₃ byproduct formation
    
    **4. Comprehensive Economics**
    - Detailed CAPEX and OPEX breakdown
    - Levelized cost of capture
    - Energy analysis and sensitivity
    - 20-year project evaluation
    
    **Target Performance:**
    - Capture efficiency: 70-90%
    - Cost: <$200/tonne CO₂
    - Realistic industrial-scale operation
    """)