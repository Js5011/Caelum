from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import io
import base64

app = Flask(__name__)

# ===================== ROUTES =====================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # --------------------- Bubble Column Inputs ---------------------
        D = float(request.form["D"])
        G = float(request.form["G"])
        y_CO2 = float(request.form["y_CO2"])
        NaOH_in = float(request.form["NaOH_in"])
        L = float(request.form["L"])
        H_sim = float(request.form["H_sim"])
        N_col = int(request.form["N_col"])

        # --------------------- Causticizing Inputs ---------------------
        V_total = float(request.form["V_total"])
        N = int(request.form["N"])
        F = float(request.form["F"])
        CaOH2_0 = float(request.form["CaOH2_0"])
        k = float(request.form["k"])
        a_s0 = float(request.form["a_s0"])
        eta_eq = float(request.form["eta_eq"])
        t_final = float(request.form["t_final"])

        # --------------------- Bubble Column Model ---------------------
        dz = H_sim / N_col
        z = np.linspace(0, H_sim, N_col)
        A = np.pi * (D / 2) ** 2
        vL = L / A
        vG = max(G / A, 0.05)

        R = 8.314
        T = 298
        P_total = 101325

        H_CO2 = 3.4e4
        kLa0 = 0.08
        k_rxn = 1.0
        n_rxn = 1

        def activity_coeff(NaOH):
            return 1 / (1 + NaOH / 2000)

        def Henry_CO2(T):
            return H_CO2 * (1 + 0.01 * (T - 298))

        C_CO2g = np.ones(N_col) * (y_CO2 * P_total / (R * T))
        C_CO2l = np.zeros(N_col)
        C_NaOH = np.ones(N_col) * NaOH_in
        C_Na2CO3 = np.zeros(N_col)

        alpha = 0.5
        tol = 1e-8
        max_iter = 20000

        for iteration in range(max_iter):
            old = np.vstack([C_CO2g, C_CO2l, C_NaOH, C_Na2CO3]).copy()
            for i in range(N_col):
                g_up = C_CO2g[i-1] if i > 0 else C_CO2g[0]
                l_up = C_CO2l[i-1] if i > 0 else 0
                naoh_up = C_NaOH[i-1] if i > 0 else NaOH_in
                na2_up = C_Na2CO3[i-1] if i > 0 else 0

                H_eff = Henry_CO2(T) / activity_coeff(naoh_up)
                P_CO2 = g_up * R * T
                C_star = P_CO2 / H_eff
                kLa_eff = kLa0 * (1 + 5 * naoh_up / (naoh_up + 2000))
                N_CO2 = kLa_eff * (C_star - l_up)
                dCO2_g = min(N_CO2 * dz / vG, g_up)
                new_g = g_up - dCO2_g
                dCO2_l = dCO2_g * (vG / vL)
                new_l = min(l_up + dCO2_l, C_star)
                r_rxn = k_rxn * new_l * (naoh_up ** n_rxn)
                dNaOH = min(2 * r_rxn * dz / vL, naoh_up)

                C_CO2g[i] = alpha * new_g + (1 - alpha) * C_CO2g[i]
                C_CO2l[i] = alpha * new_l + (1 - alpha) * C_CO2l[i]
                C_NaOH[i] = alpha * (naoh_up - dNaOH) + (1 - alpha) * C_NaOH[i]
                C_Na2CO3[i] = alpha * (na2_up + dNaOH / 2) + (1 - alpha) * C_Na2CO3[i]

            if np.max(np.abs(old - np.vstack([C_CO2g, C_CO2l, C_NaOH, C_Na2CO3]))) < tol:
                break

        capture = 1 - C_CO2g[-1] / C_CO2g[0]
        total_Na2CO3 = C_Na2CO3[-1] * A * H_sim
        C_Na2CO3_in = C_Na2CO3[-1]
        C_NaOH_in = C_NaOH[-1]

        # --------------------- Causticizing Multi-CSTR ---------------------
        V = V_total / N

        def multi_cstr(t, y):
            dydt = np.zeros_like(y)
            for i in range(N):
                idx = i * 4
                C_Na2CO3, C_NaOH, CaOH2_s, CaCO3_s = y[idx:idx+4]

                if i == 0:
                    Na2CO3_in_i = C_Na2CO3_in
                    NaOH_in_i = C_NaOH_in
                else:
                    Na2CO3_in_i = y[(i-1)*4]
                    NaOH_in_i = y[(i-1)*4 + 1]

                a_s = a_s0 * (CaOH2_s / CaOH2_0)**(2/3) if CaOH2_s > 0 else 0
                r = k * a_s * C_Na2CO3
                eta = C_NaOH / (C_NaOH + C_Na2CO3 + 1e-12)
                r_eff = r * max(0, 1 - eta / eta_eq)

                dydt[idx]   = (F/V)*(Na2CO3_in_i - C_Na2CO3) - r_eff
                dydt[idx+1] = (F/V)*(NaOH_in_i - C_NaOH) + 2*r_eff
                dydt[idx+2] = -r_eff
                dydt[idx+3] = r_eff
            return dydt

        y0 = []
        for _ in range(N):
            y0.extend([C_Na2CO3_in, C_NaOH_in, CaOH2_0, 0.0])

        t_eval = np.linspace(0, t_final, 600)
        sol = solve_ivp(multi_cstr, [0, t_final], y0, t_eval=t_eval, method="BDF")

        # --------------------- Plot Function ---------------------
        def plot_to_img(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        # Bubble Column Plot
        fig1, ax1 = plt.subplots()
        ax1.plot(C_NaOH, z, label='NaOH')
        ax1.plot(C_Na2CO3, z, label='Na₂CO₃')
        ax1.set_xlabel('Concentration (mol/m³)')
        ax1.set_ylabel('Column height (m)')
        ax1.set_title('CO₂ Capture Profiles')
        ax1.legend()
        ax1.grid(True)
        plot1 = plot_to_img(fig1)
        plt.close(fig1)

        # Multi-CSTR plots
        labels = ["Na₂CO₃", "NaOH", "Ca(OH)₂", "CaCO₃"]
        cstr_plots = []
        for s in range(4):
            fig, ax = plt.subplots()
            for i in range(N):
                ax.plot(sol.t, sol.y[i*4+s], label=f"Reactor {i+1}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Concentration (mol/m³)")
            ax.set_title(labels[s])
            ax.legend()
            ax.grid(True)
            cstr_plots.append(plot_to_img(fig))
            plt.close(fig)

        # Carbon Capture Effectiveness Plot
        fig_cc, ax_cc = plt.subplots()
        for i in range(N):
            ax_cc.plot(sol.t, sol.y[i*4+3] / (C_Na2CO3_in * V_total) * 100, label=f"Reactor {i+1}")
        ax_cc.set_xlabel("Time (s)")
        ax_cc.set_ylabel("Carbon Capture (%)")
        ax_cc.set_title("Carbon Capture Effectiveness")
        ax_cc.legend()
        ax_cc.grid(True)
        plot_cc = plot_to_img(fig_cc)
        plt.close(fig_cc)

        return render_template("results.html",
                               capture=capture,
                               NaOH_out=C_NaOH[-1],
                               Na2CO3_out=C_Na2CO3[-1],
                               total_Na2CO3=total_Na2CO3,
                               plot1=plot1,
                               cstr_plots=cstr_plots,
                               plot_cc=plot_cc,
                               N=N)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
