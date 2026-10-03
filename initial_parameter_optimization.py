"""Simple 1C charge optimization."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pybamm
from scipy.optimize import least_squares

CSV_FILE_PATH = r"Charge_1C.csv"  # Replace with your CSV path.
OUTPUT_DIR = Path("results")
# Historical starting guess: graphite fraction, LFP fraction, R (ohm), Dn multiplier.
INITIAL_VALUES = np.array([0.07, 0.80, 0.015, 3.0])


def lfp_ocp(sto):
    return (3.4510 - 0.009 * sto + 0.6687 * pybamm.exp(-35 * sto)
            - 0.5 * pybamm.exp(-210 * (1 - sto)))


def graphite_ocp(sto):
    return (1.9793 * pybamm.exp(-39.3631 * sto) + 0.2482
            - 0.0909 * pybamm.tanh(29.8538 * (sto - 0.1234))
            - 0.04478 * pybamm.tanh(14.9159 * (sto - 0.2769))
            - 0.0205 * pybamm.tanh(30.4444 * (sto - 0.6103)))


BASE_PARAMETERS = pybamm.ParameterValues("Prada2013")
BASE_PARAMETERS.update({
    "Nominal cell capacity [A.h]": 2.5,
    "Negative electrode OCP [V]": graphite_ocp,
    "Positive electrode OCP [V]": lfp_ocp,
    "Negative electrode exchange-current density [A.m-2]": 7.5,
    "Positive electrode exchange-current density [A.m-2]": 0.05,
    "Contact resistance [Ohm]": 0.010,
    "Lower voltage cut-off [V]": 2.0, "Upper voltage cut-off [V]": 3.65,
    "Initial temperature [K]": 296.15, "Ambient temperature [K]": 296.15,
    "Reference temperature [K]": 296.15,
})


def run_model(values, current_function, end_time, sample_times):
    x0, y0, resistance, diffusion_multiplier = values
    parameters = BASE_PARAMETERS.copy()
    parameters.update({
        "Initial concentration in negative electrode [mol.m-3]":
            x0 * parameters["Maximum concentration in negative electrode [mol.m-3]"],
        "Initial concentration in positive electrode [mol.m-3]":
            y0 * parameters["Maximum concentration in positive electrode [mol.m-3]"],
        "Contact resistance [Ohm]": resistance,
        "Negative electrode diffusivity [m2.s-1]": 3e-15 * diffusion_multiplier,
        "Current function [A]": current_function,
    })
    model = pybamm.lithium_ion.DFN({"contact resistance": "true"})
    solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-8, options={"dt_max": 5.0})
    mesh = {"x_n": 20, "x_s": 15, "x_p": 20, "r_n": 30, "r_p": 30}
    simulation = pybamm.Simulation(model, parameter_values=parameters, solver=solver, var_pts=mesh)
    return simulation.solve([0, float(end_time)], t_interp=sample_times)


def optimize_parameters(time, measured_voltage, current_function):
    # Same 250 observed rows as the original fit; report final RMSE on all rows.
    rows = np.unique(np.linspace(0, len(time) - 1, 250).astype(int))
    def residuals(values):
        try:
            solution = run_model(values, current_function, time[-1], time[rows])
            if solution.t[-1] < time[-1] - 1e-5:
                return np.full(len(rows), 1 + (time[-1] - solution.t[-1]) / time[-1])
            return solution["Voltage [V]"](time[rows]).ravel() - measured_voltage[rows]
        except pybamm.SolverError:
            return np.full(len(rows), 2.0)
    # Minimizing squared residuals also minimizes RMSE. R cannot fall below 8 mOhm.
    return least_squares(residuals, INITIAL_VALUES,
        bounds=([0.02, 0.74, 0.008, 0.5], [0.15, 0.95, 0.050, 30.0]),
        x_scale=[0.05, 0.1, 0.01, 3.0], diff_step=0.001, max_nfev=30, verbose=1)


def main():
    data = pd.read_csv(CSV_FILE_PATH).drop_duplicates("Time", keep="last").sort_values("Time")
    time = data.Time.to_numpy() - data.Time.iloc[0]
    voltage = data.Voltage.to_numpy()
    current = -data.Current.to_numpy()  # CSV: positive charge; PyBaMM: negative charge.
    if not np.isfinite(np.column_stack([time, current, voltage])).all() or not (np.diff(time) > 0).all():
        raise ValueError("CSV must contain finite Time, Current and Voltage with increasing time.")
    if (current > 0.05).any(): raise ValueError("This script fits a charge-only profile.")
    current_function = pybamm.Interpolant(time, current, pybamm.t)
    fit = optimize_parameters(time, voltage, current_function)
    solution = run_model(fit.x, current_function, time[-1], time)
    if solution.t[-1] < time[-1] - 1e-5: raise ValueError("Fitted simulation stopped early.")
    predicted = solution["Voltage [V]"](time).ravel()
    error = predicted - voltage
    cc = abs(current) >= 0.95 * abs(current).max()
    print("Optimizer:", fit.message)
    print(f"Full-record RMSE: {1000 * np.sqrt(np.mean(error**2)):.2f} mV")
    print(f"CC-phase RMSE: {1000 * np.sqrt(np.mean(error[cc]**2)):.2f} mV")
    table = pd.DataFrame({"Parameter": ["Initial graphite fraction", "Initial LFP fraction", "R (ohm)", "Dn multiplier"],
                          "Initial": INITIAL_VALUES, "Optimized": fit.x})
    print(table.to_string(index=False)); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "parameters.csv", index=False)
    pd.DataFrame({"Time": time, "Measured_V": voltage, "Simulated_V": predicted,
                  "Residual_mV": 1000 * error}).to_csv(OUTPUT_DIR / "voltage.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(time / 60, voltage, "k--", label="Measured")
    axes[0].plot(time / 60, predicted, label="DFN fit"); axes[0].legend()
    axes[0].set_ylabel("Voltage (V)"); axes[1].plot(time / 60, 1000 * error)
    axes[1].axhline(0, color="black", lw=0.7); axes[1].set(xlabel="Time (min)", ylabel="Residual (mV)")
    for ax in axes: ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "voltage_plot.png", dpi=160); plt.show()


if __name__ == "__main__": main()
