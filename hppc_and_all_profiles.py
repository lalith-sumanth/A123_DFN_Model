"""Assess the accepted local-history DFN on HPPC and all charge/discharge profiles."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pybamm
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"  # Change this folder to use your measurement CSVs.
OUTPUT_DIR = ROOT / "results" / "all_profiles"
PROFILES = ['Charge_1C', 'Charge_2C', 'Charge_3C', 'Charge_4C', 'Discharge_1C', 'HPPC']
USE_HYSTERESIS = True
BASE = pybamm.ParameterValues("Prada2013")
F = float(pybamm.constants.F.value)
AREA = BASE["Electrode height [m]"] * BASE["Electrode width [m]"]


def graphite_ocp(x):
    return (1.9793 * pybamm.exp(-39.3631 * x) + 0.2482
            - 0.0909 * pybamm.tanh(29.8538 * (x - 0.1234))
            - 0.04478 * pybamm.tanh(14.9159 * (x - 0.2769))
            - 0.0205 * pybamm.tanh(30.4444 * (x - 0.6103)))


def lfp_charge(y):
    return 3.451 - 0.009 * y + 0.6687 * pybamm.exp(-35 * y) - 0.5 * pybamm.exp(-210 * (1 - y))


def lfp_discharge(y):
    return 3.4077 - 0.020269 * y + 0.5 * pybamm.exp(-200 * y) - 0.9 * pybamm.exp(-30 * (1 - y))


def electrode_capacity(electrode, active_fraction):
    return (F * AREA * BASE[f"{electrode} electrode thickness [m]"] * active_fraction
            * BASE[f"Maximum concentration in {electrode.lower()} electrode [mol.m-3]"] / 3600)

# Accepted values from the previous multi-profile study; no new tuning loop.
ONE_C_PARAMETERS = {"R_Ohm": 0.016575597543094407, "Dn_m2_s": 1.7790482390716578e-14,
                    "j0n_A_m2": 7.5, "j0p_A_m2": 0.05, "eps_n": 0.58,
                    "eps_p": 0.374, "QLi_Ah": 2.731761840954426, "discharge_a0": 1.0}
PARAMETERS = {"R_Ohm": 0.008, "Dn_m2_s": 1.9756549841802243e-14,
              "j0n_A_m2": 8.328842386580629, "j0p_A_m2": 0.05, "eps_n": 0.5875,
              "eps_p": 0.394, "QLi_Ah": 2.7267618409544263, "discharge_a0": 0.3770652419539913}
HISTORY_CHARGE_AH = 0.05  # History approaches a branch over this transferred-charge scale.


def hppc_current(data):
    """Use recorded Ah to preserve current through sparse gaps and transitions."""
    time, current, charge = (data[k].to_numpy(float) for k in ["Time", "Current", "Ah"])
    if not np.isfinite(charge).all(): raise ValueError("HPPC Ah counter must be finite")
    knots, amps, repairs = [time[0]], [current[0]], []
    for k, dt in enumerate(np.diff(time)):
        start, end, a, b = time[k], time[k + 1], current[k], current[k + 1]
        delta_q = charge[k + 1] - charge[k]
        ramp = min(0.01, dt * 0.001); extra = []; method = "measured_linear"
        if dt > 30:
            plateau = (3600 * delta_q - 0.5 * ramp * (a + b)) / (dt - ramp)
            tolerance = 1e-4 * 3600 / dt + 0.01
            if not min(a, b) - tolerance <= plateau <= max(a, b) + tolerance:
                raise ValueError(f"HPPC gap {start:g}-{end:g}: inconsistent Ah/current")
            extra = [(start + ramp, plateau), (end - ramp, plateau)]
            method = "long_gap_plateau"
        elif dt >= 1 and abs(b - a) > 0.1 and abs(b - a) * dt / 3600 > 2e-4:
            switch = (3600 * delta_q - b * dt) / (a - b)
            uncertainty = 1e-4 * 3600 / abs(b - a)
            if -uncertainty <= switch <= dt + uncertainty:
                switch = np.clip(switch, ramp / 2, dt - ramp / 2)
                extra = [(start + switch - ramp / 2, a), (start + switch + ramp / 2, b)]
                method = "sparse_transition_switch"
        for t, i in extra:
            if knots[-1] < t < end: knots.append(float(t)); amps.append(float(i))
        knots.append(float(end)); amps.append(float(b))
        if extra:
            interval_t = [start] + [z[0] for z in extra] + [end]
            interval_i = [a] + [z[1] for z in extra] + [b]
            repairs.append({"start_s": start, "end_s": end, "method": method,
                            "target_Ah": delta_q, "input_Ah": np.trapezoid(interval_i, interval_t) / 3600})
    pd.DataFrame(repairs).to_csv(OUTPUT_DIR / "HPPC_current_audit.csv", index=False)
    return np.asarray(knots), np.asarray(amps)


def load_profile(name):
    filename = "HPPC_Converted.csv" if name == "HPPC" else name + ".csv"
    raw = pd.read_csv(DATA_DIR / filename)
    if not np.isfinite(raw[["Time", "Current", "Voltage"]].to_numpy(float)).all():
        raise ValueError(f"{name}: non-finite measurements")
    if (np.diff(raw.Time) < 0).any(): raise ValueError(f"{name}: time runs backwards")
    data = raw.drop_duplicates("Time", keep="last").reset_index(drop=True)
    data.Time -= data.Time.iloc[0]
    active = np.flatnonzero(abs(data.Current.to_numpy()) > 0.05)
    if len(active) == 0 or active[0] == 0: raise ValueError(f"{name}: initial rest required")
    rest_voltage = float(data.Voltage.iloc[:active[0]].tail(10).median())
    time, current = data.Time.to_numpy(float), data.Current.to_numpy(float)
    if name == "HPPC": time, current = hppc_current(data)
    # Removing exactly collinear knots preserves the measured linear input.
    slopes = np.diff(current) / np.diff(time)
    keep = np.r_[True, ~np.isclose(slopes[:-1], slopes[1:], atol=1e-12, rtol=1e-12), True]
    return data, time[keep], current[keep], rest_voltage


def initial_state(name, rest_voltage):
    qn = electrode_capacity("Negative", PARAMETERS["eps_n"])
    qp = electrode_capacity("Positive", PARAMETERS["eps_p"])
    a0 = PARAMETERS["discharge_a0"] if USE_HYSTERESIS and name.startswith("Discharge") else 1.0
    def error(x):
        y = (PARAMETERS["QLi_Ah"] - qn * x) / qp
        ocv = a0 * lfp_charge(pybamm.Scalar(y)) + (1 - a0) * lfp_discharge(pybamm.Scalar(y)) - graphite_ocp(pybamm.Scalar(x))
        return float(ocv.evaluate()) - rest_voltage
    lo = max(1e-6, (PARAMETERS["QLi_Ah"] - qp * (1 - 1e-6)) / qn)
    hi = min(1 - 1e-6, (PARAMETERS["QLi_Ah"] - qp * 1e-6) / qn)
    if lo >= hi: raise ValueError("No admissible electrode balance")
    x = brentq(error, lo, hi, xtol=1e-12)
    return x, (PARAMETERS["QLi_Ah"] - qn * x) / qp, a0, qp


def simulate(name, data, time, current, rest_voltage):
    x0, y0, a0, qp = initial_state(name, rest_voltage)
    p = BASE.copy()
    p.update({"Nominal cell capacity [A.h]": 2.5,
              "Negative electrode OCP [V]": graphite_ocp,
              "Positive electrode OCP [V]": lambda y: (lfp_charge(y) + lfp_discharge(y)) / 2,
              "Negative electrode exchange-current density [A.m-2]": PARAMETERS["j0n_A_m2"],
              "Positive electrode exchange-current density [A.m-2]": PARAMETERS["j0p_A_m2"],
              "Contact resistance [Ohm]": PARAMETERS["R_Ohm"],
              "Negative particle diffusivity [m2.s-1]": PARAMETERS["Dn_m2_s"],
              "Negative electrode active material volume fraction": PARAMETERS["eps_n"],
              "Positive electrode active material volume fraction": PARAMETERS["eps_p"],
              "Initial concentration in negative electrode [mol.m-3]": x0 * BASE["Maximum concentration in negative electrode [mol.m-3]"],
              "Initial concentration in positive electrode [mol.m-3]": y0 * BASE["Maximum concentration in positive electrode [mol.m-3]"],
              "Lower voltage cut-off [V]": 2.0, "Upper voltage cut-off [V]": 3.65,
              "Initial temperature [K]": 296.15, "Ambient temperature [K]": 296.15,
              "Reference temperature [K]": 296.15,
              "Current function [A]": pybamm.Interpolant(time, -current, pybamm.t)})
    options = {"contact resistance": "true", "open-circuit potential": "one-state hysteresis" if USE_HYSTERESIS else "single"}
    model = pybamm.lithium_ion.DFN(options)
    # Up=(1+h)/2*Up_charge+(1-h)/2*Up_discharge; initial h=2*a0-1.
    # PyBaMM updates h locally using reaction current, preserving partial history.
    p.update({"Negative electrode lithiation OCP [V]": graphite_ocp,
              "Negative electrode delithiation OCP [V]": graphite_ocp,
              "Positive electrode lithiation OCP [V]": lfp_discharge,
              "Positive electrode delithiation OCP [V]": lfp_charge,
              "Positive particle lithiation hysteresis decay rate": 2 * qp / HISTORY_CHARGE_AH,
              "Positive particle delithiation hysteresis decay rate": 2 * qp / HISTORY_CHARGE_AH,
              "Negative particle lithiation hysteresis decay rate": 1.0,
              "Negative particle delithiation hysteresis decay rate": 1.0,
              "Initial hysteresis state in positive electrode": 2 * a0 - 1,
              "Initial hysteresis state in negative electrode": -1.0}, check_already_exists=False)
    model.variables["Minimum negative surface potential [V]"] = pybamm.min(model.variables["Negative electrode surface potential difference [V]"])
    outputs = ["Voltage [V]", "X-averaged positive electrode hysteresis state", "Minimum negative surface potential [V]",
               "Minimum negative particle surface stoichiometry", "Maximum negative particle surface stoichiometry"]
    solver = pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-8, output_variables=outputs,
                               options={"dt_max": 1.0 if name == "HPPC" else 5.0, "dt_init": 1e-5})
    mesh = {"x_n": 20, "x_s": 15, "x_p": 20, "r_n": 30, "r_p": 30}
    if name == "HPPC": stops = time
    else:
        jumps = np.flatnonzero(abs(np.diff(current)) > 0.1)
        stops = time[np.unique(np.r_[jumps, jumps + 1])]
    stops = np.unique(np.r_[0, stops[stops < data.Time.iloc[-1]], data.Time.iloc[-1]])
    simulation = pybamm.Simulation(model, parameter_values=p, solver=solver, var_pts=mesh)
    solution = simulation.solve(stops, t_interp=data.Time.to_numpy())
    return solution, {"initial_x": x0, "initial_y": y0, "initial_a0": a0, "rest_voltage_V": rest_voltage}


def save_results(name, data, solution, state):
    end = float(solution.t[-1]); reached = data.Time.to_numpy() <= end + 1e-8
    data = data[["Time", "Current", "Voltage"]].copy(); data["Simulated_V"] = np.nan
    data.loc[reached, "Simulated_V"] = solution["Voltage [V]"](data.Time.to_numpy()[reached]).ravel()
    data["Residual_mV"] = 1000 * (data.Simulated_V - data.Voltage)
    data.to_csv(OUTPUT_DIR / (name + "_voltage.csv"), index=False)
    times = np.unique(np.r_[data.loc[reached, "Time"], end])
    states = pd.DataFrame({"Time": times})
    for variable in ["X-averaged positive electrode hysteresis state", "Minimum negative surface potential [V]",
                     "Minimum negative particle surface stoichiometry", "Maximum negative particle surface stoichiometry"]:
        states[variable] = solution[variable](times).ravel()
    states.to_csv(OUTPUT_DIR / (name + "_states.csv"), index=False)
    error = data.loc[reached, "Residual_mV"].to_numpy(); rmse = np.sqrt(np.mean(error**2))
    complete = end >= data.Time.iloc[-1] - 1e-5
    row = {"profile": name, **state, "end_s": end, "end_h": end / 3600, "complete": complete,
           "coverage_percent": 100 * end / data.Time.iloc[-1], "termination": str(solution.termination),
           "prefix_RMSE_mV": rmse, "full_RMSE_mV": rmse if complete else np.nan,
           "MAE_mV": np.mean(abs(error)), "bias_mV": np.mean(error), "max_error_mV": max(abs(error)),
           "initial_DFN_voltage_V": float(solution["Voltage [V]"](0).ravel()[0]),
           "min_anode_surface_potential_V": states["Minimum negative surface potential [V]"].min()}
    masks = {"rest": abs(data.Current) <= 0.05, "active": abs(data.Current) > 0.05}
    if name.startswith("Charge"):
        masks["CC"] = abs(data.Current) >= 0.95 * abs(data.Current).max()
        masks["CV"] = masks["active"] & ~masks["CC"]
    if name == "HPPC": masks.update({"charge_pulse": data.Current >= 5, "discharge_pulse": data.Current <= -5})
    for group, mask in masks.items():
        e = data.loc[mask & reached, "Residual_mV"].to_numpy()
        row[group + "_RMSE_mV"] = np.sqrt(np.mean(e**2)) if len(e) else np.nan
        row[group + "_samples"] = int(mask.sum()); row[group + "_reached_samples"] = len(e)
    print(f"{name}: prefix RMSE {rmse:.2f} mV; coverage {row['coverage_percent']:.2f}%; end {end / 3600:.4f} h; {solution.termination}", flush=True)
    return data, row


def plot_profiles(traces):
    for column, ylabel, filename in [("Simulated_V", "Voltage (V)", "voltage_plots.png"),
                                      ("Residual_mV", "Residual (mV)", "residual_plots.png")]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        for (name, data), ax in zip(traces.items(), axes.flat):
            scale = 3600 if name == "HPPC" else 60
            if column == "Simulated_V": ax.plot(data.Time / scale, data.Voltage, "k--", lw=1, label="Measured")
            else: ax.axhline(0, color="black", lw=0.7)
            ax.plot(data.Time / scale, data[column], lw=1, label="DFN")
            ax.set(title=name.replace("_", " "), xlabel="Time (h)" if name == "HPPC" else "Time (min)", ylabel=ylabel)
            ax.grid(alpha=0.3); ax.legend()
        for ax in axes.flat[len(traces):]: ax.set_visible(False)
        fig.savefig(OUTPUT_DIR / filename, dpi=160); plt.close(fig)


def plot_hppc_detail(data):
    pulses = data.loc[abs(data.Current) >= 5, "Time"]
    if pulses.empty: return
    start = max(0, pulses.iloc[0] - 60); end = start + 1200
    data = data[(data.Time >= start) & (data.Time <= end)]
    history = pd.read_csv(OUTPUT_DIR / "HPPC_states.csv")
    history = history[(history.Time >= start) & (history.Time <= end)]
    waveform = pd.read_csv(OUTPUT_DIR / "HPPC_current_input.csv")
    waveform = waveform[(waveform.Time >= start) & (waveform.Time <= end)]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout="constrained")
    axes[0].plot(data.Time / 60, data.Voltage, "k--", label="Measured")
    axes[0].plot(data.Time / 60, data.Simulated_V, label="DFN"); axes[0].legend()
    axes[0].set_ylabel("Voltage (V)")
    axes[1].plot(waveform.Time / 60, waveform.Current_CSV_A); axes[1].set_ylabel("Current (A)")
    axes[2].plot(history.Time / 60, (1 + history["X-averaged positive electrode hysteresis state"]) / 2)
    axes[2].set(xlabel="Time (min)", ylabel="LFP charge-branch weight", ylim=(-0.05, 1.05))
    for ax in axes: ax.grid(alpha=0.3)
    fig.savefig(OUTPUT_DIR / "HPPC_pulse_detail.png", dpi=160); plt.close(fig)


def main():
    if PARAMETERS["R_Ohm"] < 0.008: raise ValueError("Resistance must be at least 8 mOhm")
    for electrode, key in [("Negative", "eps_n"), ("Positive", "eps_p")]:
        if PARAMETERS[key] + BASE[f"{electrode} electrode porosity"] > 0.98:
            raise ValueError("Leave at least 2 percent inactive material")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({"Parameter": list(PARAMETERS), "Previous_1C": list(ONE_C_PARAMETERS.values()), "Final": list(PARAMETERS.values())})
    print(table.to_string(index=False), flush=True)
    table.to_csv(OUTPUT_DIR / "parameters.csv", index=False)
    traces, metrics = {}, []
    for name in PROFILES:
        data, time, current, rest_voltage = load_profile(name)
        pd.DataFrame({"Time": time, "Current_CSV_A": current}).to_csv(OUTPUT_DIR / (name + "_current_input.csv"), index=False)
        solution, state = simulate(name, data, time, current, rest_voltage)
        traces[name], row = save_results(name, data, solution, state); metrics.append(row)
    pd.DataFrame(metrics).to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    plot_profiles(traces)
    plot_hppc_detail(traces["HPPC"])
    print("Saved voltage plots, residuals, metrics, parameters and states to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
