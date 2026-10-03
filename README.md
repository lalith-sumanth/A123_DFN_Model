# Simple A123 DFN code

The files use plain functions, explicit parameters and CSV input. The four original files run independently. The separate optimization pipeline uses `optimization_model.py` for the physical model and plotting, keeping the main search short.

| File | What it does |
|---|---|
| `baseline_run.py` | Runs all six profiles using stock Prada physical parameters, the supplied OCP equations, constant exchange-current densities of 7.5 and 0.05 A/m², and 10 mΩ contact resistance. It uses the previous current-direction LFP branch rule. |
| `initial_parameter_optimization.py` | The unchanged 110-line 1C charge optimizer. Fits the two initial lithium fractions, resistance and graphite diffusivity multiplier. |
| `multi_profile_optimization.py` | Runs all four charges and 1C discharge with the accepted multi-profile parameter values and local LFP history. The values are supplied directly; no optimization loop runs in this file. |
| `multi_profile_optimization_pipeline.py` | Repeats the original staged optimization from the earlier 1C-based reference: discharge-history fitting, individual moves and paired moves with early/late HPPC checks. Requires `optimization_model.py` and the supplied input folders. |
| `optimization_model.py` | Companion model definitions, original phase-balanced sampling, rest-voltage initialization and result plots; it is imported by the pipeline. |
| `hppc_and_all_profiles.py` | Uses exactly the same accepted parameters and history model for HPPC and all five charge/discharge records. Also creates a pulse-detail plot with current and history. |

## Run

Extract the ZIP and run from this folder with Python 3.12:

```bash
python -m pip install -r requirements.txt
python baseline_run.py
python initial_parameter_optimization.py
python multi_profile_optimization.py
python hppc_and_all_profiles.py
python multi_profile_optimization_pipeline.py
```

For your files, edit `DATA_DIR` and `PROFILES` in the three assessment scripts. The 1C optimizer has its own `CSV_FILE_PATH`. Required CSV columns are `Time` (seconds), `Current` (amperes, positive on charge), and `Voltage` (volts). HPPC also needs the recorded cumulative `Ah` column. The baseline and history-model assessments require an initial rest so starting concentrations can be matched to its voltage. Keep the supplied profile names when replacing these measurement files.

The code keeps the last duplicate timestamp, shifts time to zero, checks the data and converts the current sign for PyBaMM. Charge and discharge inputs use the measured linear current interpolant. HPPC retains the previously audited processing: the Ah counter constrains 13 long gaps and informative sparse transitions. It does not create voltage measurements inside those gaps. The reconstructed current and repair audit are saved so they can be inspected.

Each new script saves a parameter table, voltage traces, state traces, complete/prefix errors, phase errors, coverage, cutoff reason, and voltage/residual figures in its own `results/` subfolder. The unchanged 1C optimizer writes its new outputs directly to `results/`; its included tested output is in `results/1C_charge/`. Run logs and a baseline/final voltage comparison are included.

## Parameters and initial states

The accepted multi-profile values are reproduced from the previous study. The original fixed-value scripts keep those values. The separate pipeline starts there and executes a genuine fit; its selected output is saved separately.

| Parameter | Why it affects the fit |
|---|---|
| Effective contact resistance, 8 mΩ | Adds current-dependent voltage loss through IR. Every script enforces the requested 8 mΩ minimum. This term does not represent all internal cell polarization. |
| Graphite diffusivity, 1.97565 × 10⁻¹⁴ m²/s | Controls concentration polarization and relaxation; diffusion time scales as particle-radius squared divided by diffusivity. |
| Graphite exchange-current density, 8.32884 A/m² | Controls charge-transfer overpotential. LFP remains at 0.05 A/m². |
| Active fractions, 0.5875 graphite and 0.394 LFP | Set both electrode capacity and reaction area. |
| Lithium inventory, 2.72676 Ah-equivalent | Sets electrode balance and the available stoichiometric range; it is not the rated cell capacity. |
| Discharge initial charge-branch weight, 0.377065 | Represents a partly mixed initial LFP history for the independently measured discharge. Charge and HPPC start with weight one. |

Particle sizes, maximum concentrations, geometry, positive diffusivity, porosities and electrolyte properties remain at stock Prada values. Each assessment solves two constraints: electrode lithium inventory and the median voltage of the last ten initial-rest samples. It does not take the simulated end of 1C charge as the start of discharge or HPPC.

## Partial history

The native PyBaMM one-state LFP model uses

`Up = (1+h)/2 * Up_charge + (1-h)/2 * Up_discharge`,

`dh/dt = gamma * ivol / (F * cs_max * eps) * (1-sign(ivol)*h)/2`.

Here `ivol` is the local volumetric reaction current, and `h` lies between −1 and 1. Initial `h = 2*a0-1`; for discharge this is approximately −0.24587. The history approaches the relevant branch as lithium reacts locally, rather than instantly switching according to terminal current. The charge scale is 0.05 Ah, with `gamma = 2*Qp/0.05`. Zero local reaction retains history; terminal rest can still contain local redistribution. The exported history is averaged through the positive electrode. Graphite's two branch equations remain identical, so its hysteresis voltage amplitude is zero.

Use the pinned PyBaMM 25.12.2 version: this code uses its post-25.10 dimensionless history-decay convention. The OCP expressions are the supplied Chen graphite equation and Afshar LFP branches, retained from the accepted model. This remains an isothermal DFN with one representative particle size per electrode.

See `RESULTS.md` for the actual errors and HPPC limitation. Earlier parameter selection used HPPC, so this is reproduction and assessment, not independent validation.

## Multi-parameter optimization pipeline

Run `python multi_profile_optimization_pipeline.py` from the extracted folder after installing `requirements.txt`. The main search is 180 lines of plain functions and loops. The companion `optimization_model.py` keeps the original physical model, CSV loading and result plotting separate. Results go to `results/optimization_pipeline/`. The other four main programs and their accepted parameters remain unchanged.

This file reproduces the original successful method, replacing the later joint least-squares/backtracking attempt. It starts from the saved earlier 1C-based reference in `optimization_inputs/1C_optimized_parameters.json`, with resistance set to the previously selected 8 mOhm reference and charged initial history. It does not initialize from the final multi-profile parameter set, fit toward those final numbers, or substitute them when a search fails. The saved seed is the earlier fit used in that study; it is not automatically replaced by a newly executed charge-only optimization.

The five training CSVs are loaded directly from `data/`. HPPC uses the exact cleaned measurements and audited current input from the original run, supplied in `optimization_inputs/`. Its 13-gap reconstruction and charge-balance audit are included. Keep those HPPC measurements and waveform paired; replacing the raw HPPC recording also requires regenerating its processed inputs. This preserves the original data processing rather than reconstructing a different waveform during optimization. For a different training dataset, change the CSV paths in `optimization_model.py` while retaining the profile names and required initial rest.

The objective gives equal weight to each profile and each available phase: CC, CV and rest for charge; active and rest for discharge. Each phase uses up to 120 existing measurement rows. The full current waveform is simulated. Thus long rest periods or densely sampled CC portions cannot dominate the CV and discharge fit. If a training simulation cuts off early, the objective temporarily holds its last predicted voltage for the sampled tail and adds a coverage penalty. This is only a fitting score: published traces leave the missing tail empty and reported full RMSE is provided only for completed records. The resulting objective is not an all-sample voltage RMSE.

The search first uses bounded scalar optimization to estimate the independent discharge's initial LFP charge-branch fraction. It then makes two coordinate sweeps, testing decreases and increases in all eight quantities; the second sweep halves the step sizes. Finally, seven explicit related parameter pairs are tested at half and quarter step sizes. Each pair's alternatives start from the same current point, and the best passing alternative becomes the next point. Resistance is in mOhm, diffusivity and exchange-current densities use base-10 logarithms, and other parameters use their physical fractions or Ah-equivalent units. Log steps keep positive quantities positive and make changes proportional.

The eight quantities are resistance, graphite diffusivity, graphite and LFP exchange-current densities, both active fractions, lithium inventory and discharge initial history. They control IR loss, concentration polarization, reaction kinetics, electrode capacity/reaction area, electrode balance and initial OCP history. A rejected isolated change can still participate in an accepted paired move. Parameters are not frozen merely because changing them might affect pulses. Particle sizes, maximum concentrations, geometry, electrolyte properties and the 0.05 Ah history charge scale stay at their original values.

Bounds are 8–25 mOhm for resistance, 1e-15–1e-12 m2/s for graphite diffusivity, 1–100 / 0.005–2 A/m2 for graphite/LFP exchange-current densities, 0.48–0.62 / 0.28–0.52 for graphite/LFP active fractions, 2.2–3.4 Ah-equivalent for lithium inventory, and 0–1 for discharge initial history. These are the original declared working bounds. Initial concentrations are recalculated at every trial from that record's rest voltage and the trial lithium inventory. Local LFP hysteresis and the post-25.10 decay convention are unchanged. HPPC and charge initial history remain fully on the charge branch; only discharge initial history is fitted.

Acceptance requires a training-objective improvement of at least 0.05 mV and an early HPPC check over 0–2 h. A passing, improving proposal then undergoes a full HPPC replay and a late check over 2 h to one second before the original reference cutoff. In each window, charge and discharge are checked separately using measured current >=5 A and <=-5 A. Pulse RMSE may increase by at most 1 mV, pulse peak absolute error by at most 5 mV, and rest RMSE by at most 2 mV, relative to the original reference. Duration may shorten by at most 1 second. These are declared tolerances, not measured noise estimates. Missing predictions fail the corresponding checks. Every accepted move must pass all these tests before the next search stage proceeds.

After search, fresh detailed simulations assess both the starting reference and fitted set at every measured timestamp. The outputs include the initial/final parameter table, every accepted/rejected proposal with reasons, the original reference checks, voltage/state traces, full/prefix/phase and common-window errors, pulse errors, coverage, cutoffs, voltage/residual comparisons, HPPC voltage/current/error and anode-potential plots. Local history and voltage-loss terms are exported in state CSVs. The independent reproduction checks are in `verification.json` and `reproduction_trials.csv`.

This is a finite, gradient-free local search with a bounded scalar history fit. Its objective, bounds, parameter order, paired choices and step sizes are preserved from the original method. It establishes a reproducible search path, not a global optimum or uniquely measured material properties. HPPC guides acceptance and therefore participates in calibration; it is not an independent holdout validation. See `RESULTS.md` for the actual executed outcome and the remaining HPPC cutoff limitation.
