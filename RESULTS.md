# A123 simplified-code results

The three original assessment scripts were executed. Their parameters and input handling reproduce the accepted baseline and multi-profile runs. Across all 17 runs, the largest RMSE difference is 0.00765 mV, and the largest endpoint difference is 0.00906 s. 196 checks passed.

| Profile | Baseline prefix RMSE (mV) | Baseline coverage | Final RMSE (mV) | Final coverage | Final endpoint |
|---|---:|---:|---:|---:|---|
| Charge 1C | 40.78 | 48.82% | 27.35 (full) | 100.00% | Complete |
| Charge 2C | 41.03 | 33.31% | 25.57 (full) | 100.00% | Complete |
| Charge 3C | 54.19 | 22.28% | 26.94 (full) | 100.00% | Complete |
| Charge 4C | 64.10 | 15.99% | 20.07 (full) | 100.00% | Complete |
| Discharge 1C | 219.63 | 65.04% | 26.49 (full) | 100.00% | Complete |
| HPPC | 69.30 | 10.47% | 35.82 (prefix) | 72.53% | 18.273575 h, UV cutoff |

HPPC stops at **65784.868 s = 18.273575 h**, with **72.529%** time coverage. Its **35.82 mV** error describes the reached part only. No full-record HPPC RMSE is claimed; the missing voltage tail is left empty. Baseline prefix errors and final full-record errors cover different intervals; `results/common_window_metrics.csv` supplies a comparison over identical reached windows.

HPPC reached-sample charge-pulse RMSE is **44.82 mV** (1716/2623 samples). Discharge-pulse RMSE is **84.08 mV** (2991/4022 samples). These use |I| ≥ 5 A in the corresponding direction; their errors are larger than the overall error, which includes many rest samples.

The unchanged 1C charge optimizer retains its tested **15.15 mV full-record RMSE** and **9.45 mV CC-phase RMSE**. The fixed multi-profile set uses different effective parameters and the local LFP history model; its 1C charge error above is not the charge-only optimizer result.

The `multi_profile_optimization.py` file supplies the previously accepted values directly and executes no optimizer loop. Its five charge/discharge predictions match the corresponding predictions from `hppc_and_all_profiles.py`. No optimizer loop was introduced in those fixed-value files. The separate pipeline below executes a real fit.

Plots: `results/voltage_comparison.png`, the voltage/residual figures in each case folder, and `results/all_profiles/HPPC_pulse_detail.png`. Parameter tables, measurement/prediction traces, history/state traces, phase errors, pulse sample counts, cutoff reasons, reconstruction audit and execution logs are included.

The accepted model completes the five charge/discharge records and is partially assessed on HPPC. Its early HPPC UV cutoff, larger pulse errors, assumed initial history and non-unique effective parameters remain limitations.

## Reproduced grouped optimization pipeline

`multi_profile_optimization_pipeline.py` executes the original successful search in a simpler 180-line main file. The companion `optimization_model.py` preserves the physical equations, original CSV loading, phase-balanced sampling, solver settings and plotting. It starts from the saved earlier 1C-based reference with the previously selected 8 mOhm resistance; final multi-profile values are not supplied to the search.

The full search and fresh detailed replays were executed. **All 45 proposal labels, parameter vectors, accepted/rejected decisions and rejection reasons reproduce the original record. Seven moves are accepted. The maximum difference between the original and simplified search's recorded trial objectives is 0.0 mV.** The resulting eight quantities match the fixed multi-profile case to numerical precision; the LFP exchange-current density prints as 0.049999999999999996 instead of 0.05 solely because of floating-point conversion from its logarithm.

The bounded discharge-history fit is followed by two coordinate sweeps and seven related pairs at two step sizes. Actual accepted moves are:

| Trial | Accepted move |
|---:|---|
| 1 | Fit discharge initial LFP charge-branch fraction to 0.377065241954 |
| 25 | Increase graphite active fraction to 0.585 |
| 32 | Increase graphite diffusivity and LFP active fraction together |
| 34 | Increase graphite diffusivity and graphite active fraction together |
| 36 | Increase graphite exchange-current density and LFP active fraction together |
| 43 | Decrease graphite active fraction and lithium inventory together |
| 45 | Decrease graphite diffusivity and increase graphite exchange-current density together |

Resistance remains 8 mOhm and LFP exchange-current density remains 0.05 A/m2 after testing alternatives. Both early and late pulse directions, pulse peak errors, rest errors and HPPC duration are checked before a proposed improving move can be accepted. The original allowances are retained: 1 mV for pulse RMSE, 5 mV for pulse peak error, 2 mV for rest RMSE and 1 second for duration. These are declared numerical/engineering tolerances, not measured noise estimates. They permit small deteriorations within the stated limits and do not promise improvement of every individual pulse.

The original objective is 251.561813 mV before correcting discharge history, 40.047436 mV after that correction, and 32.663682 mV at the final selected parameters. It gives equal weight to profiles and phases and includes an early-cutoff penalty plus objective-only held terminal-voltage tails. **It is not an all-sample RMSE.** Full/prefix and phase errors, coverage and cutoff reasons are independently reported in the metrics CSVs; plotted predictions after cutoff remain empty.

| HPPC pulse RMSE | Earlier reference | Reproduced optimized set |
|---|---:|---:|
| Early charge | 24.26 mV | 23.80 mV |
| Early discharge | 63.06 mV | 58.24 mV |
| Later charge | 48.58 mV | 44.90 mV |
| Later discharge | 82.33 mV | 80.19 mV |

Early means 0–2 h; late means 2 h to one second before the original reference cutoff. High-current masks use CSV current >=5 A for charge and <=-5 A for discharge. These are identical reached-window comparisons. The early charge group contains only eight measured samples; counts for all groups are in `pulse_comparison.csv`.

The reproduced optimized HPPC endpoint is **18.273574542 h**, with a lower-voltage cutoff and 72.529% coverage. All five optimized charge/discharge records complete. Its full voltage and detailed state traces, pulse errors and metrics reproduce the original successful run. **222 reproduction checks passed**, including all trial decisions, parameter bounds, initial lithium balance/rest-voltage matching, bounded history/stoichiometry, independently recalculated errors and preservation of the four original main files and requirements.

The new result folder is `results/optimization_pipeline/`. It contains `parameter_comparison.csv` (initial and optimized values), `optimized_parameters.json`, `guarded_parameters.json`, `guard_definition.json`, `candidate_history.csv`, `reproduction_trials.csv`, `verification.json`, `execution.log`, the starting/final full and common-window metrics, pulse comparisons, voltage/state traces, voltage/residual comparison plots, an HPPC voltage/error/current figure and an anode-potential figure. No final parameter is inserted as a fallback or as a fitting target.

This is a reproduced finite local search, not a global-optimum or material-identifiability claim. Initial history is an effective fitted state; actual microscopic preconditioning is not independently established. HPPC guides acceptance and is part of calibration rather than independent validation. The original 18.27-hour UV cutoff limitation remains. The earlier fixed-value scripts and the unchanged charge-only optimizer retain their existing results.
