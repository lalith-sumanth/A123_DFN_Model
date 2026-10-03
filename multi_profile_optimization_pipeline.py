"""Run beside optimization_model.py, data/ and optimization_inputs/"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import optimization_model as m

RESULTS = m.RESULTS
REFERENCE = m.START.copy(); REFERENCE[0] = 8.0
STEP = np.array([0.5, np.log10(1.15), np.log10(1.15), np.log10(1.15), 0.01, 0.02, 0.02, 0.10])
EARLY_END = 7200.0
PULSE_RMS_TOL_MV = 1.0
PULSE_MAX_TOL_MV = 5.0
REST_RMS_TOL_MV = 2.0
END_TOL_S = 1.0
TRAIN_IMPROVEMENT_MV = 0.05
SIMS = {name: m.build_simulation(name, m.FINAL_MESH)[0] for name in m.NAMES}
TRAIN_CACHE, HPPC_CACHE, LOG = {}, {}, []
REFERENCE_GATES, REFERENCE_END = {}, None

def vector_key(v,name):
    # Test-specific discharge initial history has no effect on the other tests.
    return tuple(np.round(v if name=='Discharge_1C' else v[:7],12))


def training_score(v):
    residuals=[];coverage={}
    for name in m.TRAIN:
        key=(name,vector_key(v,name))
        if key not in TRAIN_CACHE:
            d=m.DATA[name].iloc[m.FIT_ROWS[name]]
            sol,_=m.solve_profile(name,v,SIMS[name],d.Time.to_numpy())
            end=float(sol.t[-1]);valid=d.Time.to_numpy()<=end
            pred=np.full(len(d),float(sol['Voltage [V]'](end).reshape(-1)[0]))
            pred[valid]=sol['Voltage [V]'](d.Time.to_numpy()[valid]).reshape(-1)
            r=(pred-d.Voltage.to_numpy())*m.FIT_WEIGHTS[name]
            c=min(1.0,end/float(m.DATA[name].Time.iloc[-1]))
            TRAIN_CACHE[key]=(r,c)
        r,c=TRAIN_CACHE[key];residuals.extend(r);coverage[name]=c
        residuals.append(0.5*(1-c)/np.sqrt(len(m.TRAIN)))
    return float(1000*np.linalg.norm(residuals)),coverage


def hppc(v,end_time):
    key=(vector_key(v,'HPPC'),float(end_time))
    if key not in HPPC_CACHE:
        d=m.DATA['HPPC'];times=d.loc[d.Time<=end_time,'Time'].to_numpy()
        sol,_=m.solve_profile('HPPC',v,SIMS['HPPC'],times,end_time=end_time)
        end=float(sol.t[-1]);valid=times<=end
        frame=d.loc[d.Time<=end_time].copy();frame['Simulated_V']=np.nan
        frame.loc[valid,'Simulated_V']=sol['Voltage [V]'](times[valid]).reshape(-1)
        frame['Residual_mV']=1000*(frame.Simulated_V-frame.Voltage)
        HPPC_CACHE[key]=(frame,end,str(sol.termination))
    return HPPC_CACHE[key]


def gate(v,stage):
    target=EARLY_END if stage=='early' else float(m.DATA['HPPC'].Time.iloc[-1])
    frame,end,termination=hppc(v,target)
    limit=EARLY_END if stage=='early' else REFERENCE_END-END_TOL_S
    metrics=m.pulse_metrics(frame,limit,late=stage=='late',early_end=EARLY_END)
    reasons=[];ref=REFERENCE_GATES[stage]
    if end < limit:reasons.append('earlier HPPC cutoff')
    for group in ['charge_pulse','discharge_pulse','rest']:
        if metrics[group+'_valid_count']!=metrics[group+'_count']:
            reasons.append(group+' missing predictions');continue
        tol=REST_RMS_TOL_MV if group=='rest' else PULSE_RMS_TOL_MV
        if metrics[group+'_RMSE_mV']>ref[group+'_RMSE_mV']+tol:
            reasons.append(group+' RMSE worsened')
        if group!='rest' and metrics[group+'_max_error_mV']>ref[group+'_max_error_mV']+PULSE_MAX_TOL_MV:
            reasons.append(group+' peak error worsened')
    metrics.update(end_s=end,termination=termination)
    return not reasons,reasons,metrics


def attempt(v,label,best_score):
    row={'trial':len(LOG)+1,'change':label,**dict(zip(m.PARAMETER_NAMES,v.tolist()))}
    try:
        score,coverage=training_score(v)
        row.update(training_objective_mV=score,**{n+'_coverage':c for n,c in coverage.items()})
        early_ok,reasons,early=gate(v,'early')
        row.update(**{'early_'+k:value for k,value in early.items()})
        improving=score<best_score-TRAIN_IMPROVEMENT_MV
        if not improving:reasons.append('no sufficient training improvement')
        accepted=False
        if early_ok and improving:
            # Full replay checks later high-current charge/discharge pulses AND
            # coverage. Missing tails cannot be hidden behind a lower RMSE.
            late_ok,late_reasons,late=gate(v,'late')
            reasons.extend(late_reasons)
            row.update(**{'late_'+k:value for k,value in late.items()})
            accepted=late_ok
        row.update(accepted=accepted,reason='accepted' if accepted else '; '.join(reasons))
    except (m.pybamm.SolverError,ValueError) as error:
        score=np.inf;accepted=False
        row.update(training_objective_mV=None,accepted=False,reason='simulation error: '+str(error))
    LOG.append(row);pd.DataFrame(LOG).to_csv(RESULTS/'candidate_history.csv',index=False)
    print('trial',row['trial'],label,'score',round(score,3),'accepted',accepted,row['reason'],flush=True)
    return accepted,score


def test_group(best, score, proposals):
    # All alternatives start from the same point; keep the best accepted move.
    origin = best.copy(); selected = best.copy(); selected_score = score
    for label, changes in proposals:
        candidate = origin.copy()
        for index, change in changes:
            candidate[index] = np.clip(origin[index] + change, m.LOWER[index], m.UPPER[index])
        if np.array_equal(candidate, origin): continue
        accepted, new_score = attempt(candidate, label, selected_score)
        if accepted: selected, selected_score = candidate, new_score
    return selected, selected_score


def optimize():
    global REFERENCE_END, REFERENCE_GATES
    started = time.time()
    baseline, REFERENCE_END, _ = hppc(REFERENCE, float(m.DATA['HPPC'].Time.iloc[-1]))
    REFERENCE_GATES = {'early': m.pulse_metrics(baseline, EARLY_END),
                       'late': m.pulse_metrics(baseline, REFERENCE_END-END_TOL_S, late=True)}
    (RESULTS/'guard_definition.json').write_text(json.dumps({
        'reference_parameters': m.decode(REFERENCE), 'reference_end_s': REFERENCE_END,
        'reference_metrics': REFERENCE_GATES, 'high_current_threshold_A': 5.0,
        'early_window_s': [0, EARLY_END], 'late_window_s': [EARLY_END, REFERENCE_END-END_TOL_S],
        'pulse_RMSE_allowance_mV': PULSE_RMS_TOL_MV, 'pulse_max_error_allowance_mV': PULSE_MAX_TOL_MV,
        'rest_RMSE_allowance_mV': REST_RMS_TOL_MV, 'end_allowance_s': END_TOL_S,
        'HPPC_use': 'Calibration acceptance; not independent validation.'}, indent=2))
    best = REFERENCE.copy(); score, _ = training_score(best)
    print('Reference objective', score, 'mV; HPPC endpoint', REFERENCE_END/3600, 'h', flush=True)

    # Fit unknown discharge history before changing shared electrode parameters.
    def history_cost(a0):
        candidate = best.copy(); candidate[7] = a0
        return training_score(candidate)[0]
    history = minimize_scalar(history_cost, bounds=(0, 1), method='bounded',
                              options={'xatol': 0.005, 'maxiter': 18})
    candidate = best.copy(); candidate[7] = float(history.x)
    accepted, new_score = attempt(candidate, 'discharge initial history only', score)
    if accepted: best, score = candidate, new_score

    # Two coordinate sweeps: test both signs, then halve the step size.
    for sweep, factor in enumerate([1.0, 0.5], start=1):
        for index, name in enumerate(m.PARAMETER_NAMES):
            proposals = [(f'sweep {sweep}: {name} decrease', [(index, -factor*STEP[index])]),
                         (f'sweep {sweep}: {name} increase', [(index, factor*STEP[index])])]
            best, score = test_group(best, score, proposals)

    # Test the same seven related pairs as the original successful search.
    pairs = [[(1, 1), (5, 1)], [(1, 1), (4, 1)], [(2, 1), (5, 1)],
             [(2, -1), (3, 1)], [(5, 1), (6, 1)], [(4, -1), (6, -1)], [(1, -1), (2, 1)]]
    for pair in pairs:
        description = ' + '.join(m.PARAMETER_NAMES[i]+(' increase' if sign>0 else ' decrease') for i, sign in pair)
        proposals = [(f'paired step {factor}: '+description,
                      [(i, sign*factor*STEP[i]) for i, sign in pair]) for factor in [0.5, 0.25]]
        best, score = test_group(best, score, proposals)

    (RESULTS/'guarded_parameters.json').write_text(json.dumps({
        'parameter_names': m.PARAMETER_NAMES, 'optimizer_vector': best.tolist(),
        'physical_parameters': m.decode(best), 'reference_parameters': m.decode(REFERENCE),
        'training_objective_mV': score, 'runtime_s': time.time()-started,
        'candidate_trials': len(LOG), 'accepted_trials': int(sum(row['accepted'] for row in LOG)),
        'method': 'Bounded scalar history fit, coordinate sweeps and paired moves with HPPC checks.'}, indent=2))
    initial, final = m.decode(REFERENCE), m.decode(best)
    table = pd.DataFrame({'Parameter': list(final), 'Initial': list(initial.values()), 'Optimized': list(final.values())})
    table.to_csv(RESULTS/'parameter_comparison.csv', index=False)
    final['discharge_a0'] = final.pop('discharge_initial_LFP_charge_fraction')
    (RESULTS/'optimized_parameters.json').write_text(json.dumps(final, indent=2))
    print(table.to_string(index=False), flush=True)
    return best


if __name__ == '__main__':
    final = optimize()
    m.compare(final, REFERENCE, REFERENCE_END, EARLY_END, END_TOL_S)
