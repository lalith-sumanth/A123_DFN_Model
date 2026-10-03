"""A123 DFN: fit four charge profiles and one discharge profile, then check HPPC.

Model, CSV loading, OCPs and result plots for the grouped optimization pipeline.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pybamm
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
RESULTS = ROOT / 'results' / 'optimization_pipeline'
RESULTS.mkdir(parents=True, exist_ok=True)
TRAIN = ['Charge_1C', 'Charge_2C', 'Charge_3C', 'Charge_4C', 'Discharge_1C']
NAMES = TRAIN + ['HPPC']
REFERENCE = json.loads((ROOT / 'optimization_inputs' / '1C_optimized_parameters.json').read_text())
REST = pd.read_csv(ROOT / 'optimization_inputs' / 'initial_states_reference.csv').set_index('profile')
DATA = {'HPPC': pd.read_csv(ROOT / 'optimization_inputs' / 'HPPC_measured.csv')}
WAVE = {'HPPC': pd.read_csv(ROOT / 'optimization_inputs' / 'HPPC_current_input.csv')}
# The optimizer takes the five original training CSVs directly. Supplied cleaned
# copies remain available for auditing. HPPC uses the earlier gap reconstruction.
for name in TRAIN:
    d=pd.read_csv(ROOT/'data'/f'{name}.csv')
    d=d.drop_duplicates('Time').dropna(subset=['Time','Current','Voltage']).copy()
    d.Time=d.Time-d.Time.iloc[0]
    if not np.all(np.diff(d.Time)>0):
        raise ValueError(f'{name}: time must increase after duplicate removal.')
    DATA[name]=d[['Time','Current','Voltage']].reset_index(drop=True)
    WAVE[name]=pd.DataFrame({'Time':d.Time,'Current_PyBaMM_A':-d.Current})
    first_active=np.flatnonzero(np.abs(d.Current.to_numpy())>0.05)[0]
    initial_rest=d.iloc[:first_active]
    REST.loc[name,'rest_voltage_V']=initial_rest.Voltage.tail(10).median()
    REST.loc[name,'rest_duration_s']=initial_rest.Time.iloc[-1]

# Existing literature OCPs: their shapes are not fitted to individual tests.
def graphite(x):
    return (1.9793 * pybamm.exp(-39.3631*x) + 0.2482
            - 0.0909*pybamm.tanh(29.8538*(x-0.1234))
            - 0.04478*pybamm.tanh(14.9159*(x-0.2769))
            - 0.0205*pybamm.tanh(30.4444*(x-0.6103)))

def lfp_charge(y):
    return 3.4510-0.009*y+0.6687*pybamm.exp(-35*y)-0.5*pybamm.exp(-210*(1-y))

def lfp_discharge(y):
    return 3.4077-0.020269*y+0.5*pybamm.exp(-200*y)-0.9*pybamm.exp(-30*(1-y))

BASE = pybamm.ParameterValues('Prada2013')
BASE.update({'Nominal cell capacity [A.h]': 2.5,
             'Negative electrode OCP [V]': graphite,
             'Positive electrode OCP [V]': lambda y: (lfp_charge(y)+lfp_discharge(y))/2,
             'Lower voltage cut-off [V]': 2.0, 'Upper voltage cut-off [V]': 3.65,
             'Initial temperature [K]': 296.15, 'Ambient temperature [K]': 296.15,
             'Reference temperature [K]': 296.15})
F = 96485.33212
AREA = BASE['Electrode height [m]'] * BASE['Electrode width [m]']
def electrode_capacity(electrode, active_fraction):
    return (F*AREA*BASE[f'{electrode} electrode thickness [m]']*active_fraction
            *BASE[f'Maximum concentration in {electrode.lower()} electrode [mol.m-3]']/3600)

QN_REFERENCE = electrode_capacity('Negative', 0.58)
QP_REFERENCE = electrode_capacity('Positive', 0.374)
QLI_REFERENCE = (QN_REFERENCE*REFERENCE['initial_negative_fraction']
                 + QP_REFERENCE*REFERENCE['initial_positive_fraction'])

# Seven shared physical/effective parameters plus ONE test-specific history state.
# Logs keep the positive parameters
# well scaled; resistance is in mOhm and inventory in Ah-equivalent.
PARAMETER_NAMES = ['R_mOhm', 'log10_Dn', 'log10_j0n', 'log10_j0p',
                   'negative_active_fraction', 'positive_active_fraction', 'lithium_inventory_Ah',
                   'discharge_initial_LFP_charge_fraction']
START = np.array([1000*REFERENCE['contact_resistance_Ohm'],
                  np.log10(REFERENCE['negative_diffusivity_m2_s']), np.log10(7.5),
                  np.log10(0.05), 0.58, 0.374, QLI_REFERENCE, 1.0])
LOWER = np.array([8, -15, np.log10(1), np.log10(0.005), 0.48, 0.28, 2.2, 0.0])
UPPER = np.array([25, -12, np.log10(100), np.log10(2), 0.62, 0.52, 3.4, 1.0])
SCALE = np.array([10, 1, 1, 1, 0.1, 0.1, 0.5, 1])
Q_HYSTERESIS_AH = 0.05  # fixed prior choice; HPPC informed this earlier study

def decode(v):
    return {'R_Ohm': float(v[0]/1000), 'Dn_m2_s': float(10**v[1]),
            'j0n_A_m2': float(10**v[2]), 'j0p_A_m2': float(10**v[3]),
            'eps_n': float(v[4]), 'eps_p': float(v[5]), 'QLi_Ah': float(v[6]),
            'discharge_initial_LFP_charge_fraction':float(v[7])}

def initialize(name, p, history_weight=None):
    """Use this test's rest voltage and the fitted common lithium inventory."""
    qn, qp = electrode_capacity('Negative', p['eps_n']), electrode_capacity('Positive', p['eps_p'])
    # Only discharge initial history is fitted; charge and HPPC retain their
    # previous charged-history assumption. HPPC has no fitted initial history.
    a0=p['discharge_initial_LFP_charge_fraction'] if name=='Discharge_1C' else 1.0
    if history_weight is not None:
        a0=float(history_weight)
    def error(x):
        y = (p['QLi_Ah']-qn*x)/qp
        vp=a0*lfp_charge(pybamm.Scalar(y))+(1-a0)*lfp_discharge(pybamm.Scalar(y))
        return float((vp-graphite(pybamm.Scalar(x))).evaluate())-REST.loc[name,'rest_voltage_V']
    lo = max(1e-6, (p['QLi_Ah']-qp*(1-1e-6))/qn)
    hi = min(1-1e-6, (p['QLi_Ah']-qp*1e-6)/qn)
    x = brentq(error, lo, hi, xtol=1e-12)
    y = (p['QLi_Ah']-qn*x)/qp
    return x, y, qn, qp, error(x)

# Select actual measured samples. Each profile and each available phase gets
# equal weight; dense CC samples cannot overwhelm CV or rest samples.
def phase_masks(name, data):
    current = np.abs(data.Current.to_numpy())
    rest = current <= 0.05
    if name.startswith('Charge'):
        cc = current >= 0.9*current.max()
        return {'CC': cc, 'CV': (~rest)&(~cc), 'rest': rest}
    return {'active': ~rest, 'rest': rest}

FIT_ROWS, FIT_WEIGHTS = {}, {}
for name in TRAIN:
    indices, weights = [], []
    groups = [np.flatnonzero(m) for m in phase_masks(name, DATA[name]).values() if m.any()]
    for group in groups:
        # Up to 120 measured samples per phase, distributed over that phase.
        selected = group[np.unique(np.linspace(0,len(group)-1,min(120,len(group))).astype(int))]
        indices.extend(selected)
        weights.extend(np.full(len(selected), 1/np.sqrt(len(TRAIN)*len(groups)*len(selected))))
    order = np.argsort(indices)
    FIT_ROWS[name] = np.array(indices)[order]
    FIT_WEIGHTS[name] = np.array(weights)[order]

def build_simulation(name, mesh, detailed=False):
    """Build once per profile; InputParameters avoid rebuilding during fitting."""
    model = pybamm.lithium_ion.DFN({'contact resistance':'true',
                                  'open-circuit potential':('one-state hysteresis','one-state hysteresis')})
    p = BASE.copy()
    p.update({'Contact resistance [Ohm]': '[input]',
              'Negative electrode diffusivity [m2.s-1]': '[input]',
              'Negative electrode exchange-current density [A.m-2]': '[input]',
              'Positive electrode exchange-current density [A.m-2]': '[input]',
              'Negative electrode active material volume fraction': '[input]',
              'Positive electrode active material volume fraction': '[input]',
              'Initial concentration in negative electrode [mol.m-3]': '[input]',
              'Initial concentration in positive electrode [mol.m-3]': '[input]',
              'Current function [A]': pybamm.Interpolant(WAVE[name].Time.to_numpy(),
                                  WAVE[name].Current_PyBaMM_A.to_numpy(), pybamm.t)})
    # Zero graphite amplitude; retain the native submodel for a simple consistent
    # implementation. The graphite history state therefore has no voltage effect.
    p.update({'Negative electrode lithiation OCP [V]': graphite,
              'Negative electrode delithiation OCP [V]': graphite,
              'Positive electrode lithiation OCP [V]': lfp_discharge,
              'Positive electrode delithiation OCP [V]': lfp_charge,
              'Positive particle lithiation hysteresis decay rate': '[input]',
              'Positive particle delithiation hysteresis decay rate': '[input]',
              'Negative particle lithiation hysteresis decay rate': 2*QN_REFERENCE/Q_HYSTERESIS_AH,
              'Negative particle delithiation hysteresis decay rate': 2*QN_REFERENCE/Q_HYSTERESIS_AH,
              'Initial hysteresis state in positive electrode': '[input]',
              'Initial hysteresis state in negative electrode': -1},check_already_exists=False)
    outputs = ['Voltage [V]']
    if detailed:
        model.variables['Minimum negative electrode surface potential difference [V]'] = pybamm.min(
            model.variables['Negative electrode surface potential difference [V]'])
        outputs += ['Battery open-circuit voltage [V]',
            'Negative electrode stoichiometry', 'Positive electrode stoichiometry',
            'Minimum negative particle surface stoichiometry',
            'Maximum negative particle surface stoichiometry',
            'X-averaged negative electrode open-circuit potential [V]',
            'X-averaged negative electrode surface potential difference [V]',
            'Minimum negative electrode surface potential difference [V]',
            'X-averaged positive electrode surface potential difference [V]',
            'X-averaged positive electrode hysteresis state',
            'Battery particle concentration overpotential [V]',
            'X-averaged battery reaction overpotential [V]',
            'X-averaged battery electrolyte ohmic losses [V]',
            'X-averaged battery solid phase ohmic losses [V]',
            'X-averaged battery concentration overpotential [V]']
    solver = pybamm.IDAKLUSolver(rtol=1e-6,atol=1e-8,output_variables=outputs,
                     options={'dt_max':1.0 if name=='HPPC' else 5.0, 'dt_init':1e-5})
    sim = pybamm.Simulation(model,parameter_values=p,solver=solver,var_pts=mesh)
    sim.build()
    return sim, outputs

def solve_profile(name, vector, simulation, times, history_weight=None, end_time=None):
    p = decode(vector)
    if p['R_Ohm'] < 0.008-1e-12:
        raise ValueError('Contact resistance must be at least 8 mOhm.')
    x, y, qn, qp, root_error = initialize(name,p,history_weight)
    a0=p['discharge_initial_LFP_charge_fraction'] if name=='Discharge_1C' else 1.0
    if history_weight is not None:
        a0=float(history_weight)
    inputs = {'Contact resistance [Ohm]':p['R_Ohm'],
        'Negative electrode diffusivity [m2.s-1]':p['Dn_m2_s'],
        'Negative electrode exchange-current density [A.m-2]':p['j0n_A_m2'],
        'Positive electrode exchange-current density [A.m-2]':p['j0p_A_m2'],
        'Negative electrode active material volume fraction':p['eps_n'],
        'Positive electrode active material volume fraction':p['eps_p'],
        'Initial concentration in negative electrode [mol.m-3]':x*BASE['Maximum concentration in negative electrode [mol.m-3]'],
        'Initial concentration in positive electrode [mol.m-3]':y*BASE['Maximum concentration in positive electrode [mol.m-3]'],
        'Positive particle lithiation hysteresis decay rate':2*qp/Q_HYSTERESIS_AH,
        'Positive particle delithiation hysteresis decay rate':2*qp/Q_HYSTERESIS_AH,
        'Initial hysteresis state in positive electrode':2*a0-1}
    wave = WAVE[name]
    if name=='HPPC':
        stops = wave.Time.to_numpy()
    else:
        jumps = np.flatnonzero(np.abs(np.diff(wave.Current_PyBaMM_A))>0.1)
        stops = wave.Time.iloc[np.unique(np.r_[jumps,jumps+1])].to_numpy()
    end_time=float(DATA[name].Time.iloc[-1]) if end_time is None else float(end_time)
    stops = np.unique(np.r_[0,stops[stops<end_time],end_time])
    solution = simulation.solve(stops,inputs=inputs,t_interp=times)
    return solution, {'initial_x':x,'initial_y':y,'Qn_Ah':qn,'Qp_Ah':qp,
                      'inventory_Ah':qn*x+qp*y,'root_error_V':root_error,
                      'initial_LFP_charge_branch_weight':a0}

COARSE_MESH = {'x_n':12,'x_s':8,'x_p':12,'r_n':20,'r_p':20}
FINAL_MESH = {'x_n':20,'x_s':15,'x_p':20,'r_n':30,'r_p':30}

# Evaluate at every measured timestamp. A cutoff creates an explicitly missing
# prediction tail; objective-only extrapolation is NEVER used in reported plots.
def evaluate(vector, label, mesh=FINAL_MESH, names=NAMES, hppc_history_weight=None):
    metrics=[]
    for name in names:
        started=time.time()
        sim, outputs=build_simulation(name,mesh,detailed=True)
        data=DATA[name].copy()
        history=hppc_history_weight if name=='HPPC' else None
        sol,state=solve_profile(name,vector,sim,data.Time.to_numpy(),history)
        end=float(sol.t[-1]); valid=data.Time.to_numpy()<=end
        data['Simulated_V']=np.nan
        data.loc[valid,'Simulated_V']=sol['Voltage [V]'](data.Time.to_numpy()[valid]).reshape(-1)
        data['Residual_mV']=1000*(data.Simulated_V-data.Voltage)
        data.to_csv(RESULTS/f'{name}_{label}_voltage.csv',index=False)
        times=np.unique(np.r_[data.Time.to_numpy()[valid],end])
        states=pd.DataFrame({'Time':times})
        for variable in outputs:
            states[variable]=sol[variable](times).reshape(-1)
        states['Imposed_CSV_current_A']=-np.interp(times,WAVE[name].Time,WAVE[name].Current_PyBaMM_A)
        states['Contact_voltage_contribution_V']=decode(vector)['R_Ohm']*states.Imposed_CSV_current_A
        states.to_csv(RESULTS/f'{name}_{label}_states.csv',index=False)
        qt=np.unique(np.r_[WAVE[name].loc[WAVE[name].Time<=end,'Time'],end])
        qi=-np.interp(qt,WAVE[name].Time,WAVE[name].Current_PyBaMM_A)
        complete=end>=float(data.Time.iloc[-1])-1e-5
        row={'profile':name,'variant':label,**state,'end_s':end,
             'requested_end_s':float(data.Time.iloc[-1]),'complete':bool(complete),
             'time_coverage_percent':100*end/float(data.Time.iloc[-1]),
             'termination':str(sol.termination),'final_voltage_V':float(states['Voltage [V]'].iloc[-1]),
             'net_discharge_Ah':float(-np.trapezoid(qi,qt)/3600),
             'initial_DFN_voltage_V':float(sol['Voltage [V]'](0).reshape(-1)[0]),
             'minimum_anode_surface_potential_V':float(states['Minimum negative electrode surface potential difference [V]'].min())}
        e=data.loc[valid,'Residual_mV'].to_numpy()
        row.update(prefix_RMSE_mV=float(np.sqrt(np.mean(e**2))),
                   prefix_MAE_mV=float(np.mean(abs(e))),prefix_bias_mV=float(np.mean(e)),
                   prefix_max_absolute_error_mV=float(np.max(abs(e))),
                   full_RMSE_mV=float(np.sqrt(np.mean(e**2))) if complete else None)
        t=data.loc[valid,'Time'].to_numpy()
        row['prefix_time_weighted_RMSE_mV']=float(np.sqrt(np.trapezoid(e**2,t)/(t[-1]-t[0])))
        for phase,mask in phase_masks(name,data).items():
            v=data.loc[mask&valid,'Residual_mV'].dropna().to_numpy()
            row[phase+'_prefix_RMSE_mV']=float(np.sqrt(np.mean(v**2))) if len(v) else None
        metrics.append(row)
        print(label,name,'end_s',round(end,2),'RMSE_mV',round(row['prefix_RMSE_mV'],2),
              'complete',complete,'runtime_s',round(time.time()-started,1),flush=True)
    pd.DataFrame(metrics).to_csv(RESULTS/f'{label}_metrics.csv',index=False)
    return pd.DataFrame(metrics)



def pulse_metrics(frame,limit,late=False,early_end=7200.0,high_current=5.0):
    d=frame[(frame.Time<=limit)&(frame.Time>early_end if late else frame.Time>=0)]
    masks={'charge_pulse':d.Current>=high_current,
           'discharge_pulse':d.Current<=-high_current,
           'rest':abs(d.Current)<=0.05}
    metrics={}
    for group,mask in masks.items():
        e=d.loc[mask,'Residual_mV']
        metrics[group+'_count']=int(len(e))
        metrics[group+'_valid_count']=int(e.notna().sum())
        metrics[group+'_RMSE_mV']=float(np.sqrt(np.mean(e.dropna()**2))) if e.notna().any() else np.nan
        metrics[group+'_max_error_mV']=float(abs(e.dropna()).max()) if e.notna().any() else np.nan
    return metrics


def compare(final, reference, reference_end, early_end=7200.0, end_tolerance=1.0):
    # Fresh detailed model replay at every measured timestamp.
    baseline=evaluate(reference,'reference')
    guarded=evaluate(final,'pulse_checked')
    all_metrics=pd.concat([baseline,guarded],ignore_index=True)
    for name in NAMES:
        end=all_metrics.loc[all_metrics.profile==name,'end_s'].min()
        for label in ['reference','pulse_checked']:
            d=pd.read_csv(RESULTS/f'{name}_{label}_voltage.csv')
            e=d.loc[(d.Time<=end+1e-9)&(d.Time>REST.loc[name,'rest_duration_s']+1e-9),'Residual_mV'].dropna()
            all_metrics.loc[(all_metrics.profile==name)&(all_metrics.variant==label),'common_window_RMSE_mV']=np.sqrt(np.mean(e**2))
    all_metrics.to_csv(RESULTS/'comparison_metrics.csv',index=False)
    rows=[]
    for label in ['reference','pulse_checked']:
        d=pd.read_csv(RESULTS/f'HPPC_{label}_voltage.csv')
        for phase in ['early','late']:
            limit=early_end if phase=='early' else reference_end-end_tolerance
            rows.append({'variant':label,'window':phase,**pulse_metrics(d,limit,late=phase=='late',early_end=early_end)})
    pd.DataFrame(rows).to_csv(RESULTS/'pulse_comparison.csv',index=False)
    for kind in ['voltage','residual']:
        fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
        for name,ax in zip(NAMES,axes.flat):
            factor=3600 if name=='HPPC' else 60
            d=DATA[name]
            if kind=='voltage':ax.plot(d.Time/factor,d.Voltage,color='black',lw=1,label='Measured')
            else:ax.axhline(0,color='black',lw=.7)
            for label,color,text in [('reference','#d97732','Previous 18.27 h reference'),('pulse_checked','#087f8c','Pulse-checked fit')]:
                d=pd.read_csv(RESULTS/f'{name}_{label}_voltage.csv')
                ax.plot(d.Time/factor,d.Simulated_V if kind=='voltage' else d.Residual_mV,color=color,lw=1,label=text)
            ax.set(title=name.replace('_',' '),xlabel='Time (h)' if name=='HPPC' else 'Time (min)',ylabel='Voltage (V)' if kind=='voltage' else 'Prediction − measured (mV)');ax.grid(alpha=.2)
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=3)
        fig.suptitle('Multi profile optimization with HPPC pulse acceptance checks')
        fig.savefig(RESULTS/f'all_profiles_{kind}.png',dpi=180);plt.close(fig)
    # Explicit pulse views and errors in each current direction.
    fig,axes=plt.subplots(3,1,figsize=(14,9),sharex=True,layout='constrained')
    d=DATA['HPPC'];axes[0].plot(d.Time/3600,d.Voltage,color='black',lw=.7,label='Measured')
    for label,color,text in [('reference','#d97732','Reference'),('pulse_checked','#087f8c','Pulse-checked')]:
        d=pd.read_csv(RESULTS/f'HPPC_{label}_voltage.csv')
        axes[0].plot(d.Time/3600,d.Simulated_V,color=color,lw=.8,label=text)
        axes[1].plot(d.Time/3600,d.Residual_mV,color=color,lw=.7,label=text)
    d=DATA['HPPC'];axes[2].plot(d.Time/3600,d.Current,color='#536878',lw=.7)
    axes[0].set(ylabel='Voltage (V)');axes[1].set(ylabel='Residual (mV)');axes[2].set(ylabel='CSV current (A)',xlabel='Time (h)')
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend();fig.savefig(RESULTS/'HPPC_voltage_residual_current.png',dpi=180);plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    for name in NAMES:
        state = pd.read_csv(RESULTS / f'{name}_pulse_checked_states.csv')
        ax.plot(state.Time / 3600, state['Minimum negative electrode surface potential difference [V]'], label=name.replace('_', ' '))
    ax.axhline(0, color='black', linestyle='--', label='0 V vs Li/Li+')
    ax.set(xlabel='Time (h)', ylabel='Minimum anode potential (V vs Li/Li+)')
    ax.grid(alpha=0.3); ax.legend()
    fig.savefig(RESULTS / 'anode_potentials.png', dpi=160); plt.close(fig)
    return all_metrics

