
"""
SolomodReconstruction.py

This script performs a reconstruction of past solar modulation using
radiocarbon (Δ14C) data and a multi-box carbon cycle model.

Workflow:
- Load Δ14C measurements from an Excel dataset and select a specified
  time window in years BP.
- Reconstruct time-dependent 14C production rates using a Monte Carlo
  inversion of a box-model carbon cycle.
- Propagate production uncertainties via Monte Carlo sampling.
- Convert reconstructed production rates into solar modulation
  parameters using the solarModCalc routine.
- Visualize Δ14C data, reconstructed Δ14C, production rates, and
  associated uncertainties.

The reconstruction is performed on a fixed time grid defined by the
box model time step. All production uncertainties are evaluated on
the same grid to ensure consistency when computing solar modulation
bounds.

Intended use:
Standalone script for scientific analysis and visualization of
solar modulation over the selected time interval.
"""


from Library.BoxModel import *
from Library.plotfunctions import *
from Library.MCMCFunctions import *


#timerange in years BP for reconstruction
t0 = 950
t1 = 1050



dt = 0.1
totprod = 6.6e-12


def gaussfunc(t, amp, times, width=0.15):
    return np.exp(-1 / 2 * (t - times) ** 2 / width ** 2) * amp*1e12


prepostyears = 15

data = loadexcel(projectPath/ Path('Data\\C14Records\\ETHALL2026-04-16.xlsx'))


idx = np.where((data['bp']>=1950-t1)&(data['bp']<=1950-t0))[0]
df = {}
for key in data.keys():
    df[key] = data[key][idx]
[delta, deltasigm, years] = getDeltafromDataframe(df)



Sim = BoxSimulator()
Sim.getIntCalStartState(min(years))
Sim.monteCarloProdReconstruction([delta, deltasigm, years], 12, N=100, smoothing='Golay112', kind='linear')
times, production, deltas = Sim.getSimulationResults()

ptimes = Sim.dttimes
production = Sim.production[0][0](ptimes)
prodsigm = Sim.dtmonteprod_sig
solarmod = solarModCalc(ptimes, production, totprod)
solarmodlow = solarModCalc(ptimes, production - prodsigm, totprod)
solarmodhigh = solarModCalc(ptimes, production + prodsigm, totprod)

fig, ax = plt.subplots(3, sharex=True, figsize=(14, 10))
ax[0].errorbar(years, delta, yerr=deltasigm, fmt='o', label='Data', color='C0')
ax[0].set_ylabel('Δ14C (‰)')
ax[0].legend()
ax[0].grid(ls=':')
ax[0].plot(times, deltas[12], 'k-', label='Reconstruction')
ax[1].fill_between(ptimes, (production - prodsigm)*1e12+totprod*1e12, (production + prodsigm)*1e12+totprod*1e12, color='gray', alpha=0.5, label='Uncertainty')
ax[1].set_ylabel('Production (kg/year)')
ax[1].legend()
ax[1].grid(ls=':')
ax[2].fill_between(ptimes, solarmodlow, solarmodhigh, color='gray', alpha=0.5, label='Uncertainty')
ax[2].set_ylabel('Solar modulation (MeV)')
plt.show()



