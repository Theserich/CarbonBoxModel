# CarbonBoxModel

A multi-reservoir carbon-cycle box model for reconstructing ¹⁴C production rates from measured Δ¹⁴C records and fitting past radiocarbon production spikes — including solar proton events and potential supernova signatures — with full MCMC-based uncertainty quantification.

---

## Overview

The model solves the coupled system of ordinary differential equations governing ¹⁴C exchange between atmospheric, oceanic, and biospheric reservoirs. Given a time series of measured Δ¹⁴C values, it can either reconstruct the implied production history or fit a parametric production model (sinusoidal solar-cycle baseline + discrete spike event) directly to the data via Bayesian inference.

The production rate Q(t) is modelled as:

$$Q(t) = Q_\text{base}(t) + Q_\text{event}(t)$$

where the baseline follows a sinusoidal solar cycle

$$Q_\text{base}(t) = B + A\,\sin\!\left(\frac{2\pi\,t}{P} + \phi\right)$$

and the event is a Gaussian pulse centred on the spike year.

---

## Repository Structure

```
CarbonBoxModel/
│
├── Library/
│   ├── BoxModel.py                    # Carbon-cycle box model + Numba ODE kernel
│   ├── MCMSpikefitterWithCycle.py     # MCMC spike + solar-cycle fitter
│   ├── MCMCFunctions.py               # MCMC utilities
│   ├── plotfunctions.py               # Plotting helpers
│   ├── Functions.py                   # Shared numerical utilities
│   ├── dfFunctions.py                 # Data-frame and Δ14C helpers
│   ├── GlobalPathsAndConstants.py     # Physical constants and project paths
│   └── timerfunction.py               # Simple timing decorator
│
├── SolomodReconstruction.py           # Solar modulation reconstruction (production → Φ)
├── cycleEventFitter.py                # MCMC spike + cycle analysis script
│
├── Data/
│   ├── C14Records/                    # Measured Δ14C input data (Excel)
│   └── IntCal/                        # IntCal20 calibration data
│
├── SimulationResults/
│   └── SteadyStates/                  # Cached steady-state reservoir vectors
│
└── Fluxes/
    └── StandartFluxes.xlsx            # Inter-reservoir flux matrix and production fractions

```

---

## Key Components

### `BoxModel.py` — BoxSimulator

The core simulation engine. On first use it computes and caches the model's steady-state reservoir vector; subsequent runs load it from disk. An IntCal20-derived state trajectory is likewise cached so that arbitrary start states can be recovered in O(1).

| Method | Description |
|---|---|
| `__init__` | Loads fluxes, initialises steady state, builds or loads the IntCal20 cache |
| `getstartState` | Spins up the model for a given start Δ¹⁴C value |
| `getIntCalStartState` | Recovers the reservoir state at a given calendar year from the IntCal20 cache |
| `productionReconstruction` | Inverts observed Δ¹⁴C to a production time series using a single-step predictor–corrector |
| `simulate` | Forward-integrates the box model with background, solar-cycle, and event production inputs |
| `getSimulationResults` | Returns simulation times, production history, and per-reservoir Δ¹⁴C |
| `monteCarloProdReconstruction` | Monte Carlo uncertainty propagation for the reconstructed production rate |
 vector.

### `cycleEventFitter.py` — Main Analysis Script

Loads a measured Δ¹⁴C record, runs the MCMC spike+cycle fitter, converts posterior samples to physical units, and produces two figure panels:

1. **Time-series panel** — observed Δ¹⁴C with IntCal20 background, MAP fit, and posterior ensemble; inferred production rate with posterior ensemble.
2. **Posterior panel** — 2D marginal scatter plots (coloured by importance weight) for selected parameter pairs, each decorated with 1D marginal histograms:
   - Event production (kg) vs. baseline (kg/yr)
   - Event year vs. event production (kg)
   - Cycle amplitude (kg/yr) vs. cycle period (yrs)
   - Event year vs. event phase

**Fitted parameters and their units:**

| Parameter | Symbol | Unit |
|---|---|---|
| Event production | — | kg of ¹⁴C |
| Baseline production | B | kg/yr |
| Event year | t₀ | CE / BCE |
| Solar-cycle phase | φ | rad |
| Solar-cycle amplitude | A | kg/yr |
| Solar-cycle period | P | years |

---

## Usage

### 1. Configure the target event

In `cycleEventFitter.py`, set the calendar year of interest and the time window around it:

```python
year          = 840   # target year (CE positive, BCE negative)
prepostyears  = 15    # years before and after to include
meandata      = True  # average repeated measurements per year
```

### 2. Run the fitter

```bash
python cycleEventFitter.py
```

The script will:
- Load and filter the Δ¹⁴C record from `Data/C14Records/`
- Run the MCMC sampler (N = 5000 draws, 1000 burn-in)
- Draw posterior predictive simulations
- Print MAP estimates and posterior means ± standard deviations
- Display the combined time-series and posterior figure

### 3. Use `BoxSimulator` directly

```python
from Library.BoxModel import FastSimulator

sim = FastSimulator(fluxFile='StandartFluxes.xlsx', dt=0.1)

# Recover the reservoir state at a given year from the IntCal20 trajectory
sim.getIntCalStartState(t0=775, startdelta=-5.0, boxindex=sim.refbox)

# Forward-simulate from t0 to tmax
sim.simulate(t0=760, tmax=800)
times, production, deltas = sim.getSimulationResults()
```

---

## Data Requirements

| File | Description |
|---|---|
| `Data/C14Records/ETHALL<date>.xlsx` | Δ¹⁴C measurements with columns `bp`, `delta`, `delta_sig` |
| `Data/IntCal/Intcal20.xlsx` | IntCal20 calibration curve (same column format) |
| `Fluxes/StandartFluxes.xlsx` | Inter-reservoir flux matrix, production fractions, reference box index, and steady-state ¹²C contents |

---

## Dependencies

- Python ≥ 3.9
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `numba`
- `openpyxl` (for `.xlsx` reading)

Install with:

```bash
pip install numpy scipy pandas matplotlib numba openpyxl
```

---

## Physical Constants

- **Mean ¹⁴C production rate:** 6.6 × 10⁻¹² (relative to ¹²C; configurable via `totprod`)
- **Radioactive decay constant λ:** defined in `GlobalPathsAndConstants.py` as `landa`
- **Event production distribution:** 45 % stratosphere box 0, 5 % troposphere box 1, 45 % stratosphere box 11, 5 % troposphere box 12

---

## Output

Printed to console:
```
Excess      = 3.21 ± 0.45   [kg of ¹⁴C]
Baseline    = 6.10 ± 0.08   [kg/yr]
Event time  = 774.83 ± 0.12 [CE]
```

Displayed figures:
- Δ¹⁴C time series with data, IntCal20 band, MAP fit, and posterior ensemble
- ¹⁴C production rate time series with posterior ensemble
- 2D marginal posteriors with 1D histograms for key parameter pairs
