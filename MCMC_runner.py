"""
MCMC_runner.py

Fits a radiocarbon (¹⁴C) production model to measured Δ¹⁴C records around a
user-specified year, using Markov Chain Monte Carlo (MCMC) sampling. The model
decomposes the inferred production rate into three components: a sinusoidal
solar cycle baseline, a discrete spike event (e.g. a solar proton event or
supernova), and a constant background flux. Posterior samples are converted to
physical units (kg and kg/yr) and visualised as time-series plots and 2D
marginal scatter plots with 1D posterior histograms.
"""
import itertools
from Library.MCMCSpikeFitter import *
from Library.BoxModel import *
from Library.plotfunctions import *
from Library.MCMCFunctions import *
from Library.EventDetrend import eventdetrenddataframe
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--year_start', type=float, required=True)
parser.add_argument('--year_end', type=float, required=True)
parser.add_argument('--eventdetrend', type=str, default='False')
args = parser.parse_args()

year_start = args.year_start
year_end = args.year_end
eventdetrend = args.eventdetrend.lower() == 'true'

# Loop over all years in this chunk
for year in range(int(year_start), int(year_end)):
    print(f"Processing year {year}", flush=True)

    dt = 0.1
    totprod = 6.6e-12
    meandata = True
    prepostyears = 15

    datalabel = 'Alldata2026-06-18'
    data = loadexcel(projectPath / Path(f'Data/C14Records/{datalabel}.xlsx'))
    data = calcD14C(data)

    t0 = year - prepostyears
    t1 = year + prepostyears
    idx = np.where((data['bp'] >= 1950 - t1) & (data['bp'] <= 1950 - t0))[0]
    df = {}
    for key in data.keys():
        df[key] = data[key][idx]
    if meandata:
        [delta, deltasigm, years_data] = getDeltafromDataframe(df)
    else:
        delta, deltasigm, years_data = df['delta'], df['delta_sig'], df['year']

    simtimes, production, simdeltas, samples, weights, theta_map = MCMCCycleSpikefitterprior(
        delta, deltasigm, years_data, logprior
    )