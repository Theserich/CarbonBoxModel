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
from Library.dfFunctions import calcD14C, getDeltafromDataframe
from Library.Functions import getExcelData
from scipy.stats import norm
import argparse
from pathlib import Path
import re

logprior = logpriorcycle

parser = argparse.ArgumentParser()
parser.add_argument('--year_start', type=float, required=True)
parser.add_argument('--year_end', type=float, required=True)
parser.add_argument('--eventdetrend', type=str, default='False')
parser.add_argument('--datalabel', type=str, default='Alldata')
args = parser.parse_args()

year_start = args.year_start
year_end = args.year_end
eventdetrend = args.eventdetrend.lower() == 'true'
meandata = True
datalabel = args.datalabel

# --- Load the raw data ONCE. This is what gets hashed for caching, since it
# is identical byte-for-byte regardless of which machine loads it. ---
rawdata = calcD14C(getExcelData(datalabel))

# --- Compute the detrended version (if requested) ONCE, up front, not per
# year in the loop. Detrending is a stochastic MCMC fit under the hood, so
# its output can legitimately differ slightly between local and Euler --
# that's fine, since we never hash the detrended values directly. ---
if eventdetrend:
    data = eventdetrenddataframe(rawdata)
else:
    data = rawdata

# Loop over all years in this chunk
for year in range(int(year_start), int(year_end)):
    print(f"Processing year {year}", flush=True)
    dt = 0.1
    totprod = 6.6e-12
    prepostyears = 15

    t0 = year - prepostyears
    t1 = year + prepostyears

    # Slice both the (possibly detrended) working data and the raw data with
    # the exact same index, so raw_delta/raw_deltasigm line up one-to-one
    # with delta/deltasigm even though they come from different dataframes.
    idx = np.where((data['bp'] >= 1950 - t1) & (data['bp'] <= 1950 - t0))[0]

    df = {}
    rawdf = {}
    for key in data.keys():
        df[key] = data[key][idx]
    for key in rawdata.keys():
        rawdf[key] = rawdata[key][idx]

    if len(df['bp']) < prepostyears / 3:
        continue

    if meandata:
        [delta, deltasigm, years_data] = getDeltafromDataframe(df)
        [raw_delta, raw_deltasigm, raw_years_data] = getDeltafromDataframe(rawdf)
    else:
        delta, deltasigm, years_data = df['delta'], df['delta_sig'], df['year']
        raw_delta, raw_deltasigm, raw_years_data = rawdf['delta'], rawdf['delta_sig'], rawdf['year']

    result = MCMCCycleSpikefitterprior(
        delta, deltasigm, years_data, logprior,
        dt=dt, totprod=totprod,
        raw_delta=raw_delta, raw_deltasigm=raw_deltasigm, detrend=eventdetrend
    )