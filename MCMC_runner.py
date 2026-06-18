"""
cycleEventFitter.py

Fits a radiocarbon (¹⁴C) production model to measured Δ¹⁴C records around a
user-specified year, using Markov Chain Monte Carlo (MCMC) sampling. The model
decomposes the inferred production rate into three components: a sinusoidal
solar cycle baseline, a discrete spike event (e.g. a solar proton event or
supernova), and a constant background flux. Posterior samples are converted to
physical units (kg and kg/yr) and visualised as time-series plots and 2D
marginal scatter plots with 1D posterior histograms.
"""


import itertools
from Library.MCMSpikefitterWithCycle import *
import matplotlib.pyplot as plt
import numpy as np
from Library.BoxModel import *
from Library.plotfunctions import *
from Library.MCMCFunctions import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--year', type=float, required=True)
args = parser.parse_args()

year = 1950 - args.year



dt = 0.1
totprod = 6.6e-12


def gaussfunc(t, amp, times, width=0.15):
    return np.exp(-1 / 2 * (t - times) ** 2 / width ** 2) * amp*1e12


meandata = True
prepostyears = 15

data = loadexcel(projectPath/ Path('Data\\C14Records\\ETHALL2026-04-16.xlsx'))



data = calcD14C(data)
t0 = year-prepostyears
t1 = year+prepostyears
idx = np.where((data['bp']>=1950-t1)&(data['bp']<=1950-t0))[0]
df = {}
for key in data.keys():
    df[key] = data[key][idx]
if meandata:
    [delta, deltasigm, years] = getDeltafromDataframe(df)
else:
    delta, deltasigm, years = df['delta'], df['delta_sig'], df['year']

simtimes, prodcution, simdeltas, samples, weights, theta_map = MCMCCycleSpikefitter(delta, deltasigm, years, eventyear=None, N=5000,burnin=1000,thin=1)#year+0.5
times, allsimprods, allsimdeltas = getsimulations(delta, deltasigm, years,samples,thin=350)

