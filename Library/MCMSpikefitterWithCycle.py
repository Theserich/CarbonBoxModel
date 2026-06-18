from Library.BoxModel import *
import emcee
from Library.MCMCFunctions import emcee_weights
from scipy.stats import norm
import copy
from scipy.stats import halfnorm


@cache_results(file_format='npz',recalc=False, cache_dir="CycleSpikeFitterCache")
def MCMCCycleSpikefitter(delta, deltasigm, years, eventyear=None, dt=0.1, totprod=6.6e-12, N=1000, burnin=100, thin=1,intcal=True):
    sig0 = deltasigm[0]
    startdelta = np.mean(delta[:4])
    Sim = BoxSimulator(fluxFile='StandartFluxes.xlsx', totprod=totprod, dt=dt)
    if intcal:
        box0 = copy.deepcopy(Sim.getIntCalStartState(min(years) - 1, startdelta, 12))
    else:
        box0 = copy.deepcopy(Sim.getstartState(startdelta, 12, preTime=10))
    fixsimtimes = np.arange(min(years) - 5, max(years) + 5, dt)
    if eventyear is None:
        maxyear = max(years)-5
        minyear = min(years)+5
        yearstep = (max(years) - min(years)) / 3
        diffs = np.diff(delta)
        spikeyear = years[np.argmax(diffs)]
        initial_parameters = np.array([totprod, 0, spikeyear,np.pi,2e-12,11,startdelta])
    else:
        minyear = eventyear - 0.45
        maxyear = eventyear + 0.45
        yearstep = 0.2
        initial_parameters = np.array([totprod, 0, eventyear,np.pi,2e-12,11,startdelta])

    def gausseventfunc(t, amp, times, width=0.15):
        return np.exp(-0.5 * (t - times) ** 2 / width ** 2) * amp

    def baseeventfunc(baseline, times, phase, amp, period):
        return np.ones(len(times)) * baseline + amp * np.sin(2 * np.pi * times / period + phase)

    def log_likelihood(theta):
        eventamp, baseline, t,phase,amp,period,delta0 = theta
        Sim.box0 = box0
        Sim.production = []
        Sim.eventproduction = []
        Sim.box0 = Sim.getstartState(delta0, 12, preTime=3)
        Sim.production = [[interp1d(fixsimtimes, baseeventfunc(baseline, fixsimtimes,phase,amp,period), fill_value=baseline, bounds_error=False,kind='linear')]]
        Sim.eventproduction = [[interp1d(fixsimtimes, gausseventfunc(t, eventamp, fixsimtimes), fill_value=0, bounds_error=False,kind='linear')]]
        Sim.simulate(min(years) - 1, max(years))
        simtimes, prod, deltas = Sim.getSimulationResults()
        interpoldeltas = interp1d(simtimes, deltas[12])
        res = (interpoldeltas(years) - delta) / deltasigm
        return -0.5 * np.sum(res ** 2 + np.log(2 * np.pi * deltasigm ** 2))

    def log_prior(theta):
        eventamp, baseline, time, phase, amp, period, delta0 = theta
        # Check all conditions
        if not (
                0 <= eventamp < 10 * totprod and
                -totprod < baseline < 10 * totprod and
                minyear < time < maxyear and
                0 <= phase < 2 * np.pi and
                5 < period < 18 and
                amp > 0 and
                abs(amp + baseline) < totprod
        ):
            return -np.inf
        lp_time = 0.0
        lp_phase = 0.0
        lp_baseline = 0.0
        lp_amp = 0
        lp_period = norm.logpdf(period, loc=11, scale=1)
        lp_eventamp = 0
        lp_delta0 = norm.logpdf(delta0, loc=startdelta, scale=sig0)
        return lp_eventamp + lp_amp + lp_period + lp_time + lp_phase + lp_baseline + lp_delta0

    def log_posterior(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    ndim = len(initial_parameters)
    nwalkers = 4 * ndim
    step_sizes = np.array([
        0.1 * totprod,  # eventamp
        0.1 * totprod,  # baseline
        0.1 * yearstep,  # time
        0.1,  # phase
        0.1 * totprod,  # amp
        0.2,  # period
        sig0 #delta0sig
    ])
    pos = initial_parameters + step_sizes * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior,
        moves=[
            (emcee.moves.DEMove(), 0.8),
            (emcee.moves.DESnookerMove(), 0.2),
        ]
    )
    sampler.run_mcmc(pos, N, progress=True)
    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    log_prob = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
    weights = emcee_weights(sampler, burnin=burnin, thin=thin)
    autocorrtime = sampler.get_autocorr_time(quiet=True)
    print(f"Autocorrelation time estimate: {autocorrtime}")
    print(f"Chain length / tau: {N / autocorrtime}")
    print(f"Effective sample size: {N * nwalkers / autocorrtime}")
    imax = np.argmax(log_prob)
    theta_map = samples[imax]
    thetabest = theta_map
    Sim.box0 = box0
    delta0 = thetabest[6]
    Sim.production = []
    Sim.eventproduction = []
    Sim.box0 = Sim.getstartState(delta0, 12, preTime=3)
    Sim.production = [[
        interp1d(fixsimtimes, baseeventfunc(thetabest[1], fixsimtimes,thetabest[3],thetabest[4],thetabest[5]), fill_value=0, kind='linear',
                 bounds_error=False)]]
    Sim.eventproduction = [
        [interp1d(fixsimtimes, gausseventfunc(thetabest[2], thetabest[0], fixsimtimes), fill_value=0, kind='linear',
                  bounds_error=False)]]

    Sim.simulate(min(years) - 1, max(years))
    times, p, deltas = Sim.getSimulationResults()
    return Sim.times, p[0][0](Sim.times) + Sim.eventproduction[0][0](Sim.times), deltas[12], samples, weights, thetabest


@cache_results(file_format='npz',recalc=False, cache_dir="getsimulationsCache")
def getsimulations(delta, deltasigm, years,samples, intcal=True,dt=0.1, totprod=6.6e-12,thin=1):
    startdelta = np.mean(delta[:4])
    Sim = BoxSimulator(fluxFile='StandartFluxes.xlsx', totprod=totprod, dt=dt)
    if intcal:
        box0 = copy.deepcopy(Sim.getIntCalStartState(min(years) - 1, startdelta, 12))
    else:
        box0 = copy.deepcopy(Sim.getstartState(startdelta, 12, preTime=10))
    fixsimtimes = np.arange(min(years) - 5, max(years) + 5, dt)
    def gausseventfunc(t, amp, times, width=0.15):
        return np.exp(-0.5 * (t - times) ** 2 / width ** 2) * amp

    def baseeventfunc(baseline, times, phase, amp, period):
        return np.ones(len(times)) * baseline + amp * np.sin(2 * np.pi * times / period + phase)

    if len(samples) == 0:
        empty_2d = np.empty((0, len(fixsimtimes)))
        return fixsimtimes, empty_2d, empty_2d
    sample = samples[0]
    eventamp, baseline, t, phase, amp, period, delta0 = sample
    Sim.box0 = box0
    Sim.production = []
    Sim.eventproduction = []
    #Sim.box0 = Sim.getstartState(delta0, 12, preTime=3)
    Sim.production = [
        [interp1d(fixsimtimes, baseeventfunc(baseline, fixsimtimes, phase, amp, period), fill_value=baseline,
                  bounds_error=False, kind='linear')]]
    Sim.eventproduction = [
        [interp1d(fixsimtimes, gausseventfunc(t, eventamp, fixsimtimes), fill_value=0, bounds_error=False,
                  kind='linear')]]
    Sim.box0 = Sim.getstartState(delta0, 12, preTime=3)
    Sim.simulate(min(years) - 1, max(years))
    times, p, deltas = Sim.getSimulationResults()

    n_samples = len(samples[::thin])
    simtimes, prod, deltas = Sim.getSimulationResults()
    n_times = len(Sim.times)
    alldeltas = np.empty((n_samples, n_times))
    allprods = np.empty((n_samples, n_times))
    alldeltas[0] = deltas[12]
    allprods[0] = prod[0][0](Sim.times) + Sim.eventproduction[0][0](Sim.times)
    for i, sample in enumerate(samples[::thin][1:], start=1):
        eventamp, baseline, t, phase, amp, period, delta0 = sample
        Sim.box0 = box0
        Sim.production = []
        Sim.eventproduction = []

        Sim.production = [
            [interp1d(fixsimtimes, baseeventfunc(baseline, fixsimtimes, phase, amp, period), fill_value=baseline,
                      bounds_error=False, kind='linear')]]
        Sim.eventproduction = [
            [interp1d(fixsimtimes, gausseventfunc(t, eventamp, fixsimtimes), fill_value=0, bounds_error=False,
                      kind='linear')]]
        Sim.box0 = Sim.getstartState(delta0, 12, preTime=3)
        Sim.simulate(min(years) - 1, max(years))
        simtimes, prod, deltas = Sim.getSimulationResults()
        alldeltas[i] = deltas[12]
        allprods[i] = prod[0][0](Sim.times) + Sim.eventproduction[0][0](Sim.times)
    return Sim.times, allprods, alldeltas









