import itertools
from Library.MCMCSpikeFitter import *
from Library.BoxModel import *
from Library.plotfunctions import *
from Library.MCMCFunctions import *
import shutil, os
from Library.EventDetrend import eventdetrenddataframe


dt = 0.1
totprod = 6.6e-12


def gaussfunc(t, amp, times, width=0.15):
    return np.exp(-1 / 2 * (t - times) ** 2 / width ** 2) * amp*1e12

eventdetrend = False
allyears = np.arange(0,50,dtype=int)

datalabel = 'Alldata2026-06-18'
data = loadexcel(projectPath/ Path(f'Data/C14Records/{datalabel}.xlsx'))
if eventdetrend:
    data = eventdetrenddataframe(data, plotfit=False)
data = calcD14C(data)

meandata = True


folder = Path(r'Graphs\Events') / Path(datalabel)
os.makedirs(folder,exist_ok=True)
shutil.rmtree(folder)
os.makedirs(folder)

prepostyears = 15

data = calcD14C(data)
intcal = True
for year in allyears:
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
    simtimes, prodcution, simdeltas, samples, weights, theta_map = MCMCCycleSpikefitterprior(delta, deltasigm, years,logprior, eventyear=None, N=2000,burnin=1000,thin=1,intcal=True)
    times, allsimprods, allsimdeltas = getsimulations(delta, deltasigm, years,samples, intcal=intcal,thin=100)
    Sim = BoxSimulator(fluxFile='StandartFluxes.xlsx', totprod=totprod, dt=dt)


    alldelta, alldeltasigm, allyears = df['delta'], df['delta_sig'], df['year']


    theta_map_phys = theta_map.copy()
    theta_map_phys[0] = np.sum(
        gaussfunc(theta_map[2], theta_map[0], simtimes, width=0.15) * dt
    )
    theta_map_phys[1] = theta_map[1] * 1e12 + totprod * 1e12
    theta_map_phys = theta_map.copy()
    theta_map_phys[0] = np.sum(
        gaussfunc(theta_map_phys[2], theta_map[0], simtimes, width=0.15) * dt
    )
    theta_map_phys[1] = theta_map[1] * 1e12 + totprod * 1e12
    samples_phys = samples.copy()
    for i, sample in enumerate(samples_phys[:, 0]):
        samples_phys[i, 0] = np.sum(
            gaussfunc(theta_map[2], sample, simtimes, width=0.15) * dt
        )
    samples_phys[:, 1] = samples_phys[:, 1] * 1e12 + totprod * 1e12
    samples_phys[:, 3] = (samples_phys[:,3] + (2 * np.pi / samples_phys[:, 5]) * samples_phys[:, 2]) % (2 * np.pi)
    samples_phys[:,4] = samples_phys[:,4]*1e12
    samples_phys[:,5] = samples_phys[:,5]

    params_settings = {0: {'name': 'Event production\n(kg)','binsize':0.2,'limits':(0,10.5)},
                       1: {'name': 'Baseline\n(kg/yr)','binsize':0.01,'limits':(4,8)},
                       2: {'name': 'Event year','binsize':0.1,'limits':(min(years), max(years))},
                       3: {'name': 'Event phase','binsize':0.3,'limits':(0,np.pi*2)},
                       4: {'name': 'Cycle amp\n(kg/yr)','binsize':0.05,'limits':(0,1.4)},
                       5: {'name': 'Cycle period\n(yrs)','binsize':0.3,'limits':(5,43)},
                       }

    #bins = build_bins(samples, nbins=20)
    for key in params_settings:
        samples = samples_phys[:, key]
        minimum = samples.min()
        maximum = samples.max()
        grid = np.arange(minimum, maximum + params_settings[key].get('binsize', 1), params_settings[key].get('binsize', 1))
        bins2 = np.arange(
            minimum - params_settings[key]['binsize'] / 2,
            maximum + params_settings[key]['binsize'] / 2 + params_settings[key]['binsize'],
            params_settings[key]['binsize']
        )
        posterior_counts, edges = np.histogram(samples, bins=bins2)
        posterior = posterior_counts / posterior_counts.sum()
        grid = 0.5 * (edges[:-1] + edges[1:])
        params_settings[key]['grid'] = grid
        params_settings[key]['posterior'] = posterior

    pairs = [(0,1) ,(2, 0),(4,5),(3,5)]
    def get_stats(samples, weights):
        # Weighted mean (Peak/Centroid)
        mean = np.average(samples)
        # Weighted variance
        var = np.average((samples - mean)**2)
        return mean, np.sqrt(var)

    # Calculate stats for each physical parameter directly
    stats = {}
    for i in range(6):
        stats[i] = get_stats(samples_phys[:, i], weights)

    # Map them to your variables
    Excess, Excess_sig = stats[0]
    Baseline, Baseline_sig = stats[1]
    fiteventtime, eventtime_sig = stats[2]

    # Re-populate intervals for your printing logic if needed
    intervals = [((stats[0]), (stats[1])), ((stats[2]), (stats[0])), ((stats[2]), (stats[1]))]

    print(f"Excess      = {Excess:.2f} ± {Excess_sig:.2f}")
    print(f"Baseline    = {Baseline:.2f} ± {Baseline_sig:.2f}")
    print(f"Event time  = {fiteventtime:.2f} ± {eventtime_sig:.2f}")
    figsize = (14, 14)
    fontsize = 15
    setPlotParams(fontsize, figsize=figsize)
    colors = itertools.cycle([f'C{i}' for i in range(10)])
    fig, ax, ax0 = subplots(2)
    ax[1].plot(convertCalendarToBCE(simtimes), (prodcution + totprod) * 1e12, color='C3', lw=2, zorder=-10)


    for simprod, simdelta in zip(allsimprods, allsimdeltas):
        ax[1].plot(convertCalendarToBCE(times), (simprod + totprod) * 1e12, color='C3', lw=1, alpha=0.05, zorder=-10)
        ax[0].plot(convertCalendarToBCE(times), simdelta, color='C0', lw=1, alpha=0.05, zorder=-10)


    ax[1].set_ylabel(r'$^{14}$C production rate (kg/year)')
    ax[0].set_ylabel(r'$\Delta^{14}$C (‰)')

    # --- Corrected posterior panel layout ---
    # The main panel occupies (1 - posterior_width) of both dimensions.
    # posterior_right_gap:  horizontal space between the right posterior and the
    #                       y-axis label of the NEXT panel (prevents overlap).
    # posterior_top_gap:    vertical space between the top posterior and the top
    #                       edge of the main axes (prevents overlap with tick labels).
    left = 0.2
    bottom = 0.1
    height = 0.12
    posterior_width = 0.2          # fraction of panel occupied by each marginal
    posterior_right_gap = 0.08   # extra gap so right posterior does not overlap next y-axis label
    posterior_top_gap  = 0.005    # extra gap so top posterior does not overlap x tick labels

    width = height / figsize[0] * figsize[1]

    # Sizes of sub-axes expressed as figure fractions
    main_w  = width  * (1 - posterior_width)
    main_h  = height * (1 - posterior_width)
    post_w  = width  * posterior_width        # right marginal width
    post_h  = height * posterior_width        # top  marginal height

    main_box  = [left,                         bottom,                          main_w, main_h]
    top_box   = [left,                         bottom + main_h + posterior_top_gap, main_w, post_h]
    right_box = [left + main_w,                bottom,                          post_w, main_h]

    # Step between panels: main + right posterior + gap before next panel's y-label
    panel_step = width + posterior_right_gap

    axs = []
    axs_top = []
    axs_right = []
    for i in range(len(pairs)):
        axs.append(fig.add_axes(main_box))
        axs_top.append(fig.add_axes(top_box, sharex=axs[i]))
        axs_top[i].axis('off')
        axs_right.append(fig.add_axes(right_box, sharey=axs[i]))
        axs_right[i].axis('off')
        main_box[0]  += panel_step
        top_box[0]   += panel_step
        right_box[0] += panel_step

    for i, (a, b) in enumerate(pairs):
        x = samples_phys[:, a]
        y = samples_phys[:, b]
        axs[i].scatter(
            x,
            y,
            c=weights,
            s=8,
            cmap="plasma",  # high-contrast: dark purple → yellow
            alpha=0.1,
            linewidths=0,
            vmin=np.percentile(weights, 10),  # clip low-weight points to boost contrast
            vmax=np.percentile(weights, 99)
        )
        axs[i].set_xlabel(params_settings[a]['name'])
        axs[i].set_ylabel(params_settings[b]['name'])
        axs_top[i].plot(params_settings[a]['grid'], params_settings[a]['posterior'], color='black', lw=1.5)
        axs_right[i].plot(params_settings[b]['posterior'], params_settings[b]['grid'], color='black', lw=1.5)
        axs[i].xaxis.get_major_formatter().set_useOffset(False)
        axs[i].yaxis.get_major_formatter().set_useOffset(False)
        x_low, x_high = weighted_quantile(x, [0.000005, 0.999995], weights)
        y_low, y_high = weighted_quantile(y, [0.000005, 0.999995], weights)
        axs[i].set_xlim(x_low, x_high)
        axs[i].set_ylim(y_low, y_high)

    ax[0].errorbar(convertCalendarToBCE(years), delta, yerr=deltasigm, fmt='o', capsize=3, label='Data', color='k', zorder=10)
    ax[0].errorbar(convertCalendarToBCE(allyears), alldelta, yerr=alldeltasigm, fmt='s', capsize=3, label='Data', color='grey', zorder=-2,alpha=0.2)
    ax[0].plot(convertCalendarToBCE(simtimes), simdeltas)

    xlim = ax[0].get_xlim()
    intcaldf = loadexcel(projectPath /'Data/IntCal/Intcal20.xlsx')
    intCalcurveidx = np.where((intcaldf['bp'] > 1950 - t0 - 50) & (intcaldf['bp'] < 1950 - t1 + 50))[0]
    for key in intcaldf.keys():
        intcaldf[key] = intcaldf[key][intCalcurveidx]
    idelta, ideltasigm, iyears = getDeltafromDataframe(intcaldf)
    ax[0].fill_between(iyears, idelta - ideltasigm, idelta + ideltasigm, alpha=0.2, color='grey')
    ax[0].set_xlim(xlim)
    separateSubplots(ax, overlap=0.4, ylabelx=(0.07, 0.04), plotlabels=False)
    ax[0].xaxis.set_major_formatter(FuncFormatter(CE_BCE_format))
    fig.subplots_adjust(bottom=0.26)
    eventamp, baseline, t,phase,amp,period,delta0 = theta_map
    ymin, ymax = ax[1].get_ylim()
    fiteventtime, fiteventtime_sig = stats[2]
    phase_val, phase_sig = stats[3]
    amp_val, amp_sig = stats[4]
    period_val, period_sig = stats[5]

    formulastring = (
        f'$Q_{{\\text{{base}}}}(t) = ({Baseline:.2f} \\pm {Baseline_sig:.2f}) + ({amp_val:.2f} \\pm {amp_sig:.2f}) '
        f'\\sin\\!\\left(\\frac{{2\\pi (t - ({fiteventtime:.1f} \\pm {fiteventtime_sig:.1f}))}}{{{period_val:.1f} \\pm {period_sig:.1f}}} '
        f'+ ({phase_val:.2f} \\pm {phase_sig:.2f})\\right)\\,\\text{{kg/yr}}$\n'
        f'with {Excess:.1f} ± {Excess_sig:.1f} kg modelled event production'
    )
    print(Baseline)
    yrange = ymax - ymin
    ax[1].set_ylim(ymin - 0.09 * yrange, ymax)
    ymin, ymax = ax[1].get_ylim()
    print(amp)
    ax[1].text(min(years)-2, Baseline-1.2*amp*1e12, formulastring, fontsize=fontsize,va='top')
    year_str = f'{year:+06d}'
    plt.savefig(folder/f'{year_str}cycleEventFitter.png',bbox_inches='tight')