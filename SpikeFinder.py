import itertools
from Library.plotfunctions import *
from Library.MCMCSpikeFitter import *
from scipy.signal import find_peaks
from Library.EventDetrend import eventdetrenddataframe
eventdetrend = False
preposttime = 15
t0,t1  = 1000, 1200




dt = 0.1
totprod = 6.6e-12

meandata = True
prepostyears = 15
eventdetrend = False
threshold = 3


dataframe = getExcelData('Alldata')
colors = itertools.cycle([f'C{i}' for i in range(10)])
setPlotParams(14,figsize=(16, 16))


if eventdetrend:
    dataframe = eventdetrenddataframe(dataframe, plotfit=False)

dfs = [dataframe]
labels = [datalabel]

fig, ax = plt.subplots(len(dfs),sharex=True)
for k,data in enumerate(dfs):
    print(f'Processing dataset: {labels[k]}')
    color = next(colors)
    yearproddict, yearbasedict,yearprobdict = get_yeardict(t0,t1,data,logprior,threshold=threshold)
    if len(dfs) == 1:
        ax = [ax]
    years = np.arange(t0,t1)
    probabilities = np.zeros(len(years))
    prob_uncertainties = np.zeros(len(years))
    prods = np.zeros(len(years))
    prods_sig = np.zeros(len(years))
    increasefactors = np.zeros(len(years))
    increasefactors_sig = np.zeros(len(years))
    for i, t in enumerate(years):
        if t in yearproddict:
            prob = yearprobdict[t]
            increasefactor = np.mean(yearproddict[t]/yearbasedict[t])
            increasefactor_sig = np.std(yearproddict[t]/yearbasedict[t])
            probabilities[i] = yearprobdict[t]
            prods[i] = np.mean(yearproddict[t])
            prods_sig[i] = np.std(yearproddict[t])
            increasefactors[i] = increasefactor
            increasefactors_sig[i] = increasefactor_sig
    ax[k].plot(years, probabilities, color=color,label=labels[k])
    ax[k].legend()
    peaks, _ = find_peaks(probabilities, height=0.3,distance=8)
    print(list(years[peaks]))
    stronglabel = ''
    printlabel = ''
    for peak in peaks:
        peaklabel = f'{years[peak]}\n ({prods[peak]:.1f} ± {prods_sig[peak]:.1f} kg\n x({increasefactors[peak]:.1f} ± {increasefactors_sig[peak]:.1f}'
        peaklabel3 = f'{years[peak]}\n {prods[peak]:.1f} ± {prods_sig[peak]:.1f} kg\n'
        peaklabel4 = f'{years[peak]}\t {prods[peak]:.1f} ± {prods_sig[peak]:.1f} kg\t {probabilities[peak]*100:.0f} %\n'
        peaklabel2 = f'{years[peak]}'
        label = f'{years[peak]}\t {prods[peak]:.1f} ± {prods_sig[peak]:.1f} kg\t {increasefactors[peak]:.1f} ± {increasefactors_sig[peak]:.1f}\n'
        printlabel+= peaklabel4
        if prods[peak] > 6:
            stronglabel += peaklabel4
        ax[k].plot(years[peak], probabilities[peak],'o',color=color)
        ax[k].text(years[peak], probabilities[peak]+0.05,peaklabel3 , rotation=90, color='k',
                       ha='center',
                       fontsize= 12* 0.8)
    print(printlabel)
    print('\n')
    print(stronglabel)
    ax[k].set_ylabel('Probability of event\n> %.1f kg' % threshold)
    ax[k].set_ylim(0,1.4)
#ax[-1].set_xlabel('Year')

xticks = ax[-1].get_xticks()
locs = np.arange(min(xticks),max(xticks),100)
minorLocator = FixedLocator(locs)
ax[-1].xaxis.set_minor_locator(minorLocator)
ax[-1].xaxis.set_major_formatter(FuncFormatter(CE_BCE_format))
plt.subplots_adjust(hspace=0)
folder = Path(r'C:\Users\nbrehm.D.000\SynologyDrive\ETHPostdoc\Paper\Millennium\Graphs')
plotname = folder / f'SpikeFinderCycle_{labels}{threshold}kg.png'
plt.savefig(plotname, dpi=300,bbox_inches='tight')
plt.show()


