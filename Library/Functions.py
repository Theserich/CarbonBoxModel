from copy import copy
from scipy.interpolate import RegularGridInterpolator
from matplotlib import pyplot as plt
from scipy import signal
from Library.dfFunctions import *
from scipy.interpolate import interp1d, UnivariateSpline, CubicSpline
from Library.cache_function import cache_results
from Library.GlobalPathsAndConstants import *






def lowsmoothdata(x, y, lowcut, order=4):
    minx = min(x)
    maxx = max(x)
    interpoly = interp1d(x, y)
    interx = np.arange(minx, maxx + 1, 1)
    smoothed = interp1d(interx, butter_lowpass_filter(interpoly(interx), 1 / lowcut, fs=1, order=order))
    curve = smoothed(x)
    return curve


def chooseSmoothing(years, delta, smoothing):
    if smoothing == 'None' or smoothing is None:
        pass
    elif smoothing == 'Spline':
        sp1 = UnivariateSpline(years, delta, s=100)
        # sp1.set_smoothing_factor(2)
        delta = sp1(years)
    elif smoothing[0] + smoothing[1] + smoothing[2] + smoothing[3] + smoothing[4] == 'Golay':
        a = int(smoothing[-3] + smoothing[-2])
        b = int(smoothing[-1])
        # print(a)
        # print(b)
        delta = signal.savgol_filter(delta, a, b)
    elif smoothing == 'lowpass4':
        deltainterpol = interp1d(years, delta)
        interpoltimes = np.arange(min(years), max(years) + 1)
        lpdelta = butter_lowpass_filter(deltainterpol(interpoltimes), 1 / 4, fs=1)
        interpoldeltas = interp1d(interpoltimes, lpdelta)
        delta = interpoldeltas(years)
    elif smoothing == 'lowpass10':
        deltainterpol = interp1d(years, delta)
        interpoltimes = np.arange(min(years), max(years) + 1)
        lpdelta = butter_lowpass_filter(deltainterpol(interpoltimes), 1 / 10, fs=1)
        interpoldeltas = interp1d(interpoltimes, lpdelta)
        delta = interpoldeltas(years)
    elif smoothing == 'lowpass5':
        deltainterpol = interp1d(years, delta)
        interpoltimes = np.arange(min(years), max(years))
        lpdelta = butter_lowpass_filter(deltainterpol(interpoltimes), 1 / 4, fs=1)
        interpoldeltas = interp1d(interpoltimes, lpdelta)
        delta = interpoldeltas(years)
    else:
        pass
    return delta



def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs=12, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y




def setPlotParams(fontsize, figsize=(15, 10), auto=False, lw=1.5, inline=True, xinline=True,enable_minor=False):
    import matplotlib.pylab as pylab
    params = {'figure.autolayout': auto,
              'legend.fontsize': fontsize,
              'figure.figsize': figsize,
              'axes.labelsize': fontsize,
              'axes.titlesize': fontsize,
              'axes.linewidth': lw,
              'xtick.labelsize': fontsize,
              'xtick.major.size': 10,
              'xtick.minor.size': 5,
              'ytick.major.size': 10,
              'ytick.minor.size': 5,
              'xtick.major.width': lw,
              'xtick.minor.width': lw,
              'ytick.major.width': lw,
              'ytick.minor.width': lw,
              'ytick.labelsize': fontsize}
    if inline:
        params['ytick.direction'] = 'in'
    if xinline:
        params['xtick.direction'] = 'in'
    else:
        params['xtick.direction'] = 'out'
    pylab.rcParams.update(params)
    if enable_minor:
        # Activate minor ticks for all current figures
        for ax in plt.gcf().get_axes():
            ax.minorticks_on()

        # Monkey-patch plt.subplots to auto-enable minor ticks on new axes
        import types
        original_subplots = plt.subplots

        def patched_subplots(*args, **kwargs):
            fig, axs = original_subplots(*args, **kwargs)
            if isinstance(axs, (list, np.ndarray)):
                for ax in axs.flat:
                    ax.minorticks_on()
            else:
                axs.minorticks_on()
            return fig, axs

        plt.subplots = types.FunctionType(
            patched_subplots.__code__,
            globals(),
            name=patched_subplots.__name__,
            argdefs=patched_subplots.__defaults__,
            closure=patched_subplots.__closure__,
        )





def convertCalendarToBCE(t,bp=False):
    if bp == False:
        try:
            res = np.array(copy(t))
            neginds = np.where(res <= 0)
            res[neginds] = res[neginds] - 1
        except:
            res = copy(t)
            if res < 0:
                res -= 1
    else:
        return 1950-t
    return res


def getSimulationResults(Sim):
    # steadystate = np.loadtxt(Sim.steadystateName)
    times = np.array(Sim.times)
    n = Sim.nBoxes
    ref = Sim.ref
    production = Sim.production
    result = np.array(Sim.simResult)
    c12 = result[:, :n]  # shape (T, nBoxes)
    c14 = result[:, n:2 * n]  # shape (T, nBoxes)
    deltas = (c14 / c12 - ref) / ref * 1000
    deltas = deltas.transpose()
    return times, production, np.array(deltas)


def getMeanVADM(index=1):
    if index == 0:
        filename = projectPath/'Data/MagneticField/CALS10k2m.txt'
        vadm = []
        time = []
        with open(filename) as f:
            lines = f.readlines()
        for a in lines:
            b = a.strip('\n')
            b = b.split('\t')
            time.append(float(b[0]))
            vadm.append(float(b[1]))
    elif index == 1:
        filename = projectPath/'Data/MagneticField/covarch_mean.txt'
        vadm = []
        time = []
        with open(filename) as f:
            lines = f.readlines()
        for a in lines:
            b = a.strip('\n')
            b = b.split('\t')
            time.append(float(b[0]))
            vadm.append(float(b[1]))
    elif index == 2:
        filename = projectPath/'Data/MagneticField/HFMOL1A1m.txt'
        vadm = []
        time = []
        with open(filename) as f:
            lines = f.readlines()
        for a in lines:
            b = a.strip('\n')
            b = b.split('\t')
            time.append(float(b[0]))
            vadm.append(float(b[1]))
    elif index == 3:
        filename = projectPath/'Data/MagneticField/covlake_meanm.txt'
        vadm = []
        time = []
        with open(filename) as f:
            lines = f.readlines()
        for a in lines:
            b = a.strip('\n')
            b = b.split('\t')
            time.append(float(b[0]))
            vadm.append(float(b[1]))
    elif index == 4:
        filename = projectPath + 'Data/MagneticField/SHAWQ2k.txt'
        vadm = []
        time = []
        with open(filename) as f:
            lines = f.readlines()
        for a in lines:
            b = a.strip('\n')
            b = b.split('\t')
            time.append(float(b[0]))
            vadm.append(float(b[1]))
    elif index == 5:
        filename = projectPath/'Data/MagneticField/knudsenCalculation.xls'
        magnDf = pd.read_excel(filename)
        time = 1950 - magnDf['Year BP']
        vadm = magnDf['VADM']
    elif index == 6:
        filename = projectPath/'Data/MagneticField/DM_GGF100k.dat'
        datContent = [i.strip().split() for i in open(filename).readlines()]
        dattimes = []
        datmag = []
        for dat in datContent[1:]:
            dattimes.append(float(dat[0]) * 1000)
            datmag.append(float(dat[1]))
        dattimes = np.array(dattimes)
        datmag = np.array(datmag)
        time = 1950 - dattimes
        vadm = datmag
    return time, vadm




def getProductionData():
    filename = projectPath/'Data/CosmicData/C14Kovaltsov 2012.xlsx'
    data = pd.read_excel(filename)
    M = []
    Phi = []
    for i in range(0, 2001, 100):
        Phi.append(i)
    for i in range(0, 12 * 5 + 1, 1):
        M.append(i / 5)
    Datamatrix = np.zeros((len(M), len(Phi)))

    for i in range(len(M)):
        for j in range(len(Phi)):
            Datamatrix[i][j] = data[int(Phi[j])][i]
    X, Y = np.meshgrid(Phi, M)
    gridInterpol = RegularGridInterpolator((M, Phi), Datamatrix)
    return gridInterpol


@cache_results(file_format="pickle",cache_dir='SolarModCalc')
def solarModCalc(times, production, totprod, magindex=6):
    prodrateprod = (production + totprod) * 10 ** 12 * 1000 * 6.022 * 10 ** 23 / massC14 / AEarth / secondsPerYear
    vadmtime, refvadm = getMeanVADM(index=magindex)
    refvadm = interp1d(vadmtime, refvadm)
    refvadm = np.array(refvadm(times))
    productionData = getProductionData()
    solarmodlinspace = np.linspace(0, 2000, 61)
    solarMod = np.zeros(len(prodrateprod))
    for i in range(len(prodrateprod)):
        x = np.zeros(len(solarmodlinspace))
        for j in range(len(solarmodlinspace)):
            x[j] = productionData((refvadm[i], solarmodlinspace[j]))
        curve = interp1d(x, solarmodlinspace)
        if prodrateprod[i] >= max(x):
            solarMod[i] = curve(max(x))
        elif prodrateprod[i] <= min(x):
            solarMod[i] = curve(min(x))
        else:
            solarMod[i] = curve(prodrateprod[i])
    return solarMod


def getIntcalRawData(t0, t1):
    df = pd.read_excel(projectPath/'Data/IntCal/Intcalrawdelta.xlsx')
    data = {'Time': np.array(df['cal']), 'dTime': np.array(df['calsig']), 'delta': np.array(df['d14c']),
            'delta_sig': np.array(df['d14csig']),'years':1950-np.array(df['cal'])}
    ind = np.where((data['Time'] > t0) & (data['Time'] < t1))[0]
    for key in data.keys():
        data[key] = data[key][ind]
    return data

def getIntcalData(t0, t1):
    df = pd.read_excel(projectPath/'Data/IntCal/Intcal20.xlsx')
    data = {'Time': np.array(df['bp']), 'delta': np.array(df['Delta14C']), 'delta_sig': np.array(df['Sigm2']),
            'fm': np.array(df['fm']), 'fm_sig': np.array(df['fm_sig']), 'bp': np.array(df['bp']),'years':1950-np.array(df['bp'])}
    ind = np.where((data['Time'] > t0) & (data['Time'] < t1))[0]
    for key in data.keys():
        data[key] = data[key][ind]
    return data



