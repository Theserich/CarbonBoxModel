


from numba import njit
from Library.Functions import *
from Library.GlobalPathsAndConstants import *

class BoxSimulator:

    def __init__(self, totprod=6.6e-12, fluxFile='StandartFluxes.xlsx', dt=0.1):
        self.dt = dt
        self.fluxFile = fluxFile
        self.fluxdf = pd.read_excel(projectPath/fluxesPath/self.fluxFile)
        self.refbox = int(self.fluxdf['ref'][0])
        self.nBoxes = len(self.fluxdf['Contents'])
        self.steadyC12 = np.array(self.fluxdf['Contents'])
        self.inv_steady12 = 1.0 / self.steadyC12
        self.dprod = np.array(self.fluxdf['Production'])
        self.dprodevent = np.zeros(self.nBoxes)
        self.dprodevent[0] = 0.45
        self.dprodevent[1] = 0.05
        self.dprodevent[11] = 0.45
        self.dprodevent[12] = 0.05
        self.eventproduction = [[interp1d(np.arange(0,2),np.zeros(2),fill_value=0,bounds_error=False)]]
        self.meanProduction = totprod
        self.production = []
        self.prodlimit = 1
        self.prodlim = True
        fluxes = pd.read_excel(projectPath/fluxesPath/self.fluxFile)

        fluxes12C = []
        for i in range(self.nBoxes):
            row = []
            for j in range(self.nBoxes):
                row.append(fluxes[j][i])
            fluxes12C.append(row)
        fluxes12C = np.array(fluxes12C)
        self.tflux = fluxes12C
        self.tflux_T = fluxes12C.T

        self.prodvec = self.meanProduction * self.dprod
        self.decay = -landa

        self.steadystateName = projectPath/('SimulationResults/SteadyStates/MiniMiniSteadystate'+str(self.meanProduction) + self.fluxFile[:-5] +'.txt')

        self.initializeSteadystate()

        intcalFileName = projectPath/(f'Data/IntCal/FullYearlyIntCalFastMiniMini{dt}' + str(self.meanProduction) + self.fluxFile[:-5] + '.txt')
        if Path(intcalFileName).is_file():
            self.intCalData = np.loadtxt(intcalFileName)
        else:
            filename = projectPath/'Data/IntCal/Intcal20.xlsx'
            intcaldata = loadexcel(filename)
            [delta, sigm, year] = getDeltafromDataframe(intcaldata)
            self.getstartState(delta[0], self.refbox, preTime=1000)
            self.productionReconstruction([delta, sigm, year], self.refbox)
            self.production = []
            allboxes = self.simResult
            yearlyboxes = allboxes[::int(1/dt)]
            np.savetxt(intcalFileName, np.array(yearlyboxes))
            self.intCalData = np.array(yearlyboxes)


    def getstartState(self, startdelta, boxindex, preTime=10000):
        delta = np.ones(preTime) * startdelta
        years = np.arange(preTime)
        sigm = np.zeros(preTime)
        self.productionReconstruction([delta, sigm, years], boxindex)
        self.box0 = self.simResult[-1]
        del self.production[-1]
        return self.box0

    def getIntCalStartState(self,t0, startdelta=0, boxindex=12):
        intcalt0 = -53050.0
        if t0 < 1950:
            self.box0 = self.intCalData[int(t0 - intcalt0)]
        else:
            print('No intcal startstate found: startstate is set to steadystate')
            self.getstartState(startdelta, boxindex, preTime=100)
        return self.box0

    def productionReconstruction(self, data, boxindex,smoothing=None):
        [delta, deltasigm, years] = data
        delta = chooseSmoothing(years, delta, smoothing)
        years = np.append(years[0]-1,years)
        delta0 = (self.box0[boxindex] / self.steadyC12[boxindex] - self.ref) / self.ref * 1000
        delta = np.append(delta0,delta)
        interpol = interp1d(years, delta, fill_value="extrapolate")
        self.times = np.arange(min(years), max(years) + self.dt, self.dt)
        boxes = self.box0.copy()
        nT = len(self.times)
        result = np.empty((nT,self.nBoxes))
        reconstprod = np.empty(len(self.times))
        for i, t in enumerate(self.times):
            result[i] = boxes
            deltaData = interpol(t + self.dt)
            boxes1 = boxes + self.dt * boxDGL_numba(
                boxes, self.tflux, self.tflux_T,
                self.inv_steady12, self.prodvec,
                self.decay, self.nBoxes
            )
            p = ((deltaData / 1000 * self.ref + self.ref) * self.steadyC12[boxindex]
                 - boxes[boxindex]) / (self.dprod[boxindex]*self.dt)
            if self.prodlim and p < -self.meanProduction * self.prodlimit:
                p = -self.meanProduction * self.prodlimit
            boxes1 += self.dt * p * self.dprod
            reconstprod[i] = p
            boxes = boxes1

        self.production.append([interp1d(self.times, reconstprod, fill_value=0, bounds_error=False)])
        self.simResult = np.array(result)

    def simulate(self, t0, tmax):
        self.times = np.arange(t0, tmax + 2*self.dt, self.dt)
        nT = len(self.times)
        boxes = self.box0.copy()
        result = np.empty((nT,self.nBoxes))
        self.prod_values = np.vstack([
            p[0](self.times) for p in self.production
        ])
        self.event_prod_values = np.vstack([
            p[0](self.times) for p in self.eventproduction
        ])
        for i in range(nT):
            result[i] = boxes
            boxes += self.dt * boxDGL_numba(
                boxes, self.tflux, self.tflux_T,
                self.inv_steady12, self.prodvec,
                self.decay, self.nBoxes
            )
            boxes += self.dt * np.sum(self.prod_values[:,i,None] * self.dprod, axis=0)
            boxes += self.dt * np.sum(self.event_prod_values[:,i,None] * self.dprodevent, axis=0)
        self.simResult = result

    def initializeSteadystate(self):
        if Path(self.steadystateName).is_file():
            self.box0 = np.loadtxt(self.steadystateName)
            self.steadystate = self.box0.copy()
        else:
            boxes = pd.read_excel(projectPath/fluxesPath/self.fluxFile)
            steadyboxes = [0]*self.nBoxes
            self.box0 = np.array(steadyboxes, dtype=float)
            for _ in range(3600000):
                self.box0 += self.dt * boxDGL_numba(
                    self.box0,
                    self.tflux,
                    self.tflux_T,
                    self.inv_steady12,
                    self.prodvec,
                    self.decay,
                    self.nBoxes
                )
            self.steadystate = self.box0.copy()
            np.savetxt(self.steadystateName, self.steadystate)

        self.ref = self.steadystate[self.refbox] / self.steadyC12[self.refbox]

    def getSimulationResults(self):
        # steadystate = np.loadtxt(Sim.steadystateName)
        times = self.times
        n = self.nBoxes
        ref = self.ref
        production = self.production
        result = np.array(self.simResult)
        c12 = self.steadyC12  # shape (T, nBoxes)
        c14 = result # shape (T, nBoxes)
        deltas = (c14 / c12 - ref) / ref * 1000
        deltas = deltas.transpose()
        return times, production, np.array(deltas)

    @timer
    def AnnualmonteCarloProdReconstruction(self, data, boxindex, N=1000, smoothing='Golay112', kind='linear'):
        [delta, deltasigm, years] = data
        distributedDelta = np.random.normal(loc=delta,scale=deltasigm,size=(N, len(delta)))
        self.productionReconstruction([delta, deltasigm, years], boxindex, smoothing=smoothing)
        reconstproduction = self.production[0][0](self.times)
        binsize = int(1 / self.dt)
        n_years = len(reconstproduction) // binsize
        prod_idx = np.arange(n_years * binsize).reshape(n_years, binsize)
        self.annualtimes = np.arange(min(self.times), min(self.times) + n_years)
        self.annualproductions = reconstproduction[prod_idx].mean(axis=1)
        self.annualmonteprods = np.empty(shape=(N, len(self.annualtimes)))
        for i in range(N):
            self.productionReconstruction([distributedDelta[i], deltasigm, years], boxindex, smoothing=smoothing)
            reconstproduction = self.production[0][0](self.times)
            self.annualmonteprods[i] = reconstproduction[prod_idx].mean(axis=1)
            self.production = []
        self.production = [[interp1d(self.annualtimes, self.annualproductions, fill_value=0, kind='previous', bounds_error=False)]]
        self.simulate(min(self.annualtimes), max(self.annualtimes)+1)
        self.annualmonteprod_sig = self.annualmonteprods.std(axis=0,ddof=1)

    @timer
    def monteCarloProdReconstruction(self, data, boxindex, N=1000, smoothing='Golay112', kind='linear'):
        [delta, deltasigm, years] = data
        distributedDelta = np.random.normal(loc=delta, scale=deltasigm, size=(N, len(delta)))
        self.productionReconstruction([delta, deltasigm, years], boxindex, smoothing=smoothing)
        self.dttimes = self.times.copy()
        self.dtproductions = self.production[0][0](self.dttimes)
        self.dtmonteprods = np.empty(shape=(N, len(self.dttimes)))
        for i in range(N):
            self.productionReconstruction([distributedDelta[i], deltasigm, years], boxindex, smoothing=smoothing)
            self.dtmonteprods[i] = self.production[0][0](self.dttimes)
            self.production = []
        self.production = [[interp1d(self.dttimes, self.dtproductions, fill_value=0, bounds_error=False)]]
        self.simulate(min(self.dttimes), max(self.dttimes))
        self.dtmonteprod_sig = self.dtmonteprods.std(axis=0, ddof=1)





@njit
def boxDGL_numba(boxes, tflux, tflux_T, inv_steady12, prodvec, decay, n):
    boxes14 = boxes
    d14scaler = boxes14 * inv_steady12
    ones = np.ones(n)
    dflux14 = -(tflux @ ones) * d14scaler + tflux_T @ d14scaler
    dboxes = dflux14 + decay * boxes14 + prodvec
    return dboxes
