import numpy as np
import pandas as pd
import xlsxwriter
from pathlib import Path
from Library.timerfunction import timer

class datadict(dict):
    def __init__(self, *args, **kwargs):
        super(datadict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __add__(self, other):
        keys = list(self.keys())
        otherkeys = list(other.keys())
        if len(otherkeys) == 0:
            return self
        otherlen = len(other[otherkeys[0]])
        if len(keys) == 0:
            thislen = 0
            for key in otherkeys:
                self[key] = np.full(thislen, np.nan)
        else:
            thislen = len(self[keys[0]])
        for key in keys:
            if key in other.keys():
                self[key] = np.append(self[key], other[key])
            else:
                self[key] = np.append(self[key], np.full(otherlen, np.nan))
        for key in otherkeys:
            if key not in keys:
                self[key] =np.append(np.full(thislen, np.nan),other[key])
        return self


def calcD14C(df):
    newdf = {}
    #for i,time in enumerate(df['bp']):
    #    df['bp'][i] = round(time,0)
    for key in df.keys():
        newdf[key] = np.array(df[key])
    newdf['fm'] = np.array(newdf['fm'],dtype=float)
    newdf['bp'] = np.array(newdf['bp'], dtype=float)
    newdf['year'] = 1950-newdf['bp']
    newdf['fm_sig'] = np.array(newdf['fm_sig'], dtype=float)
    newdf['d14C'] = (newdf['fm']*np.exp(newdf['bp']/8267)-1)*1000
    newdf['delta'] = (newdf['fm']*np.exp(newdf['bp']/8267)-1)*1000
    newdf['d14C_sig'] = newdf['fm_sig']*np.exp(newdf['bp']/8267)*1000
    newdf['delta_sig'] = newdf['fm_sig']*np.exp(newdf['bp']/8267)*1000
    newdf['c14_age'] = -8033*np.log(newdf['fm'])
    newdf['c14_age_sig'] = 8033/newdf['fm']*newdf['fm_sig']
    newdf['age'] = -8033*np.log(newdf['fm'])
    newdf['age_sig'] = 8033/newdf['fm']*newdf['fm_sig']
    return newdf



def groupdf(df, sortkey):
    data = {}
    for key in df.keys():
        data[key] = np.array(df[key])
    _, idx = np.unique(data[sortkey], return_index=True)
    keys = data[sortkey][np.sort(idx)]
    result = {}
    for key in keys:
        idx = np.where(data[sortkey]==key)
        result[key] = {}
        for key2 in data.keys():
            result[key][key2] = data[key2][idx]
    return result

def sortdf(df,sortkey,order='normal'):
    sortedind = df[sortkey].argsort(kind='stable')
    if order!='normal':
        sortedind = sortedind[::-1]
    for key in df.keys():
        df[key] = df[key][sortedind]
    return df


def getDeltafromDataframe(df):
    delta = []
    deltasigm = []
    years = []
    for i,time in enumerate(df['bp']):
        df['bp'][i] = round(time,0)
    bpdf = groupdf(df, 'bp')
    halftime = 8267
    for i,bp in enumerate(bpdf.keys()):
        N = len(bpdf[bp]['fm'])
        weight = 1/bpdf[bp]['fm_sig']**2
        #weight = np.ones(N)
        sig = bpdf[bp]['fm_sig']
        fm = float(sum(weight*bpdf[bp]['fm'])/sum(weight))
        #fm_sig = float(sum(bpdf[bp]['fm_sig']**2)**0.5/N)
        fm_sig = float(sum(weight**2*bpdf[bp]['fm_sig']**2)**0.5)/sum(weight)
        #fm_sig = np.sqrt(1/sum(sig**2)/N)
        fbp = float(bp)
        years.append(1950-fbp)
        delta.append((fm*np.exp(fbp/halftime)-1)*1000)
        deltasigm.append(np.exp(fbp/halftime)*1000*fm_sig)
    delta = np.array(delta)
    deltasigm = np.array(deltasigm)
    years = np.array(years)
    sortind = np.argsort(years)
    delta = delta[sortind]
    deltasigm = deltasigm[sortind]
    years = years[sortind]
    return np.array(delta), np.array(deltasigm), np.array(years)



def getF14CfromDataframe(df):
    fms = []
    fm_sigs = []
    years = []
    for i,time in enumerate(df['bp']):
        df['bp'][i] = round(time,0)
    bpdf = groupdf(df, 'bp')
    halftime = 8267
    for i,bp in enumerate(bpdf.keys()):
        N = len(bpdf[bp]['fm'])
        weight = 1/bpdf[bp]['fm_sig']**2
        #weight = np.ones(len(bpdf[bp]['fm_sig']))
        fm = float(sum(weight*bpdf[bp]['fm'])/sum(weight))
        #fm_sig = float(sum(bpdf[bp]['fm_sig']**2)**0.5/N)
        fm_sig = float(sum(weight ** 2 * bpdf[bp]['fm_sig'] ** 2) ** 0.5) / sum(weight)
        fbp = float(bp)
        years.append(1950-fbp)
        fms.append(fm)
        fm_sigs.append(fm_sig)
    fms = np.array(fms)
    fm_sigs = np.array(fm_sigs)
    years = np.array(years)
    sortind = np.argsort(years)
    years = years[sortind]
    fms = fms[sortind]
    fm_sigs = fm_sigs[sortind]
    return fms, fm_sigs, years

def loadexcel(file):
    df = pd.read_excel(file)
    retdf = {}
    for key in df.keys():
        retdf[key] = np.array(df[key])
    return retdf

def killNans(df,killkeys):
    for key in killkeys:
        naninds = ~np.isnan(df[key])
        for key in df:
            df[key] = df[key][naninds]
    return df



def loadexcel(filename):
    edf = pd.read_excel(filename)
    df = {}
    for key in edf:
        df[key] = np.array(edf[key])
    return df


def loadcsv(filename):
    edf = pd.read_csv(filename)
    df = {}
    for key in edf:
        df[key] = np.array(edf[key])
    return df







