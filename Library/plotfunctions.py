import matplotlib.pyplot as plt
#from SimFunctions import *
#from fftStuff import *
from matplotlib.axis import Axis
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from numpy import searchsorted

def CE_BCE_format(x,pos):
    if x==0:
        return '1 CE'
    elif x<0:
        return f'{int(-x)} BCE'
    else: return f'{int(x)} CE'


def CE_BCE_format_nolabel(x,pos):
    if x==0:
        return '1'
    elif x<0:
        return f'{-x:.0f}'
    else: return f'{x:.0f}'


def subplots(nplots):
    fig, firstax = plt.subplots()
    firstax.spines['bottom'].set_visible(False)
    firstax.spines['top'].set_visible(False)
    firstax.spines['left'].set_visible(False)
    firstax.spines['right'].set_visible(False)
    firstax.set_yticks([])
    ax = []
    for i in range(nplots):
        ax.append(firstax.twinx())
    return fig, ax,firstax


def separateSubplots(ax,overlap = 0.2,ylabelx = (0.08,0.05),fontsize=18,plotlabels=True,starlabelind = 0,labels=None,leftright=0):
    if labels is None:
        labels = ['a', 'b', 'c', 'd','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u']
    n = len(ax)
    xlim = ax[0].get_xlim()
    xticks = ax[0].get_xticks()
    for i, x in enumerate(ax):
        ticks = x.get_yticks()
        dticks = ticks[1]-ticks[0]
        maxy = max(ticks)
        miny = min(ticks)
        dy = maxy - miny
        nonscaletotheight = n * dy
        totheight = nonscaletotheight - (n - 1) * overlap * dy
        top = maxy + i * (1 - overlap) * dy
        bottom = top - totheight
        ylabelheight = (miny - bottom+dy/2) / totheight
        labelheight = (miny+0.8*dy - bottom) / totheight
        if i % 2 == leftright:
            x.yaxis.tick_left()
            x.spines['left'].set_bounds((ticks[1], ticks[-2]))
            x.spines['right'].set_visible(False)
            Axis.set_label_coords(x.yaxis,-ylabelx[0],ylabelheight)
        else:
            x.yaxis.tick_right()
            x.spines['right'].set_bounds((ticks[1], ticks[-2]))
            x.spines['left'].set_visible(False)
            Axis.set_label_coords(x.yaxis,1+ylabelx[1],ylabelheight)
        x.set_ylim(top=top, bottom=bottom)
        x.set_yticks(ticks[1:-1])
        x.spines['top'].set_visible(False)
        if plotlabels:
            x.text(0.03, labelheight, labels[i+starlabelind], fontsize=fontsize, horizontalalignment='left', verticalalignment='center',
                   transform=x.transAxes)
        if i ==0:
            x.spines['bottom'].set_bounds((xticks[1],xticks[-2]))
        else:
            x.spines['bottom'].set_visible(False)