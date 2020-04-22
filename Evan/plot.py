#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, ListedColormap
from comp import sort_tuning_curves


def trace(traces, index=0):
    plt.plot(range(np.size(traces, 1)), traces[index,])
    plt.xlabel("Frame #")
    plt.ylabel("Fluorescence ($F$)")


def tuning_curve(curves, ci, index=0):
    plt.errorbar(range(np.size(curves, 1)), curves[index,], ci[index,].T, fmt=".")
    plt.xlabel("Stimulus")
    plt.ylabel("Response ($\Delta F/F$)")


def tuning_curve_matrix(curves, sort=None, vertbars=True):
    if sort==True: curves = curves[sort_tuning_curves(curves),:]
    elif sort:     curves = curves[sort,:]
    plt.imshow(curves, interpolation="none", aspect="auto") #, extent=[-.5, np.size(curves,1)-.5, np.size(curves,0)-.5, -.5]
    plt.plot([-.5,np.size(curves,1)+.5], [-.5,np.size(curves,0)+.5], 'w--', linewidth=1)
    if vertbars:
        if vertbars==True:
            vertbars=[.5,5.5,15.5,25.5,30.5]
            if np.size(curves,1)==31: # no catch
                vertbars = vertbars[1:]-1
            elif np.size(curves,1)==26: # multi-piston stimuli only
                vertbars = vertbars[2:]-6
        for v in vertbars:
            plt.axvline(v, color='w', linestyle='--', linewidth=.5)
    plt.xlabel("Stimulus")
    plt.ylabel("Neuron")
    plt.colorbar()