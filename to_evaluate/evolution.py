from radial_constrained_cluster_copy import *
from dataset_managment import *
from report_managment import *
from EOF_various import *
from clustering import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import xarray as xr
from eofs.multivariate.standard import MultivariateEof
from shapely.geometry import mapping
from sklearn.cluster import KMeans
from scipy import signal
from obspy.signal.tf_misfit import cwt
import pymannkendall as mk
import dask
import time
import sys
import os
import threading
from time import sleep
from tqdm import tqdm


def creating_periods(dataset, step):

    year =  np.unique(dataset.time['time.year'].to_numpy())
    periods = np.arange(year[0+step],year[-1-step],1)

    time_series = []

    for i in tqdm(np.arange(0,len(periods),1)):

        sleep(3)
    
        to_analyze = dataset.sel(time = slice(str(periods[i]-step),str(periods[i]+step)))
        to_analyze = to_analyze.groupby('time.dayofyear').mean()
        time_series.append(to_analyze)

    return time_series, periods


def adjust_list_to_vec(vec):
    
    vec = np.array(vec)
    vec = np.squeeze(vec)
    vec = np.transpose(vec)

    return vec


def variable_evolution(variable, n_seas, periods, step, colors, name_for_title):

    plt.figure(figsize=(15,7))
    values = []

    for i in range(n_seas):
        x = periods
        y = variable[i]
    
        idx = np.isfinite(x) & np.isfinite(y)
        ab = np.polyfit(x[idx], y[idx], 1)
    
        plt.plot(x, ab[1]+x*ab[0], c = colors[i])
        plt.plot(x,variable[i], '-.*', label = f'season {i}', c = colors[i], markersize = 15)
        result = mk.original_test(variable[i])
        
        values.append([f'season {i} - Trend {ab[0]} - sign {result.h} - p_value {result.p}'])
    
    plt.xlabel('year')
    plt.grid()
    plt.legend()
    plt.title(name_for_title)

    return plt, values


def breakpoints_evolution(b_p, n_seas, periods, step, colors):

    plt.figure(figsize=(15,7))

    values = []

    for i in range(n_seas):
        x = periods
        y = b_p[i]
    
        idx = np.isfinite(x) & np.isfinite(y)
        ab = np.polyfit(x[idx], y[idx], 1)
    
        if i == n_seas-1:
        
            plt.plot(x, ab[1]+x*ab[0], c = colors[0])
            plt.plot(x,b_p[i], '-.*', label = f'season {0}', c = colors[0], markersize = 15)
            result = mk.original_test(b_p[i])
            values.append([f'season {0} - Trend {ab[0]} - sign {result.h} - p_value {result.p}'])
        
        else:
    
            plt.plot(x, ab[1]+x*ab[0], c = colors[i+1])
            plt.plot(x,b_p[i], '-.*', label = f'season {i+1}', c = colors[i+1], markersize = 15)
            result = mk.original_test(b_p[i])
            values.append([f'season {i+1} - Trend {ab[0]} - sign {result.h} - p_value {result.p}'])
    
    plt.xlabel('year')
    plt.ylabel('season breakpoint [julian day]')
    plt.yticks([0,31,60,91,121,152,182,213,244,274,305,335,366],['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec','jan'])
    plt.grid()
    plt.legend()
    plt.title('Breakpoints evolution')

    return plt, values



def compare_cycles(time_series, fields, indices, roll_mean_step, periods, step):

    fig = plt.figure(figsize = (15,7))

    for i in indices:

        getattr(time_series[i], fields[0]).rolling(dayofyear=roll_mean_step, center=True).mean().mean(['lat','lon']).plot(label = f'{periods[i]-step} - {periods[i]+step}')

    plt.xticks([0,31,60,91,121,152,182,213,244,274,305,335,366],['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec','jan'])
    plt.legend()
    plt.grid()

    return fig