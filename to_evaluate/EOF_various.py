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




def dataset_to_eof(dataset, fields, n_eof_comp):

    '''
        This function returns the basis of the EOF analysis. In order to find seasonal cluster,
        data are normalized with max division
    '''

    status = 0
    eofs = 0
    old_shape = 0 

    # Creating list for fields numpy array storing
    to_eof = []

    # Standardizing data and saving field as np array
    if status == 0:

        try:

            for i in fields:
                supp = getattr(dataset,i)
                supp = supp/supp.max('time')
                to_eof.append(supp.to_numpy())

        except:
            status = 'Something went wrong with dataset normalization and storing'

    # Creating EOF's basis
    if status == 0:

        try:
            solver = MultivariateEof(to_eof, center = False)
            eofs = solver.eofs(neofs = n_eof_comp)
            exp = solver.varianceFraction()
    
            for j in range(n_eof_comp):
                eofs[0][j][:][:]= eofs[0][j][:][:]*exp[j]

        except:
            status = 'Something went wrong with EOFs computing'

    # Reshaping EOF's
    if status == 0:

        try:
            old_shape = np.shape(eofs)
            new_shape = [np.size(eofs, axis = 0)*np.size(eofs, axis = 1), np.size(eofs, axis = 2)*np.size(eofs, axis = 3)]
            eofs = np.reshape(eofs, new_shape)
            # Now we have n_fields*366 x lon*lat
            # We want x - > lon- lat, y-> features so we have to reshape
            eofs = np.transpose(eofs)

        except:
            status = 'Could not reshape eofs'
    
    return eofs, old_shape, status




def dataset_to_pc(dataset, fields, n_eof_comp):

    to_eof = []
    for i in fields:
        supp = getattr(dataset,i)
        supp = (supp-supp.mean())/supp.std()
        to_eof.append(supp.to_numpy())

    solver = MultivariateEof(to_eof)
    pc = solver.pcs(npcs = n_eof_comp)

    fig_1, axs = plt.subplots(2,1, figsize = (18,6), sharex = True, sharey=True)
    axs[0].plot(pc[:,0])
    axs[0].grid()
    axs[0].set_ylabel('PC 1')

    axs[1].plot(pc[:,1])
    axs[1].set_ylabel('PC 2')
    axs[1].set_xticks([0,31,60,91,121,152,182,213,244,274,305,335,366],['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec','jan'])
    axs[1].grid()

    fig_2 = plt.figure(figsize = (10,10))
    plt.scatter(pc[:,0],pc[:,1])
    plt.grid()
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')

    return pc, fig_1, fig_2
