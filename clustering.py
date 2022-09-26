from radial_constrained_cluster_copy import *


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





def elbow_K_MEAN(K_spatial_elbow, to_fit):

    distortions = []

    for k in tqdm(K_spatial_elbow):
        sleep(1)
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(to_fit)
        distortions.append(kmeanModel.inertia_)

    fig = plt.figure(figsize=(15,7))
    plt.plot(K_spatial_elbow, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow graph')
    plt.grid()

    return fig


def spatial_K_MEANS(n_reg, to_fit, old_shape, dataset, custom_region, custom_name, lon_boundaries, lat_boundaries, shp):



    model = KMeans(n_clusters=n_reg, max_iter=300, random_state=1).fit(to_fit)
    prediction = model.predict(to_fit)
    pred = np.reshape(prediction,[old_shape[2], old_shape[3]])
    pred =  pred.astype(float)
    pred = xr.DataArray(pred).rename('pred').rename(dim_0='lat', dim_1='lon')

    dataset = xr.merge([pred,dataset], compat='override')


    plt.figure(figsize = (10,10))

    base = shp.boundary.plot(figsize=(10,10), color = 'k')
    axs = dataset.pred.plot(ax = base)
    #region_division.boundary.plot(ax = base, color = 'r', linewidth=0.2, label = 'States boundaries')

    custom_region.boundary.plot(ax = base, color = 'g', linewidth=2, label = custom_name)

    plt.xlim([np.min(lon_boundaries), np.max(lon_boundaries)])
    plt.ylim([np.min(lat_boundaries), np.max(lat_boundaries)])
    plt.title('Region clustering')
    plt.xlabel('Longitude [°E]')
    plt.ylabel('Latitude [°N]')
    #plt.legend()
    plt.grid()

    return dataset, plt





def elbow_RCC(RCC_temporal_elbow, to_fit, n_iter, learning_rate, l_r_scheduler, scheduling_factor, len_consistancy_check, min_len):

    err = []

    for i in tqdm(RCC_temporal_elbow):

        model = Radially_Constrained_Cluster(to_fit, i, n_iter, 
                                learning_rate, l_r_scheduler, scheduling_factor, 
                                len_consistancy_check, min_len)

        model.fit()

        err.append(model.get_final_error())

    plt.figure(figsize=(18,8))
    plt.plot(RCC_temporal_elbow,err, '*--')
    plt.grid()
    plt.xlabel('N° clusters')
    plt.ylabel('Within-Cluster Sum of Square')
    plt.title('Elbow graph')

    return plt



def season_clustering(to_fit, n_seas, n_iter, learning_rate, l_r_scheduler, scheduling_factor, len_consistancy_check, min_len, dataset, fields):

    model = Radially_Constrained_Cluster(to_fit, n_seas, n_iter, 
                                learning_rate, l_r_scheduler, scheduling_factor, 
                                len_consistancy_check, min_len)

    model.fit()
    prediction = model.get_prediction()

    fig_learning = plt.figure(figsize = (15,6))
    plt.plot(model.error_story)
    plt.grid()
    plt.xlabel('N° iterations')
    plt.ylabel('Within-Cluster Sum of Square')

    cc = model.get_centroids()  
    bb = model.breakpoints

    fig_clustered_field, ax = plt.subplots(figsize = (15,6))
    scatter = ax.scatter(range(np.size(to_fit, axis=0)), getattr(dataset,fields[0]).mean(['lat','lon']), c = prediction)
    legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Seasons")
    ax.set_title('Seasonal clustering')
    ax.add_artist(legend1)
    ax.grid()
    ax.set_xlabel('julian day')
    ax.set_ylabel('Tp [mm/day]')
    ax.set_xticks([0,31,60,91,121,152,182,213,244,274,305,335,366],['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec','jan'])


    fig_clustered_var, axs = plt.subplots(1,2, figsize = (18,6), sharey = True, sharex=True)
    axs[0].scatter(to_fit[:,0],to_fit[:,1], c = prediction )
    for i in range(n_seas):
        axs[0].scatter(cc[i][0],cc[i][1], s = 200)
        axs[0].scatter(cc[i][0],cc[i][1], s = 200)
    axs[0].grid()
    axs[0].set_xlabel('PC 1')
    axs[0].set_ylabel('PC 2')

    axs[1].scatter(to_fit[:,0],to_fit[:,2], c = prediction)
    axs[1].grid()
    axs[1].set_xlabel('PC 1')
    axs[1].set_ylabel('PC 3')

    return bb, prediction, fig_learning, fig_clustered_field, fig_clustered_var