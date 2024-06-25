import numpy as np
import xarray as xr

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


def import_singledataset(dataset_path, lat_boundaries, lon_boundaries, lat_name, lon_name, time_boundaries):

    '''
        Importation for a single dataset (or all files contained in a folder)
        Dataset files have to be in .nc format

        -> Dataset is imported as a chunked xarray with automatic size
        -> Latitude and longitude names are standardized in lat lon
        -> Lat and lon are sorted 
        -> Spatial domain is cutted with custom values ('Original' keeps all input data)
        -> Time domain is cutted ('Original' keeps all input data)

    '''
    dataset = 0
    status = 0

    # Importing the dataset
    if status == 0:

        try:
            dataset = xr.open_mfdataset(dataset_path, chunks='auto', parallel=True)
    
        except:
            status = 'Cannot import dataset - please check your path'


    # Standardizing lat lon names
    if status == 0:

        try:
            if lat_name != 'lat':        
                dataset.rename({lat_name : 'lat'})

            if lon_name != 'lon':       
                dataset.rename({lon_name : 'lon'})

        except:
            status = 'Cannot rename lat lon - please check input names'

        

    # Standardizing lat lon sorting
    if status == 0:

        try:
            dataset = dataset.sortby(['lat']).sortby(['lon'])

        except: 
            status = 'Cannot sort lat-lon - Unknown error'

    # Cutting space domain
    if status == 0:

        try:
            if lat_boundaries != 'Original':
                dataset = dataset.sel(lat=slice(np.min(lat_boundaries),np.max(lat_boundaries)))

            if lon_boundaries != 'Original':
                dataset = dataset.sel(lon=slice(np.min(lon_boundaries),np.max(lon_boundaries)))

        except:
            status = 'Cannot cut space domain - pleas check domain consistance'

    # Cutting time domain (if None keeping original)
    if status == 0:

        try:
            if time_boundaries != 'Original':
                dataset = dataset.sel(time=slice(time_boundaries[0], time_boundaries[1]))

        except:
            status = 'Cannot cut time domain - pleas check domain consistance'

    # Removing useless dim
    if status == 0:

        try:
            dataset = remove_useless_dim(dataset)

        except:
            status = 'Cannot remove useless dim'

    return dataset, status




def import_multipledataset(dataset_paths, lat_boundaries, lon_boundaries, lat_name, lon_name, time_boundaries):

    '''
        Importation for multiple dataset (or all files contained in multiple folder)
        Datasets must have the same spatial dimension and resolution and must be time contigous
        Dataset files have to be in .nc format

        -> Datasets are imported with import single dataset
        -> Datasets are concatenated on time dimension

    '''
    dataset = 0
    # List for datasets storing
    d = []


    for i in range(len(dataset_paths)):
        
        # Opening dataset keeping original time boundaries
        dat, status = import_singledataset(dataset_paths[i], lat_boundaries, lon_boundaries, lat_name, lon_name, 'Original')

        if status == 0:
            d.append(dat)


    # Concatenating datasets on time dimension
    if status == 0:
        dataset = xr.concat(d,'time')



    return dataset, status




def remove_useless_dim(dataset):

    '''
        This function removes some variables which are automatically created when importing
        a dataset with open_mfdataset. Removing these variable is necessary in order to avoid
        future problems.

    '''

    status = 0

    # Creating a list with alla dataset attributes
    a = dir(dataset)
    
    # A list of forbidden dims
    forbidden = ['time_bnds','lat_bnds','lon_bnds']

    # Removing forbidden dims
    for i in range(len(a)):
        for j in range(len(forbidden)):
            if a[i] == forbidden[j]:
                dataset = dataset.drop(forbidden[j])


    return dataset




def detrend(dataset, variable, dimension, deg):

    '''
        This function removes the time trend from the dataset
    '''

    dataset = 0
    status = 0

    try: 
        # Fitting dataset with polynomial
        p = getattr(dataset, field).polyfit(dim=dimension, deg=deg)
        fit = xr.polyval(dataset.tp[dimension], p.polyfit_coefficients)

        # Detrending dataset
        detrended = getattr(dataset,variable)-fit

        # Removing trended variable and adding detrended variable with same name
        detrended = xr.DataArray(detrended).rename(variable)
        dataset = dataset.drop(variable)
        dataset = xr.merge([dataset,detrended])

    except:
        status = 'Cannot remove trend - unknow error'
    
    return dataset, status




def cycle_doy_mask(dataset, shp, to_fillna):

    ''' 
        This funtion creates a day of year dataset of a given dataset and 
        masks it whith a given shp which has to be a GeoPandas GeoDataFrame. 
        Nans are filled with to_fillna
    '''

    dataset_doy = 0 
    status = 0

    # Creating dayofyear dataset
    if status == 0:

        try:
            dataset_doy = dataset.groupby('time.dayofyear').mean().rename(dayofyear = 'time')

        except:
            status = 'Cannot create dayofyear dataset'

    # Masking dataset with given shp and filling nan
    if status == 0:

        try:
            dataset_doy = dataset_doy.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
            dataset_doy = dataset_doy.rio.write_crs("epsg:4326", inplace=True)
            dataset_doy = dataset_doy.rio.clip(shp.geometry.apply(mapping), shp.crs, drop=False)
            dataset_doy = dataset_doy.fillna(to_fillna)

        except:
            status = 'Cannot mask dataset - check shp requirements'

    return dataset_doy, status





def region_selection(to_see, dataset, custom_region, custom_name, fields):

    ''' 
        This funtion selects a region from the dataset. The region could be one of the previous cluster 
        or a custom region (GeoPandas GeoDataFrame) with his custom name (str).
        A graph with the time serie and a graph with che annual cycle are also created.
    '''

    data = 0
    data_doy = 0
    status = 0
    fig = 0

  
    if status == 0:

        # Check if to_see is the custom region
        if to_see == custom_name:

            try:
                # Mask and create cycle
                data = dataset.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
                data = data.rio.write_crs("epsg:4326", inplace=True)
                data = data.rio.clip(custom_region.geometry.apply(mapping), custom_region.crs, drop=False)
                data_doy = data.groupby('time.dayofyear').mean()

            except:
                status = 'Cannot exctract selected region - check custom region requirements'
    

        else:

            try:
                data = dataset.where(dataset.pred == to_see)
                data_doy = data.groupby('time.dayofyear').mean()

            except:
                status = 'Cannot exctract selected region - check you have selected a valid region'

    
    # Making the plots
    if status == 0:

        # Creating the plot to return 
        fig, axs = plt.subplots(1,2, figsize = (18,6), sharey = False)

        # Plotting time serie
        getattr(data,fields[0]).mean(['lat','lon']).plot(ax=axs[0])
        axs[0].set_title(f'Time serie precipitation over region {to_see}')
        axs[0].grid()
        axs[0].set_ylabel('Total precipitation [mm/day]')
        axs[0].set_xlabel('Year')

        # Plotting seasonal cycle
        getattr(data_doy,fields[0]).mean(['lat','lon']).plot(ax=axs[1])
        axs[1].set_title(f'Annual rainfall cycle over region {to_see}')
        axs[1].set_xticks([0,31,60,91,121,152,182,213,244,274,305,335,366],['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec','jan'])
        axs[1].set_xlabel('Julian day')
        axs[1].grid()

    return data, data_doy, fig



