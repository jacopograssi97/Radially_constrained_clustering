from radial_constrained_cluster_copy import *
from dataset_managment import *
from report_managment import *
from EOF_various import *
from clustering import *
from evolution import *

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


font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

dask.config.set(scheduler="synchronous")

#################################################################################################################################
#################################################################################################################################
#################################################################################################################################

lat_boundaries = [5,40]
lon_boundaries = [65,100]
time_boundaries = 'Original'

n_eof_comp = 4
do_detrend = False
fields = ['tp']

space_elbow = True
temporal_elbow = True
K_spatial_elbow = range(2,3)
n_reg = 5

to_see = 'hkk'

dataset_paths = ["~/work/jacopo/DATA/ERA5_prec_day_INDIA.nc"]

lon_name = 'lon'
lat_name = 'lat'

#"~/work/jacopo/DATA/ERA5_prec_day_INDIA.nc"
#"/work/datasets/obs/ERA5/total_precipitation/day/ERA5_total_precipitation_day_0.25x0.25_sfc_1950-2020.nc"
#"~/work/jacopo/DATA/APHRODITE_1951_2007_daily_INDIA.nc"
#"/archive/paolo/cmip6/CMIP6/model-output/EC-Earth-Consortium/EC-Earth3/historical/atmos/day/r1i1p1f1/pr/*.nc"
#"/home/paolo/archive/cmip6/CMIP6/model-output/EC-Earth-Consortium/EC-Earth3/ssp585/atmos/day/r1i1p1f1/pr/*.nc"

# Add a shapefile for masking
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
#world = world.translate(-180,0)
region_division = world

# Add custom shapefiles o box for region selection
#custom_region = gpd.GeoDataFrame({'geometry': [Polygon([(78, 25), (93, 25), (93, 32), (78, 32)])]})   #Himalaya
custom_region = gpd.GeoDataFrame({'geometry': [Polygon([(71, 32), (78, 32), (78, 37), (71, 37)])]})  #hkk
custom_name = 'hkk'


len_consistancy_check = True
min_len = 10

learning_rate = 10
scheduling_factor = 1.1
l_r_scheduler = True

n_iter = 2000


RCC_temporal_elbow = range(1,15)
n_seas = 3

step = 10

report_name = 'ERA5_HKK'
dataset_name = 'ERA_5'
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################

#################################################################################################################################
## PRELIMINARY OPERATION

# Clear screen for output
os.system('cls' if os.name == 'nt' else 'clear')

# Change in Report dir for report save
try: 
    os.chdir('Report/')

except:
    raise ValueError('Please create Report directory in this path. Report will be now saved in current path')

# Creating the report
try:
    report = Report(report_name, 810, 19, 50)
    report.add_header()

except:

    raise ValueError('Could not create the report')
    



#################################################################################################################################
## DATASET MANAGMENT

# Importing dataset
print('Importing dataset: ', end="")
with Spinner():

    # If only one path provided open single dataset
    if len(dataset_paths) == 1:
        dataset, status = import_singledataset(dataset_paths[0], lat_boundaries, lon_boundaries, lat_name, lon_name, time_boundaries)

    # If more paths provided open multiple dataset and concatenate on time dimension
    else:
        dataset, status = import_multipledataset(dataset_paths, lat_boundaries, lon_boundaries, lat_name, lon_name, time_boundaries)

    # Checkig if all went good - othervise abort
    if status != 0:
        report.early_exit('An error occoured in dataset importation', status)

print('\b', 'Done')


dataset = dataset.load()

# Detrending dataset if choosen
if do_detrend == True:

    print('Detrending dataset: ', end="")
    with Spinner():

        dataset, status = detrend(dataset, fields[0], 'time', 1)

        if status != 0:
            report.early_exit('An error occoured in dataset detrending', status)

    print('\b', 'Done')

else:

    print('Detrending dataset: skipped')


# Saving in report the dataset specifications
lat_boundaries = [float(dataset.lat[0]),float(dataset.lat[-1])]
lon_boundaries = [float(dataset.lon[0]),float(dataset.lon[-1])]

report.add_line("Dataset specifications",12)
report.add_line(f"Dataset name: {dataset_name}",10)
report.add_line(f"Latitude boundaries: {np.min(lat_boundaries)} - {np.max(lat_boundaries)} ---- Resolution: {np.abs(float(dataset.lat[0]-dataset.lat[1]))}",10)
report.add_line(f"Longitude boundaries: {np.min(lon_boundaries)} - {np.max(lon_boundaries)} ---- Resolution: {np.abs(float(dataset.lon[0]-dataset.lon[1]))}",10)
report.add_line(f"Time boundaries: {int(dataset.time[0]['time.year'])} - {int(dataset.time[-1]['time.year'])}", 10)
report.add_line(f"Dataset linear detrending: {do_detrend}",10)




#################################################################################################################################
## SPATIAL CLUSTERING

# Creating day of year dataset and masking
print('Creating seasonal cycle for spatial clustering: ', end="")
with Spinner():

    dataset_doy, status = cycle_doy_mask(dataset, world, 1)

    if status != 0:
        report.early_exit('An error occoured in dataset masking or seasonal cycle creation', status)

print('\b', 'Done')



####### DIM REDUCTION FOR SPATIAL CLUSTERING
# Insert here the method for dim reduction
# Please note that the method has to return:
# to_fit -> np.array() 2D with x-> grid points, y-> features
# old shape -> the shape of the map before reshaping

print('Performing EOF: ', end="")
with Spinner():

    to_fit, old_shape, status = dataset_to_eof(dataset_doy, fields, n_eof_comp)

    if status != 0:
        report.early_exit('An error occoured EOF computing: ', status)
    
print('\b', 'Done')

######### END OF DIM REDUCTION


if space_elbow == True:

    print('Computing elbow graph for spatial clustering: ')
    
    fig = elbow_K_MEAN(K_spatial_elbow, to_fit)
    report.add_graph(fig, 400, 200)
    report.add_line("",12)

    print('\b', 'Done')

else:

    print('Computing elbow graph for spatial clustering: skipped')


report.add_line(f"Number of regions: {n_reg}",11)


print('Performing spatial clustering: ', end="")
with Spinner():

    dataset, fig_spat_clust = spatial_K_MEANS(n_reg, to_fit, old_shape, dataset, custom_region, custom_name, lon_boundaries, lat_boundaries, world)
    report.add_graph(fig_spat_clust, 600, 600)

print('\b', 'Done')


print('Selecting region: ', end="")
with Spinner():

    data, data_doy, fig_cycle_region = region_selection(to_see, dataset, custom_region, custom_name, fields)
    report.add_graph(fig_cycle_region, 550, 200)

print('\b', 'Done')


report.add_line("",12)
report.add_line("Spatial clustering",12)
report.add_line(f"Number of EOF's component: {n_eof_comp}",10)
report.add_line(f"Region selected: {to_see}",10)



#################################################################################################################################
## SEASONAL CLUSTERING


####### DIM REDUCTION FOR SEASONAL CLUSTERING
# Insert here the method for dim reduction
# Please note that the method has to return:
# to_fit -> np.array() 2D with x-> dayofyear value, y-> features
# old shape -> the shape of the map before reshaping

print('Performing PC: ', end="")
with Spinner():

    to_fit_seas, fig_1, fig_2 = dataset_to_pc(data_doy, fields, n_eof_comp)
    report.add_graph(fig_1, 550, 200)
    report.add_graph(fig_2, 200, 200)

print('\b', 'Done')

######### END OF DIM REDUCTION


if temporal_elbow == True:

    print('Computing elbow graph for temporal clustering: ')
    
    fig = elbow_RCC(RCC_temporal_elbow, to_fit_seas, n_iter, learning_rate, l_r_scheduler, scheduling_factor, len_consistancy_check, min_len)
    report.add_graph(fig, 400, 200)
    report.add_line("",12)

    print('\b', 'Done')

else:

    print('Computing elbow graph for temporal clustering: skipped')


report.add_line("",11)
report.add_line(f"N season: {n_seas}",10)


print('Performing seasonal clustering: ', end="")
with Spinner():

    bb, prediction, fig_learning, fig_clustered_field, fig_clustered_var = season_clustering(to_fit_seas, n_seas, n_iter, learning_rate, l_r_scheduler, scheduling_factor, len_consistancy_check, min_len, data_doy, fields)
    report.add_graph(fig_learning, 550, 200)
    report.add_graph(fig_clustered_field, 550, 200)
    report.add_graph(fig_clustered_var, 550, 200)

print('\b', 'Done')


#################################################################################################################################
## SEASON EVOLUTION

print('Creating periods for season evolution: ')

time_series, periods = creating_periods(data, step)

print('\b', 'Done')




b_p = []
s_len = []
var_mean = []
var_min = []
var_max = []
var_std = []


print('Processing each period:')
for h in tqdm(time_series):

    sleep(3)

    ####### DIM REDUCTION FOR SEASONAL CLUSTERING
    # Insert here the method for dim reduction
    # Please note that the method has to return:
    # to_fit -> np.array() 2D with x-> dayofyear value, y-> features

    to_fit_per, fig_1, fig_2 = dataset_to_pc(h, fields, n_eof_comp)

    ######### END OF DIM REDUCTION

    breakpoints, prediction, fig_learning, fig_clustered_field, fig_clustered_var = season_clustering(to_fit_per, n_seas, n_iter, learning_rate, l_r_scheduler, scheduling_factor, len_consistancy_check, min_len, data_doy, fields)
    b_p.append(breakpoints)
    prediction = np.squeeze(np.int32(prediction))

    season_len = np.empty((n_seas,1))
    season_len[:] = np.nan
    
    season_mean = np.empty((n_seas,1))
    season_mean[:] = np.nan

    season_min = np.empty((n_seas,1))
    season_min[:] = np.nan

    season_max = np.empty((n_seas,1))
    season_max[:] = np.nan

    season_std = np.empty((n_seas,1))
    season_std[:] = np.nan

    tr = getattr(h,fields[0]).mean(['lon','lat']).to_numpy()

    
    for j in range(n_seas):
             
        gr = tr[prediction == j]
        season_len[j] = len(gr) 
        season_mean[j] = np.mean(gr)
        season_min[j] = np.min(gr)
        season_max[j] = np.max(gr)
        season_std[j] = np.std(gr)
        
    s_len.append(season_len)
    var_mean.append(season_mean)
    var_min.append(season_min)
    var_max.append(season_max)
    var_std.append(season_std)



b_p = adjust_list_to_vec(b_p)
s_len = adjust_list_to_vec(s_len)
var_mean = adjust_list_to_vec(var_mean)
var_min = adjust_list_to_vec(var_min)
var_max = adjust_list_to_vec(var_max)
var_std = adjust_list_to_vec(var_std)

colors = ['red','blue','green','purple']


report.add_line('',10)
report.add_line(f'Step for rolling mean:  {step*2}',10)



fig_bp, report_bp = breakpoints_evolution(b_p, n_seas, periods, step, colors)
report.add_graph(fig_bp, 550, 200)
report.add_line("",10)
for i in range(len(report_bp)):
    report.add_line(f'{report_bp[i]}',8)

fig_len, report_len = variable_evolution(s_len, n_seas, periods, step, colors, 'Length')
report.add_graph(fig_len, 550, 200)
report.add_line("",10)
for i in range(len(report_len)):
    report.add_line(f'{report_len[i]}',8)

fig_mean, report_mean = variable_evolution(var_mean, n_seas, periods, step, colors, 'Mean')
report.add_graph(fig_mean, 550, 200)
report.add_line("",10)
for i in range(len(report_mean)):
    report.add_line(f'{report_mean[i]}',8)

fig_min, report_min = variable_evolution(var_min, n_seas, periods, step, colors, 'Min')
report.add_graph(fig_min, 550, 200)
report.add_line("",10)
for i in range(len(report_min)):
    report.add_line(f'{report_min[i]}',8)

fig_max, report_max = variable_evolution(var_max, n_seas, periods, step, colors, 'Max')
report.add_graph(fig_max, 550, 200)
report.add_line("",10)
for i in range(len(report_max)):
    report.add_line(f'{report_max[i]}',8)

fig_std, report_std = variable_evolution(var_std, n_seas, periods, step, colors, 'Std')
report.add_graph(fig_std, 550, 200)
report.add_line("",10)
for i in range(len(report_std)):
    report.add_line(f'{report_std[i]}',8)



roll_mean_step = 30
indices = [0, int(len(time_series)/3), int(2*len(time_series)/3), -1]

report.add_line('',10)
report.add_line(f'Step for rolling mean:  {roll_mean_step}',10)

fig_compare = compare_cycles(time_series, fields, indices, roll_mean_step, periods, step)
report.add_line("",10)
report.add_graph(fig_compare, 550, 200)



report.save()
print('Report saved at ', os.getcwd())