# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 18:02:04 2021

@author: u0117123
"""
#Import modules
import os, random, sys, subprocess, math, time

import pylab as pl
from pylab import savefig
import matplotlib.pyplot as plt
import seaborn as sns

from descartes import PolygonPatch

from pathlib import Path

import pandas as pd
from pandas import DataFrame
import geopandas as gpd
from geopandas import GeoDataFrame

import numpy as np
from numpy import genfromtxt

import laspy
from laspy.file import File

import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import Polygon

import fiona

from scipy.spatial import Delaunay

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

from osgeo import ogr


from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#clustering packages
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.neighbors import KDTree
#%%
# Import function to create concave hulls
work_dir = r'C:\Users\u0117123\Box Sync\FWO\WP1\Point-cloud-extractions\scripts\preliminary'
os.chdir(work_dir)

import alpha_shape_function as alpha_shape_function

#%% Input variables
#RF classified veg object points
Folder = "C:\\Users\\u0117123\\Box Sync\\FWO\\WP1\\Point-cloud-extractions\\processing\\3_RF_classification\\3SA\\OutputRF\\AllFeatures\\"
Folder_cluster = "C:\\Users\\u0117123\\Box Sync\\FWO\\WP1\\Point-cloud-extractions\\processing\\5_Clustering_classification\\3SA\\r10\\Clustered_RFall_vegObj_points\\"

#Temp folder to store concave hulls
conc_hull_folder = r'C:\Users\u0117123\Box Sync\FWO\WP1\Point-cloud-extractions\processing\5_Clustering_classification\3SA\r10\RF_all\hulls\noFruit'

Points_file_Tervuren = "y_PredictedAsVeg_allF_TestTervuren_fixedDist_veg_nonveg_moreFeatures_r10.csv"
Points_file_Huldenberg = "y_PredictedAsVeg_allF_TestHuldenberg_fixedDist_veg_nonveg_moreFeatures_r10.csv"
#Points_file_Velm = "y_PredictedAsVeg_allF_TestVelm_fixedDist_veg_nonveg_moreFeatures_r10.csv"
Points_file_Velm = "y_PredictedAsVeg_allF_TestVelm_noFruit2.txt"
#%% CLUSETERING ###############################################################
###############################################################################
###############################################################################
### Timestamp 
t = time.localtime()
timestamp = time.strftime('%Y%m%d_%H%M', t)
#%%
df_V= pd.read_csv(
    Folder + Points_file_Velm , delimiter=",") # import csv file as panda df
points_CSV_V = GeoDataFrame(
    df_V.drop(['OID', 'x', 'y', 'z', 'Classifica','UserData', 'y_ref', 'y_pred_ful'], axis=1),
    crs={'init': 'epsg:31370'},
    geometry=[Point(xy) for xy in zip(df_V.x, df_V.y)]) #to geopanda dataframe

df_T = pd.read_csv(
    Folder + Points_file_Tervuren , delimiter=",") # import csv file as panda df
points_CSV_T = GeoDataFrame(
    df_T.drop(['x', 'y', 'z', 'Classification','UserData', 'y_ref', 'y_pred_full'], axis=1),
    crs={'init': 'epsg:31370'},
    geometry=[Point(xy) for xy in zip(df_T.x, df_T.y)]) #to geopanda dataframe

df_H = pd.read_csv(
    Folder + Points_file_Huldenberg , delimiter=",") # import csv file as panda df
points_CSV_H = GeoDataFrame(
    df_H.drop(['x', 'y', 'z', 'Classification','UserData', 'y_ref', 'y_pred_full'], axis=1),
    crs={'init': 'epsg:31370'},
    geometry=[Point(xy) for xy in zip(df_H.x, df_H.y)]) #to geopanda dataframe


#%%## Clustering
#Parameters
dimension = 'xy'
#parameters epsilon estimation
leaf_size = 2 
#Parameters dbscan
min_samples_nr_Velm = 15 #min sample points in cluster
min_samples_nr_Neerijse = 62


#%% epsilon estimation Velm
data_xy = df_V.iloc[:,: 2]
data_xyz = df_V.iloc[:,: 3]
#Distance matrix
tree = KDTree(globals()['data_' + dimension], leaf_size)              
dist, ind = tree.query(globals()['data_' + dimension], k=min_samples_nr_Velm + 1)   # k closest neighbors (1st is point itself)
dist_sample = dist[:,min_samples_nr_Velm]

dist_sample_sort = np.sort(dist_sample)[::-1] #Sort from largest to smallest distance
n = len(dist_sample_sort) 
nr = x = np.arange(0, n)
plt.plot(nr, dist_sample_sort)

# Find knee to estmate epsilon value
from kneed import KneeLocator
kn = KneeLocator(nr, dist_sample_sort, curve='convex', direction='decreasing')
import matplotlib.pyplot as plt
plt.xlabel('number of clusters k')
plt.ylabel('Sum of squared distances')
plt.plot(nr, dist_sample_sort)
plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
kn_x = kn.knee #knee x value
eps_nr_Velm = dist_sample_sort[kn_x] # knee y value = epsilon


#%% Clustering with DBSCAN
outputCSV_Velm = Folder_cluster + 'Velm_RF_veg_clustered_eps' + str(eps_nr_Velm) + '_minS2' + str(min_samples_nr_Velm) + 'data_' + dimension + '_' + str(timestamp) + '.csv'

model_DBSCAN = DBSCAN(eps=eps_nr_Velm, min_samples=min_samples_nr_Velm)
model_DBSCAN.fit_predict(globals()['data_' + dimension])
pred = model_DBSCAN.fit_predict(globals()['data_' + dimension])
print("number of cluster found: {}".format(len(set(model_DBSCAN.labels_))))
nr_clusters = len(set(model_DBSCAN.labels_))
print('cluster for each point: ', model_DBSCAN.labels_)
labels_DBSCAN = model_DBSCAN.labels_
df_V['cluster'] = pred
df_V = df_V[df_V.cluster != -1]
df_V.to_csv(outputCSV_Velm,index=False)


#%% epsilon estimation Huldenberg
data_xy = df_H.iloc[:,: 2]
data_xyz = df_H.iloc[:,: 3]

##Distance matrix
tree = KDTree(globals()['data_' + dimension], leaf_size)              
dist, ind = tree.query(globals()['data_' + dimension], k=min_samples_nr_Neerijse + 1)   # k closest neighbors (1st is point itself)
dist_sample = dist[:,min_samples_nr_Neerijse]

dist_sample_sort = np.sort(dist_sample)[::-1] #Sort from largest to smallest distance
n = len(dist_sample_sort) 
nr = x = np.arange(0, n)
plt.plot(nr, dist_sample_sort)

# Find knee to estmate epsilon value
from kneed import KneeLocator
kn = KneeLocator(nr, dist_sample_sort, curve='convex', direction='decreasing')
import matplotlib.pyplot as plt
plt.xlabel('number of clusters k')
plt.ylabel('Sum of squared distances')
plt.plot(nr, dist_sample_sort)
plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
kn_x = kn.knee #knee x value
eps_nr_Neerijse = dist_sample_sort[kn_x] # knee y value = epsilon
#%%

# Clustering with DBSCAN
outputCSV_Huldenberg = Folder_cluster + 'Huldenberg_RF_veg_clustered_eps' + str(eps_nr_Neerijse) + '_minS2' + str(min_samples_nr_Neerijse) + 'data_' + dimension + '_' + str(timestamp) + '.csv'

model_DBSCAN = DBSCAN(eps=eps_nr_Neerijse, min_samples=min_samples_nr_Neerijse)
model_DBSCAN.fit_predict(globals()['data_' + dimension])
pred = model_DBSCAN.fit_predict(globals()['data_' + dimension])
print("number of cluster found: {}".format(len(set(model_DBSCAN.labels_))))
nr_clusters = len(set(model_DBSCAN.labels_))
print('cluster for each point: ', model_DBSCAN.labels_)
labels_DBSCAN = model_DBSCAN.labels_
df_H['cluster'] = pred
df_H = df_H[df_H.cluster != -1]
df_H.to_csv(outputCSV_Huldenberg,index=False)


#%% epsilon estimation Tervuren
data_xy = df_T.iloc[:,: 2]
data_xyz = df_T.iloc[:,: 3]

##Distance matrix
tree = KDTree(globals()['data_' + dimension], leaf_size)              
dist, ind = tree.query(globals()['data_' + dimension], k=min_samples_nr_Neerijse + 1)   # k closest neighbors (1st is point itself)
dist_sample = dist[:,min_samples_nr_Neerijse]

dist_sample_sort = np.sort(dist_sample)[::-1] #Sort from largest to smallest distance
n = len(dist_sample_sort) 
nr = x = np.arange(0, n)
plt.plot(nr, dist_sample_sort)

# Find knee to estmate epsilon value
from kneed import KneeLocator
kn = KneeLocator(nr, dist_sample_sort, curve='convex', direction='decreasing')
import matplotlib.pyplot as plt
plt.xlabel('number of clusters k')
plt.ylabel('Sum of squared distances')
plt.plot(nr, dist_sample_sort)
plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
kn_x = kn.knee #knee x value
eps_nr_Neerijse = dist_sample_sort[kn_x] # knee y value = epsilon


#%% Clustering with DBSCAN
outputCSV_Tervuren= Folder_cluster + 'Tervuren_RF_veg_clustered_eps' + str(eps_nr_Neerijse) + '_minS2' + str(min_samples_nr_Neerijse) + 'data_' + dimension + '_' + str(timestamp) + '.csv'
model_DBSCAN = DBSCAN(eps=eps_nr_Neerijse, min_samples=min_samples_nr_Neerijse)
model_DBSCAN.fit_predict(globals()['data_' + dimension])
pred = model_DBSCAN.fit_predict(globals()['data_' + dimension])
print("number of cluster found: {}".format(len(set(model_DBSCAN.labels_))))
nr_clusters = len(set(model_DBSCAN.labels_))
print('cluster for each point: ', model_DBSCAN.labels_)
labels_DBSCAN = model_DBSCAN.labels_
df_T['cluster'] = pred
df_T = df_T[df_T.cluster != -1]
df_T.to_csv(outputCSV_Tervuren,index=False)
#%% Add location name + give unique cluster number 
df_T_features = df_T
df_H_features = df_H 
df_V_features = df_V

df_T_features['Location'] = 'Tervuren'
df_H_features['Location'] = 'Huldenberg'
df_V_features['Location'] = 'Velm'

df_T_features['cluster_unique'] = df_T_features['cluster']
df_H_features['cluster_unique'] = df_H_features['cluster'] + (max(df_T_features['cluster']) + 1)
df_V_features['cluster_unique'] = df_V_features['cluster'] + (max(df_H_features['cluster_unique']) + 1)

frames = [df_T_features, df_H_features, df_V_features]
df_features = pd.concat(frames)
df_features.to_csv(Folder_cluster + 'df_features_clusterNr.csv', index = False)


#%% CREATE CONCAVE HULLS + SAVE AS SHAPEFILE
nr_i = max(df_features['cluster_unique']) + 1
column_names = ["location", "object_type", "id", "min_z", "max_z", "len_counts_dens", "min_slope_rel", "max_slope_rel"]
data_density_loop = pd.DataFrame(columns= column_names)

for i in range (0, nr_i):
    points = df_features.loc[df_features['cluster_unique'] == i]
    filename_raw = "Cluster_" + str(i) + "_" + str(timestamp)
    #create concave hull
    points_coordinates = points[['x', 'y']] #select xy columns from dataframe
    points_list = points_coordinates.values.tolist() #convert xy columns df to list
    append_data=[] #append data to empty list
    for a in points_list:
        p = geometry.Point(a)
        append_data.append(p) #for a in series --> list_points = geometry.Point(list)
    concave_hull, edge_points = alpha_shape_function.alpha_shape(append_data, alpha=0.1)
    #store concave hull as shapefile with OGR
    path = conc_hull_folder #path to store shapefile
    os.chdir(path) #change workdir to output path
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource('hull' + filename_raw + '.shp')
    layer = ds.CreateLayer('', None, ogr.wkbPolygon)
   
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger)) #add id-field to attribute table
    defn = layer.GetLayerDefn()

    ## If there are multiple geometries, put the "for" loop here

    # Create a new feature (attribute and geometry)
    id_nr = filename_raw.split('_')[1]
    feat = ogr.Feature(defn)
    feat.SetField('id', id_nr)

    # Make a geometry, from Shapely object
    geom = ogr.CreateGeometryFromWkb(concave_hull.wkb)
    feat.SetGeometry(geom)

    layer.CreateFeature(feat)
    feat = geom = None  # destroy these

    # Save and close everything
    ds = layer = feat = geom = None

# Merge concave hull shapefiles 
folder = Path(conc_hull_folder)
shapefiles = folder.glob("*.shp")
gdf = pd.concat([
    gpd.read_file(shp)
    for shp in shapefiles
]).pipe(gpd.GeoDataFrame)
gdf.crs = {'init' :'epsg:31370'}
gdf.to_file(folder / 'compiled_hull_RFall_merge_noFruit.shp')
gdf['dissolve_field'] = 1 
gdf_dissolve = gdf.dissolve(by='dissolve_field')
gdf_dissolve.to_file(folder / 'compiled_hull_RFall_merge_dissolve_noFruit.shp')
