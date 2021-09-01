# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:04:20 2021

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
#folder to store concave hulls
conc_hull_folder = "C:\\Users\\u0117123\\Box Sync\\FWO\\WP1\\Point-cloud-extractions\\processing\\5_Clustering_classification\\Reference\\hulls\\"

#Cloudcompare files folder
CCFolder = "C:\\Users\\u0117123\\Box Sync\\FWO\\WP1\\Point-cloud-extractions\\processing\\5_Clustering_classification\\Reference\\CloudComparePoints\\"

#%%## Timestamp 
t = time.localtime()
timestamp = time.strftime('%Y%m%d_%H%M', t)


#%%
# Import function to create concave hulls
work_dir = r'C:\Users\u0117123\Box Sync\FWO\WP1\Point-cloud-extractions\scripts\preliminary'
os.chdir(work_dir)

import alpha_shape_function as alpha_shape_function

#%% CREATE CONCAVE HULLS + SAVE AS SHAPEFILE
#Empty dataframe
column_names = ["location", "object_type", "id", "min_z", "max_z", "len_counts_dens", "min_slope_rel", "max_slope_rel"]
data_density_loop = pd.DataFrame(columns= column_names)

for filename in os.listdir(CCFolder):
    if filename.endswith(".dat"):
        points = pd.read_csv(CCFolder + filename, header=None, sep = '\t')
        points['file_name'] = filename
        colNames = ["x", "y", "_z_orig", "Intensity", "EchoNumber", "NrOfEchos", "Classification", 
                    "_UserData", "Red", "Green", "Blue", "_EuclDistLbgebrperc", "_EuclDistWbn", "z", 
                    "_norm_returns", "_roughness100000", "_NormalE1100000", "_NormalE2100000", 
                    "_NormalE3100000", "_changeofcurvature100000", "_linearity100000", "_planarity100000", 
                    "_scattering100000", "_omnivariance100000", "_anisotropy100000", "_eigentropy100000", 
                    "_sumeigen100000", "_NormalZ100000", "_ZASLRange100000", "_ZASLStdDev100000", 
                    "_ZASLPdens100000", "_ZASLPcount100000", "_ZASLMin100000", "_ZASLMax100000", 
                    "_ZASLVariance100000", "_ZASLMean100000", "_RedMean100000", "_RedVariance100000", 
                    "_GreenMean100000", "_GreenVariance100000", "_BlueMean100000", "_BlueVariance100000", 
                    "_IntensityMean100000", "_IntensityVariance100000", "filename"]
        points.columns = colNames
        filename_raw = filename.split('_merge')[0] #create filename for title plot
        filename_raw = filename_raw.split('_var')[0] #create filename for title plot
                
        #create concave hull
        points_coordinates = points[['x', 'y']] #select xy columns from dataframe
        points_list = points_coordinates.values.tolist() #convert xy columns df to list
        append_data=[] #append data to empty list
        for i in points_list:
            p = geometry.Point(i)
            append_data.append(p) #for i in series --> list_points = geometry.Point(list)
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
        id_nr = filename.split('_')[2]
        feat = ogr.Feature(defn)
        feat.SetField('id', id_nr)
        
                # Make a geometry, from Shapely object
        geom = ogr.CreateGeometryFromWkb(concave_hull.wkb)
        feat.SetGeometry(geom)

        layer.CreateFeature(feat)
        feat = geom = None  # destroy these

        # Save and close everything
        ds = layer = feat = geom = None
#%%
# Merge concave hull shapefiles 
folder = Path(conc_hull_folder)
shapefiles = folder.glob("*.shp")
gdf = pd.concat([
    gpd.read_file(shp)
    for shp in shapefiles
]).pipe(gpd.GeoDataFrame)
gdf.crs = {'init' :'epsg:31370'}
gdf.to_file(folder / 'compiled_hull_Reference_merge.shp')
gdf['dissolve_field'] = 1 
gdf_dissolve = gdf.dissolve(by='dissolve_field')
gdf_dissolve.to_file(folder / 'compiled_hull_Reference_merge_dissolve.shp')
