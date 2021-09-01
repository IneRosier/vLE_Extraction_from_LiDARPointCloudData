# -*- coding: utf-8 -*-
"""
Created on Mon May 17 12:10:55 2021

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

#%% Input variables
#RF classified veg object points
Folder = "C:\\Users\\u0117123\\Box Sync\\FWO\\WP1\\Point-cloud-extractions\\processing\\3_RF_classification\\3SA\\OutputRF\\AllFeatures\\"

Points_file_Velm = "y_PredictedAsVeg_allF_TestVelm_fixedDist_veg_nonveg_moreFeatures_r10.csv"

#%% Load Points Velm
df_V= pd.read_csv(
    Folder + Points_file_Velm , delimiter=",") # import csv file as panda df
points_CSV_V = GeoDataFrame(
    df_V.drop(['x', 'y', 'z', 'Classification','UserData', 'y_ref', 'y_pred_full'], axis=1),
    crs={'init': 'epsg:31370'},
    geometry=[Point(xy) for xy in zip(df_V.x, df_V.y)]) #to geopanda dataframe


#%%
#Load Polygon Velm without fruit trees
path_Velm_noFruit = r'C:\Users\u0117123\Box Sync\FWO\WP1\Point-cloud-extractions\processing\Velm\VegFeatures'
Filename_Velm_noFruit = "\Velm_noFruitTrees.shp"
polygons_noFruit = gpd.read_file(path_Velm_noFruit + Filename_Velm_noFruit)
polygons_noFruit.crs = {"init":"epsg:31370"}


#Load Polygon Velm with fruit trees
path_Velm_Fruit = r'C:\Users\u0117123\Box Sync\FWO\WP1\Point-cloud-extractions\processing\Velm\VegFeatures'
Filename_Velm_Fruit = "\Velm_FruitTrees.shp"
polygons_Fruit = gpd.read_file(path_Velm_Fruit + Filename_Velm_Fruit)
polygons_Fruit.crs = {"init":"epsg:31370"}

#%% #clip fruit tree points Velm
#points_clip = gpd.clip(points_CSV_V, polygons)
Intersect = polygons_noFruit.intersection(points_CSV_V)
Difference1 = polygons_Fruit.difference(points_CSV_V)
Difference2 = points_CSV_V.difference(polygons_Fruit)