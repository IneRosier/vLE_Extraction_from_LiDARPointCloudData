# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 18:28:15 2021

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

# Import function to create concave hulls
work_dir = r'C:\Users\u0117123\Box Sync\FWO\WP1\Point-cloud-extractions\scripts\preliminary'
os.chdir(work_dir)
import alpha_shape_function as alpha_shape_function

#%%
#Input variables
#Output path clipped ground las files
output_path = r'C:\Users\u0117123\Box Sync\FWO\WP1\Point-cloud-extractions\processing\5_Clustering_classification\Reference\groundpoints_hull'

#location to store reference objects with calculated features
refObjectPath = r'C:\Users\u0117123\Box Sync\FWO\WP1\Point-cloud-extractions\processing\5_Clustering_classification\Reference'

#Cloudcompare files folder
CCFolder = "C:\\Users\\u0117123\\Box Sync\\FWO\\WP1\\Point-cloud-extractions\\processing\\5_Clustering_classification\\Reference\\CloudComparePoints\\"


#points with unique userdata 
df_features_path = r'C:\Users\u0117123\Box Sync\FWO\WP1\Point-cloud-extractions\processing\5_Clustering_classification\Clustered_RefPoints\df_features_UserDataUnique.csv'

percentage = 0.10
#%%
### Timestamp 
t = time.localtime()
timestamp = time.strftime('%Y%m%d_%H%M', t)

#%% IMPORT LIDAR GROUND POINTS
#Read clipped las ground points
#Dataframe to store ground points
column_names = ['x', 'y', 'z', 'Intensity', 'Classification', 'UserData', 'geometry']
Ground_points_df = pd.DataFrame(columns= column_names)

for filename in os.listdir(output_path):
    if filename.endswith(".las"):
        file = output_path + '/' + filename
        inFile = File(file, mode='r')
        #Import LAS into numpy array (X=raw integer value x=scaled float value)
        lidar_points = np.array((inFile.x,inFile.y,inFile.z,inFile.intensity,
                                 inFile.raw_classification,inFile.scan_angle_rank)).transpose()
        #Transform to pandas DataFrame
        lidar_df=DataFrame(lidar_points)
        #Column names pandas Dataframe
        lidar_df.columns = ['x', 'y', 'z', 'Intensity', 'Classification', 'UserData' ]
        #Transform to geopandas GeoDataFrame
        crs = None
        geometry_points = [Point(xyz) for xyz in zip(inFile.x,inFile.y,inFile.z)]
        lidar_geodf = GeoDataFrame(lidar_df, crs=crs, geometry=geometry_points)
        lidar_geodf.crs = {'init' :'epsg:31370'} # set correct spatial reference
        Ground_points_df = pd.concat([Ground_points_df, lidar_geodf], axis = 0)

ground_points = MultiPoint(list(zip(Ground_points_df ['x'], Ground_points_df ['y']))) #geodataframe to shapely multipoint object

#%% FEATURE CALCULATION PER CONCAVE HULL

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
        
        
        #Calculate density distribution features
        points_low_elim = points[points['z'] > 0.2] 
        points_sorted = points_low_elim.sort_values(by=['z'])
        n=len(points_sorted)
        perc = round(percentage * n) #mark percentage lowest points
        points_sorted['perc']=0
        points_sorted.iloc[:perc, points_sorted.columns.get_loc('perc')] = 1
        filename_raw = filename.split('_merge')[0] #create filename for title plot
        filename_raw = filename_raw.split('_var')[0] #create filename for title plot
        filename_title = filename_raw + "_break" +str(percentage)
        sns.scatterplot(x="x", y="z", hue="perc", data=points_sorted, s=2, legend=False)
        plot = plt.title(filename_title)
        #plot=savefig(PlotFolder + filename_raw + "_" + str(percentage) + ".png" )
        plt.clf()
        
        #calculate area
        area=concave_hull.area
        
        points_coordinates = points_low_elim[['x', 'y']] #select xy columns from dataframe
        points_list = points_coordinates.values.tolist() #convert xy columns df to list
        append_data=[] #append data to empty list
        for i in points_list:
            p = geometry.Point(i)
            append_data.append(p) #for i in series --> list_points = geometry.Point(list)
        
        #Calculate ground density
        Intersect = concave_hull.intersection(ground_points) #intersect ground points with concave hull
        ground_p_density = len(Intersect)/area #ground point density = number of points/area hull
        
        
        #statitistics
        #mean
        list_features = list(points.columns.values)
        list_features = list_features[:-1]
        df_mean = points[list_features].mean()
        df_mean = df_mean.add_prefix('m_')
        #stdev
        df_std = points[list_features].std()
        df_std = df_std.add_prefix('std_')
        #variance
        df_var = points[list_features].var()
        df_var = df_var.add_prefix('var_')
        #range
        df_min = points[list_features].min()
        df_max = points[list_features].max()
        df_range = df_max - df_min
        df_min = df_min.add_prefix('min_')
        df_max = df_max.add_prefix('max_')
        df_range = df_range.add_prefix('r_')
            
        #create density plots
        mymax = points_low_elim["z"].max()
        mymax_round = (10*(mymax-0.2)) + (5-(10*(mymax-0.2))%5) if ((mymax-0.2)*10)%5!=0 else ((mymax-0.2)*10) 
        my_range = range(2, math.floor((mymax_round+7)), 5)
        my_range_lst = list(my_range)
        my_range_lst_flt = [float(i) for i in my_range_lst]
        my_range = (pd.DataFrame(my_range_lst_flt))/10
        my_range.columns = ['range']
        my_range_list = my_range['range'].to_list()
        my_x = my_range.iloc[1:]
        my_x = my_x.reset_index(drop=True)
        counts = points_low_elim.groupby(pd.cut(points_low_elim.z, my_range_list)).count()
        counts = counts.y
        counts_df = counts.to_frame().reset_index()
        counts_df['x'] = my_x
        counts_dens = counts/area
        counts_dens = counts_dens.to_frame().reset_index()
        counts_df['dens'] = counts_dens.y 
        plt.scatter(counts_df.x, counts_df.dens, s=4)
        plt.title(filename_title + "_density")
        plt.xlabel('height (m)')
        plt.ylabel('points per 0.5 m³')
        #plt.savefig(PlotFolder + filename_raw + "_" + 'density' + ".png")
        plt.clf()
        
        y_dens = counts_df.dens
        y_dens_list = list(y_dens)
        max_list = y_dens_list.index(max(y_dens_list))
        len_list = len(y_dens_list)
        rel_dens = max_list/len_list
        #print(filename_raw)
        #print(rel_dens)
        
        #cumuative density plot
        Total = counts_df['y'].sum()
        counts_df['y_rel'] = counts_df.y/Total
        counts_df['y_rel_cum_sum'] = counts_df['y_rel'].cumsum()
        plt.plot(counts_df.x, counts_df.y_rel_cum_sum)
        plt.title(filename_title + "_cum_density")
        plt.xlabel('height (m)')
        plt.ylabel('cumulative relative density')
        #plt.savefig(PlotFolder + filename_raw + "_" + 'cum_density' + ".png")
        plt.clf()
        x = counts_df.x
        y = counts_df.y_rel_cum_sum

        try: 
            steepest = np.argmax(np.diff(np.array(y)))
            length = len(np.diff(np.array(y)))
            max_slope_rel = steepest/length
            flattest = np.argmin(np.diff(np.array(y)))
            length = len(np.diff(np.array(y)))
            min_slope_rel = flattest/length
        except ValueError:
            print(filename + "pass")
            
        result_index1 = counts_df['y_rel_cum_sum'].sub(0.1).abs().idxmin()
        height1 = (counts_df["z"].iloc[result_index1]).mid
        result_index2 = counts_df['y_rel_cum_sum'].sub(0.2).abs().idxmin()
        height2 = (counts_df["z"].iloc[result_index2]).mid
        result_index3 = counts_df['y_rel_cum_sum'].sub(0.3).abs().idxmin()
        height3 = (counts_df["z"].iloc[result_index3]).mid
        result_index4 = counts_df['y_rel_cum_sum'].sub(0.4).abs().idxmin()
        height4 = (counts_df["z"].iloc[result_index4]).mid
        result_index5 = counts_df['y_rel_cum_sum'].sub(0.5).abs().idxmin()
        height5 = (counts_df["z"].iloc[result_index5]).mid
        result_index6 = counts_df['y_rel_cum_sum'].sub(0.6).abs().idxmin()
        height6 = (counts_df["z"].iloc[result_index6]).mid
        result_index7 = counts_df['y_rel_cum_sum'].sub(0.7).abs().idxmin()
        height7 = (counts_df["z"].iloc[result_index7]).mid
        result_index8 = counts_df['y_rel_cum_sum'].sub(0.8).abs().idxmin()
        height8 = (counts_df["z"].iloc[result_index8]).mid
        result_index9 = counts_df['y_rel_cum_sum'].sub(0.9).abs().idxmin()
        height9 = (counts_df["z"].iloc[result_index9]).mid
        result_index10 = counts_df['y_rel_cum_sum'].sub(1).abs().idxmin()
        height10 = (counts_df["z"].iloc[result_index10]).mid
        
        
        print(filename_raw)
        if len(points_low_elim) > 0:
            print("min z value points: " + str(min(points_low_elim.z)))
            print("max z value points: " + str(max(points_low_elim.z)))
        print("lenght points: " + str(len(points_low_elim)))
        print("length counts_dens: " + str(len(counts_dens)))
        print("min_slope_rel: " + str(min_slope_rel))
        column_names = ["location", "object_type", "id","min_z", "max_z", 
                        "len_counts_dens", "min_slope_rel", "max_slope_rel", 
                        "area", "m_z_chm", "m_nr_returns", "3D_dens", 
                        "ground_p_density", "height1", "height2","height3",
                        "height4","height5","height6","height7","height8",
                        "height9","height10"]
        
        data_density_new = pd.DataFrame(columns= column_names)
        
        data_density_new = {'location': filename.split('_')[0],
                            'object_type': filename.split('_')[1],
                            'id': filename.split('_')[2],
                            'min_z': min(points_low_elim.z), 
                            'max_z': max(points_low_elim.z),
                            'len_counts_dens': len(counts_dens), 
                            'min_slope_rel': min_slope_rel, 
                            'max_slope_rel': max_slope_rel,
                            'area': area,
                            'm_z_chm': df_mean.m_z,
                            'm_nr_returns': df_mean.m_NrOfEchos,
                            '3D_dens': (len(points)/area)/df_range.r_z,
                            'ground_p_density': ground_p_density,
                            'height1': height1,
                            'height2': height2,
                            'height3': height3,
                            'height4': height4,
                            'height5': height5,
                            'height6': height6,
                            'height7': height7,
                            'height8': height8,
                            'height9': height9,
                            'height10': height10}
        data_density_new = pd.DataFrame (data_density_new, columns = column_names, index=[0])
        
        data_density_loop = pd.concat([data_density_loop, data_density_new], axis = 0)
        shrub_related = ["Bush", "Hedgerow","WoodyEdge"]
        tree_related = ["Tree","TreeLine","TreeGroup"]
        data_density_loop["Type"] = np.where(data_density_loop["object_type"].isin(shrub_related), "shrub", "tree")
data_density_loop.to_csv(refObjectPath + '\data_density_loop_Reference.csv', sep=';', header=True)


#%%
data_density_loop['height7_1'] = data_density_loop['height7']/data_density_loop['height1']
data_density_loop['height7_2'] = data_density_loop['height7']/data_density_loop['height2']
data_density_loop['height5_1'] = data_density_loop['height5']/data_density_loop['height1']
data_density_loop['height10_2'] = data_density_loop['height10']/data_density_loop['height2']
data_density_loop['height10_1'] = data_density_loop['height10']/data_density_loop['height1']
