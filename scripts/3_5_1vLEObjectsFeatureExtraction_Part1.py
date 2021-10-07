# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:46:55 2021

@author: IneR
"""
##############################################################################
################################# IMPORTANT NOTE #############################
# Run after this script the two bat files
##############################################################################

#%%
#Import modules
import os

#%%
#Input variables

#Output path clipped ground las files
output_path = r'C:\Users\u0117123\Box Sync\FWO\WP1\Point-cloud-extractions\processing\5_Clustering_classification\3SA\r10\RF_all\groundpoints_hull'

#lasTools
lasclip = r'C:\Users\u0117123\Documents\Programmas\LAStools\LAStools\bin\lasclip.exe'
directory = r'C:\Users\u0117123\Documents\Programmas\LAStools\LAStools\bin'

#Folder where concave hulls are stored
conc_hull_folder = r'C:\Users\u0117123\Box Sync\FWO\WP1\Point-cloud-extractions\processing\5_Clustering_classification\3SA\r10\RF_all\hulls\noFruit'
#name concave hull merged & dissolved
conc_hull = '\compiled_hull_RFall_merge_dissolve_noFruit.shp'

#Input ground las points
Folder_Velm = "D:\\Data\\Processed data\\LAS\\Velm_Selection\\Velm_Selection_class2\\" #Folder with ground points Velm
Folder_Neerijse = "D:\\Data\\Processed data\\LAS\\Huldenberg_32_Selection\\Huldenberg_32_Selection_class2\\" #Folder with ground points Neerijse 


#%% Create bat files to clip las ground points
#input path polygons
polygon_path = conc_hull_folder + conc_hull

#Input las paths

input_path_Neerijse = Folder_Neerijse + "*.las"
input_path_Velm = Folder_Velm + "*.las"


#create bat files
with open(os.path.join(directory, 'batch_temp_clustered_Neerijse.bat'), 'w') as OPATH:
      OPATH.writelines(["lasclip ",
                        "-i {0} ".format('"' + input_path_Neerijse + '"'),
                        "{0} ".format("-poly"),
                        "{0} ".format('"' + polygon_path + '"'),
                        "{0} ".format("-odir"),
                        "{0} ".format('"' + output_path + '"'),
                        "{0} ".format("-olas"),
                        "\n "])

with open(os.path.join(directory, 'batch_temp_clustered_Velm.bat'), 'w') as OPATH:
      OPATH.writelines(["lasclip ",
                        "-i {0} ".format('"' + input_path_Velm + '"'),
                        "{0} ".format("-poly"),
                        "{0} ".format('"' + polygon_path + '"'),
                        "{0} ".format("-odir"),
                        "{0} ".format('"' + output_path + '"'),
                        "{0} ".format("-olas"),
                        "\n "])
