# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:06:59 2021

@author: IneR
"""

import pandas as pd
# Set input
Study_area = "Huldenberg"

#folder containing parameter csv files 
csv_Folder_Huldenberg ="I:\\Neerijse\\VegFeatures\\ReferenceData\\changedUserData\\csv\\Huldenberg\\"
csv_Folder_Tervuren ="I:\\Neerijse\\VegFeatures\\ReferenceData\\changedUserData\\csv\\Tervuren\\"
csv_Folder_Velm = "I:\\Las\\Velm\\VegFeatures\\ChangedUserData\\Class1_perSquare_CSV\\"

#%%

#column names merged per square dataframes
colNames = ["x", "y", "z", "Intensity", "EchoNumber", "NrOfEchos", "Classification", "UserData", "Red", "Green", "Blue", "EuclDistLbgebrperc", 
             "EuclDistWbn", "z_CHM", "norm_returns", "roughness1", "NormalE1_1", "NormalE2_1", "NormalE3_1", "changeofcurvature1", "linearity1", 
             "planarity1", "scattering1", "omnivariance1", "anisotropy1", "eigentropy1", "sumeigen1", "NormalZ1", "z_CHMRange1", "z_CHMStdDev1", 
             "z_CHMPdens1", "z_CHMPcount1", "z_CHMMin1", "z_CHMMax1", "z_CHMVariance1", "z_CHMMean1", "RedMean1", "RedVariance1", "GreenMean1", 
             "GreenVariance1", "BlueMean1", "BlueVariance1", "IntensityMean1", "IntensityVariance1", "roughness2", "NormalE1_2", "NormalE2_2", 
             "NormalE3_2", "changeofcurvature2", "linearity2", "planarity2", "scattering2", "omnivariance2", "anisotropy2", "eigentropy2", 
             "sumeigen2", "NormalZ2", "z_CHMRange2", "z_CHMStdDev2", "z_CHMPdens2", "z_CHMPcount2", "z_CHMMin2", "z_CHMMax2", "z_CHMVariance2", 
             "z_CHMMean2", "RedMean2", "RedVariance2", "GreenMean2", "GreenVariance2", "BlueMean2", "BlueVariance2", "IntensityMean2", 
             "IntensityVariance2", "roughness5", "NormalE1_5", "NormalE2_5", "NormalE3_5", "changeofcurvature5", "linearity5", "planarity5", 
             "scattering5", "omnivariance5", "anisotropy5", "eigentropy5", "sumeigen5", "NormalZ5", "z_CHMRange5", "z_CHMStdDev5", "z_CHMPdens5", 
             "z_CHMPcount5", "z_CHMMin5", "z_CHMMax5", "z_CHMVariance5", "z_CHMMean5", "RedMean5", "RedVariance5", "GreenMean5", "GreenVariance5",
             "BlueMean5", "BlueVariance5", "IntensityMean5", "IntensityVariance5", "roughness10", "NormalE1_10", "NormalE2_10", "NormalE3_10", 
             "changeofcurvature10", "linearity10", "planarity10", "scattering10", "omnivariance10", "anisotropy10", "eigentropy10", "sumeigen10", 
             "NormalZ10", "z_CHMRange10", "z_CHMStdDev10", "z_CHMPdens10", "z_CHMPcount10", "z_CHMMin10", "z_CHMMax10", "z_CHMVariance10", 
             "z_CHMMean10", "RedMean10", "RedVariance10", "GreenMean10", "GreenVariance10", "BlueMean10", "BlueVariance10", "IntensityMean10", 
             "IntensityVariance10","roughness20", "NormalE1_20", "NormalE2_20", "NormalE3_20", "changeofcurvature20", "linearity20", 
             "planarity20", "scattering20", "omnivariance20", "anisotropy20", "eigentropy20", "sumeigen20", "NormalZ20", "z_CHMRange20", 
             "z_CHMStdDev20", "z_CHMPdens20", "z_CHMPcount20", "z_CHMMin20", "z_CHMMax20", "z_CHMVariance20", "z_CHMMean20", "RedMean20", 
             "RedVariance20", "GreenMean20", "GreenVariance20", "BlueMean20", "BlueVariance20", "IntensityMean20", "IntensityVariance20"]

#loop over files to find square numbers
list_squares = []
import glob, os
os.chdir(globals()["csv_Folder_{}".format(Study_area)])
for file in glob.glob("*.dat"):
    fileS=file.split("square",1)[1]
    Square = fileS.split("_",2)[0]
    list_squares.append(Square)
# min_square = int(min(list_squares))
# max_square = int(max(list_squares))
list_squares_unique = []

for x in list_squares:
    if x not in list_squares_unique:
        list_squares_unique.append(x)
for i in range(0, len(list_squares_unique)): 
    list_squares_unique[i] = int(list_squares_unique[i])

#per square file : remove excess columns, 
f = {} #dictionary with dataframes for all squares
cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
#for i in range(min_square,max_square):
for i in list_squares_unique:
    e={} #dictionary per square
    for file in glob.glob("*.dat"):
        fileS=file.split("square",1)[1]
        Square = fileS.split("_",2)[0]
        dist = fileS.split("_",2)[1]
        if dist == "1m":
            dist_nr = 1
        elif dist == "2m":
            dist_nr = 2
        elif dist == "5m":
            dist_nr = 3
        elif dist == "10m":
            dist_nr = 4
        elif dist == "20m":
            dist_nr = 5
        
        Square_dist = Square + "_" + str(dist_nr)
        if i == int(Square):
            print(Square)
            d = {}
            d["csv{0}".format(Square_dist)] = file
            e["parameter{0}".format(Square_dist)] = pd.read_csv(globals()["csv_Folder_{}".format(Study_area)] + d["csv{0}".format(Square_dist)], header=None, sep='\t')
        
            if dist_nr != 1:
                e["parameter{0}".format(Square_dist)] = e["parameter{0}".format(Square_dist)].drop(e["parameter{0}".format(Square_dist)].columns[cols],axis=1)
            import collections
            oe = collections.OrderedDict(sorted(e.items())) #dictionary per square, ordered by distance
            f["parameter{0}".format(Square)] = pd.concat(oe, axis=1)
#    
tp = {}
tn = {}
tpm = {}
tnm = {}


# 10 = tree group; 11 = tree line; 12 = tree; 13 = hedgerow; 14 = woody edge; 15 = bush; 16 = forest patch
# 1 = vegetation but not a feature  
for i in list_squares_unique:
    f["parameter{0}".format(i)].dropna()            
    f["parameter{0}".format(i)].columns = colNames
    tp["parameter{0}".format(i)] = f["parameter{0}".format(i)][f["parameter{0}".format(i)]['Classification'].isin([10, 11, 12, 13, 14, 15])]
    tn["parameter{0}".format(i)] = f["parameter{0}".format(i)][f["parameter{0}".format(i)]['Classification'].isin([1])]
    
tpm = pd.concat(tp, axis=0)
tnm = pd.concat(tn, axis=0)   

tnm['veg'] = 1
tpm['veg'] = 2
#%% 
print("save TP")
tpm.to_csv("I:\\Las\\InputRF\\ReferenceData\\FixedDistance\\TP_" + str (Study_area) + "_Features_fixedDist_MoreFeatures_r20.csv")
print("save TN")
tnm.to_csv("I:\\Las\\InputRF\\ReferenceData\\FixedDistance\\TN_" + str (Study_area) + "_Features_fixedDist_MoreFeatures_r20.csv")


