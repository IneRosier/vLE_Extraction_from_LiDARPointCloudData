# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:59:56 2021

@author: IneR
"""
import opals
import numpy as np
import pandas as pd

import os

#from opals import *  # @NoMove @UnusedWildImport
from opals.Import import Import  # @Reimport
from opals.AddInfo import AddInfo  # @Reimport
from opals.PointStats import PointStats  # @Reimport
from opals.Normals import Normals  # @Reimport
from opals.Export import Export  # @Reimport
imp = Import()
add = AddInfo()
ps = PointStats()
nor = Normals()
exp = Export()

#%% Input small test set parameters
#Input parameters

CSVFolder = "I:\\Neerijse\\VegFeatures\\ReferenceData\\changedUserData\\csv\\"
ascFolder = "I:\\Neerijse\\VegFeatures\\ReferenceData\\changedUserData\\csv\\withExtra2\\"
#var_searchRadius = [1,2,5,10,20]
var_searchRadius = [1] # run script for value 1, 2, 5, 10 and 20
#%%input rasters
import rasterio
EuclDistLbgebrperc = rasterio.open("I:\\Neerijse\\EuclDist_lbgebrperc.tif")
EuclDistWbn = rasterio.open("I:\\Neerijse\\EuclDist_Wbn.tif")
TN_UserData = rasterio.open("I:\\Neerijse\\Tervuren_Huldenberg_TN_UserData2.tif")
DTM = rasterio.open("I:\\Neerijse\\dtm\\GeoTIFF\\DHMVIIDTMRAS1m_k32.tif")

#%%
################# PART 1 ######################################################
#loop throug csv files

#calculate Euclidean distance to agricultural parcel boundary, 
##euclidean distance to road and normalized Z value

for filename in os.listdir(CSVFolder):
    if filename.endswith(".csv"):
        print (filename)
        csv_path = CSVFolder + filename
        parameters = pd.read_csv(csv_path)
        parameters=parameters.drop(['ScanDirectionFlag','EdgeOfFlightLine','ScanAngleRank','PointSourceId','GpsTime','ScanChannel','ClassFlags'], axis=1)
        parameters.columns = ['x','y','z','Intensity','EchoNumber', 'NrOfEchos','Classification','UserData', 'Red', 'Green', 'Blue']
        coords = [(x,y) for x, y in zip(parameters.x, parameters.y)]
        print("Calculate euclidean distance to Lbgebrprc..")
        parameters['EuclDistLbgebrperc'] = [x[0] for x in EuclDistLbgebrperc.sample(coords)]
        print("Calculate euclidean distance to Wbn..")
        parameters['EuclDistWbn'] = [x[0] for x in EuclDistWbn.sample(coords)]
        print("extract dtm value..")
        parameters['DTM'] = [x[0] for x in DTM.sample(coords)]
        parameters['ZASL'] = parameters['z'] - parameters['DTM']
        print("extract TN User data..")
        parameters['TN_UserData'] = [x[0] for x in TN_UserData.sample(coords)]
        classes = [1,4,5]
        mask=(parameters['Classification'].isin(classes))
        parameters_valid = parameters[mask]
        parameters.loc[mask,'UserData'] = parameters_valid['UserData'] + parameters_valid['TN_UserData']
        parameters = parameters.drop(['TN_UserData'], axis=1)
        # Output data
        print("creating output data..")
        csv_path_root = os.path.splitext(CSVFolder + "withExtra2\\" + filename)[0]
        out_filename = '{}_params_veg_euclDist.asc'.format(csv_path_root)
        parameters.to_csv(out_filename, sep='\t', index=False,header=False)       
        


#%%
################# PART 2 ######################################################
for filename in os.listdir(ascFolder):
    if filename.endswith(".asc"):
        filename_raw = os.path.splitext(filename)[0]
        filename_raw = filename_raw.split('_')[0]
        print(filename_raw)
        odmFile = CSVFolder + filename_raw + "var.odm"
        outputFile = CSVFolder + filename_raw + "_20m_" + "var_exp.dat"
        imp.reset()
        imp.inFile = ascFolder + filename
        imp.iFormat = "I:/xmlFiles/Import_asc - Copy.xml"
        imp.outFile = odmFile
        imp.run()
        
        add.inFile = odmFile
        add.attribute = '_norm_returns = (EchoNumber) / (NrOfEchos)'
        add.run()
        add.reset()
        
        for i in var_searchRadius:
            print (i)
            print("neighbors search + calculation eigenvalues")
            nor.reset()
            nor.inFile = odmFile 
            nor.storeMetaInfo = opals.Types.NormalsMetaInfo.medium
            nor.searchRadius = i
            nor.neighbours = 500
            nor.run()
            add.inFile = odmFile
            add.attribute = '_roughness' + str(int(i*100)) + '(float) = NormalSigma0'
            add.run()
            add.reset()
            
            # add eigenvalues
            #-----------------------------------------------------------------
            print("add eigenvalues")
            add.inFile =  odmFile
            add.attribute = '_NormalE1' + str(int(i*100)) + '(float) = NormalEigenvalue1'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_NormalE2' + str(int(i*100)) + '(float)= NormalEigenvalue2'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_NormalE3' + str(int(i*100)) + '(float) = NormalEigenvalue3'
            add.run()
            add.reset()
            
            # Calculate and add curvature, linearity, planarity and sphericity
            #-----------------------------------------------------------------
            print("Calculate and add curvature, linearity, planarity and sphericity")
            add.inFile =  odmFile
            add.attribute = '_changeofcurvature' + str(int(i*100)) + '(float) = NormalEigenvalue3 / (NormalEigenvalue1 + NormalEigenvalue2 + NormalEigenvalue3)'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_linearity' + str(int(i*100)) + '(float)= (NormalEigenvalue1 - NormalEigenvalue2) / NormalEigenvalue1'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_planarity' + str(int(i*100)) + '(float) = (NormalEigenvalue2 - NormalEigenvalue3) / NormalEigenvalue1'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_scattering' + str(int(i*100)) + '(float) = NormalEigenvalue3 / NormalEigenvalue1'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_omnivariance' + str(int(i*100)) + '(float) = pow((NormalEigenvalue1 * NormalEigenvalue2 * NormalEigenvalue3), (1./3))'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_anisotropy' + str(int(i*100)) + '(float) = (NormalEigenvalue1 - NormalEigenvalue3) / NormalEigenvalue1'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_eigentropy' + str(int(i*100)) + '(float) = - ((NormalEigenvalue1 * log(NormalEigenvalue1)) + (NormalEigenvalue2 * log(NormalEigenvalue2)) + (NormalEigenvalue3 * log(NormalEigenvalue3)))'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_sumeigen' + str(int(i*100)) + '(float) = NormalEigenvalue1 + NormalEigenvalue2 + NormalEigenvalue3'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_NormalZ' + str(int(i*100)) + '(float) = NormalZ'
            add.run()
            add.reset()
            
            # Calculate and add height range, height standard devatiation, point density, point count, height min, height max, height variance,
            # and height mean
            #-----------------------------------------------------------
            print ("Calculate and add height range, height standard devatiation, point density, point count, height min, height max, height variance and height mean")
            ps.inFile =  odmFile
            ps.feature =  'range', 'stdDev', 'pdens', 'pcount', 'min', 'max', 'variance', 'mean'
            ps.attribute = 'z'
            ps.refModel = "zeroPlane"
            ps.searchRadius = i
            ps.run()
            ps.reset()
            add.inFile =  odmFile
            add.attribute = '_ZASLRange' + str(int(i*100)) + '(float)= _zRange'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_ZASLStdDev' + str(int(i*100)) + '(float)= _zStdDev'
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_ZASLPdens' + str(int(i*100)) + '(float)= _zPdens'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_ZASLPcount' + str(int(i*100)) + '(float)= _zPcount'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_ZASLMin' + str(int(i*100)) + '(float)= _zMin'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_ZASLMax' + str(int(i*100)) + '(float)= _zMax'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_ZASLVariance' + str(int(i*100)) + '(float)= _zVariance'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_ZASLMean' + str(int(i*100)) + '(float)= _zMean'
            add.run()
            add.reset()
            
            #Calculate color features
            #-----------------------------------------------------------
            print("Calculate color features")
            ps.inFile =  odmFile
            ps.feature = 'mean', 'variance'
            ps.attribute = 'Red'
            ps.refModel = "zeroPlane"
            ps.searchRadius = i
            ps.run()
            ps.reset()
            add.inFile =  odmFile
            add.attribute = '_RedMean' + str(int(i*100)) + '(float)= _RedMean'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_RedVariance' + str(int(i*100)) + '(float)= _RedVariance'
            add.run()
            add.reset()            
            ps.inFile =  odmFile
            ps.feature = 'mean', 'variance'
            ps.attribute = 'Green'
            ps.refModel = "zeroPlane"
            ps.searchRadius = i
            ps.run()
            ps.reset()
            add.inFile =  odmFile
            add.attribute = '_GreenMean' + str(int(i*100)) + '(float)= _GreenMean'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_GreenVariance' + str(int(i*100)) + '(float)= _GreenVariance'
            add.run()
            add.reset()  
            ps.inFile =  odmFile
            ps.feature = 'mean', 'variance'
            ps.attribute = 'Blue'
            ps.refModel = "zeroPlane"
            ps.searchRadius = i
            ps.run()
            ps.reset()
            add.inFile =  odmFile
            add.attribute = '_BlueMean' + str(int(i*100)) + '(float)= _BlueMean'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_BlueVariance' + str(int(i*100)) + '(float)= _BlueVariance'
            add.run()
            add.reset()  
            
            #Calculate intensity features
            #-----------------------------------------------------------
            print("Calculate Intensity features")
            ps.inFile =  odmFile
            ps.feature = 'mean', 'variance'
            ps.attribute = 'Intensity'
            ps.refModel = "zeroPlane"
            ps.searchRadius = i
            ps.run()
            ps.reset()
            add.inFile =  odmFile
            add.attribute = '_IntensityMean' + str(int(i*100)) + '(float)= _IntensityMean'
            add.run()
            add.reset()
            add.inFile =  odmFile
            add.attribute = '_IntensityVariance' + str(int(i*100)) + '(float)= _IntensityVariance'
            add.run()
            add.reset()
        
        exp.inFile = odmFile
        exp.oFormat = 'I:/xmlFiles/Export_geom_var3_20m.xml'
        exp.outFile = outputFile
        exp.run()




