# vLE Extraction from LiDAR Point CloudData

Scripts and reference Data for the publication: 'A workflow to extract the geometry and type of vegetated landscape elements from airborne LiDAR point clouds (submitted)'.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
SCRIPTS

2_6_LiDARFeatureCalculation.py:
Change directories in this script;
Run this script to calculate features described in section 2.6 of the publication.

2_7_1_ClassificationOfPointData_DataPreparation.py:
Change directories in this script;
Run this script for data preparation needed to do the random forest classification described in section 2.7 of the publication.

2_7_2_ClassificationOfPointData.py:
Change directories in this script;
Run this script for to perform the Random Forest classificqation described in section 2.7 of the publication.

2_7_3_ClassificationOfPointData_Postprocessing.py:
Change directories in this script;
Run this script for the postprocessing step of the random forest classification described in section 2.7 of the publication.

2_8_vLEObjectSegmentation.py:
Change directories in this script;
Run this script to cluster and segment the classified LiDAR points as described in section 2.8 of the publication.

2_9_1vLEObjectsFeatureExtraction_Part1.py:
Change directories in this script;
Run this script to calculate the predictor variables of the modelled vLE objects as described in section 2.9 of the publication.

2_9_2vLEObjectsFeatureExtraction_Part2.py:
Change directories in this script;
Run this script to calculate the predictor variables of the modelled vLE objects as described in section 2.9 of the publication (continued).

2_9_3_vLEReferenceObjectSegmentation.py:
Change directories in this script;
Run this script to cluster and segment the Reference LiDAR points.

2_9_4vLERerferenceObjectsFeatureExtraction_Part1.py:
Change directories in this script;
Run this script to calculate the predictor variables of the reference vLE objects.

2_9_5vLERerferenceObjectsFeatureExtraction_Part2.py:
Change directories in this script;
Run this script to calculate the predictor variables of the reference vLE objects (continued).

2_10_ClassificationvLEObjects.py:
Change directories in this script;
Run this script to classify the modelled vLE objects using a logistic regression model as described in section 2.10 of the publication.

alpha_shape_function.py:
Script used in the scripts '2_8_vLEObjectSegmentation.py', '2_9_1vLEObjectsFeatureExtraction_Part1.py', '2_9_3_vLEReferenceObjectSegmentation.py', and '2_9_4vLERerferenceObjectsFeatureExtraction_Part1.py' to segment point cloud data. Script does not need to be run but needes to be saved in the same folder as the four scripts

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Refernce Data

ReferencePointClouds:
LiDAR point clouds of the reference vLE objects used in the publication

StudyAreaBoundaries:
boundaries of the three study areas used in the publication
