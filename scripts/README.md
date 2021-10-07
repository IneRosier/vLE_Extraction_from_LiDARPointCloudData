This folder contains the script for the publication 'A workflow to extract the geometry and type of vegetated 2 landscape elements from airborne LiDAR point clouds'

3_2_LiDARFeatureCalculation.py: Change directories in this script; Run this script to calculate features described in section 3.2 of the publication.

3_3_1_ClassificationOfPointData_DataPreparation.py: Change directories in this script; Run this script for data preparation needed to do the random forest classification described in section 3.3 of the publication.

3_3_2_ClassificationOfPointData.py: Change directories in this script; Run this script for to perform the Random Forest classificqation described in section 3.3 of the publication.

3_3_3_ClassificationOfPointData_Postprocessing.py: Change directories in this script; Run this script for the postprocessing step of the random forest classification described in section 3.3 of the publication.

3_4_vLEObjectSegmentation.py: Change directories in this script; Run this script to cluster and segment the classified LiDAR points as described in section 3.4 of the publication.

3_5_1vLEObjectsFeatureExtraction_Part1.py: Change directories in this script; Run this script to calculate the predictor variables of the modelled vLE objects as described in section 3.5 of the publication.

3_5_2vLEObjectsFeatureExtraction_Part2.py: Change directories in this script; Run this script to calculate the predictor variables of the modelled vLE objects as described in section 3.5 of the publication (continued).

3_5_3_vLEReferenceObjectSegmentation.py: Change directories in this script; Run this script to cluster and segment the Reference LiDAR points as described in section 3.5 of the publication.

3_5_4vLERerferenceObjectsFeatureExtraction_Part1.py: Change directories in this script; Run this script to calculate the predictor variables of the reference vLE objects as described in section 3.5 of the publication.

3_5_5vLERerferenceObjectsFeatureExtraction_Part2.py: Change directories in this script; Run this script to calculate the predictor variables of the reference vLE objects as described in section 3.5 of the publication (continued).

3_6_ClassificationvLEObjects.py: Change directories in this script; Run this script to classify the modelled vLE objects using a logistic regression model as described in section 3.6 of the publication.

alpha_shape_function.py: Script used in the scripts '3_4_vLEObjectSegmentation.py', '3_5_1vLEObjectsFeatureExtraction_Part1.py', '3_5_3_vLEReferenceObjectSegmentation.py', and '3_5_4vLERerferenceObjectsFeatureExtraction_Part1.py' to segment point cloud data. Script does not need to be run but needs to be saved in the same folder as the four scripts.
