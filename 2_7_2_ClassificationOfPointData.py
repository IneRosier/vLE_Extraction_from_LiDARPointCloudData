# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:52:26 2021

@author: IneR
"""
#set inputs
#Folder (Adapt!!) 
Folder = 'I:\\Las\\InputRF\\ReferenceData\\FixedDistance\\'
#%%
#import modules
import os
import glob
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection as ms

Test = "Velm"

#%%
########################## Import variables ################################### 
###############################################################################
###############################################################################

#import True positive
TruePos_H = pd.read_csv(Folder + 'TP_Huldenberg_Features_fixedDist_MoreFeatures_r20.csv', index_col=0)
TruePos_T = pd.read_csv(Folder + 'TP_Tervuren_Features_fixedDist_MoreFeatures_r20.csv', index_col=0)
TruePos_V = pd.read_csv(Folder + 'TP_Velm_Features_fixedDist_MoreFeatures_r20.csv', index_col=0)
#import True negative
TrueNeg_H = pd.read_csv(Folder + 'TN_Huldenberg_Features_fixedDist_MoreFeatures_r20.csv', index_col=0)
TrueNeg_T = pd.read_csv(Folder + 'TN_Tervuren_Features_fixedDist_MoreFeatures_r20.csv', index_col=0)
TrueNeg_V = pd.read_csv(Folder + 'TN_Velm_Features_fixedDist_MoreFeatures_r20.csv', index_col=0)

#

dataRF_H = pd.concat([TruePos_H, TrueNeg_H])
dataRF_V = pd.concat([TruePos_V, TrueNeg_V])
dataRF_T = pd.concat([TruePos_T, TrueNeg_T])

#Get feature names --> no feature columns: 0, 1, 2, 3, 5, 7, 8, 161 --> drop them
cols = [0, 1, 2, 3, 5, 7, 8, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 
        142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 
        156, 157, 158, 159, 160, 161]

features_df = dataRF_H.drop(dataRF_H.columns[cols],axis=1)
features = features_df.columns[:125] #column names feat

#dataRF: 1,2,3 = x,y,z     4,6,9:160 = features    response = 161
cols2 = [0, 5, 7, 8, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 
         144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 
         158, 159, 160, 161] #all columns but xyz and features


if Test == "Huldenberg":
    Train1 = "Tervuren"
    Train2 = "Velm"
    dataRF_train = pd.concat([TruePos_T, TrueNeg_T, TruePos_V, TrueNeg_V])
    dataRF_train1 = dataRF_T
    dataRF_train2 = dataRF_V
    dataRF_test = dataRF_H
    
    X_xyz_values_train = dataRF_train.drop(dataRF_train.columns[cols2], axis=1) #only keep xyz and features
    X_xyz_values_train1 = dataRF_T.drop(dataRF_T.columns[cols2], axis=1) #only keep xyz and features
    X_xyz_values_train2 = dataRF_V.drop(dataRF_V.columns[cols2], axis=1) #only keep xyz and features
    X_xyz_values_test = dataRF_H.drop(dataRF_H.columns[cols2], axis=1) #only keep xyz and features
    
    X_test = X_xyz_values_test.iloc[ : , 3:157]
    X_train1 = X_xyz_values_train1.iloc[ : , 3:157]
    X_train2 = X_xyz_values_train2.iloc[ : , 3:157]
    
    y_test = dataRF_H.iloc[:,161].values
    y_train1 = dataRF_T.iloc[:,161].values
    y_train2 = dataRF_V.iloc[:,161].values


if Test == "Tervuren":
    Train1 = "Huldenberg"
    Train2 = "Velm"
    dataRF_train = pd.concat([TruePos_H, TrueNeg_H, TruePos_V, TrueNeg_V])
    dataRF_train1 = dataRF_H
    dataRF_train2 = dataRF_V
    dataRF_test = dataRF_T
    
    X_xyz_values_train = dataRF_train.drop(dataRF_train.columns[cols2], axis=1) #only keep xyz and features
    X_xyz_values_train1 = dataRF_H.drop(dataRF_H.columns[cols2], axis=1) #only keep xyz and features
    X_xyz_values_train2 = dataRF_V.drop(dataRF_V.columns[cols2], axis=1) #only keep xyz and features
    X_xyz_values_test = dataRF_T.drop(dataRF_T.columns[cols2], axis=1) #only keep xyz and features

    X_test = X_xyz_values_test.iloc[ : , 3:157]
    X_train1 = X_xyz_values_train1.iloc[ : , 3:157]
    X_train2 = X_xyz_values_train2.iloc[ : , 3:157]
    
    y_test = dataRF_T.iloc[:,161].values
    y_train1 = dataRF_H.iloc[:,161].values
    y_train2 = dataRF_V.iloc[:,161].values    
    
   
if Test =="Velm":
    Train1 = "Huldenberg"
    Train2 = "Tervuren"
    dataRF_train = pd.concat([TruePos_T, TrueNeg_T, TruePos_H, TrueNeg_H])
    dataRF_train1 = dataRF_H
    dataRF_train2 = dataRF_T
    dataRF_test = dataRF_V
    
    X_xyz_values_train = dataRF_train.drop(dataRF_train.columns[cols2], axis=1) #only keep xyz and features
    X_xyz_values_train1 = dataRF_H.drop(dataRF_H.columns[cols2], axis=1) #only keep xyz and features
    X_xyz_values_train2 = dataRF_T.drop(dataRF_T.columns[cols2], axis=1) #only keep xyz and features    
    X_xyz_values_test = dataRF_V.drop(dataRF_V.columns[cols2], axis=1) #only keep xyz and features
    
    X_test = X_xyz_values_test.iloc[ : , 3:157]
    X_train1 = X_xyz_values_train1.iloc[ : , 3:157]
    X_train2 = X_xyz_values_train2.iloc[ : , 3:157]
    
    y_test = dataRF_V.iloc[:,161].values     
    y_train2 = dataRF_T.iloc[:,161].values
    y_train1 = dataRF_H.iloc[:,161].values

    
X_train = X_xyz_values_train.iloc[ : , 3:157]
y_train = dataRF_train.iloc[:,161].values
xyz_train = X_xyz_values_train.iloc[ : , 0:3]
xyz_test = X_xyz_values_test.iloc[ : , 0:3]
#
################################## Random Forest ############################## 
###############################################################################
###############################################################################

#
##############################################################################
##############################################################################
#### Random forest, reduced model: train on velm and neerijsehuldenberg ######
##############################################################################
##############################################################################
##############################################################################
print("RF Reduced model, test= " + Test )
Classifier_reduced = RandomForestClassifier(n_estimators=100, 
                                            random_state=0, 
                                            class_weight= "balanced") 

##############################################################################
############################feature selection#################################
##############################################################################
sel = SelectFromModel(Classifier_reduced) 
sel.fit(X_train, y_train)

#show feature importances
sel.get_support()

#count selected features
selected_features = features_df.columns[(sel.get_support())]
print("Selected feature count: " + str(len(selected_features)))
index = [features.get_loc(str(i)) for i in selected_features]
#names of selected features
print(selected_features)

############################################################################
########################## RF with selected features #########################
##############################################################################
X_train_select = X_train.iloc[:,index]
X_test_select = X_test.iloc[:,index]
X_train1_select = X_train1.iloc[:,index]
X_train2_select = X_train2.iloc[:,index]

cv=ms.StratifiedKFold(n_splits=10, shuffle=True)
#
scores_reduced = cross_val_score(Classifier_reduced, 
                                 X_train_select, 
                                 y_train, cv=cv)
print("Accuracy reduced model: %0.2f (+/- %0.2f)" % (scores_reduced.mean(), 
                                                     scores_reduced.std() * 2))

#build model
forest_reduced = Classifier_reduced.fit(X_train_select, y_train)
#% Plot and save feature importances

importances_reduced = forest_reduced.feature_importances_

#
#predict reduced model
y_pred_select_test = Classifier_reduced.predict(X_test_select)
#
# accuracy of predictions
print("accuracy " + Test)
print(confusion_matrix(y_test,y_pred_select_test))
print(classification_report(y_test,y_pred_select_test))
print(accuracy_score(y_test, y_pred_select_test))

#
#predict reduced model
y_pred_select_train1 = Classifier_reduced.predict(X_train1_select)
y_pred_select_train2 = Classifier_reduced.predict(X_train2_select)

# accuracy of predictions
print("accuracy " + Train1)
print(confusion_matrix(y_train1,y_pred_select_train1))
print(classification_report(y_train1,y_pred_select_train1))
print(accuracy_score(y_train1, y_pred_select_train1))

print("accuracy " + Train2)
print(confusion_matrix(y_train2,y_pred_select_train2))
print(classification_report(y_train2,y_pred_select_train2))
print(accuracy_score(y_train2, y_pred_select_train2))

#Save CSVs
#Test
Folder_test = "I:\\Las\\OutputRF\\SelectedFeatures\\test_" + Test
import os
if not os.path.exists(Folder_test):
    os.makedirs(Folder_test)
    
y_test_ref_pred = pd.DataFrame({'y_ref': y_test, 'y_pred_red': y_pred_select_test})
xyz_features_test = dataRF_test.iloc[:,1:9]
zasl_test = dataRF_test.iloc[:,14]
csv_test = pd.concat([xyz_features_test.reset_index(drop=True), zasl_test.reset_index(drop=True), y_test_ref_pred.reset_index(drop=True)], axis=1)
csv_test.to_csv(Folder_test + "\\y_Pred_selectF_Test" + Test + "_fixedDist_veg_nonveg_moreFeatures_r10.csv", index = False)
csv_test_classified_veg = csv_test.loc[csv_test['y_pred_red'] == 2]
csv_test_classified_veg.to_csv(Folder_test + "\\y_PredictedAsVeg_selectF_Test" + Test + "_fixedDist_veg_nonveg_moreFeatures_r10.csv", index = False)

#Train1 
y_train1_ref_pred = pd.DataFrame({'y_ref': y_train1, 'y_pred_red': y_pred_select_train1})
xyz_features_train1 = dataRF_train1.iloc[:,1:9]
zasl_train1 = dataRF_train1.iloc[:,14]
csv_train1 = pd.concat([xyz_features_train1.reset_index(drop=True), zasl_train1.reset_index(drop=True), y_train1_ref_pred.reset_index(drop=True)], axis=1)
csv_train1.to_csv(Folder_test + "\\y_Pred_selectF_Train1" + Train1 + "_fixedDist_veg_nonveg_moreFeatures_r10.csv", index = False)
csv_train1_classified_veg = csv_train1.loc[csv_train1['y_pred_red'] == 2]
csv_train1_classified_veg.to_csv(Folder_test + "\\y_PredictedAsVeg_selectF_Train1" + Train1 + "_fixedDist_veg_nonveg_moreFeatures_r10.csv", index = False)


#Train2
y_train2_ref_pred = pd.DataFrame({'y_ref': y_train2, 'y_pred_red': y_pred_select_train2})
xyz_features_train2 = dataRF_train2.iloc[:,1:9]
zasl_train2 = dataRF_train2.iloc[:,14]
csv_train2 = pd.concat([xyz_features_train2.reset_index(drop=True), zasl_train2.reset_index(drop=True), y_train2_ref_pred.reset_index(drop=True)], axis=1)
csv_train2.to_csv(Folder_test + "\\y_Pred_selectF_Train2" + Train2 + "_fixedDist_veg_nonveg_moreFeatures_r10.csv", index = False)
csv_train2_classified_veg = csv_train2.loc[csv_train2['y_pred_red'] == 2]
csv_train2_classified_veg.to_csv(Folder_test + "\\y_PredictedAsVeg_selectF_Train2" + Train2 + "_fixedDist_veg_nonveg_moreFeatures_r10.csv", index = False)


#
############################################################################
############################ RF with all features ############################
##############################################################################
print("RF Full model, test= " + Test )
Classifier_full = RandomForestClassifier(n_estimators=100, 
                                         random_state=0, 
                                         class_weight= "balanced")
cv=ms.StratifiedKFold(n_splits=10, shuffle=True)
scores_full = cross_val_score(Classifier_full, 
                              X_train, 
                              y_train, 
                              cv=cv)
print("Accuracy full model: %0.2f (+/- %0.2f)" % (scores_full.mean(),
                                                  scores_full.std() * 2))

#build model
forest_full = Classifier_full.fit(X_train, y_train)
#% Plot and save feature importances

#importances_transition = forest_transition.feature_importances_
importances_full = forest_full.feature_importances_


#
#predict full model
y_pred_test = Classifier_full.predict(X_test)

# accuracy of predictions
print(confusion_matrix(y_test,y_pred_test))
print(classification_report(y_test,y_pred_test))
print(accuracy_score(y_test, y_pred_test))

#
#predict Full model
y_pred_train1 = Classifier_full.predict(X_train1)
y_pred_train2 = Classifier_full.predict(X_train2)

# accuracy of predictions
print("accuracy " + Train1)
print(confusion_matrix(y_train1,y_pred_train1))
print(classification_report(y_train1,y_pred_train1))
print(accuracy_score(y_train1, y_pred_train1))

print("accuracy " + Train2)
print(confusion_matrix(y_train2,y_pred_train2))
print(classification_report(y_train2,y_pred_train2))
print(accuracy_score(y_train2, y_pred_train2))

#Save CSVs
#Test
Folder_test = "I:\\Las\\OutputRF\\AllFeatures\\test_" + Test
import os
if not os.path.exists(Folder_test):
    os.makedirs(Folder_test)
#

y_test_ref_pred = pd.DataFrame({'y_ref': y_test, 'y_pred_full': y_pred_test})
xyz_features_test = dataRF_test.iloc[:,1:9]
zasl_test = dataRF_test.iloc[:,14]
csv_test = pd.concat([xyz_features_test.reset_index(drop=True), zasl_test.reset_index(drop=True), y_test_ref_pred.reset_index(drop=True)], axis=1)
csv_test.to_csv(Folder_test + "\\y_Pred_allF_Test" + Test + "_fixedDist_veg_nonveg_moreFeatures_r10.csv", index = False)
csv_test_classified_veg = csv_test.loc[csv_test['y_pred_full'] == 2]
csv_test_classified_veg.to_csv(Folder_test + "\\y_PredictedAsVeg_allF_Test" + Test + "_fixedDist_veg_nonveg_moreFeatures_r10.csv", index = False)

#Train1 
y_train1_ref_pred = pd.DataFrame({'y_ref': y_train1, 'y_pred_full': y_pred_train1})
xyz_features_train1 = dataRF_train1.iloc[:,1:9]
zasl_train1 = dataRF_train1.iloc[:,14]
csv_train1 = pd.concat([xyz_features_train1.reset_index(drop=True), zasl_train1.reset_index(drop=True), y_train1_ref_pred.reset_index(drop=True)], axis=1)
csv_train1.to_csv(Folder_test + "\\y_Pred_allF_Train1" + Train1 + "_fixedDist_veg_nonveg_moreFeatures_r10.csv", index = False)
csv_train1_classified_veg = csv_train1.loc[csv_train1['y_pred_full'] == 2]
csv_train1_classified_veg.to_csv(Folder_test + "\\y_PredictedAsVeg_allF_Train1" + Train1 + "_fixedDist_veg_nonveg_moreFeatures_r10.csv", index = False)

#Train2
y_train2_ref_pred = pd.DataFrame({'y_ref': y_train2, 'y_pred_full': y_pred_train2})
xyz_features_train2 = dataRF_train2.iloc[:,1:9]
zasl_train2 = dataRF_train2.iloc[:,14]
csv_train2 = pd.concat([xyz_features_train2.reset_index(drop=True), zasl_train2.reset_index(drop=True), y_train2_ref_pred.reset_index(drop=True)], axis=1)
csv_train2.to_csv(Folder_test + "\\y_Pred_allF_Train2" + Train2 + "_fixedDist_veg_nonveg_moreFeatures_r10.csv", index = False)
csv_train2_classified_veg = csv_train2.loc[csv_train2['y_pred_full'] == 2]
csv_train2_classified_veg.to_csv(Folder_test + "\\y_PredictedAsVeg_allF_Train2" + Train2 + "_fixedDist_veg_nonveg_moreFeatures_r10.csv", index = False)
#%%
#save reference data
csv_train1_ref_veg = csv_train1.loc[csv_train1['y_ref'] == 2]
csv_test_ref_veg = csv_test.loc[csv_test['y_ref'] == 2]
csv_train2_ref_veg = csv_train2.loc[csv_train2['y_ref'] == 2]
csv_train1_ref_veg.to_csv("I:\\Las\\OutputRF\\Reference\\y_reference_" + Train1 + "_features.csv", index = False)
csv_test_ref_veg.to_csv("I:\\Las\\OutputRF\\Reference\\y_reference_" + Test + "_features.csv", index = False)
csv_train2_ref_veg.to_csv("I:\\Las\\OutputRF\\Reference\\y_reference_" + Train2 + "_features.csv", index = False)


