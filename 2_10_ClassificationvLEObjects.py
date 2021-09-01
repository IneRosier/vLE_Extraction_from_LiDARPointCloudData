# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 19:41:56 2021

@author: u0117123
"""
#Import modules

import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

#Input variables
Validation_Area="Tervuren"
#Referece objects with features path
refObjectPath = r'C:\Users\u0117123\Box Sync\FWO\WP1\Point-cloud-extractions\processing\5_Clustering_classification\Reference'

ClusteredObjectPath = r'C:\Users\u0117123\Box Sync\FWO\WP1\Point-cloud-extractions\processing\5_Clustering_classification\3SA\r10\RF_all'

#%% LOGISTIC REGRESSION MODEL

### STEP 1 ### IMPORT DATA
data_density_loop_all = pd.read_csv(refObjectPath + "\data_density_loop_Reference.csv", sep=";", index_col=(0))
data_density_loop = data_density_loop_all.loc[data_density_loop_all['location'] != Validation_Area]
data_density_loop['height7_1'] = data_density_loop['height7']/data_density_loop['height1']
data_density_loop['height7_2'] = data_density_loop['height7']/data_density_loop['height2']
data_density_loop['height5_1'] = data_density_loop['height5']/data_density_loop['height1']
data_density_loop['height10_2'] = data_density_loop['height10']/data_density_loop['height2']
data_density_loop['height10_1'] = data_density_loop['height10']/data_density_loop['height1']

columns_x = ["min_z", "max_z", "min_slope_rel", "max_slope_rel", "area", 
           "m_z_chm","m_nr_returns", "3D_dens","height7_1", "height5_1", 
           "height10_2", "height10_1", "height7_2"]
data_density_loop_x = data_density_loop[columns_x] #independent variables
data_density_loop_ground_p_density =  data_density_loop[["ground_p_density"]]
data_density_loop_y = data_density_loop[["Type"]] #Response variable

#Convert response variable to binary values (shrub = 1; tree = 0)
shrub = ["shrub"]
data_density_loop_y["y"] = np.where(data_density_loop_y["Type"].isin(shrub), "1", "0")
data_density_loop_y = data_density_loop_y.drop(['Type'], axis=1)

# convert dataframe response variable to matrix
conv_arr = data_density_loop_y.values
y_array = conv_arr.ravel()



#%%## STEP 2 ### Check for correlations
import matplotlib.pyplot as plt
import seaborn as sns

# Create correlation matrix & selecting upper triangle
cor_matrix = data_density_loop_x.corr().abs()

plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(data_density_loop_x.corr().abs(),annot = True)
plt.show()

upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
#print(upper_tri)

# Droping the column with correlation > 95%
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)] #height5_1, height10_1
#print(); print(to_drop)

data_density_loop_x_dropCorr = data_density_loop_x.drop(to_drop, axis=1)
#print(); print(data_density_loop_x_dropCorr.head())



#%%## STEP 3 ### Cross validation loop

#merge independent variables and dependent variable
data_density_loop_xy_dropCorr = pd.concat([data_density_loop_x_dropCorr,data_density_loop_y], axis=1)
data_density_loop_xy_dropCorr = data_density_loop_xy_dropCorr.reset_index(drop=True)

#split in 10 parts
data_density_loop_xy_dropCorr_shuffled = data_density_loop_xy_dropCorr.sample(frac=1, random_state=1) #shuffle dataframe
data_density_loop_xy_dropCorr_shuffled_List = np.array_split(data_density_loop_xy_dropCorr_shuffled, 10)

#Empty dataframes
rfe_features_append = []
sp_features_append = []
accuracy_append = []

#for loop cross validation
for x in range(10):
    trainList = []
    for y in range(10):
        if y == x :
            testdf = data_density_loop_xy_dropCorr_shuffled_List[y]
        else:
            trainList.append(data_density_loop_xy_dropCorr_shuffled_List[y])
    traindf = pd.concat(trainList)
    
    #independent variables and response variable
    X_train = traindf.drop(columns=['y'])
    y_train = traindf['y']
    X_test = testdf.drop(columns=['y'])
    y_test = testdf['y']
    
 
    ### STEP 3.1 ### Create scaler
    from sklearn import preprocessing
    import numpy as np
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_train_scaled = pd.DataFrame(data = X_train_scaled, columns=X_train.columns)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(data = X_test_scaled, columns=X_test.columns)
    
    
    ### STEP 3.2 ### Feature selection
    ### Step 3.2.1 Recursive Feature Elimination
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()
    rfe = RFE(logreg, n_features_to_select = 5)   # running RFE with 5 variables as output
    rfe = rfe.fit(X_train_scaled, y_train)
    #create training and testing dataframe with selected features
    col_rfe = X_train_scaled.columns[rfe.support_]
    X_train_scaled_rfe = X_train_scaled[col_rfe]
    X_test_scaled_rfe = X_test_scaled[col_rfe]
    #create dataframe with selected features per fold
    rfe_features_columns = ["fold", "features"]
    rfe_features = pd.DataFrame(columns = rfe_features_columns)
    rfe_features["features"] = X_train_scaled_rfe.columns
    rfe_features["fold"] = x
    rfe_features_append.append(rfe_features)
       
    ### STEP 3.2.2 Select Percentile (ANOVA F-value, retain features with 50% highest score)
    from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif
    sp = SelectPercentile(f_classif, percentile=70).fit(X_train_scaled, y_train)
    index_selfeat = (sp.get_support(indices=True)).tolist()
    X_train_scaled_sp = X_train_scaled.iloc[:,index_selfeat] 
    X_test_scaled_sp = X_test_scaled.iloc[:,index_selfeat]
    #create dataframe with selected features per fold
    sp_features_columns = ["fold", "features"]
    sp_features = pd.DataFrame(columns = sp_features_columns)
    sp_features["features"] = X_train_scaled_sp.columns
    sp_features["fold"] = x
    sp_features_append.append(sp_features)
    
    
    ### STEP 4 ### Build models using all or selected features
    ### STEP 4.1 Full model
    logreg_Full = LogisticRegression(random_state=0).fit(X_train_scaled, y_train)
    # print('Logistic Regression score for training set: %f' % logreg_Full.score(X_train_scaled, y_train))
    y_pred_full = logreg_Full.predict(X_test_scaled)
    score_full = logreg_Full.score(X_test_scaled, y_test) # Use score method to get accuracy of model

    
    ### STEP 4.2 Recursive Feature Elimination
    logreg_RFE = LogisticRegression(random_state=0).fit(X_train_scaled_rfe, y_train)
    # print('Logistic Regression score for training set: %f' % logreg_RFE.score(X_train_scaled_rfe, y_train))
    y_pred_rfe = logreg_RFE.predict(X_test_scaled_rfe) 
    score_rfe = logreg_RFE.score(X_test_scaled_rfe, y_test) # Use score method to get accuracy of model

    
    ### STEP 4.3 Select Percentile
    logreg_SP = LogisticRegression(random_state=0).fit(X_train_scaled_sp, y_train)
    # print('Logistic Regression score for training set: %f' % logreg_SP.score(X_train_scaled_sp, y_train))
    y_pred_sp = logreg_SP.predict(X_test_scaled_sp) 
    score_sp = logreg_SP.score(X_test_scaled_sp, y_test) # Use score method to get accuracy of model

    #create dataframe with scores per fold
    accuracy_columns = ["fold", "accuracy_full", "accuracy_rfe", "accuracy_sp"]
    accuracy = pd.DataFrame(columns = accuracy_columns)
    new_row = {'accuracy_full':score_full, 'accuracy_rfe':score_rfe, 'accuracy_sp':score_sp, 'fold':x}
    accuracy = accuracy.append(new_row, ignore_index=True)
    accuracy_append.append(accuracy)

rfe_features_append = pd.concat(rfe_features_append)
sp_features_append = pd.concat(sp_features_append)
accuracy_append = pd.concat(accuracy_append)

#calculate mean performance score
print(accuracy_append['accuracy_full'].mean()) #0.862087912087912
print(accuracy_append['accuracy_rfe'].mean()) #0.8543956043956044
print(accuracy_append['accuracy_sp'].mean()) #0.8543956043956044


#%%## STEP 5 ### Fit model on all Reference data
from sklearn import preprocessing
import numpy as np

#Features
rfe_unique = rfe_features_append["features"].unique()
sp_unique = sp_features_append["features"].unique()

#Scale independent variables
scaler = preprocessing.StandardScaler().fit(data_density_loop_x_dropCorr)
X_scaled = scaler.transform(data_density_loop_x_dropCorr)
X_scaled = pd.DataFrame(data = X_scaled, columns=X_train.columns)

### STEP 5.1 full model
logreg_Full = LogisticRegression(random_state=0).fit(X_scaled, y_array)
print('Logistic Regression score for training set: %f' % logreg_Full.score(X_scaled, y_array)) #0.891304


### STEP 5.2 rfe
X_scaled_rfe = X_scaled[rfe_unique] #select columns
logreg_rfe= LogisticRegression(random_state=0).fit(X_scaled_rfe, y_array)
print('Logistic Regression score for training set: %f' % logreg_rfe.score(X_scaled_rfe, y_array)) #0.898551

### STEP 5.3 sp
X_scaled_sp = X_scaled[sp_unique] #select columns
logreg_sp= LogisticRegression(random_state=0).fit(X_scaled_sp, y_array)
print('Logistic Regression score for training set: %f' % logreg_sp.score(X_scaled_sp, y_array)) #0.905797

#%%## STEP 6 ### Apply model on RF clustered points/Segmented opbjects

## STEP 6.1 - Import data
data_density_loop_cluster = pd.read_csv(ClusteredObjectPath + '\data_density_loop_clustered_noFruit.csv', sep=";", index_col=(0))

data_density_loop_cluster['height7_1'] = data_density_loop_cluster['height7']/data_density_loop_cluster['height1']
data_density_loop_cluster['height7_2'] = data_density_loop_cluster['height7']/data_density_loop_cluster['height2']
data_density_loop_cluster['height5_1'] = data_density_loop_cluster['height5']/data_density_loop_cluster['height1']
data_density_loop_cluster['height10_2'] = data_density_loop_cluster['height10']/data_density_loop_cluster['height2']
data_density_loop_cluster['height10_1'] = data_density_loop_cluster['height10']/data_density_loop_cluster['height1']
### STEP 1 ### IMPORT DATA
columns_x = ["min_z", "max_z", "min_slope_rel", "max_slope_rel", "area", 
           "m_z_chm","m_nr_returns", "3D_dens", 
           "height7_1", "height5_1", "height10_2", "height10_1", "height7_2"]
data_density_loop_cluster_x = data_density_loop_cluster[columns_x]
data_density_loop_ground_p_density = data_density_loop_cluster[["ground_p_density"]]

#%%## STEP 6.2 - Check for correlations
import matplotlib.pyplot as plt
import seaborn as sns
# drop correlated features (>95% correlation)
X_clustered_df_dropCorr = data_density_loop_cluster_x.drop(to_drop, axis=1)

#%%## STEP 6.3 - apply scaler
from sklearn import preprocessing
import numpy as np
X_clustered_scaled = scaler.transform(X_clustered_df_dropCorr)
X_clustered_scaled = pd.DataFrame(data=X_clustered_scaled, columns=X_clustered_df_dropCorr.columns)
#### STEP 6.4 - apply model
#Full model
y_pred_full = logreg_Full.predict(X_clustered_scaled)

#rfe model
X_clustered_scaled_rfe = X_clustered_scaled[rfe_unique] #select columns
y_pred_rfe = logreg_rfe.predict(X_clustered_scaled_rfe)

#sp model
X_clustered_scaled_sp = X_clustered_scaled[sp_unique] #select columns
y_pred_sp = logreg_sp.predict(X_clustered_scaled_sp)

#%%
y_pred_full_df = pd.DataFrame(y_pred_full, columns=['y_pred_full'])
y_pred_RFE_df = pd.DataFrame(y_pred_rfe, columns=['y_pred_rfe'])
y_pred_SP_df = pd.DataFrame(y_pred_sp, columns=['y_pred_sp'])



data_density_loop_clusters_prediction = pd.concat([data_density_loop_cluster.reset_index(drop=True),
                                                   y_pred_full_df.reset_index(drop=True),
                                                   y_pred_RFE_df.reset_index(drop=True),
                                                   y_pred_SP_df.reset_index(drop=True)], axis=1)
data_density_loop_clusters_prediction.to_csv(ClusteredObjectPath + '\data_density_loop_clusters_cv_Full_RFE_SP_prediction_Validate_' + Validation_Area + '.csv', sep=';', header=True)