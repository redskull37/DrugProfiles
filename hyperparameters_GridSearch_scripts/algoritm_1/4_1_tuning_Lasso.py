import pandas as pd
import numpy as np
import time
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import scipy.optimize as opt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import gc

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import MinMaxScaler
import os

import scipy as sp
np.random.seed(123)

# _FOLDER = "/home/acq18mk/master/results/results/" 
_FOLDER = "results/"

### Coding Part

def LeaveOneOutError(model, X, y, metrics = "mse"):
    errors = []
    splitter_loo = LeaveOneOut()
    
    for train_index, test_index in splitter_loo.split(X):
        X_train_loo, X_test_loo = X[train_index, :], X[test_index,:]
        y_train_loo, y_test_loo = y[train_index], y[test_index]
        
        model = model.fit(X_train_loo, y_train_loo)
        if metrics == "mse":
            mse = mean_squared_error(y_test_loo, model.predict(X_test_loo))
            errors.append(mse)
        elif metrics == "mae":
            mae = mean_absolute_error(y_test_loo, model.predict(X_test_loo))
            errors.append(mae)
    
    return (sum(errors)/ len(errors)) 

with open(_FOLDER + "drug_ids_50.txt", 'r') as f:
    drug_ids_50 = [np.int32(line.rstrip('\n')) for line in f]
    
#columns to normalise:
with open(_FOLDER+"columns_to_normalise.txt", 'r') as f:
    columns_to_normalise = [line.rstrip('\n') for line in f]
# *****************************************

with open(_FOLDER+"X_features_cancer_cell_lines.txt", 'r') as f:
    X_cancer_cell_lines = [line.rstrip('\n') for line in f]
# *****************************************

with open(_FOLDER+"X_PubChem_properties.txt", 'r') as f:
    X_PubChem_properties = [line.rstrip('\n') for line in f]
# *****************************************

with open(_FOLDER+"X_features_Targets.txt", 'r') as f:
    X_targets = [line.rstrip('\n') for line in f]
# *****************************************

with open(_FOLDER+"X_features_Target_Pathway.txt", 'r') as f:
    X_target_pathway = [line.rstrip('\n') for line in f]
# *****************************************

all_columns = X_cancer_cell_lines + X_PubChem_properties + X_targets + X_target_pathway +["MAX_CONC"]

train_df = pd.read_csv(_FOLDER+"train08_merged_fitted_sigmoid4_123_with_drugs_properties_min10.csv").drop(["Unnamed: 0","Unnamed: 0.1"], axis=1)
test_df = pd.read_csv(_FOLDER+"test02_merged_fitted_sigmoid4_123_with_drugs_properties_min10.csv").drop(["Unnamed: 0","Unnamed: 0.1"], axis=1)               

train_df_50 = train_df.set_index("DRUG_ID").loc[drug_ids_50, :].copy()
test_df_50 = test_df.set_index("DRUG_ID").loc[drug_ids_50, :].copy()

datasets = ["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4"]

X_feat_dict = {"Dataset 1": X_cancer_cell_lines ,
               "Dataset 2": ["MAX_CONC"] + X_targets + X_target_pathway + X_cancer_cell_lines ,
               "Dataset 3": ["MAX_CONC"] + X_PubChem_properties +  X_cancer_cell_lines,
               "Dataset 4": ["MAX_CONC"] + X_PubChem_properties +  X_targets + X_target_pathway + X_cancer_cell_lines}

param_tested_alphas = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 300, 500, 1000]

param_grid = dict(alpha = param_tested_alphas)

splitter_loo = LeaveOneOut()
grid = GridSearchCV(Lasso(), param_grid=param_grid, cv=splitter_loo, scoring = "neg_mean_absolute_error")

### Coefficient 1
print("Coefficient 1 ....")
results = pd.DataFrame()
results["COSMIC_ID"]= test_df_50["COSMIC_ID"]

train_drug = train_df_50.copy()
test_drug = test_df_50.copy()
y_train_drug = train_drug["param_1"].values
y_test_drug =  test_drug["param_1"].values
    
for i, data_set in list(enumerate(datasets)):
    X_columns = X_feat_dict[data_set]
    scaler = MinMaxScaler().fit(train_drug[X_columns])
    Xtrain_drug = scaler.transform(train_drug[X_columns])
    grid.fit(Xtrain_drug, y_train_drug)
        
    # Pick the best parameterds, train again and predict on the test data
    model = Lasso(alpha=grid.best_params_["alpha"])
    model.fit(Xtrain_drug, y_train_drug)
    Xtest_drug = scaler.transform(test_drug[X_columns])
        
    y_pred = model.predict(Xtest_drug)  
    
    mse = mean_squared_error(y_test_drug, y_pred)
    mae = mean_absolute_error(y_test_drug, y_pred)
    
    results["pred_coef1"+str(i)]= y_pred        
    print("Dataset:", i, "best alpha:", grid.best_params_["alpha"])

results.to_csv(_FOLDER+"all_lasso_coef1.csv")

### Coefficient 2
print("Coefficient 2 ....")
results = pd.DataFrame()
results["COSMIC_ID"]= test_df_50["COSMIC_ID"]

train_drug = train_df_50.copy()
test_drug = test_df_50.copy()
y_train_drug = train_drug["param_2"].values
y_test_drug =  test_drug["param_2"].values
    
for i, data_set in list(enumerate(datasets)):
    X_columns = X_feat_dict[data_set]
    scaler = MinMaxScaler().fit(train_drug[X_columns])
    Xtrain_drug = scaler.transform(train_drug[X_columns])
    grid.fit(Xtrain_drug, y_train_drug)
        
    # Pick the best parameterds, train again and predict on the test data
    model = Lasso(alpha=grid.best_params_["alpha"])
    model.fit(Xtrain_drug, y_train_drug)
    Xtest_drug = scaler.transform(test_drug[X_columns])
        
    y_pred = model.predict(Xtest_drug)  
    
    mse = mean_squared_error(y_test_drug, y_pred)
    mae = mean_absolute_error(y_test_drug, y_pred)
    
    results["pred_coef2"+str(i)]= y_pred        
    print("Dataset:", i, "best alpha:", grid.best_params_["alpha"])

results.to_csv(_FOLDER+"all_lasso_coef2.csv")

### Coefficient 3
print("Coefficient 3 ....")
results = pd.DataFrame()
results["COSMIC_ID"]= test_df_50["COSMIC_ID"]

train_drug = train_df_50.copy()
test_drug = test_df_50.copy()
y_train_drug = train_drug["param_3"].values
y_test_drug =  test_drug["param_3"].values
    
for i, data_set in list(enumerate(datasets)):
    X_columns = X_feat_dict[data_set]
    scaler = MinMaxScaler().fit(train_drug[X_columns])
    Xtrain_drug = scaler.transform(train_drug[X_columns])
    grid.fit(Xtrain_drug, y_train_drug)
        
    # Pick the best parameterds, train again and predict on the test data
    model = Lasso(alpha=grid.best_params_["alpha"])
    model.fit(Xtrain_drug, y_train_drug)
    Xtest_drug = scaler.transform(test_drug[X_columns])
        
    y_pred = model.predict(Xtest_drug)  
    
    mse = mean_squared_error(y_test_drug, y_pred)
    mae = mean_absolute_error(y_test_drug, y_pred)
    
    results["pred_coef3"+str(i)]= y_pred        
    print("Dataset:", i, "best alpha:", grid.best_params_["alpha"])

results.to_csv(_FOLDER+"all_lasso_coef3.csv")

### Coefficient 4
print("Coefficient 4 ....")
results = pd.DataFrame()
results["COSMIC_ID"]= test_df_50["COSMIC_ID"]

train_drug = train_df_50.copy()
test_drug = test_df_50.copy()
y_train_drug = train_drug["param_4"].values
y_test_drug =  test_drug["param_4"].values
    
for i, data_set in list(enumerate(datasets)):
    X_columns = X_feat_dict[data_set]
    scaler = MinMaxScaler().fit(train_drug[X_columns])
    Xtrain_drug = scaler.transform(train_drug[X_columns])
    grid.fit(Xtrain_drug, y_train_drug)
        
    # Pick the best parameterds, train again and predict on the test data
    model = Lasso(alpha=grid.best_params_["alpha"])
    model.fit(Xtrain_drug, y_train_drug)
    Xtest_drug = scaler.transform(test_drug[X_columns])
        
    y_pred = model.predict(Xtest_drug)  
    
    mse = mean_squared_error(y_test_drug, y_pred)
    mae = mean_absolute_error(y_test_drug, y_pred)
    
    results["pred_coef4"+str(i)]= y_pred        
    print("Dataset:", i, "best alpha:", grid.best_params_["alpha"])

results.to_csv(_FOLDER+"all_lasso_coef4.csv")