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

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import gc

from sklearn.model_selection import LeaveOneOut
from IPython.print import print

import os
from data_preprocessing import FilteringCurves, ShowResponseCurves
from fitting_curves import FittingColumn, ShowResponseCurvesWithFitting, compute_r2_score
_FOLDER = "/home/acq18mk/master/results/"

### Coding Part

def LeaveOneOutError(model, X, y, metrics = "mse"):
    errors = []
    splitter_loo = LeaveOneOut()
#     print(splitter_loo.get_n_splits(X))
    
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


def RunCrossValidation(merged_df, drug_ids, number_coefficients, column_not_to_use =[], param_tested = "alpha", param_tested_values = [],
                       alpha=1, solver= "auto", print_results=True):
    
    param1 = ["param_" +str(i) for i in range(10)]
    param2 = ["param" +str(i) for i in range(10)] 
    norm_response  = ["norm_cells_"+str(i) for i in range(10)]
    con_columns  = ["fd_num_"+str(i) for i in range(10)]

    not_X_columns = param1 + param2 + norm_response + con_columns + column_not_to_use
    X_columns = set(df.columns) - set(not_X_columns)
    
    df_errors = pd.DataFrame()

    for drug_id in drug_ids:
        merged_df_i = merged_df[merged_df["DRUG_ID"]==drug_id]
        # merged_df_i has lower shape
        np.random.seed(123)
        indexes = np.random.permutation(merged_df_i.index)
        train_size = int(merged_df_i.shape[0]*0.8)
        indexes_train = indexes[:train_size]
        X_train = merged_df_i.loc[indexes_train, X_columns].values
    
        for i in range(number_coefficients):
            #check whether each coefficient needs its own parameters
            if type(alpha)==dict:
                alpha_value = alpha[i+1]
            else:
                alpha_value = alpha
                
            if type(solver)==solver:
                solver_value = solver[i+1]
            else:
                solver_value = solver
                
            y_train = merged_df_i.loc[indexes_train, "param_"+str(i+1)].values
            
            for param in param_tested_values:
    
                #check whether each coefficient needs its own parameters
                if param_tested == "alpha":
                    model = Ridge(alpha=param, solver= solver_value)
                elif param_tested == "solver":
                    model = Ridge(alpha=alpha_value, solver=param)
                    
                else:
                    print("ERROR: Unknown parameters")
                
                # mse is more sensitive to different parameters choice
                mse = LeaveOneOutError(model, X_train, y_train, metrics="mse")
                df_errors.loc[drug_id, "mse_coef"+str(i+1)+" "+str(param)] = mse

        
    best_values = {}
    for coef in range(number_coefficients):
        df_results = df_errors[["mse_coef"+str(coef+1)+" "+str(param) for param in param_tested_values]].describe().loc[["mean", "min","max"], :]
        if param_tested != "solver":
            best_param = np.float32(df_results.loc["mean",:].idxmin().split(" ")[1])
        else:
            best_param = df_results.loc["mean",:].idxmin().split(" ")[1]
        best_values[coef+1] = best_param
        if print_results:
            print(df_results)
            if type(best_param) != str:
                print("Coefficient %d: ,  Best %s: %.5f" % (coef+1, param_tested, best_param))
            else:
                print("Coefficient %d: ,  Best %s: %s" % (coef+1, param_tested, best_param))
        
    del df_errors
    print("\nBest values for parameter:", param_tested)
    print(best_values)
    return best_values

def TuneParameters(merged_df, drug_ids, number_coefficients, column_not_to_use =[], 
                   param_tested_alphas = [], param_tested_solvers = [], print_results=True):
    
    results = {}
    
    start_time = time.time()
    best_solver = RunCrossValidation(merged_df, drug_ids, 4, column_not_to_use= column_not_to_use, 
                                     param_tested = "solver", 
                                     param_tested_values = param_tested_solvers, 
                                     alpha = 1,
                                     print_results=print_results)

    results["solver"] = best_solver
    print("\n Execution time for tuning solver: %.3f seconds" % (time.time() - start_time))
    
    start_time = time.time()
    best_alpha = RunCrossValidation(merged_df, drug_ids, 4, column_not_to_use= column_not_to_use, 
                                    param_tested = "alpha", 
                                    param_tested_values = param_tested_alphas, 
                                    solver = best_solver,
                                    print_results=print_results)
            
    print("\n Execution time for tuning alpha: %.3f seconds" % (time.time() - start_time))
    results["alpha"] = best_alpha
    
    

    return  results

def TestTunedModel(merged_df, drug_ids, number_coefficients, column_not_to_use=[], alpha=1, solver ="auto", 
                     metrics = "mse", print_results=True):
    """Training and testing Kernels with the best found hyperparameters"""
    
    param1 = ["param_" +str(i) for i in range(10)]
    param2 = ["param" +str(i) for i in range(10)] 
    norm_response  = ["norm_cells_"+str(i) for i in range(10)]
    con_columns  = ["fd_num_"+str(i) for i in range(10)]

    not_X_columns = param1 + param2 + norm_response + con_columns + column_not_to_use
    X_columns = set(df.columns) - set(not_X_columns)
    
    df_errors_test = pd.DataFrame()

    for drug_id in drug_ids:
        # merged_df_i has lower shape
        merged_df_i = merged_df[merged_df["DRUG_ID"]==drug_id]
        
        np.random.seed(123)
        indexes = np.random.permutation(merged_df_i.index)
        train_size = int(merged_df_i.shape[0]*0.8)
        indexes_train = indexes[:train_size]
        indexes_test= indexes[train_size:]
        X_train = merged_df_i.loc[indexes_train, X_columns].values
        X_test = merged_df_i.loc[indexes_test, X_columns].values
    
        for i in range(number_coefficients):
#             param = best_param[i+1]
            y_train = merged_df_i.loc[indexes_train, "param_"+str(i+1)].values
            y_test = merged_df_i.loc[indexes_test, "param_"+str(i+1)].values
            
            #check whether each coefficient needs its own parameters
            if type(alpha)==dict:
                alpha_value = alpha[i+1]
            else:
                alpha_value = alpha
                
            if type(solver)==dict:
                solver_value = solver[i+1]
            else:
                solver_value = solver
                
            lin_reg = Ridge(alpha = alpha_value, solver = solver_value)
            lin_reg.fit(X_train, y_train)
            y_pred = np.exp(lin_reg.predict(X_test))
                                
            # mse is more sensitive to different parameters choice
            if metrics == "mse":
                error = mean_squared_error(y_test, y_pred)
            elif metrics == "mae":
                error = mean_absolute_error(y_test, y_pred)
            else:
                print("ERROR: Unknown metrics")
            df_errors_test.loc[drug_id, "mse_coef"+str(i+1)] = error
    
    df_results = df_errors_test.describe().loc[["mean", "min","max"], :]
    if print_results: 
        print(df_results)
    return df_results

### Analytical Part

# **Data Preprocessing pipeline:**
#     1. filter drug_profiles data 
#     (123 - three stages of filtration, 23 - two stages of filtration):
#         - "results/filtered_drug_profiles_123" (less data)
#         - "results/filtered_drug_profiles_23" (more data)
#     2. add drug features to drug data
#     - "data/Drug_Features.csv" (original data)
#     - "results/drug_features_with_properties2.csv" (data with pubchem properties)
#     3. merged drug_profiles and drug_features
# **For goog comparison:**
#     filter merged data so that they have only drug with features 
#     <br>for both data frames (original drug features and with added pubchem features)

### Finding optimal parameters for just drug profiles and cell lines

print("\nFinding optimal parameters for just drug profiles and cell lines\n")
df = pd.read_csv('results/merged_fitted_sigmoid4_123_with_drugs_description.csv').drop(["Drug_Name","Target_Pathway"], axis=1)

column_not_to_use = ["Unnamed: 0", "COSMIC_ID", "DRUG_ID", "Drug_Name", "Synonyms", "Target", "deriv_found", "PubChem_ID",
                     "elements", "inchi_key", "canonical_smiles", "inchi_string", "third_target", "first_target", "molecular_formula", "second_target", "Target_Pathway"]

gr = df.groupby(["DRUG_ID"])["COSMIC_ID"].count()
drug_ids = list(gr[gr > 50].index)
len(drug_ids)

param_tested_alphas = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 300, 500]
param_tested_solvers = ["auto", "svd", "cholesky", "lsqr", "sparse_cg"]


results = TuneParameters(df, drug_ids, 4, column_not_to_use=column_not_to_use, param_tested_alphas=param_tested_alphas,
                         param_tested_solvers = param_tested_solvers, print_results=False)

TestTunedModel(df, drug_ids, 4, column_not_to_use= column_not_to_use,
                                     alpha=results["alpha"],
                                     solver = results["solver"],
                                    metrics = "mse", print_results=False)


### Finding optimal parameters for drug profiles, cell lines and drug description

print("\nFinding optimal parameters for drug profiles, cell lines and drug description\n")
df = pd.read_csv(_FOLDER+'merged_fitted_sigmoid4_123_with_drugs_description.csv').drop(["Drug_Name","Target_Pathway"], axis=1)

# OHE and dumnies columns for Target_Pathway - 21 new columns
df = pd.concat([df, pd.get_dummies(df["Target_Pathway"])], axis=1).drop("Target_Pathway", axis=1)

column_not_to_use = ["Unnamed: 0", "COSMIC_ID", "DRUG_ID", "Drug_Name", "Synonyms", "Target", "deriv_found", "PubChem_ID",
                     "elements", "inchi_key", "canonical_smiles", "inchi_string", "third_target", "first_target", "molecular_formula", "second_target", "Target_Pathway"]

gr = df.groupby(["DRUG_ID"])["COSMIC_ID"].count()
drug_ids = list(gr[gr > 50].index)
len(drug_ids)

param_tested_alphas = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 300, 500]
param_tested_solvers = ["auto", "svd", "cholesky", "lsqr", "sparse_cg"]


results = TuneParameters(df, drug_ids, 4, column_not_to_use=column_not_to_use, param_tested_alphas=param_tested_alphas,
                         param_tested_solvers = param_tested_solvers, print_results=False)

TestTunedModel(df, drug_ids, 4, column_not_to_use= column_not_to_use,
                                     alpha=results["alpha"],
                                     solver = results["solver"],
                                    metrics = "mse", print_results=False)

### Finding optimal parameters for drug profiles, cell lines and drug features

print("\nFinding optimal parameters for drug profiles, cell lines and drug features\n")
df = pd.read_csv(_FOLDER+'merged_fitted_sigmoid4_123_with_drugs_description.csv')

column_not_to_use = ["Unnamed: 0", "COSMIC_ID", "DRUG_ID", "Drug_Name", "Synonyms", "Target", "deriv_found", "PubChem_ID",
                     "elements", "inchi_key", "canonical_smiles", "inchi_string", "third_target", "first_target", "molecular_formula", "second_target", "Target_Pathway"]

gr = df.groupby(["DRUG_ID"])["COSMIC_ID"].count()
drug_ids = list(gr[gr > 50].index)
len(drug_ids)

param_tested_alphas = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 300, 500]
param_tested_solvers = ["auto", "svd", "cholesky", "lsqr", "sparse_cg"]


results = TuneParameters(df, drug_ids, 4, column_not_to_use=column_not_to_use, param_tested_alphas=param_tested_alphas,
                         param_tested_solvers = param_tested_solvers, print_results=False)

TestTunedModel(df, drug_ids, 4, column_not_to_use= column_not_to_use,
                                     alpha=results["alpha"],
                                     solver = results["solver"],
                                    metrics = "mse", print_results=False)