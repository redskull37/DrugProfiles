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

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import gc

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
import os
from data_preprocessing import FilteringCurves, ShowResponseCurves
from fitting_curves import FittingColumn, ShowResponseCurvesWithFitting, compute_r2_score

# from IPython.display import display
#_FOLDER = "results/"
_FOLDER = "/home/acq18mk/master/results/"

### Coding Part

def LeaveOneOutError(kernel_model, X, y, metrics = "mse"):
    errors = []
    splitter_loo = LeaveOneOut()
#     print(splitter_loo.get_n_splits(X))
    
    for train_index, test_index in splitter_loo.split(X):
        X_train_loo, X_test_loo = X[train_index, :], X[test_index,:]
        y_train_loo, y_test_loo = y[train_index], y[test_index]
        
        model = kernel_model.fit(X_train_loo, y_train_loo)
        if metrics == "mse":
            mse = mean_squared_error(y_test_loo, model.predict(X_test_loo))
            errors.append(mse)
        elif metrics == "mae":
            mae = mean_absolute_error(y_test_loo, model.predict(X_test_loo))
            errors.append(mae)
    
    return (sum(errors)/ len(errors)) 

# gamma for RBF, laplacian, polynomial, exponential chi2 and sigmoid kernels

def RunCrossValidation(df_train, number_coefficients, column_not_to_use =[], kernel='linear', param_tested = "alpha", 
                       param_tested_values = [], alpha=1, gamma=None, degree=3, coef0=1,features_to_scale=[], scaling=False, 
                      print_results=True):
    
    param1 = ["param_" +str(i) for i in range(10)]
    param2 = ["param" +str(i) for i in range(10)] 
    norm_response  = ["norm_cells_"+str(i) for i in range(10)]
    con_columns  = ["fd_num_"+str(i) for i in range(10)]
    not_X_columns = param1 + param2 + norm_response + con_columns+column_not_to_use
    X_columns = set(df_train.columns) - set(not_X_columns)
    
#     print("RunCrossValidation - string checking...")
#     string_col =[]
#     for col in X_columns:
#         if df_train[col].dtype=="object":
#             string_col.append(col)
#     if len(string_col)>0:
#         print(col)
#     else:
#         print("no string are found\n")
        
    
    df_errors = pd.DataFrame(index=["mse"])
    
    if scaling:
        train=df_train.copy()
        scaler = MinMaxScaler()
        train[columns_for_normalisation] = scaler.fit_transform(train[columns_for_normalisation])
#         print("CV1", "first_target" in )
        X_train = train[X_columns].values     
    else:
        X_train = df_train[X_columns].values
    
    for i in range(number_coefficients):
        #check whether each coefficient needs its own parameters
        if type(alpha)==dict:
            alpha_value = alpha[i+1]
        else:
            alpha_value = alpha
                
        if type(gamma)==dict:
            gamma_value = gamma[i+1]
        else:
            gamma_value = gamma
            
        if type(degree)==dict:
            degree_value = degree[i+1]
        else:
            degree_value = degree
                
        if type(coef0)==dict:
            coef0_value = coef0[i+1]
        else:
            coef0_value = coef0
            
        y_train = df_train["param_"+str(i+1)].values

        for param in param_tested_values:
            if param_tested == "alpha":
                kernel_model = KernelRidge(kernel=kernel, 
                                               alpha=param, 
                                               gamma=gamma_value, 
                                               degree=degree_value, 
                                               coef0=coef0_value)
            elif param_tested == "gamma":
                kernel_model = KernelRidge(kernel=kernel, 
                                               alpha=alpha_value, 
                                               gamma=param, 
                                               degree=degree_value,
                                               coef0=coef0_value)
            elif param_tested == "degree":
                kernel_model = KernelRidge(kernel=kernel, 
                                               alpha=alpha_value, 
                                               gamma=gamma_value,
                                               degree=param, 
                                               coef0=coef0_value)
            elif param_tested == "coef0":
                kernel_model = KernelRidge(kernel=kernel, 
                                               alpha=alpha_value,  
                                               gamma=gamma_value,
                                               degree=degree_value,
                                               coef0=param)
            else:
                print("ERROR: Unknown parameters")
                
            # mse is more sensitive to different parameters choice
            mse = LeaveOneOutError(kernel_model, X_train, y_train, metrics="mse")
            df_errors.loc["mse", "mse_coef"+str(i+1)+"_"+str(param)] = mse

    best_values = {}
    for coef in range(number_coefficients):
        best_param = np.float32(df_errors.idxmin(axis=1)[0].split("_")[-1])
        best_values[coef+1] = best_param
        if print_results:
            print(df_results)
            print("Coefficient %d: ,  Best %s: %.5f" % (coef+1, param_tested, best_param))
        
    del df_errors
    print("%s kernel, best values for parameter: %s" % (kernel, param_tested))
    print(best_values)
    return best_values

def TestTunedKernels(df_train, df_test, number_coefficients, kernel, column_not_to_use =[], alpha=1, gamma=None, degree=3, coef0=1, 
                     metrics = "mse", features_to_scale=[], scaling=False, print_results=True):
    """Training and testing Kernels with the best found hyperparameters"""
    
    param1 = ["param_" +str(i) for i in range(10)]
    param2 = ["param" +str(i) for i in range(10)] 
    norm_response  = ["norm_cells_"+str(i) for i in range(10)]
    con_columns  = ["fd_num_"+str(i) for i in range(10)]

    not_X_columns = param1 + param2 + norm_response + con_columns+column_not_to_use
    X_columns = set(df_train.columns) - set(not_X_columns)
    print("Number of X_columns:", len(X_columns))
    
    df_errors_test = pd.DataFrame(index=["mse"])
    if scaling:
        train = df_train.copy()
        test = df_test.copy()
        scaler = MinMaxScaler()
        scaler.fit(train[columns_for_normalisation])
        train[columns_for_normalisation] = scaler.transform(train[columns_for_normalisation])
        X_train = train[X_columns].values  
        test[columns_for_normalisation] = scaler.transform(test[columns_for_normalisation])
        X_test = test[X_columns].values
    else:
        X_train = df_train[X_columns].values
        X_test = df_test[X_columns].values
    
    for i in range(number_coefficients):
        y_train = df_train["param_"+str(i+1)].values
        y_test = df_test["param_"+str(i+1)].values
            
        #check whether each coefficient needs its own parameters
        if type(alpha)==dict:
            alpha_value = alpha[i+1]
        else:
            alpha_value = alpha
                
        if type(gamma)==dict:
            gamma_value = gamma[i+1]
        else:
            gamma_value = gamma
            
        if type(degree)==dict:
            degree_value = degree[i+1]
        else:
            degree_value = degree
                
        if type(coef0)==dict:
            coef0_value = coef0[i+1]
        else:
            coef0_value = coef0
                
        kr_lin = KernelRidge(kernel = kernel, alpha = alpha_value, gamma=gamma_value, 
                                 degree=degree_value, coef0=coef0_value)
        kr_lin.fit(X_train, y_train)
        y_pred = kr_lin.predict(X_test)
                                
        # mse is more sensitive to different parameters choice
        if metrics == "mse":
            error = mean_squared_error(y_test, y_pred)
        elif metrics == "mae":
            error = mean_absolute_error(y_test, y_pred)
        else:
            print("ERROR: Unknown metrics")
        df_errors_test.loc["mse", kernel+" mse_coef"+str(i+1)] = error

    return df_errors_test


def TuneParameters(train, number_coefficients, kernels = [], column_not_to_use =[], param_tested = "alpha", 
                       param_tested_values = [], alpha=1, gamma=None, degree=3, coef0=1, features_to_scale=[], scaling=False, 
                      print_results=True):
    results = {}
    for kernel in kernels:
        start_time = time.time()
        if kernel == "linear":
            best_alpha = RunCrossValidation(train, number_coefficients, kernel=kernel, column_not_to_use=column_not_to_use, 
                                            param_tested = "alpha", param_tested_values = [0.1, 0.5, 1, 5, 7, 10, 30, 50, 100, 200, 300, 500],
                                            features_to_scale = features_to_scale, scaling = scaling,
                                            print_results=print_results)
            
            print("\n%s kernel: Execution time: %.3f seconds \n" % (kernel, (time.time() - start_time)))
            results[kernel]={}
            results[kernel]["alpha"] = best_alpha
            
        elif kernel == "polynomial":
            start_time = time.time()
            best_gamma = RunCrossValidation(train, number_coefficients, kernel='polynomial', column_not_to_use=column_not_to_use, param_tested = "gamma", 
                                            param_tested_values = [0.00001, 0.0001, 0.01, 0.1, 1], 
                                            features_to_scale = features_to_scale, scaling = scaling,
                                            print_results=print_results)

            best_degree = RunCrossValidation(train, number_coefficients, kernel='polynomial', column_not_to_use=column_not_to_use, param_tested = "degree", 
                                             gamma= best_gamma, param_tested_values = [1,2,3,4,5], 
                                             features_to_scale = features_to_scale, scaling = scaling,
                                             print_results=print_results)

            best_alpha = RunCrossValidation(train, number_coefficients, kernel='polynomial', column_not_to_use=column_not_to_use, param_tested = "alpha", 
                                            gamma= best_gamma, degree = best_degree,
                                            param_tested_values = [0.001, 0.01, 0.1, 1, 5, 7], 
                                            features_to_scale = features_to_scale, scaling = scaling,
                                            print_results=print_results)            
            
            print("\n%s kernel: Execution time: %.3f seconds \n" % (kernel, (time.time() - start_time)))
            results[kernel]={}
            results[kernel]["alpha"] = best_alpha
            results[kernel]["gamma"] = best_gamma
            results[kernel]["degree"] = best_degree
            
        else: 
            start_time = time.time()
            best_gamma = RunCrossValidation(train, number_coefficients, kernel = kernel, column_not_to_use=column_not_to_use, param_tested = "gamma", 
                                            param_tested_values = [0.00001, 0.0001, 0.01, 0.1, 0.5, 1], 
                                            features_to_scale = features_to_scale, scaling = scaling,
                                            print_results=print_results)

            
            best_alpha = RunCrossValidation(train, number_coefficients, kernel=kernel, column_not_to_use=column_not_to_use, param_tested = "alpha", 
                                            param_tested_values = [0.1, 0.5, 1, 5, 7, 10, 30, 50, 100, 200, 300, 500], 
                                            features_to_scale = features_to_scale, scaling = scaling,
                                            print_results=print_results)
            
            best_coef0 = RunCrossValidation(train, number_coefficients, kernel=kernel, column_not_to_use=column_not_to_use, gamma= best_gamma, 
                                            param_tested = "coef0", alpha=best_alpha,
                                            param_tested_values = [-0.1, 0, 0.1, 0.5, 1,  5, 10], 
                                            features_to_scale = features_to_scale, scaling = scaling,
                                            print_results=print_results)

            print("\n%s kernel: Execution time: %.3f seconds \n" % (kernel, (time.time() - start_time)))
            results[kernel]={}
            results[kernel]["alpha"] = best_alpha
            results[kernel]["gamma"] = best_gamma
            results[kernel]["coef0"] = best_coef0
            
    return  results

def TrainTestBestParameters(train, test, number_coefficients, kernels =[], column_not_to_use =[], best_parameters_dict={}, 
                     metrics = "mse", features_to_scale=[], scaling=False, print_results=True):
    
    df_results = pd.DataFrame()
    for kernel in kernels:
        if kernel == "linear":
            test_results = TestTunedKernels(train, test, number_coefficients, kernel='linear', column_not_to_use=column_not_to_use,
                                               alpha=best_parameters_dict[kernel]["alpha"], 
                                               metrics = "mse", 
                                               features_to_scale = features_to_scale, scaling = scaling,
                                               print_results=print_results)
            df_results = pd.concat([df_results, test_results], axis=1)
            
        elif kernel == "polynomial":
            test_results = TestTunedKernels(train, test, number_coefficients, kernel='polynomial', column_not_to_use=column_not_to_use,
                                                   alpha=best_parameters_dict[kernel]["alpha"], 
                                                   gamma= best_parameters_dict[kernel]["gamma"], 
                                                   degree=best_parameters_dict[kernel]["degree"], 
                                                   metrics = "mse", 
                                                   features_to_scale = features_to_scale, scaling = scaling,
                                                   print_results=print_results)
            df_results = pd.concat([df_results, test_results], axis=1)
            
        else:
            test_results = TestTunedKernels(train, test, number_coefficients, kernel=kernel, column_not_to_use=column_not_to_use,
                                             alpha=best_parameters_dict[kernel]["alpha"], 
                                             gamma= best_parameters_dict[kernel]["gamma"],
                                             coef0= best_parameters_dict[kernel]["coef0"],
                                             degree=1, metrics = "mse", 
                                             features_to_scale = features_to_scale, scaling = scaling,
                                             print_results=print_results)
            df_results = pd.concat([df_results, test_results], axis=1)
    
#     print("TrainTestBestParameters",df_results)
    best_kernels = {}
    for coef in range(1, number_coefficients+1):
        columns_names = [kernel+" mse_coef"+str(coef)]
        best_kernels[coef] ={}
        best_kernels[coef]["kernel"] = df_results[columns_names].idxmin(axis=1)[0].split(" ")[0]
        best_kernels[coef]["mse"] = df_results[columns_names].min(axis=1)[0]
    
    return best_kernels, df_results

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

column_not_to_use = ["Unnamed: 0", "Unnamed: 0.1","COSMIC_ID", "DRUG_ID", "Drug_Name", "Synonyms", "Target", "deriv_found", "PubChem_ID",
                     "elements", "inchi_key", "canonical_smiles", "inchi_string", "third_target", "first_target", "molecular_formula", "second_target", "Target_Pathway"]

param1 = ["param_" +str(i) for i in range(10)]
param2 = ["param" +str(i) for i in range(10)] 
norm_response  = ["norm_cells_"+str(i) for i in range(10)]

print("Dealing with ready train and test data!!!")

### 1. Finding optimal parameters for just drug profiles and cell lines

print("\n1. Finding optimal parameters for just drug profiles and cell lines\n")

train = pd.read_csv(_FOLDER+"train08_merged_fitted_sigmoid4_123_with_drugs_description.csv")
test = pd.read_csv(_FOLDER+"test02_merged_fitted_sigmoid4_123_with_drugs_description.csv")

kernels_to_test = ["linear", "sigmoid", "rbf", "polynomial", "additive_chi2", "laplacian"]
results = TuneParameters(train, 4, kernels = kernels_to_test, column_not_to_use=column_not_to_use, print_results=False)

print("Tuned parameters:")
print(results)
print("\nBetter presentation:")
for key in results:
    print(key,"\t", results[key])

best_kernels, df_results = TrainTestBestParameters(train, test, 4, kernels = kernels_to_test, column_not_to_use=column_not_to_use, best_parameters_dict = results, print_results=True)
print("Best Kernels:", best_kernels)

print("\nAll results\n")
print(df_results)
df_results.to_csv(_FOLDER+"kernel_learning_1_2.csv")
      

### 2. Finding optimal parameters for drug profiles, cell lines and drug description

print("\n2. Finding optimal parameters for drug profiles, cell lines and drug description\n")

train = pd.read_csv(_FOLDER+"train08_merged_fitted_sigmoid4_123_with_drugs_description.csv")
test = pd.read_csv(_FOLDER+"test02_merged_fitted_sigmoid4_123_with_drugs_description.csv")

# OHE and dumnies columns for Target_Pathway - 21 new columns
df = pd.concat([df, pd.get_dummies(df["Target_Pathway"])], axis=1).drop("Target_Pathway", axis=1)

kernels_to_test = ["linear", "sigmoid", "rbf", "polynomial", "additive_chi2", "laplacian"]
results = TuneParameters(train, 4, kernels = kernels_to_test, column_not_to_use=column_not_to_use, print_results=False)

print("Tuned parameters:")
print(results)
print("\nBetter presentation:")
for key in results:
    print(key,"\t", results[key])

best_kernels, df_results = TrainTestBestParameters(train, test, 4, kernels = kernels_to_test, column_not_to_use=column_not_to_use, best_parameters_dict = results, print_results=True)
print("Best Kernels:", best_kernels)

print("\nAll results\n")
print(df_results)
df_results.to_csv(_FOLDER+"kernel_learning_2_2.csv")

### 3. Finding optimal parameters for drug profiles, cell lines and drug features

print("\n3. Finding optimal parameters for drug profiles, cell lines and drug features\n")

train = pd.read_csv(_FOLDER+"train08_merged_fitted_sigmoid4_123_with_drugs_properties.csv")
test = pd.read_csv(_FOLDER+ "test02_merged_fitted_sigmoid4_123_with_drugs_properties.csv")

kernels_to_test = ["linear", "sigmoid", "rbf", "polynomial", "additive_chi2", "laplacian"]
results = TuneParameters(train, 4, kernels = kernels_to_test, column_not_to_use=column_not_to_use, print_results=False)

print("Tuned parameters:")
print(results)
print("\nBetter presentation:")
for key in results:
    print(key,"\t", results[key])

best_kernels, df_results = TrainTestBestParameters(train, test, 4, kernels = kernels_to_test, column_not_to_use=column_not_to_use, best_parameters_dict = results, print_results=True)
print("Best Kernels:", best_kernels)

print("\nAll results\n")
print(df_results)
df_results.to_csv(_FOLDER+"kernel_learning_3_2.csv")

### 4. Finding optimal parameters for drug profiles, cell lines and drug features with SCALING

print("\n4. Finding optimal parameters for drug profiles, cell lines and drug features with scaling\n")

train_set = pd.read_csv(_FOLDER+"train08_merged_fitted_sigmoid4_123_with_drugs_properties.csv")
test_set = pd.read_csv(_FOLDER+ "test02_merged_fitted_sigmoid4_123_with_drugs_properties.csv")

potential_columns_for_normalisation = []
for col in train_set.columns:
    if (train_set[col].nunique()>2) & (train_set[col].dtype != "O"):
        potential_columns_for_normalisation.append(col)
        
param1 = ["param_" +str(i) for i in range(10)]
param2 = ["param" +str(i) for i in range(10)] 
conc_columns= ["fd_num_"+str(i) for i in range(10)]
norm_response  = ["norm_cells_"+str(i) for i in range(10)]
column_not_to_use = ["Unnamed: 0", "Unnamed: 0.1","COSMIC_ID", "DRUG_ID", "Drug_Name", "Synonyms", "Target", "deriv_found", "PubChem_ID",
                     "elements", "inchi_key", "canonical_smiles", "inchi_string", "third_target", "first_target", "molecular_formula", "second_target", "Target_Pathway"]


columns_for_normalisation = list(set(potential_columns_for_normalisation) - set(norm_response) - set(param1) - set(param2) -set(['Unnamed: 0', 'DRUG_ID', 'COSMIC_ID',]))

kernels_to_test_set = ["linear", "sigmoid"]#, "rbf", "polynomial", "additive_chi2", "laplacian"]
results = TuneParameters(train_set, number_coefficients=4, kernels=kernels_to_test_set, 
                         column_not_to_use=column_not_to_use,  
                         alpha=1, gamma=None, degree=3, coef0=1, 
                         features_to_scale=columns_for_normalisation, 
                         scaling=True, print_results=True)

print("Tuned parameters:")
print(results)
print("\nBetter presentation:")
for key in results:
    print(key,"\t", results[key])

best_kernels, df_results = TrainTestBestParameters(train, test, 4, kernels = kernels_to_test, 
                                                   column_not_to_use=column_not_to_use, 
                                                   best_parameters_dict = results, 
                                                   features_to_scale=columns_for_normalisation, scaling=True,
                                                   print_results=True)
print("Best Kernels:", best_kernels)

print("\nAll results\n")
print(df_results)
df_results.to_csv(_FOLDER+"kernel_learning_4_2.csv")