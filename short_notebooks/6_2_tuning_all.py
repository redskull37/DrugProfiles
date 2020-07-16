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
from sklearn.preprocessing import StandardScaler
import os
from data_preprocessing import FilteringCurves, ShowResponseCurves
from fitting_curves import FittingColumn, ShowResponseCurvesWithFitting, compute_r2_score

#from IPython.display import display
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


def RunCrossValidation(train, drug_ids, number_coefficients, train_ratio=0.8, column_not_to_use =[], kernel='linear', param_tested = "alpha", 
                       param_tested_values = [], alpha=1, gamma=None, degree=3, coef0=1, features_to_scale=[], scaling=False, 
                      print_results=True):
    
    param1 = ["param_" +str(i) for i in range(10)]
    param2 = ["param" +str(i) for i in range(10)] 
    norm_response  = ["norm_cells_"+str(i) for i in range(10)]
    con_columns  = ["fd_num_"+str(i) for i in range(10)]
    not_X_columns = param1 + param2 + norm_response + con_columns + column_not_to_use
    X_columns = set(df.columns) - set(not_X_columns)
    print("Number of X_columns:", len(X_columns))
    
    df_errors = pd.DataFrame()
    #check whether each coefficient needs its own parameters
    if scaling:
        train = train.copy()
        scaler = StandardScaler()
        train[features_to_scale] = scaler.fit_transform(train[features_to_scale])
    else:
        pass
    
    for drug_id in drug_ids:
        X_train = train[train["DRUG_ID"]== drug_id ][X_columns].values
    
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
            
            y_train = train[train["DRUG_ID"]== drug_id ]["param_"+str(i+1)].values

            for param in param_tested_values:
    
                #check whether each coefficient needs its own parameters
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
                df_errors.loc[drug_id, "mse_coef"+str(i+1)+"_"+str(param)] = mse

        
    best_values = {}
    for coef in range(number_coefficients):
        df_results = df_errors[["mse_coef"+str(coef+1)+"_"+str(param) for param in param_tested_values]].describe().loc[["mean", "min","max"], :]
        best_param = np.float32(df_results.loc["mean",:].idxmin().split("_")[-1])
        best_values[coef+1] = best_param
        if print_results:
            print(df_results)
            print("Coefficient %d: ,  Best %s: %.5f" % (coef+1, param_tested, best_param))
        
    del df_errors
    print("%s kernel, best values for parameter: %s" % (kernel, param_tested))
    print(best_values)
    return best_values

def TestTunedKernels(train, test, drug_ids, number_coefficients, kernel, column_not_to_use =[], alpha=1, gamma=None, degree=3, coef0=1, 
                     metrics = "mse", features_to_scale=[], scaling=False, print_results=True):
    """Training and testing Kernels with the best found hyperparameters"""
    
    param1 = ["param_" +str(i) for i in range(10)]
    param2 = ["param" +str(i) for i in range(10)] 
    norm_response  = ["norm_cells_"+str(i) for i in range(10)]
    con_columns  = ["fd_num_"+str(i) for i in range(10)]

    not_X_columns = param1 + param2 + norm_response + con_columns+column_not_to_use
    X_columns = set(df.columns) - set(not_X_columns)
    print("Number of X_columns:", len(X_columns))
    
    if scaling:
        train = train.copy()
        scaler = StandardScaler()
        scaler.fit(train[features_to_scale])
        train[features_to_scale] = scaler.transform(train[features_to_scale])
        test[features_to_scale] = scaler.transform(test[features_to_scale])
    else:
        pass
    
    df_errors_test = pd.DataFrame()
    
    for drug_id in drug_ids:
        
        X_train = train[train["DRUG_ID"]==drug_id][X_columns].values
        X_test = test[test["DRUG_ID"]==drug_id][X_columns].values
    
        for i in range(number_coefficients):
            y_train = train[train["DRUG_ID"]== drug_id ]["param_"+str(i+1)].values
            y_test = test[test["DRUG_ID"]== drug_id ]["param_"+str(i+1)].values
            
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
            y_pred = np.exp(kr_lin.predict(X_test))
                                
            # mse is more sensitive to different parameters choice
            if metrics == "mse":
                error = mean_squared_error(y_test, y_pred)
            elif metrics == "mae":
                error = mean_absolute_error(y_test, y_pred)
            else:
                print("ERROR: Unknown metrics")
            df_errors_test.loc[drug_id, kernel+"_mse_coef"+str(i+1)] = error
    
    df_results = df_errors_test.describe().loc[["mean", "min","max"], :]
    if print_results: 
        print(df_results)
    return df_results

def TuneParameters(train, drug_ids, number_coefficients, kernels = [], column_not_to_use =[], param_tested = "alpha", 
                       param_tested_values = [], alpha=1, gamma=None, degree=3, coef0=1, features_to_scale=[], scaling=False, 
                      print_results=True):
    results = {}
    for kernel in kernels:
        start_time = time.time()
        if kernel == "linear":
            best_alpha = RunCrossValidation(train, drug_ids, 4, kernel=kernel, column_not_to_use=column_not_to_use, 
                                            param_tested = "alpha", param_tested_values = [0.1, 0.5, 1, 5, 7, 10, 30, 50, 100, 200, 300, 500],
                                            features_to_scale = features_to_scale, scaling = scaling,
                                            print_results=print_results)
            
            print("\n%s kernel: Execution time: %.3f seconds \n" % (kernel, (time.time() - start_time)))
            results[kernel]={}
            results[kernel]["alpha"] = best_alpha
            
        elif kernel == "polynomial":
            start_time = time.time()
            best_gamma = RunCrossValidation(train, drug_ids, 4, kernel='polynomial', column_not_to_use=column_not_to_use, param_tested = "gamma", 
                                            param_tested_values = [0.00001, 0.0001, 0.01, 0.1, 1], 
                                            features_to_scale = features_to_scale, scaling = scaling,
                                            print_results=print_results)

            best_degree = RunCrossValidation(train, drug_ids, 4, kernel='polynomial', column_not_to_use=column_not_to_use, param_tested = "degree", 
                                             gamma= best_gamma, param_tested_values = [1,2,3,4,5], 
                                             features_to_scale = features_to_scale, scaling = scaling,
                                             print_results=print_results)

            best_alpha = RunCrossValidation(train, drug_ids, 4, kernel='polynomial', column_not_to_use=column_not_to_use, param_tested = "alpha", 
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
            best_gamma = RunCrossValidation(train, drug_ids, 4, kernel = kernel, column_not_to_use=column_not_to_use, param_tested = "gamma", 
                                            param_tested_values = [0.00001, 0.0001, 0.01, 0.1, 0.5, 1], 
                                            features_to_scale = features_to_scale, scaling = scaling,
                                            print_results=print_results)

            
            best_alpha = RunCrossValidation(train, drug_ids, 4, kernel=kernel, column_not_to_use=column_not_to_use, param_tested = "alpha", 
                                            param_tested_values = [0.1, 0.5, 1, 5, 7, 10, 30, 50, 100, 200, 300, 500], 
                                            features_to_scale = features_to_scale, scaling = scaling,
                                            print_results=print_results)
            
            best_coef0 = RunCrossValidation(train, drug_ids, 4, kernel=kernel, column_not_to_use=column_not_to_use, gamma= best_gamma, 
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

def TrainTestBestParameters(train, test, drug_ids, number_coefficients, kernels =[], column_not_to_use =[], best_parameters_dict={}, 
                     metrics = "mse", features_to_scale=[], scaling=False, print_results=True):
    tests={}
    for kernel in kernels:
        if kernel == "linear":
            tests["linear"] = TestTunedKernels(train, test, drug_ids, 4, kernel='linear', column_not_to_use=column_not_to_use,
                                               alpha=best_parameters_dict[kernel]["alpha"], 
                                               metrics = "mse", 
                                               features_to_scale = features_to_scale, scaling = scaling,
                                               print_results=print_results)
        elif kernel == "polynomial":
            tests['polynomial'] = TestTunedKernels(train, test, drug_ids, 4, kernel='polynomial', column_not_to_use=column_not_to_use,
                                                   alpha=best_parameters_dict[kernel]["alpha"], 
                                                   gamma= best_parameters_dict[kernel]["gamma"], 
                                                   degree=best_parameters_dict[kernel]["degree"], 
                                                   metrics = "mse", 
                                                   features_to_scale = features_to_scale, scaling = scaling,
                                                   print_results=print_results)
        else:
            tests[kernel] = TestTunedKernels(train, test, drug_ids, 4, kernel=kernel, column_not_to_use=column_not_to_use,
                                             alpha=best_parameters_dict[kernel]["alpha"], 
                                             gamma= best_parameters_dict[kernel]["gamma"],
                                             coef0= best_parameters_dict[kernel]["coef0"],
                                             degree=1, metrics = "mse", 
                                             features_to_scale = features_to_scale, scaling = scaling,
                                             print_results=print_results)
    best_kernels = {}
    coef_names= ["coef_"+str(i) for i in range(1, number_coefficients+1)]
    compared_means = pd.DataFrame(index=coef_names, columns= kernels)
    for i in range(number_coefficients):
        test_kernels_comparison = pd.DataFrame(index=["mean", "min", "max"])
        for kernel in kernels:
            test_kernels_comparison[kernel] = tests[kernel][tests[kernel].columns[i]]
        
        compared_means.loc["coef_"+str(i+1), :] = test_kernels_comparison.loc["mean", :]
        print(test_kernels_comparison)
        best_kernels[i+1]= test_kernels_comparison.loc["mean", :].idxmin(axis=1)
        print("Coefficient: %d, best kernel: %s" % (i+1, best_kernels[i+1]))
    
    return best_kernels, compared_means

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

column_not_to_use = ["Unnamed: 0", "COSMIC_ID", "DRUG_ID", "Drug_Name", "Synonyms", "Target", "deriv_found", "PubChem_ID",
                     "elements", "inchi_key", "canonical_smiles", "inchi_string", "third_target", "first_target", "molecular_formula", "second_target", "Target_Pathway"]

param1 = ["param_" +str(i) for i in range(10)]
param2 = ["param" +str(i) for i in range(10)] 
norm_response  = ["norm_cells_"+str(i) for i in range(10)]

### Finding optimal parameters for just drug profiles and cell lines

print("\nFinding optimal parameters for just drug profiles and cell lines\n")
df = pd.read_csv(_FOLDER+'merged_fitted_sigmoid4_123_with_drugs_description.csv').drop(["Drug_Name","Target_Pathway"], axis=1)
train = pd.read_csv(_FOLDER+'train08_merged_fitted_sigmoid4_123_with_drugs_description.csv')
test = pd.read_csv(_FOLDER+'test02_merged_fitted_sigmoid4_123_with_drugs_description.csv')

conc_columns= ["fd_num_"+str(i) for i in range(10)]
response_norm = ['norm_cells_'+str(i) for i in range(10)]

gr = df.groupby(["DRUG_ID"])["COSMIC_ID"].count()
drug_ids = list(gr[gr > 50].index)
print("Number of drugs for training:", len(drug_ids))

kernels_to_test = ["linear", "sigmoid", "rbf", "polynomial", "additive_chi2", "laplacian"]
results = TuneParameters(train, drug_ids, 4, kernels = kernels_to_test, column_not_to_use=column_not_to_use, print_results=False)

print("Tuned parameters:")
print(results)
print("\nBetter presentation:")
for key in results:
    print(key,"\t", results[key])

best_kernels, compared_means = TrainTestBestParameters(train, test, drug_ids, 4, kernels = kernels_to_test, column_not_to_use=column_not_to_use, best_parameters_dict = results, print_results=True)
print("Best Kernels:", best_kernels)
compared_means.to_csv(_FOLDER+"kernel_learning_1_2.csv")
print(compared_means)

### Finding optimal parameters for drug profiles, cell lines and drug description

print("\nFinding optimal parameters for drug profiles, cell lines and drug description\n")
df = pd.read_csv(_FOLDER+'merged_fitted_sigmoid4_123_with_drugs_description.csv').drop(["Drug_Name","Target_Pathway"], axis=1)
train = pd.read_csv(_FOLDER+'train08_merged_fitted_sigmoid4_123_with_drugs_description.csv')
test = pd.read_csv(_FOLDER+'test02_merged_fitted_sigmoid4_123_with_drugs_description.csv')

# OHE and dumnies columns for Target_Pathway - 21 new columns
train = pd.concat([train, pd.get_dummies(train["Target_Pathway"])], axis=1).drop("Target_Pathway", axis=1)
test = pd.concat([test, pd.get_dummies(test["Target_Pathway"])], axis=1).drop("Target_Pathway", axis=1)

conc_columns= ["fd_num_"+str(i) for i in range(10)]
response_norm = ['norm_cells_'+str(i) for i in range(10)]

gr = df.groupby(["DRUG_ID"])["COSMIC_ID"].count()
drug_ids = list(gr[gr > 50].index)
print("Number of drugs for training:", len(drug_ids))

kernels_to_test = ["linear", "sigmoid", "rbf", "polynomial", "additive_chi2", "laplacian"]
results = TuneParameters(train, drug_ids, 4, kernels = kernels_to_test, column_not_to_use=column_not_to_use, print_results=False)

print("Tuned parameters:")
print(results)
print("\nBetter presentation:")
for key in results:
    print(key,"\t", results[key])

best_kernels, compared_means = TrainTestBestParameters(train, test, drug_ids, 4, kernels = kernels_to_test, column_not_to_use=column_not_to_use, best_parameters_dict = results, print_results=True)
print("Best Kernels:", best_kernels)
compared_means.to_csv(_FOLDER+"kernel_learning_2_2.csv")
print(compared_means)

### Finding optimal parameters for drug profiles, cell lines and drug features

print("\nFinding optimal parameters for drug profiles, cell lines and drug features\n")
df = pd.read_csv(_FOLDER+'merged_fitted_sigmoid4_123_with_drugs_properties.csv')
train = pd.read_csv(_FOLDER+'train08_merged_fitted_sigmoid4_123_with_drugs_properties.csv')
test = pd.read_csv(_FOLDER+'test02_merged_fitted_sigmoid4_123_with_drugs_properties.csv')

gr = df.groupby(["DRUG_ID"])["COSMIC_ID"].count()
drug_ids = list(gr[gr > 50].index)
print("Number of drugs for training:", len(drug_ids))

kernels_to_test = ["linear", "sigmoid", "rbf", "polynomial", "additive_chi2", "laplacian"]
results = TuneParameters(train, drug_ids, 4, kernels = kernels_to_test, column_not_to_use=column_not_to_use, print_results=False)

print("Tuned parameters:")
print(results)
print("\nBetter presentation:")
for key in results:
    print(key,"\t", results[key])

best_kernels, compared_means = TrainTestBestParameters(train, test, drug_ids, 4, kernels = kernels_to_test, column_not_to_use=column_not_to_use, best_parameters_dict = results, print_results=True)
print("Best Kernels:", best_kernels)
compared_means.to_csv(_FOLDER+"kernel_learning_3_2.csv")
print(compared_means)

### Finding optimal parameters for drug profiles, cell lines and drug features with SCALING

print("\nFinding optimal parameters for drug profiles, cell lines and drug features with scaling\n")
df = pd.read_csv(_FOLDER+'merged_fitted_sigmoid4_123_with_drugs_properties.csv')
train = pd.read_csv(_FOLDER+'train08_merged_fitted_sigmoid4_123_with_drugs_properties.csv')
test = pd.read_csv(_FOLDER+'test02_merged_fitted_sigmoid4_123_with_drugs_properties.csv')

potential_columns_for_normalisation = []
for col in df.columns:
    if (df[col].nunique()>2) & (df[col].dtype != "O"):
        potential_columns_for_normalisation.append(col)

columns_for_normalisation = list(set(potential_columns_for_normalisation) - set(norm_response) - set(param1) - set(param2) -set(['Unnamed: 0', 'DRUG_ID', 'COSMIC_ID',]))
gr = df.groupby(["DRUG_ID"])["COSMIC_ID"].count()
drug_ids = list(gr[gr > 50].index)
print("Number of drugs for training:", len(drug_ids))

kernels_to_test = ["linear", "sigmoid", "rbf", "polynomial", "additive_chi2", "laplacian"]
results = TuneParameters(train, drug_ids, 4, kernels = kernels_to_test, column_not_to_use=column_not_to_use, 
                         features_to_scale=columns_for_normalisation, scaling = True,
                         print_results=False)

print("Tuned parameters:")
print(results)
print("\nBetter presentation:")
for key in results:
    print(key,"\t", results[key])

best_kernels, compared_means = TrainTestBestParameters(train, test, drug_ids, 4, kernels = kernels_to_test, 
                                                       column_not_to_use=column_not_to_use, 
                                                       best_parameters_dict = results, 
                                                       features_to_scale=columns_for_normalisation, scaling = True,
                                                       print_results=True)
print("Best Kernels:", best_kernels)
compared_means.to_csv(_FOLDER+"kernel_learning_4_2.csv")
print(compared_means)
