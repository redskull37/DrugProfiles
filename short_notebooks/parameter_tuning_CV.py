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
from IPython.display import display

import os
from data_preprocessing import FilteringCurves, ShowResponseCurves
from fitting_curves import FittingColumn, ShowResponseCurvesWithFitting, compute_r2_score
_FOLDER = "./data/"


def LeaveOneOutError(kernel_model, X, y, metrics = "mse"):
    errors = []
    splitter_loo = LeaveOneOut()
    
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

def RunCrossValidation(drug_ids, number_coefficients, kernel='linear', param_tested = "alpha", 
                       param_tested_values = [], alpha=1, gamma=None, degree=3, coef0=1,
                      display_results=True):
    
    df_errors = pd.DataFrame()
    #check whether each coefficient needs its own parameters
    

    for drug_id in drug_ids:
        merged_df_i = merged_df[merged_df["DRUG_ID"]==drug_id]
        # merged_df_i has lower shape
        np.random.seed(123)
        indexes = np.random.permutation(merged_df_i.index)
        train_size = int(merged_df_i.shape[0]*0.8)
        indexes_train = indexes[:train_size]
        X_train = merged_df_i.loc[indexes_train, merged_df_i.columns[26:-4]].values
    
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
            
            y_train = merged_df_i.loc[indexes_train, "param_"+str(i+1)].values
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
        
    print("Tuning hyperparameter %s" % param_tested)
    print("x0, L, k, b   y = 1/ (L + np.exp(-k*(x-x0)))+b \n")
    best_values = {}
    for coef in range(number_coefficients):
#         print("Results for coefficient:", coef+1)
        df_results = df_errors[["mse_coef"+str(coef+1)+"_"+str(param) for param in param_tested_values]].describe().loc[["mean", "min","max"], :]
        best_param = np.float32(df_results.loc["mean",:].idxmin().split("_")[-1])
        best_values[coef+1] = best_param
        if display_results:
            display(df_results)
            print("Coefficient %d: ,  Best %s: %.5f" % (coef+1, param_tested, best_param))
        
    del df_errors
    print("%s kernel, best values for parameter: %s" % (kernel, param_tested))
    print(best_values)
    return best_values

def TestTunedKernels(drug_ids, number_coefficients, kernel, alpha=1, gamma=None, degree=3, coef0=1, 
                     metrics = "mse", display_results=True):
    """Training and testing Kernels with the best found hyperparameters"""
    
    df_errors_test = pd.DataFrame()

    for drug_id in drug_ids:
        # merged_df_i has lower shape
        merged_df_i = merged_df[merged_df["DRUG_ID"]==drug_id]
        
        np.random.seed(123)
        indexes = np.random.permutation(merged_df_i.index)
        train_size = int(merged_df_i.shape[0]*0.8)
        indexes_train = indexes[:train_size]
        indexes_test= indexes[train_size:]
        X_train = merged_df_i.loc[indexes_train, merged_df_i.columns[26:-4]].values
        X_test = merged_df_i.loc[indexes_test, merged_df_i.columns[26:-4]].values
    
        for i in range(number_coefficients):
#             param = best_param[i+1]
            y_train = merged_df_i.loc[indexes_train, "param_"+str(i+1)].values
            y_test = merged_df_i.loc[indexes_test, "param_"+str(i+1)].values
            
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
    if display_results:
        print("Testing %s kernel with tuned hyperparameters\n" % kernel)
        print("Coefficients x0, L, k, b   y = 1/ (L + np.exp(-k*(x-x0)))+b") 
        display(df_results)
    return df_results