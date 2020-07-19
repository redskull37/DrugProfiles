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


def TrainTestBestParameters(merged_df, drug_ids, number_coefficients, kernels =[], 
                            column_not_to_use =[], best_parameters_dict={}, 
                            metrics = "mse", features_to_scale=[], 
                            scaling=False, print_results=True):
    tests={}
    for kernel in kernels:
        if kernel == "linear":
            tests["linear"] = TestTunedKernels(merged_df, drug_ids, number_coefficients, 
                                               kernel = kernel, 
                                               column_not_to_use=column_not_to_use,
                                               C = best_parameters_dict[kernel]["C"], 
                                               epsilon =best_parameters_dict[kernel]["epsilon"],
                                               metrics = "mse", 
                                               features_to_scale = features_to_scale, scaling = scaling,
                                               print_results=print_results)
        elif kernel == "poly":
            tests['poly'] = TestTunedKernels(merged_df, drug_ids, number_coefficients, 
                                             kernel= kernel, 
                                             column_not_to_use  =column_not_to_use,
                                             C = best_parameters_dict[kernel]["C"],  
                                             degree = best_parameters_dict[kernel]["degree"], 
                                             epsilon = best_parameters_dict[kernel]["epsilon"],
                                             coef0 = best_parameters_dict[kernel]["coef0"],
                                             metrics = "mse", 
                                             features_to_scale = features_to_scale, scaling = scaling,
                                             print_results=print_results)
        else:
            tests[kernel] = TestTunedKernels(merged_df, drug_ids, number_coefficients, 
                                             kernel = kernel, 
                                             column_not_to_use = column_not_to_use,
                                             C = best_parameters_dict[kernel]["C"], 
                                             coef0 = best_parameters_dict[kernel]["coef0"],
                                             epsilon = best_parameters_dict[kernel]["epsilon"],
                                             metrics = "mse", 
                                             features_to_scale = features_to_scale, scaling = scaling,
                                             print_results = print_results)
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


def RunCrossValidation(merged_df, drug_ids, number_coefficients, train_ratio=0.8, column_not_to_use =[], 
                       kernel='linear', param_tested = "C", param_tested_values = [], 
                       degree=3, gamma="scale", coef0=0.0, C=1.0, epsilon=0.1, cache_size=200,
                       features_to_scale=[], scaling=False, print_results=True):
    
    param1 = ["param_" +str(i) for i in range(10)]
    param2 = ["param" +str(i) for i in range(10)] 
    norm_response  = ["norm_cells_"+str(i) for i in range(10)]
    con_columns  = ["fd_num_"+str(i) for i in range(10)]
    not_X_columns = param1 + param2 + norm_response + con_columns+column_not_to_use
    X_columns = set(merged_df.columns) - set(not_X_columns)
    print("Number of X_columns:", len(X_columns))
    
    df_errors = pd.DataFrame()
    #check whether each coefficient needs its own parameters
    

    for drug_id in drug_ids:
        merged_df_i = merged_df[merged_df["DRUG_ID"]==drug_id]
        # merged_df_i has lower shape
        np.random.seed(123)
        indexes = np.random.permutation(merged_df_i.index)
        train_size = int(merged_df_i.shape[0]*train_ratio)
        indexes_train = indexes[:train_size]
        if scaling:
            train=merged_df_i.loc[indexes_train, X_columns].copy()
            scaler = MinMaxScaler()
            train[columns_for_normalisation] = scaler.fit_transform(train[columns_for_normalisation])
            X_train = train.values     
        else:
            X_train = merged_df_i.loc[indexes_train, X_columns].values
    
        for i in range(number_coefficients):
            #check whether each coefficient needs its own parameters
            
            if type(cache_size)==dict:
                cache_size_value = cache_size[i+1]
            else:
                cache_size_value = cache_size
            
            if type(epsilon)==dict:
                epsilon_value = epsilon[i+1]
            else:
                epsilon_value = epsilon
                
            if type(C)==dict:
                C_value = C[i+1]
            else:
                C_value = C
                
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
                if param_tested == "cache_size":
                    kernel_model = SVR(kernel = kernel, 
                                       cache_size = param, 
                                       epsilon = epsilon_value,
                                       C = C_value,
                                       gamma = gamma_value, 
                                       degree = degree_value, 
                                       coef0 = coef0_value)
                
                elif param_tested == "epsilon":
                    kernel_model = SVR(kernel = kernel, 
                                       epsilon = param, 
                                       cache_size=cache_size_value,
                                       C = C_value,
                                       gamma = gamma_value, 
                                       degree = degree_value, 
                                       coef0 = coef0_value)
        
                elif param_tested == "C":
                    kernel_model = SVR(kernel = kernel,
                                       C = param,
                                       epsilon = epsilon_value,
                                       cache_size=cache_size_value,
                                       gamma = gamma_value, 
                                       degree = degree_value, 
                                       coef0 = coef0_value)
                    
                elif param_tested == "gamma":
                    kernel_model = SVR(kernel = kernel, 
                                       C = C_value, 
                                       epsilon = epsilon_value,
                                       cache_size=cache_size_value,
                                       gamma = param, 
                                       degree = degree_value,
                                       coef0 = coef0_value)
                    
                elif param_tested == "degree":
                    kernel_model = SVR(kernel = kernel, 
                                       C = C_value,
                                       epsilon = epsilon_value,
                                       cache_size=cache_size_value,
                                       gamma = gamma_value,
                                       degree = param, 
                                       coef0 = coef0_value)
                elif param_tested == "coef0":
                    kernel_model = SVR(kernel = kernel, 
                                       C = C_value,
                                       epsilon = epsilon_value,
                                       cache_size=cache_size_value,
                                       gamma = gamma_value,
                                       degree = degree_value,
                                       coef0 = param)
                else:
                    print("ERROR: Unknown parameters")
                
                # mse is more sensitive to different parameters choice
                mse = LeaveOneOutError(kernel_model, X_train, y_train, metrics="mse")
                df_errors.loc[drug_id, "mse_coef"+str(i+1)+"_"+str(param)] = mse

        
    best_values = {}
    for coef in range(number_coefficients):
        df_results = df_errors[["mse_coef"+str(coef+1)+"_"+str(param) for param in param_tested_values]].describe().loc[["mean", "min","max"], :]
        best_param = df_results.loc["mean",:].idxmin().split("_")[-1]
#         print(best_param)
        if param!= "gamma":
            best_param = np.float32(best_param)
        best_values[coef+1] = best_param
        if print_results:
            print(df_results)
            print("Coefficient %d: ,  Best %s: %.5f" % (coef+1, param_tested, best_param))
        
    del df_errors
    print("%s kernel, best values for parameter: %s" % (kernel, param_tested))
    print(best_values)
    return best_values

def TuneParameters(merged_df, drug_ids, number_coefficients, kernels = [], column_not_to_use =[], 
                   param_tested = "C", param_tested_values = [], 
                   degree=3, gamma='scale', coef0=0.0, C=1.0, epsilon=0.1, cache_size=200,
                   features_to_scale=[], scaling=False, print_results=True):
                         
    results = {}
    for kernel in kernels:
        start_time = time.time()
        if kernel == "linear":
#             best_cache_size = RunCrossValidation(merged_df, drug_ids, number_coefficients, 
#                                             kernel=kernel, 
#                                             column_not_to_use=column_not_to_use, 
#                                             param_tested = "cache_size", 
#                                             param_tested_values = [5, 10, 20, 50],
#                                             features_to_scale = features_to_scale, scaling = scaling,
#                                             print_results=print_results)
            
            best_epsilon = RunCrossValidation(merged_df, drug_ids, number_coefficients, 
                                            kernel=kernel, 
                                            column_not_to_use=column_not_to_use, 
                                            param_tested = "epsilon", 
                                            param_tested_values = [0.001, 0.01, 0.1, 1, 2, 5],
                                            features_to_scale = features_to_scale, scaling = scaling,
                                            print_results=print_results)
                         
            best_C = RunCrossValidation(merged_df, drug_ids, number_coefficients, 
                                        kernel=kernel, 
                                        column_not_to_use=column_not_to_use, 
                                        param_tested = "C", 
                                        param_tested_values = [0.1, 0.5, 1, 5, 7, 10, 30, 50, 100, 200, 300, 500],
                                        features_to_scale = features_to_scale, scaling = scaling,
                                        epsilon = best_epsilon,
                                        print_results=print_results)
            
            print("\n%s kernel: Execution time: %.3f seconds \n" % (kernel, (time.time() - start_time)))
            results[kernel]={}
            results[kernel]["epsilon"] = best_epsilon
            results[kernel]["C"] = best_C
            
        elif kernel == "poly":
            start_time = time.time()
#             best_cache_size = RunCrossValidation(merged_df, drug_ids, number_coefficients, 
#                                             kernel =k ernel, 
#                                             column_not_to_use = column_not_to_use, 
#                                             param_tested = "cache_size", 
#                                             param_tested_values = [5, 10, 20, 50],
#                                             features_to_scale = features_to_scale, scaling = scaling,
#                                             print_results=print_results)
            
            best_epsilon = RunCrossValidation(merged_df, drug_ids, number_coefficients, 
                                            kernel = kernel, 
                                            column_not_to_use = column_not_to_use, 
                                            param_tested = "epsilon", 
                                            param_tested_values = [0.001, 0.01, 0.1, 1, 2, 5],
                                            features_to_scale = features_to_scale, scaling = scaling,
                                            print_results=print_results)

            best_degree = RunCrossValidation(merged_df, drug_ids, number_coefficients, 
                                             kernel = kernel, 
                                             column_not_to_use = column_not_to_use, 
                                             param_tested = "degree", 
                                             param_tested_values = [1,2,3,4,5], 
                                             epsilon = best_epsilon,
                                             features_to_scale = features_to_scale, scaling = scaling,
                                             print_results=print_results)
            
            best_coef0 = RunCrossValidation(merged_df, drug_ids, number_coefficients, 
                                            kernel = kernel, 
                                            column_not_to_use = column_not_to_use, 
                                            param_tested = "coef0", 
                                            param_tested_values = [-0.1, 0, 0.1, 0.5, 1,  5, 10], 
                                            epsilon = best_epsilon,
                                            degree = best_degree,
                                            features_to_scale = features_to_scale, scaling = scaling,
                                            print_results=print_results)

            best_C = RunCrossValidation(merged_df, drug_ids, number_coefficients, 
                                        kernel = kernel, 
                                        column_not_to_use = column_not_to_use, 
                                        param_tested = "C", 
                                        degree = best_degree,
                                        coef0 = best_coef0,
                                        param_tested_values = [0.001, 0.01, 0.1, 1, 5, 7], 
                                        features_to_scale = features_to_scale, scaling = scaling,
                                        print_results=print_results)            
            
                         
            print("\n%s kernel: Execution time: %.3f seconds \n" % (kernel, (time.time() - start_time)))
            results[kernel]={}
            results[kernel]["C"] = best_C
            results[kernel]["degree"] = best_degree
            results[kernel]["coef0"] = best_coef0
            results[kernel]["epsilon"] = best_epsilon
            
        else: 
            start_time = time.time()
#             best_cache_size = RunCrossValidation(merged_df, drug_ids, number_coefficients, 
#                                             kernel =kernel, 
#                                             column_not_to_use = column_not_to_use, 
#                                             param_tested = "cache_size", 
#                                             param_tested_values = [5, 10, 20, 50],
#                                             features_to_scale = features_to_scale, scaling = scaling,
#                                             print_results=print_results
            
            best_epsilon = RunCrossValidation(merged_df, drug_ids, number_coefficients, 
                                            kernel = kernel, 
                                            column_not_to_use = column_not_to_use, 
                                            param_tested = "epsilon", 
                                            param_tested_values = [0.001, 0.01, 0.1, 1, 2, 5],
                                            features_to_scale = features_to_scale, scaling = scaling,
                                            print_results=print_results)
            
            best_coef0 = RunCrossValidation(merged_df, drug_ids, number_coefficients, 
                                            kernel = kernel, 
                                            column_not_to_use = column_not_to_use, 
                                            param_tested = "coef0",
                                            param_tested_values = [-0.1, 0, 0.1, 0.5, 1,  5, 10], 
                                            epsilon = best_epsilon,
                                            features_to_scale = features_to_scale, scaling = scaling,
                                            print_results=print_results)

            best_C = RunCrossValidation(merged_df, drug_ids, number_coefficients, 
                                        kernel = kernel, 
                                        column_not_to_use = column_not_to_use, 
                                        param_tested = "C", 
                                        coef0 = best_coef0,
                                        param_tested_values = [0.001, 0.01, 0.1, 1, 5, 7], 
                                        features_to_scale = features_to_scale, scaling = scaling,
                                        print_results=print_results)
            print("\n%s kernel: Execution time: %.3f seconds \n" % (kernel, (time.time() - start_time)))
            results[kernel]={}
            results[kernel]["C"] = best_C
            results[kernel]["coef0"] = best_coef0
            results[kernel]["epsilon"] = best_epsilon
            
    return  results


def TestTunedKernels(merged_df, drug_ids, number_coefficients, kernel, train_ratio =0.8, column_not_to_use =[], 
                     degree=3, gamma='scale', coef0=0.0, C=1.0, epsilon=0.1, cache_size=200,
                     metrics = "mse", features_to_scale=[], scaling=False, print_results=True):
    """Training and testing Kernels with the best found hyperparameters"""
    
    param1 = ["param_" +str(i) for i in range(10)]
    param2 = ["param" +str(i) for i in range(10)] 
    norm_response  = ["norm_cells_"+str(i) for i in range(10)]
    con_columns  = ["fd_num_"+str(i) for i in range(10)]

    not_X_columns = param1 + param2 + norm_response + con_columns+column_not_to_use
    X_columns = set(df.columns) - set(not_X_columns)
    print("Number of X_columns:", len(X_columns))
    
    df_errors_test = pd.DataFrame()

    for drug_id in drug_ids:
        # merged_df_i has lower shape
        merged_df_i = merged_df[merged_df["DRUG_ID"]==drug_id]
        
        np.random.seed(123)
        indexes = np.random.permutation(merged_df_i.index)
        train_size = int(merged_df_i.shape[0]*train_ratio)
        indexes_train = indexes[:train_size]
        indexes_test= indexes[train_size:]
        
        if scaling:
            train = merged_df_i.loc[indexes_train, X_columns].copy()
            test = merged_df_i.loc[indexes_test, X_columns].copy()
            scaler = MinMaxScaler()
            scaler.fit(train[columns_for_normalisation])
            train[columns_for_normalisation] = scaler.transform(train[columns_for_normalisation])
            X_train = train.values  
            test[columns_for_normalisation] = scaler.transform(test[columns_for_normalisation])
            X_test = test.values
        else:
            X_train = merged_df_i.loc[indexes_train, X_columns].values
            X_test = merged_df_i.loc[indexes_test, X_columns].values
    
        for i in range(number_coefficients):
#             param = best_param[i+1]
            y_train = merged_df_i.loc[indexes_train, "param_"+str(i+1)].values
            y_test = merged_df_i.loc[indexes_test, "param_"+str(i+1)].values
            
            #check whether each coefficient needs its own parameters
            if type(cache_size)==dict:
                cache_size_value = cache_size[i+1]
            else:
                cache_size_value = cache_size
            
            if type(epsilon)==dict:
                epsilon_value = epsilon[i+1]
            else:
                epsilon_value = epsilon
                
            if type(C)==dict:
                C_value = C[i+1]
            else:
                C_value = C
                
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
                
            kr_lin = SVR(kernel = kernel, C = C_value,
                                       epsilon = epsilon_value,
                                       cache_size=cache_size_value,
                                       gamma = gamma_value,
                                       degree = degree_value, 
                                       coef0 = coef0_value)
            
            kr_lin.fit(X_train, y_train)
            y_pred = kr_lin.predict(X_test)
                                
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

column_not_to_use = ["Unnamed: 0", "COSMIC_ID", "DRUG_ID", "Drug_Name", "Synonyms", "Target", 
                     "deriv_found", "PubChem_ID", "elements", "inchi_key", "canonical_smiles", 
                     "inchi_string", "molecular_formula", "Target",
                     "third_target", "first_target", "second_target", "Target_Pathway"]

param1 = ["param_" +str(i) for i in range(10)]
param2 = ["param" +str(i) for i in range(10)] 
norm_response  = ["norm_cells_"+str(i) for i in range(10)]

### 1. Finding optimal parameters for just drug profiles and cell lines

print("\n1. Finding optimal parameters for just drug profiles and cell lines\n")
df = pd.read_csv(_FOLDER+'merged_fitted_sigmoid4_123_with_drugs_description.csv')

conc_columns= ["fd_num_"+str(i) for i in range(10)]
response_norm = ['norm_cells_'+str(i) for i in range(10)]

gr = df.groupby(["DRUG_ID"])["COSMIC_ID"].count()
drug_ids = list(gr[gr > 50].index)
print("Number of drugs for training:", len(drug_ids))

kernels_to_test = ["linear", "sigmoid", "poly", "rbf"]
results = TuneParameters(df, drug_ids, 4, kernels = kernels_to_test, 
                         column_not_to_use=column_not_to_use, print_results=False)

print("Tuned parameters:")
print(results)
print("\nBetter presentation:")
for key in results:
    print(key,"\t", results[key])

best_kernels, compared_means = TrainTestBestParameters(df, drug_ids, 4, kernels = kernels_to_test, 
                                                       column_not_to_use=column_not_to_use, 
                                                       best_parameters_dict = results, print_results=True)
print("Best Kernels:", best_kernels)
compared_means.to_csv(_FOLDER+"svr_learning_1.csv")
print("\ncompared_means\n")
print(compared_means)
      

### 2. Finding optimal parameters for drug profiles, cell lines and drug description

print("\n2. Finding optimal parameters for drug profiles, cell lines and drug description\n")
df = pd.read_csv(_FOLDER+'merged_fitted_sigmoid4_123_with_drugs_description.csv')

# OHE and dumnies columns for Target_Pathway - 21 new columns
df = pd.concat([df, pd.get_dummies(df["Target_Pathway"])], axis=1)

conc_columns= ["fd_num_"+str(i) for i in range(10)]
response_norm = ['norm_cells_'+str(i) for i in range(10)]

gr = df.groupby(["DRUG_ID"])["COSMIC_ID"].count()
drug_ids = list(gr[gr > 50].index)
print("Number of drugs for training:", len(drug_ids))

kernels_to_test = ["linear", "sigmoid", "poly", "rbf"]
results = TuneParameters(df, drug_ids, 4, kernels = kernels_to_test, 
                         column_not_to_use=column_not_to_use, print_results=False)

print("Tuned parameters:")
print(results)
print("\nBetter presentation:")
for key in results:
    print(key,"\t", results[key])

best_kernels, compared_means = TrainTestBestParameters(df, drug_ids, 4, kernels = kernels_to_test, 
                                                       column_not_to_use=column_not_to_use, 
                                                       best_parameters_dict = results, print_results=True)
print("Best Kernels:", best_kernels)
compared_means.to_csv(_FOLDER+"svr_learning_2.csv")

print("\ncompared_means\n")
print(compared_means)

### 3. Finding optimal parameters for drug profiles, cell lines and drug features

print("\n3. Finding optimal parameters for drug profiles, cell lines and drug features\n")
df = pd.read_csv(_FOLDER+'merged_fitted_sigmoid4_123_with_drugs_properties.csv')

gr = df.groupby(["DRUG_ID"])["COSMIC_ID"].count()
drug_ids = list(gr[gr > 50].index)
print("Number of drugs for training:", len(drug_ids))

kernels_to_test = ["linear", "sigmoid", "poly", "rbf"]
results = TuneParameters(df, drug_ids, 4, kernels = kernels_to_test, 
                         column_not_to_use=column_not_to_use, print_results=False)

print("Tuned parameters:")
print(results)
print("\nBetter presentation:")
for key in results:
    print(key,"\t", results[key])

best_kernels, compared_means = TrainTestBestParameters(df, drug_ids, 4, kernels = kernels_to_test, 
                                                       column_not_to_use=column_not_to_use, 
                                                       best_parameters_dict = results, print_results=True)
print("Best Kernels:", best_kernels)
compared_means.to_csv(_FOLDER+"svr_learning_3.csv")
print("\ncompared_means\n")
print(compared_means)

### 4. Finding optimal parameters for drug profiles, cell lines and drug features with SCALING

print("\n4. Finding optimal parameters for drug profiles, cell lines and drug features with scaling\n")
df = pd.read_csv(_FOLDER+'merged_fitted_sigmoid4_123_with_drugs_properties.csv')

potential_columns_for_normalisation = []
for col in df.columns:
    if (df[col].nunique()>2) & (df[col].dtype != "O"):
        potential_columns_for_normalisation.append(col)

columns_for_normalisation = list(set(potential_columns_for_normalisation) - set(norm_response) - set(param1) - set(param2) -set(['Unnamed: 0', 'DRUG_ID', 'COSMIC_ID',]))
gr = df.groupby(["DRUG_ID"])["COSMIC_ID"].count()
drug_ids = list(gr[gr > 50].index)
print("Number of drugs for training:", len(drug_ids))

kernels_to_test = ["linear", "sigmoid", "poly", "rbf"]
results = TuneParameters(df, drug_ids, 4, kernels = kernels_to_test, column_not_to_use=column_not_to_use, 
                         features_to_scale=columns_for_normalisation, scaling = True,
                         print_results=False)

print("Tuned parameters:")
print(results)
print("\nBetter presentation:")
for key in results:
    print(key,"\t", results[key])

best_kernels, compared_means = TrainTestBestParameters(df, drug_ids, 4, kernels = kernels_to_test, 
                                                       column_not_to_use=column_not_to_use, 
                                                       best_parameters_dict = results, 
                                                       features_to_scale=columns_for_normalisation, scaling = True,
                                                       print_results=True)
print("Best Kernels:", best_kernels)
compared_means.to_csv(_FOLDER+"svr_learning_4.csv")
print("\ncompared_means\n")
print(compared_means)