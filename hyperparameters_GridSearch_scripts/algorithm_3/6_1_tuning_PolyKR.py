import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import MinMaxScaler

np.random.seed(123)

_FOLDER = "/home/acq18mk/master/results/results/"
# _FOLDER = "../drug_results/"

### Coding Part

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
train_df_50 = train_df.set_index("DRUG_ID").loc[drug_ids_50, :].copy()

train_drug = pd.DataFrame()

for i in range(10):
    train_drug = pd.concat([train_drug, train_df_50[["COSMIC_ID", "fd_num_"+str(i), "norm_cells_"+str(i)]+all_columns].rename(
                                    columns={"fd_num_"+str(i): "scaled_x", 
                                             "norm_cells_"+str(i): "norm_y"})],
                          axis=0, ignore_index = True) 

X_columns = ["scaled_x"] + ["MAX_CONC"] + X_PubChem_properties +  X_targets + X_target_pathway + X_cancer_cell_lines

scaler = MinMaxScaler().fit(train_drug[X_columns])
Xtrain_drug = scaler.transform(train_drug[X_columns])
y_train_drug = train_drug["norm_y"].values

print("Polynomial KernelRidge")

param_tested_alphas = [5, 10, 50, 100, 500]
param_tested_gamma = [0.00001, 0.0001, 0.01]
param_tested_coef0 = [0.01, 0.1, 1]
param_tested_degree = [2, 3, 4]

param_grid = dict(alpha = param_tested_alphas, gamma = param_tested_gamma, coef0 = param_tested_coef0,
                 degree = param_tested_degree)

splitter_loo = LeaveOneOut()
grid = GridSearchCV(KernelRidge(kernel = "polynomial"), param_grid = param_grid, cv = splitter_loo, scoring= "neg_mean_absolute_error")
grid.fit(Xtrain_drug, y_train_drug)
      
print("Dataset: 4, best alpha:", grid.best_params_["alpha"])
print("Dataset: 4, best_gamma" , grid.best_params_["gamma"])
print("Dataset: 4, best_coef0", grid.best_params_["coef0"])
print("Dataset: 4, best_degree", grid.best_params_["degree"])