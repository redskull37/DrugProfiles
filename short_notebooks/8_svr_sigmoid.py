import numpy as np
import pandas as pd
from scipy.stats import norm
import os
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.svm import SVR
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
# import time
from sklearn.model_selection import GridSearchCV

### Average sigmoid

# Main idea: get the DataFrame:
#         'DRUG_ID', 'COSMIC_ID', 'Drug_Name', x1 = conc_1, y1 = response_norm_1
#         'DRUG_ID', 'COSMIC_ID', 'Drug_Name', x2, y2
#         ....
#         'DRUG_ID', 'COSMIC_ID', 'Drug_Name', x10, y10
# Train non-linear regression (of sigmoid type) to obtain a unified/"average" functional dependence between x and y

# So, the first step is to split and concat the new dataset

train_123 = pd.read_csv("results/train08_merged_fitted_sigmoid4_123_with_drugs_properties.csv").drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
test_123 = pd.read_csv("results/test02_merged_fitted_sigmoid4_123_with_drugs_properties.csv").drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)


norm_response  = ["norm_cells_"+str(i) for i in range(10)]
conc_columns  = ["fd_num_"+str(i) for i in range(10)]

col_not_to_use = ["Unnamed: 0", "Unnamed: 0.1", 'DRUG_ID', 'COSMIC_ID', 'Drug_Name', 
                  "Synonyms", "Target", "deriv_found", "PubChem_ID","elements", "inchi_key",
                  "canonical_smiles", "inchi_string", "molecular_formula",
                  "third_target", "first_target", "second_target", "Target_Pathway"]

columns_to_use = ['DRUG_ID', 'COSMIC_ID', 'Drug_Name'] + list(set(train_123.columns)
                                                              -set(norm_response)-set(conc_columns)
                                                              -set(col_not_to_use))
train = np.zeros([1, len(columns_to_use)+2])
for i in range(10):
    train = np.vstack((train, train_123[columns_to_use + ["fd_num_"+str(i), "norm_cells_"+str(i)]].values))

train_df_full = pd.DataFrame(data= train[1:,:], columns = columns_to_use + ["x_conc", "y_response"]).fillna(0)
train_df_full["x_conc"]= np.float32(train_df_full["x_conc"])
train_df_full["y_response"]= np.float32(train_df_full["y_response"])

# test = np.zeros([1, len(columns_to_use)+2])
# for i in range(10):
#     test = np.vstack((test, test_123[columns_to_use + ["fd_num_"+str(i), "norm_cells_"+str(i)]].values))

# test_df_full = pd.DataFrame(data= test[1:,:], columns = columns_to_use + ["x_conc", "y_response"]).fillna(0)
# test_df_full["x_conc"]= np.float32(test_df_full["x_conc"])
# test_df_full["y_response"]= np.float32(test_df_full["y_response"])

int_columns = []
float_columns = []
object_columns = []
for col in train_123.columns:
    if train_123[col].dtype == "int64":
        int_columns.append(col)
    elif (train_123[col].dtype == "float64") | (train_123[col].dtype == "float32"):
        float_columns.append(col)
    else:
        object_columns.append(col)

float_columns2 = list(set(float_columns) - set(norm_response) - set(conc_columns)-set(["param_"+str(i) for i in range(1,5)]))

%%time
for col in int_columns[4:]:
    train_df_full[col] = np.int32(train_df_full[col])
#     test_df_full[col] = np.int32(test_df_full[col])
for col in float_columns2:
    train_df_full[col] = np.float32(train_df_full[col])
#     test_df_full[col] = np.float32(test_df_full[col])

### Support Vector Regression 

columns_for_normalisation = ['molecular_weight','rotatable_bond_count', 'h_bond_acceptor_count',
 'undefined_atom_stereo_count', 'bond_stereo_count', 'defined_atom_stereo_count',
 'complexity', 'atom_stereo_count','covalent_unit_count','2bonds',
 'surface_area', 'xlogp', 'heavy_atom_count', "x_conc"]

scaler = MinMaxScaler()
scaler.fit(train_df_full[columns_for_normalisation])
train_df_full[columns_for_normalisation] = scaler.transform(train_df_full[columns_for_normalisation])

X_columns = train_df_full.columns[3:-1]
X = train_df_full[X_columns].values
y = train_df_full["y_response"].values  

tuned_parameters = [{'kernel': ['rbf'], 
                                'gamma': [1e-3, 1e-4, 1e-5], 
                                'C': [1, 10, 100, 1000]},
                    {'kernel': ['sigmoid'], 
                              'gamma': [1e-3, 1e-4, 1e-5], 'C': [1, 10, 100, 1000], 
                              "epsilon": [0.001, 0.01, 0.1, 1]},
                    {'kernel': ['linear'], 
                               'C': [1, 10, 100, 1000]}]

model_search = GridSearchCV(SVR(), tuned_parameters, scoring="neg_mean_absolute_error")
model_search.fit(X, y)

print("Best parameters set found on development set:\n")
print(model_search.best_params_)