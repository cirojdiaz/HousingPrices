# Main program
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import Cleaning

Train,Test = Cleaning.get_data()

# Joining Train and Test to be processed together
train_obj_num = len(Train)
if Cleaning.check_compatibility(Train.drop(columns='SalePrice'), Test):
    dataset = pd.concat(objs=[Train.drop(columns='SalePrice'), Test], axis=0)

# Dictionary with columns name as keys and number of missing values
ColNull = Cleaning.col_null(data=dataset, verbose=True)

# Dropping columns with too many null values according to threshold
dataset = Cleaning.drop_null(data=dataset, thresh=82, axis=1, verbose=True)

# Encoding categorical variables
cat, num = Cleaning.col_type(dataset)
dataset = pd.get_dummies(dataset, columns=cat, drop_first=True)

# Separating Train and Test
Train = pd.concat(objs=[dataset[:train_obj_num], Train['SalePrice']], axis=1)
Test = dataset[train_obj_num:].copy()

# Drooping Rows with null values
old_len_train = len(Train)
old_len_test = len(Test)
Train.dropna(axis=0, inplace=True)
Test.dropna(axis=0, inplace=True)
print('Train dropped elements', old_len_train-len(Train))
print('Test dropped elements', old_len_test-len(Test))

# getting Train, Test and Target
train_data, test_data, target = Cleaning.ttt_sets(Train=Train, Test=Test)

# Checking compatibility
Cleaning.check_compatibility(train_data, test_data, verbose=True)

# print(train_data.head())
print('Train (Number of features): ', train_data.shape[1])  # Number of features
print('Train (Number of Samples): ', train_data.shape[0])  # Number of Samples
print('')
print('Test (Number of features): ', test_data.shape[1])  # Number of features
print('Test (Number of Samples): ', test_data.shape[0])  # Number of Samples

# Implementing Models
xgb = XGBRegressor(verbosity=0)
cv_score = cross_val_score(xgb, train_data, target, cv=10)
print('CV mean score', cv_score.mean())


