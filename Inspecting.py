import Cleaning
import pandas as pd

Train, Test = Cleaning.get_data()

# Joining Train and Test to be processed together
train_obj_num = len(Train)
if Cleaning.check_compatibility(Train.drop(columns='SalePrice'), Test):
    dataset = pd.concat(objs=[Train.drop(columns='SalePrice'), Test], axis=0)

# Number of missing values per column
ColNull = Cleaning.col_null(data=dataset, verbose=True)

# Dropping columns with too many null values
dataset = Cleaning.drop_null(data=dataset, thresh=82, axis=1, verbose=True)

cat, num = Cleaning.col_type(dataset)
for c in dataset[num]:
    if dataset[c].unique().shape[0] in range(10000):
        print(c, dataset[c].unique().shape[0])