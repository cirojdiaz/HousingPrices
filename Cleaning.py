# Data processing
import pandas as pd


# reading data as DataFrame
def get_data():
    Train = pd.read_csv('train.csv')
    Test = pd.read_csv('test.csv')
    Id_of_null_outputs = Train[Train['SalePrice'].isnull()].index.tolist()
    Train.drop(labels=Id_of_null_outputs, axis=0, inplace=True)
    # train_data = Train.drop(['Id'], axis=1)
    # target = Train['SalePrice']
    # train_data = Train.drop(['SalePrice'], axis=1, inplace=False)
    # test_data = Test
    return Train, Test


#  null values per column DICT
def col_null(data: pd.DataFrame, verbose: bool):
    ColNull = {}
    for col in data:
        ColNull[col] = sum(data[col].isnull())

    # ordering dict by the # of null values
    ColNull = dict(sorted(ColNull.items(), key=lambda item: item[1]))
    if verbose:
        for col in ColNull:
            print(col, 'Null Values: ', ColNull[col])
    return ColNull


def drop_null(data: pd.DataFrame, mirror=None, thresh=1, axis=1, verbose=False):
    if axis == 1:
        ColNull = col_null(data=data, verbose=False)
        cols = [key for key, value in ColNull.items() if value > thresh]
        data.drop(cols, axis=1, inplace=True)
        if verbose:
            print(len(cols), 'Columns eliminated', cols)
        if mirror is None:
            return data
        else:
            mirror.drop(cols, axis=1, inplace=True)
    elif axis == 0:
        data.dropna(axis=0, inplace=True)
    return data, mirror


# Spliting data into Train Test and Target
def ttt_sets(Train: pd.DataFrame, Test: pd.DataFrame):
    target = Train['SalePrice']
    Train.drop(['Id', 'SalePrice'], axis=1, inplace=True)
    Test.drop(['Id'], inplace=True, axis=1)
    return Train, Test, target


def col_type(Data: pd.DataFrame):
    cat = []
    num = []
    for row in Data:
        if Data[row].dtype == 'object':
            cat.append(row)
        elif Data[row].dtype == 'int64':
            num.append(row)
    return cat, num


def check_compatibility(train: pd.DataFrame, test: pd.DataFrame, verbose=False):
    cat_1, num_1 = col_type(train)
    cat_2, num_2 = col_type(test)
    # Verify num of categorical and numerical rows are the same
    cat_dim = len(cat_1) == len(cat_2)
    num_dim = len(num_2) == len(num_1)
    if not cat_dim or not num_dim:
        E1 = 'The number of Categorical and Numeric features is different'
        if verbose:
            print(E1)
        return E1
    else:
        feature_match = cat_1.sort() == cat_2.sort() and num_1.sort() == num_2.sort()
        if not feature_match:
            E2 = ['Categorical and Numerical features mismatch', (cat_1, cat_2), (num_2, num_1)]
            if verbose:
                print(E2)
            return E2
    if verbose:
        print('Train and Test are fully compatible')
    return True
