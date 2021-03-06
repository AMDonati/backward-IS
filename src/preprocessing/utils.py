import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np

def convert_col_into_float64(df, list_cols):
    for col in list_cols:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace(',', '.')
        df[col] = df[col].astype(float)
    return df

def split_array_per_sequences(array, history=12):
    new_array = np.zeros(shape=(int(array.shape[0] / history), history, array.shape[-1]))
    for i, j in enumerate(list(range(0, array.shape[0], history))):
        if j + history >= array.shape[0]:
            break
        new_array[i] = array[j:j + history, :]
        print(j)
    return new_array

def get_array_per_step(array, step=1):
    selected_index = list(range(0, array.shape[0], step))
    new_array = array[selected_index]
    return new_array


def getIndexes(dfObj, value):
    ''' Get index positions of value in dataframe i.e. dfObj.'''
    listOfPos = list()
    # Get bool dataframe with True at positions where the given value exists
    result = dfObj.isin([value])
    # Get list of columns that contains the value
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))
    # Return a list of tuples indicating the positions of value in the dataframe
    list_rows = [r for (r, _) in listOfPos]
    list_cols = [c for (_, c) in listOfPos]
    return listOfPos, list_rows, list_cols


def fill_missing_values(df, list_cols, value=-200):
    reduced_list_of_pos, rows_reduced, _ = getIndexes(df[list_cols], value=value)
    rows_reduced = list(set(rows_reduced))
    for col in list_cols:
        df_tmp = df[col].drop(index=rows_reduced, axis=0)
        mean = np.mean(df_tmp)
        df[col].iloc[rows_reduced] = mean
    return df, rows_reduced


def get_rows_nan_values(df):
    rows_with_nan = []
    for index, row in df.iterrows():
        is_nan_series = row.isnull()
        if is_nan_series.any():
            rows_with_nan.append(index)
    return rows_with_nan


def split_dataset_into_seq(dataset, start_index, end_index, history_size, step):
    data = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset)
    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])
    print("final i", i)
    return np.array(data)


def split_synthetic_dataset(x_data, TRAIN_SPLIT, save_path=None, VAL_SPLIT=0.5, VAL_SPLIT_cv=0.9, cv=False):
    if not cv:
        train_data, val_test_data = train_test_split(x_data, train_size=TRAIN_SPLIT, shuffle=True, random_state=123)
        val_data, test_data = train_test_split(val_test_data, train_size=VAL_SPLIT, shuffle=True, random_state=123)
        if save_path is not None:
            train_data_path = os.path.join(save_path, "train")
            val_data_path = os.path.join(save_path, "val")
            test_data_path = os.path.join(save_path, "test")
            for path in [train_data_path, val_data_path, test_data_path]:
                if not os.path.isdir(path):
                    os.makedirs(path)
            np.save(os.path.join(train_data_path, "synthetic.npy"), train_data)
            np.save(os.path.join(val_data_path, "synthetic.npy"), val_data)
            np.save(os.path.join(test_data_path, "synthetic.npy"), test_data)
            print("saving train, val, and test data into .npy files...")
        return train_data, val_data, test_data
    else:
        train_val_data, test_data = train_test_split(x_data, train_size=VAL_SPLIT_cv)
        kf = KFold(n_splits=5)
        list_train_data, list_val_data = [], []
        for train_index, val_index in kf.split(train_val_data):
            train_data = train_val_data[train_index, :, :]
            val_data = train_val_data[val_index, :, :]
            list_train_data.append(train_data)
            list_val_data.append(val_data)

        return list_train_data, list_val_data, test_data
