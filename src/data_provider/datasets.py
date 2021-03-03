import os
import numpy as np
from src.preprocessing.utils import split_synthetic_dataset
import torch

class Dataset:
    def __init__(self, data_path, BATCH_SIZE=32, name="synthetic", model=None, BUFFER_SIZE=500, target_features=None, max_size_test=3000, max_samples=None):
        self.data_path = data_path
        self.data_arr = np.load(os.path.join(data_path, "raw_data.npy")) if os.path.exists(os.path.join(data_path, "raw_data.npy")) else None
        self.train_path = os.path.join(data_path, "train")
        self.val_path = os.path.join(data_path, "val")
        self.test_path = os.path.join(data_path, "test")
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.name = name
        self.model = model
        self.max_samples = max_samples
        self.target_features = list(range(self.get_data_from_folder(self.train_path).shape[-1])) if target_features is None else target_features
        self.max_size_test = max_size_test

    def split_fn(self, chunk):
        input_text = chunk[:, :-1, :]
        target_text = chunk[:, 1:, :]
        return input_text, target_text

    def get_data_from_folder(self, folder_path, extension=".npy"):
        files = []
        for file in os.listdir(folder_path):
            if file.endswith(extension):
                files.append(file)
        file_path = os.path.join(folder_path, files[0])
        data = np.load(file_path)
        return data

    def get_datasets(self):
        type = np.float32
        train_data = self.get_data_from_folder(self.train_path)
        train_data = train_data.astype(type)
        val_data = self.get_data_from_folder(self.val_path)
        val_data = val_data.astype(type)
        test_data = self.get_data_from_folder(self.test_path)
        if self.max_samples is not None:
            if train_data.shape[0] > self.max_samples:
                train_data = train_data[:self.max_samples] # to avoid memory issues at test time.
                print("reducing train dataset size to {} samples...".format(self.max_samples))
        if test_data.shape[0] > self.max_size_test:
            test_data = test_data[:self.max_size_test] # to avoid memory issues at test time.
            print("reducing test dataset size to {} samples...".format(self.max_size_test))
        test_data = test_data.astype(type)
        return train_data, val_data, test_data

    def get_features_labels(self, train_data, val_data, test_data):
        x_train, y_train = self.split_fn(train_data)
        x_val, y_val = self.split_fn(val_data)
        x_test, y_test = self.split_fn(test_data)
        y_train = y_train[:, :, self.target_features]
        y_val = y_val[:, :, self.target_features]
        y_test = y_test[:, :, self.target_features]
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def check_dataset(self, dataset):
        for (inp, tar) in dataset.take(1):
            if inp.shape == 4:
                assert inp[:,:,1:,self.target_features] == tar[:,:,:-1,self.target_features], "error in inputs/targets of dataset"
            elif inp.shape == 3:
                assert inp[:, 1:, self.target_features] == tar[:, :-1, self.target_features], "error in inputs/targets of dataset"

    def get_data_splits_for_crossvalidation(self, TRAIN_SPLIT=0.8, VAL_SPLIT_cv=0.9):
        list_train_data, list_val_data, test_data = split_synthetic_dataset(x_data=self.data_arr,
                                                                            TRAIN_SPLIT=TRAIN_SPLIT,
                                                                            VAL_SPLIT_cv=VAL_SPLIT_cv, cv=True)
        list_test_data = [test_data] * len(list_train_data)
        return list_train_data, list_val_data, list_test_data

    def data_to_dataset(self, train_data, val_data, test_data, args):
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.dataset.get_features_labels(train_data=train_data,
                                                                                                val_data=val_data,
                                                                                                test_data=test_data)
        X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
        X_val, y_val = torch.tensor(X_val).float(), torch.tensor(y_val).float()
        X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs)
        self.seq_len = X_train.shape[1]
        self.output_size = y_train.shape[-1]
        self.num_features = X_train.shape[-1]
        self.num_train_samples = X_train.shape[0]
        return dataloader_train, dataloader_val, dataloader_test

    def create_torch_datasets(self, args):
        if not args.cv:
            train_data, val_data, test_data = self.dataset.get_datasets()
            dataloader_train, dataloader_val, dataloader_test = self.data_to_dataset(train_data=train_data,
                                                                                     val_data=val_data,
                                                                                     test_data=test_data, args=args)
            return dataloader_train, dataloader_val, dataloader_test
        else:
            train_datasets, val_datasets, test_datasets = self.get_dataset_for_crossvalidation(args=args)
            return train_datasets, val_datasets, test_datasets

    def get_dataset_for_crossvalidation(self, args):
        list_train_data, list_val_data, list_test_data = self.dataset.get_data_splits_for_crossvalidation()
        train_datasets, val_datasets, test_datasets = [], [], []
        for train_data, val_data, test_data in zip(list_train_data, list_val_data, list_test_data):
            train_dataset, val_dataset, test_dataset = self.data_to_dataset(train_data=train_data, val_data=val_data,
                                                                            test_data=test_data, args=args)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_datasets.append(test_dataset)
        return train_datasets, val_datasets, test_datasets[0]


class StandardizedDataset(Dataset):
    def __init__(self, data_path, BATCH_SIZE, BUFFER_SIZE=50, name="weather", model=None, target_features=None, max_samples=None):
        super(StandardizedDataset, self).__init__(data_path=data_path, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE, name=name, model=model, target_features=target_features, max_samples=max_samples)
        self.means = np.load(os.path.join(data_path, "means.npy"))
        self.stds = np.load(os.path.join(data_path, "stds.npy"))

    def rescale_data(self):
        pass


if __name__ == '__main__':

    synthetic_dataset = Dataset(data_path='../../data/synthetic_model_1', BUFFER_SIZE=50, BATCH_SIZE=64)
    train_data, val_data, test_data = synthetic_dataset.get_datasets()
    print('train data shape', train_data.shape)
    print('val data shape', val_data.shape)
    print('test data shape', test_data.shape)

    # ----------------------------------------------- test of data_to_dataset function ----------------------------------
    train_dataset, val_dataset, test_dataset = synthetic_dataset.data_to_dataset(train_data=train_data,
                                                                                 val_data=val_data, test_data=test_data)
    for (inp, tar) in train_dataset.take(1):
        print('input example shape', inp.shape)
        print('input example', inp[0])
        print('target example shape', tar.shape)
        print('target example', tar[0])

    print("3D dataset........")

    train_dataset, val_dataset, test_dataset = synthetic_dataset.data_to_dataset(train_data=train_data,
                                                                                 val_data=val_data, test_data=test_data,
                                                                                 num_dim=3)
    print(synthetic_dataset.target_features)
    synthetic_dataset.check_dataset(train_dataset)
    for (inp, tar) in train_dataset.take(1):
        print('input example shape', inp.shape)
        print('input example', inp[0])
        print('target example shape', tar.shape)
        print('target example', tar[0])

    # ---------------------------------------------------- test standardized dataset --------------------------------------
    target_features = list(range(5))
    air_quality_dataset = StandardizedDataset(data_path="../../data/air_quality", BATCH_SIZE=32, BUFFER_SIZE=500, name="air_quality", target_features=target_features)
    train_data, val_data, test_data = air_quality_dataset.get_datasets()
    train_dataset, val_dataset, test_dataset = air_quality_dataset.data_to_dataset(train_data=train_data, val_data=val_data, test_data=test_data, num_dim=4)
    for (inp, tar) in train_dataset.take(1):
        print('input example shape', inp.shape)
        print('input example', inp[0,:,:,0])
        print('target example shape', tar.shape)
        print('target example', tar[0,:,:,0])



