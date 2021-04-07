import os
import numpy as np
import torch


class Dataset:
    def __init__(self, data_path, name="weather", max_size_test=3000, max_samples=50000):
        self.data_path = data_path
        self.data_arr = np.load(os.path.join(data_path, "raw_data.npy")) if os.path.exists(
            os.path.join(data_path, "raw_data.npy")) else None
        self.train_path = os.path.join(data_path, "train")
        self.val_path = os.path.join(data_path, "val")
        self.test_path = os.path.join(data_path, "test")
        self.name = name
        self.max_samples = max_samples
        self.max_size_test = max_size_test
        self.means = np.load(os.path.join(data_path, "means.npy"))
        self.stds = np.load(os.path.join(data_path, "stds.npy"))

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

    def get_datasets(self, shuffle_test=False):
        type = np.float32
        train_data = self.get_data_from_folder(self.train_path)
        train_data = train_data.astype(type)
        val_data = self.get_data_from_folder(self.val_path)
        val_data = val_data.astype(type)
        test_data = self.get_data_from_folder(self.test_path)
        if self.max_samples is not None:
            if train_data.shape[0] > self.max_samples:
                train_data = train_data[:self.max_samples]  # to avoid memory issues at test time.
                print("reducing train dataset size to {} samples...".format(self.max_samples))
        if test_data.shape[0] > self.max_size_test:
            if self.max_size_test == 1 and shuffle_test:
                index = np.random.randint(test_data.shape[0])
                print("index sample for test data", index)
                test_data = test_data[index]
                test_data = test_data[np.newaxis, :, :]
                self.index_test = index
            else:
                test_data = test_data[:self.max_size_test]  # to avoid memory issues at test time.
                self.index_test = None
            print("reducing test dataset size to {} samples...".format(self.max_size_test))
        test_data = test_data.astype(type)
        return train_data, val_data, test_data

    def get_features_labels(self, train_data, val_data, test_data):
        x_train, y_train = self.split_fn(train_data)
        x_val, y_val = self.split_fn(val_data)
        x_test, y_test = self.split_fn(test_data)
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def check_dataset(self, dataloader):
        inp, tar = next(iter(dataloader))
        if inp.shape == 4:
            assert inp[:, :, 1:, :] == tar[:, :, :-1, :], "error in inputs/targets of dataset"
        elif inp.shape == 3:
            assert inp[:, 1:, :] == tar[:, :-1, :], "error in inputs/targets of dataset"

    def data_to_dataset(self, args, num_dim=4):
        train_data, val_data, test_data = self.get_datasets(shuffle_test=args.shuffle_test)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.get_features_labels(train_data=train_data,
                                                                                        val_data=val_data,
                                                                                        test_data=test_data)
        if num_dim == 4:
            # adding the particle dim:
            X_train = X_train[:, np.newaxis, :, :]  # (B,1,S,F)
            y_train = y_train[:, np.newaxis, :, :]
            X_val = X_val[:, np.newaxis, :, :]
            y_val = y_val[:, np.newaxis, :, :]
            X_test = X_test[:, np.newaxis, :, :]
            y_test = y_test[:, np.newaxis, :, :]

        X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
        X_val, y_val = torch.tensor(X_val).float(), torch.tensor(y_val).float()
        X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=args.bs, drop_last=True)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        batch_size_test = args.bs if args.bs_test is None else args.bs_test
        dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, drop_last=True)
        self.seq_len = X_train.shape[-2]
        self.output_size = y_train.shape[-1]
        self.input_size = X_train.shape[-1]
        self.num_train_samples = X_train.shape[0]
        return dataloader_train, dataloader_val, dataloader_test


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
