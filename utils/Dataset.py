import copy

import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import scipy.io

class Dataset():
    def __init__(self, name):
        self.path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), './..')) + '/dataset/'
        self.name = name

    def load_data(self):



        data_path = self.path + self.name + '.mat'
        dataset = scipy.io.loadmat(data_path)

        if self.name == 'Yale15_3_165':
            data = dataset['data'][0]
            y = dataset['label'][0]
            X = list()
            for i in range(data.shape[0]):
                data[i] = data[i].transpose()
                if i != 0:
                    data[i] = self.normalize(data[i])
                X.append(data[i])
        elif self.name == 'leaves100_3_1600':
            x1, x2, x3 = dataset['data'][0]
            X = [x1, x2, x3]
            for i in range(len(X)):
                X[i] = X[i].transpose()
                X[i] = self.normalize(X[i])
            y = np.squeeze(dataset['label'])
        elif self.name == 'ORL40_3_400':
            data = dataset['data'][0]
            y = dataset['label'][0]
            X = list()
            for i in range(data.shape[0]):
                data[i] = data[i].transpose()
                if i != 0:
                    data[i] = self.normalize(data[i])
                X.append(data[i])
        elif self.name == 'YaleB10_3_650':
            data = dataset['data'][0]
            y = dataset['label'][0]
            X = list()
            for i in range(data.shape[0]):
                data[i] = data[i].transpose()
                if i != 0 :
                    data[i] = self.normalize(data[i])
                X.append(data[i])
        elif self.name == 'Reuters_dim10':  # 18758 samples
            X = []
            X.append(self.normalize(np.vstack((dataset['x_train'][0], dataset['x_test'][0]))))
            X.append(self.normalize(np.vstack((dataset['x_train'][1], dataset['x_test'][1]))))
            # X.append(np.vstack((dataset['x_train'][0], dataset['x_test'][0])))
            # X.append(np.vstack((dataset['x_train'][1], dataset['x_test'][1])))
            y = np.squeeze(np.hstack((dataset['y_train'], dataset['y_test'])))
        elif self.name == 'ALOI100_4_10800':
            x1, x2, x3, x4 = dataset['data'][0]
            x1, x2, x3, x4 = x1.transpose(), x2.transpose(), x3.transpose(), x4.transpose()
            X = [x1, x2, x3, x4]
            for i in range(len(X)):
                X[i] = self.normalize(X[i])
            y = dataset['label'][0]
        else:
            X = None
            y = None

        return X, y
    def normalize(self, x, min=0):
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def Create_incomplete_data(self, X_list, y_ture, exist_ratio):
        sample_num = len(y_ture)
        view_num = len(X_list)
        W = np.ones([sample_num, view_num])
        if int(exist_ratio) == 1:
            return X_list, y_ture, W
        loss_dataset_size = round((sample_num - sample_num * exist_ratio) / view_num)
        paired_dataset_size = sample_num - round((sample_num - sample_num * exist_ratio) / view_num) * view_num

        sample_order = np.random.permutation(sample_num)
        y_ture = y_ture[sample_order]
        X_complete_list = []
        for v in range(view_num):
            X_list[v] = X_list[v][sample_order]
            # if v == 0:
            X_complete_list.append(copy.deepcopy(X_list[v]))

            X_zeros = np.zeros(X_list[v][paired_dataset_size+v*loss_dataset_size:
                                         paired_dataset_size+(v+1)*loss_dataset_size].shape)
            X_list[v] = np.concatenate((X_list[v][:paired_dataset_size+v*loss_dataset_size],
                                        X_zeros,
                                        X_list[v][paired_dataset_size+(v+1)*loss_dataset_size:]))
            W[paired_dataset_size+v*loss_dataset_size:
                         paired_dataset_size+(v+1)*loss_dataset_size, v] = [0 for _ in range(loss_dataset_size)]

        return X_list, y_ture, W, X_complete_list

    def Get_dataloaders(self, X_list, y_ture, W, batch_size):
        from torch.utils.data import DataLoader, TensorDataset
        import torch
        y_ture = torch.from_numpy(y_ture)
        X = TensorDataset(*X_list, y_ture, W)
        train_loader = DataLoader(X, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(X, batch_size=batch_size, shuffle=False, drop_last=False)
        return train_loader, test_loader


