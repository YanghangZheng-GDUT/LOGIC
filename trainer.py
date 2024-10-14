import logging
import os

import scipy
import torch
from Clustering import clustering
import numpy as np

class Trainer():
    def __init__(self, model, optimizer, data, X_list, y_ture, W, args):
        self.model = model
        self.optimizer = optimizer
        self.data = data
        self.data_loader, self.test_loader = self.data.Get_dataloaders(X_list, y_ture, W, batch_size=args['constant']['Batch_size'])
        self.args = args
        self.W = W
        self.X_list = X_list
        self.y_ture = y_ture
        self.size = len(y_ture)
        self.view_num = len(X_list)
        self.n_clusters = len(set(y_ture))

    def pre_train(self):

        logging.info('Begin initial training!')

        self.data_loader, _ = self.data.Get_dataloaders(self.X_list, self.y_ture, self.W,
                                                     batch_size=self.args['constant']['p_Batch_size'])

        res_list = []
        loss_epoch_list = []
        for e in range(self.args['constant']['p_epochs']):
            loss_batch_list = []
            for batch_idx, Data in enumerate(self.data_loader):
                xs_list = Data[0:-2]
                labels = Data[-2]
                w = Data[-1]

                self.optimizer.zero_grad()
                loss_rec, loss_kl, n_fea_v_tensor = self.model(xs_list, w)
                loss_cont = self.model.contrastive_loss(n_fea_v_tensor)
                loss_total = self.args['para_rec'] * loss_rec + self.args['para_kl'] * loss_kl \
                             + self.args['para_cont'] * loss_cont

                loss_batch_list.append(loss_total.detach().cpu().numpy())
                loss_total.backward()
                self.optimizer.step()

            if e % self.args['constant']['show_result_epoch'] == 0 or e == self.args['constant']['epochs'] - 1:
                res, _ = self.test()
                logging.info(
                    'epoch: {}, loss:{:.5f}'.format(e, loss_total.detach().cpu().data))


            loss_epoch_list.append(np.round(np.mean(loss_batch_list), 4))
            res_list.append([res['ACC'], res['NMI'], res['PUR'], res['AR'], res['precision'], res['recall']])

        return res, res_list, loss_epoch_list

    def cycle_train(self):

        self.data_loader, _ = self.data.Get_dataloaders(self.X_list, self.y_ture, self.W,
                                                     batch_size=self.args['constant']['Batch_size'])

        loss_epoch_list = []
        res_list = []
        logging.info('Begin cycle training!')
        for e in range(self.args['constant']['epochs']):
            loss_batch_list = []
            for batch_idx, Data in enumerate(self.data_loader):
                xs_list = Data[0:-2]
                labels = Data[-2]
                w = Data[-1]

                self.optimizer.zero_grad()
                loss_rec, loss_kl, n_fea_v_tensor = self.model(xs_list, w)
                loss_cont = self.model.contrastive_loss(n_fea_v_tensor)


                loss_total = self.args['para_rec']*loss_rec + self.args['para_kl']*loss_kl \
                                 + self.args['para_cont']*loss_cont

                loss_batch_list.append(loss_total.detach().cpu().numpy())
                loss_total.backward()
                self.optimizer.step()

            if e % self.args['constant']['show_result_epoch'] == 0 or e == self.args['constant']['epochs'] - 1:
                res, confusion_matrix = self.test()
                logging.info(
                    'epoch: {}, res:{}, loss:{:.5f}'.format(e, res, loss_total.detach().cpu().data))

            loss_epoch_list.append(np.round(np.mean(loss_batch_list), 4))
            res_list.append([res['ACC'], res['NMI'], res['PUR'], res['AR'], res['precision'], res['recall']])
        return res, res_list, loss_epoch_list

    def test(self):
        with torch.no_grad():
            z_means_list = []
            for batch_idx, Data in enumerate(self.test_loader):
                xs_list = Data[0:-2]
                w = Data[-1]
                if self.args['Use_cuda']:
                    xs_list = [item.cuda() for item in xs_list]
                    w = w.cuda()
                z_means, feas = self.model.encode(xs_list, w)
                z_means_list.append(z_means.detach().cpu())
            z_means = torch.cat(z_means_list, dim=0)

            # z_means, feas = self.model.encode(self.X_list, self.W)
            y_pred, ret, confusion_matrix = clustering(z_means.detach().cpu().data.numpy(), self.y_ture)

        return ret['kmeans'], confusion_matrix






