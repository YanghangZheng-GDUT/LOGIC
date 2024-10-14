import logging
import sys
import warnings
import numpy as np
import torch
import os
import random
from utils.Dataset import Dataset
from torch import optim
from trainer import Trainer
from utils.log import log
from utils.Config import load_net_para
from model_vae import Multi_VAE
from utils.plot_pic import plot_picture

warnings.filterwarnings("ignore")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def main(data_name, para_rec, para_kl, para_cont, para_tmp, exist_ratio=0.5, run_time=0):
    logger = log('logs', data_name, exp_id=str(run_time), is_cover=None)
    logging.info('log is saved in logs/{}_{}.log'.format(data_name, run_time))
    logging.info('son_thread is {}, main_thread is {}'.format(os.getpid(), os.getppid()))
    data = Dataset(data_name)
    X_list, y_ture = data.load_data()
    size = len(y_ture)
    view_num = len(X_list)

    setup_seed(0)

    lr = 5e-4

    L_dims, latent_dim, args = load_net_para(data_name)
    X_list, y_ture, W, X_comp = data.Create_incomplete_data(X_list, y_ture, exist_ratio)

    for v in range(len(X_list)):
        X_list[v] = torch.from_numpy(X_list[v]).float()
        X_comp[v] = torch.from_numpy(X_comp[v]).float()
    W = torch.from_numpy(W).float()

    miss_ids = torch.logical_not(W[:, 0].bool()).nonzero().squeeze()[:10]
    plot_pic = plot_picture(data_name)
    plot_pic.load_data(X_comp[0][miss_ids], title='Origin')


    use_cuda = torch.cuda.is_available()
    print("cuda is available?")
    print(use_cuda)
    args['Use_cuda'] = use_cuda
    if use_cuda:
        args['device'] = 'cuda'

    args['para_rec'] = para_rec
    args['para_kl'] = para_kl
    args['para_cont'] = para_cont
    args['para_tmp'] = para_tmp


    logging.info(args)
    logging.info('exist_rate:{}'.format(exist_ratio))


    model = Multi_VAE(size=size, view_num=view_num, latent_dim=latent_dim, args=args, L_dims=L_dims)

    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    if use_cuda:
        model.cuda()
        tmp_list = []
        for v in range(len(X_list)):
            tmp_list.append(X_list[v].to('cuda'))
        X_list = tmp_list
        W = W.to('cuda')
    trainer = Trainer(model, optimizer, data, X_list, y_ture, W, args)

    trainer.pre_train()
    X_list, W = trainer.model.complete_X(trainer.X_list, trainer.W)

    plot_pic.load_data(X_list[0][miss_ids], title='Recovered')
    plot_pic.plot()

    model = Multi_VAE(size=size, view_num=view_num, latent_dim=latent_dim, args=args, L_dims=L_dims)
    if use_cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, optimizer, data, X_list, y_ture, W, args)
    trainer.cycle_train()

    return

if __name__ == '__main__':

    main('Yale15_3_165', 1, 0.001, 0.001, 0.5, exist_ratio=0.5)
    sys.exit()
