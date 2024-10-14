import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch.autograd import Variable

class LinearAE(nn.Module):
    def __init__(self, dims, latent_dim):
        super(LinearAE, self).__init__()
        self.encoder = nn.Sequential()
        for i in range(1, len(dims)):
            self.encoder.add_module('linear%d' % i, nn.Linear(dims[i-1], dims[i]))
            # self.encoder.add_module('bn%d' % i, nn.BatchNorm1d(dims[i], affine=True))  # bn层
            self.encoder.add_module('relu%d' % i, nn.ReLU())
        self.decoder = nn.Sequential()
        dims = list(reversed(dims))
        self.fea_dim = dims[0]
        if latent_dim is not None:
            self.decoder.add_module('delinear0', nn.Linear(latent_dim, dims[0]))
            self.decoder.add_module('relud0', nn.ReLU())
        for i in range(len(dims)-1):
            self.decoder.add_module('delinear%d' % (i+1), nn.Linear(dims[i], dims[i+1]))
            if i != len(dims)-2:
                self.decoder.add_module('relud%d' % (i+1), nn.ReLU())

        self.decoder.add_module('Sigmoid', nn.Sigmoid())

    def get_fea_dim(self):
        return self.fea_dim
    def encode(self, x):
        fea = self.encoder(x)
        return fea
    def decode(self, latent):
        x = self.decoder(latent)
        return x

class Common_Vae(nn.Module):
    def __init__(self, fea_dim, latent_dim, is_cuda):
        super(Common_Vae, self).__init__()

        self.mean = nn.Linear(fea_dim, latent_dim)
        self.logvar = nn.Linear(fea_dim, latent_dim)

        self.is_cuda = is_cuda

    def encode(self, fea):
        mean = self.mean(fea)
        logvar = self.logvar(fea)
        return mean, logvar

    def z(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = (
            Variable(torch.randn(std.size())).cuda() if self.is_cuda else
            Variable(torch.randn(std.size()))
        )
        return mean + std * eps

    def forward(self, fea):
        mean, logvar = self.encode(fea)
        z = self.z(mean, logvar)
        return mean, logvar, z

class Multi_VAE(torch.nn.Module):

    def __init__(self, size, view_num, latent_dim, args, L_dims=None):
        super(Multi_VAE, self).__init__()
        self.args = args
        self.net_aes = []
        self.size = size  #
        self.view_num = view_num
        self.is_partical_train = False
        if L_dims is not None:
            for i in range(len(L_dims)):
                self.net_aes.append(LinearAE(L_dims[i], latent_dim))

        self.fea_dim = self.net_aes[0].get_fea_dim()
        self.net_aes = nn.ModuleList(self.net_aes)
        self.net_comvae = Common_Vae(self.fea_dim, latent_dim, args['Use_cuda'])


    def forward(self, xs_list, w):

        summ = 0
        feas = []
        for v, ae in enumerate(self.net_aes):
            fea_i = ae.encode(xs_list[v])
            fea_i = torch.diag(w[:, v]).mm(fea_i)
            feas.append(fea_i)
            summ += fea_i
        weight = 1 / torch.sum(w, 1)
        fea_fusion = torch.diag(weight).mm(summ)

        z_mean, z_logvar, z = self.net_comvae(fea_fusion)


        xs_rec = []
        for v, ae in enumerate(self.net_aes):
            x_rec = ae.decode(z)
            x_rec = torch.diag(w[:, v]).mm(x_rec)
            xs_rec.append(x_rec)

        loss_rec_list = []
        for v in range(self.view_num):
            loss_rec_list.append(nn.BCELoss(reduction='mean')(xs_rec[v], xs_list[v]))
        
        loss_kl_list = torch.sum((z_mean ** 2 + z_logvar.exp() - 1 - z_logvar) / 2, 1)

        loss_rec = torch.mean(torch.stack(loss_rec_list))
        loss_kl = torch.mean(loss_kl_list)
        n_fea_v_tensor = torch.stack(feas).permute(1, 2, 0)

        return loss_rec, loss_kl, n_fea_v_tensor

    def encode(self, X_list, W):
        summ = 0
        feas = []
        for v, ae in enumerate(self.net_aes):
            fea_i = ae.encode(X_list[v])
            fea_i = torch.diag(W[:, v]).mm(fea_i)
            feas.append(fea_i)
            summ += fea_i
        weight = 1 / torch.sum(W, 1)
        fea_fusion = torch.diag(weight).mm(summ)

        z_mean, z_logvar, z = self.net_comvae(fea_fusion)

        return z_mean, feas

    def complete_X(self, X_list, W):
        with torch.no_grad():
            z_mean, _ = self.encode(X_list, W)
            Xs_rec = []

            for v, ae in enumerate(self.net_aes):
                X_rec = ae.decode(z_mean)
                X_rec = X_rec[torch.logical_not(W[:, v].bool())]
                Xs_rec.append(X_rec)

            for v in range(len(X_list)):
                X_list[v][torch.logical_not(W[:, v].bool())] = Xs_rec[v].detach()

            W = torch.ones([self.size, self.view_num])
            if self.args['Use_cuda']:
                W = W.cuda()
        return X_list, W


    def contrastive_loss(self, n_fea_v_tensor):
        bn, fea_dim, view_num = n_fea_v_tensor.shape
        criterion = nn.CrossEntropyLoss(reduction="sum")
        loss_list = []
        n_fea_v_tensor_ = torch.zeros(n_fea_v_tensor.shape).to(self.args['device'])
        for v in range(view_num):
            n_fea_v_tensor_[:, :, v] = normalize(n_fea_v_tensor[:, :, v], dim=1)
        n_fea_v_tensor = n_fea_v_tensor_
        for v in range(view_num):
            for w in range(v + 1, view_num):
                N = 2*bn
                h_i = n_fea_v_tensor[:, :, v]
                h_j = n_fea_v_tensor[:, :, w]
                h = torch.cat((h_i, h_j), dim=0)
                sim = torch.matmul(h, h.T) / self.args['para_tmp']
                sim_i_j = torch.diag(sim, bn)
                sim_j_i = torch.diag(sim, -bn)

                positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
                mask = self.mask_correlated_samples(N)
                negative_samples = sim[mask].reshape(N, -1)

                labels = torch.zeros(N).to(positive_samples.device).long()
                logits = torch.cat((positive_samples, negative_samples), dim=1)
                loss = criterion(logits, labels)
                loss /= N
                loss_list.append(loss)
        loss_total = sum(loss_list)
        return loss_total

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        mask = mask.bool()
        return mask



