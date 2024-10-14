import os

import torch
import matplotlib.pyplot as plt


class plot_picture():
    def __init__(self, data_name):
        self.imgdata_list = []
        self.title_list = []
        self.permute = (0, 2, 1, 3)
        if data_name in ['Yale15_3_165', 'ORL40_3_400']:
            self.img_size = [-1, 64, 64, 1]
        if data_name in ['YaleB10_3_650']:
            self.img_size = [-1, 50, 50, 1]
        if data_name in ['Notting-Hill']:
            self.img_size = [-1, 40, 50, 1]
        if data_name in ['MNIST-USPS']:
            self.img_size = [-1, 28, 28, 1]
            self.permute = (0, 1, 2, 3)
        if data_name in ['NoisyMNIST-30000']:
            self.img_size = [-1, 28, 28, 1]
        if data_name in ['rgbd_mtv']:
            self.img_size = [-1, 64, 64, 3]
        if data_name in ['handwritten_2views', 'handwritten-5view']:
            self.img_size = [-1, 16, 15, 1]
            self.permute = (0, 1, 2, 3)

        self.data_name = data_name
        project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), './..'))
        pic_dir = project_dir + '/recovery_pic'
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        self.pic_path = pic_dir + '/' + data_name + '_recovery.pdf'


    def load_data(self, imgdata:torch.Tensor, title=''):
        imgdata = imgdata.reshape(self.img_size).permute(self.permute).cpu().detach().numpy()
        imgdata = imgdata * 255
        imgdata = imgdata.astype('uint8')
        self.imgdata_list.append(imgdata)
        self.title_list.append(title)

    def plot(self):
        if len(self.imgdata_list) == 1:
            self.plot_one()
        else:
            self.plot_multi()

    def plot_one(self):
        imgdata = self.imgdata_list[0]
        title = self.title_list[0]
        n = len(imgdata)
        fig, axes = plt.subplots(1, n, figsize=(20, 3))

        plt.subplots_adjust(left=0.01, right=0.99, top=1, bottom=1, wspace=0.2, hspace=0)
        axes[0].set_title(title, fontsize=10, fontweight='bold')
        for i in range(n):
            im_result = imgdata[i]
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].imshow(im_result, cmap='gray')
        # plt.title(title, fontsize=10, fontweight='bold')
        fig.show()

    def plot_multi(self):
        dtype = len(self.imgdata_list)
        num = len(self.imgdata_list[0])
        fig, axes = plt.subplots(dtype, num, figsize=(num, dtype))
        # fig.subplots_adjust(left=0.1)
        # fig.subplots_adjust(left=0, bottom=None, right=0, top=None, wspace=None, hspace=0.3)
        for d in range(dtype):
            axes[d][0].set_title(self.title_list[d], fontsize=6 * dtype, fontweight='bold')
            for n in range(num):
                img_data = self.imgdata_list[d][n]
                axes[d][n].set_xticks([])
                axes[d][n].set_yticks([])
                axes[d][n].imshow(img_data, cmap='gray')

        plt.savefig(self.pic_path, format='pdf', bbox_inches='tight', dpi=600)
        print('recovery picture is saved in {}'.format(self.pic_path))
        fig.show()
