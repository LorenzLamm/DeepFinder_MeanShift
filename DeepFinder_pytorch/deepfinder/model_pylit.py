# ============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# ============================================================================================
# Copyright (c) 2019 - now
# Inria - Centre de Rennes Bretagne Atlantique, France
# Author: Emmanuel Moebel (serpico team); adapted by Lorenz Lamm
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# ============================================================================================


import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import numpy as np
from deepfinder.utils.weighted_mean_shift import MeanShift_weighted

from matplotlib import pyplot as plt
from torch.nn.functional import normalize
import scipy

from sklearn.cluster import MeanShift


class DeepFinder_model(pl.LightningModule):

    def __init__(self, Ncl, loss_fn, lr, weight_decay, Lrnd):
        super().__init__()
        self.Ncl = Ncl
        self.loss_fn = loss_fn
        # self.loss_fn.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epoch_acc_train = []
        self.epoch_loss_train = []
        self.epoch_acc_val = []
        self.epoch_loss_val = []
        self.save_hyperparameters()
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 32, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 32, (3, 3, 3), padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 48, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(48, 48, (3, 3, 3), padding=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(48, 64, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 64, (2, 2, 2), stride=2),
            nn.Conv3d(64, 64, (3, 3, 3), padding=1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(64 + 48, 48, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(48, 48, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(48, 48, (2, 2, 2), stride=2),
            nn.Conv3d(48, 48, (3, 3, 3), padding=1)
        )
        self.layer5 = nn.Sequential(
            nn.Conv3d(48 + 32, 32, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 32, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(32, self.Ncl, (1, 1, 1), padding=0),
            nn.Softmax(dim=1)
        )
        # self.layer1.to(self.device)
        # self.layer2.to(self.device)
        # self.layer3.to(self.device)
        # self.layer4.to(self.device)
        # self.layer5.to(self.device)
        # self.loss_fn.to(self.device)


    def forward(self, x):
        x_high = self.layer1(x)
        mid = self.layer2(x_high)
        x = self.layer3(mid)
        x = torch.cat((mid, x), dim=1)
        x = self.layer4(x)
        x = torch.cat((x_high, x), dim=1)
        x = self.layer5(x)
        return x


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        batch_data, batch_target, gt_img = batch
        img, pred, seeds = self(batch_data, batch_target.clone().detach().cpu())
        loss, _ = self.loss_fn(batch_target, pred, seeds.to(self.device))
        self.log('hp/train loss', loss)
        self.log('hp/train acc', torch.mean((((img > 0)*1.0)== ((gt_img > 0)*1.0))*1.0))
        self.epoch_acc_train.append(torch.mean((((img > 0)*1.0)== ((gt_img > 0)*1.0))*1.0))
        self.epoch_loss_train.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_data, batch_target, gt_img = batch
        # img, pred, seeds = self(batch_data)
        img, pred, seeds = self(batch_data, batch_target.clone().detach().cpu())
        loss, mins_seeds = self.loss_fn(batch_target, pred, seeds.to(self.device))

        self.eval_plots(batch_target, pred, mins_seeds, seeds, img, batch_data, gt_img.detach().cpu().numpy()[0,0,1])

        self.log('hp/val loss', loss)
        self.log('hp/val acc', torch.mean((((img > 0) * 1.0) == ((gt_img > 0) * 1.0)) * 1.0))
        self.epoch_acc_train.append(torch.mean((((img > 0) * 1.0) == ((gt_img > 0) * 1.0)) * 1.0))
        self.epoch_loss_train.append(loss)
        return loss

    def eval_plots(self, batch_target, pred, mins_seeds, seeds, img, batch_data, gt_img):
        plt.figure()
        plt.scatter(x=batch_target[0, :, 0].detach().cpu(), y=batch_target.detach().cpu()[0, :, 1], c='black', s=100)
        plt.scatter(pred.clone().detach().cpu().numpy()[0, :, 0], pred.clone().detach().cpu().numpy()[0, :, 1],
                    c=mins_seeds.clone().detach().cpu().numpy()[0])
        plt.scatter(seeds.clone().detach().cpu().numpy()[:, 0], seeds.clone().detach().cpu().numpy()[:, 1], c='yellow')
        plt.colorbar()
        plt.savefig('/fs/pool/pool-engel/Lorenz/DeepFinder_MeanShift_RNN/log_imgs/loss_vals.png')

        ms_img = img[0, 0].detach().cpu().numpy()
        points = np.argwhere(ms_img > 0.5)
        ms_weights = ms_img[ms_img > 0.5]
        ms_weights = np.reshape(ms_weights, (-1))

        # if points.shape[0] > 8 and points.shape[0] < img[0,0].shape[1] * img[0,0].shape[0]:
        if points.shape[0] > 8:  # and points.shape[0] < img[0,0].shape[1] * img[0,0].shape[0]:
            # ms = MeanShift(bandwidth=5.)
            # ms.fit(points)
            # center_coords = ms.cluster_centers_
            ms = MeanShift_weighted(bandwidth=4.)
            center_coords, center_idcs = ms.mean_shift(points, ms_weights, weighting='simple', n_pr=8, fuse_dist=2.)
            center_coords2, _ = self.mean_shift_for_seeds(torch.tensor(points),
                                                          torch.tensor(ms_weights - 0.5).to(self.device),
                                                          torch.tensor(points), bandwidth=5., max_iter=1000)
            mask = np.ones(center_coords.shape[0])
            for nr in range(center_coords.shape[0]):
                if np.sum(center_idcs == nr) < 4:
                    mask[nr] = 0
            center_coords = center_coords[mask == 1]
        else:
            center_coords = 25. * np.ones((1, 2))
        plt.figure(figsize=(20, 20))
        plt.subplot(1, 4, 1)
        plt.scatter(x=batch_target[0, :, 0].detach().cpu(), y=batch_target.detach().cpu()[0, :, 1], c='blue', s=60)
        plt.scatter(x=pred.clone().detach().cpu()[0, :, 0], y=pred.clone().detach().cpu()[0, :, 1], c='red')
        # plt.scatter(x=pred.clone().detach().cpu()[1, :, 1], y=pred.clone().detach().cpu()[1, :, 0], c='red')
        if pred.shape[0] > 2:
            plt.scatter(x=pred.clone().detach().cpu()[2, :, 1], y=pred.clone().detach().cpu()[2, :, 0], c='red')
            plt.scatter(x=pred.clone().detach().cpu()[3, :, 1], y=pred.clone().detach().cpu()[3, :, 0], c='red')
        plt.imshow(batch_data.clone().detach().cpu()[0, 0])
        plt.subplot(1, 4, 2)
        plt.scatter(x=batch_target.detach().cpu()[0, :, 0], y=batch_target.detach().cpu()[0, :, 1], c='blue')
        plt.scatter(x=pred.clone().detach().cpu()[0, :, 0], y=pred.clone().detach().cpu()[0, :, 1], c='red')
        plt.scatter(x=seeds.clone().detach().cpu()[:, 0], y=seeds.clone().detach().cpu()[:, 1], c='green')
        if points.shape[0] > 8:  # and points.shape[0] < img[0,0].shape[1] * img[0,0].shape[0]:
            # plt.scatter(x=center_coords2.clone().detach().cpu()[:, 1], y=center_coords2.clone().detach().cpu()[:, 0],
            #             c='blue')
            plt.scatter(x=center_coords[:, 1], y=center_coords[:, 0], c='black', marker='P')
        plt.imshow(img.clone().detach().cpu()[0, 0])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(1, 4, 3)
        plt.imshow(ms_img)
        # plt.scatter(x = points[:, 1], y=points[:,0])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(1,4,4)
        plt.imshow(gt_img)

        plt.savefig('/fs/pool/pool-engel/Lorenz/DeepFinder_MeanShift_RNN/log_imgs/progress.png')


    def on_epoch_end(self):
        if len(self.epoch_acc_train) > 0:
            self.log('hp/train_loss_epoch', torch.mean(torch.Tensor(self.epoch_loss_train)))
            self.log('hp/train_acc_epoch', torch.mean(torch.Tensor(self.epoch_acc_train)))
        if len(self.epoch_acc_val) > 0:
            self.log('hp/val_loss_epoch', torch.mean(torch.Tensor(self.epoch_loss_val)))
            self.log('hp/val_acc_epoch', torch.mean(torch.Tensor(self.epoch_acc_val)))
        self.epoch_acc_train = []
        self.epoch_loss_train = []
        self.epoch_acc_val = []
        self.epoch_loss_val = []

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/train loss": 0, "hp/mean_pred": 0})
        print(self.hparams)




class DeepFinder_model_2D(DeepFinder_model):
    def __init__(self, Ncl, loss_fn, lr, weight_decay, Lrnd):
        super().__init__(Ncl, loss_fn, lr, weight_decay, Lrnd)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 48, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, (3, 3), padding=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(48, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, (2, 2), stride=2),
            nn.Conv2d(64, 64, (3, 3), padding=1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64 + 48, 48, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, (3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 48, (2, 2), stride=2),
            nn.Conv2d(48, 48, (3, 3), padding=1)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(48 + 32, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, self.Ncl, (1, 1), padding=0),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x_high = self.layer1(x)
        mid = self.layer2(x_high)
        x = self.layer3(mid)
        x = torch.cat((mid, x), dim=1)
        x = self.layer4(x)
        x = torch.cat((x_high, x), dim=1)
        x = self.layer5(x)
        return x


class DeepFinder_model_2D_MS(DeepFinder_model_2D):
    def __init__(self, Ncl, loss_fn, lr, weight_decay, Lrnd):
        super().__init__(Ncl, loss_fn, lr, weight_decay, Lrnd)
        self.layer5 = self.layer5 = nn.Sequential(
            nn.Conv2d(48 + 32, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, self.Ncl, (1, 1), padding=0),
            nn.Sigmoid()
        )
    # def forward(self, x):
    #     x_high = self.layer1(x)
    #     mid = self.layer2(x_high)
    #     x = self.layer3(mid)
    #     x = torch.cat((mid, x), dim=1)
    #     x = self.layer4(x)
    #     x = torch.cat((x_high, x), dim=1)
    #     x = self.layer5(x)

    def initialize_seeds(self, x, sample_pos, num_seeds=100, margin=2):
        shape = x.shape
        all_seeds = []
        if len(shape) == 4:
            s_max1 = shape[2]
            s_max2 = shape[3]
        else:
            raise IOError('Wrong input shape!')
        while len(all_seeds) < num_seeds:
            x_comp = np.random.uniform(margin, s_max1-margin)
            y_comp = np.random.uniform(margin, s_max2-margin)
            dist = scipy.spatial.distance.cdist(sample_pos.numpy(),np.expand_dims(np.array((x_comp, y_comp)), axis=0))
            if np.min(dist, axis=0) < 8.0:
                all_seeds.append(np.array((x_comp, y_comp)))
        all_seeds = torch.from_numpy(np.stack(all_seeds))
        return all_seeds

    def cos_batch(self, a, b):
        # return sqrt(((a[None,:] - b[:,None]) ** 2).sum(2))
        b = b.float()
        a = a.float()
        num = a @ b.T
        denom = torch.norm(a, dim=1).reshape(-1, 1) * torch.norm(b, dim=1)
        return num / denom

    def euclidean_dist(self, a, b):
        dist_mat = torch.cdist(a.float(), b.float())
        return dist_mat

    def get_weight(self, dist, nn_weights, bandwidth):
        nn_weights = nn_weights.reshape(1, -1)
        thr = bandwidth
        # max = torch.tensor(1.0e+10).double().cuda()
        max = torch.tensor(1.0).double().to(self.device)
        min = torch.tensor(0.0).double().to(self.device)
        nn_weights_min = 0.5
        # nn_weights = nn_weights - nn_weights_min
        # dis=torch.where(sim>thr, 1-sim, max)
        dis = torch.where(dist < thr, max, min)
        dis = dis.to(self.device)
        # dis *= torch.exp(nn_weights)
        dis *= nn_weights
        dis = normalize(dis, dim=1, p=2)
        return dis

    def mean_shift_for_seeds(self, coords, nn_weights, seeds, bandwidth, max_iter=100):
        stop_thresh = 1e-3 * bandwidth
        iter = 0
        X = coords.to(self.device)
        S = seeds.to(self.device)
        B = torch.tensor(bandwidth).double().to(self.device)

        while True:
            weight = self.get_weight(self.euclidean_dist(S, X.float()), nn_weights, B)
            num = (weight[:, :, None] * X).sum(dim=1)
            S_old = S
            S = num / (weight.sum(1)[:, None] + 1e-7)
            iter += 1

            if (torch.norm(S - S_old, dim=1).mean() < stop_thresh or iter == max_iter):
                break

        p_num = []
        for line in weight:
            p_num.append(line[line == 1].size()[0])

        return S, p_num

    def initialize_coords(self, x):
        shape = x.shape
        if len(shape) == 4:
            s_max1 = shape[2]
            s_max2 = shape[3]
        else:
            raise IOError('Wrong input shape!')
        coords_x = torch.linspace(0, s_max1-1, s_max1)
        coords_y = torch.linspace(0, s_max2-1, s_max2)
        coords_x = coords_x.repeat(1, s_max1).view(-1, s_max1).transpose(1, 0)
        coords_y = coords_y.repeat(1, s_max1).view(-1, s_max1)
        coordinate_grid = torch.cat((coords_x.unsqueeze(2), coords_y.unsqueeze(2)), dim=2)
        coordinate_grid = coordinate_grid.transpose(1,0)
        coordinate_grid = torch.reshape(coordinate_grid, (-1, 2))
        return coordinate_grid


    def forward(self, x, gt_centers):
        x_high = self.layer1(x)
        mid = self.layer2(x_high)
        x = self.layer3(mid)
        x = torch.cat((mid, x), dim=1)
        x = self.layer4(x)
        x = torch.cat((x_high, x), dim=1)
        x = self.layer5(x)
        coords = self.initialize_coords(x)
        means = []

        for i in range(x.shape[0]):
            seeds = self.initialize_seeds(x, gt_centers[i], num_seeds=32, margin=2.)
            if i == 0:
                seeds0 = seeds.clone()
            mean, p_num = self.mean_shift_for_seeds(coords, x[i,0], seeds, bandwidth=4., max_iter=100)
            means.append(mean)
        means = torch.stack(means)
        return x, means, seeds0