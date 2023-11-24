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

from deepfinder.utils.data_utils import store_tomogram

from sklearn.cluster import MeanShift
from deepfinder.eval_plots import eval_plots_2D, eval_plots_3D
from deepfinder.mean_shift_utils import MeanShiftForwarder



def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum()
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum().sum() + target.sum().sum() + smooth)))
    return loss

class DeepFinder_model(pl.LightningModule):

    def __init__(self, Ncl, loss_fn, lr, weight_decay, Lrnd, bandwidth, num_seeds, max_iter, bce_loss=None, bce_fac=1.0, dice_loss=False, dice_fac=1.0, eval_plots=False,
        log_img_folder="/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MeanShift_Loss/DeepFinder_MeanShift/DeepFinder_pytorch/log_imgs"):
        super().__init__()
        self.Ncl = Ncl
        self.loss_fn = loss_fn
        self.bce_loss = bce_loss
        self.bce_fac = bce_fac
        self.dice_loss = dice_loss
        self.dice_fac = dice_fac
        self.log_img_folder = log_img_folder
        self.eval_plots = eval_plots
        # self.loss_fn.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epoch_acc_train = []
        self.epoch_loss_train = []
        self.epoch_acc_val = []
        self.epoch_MS_loss_val = []
        self.epoch_MS_loss_seeds_val = []
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
            # nn.Softmax(dim=1)
            nn.Sigmoid()
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ms_forwarder = MeanShiftForwarder(bandwidth=bandwidth, num_seeds=num_seeds, max_iter=max_iter, device=device)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        batch_data, batch_target, gt_img = batch
        if len(batch_target.shape) == 2:
            batch_target = batch_target.unsqueeze(1)
        img, pred, seeds = self(batch_data, batch_target.clone().detach().cpu())
        
        loss, _, jic_losses = self.loss_fn(batch_target, pred, seeds.to(self.device))
        if self.bce_loss is not None:
            loss += (self.bce_fac * self.bce_loss(img[:, 0], gt_img[:, 0, 1, :, :]))
        if self.dice_loss:
            loss += (self.dice_fac * dice_loss(img[:, 0], gt_img[:, 0, 1, :, :]))
        self.log('hp/train loss', loss)
        self.log('hp/train acc', torch.mean((((img > 0)*1.0)== ((gt_img > 0)*1.0))*1.0))
        self.epoch_acc_train.append(torch.mean((((img > 0)*1.0)== ((gt_img > 0)*1.0))*1.0))
        self.epoch_loss_train.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_data, batch_target, gt_img = batch
        if len(batch_target.shape) == 2:
            batch_target = batch_target.unsqueeze(1)
        # img, pred, seeds = self(batch_data)
        img, pred, seeds = self(batch_data, batch_target.clone().detach().cpu())
        loss, mins_seeds, jic_losses = self.loss_fn(batch_target, pred, seeds.to(self.device))
        print(loss, "normal loss, val")
        if self.bce_loss is not None:
            loss += self.bce_loss(img[:, 0], gt_img[:, 0, 1, :, :])
            print(loss, "after bce, val")
        if self.dice_loss:
            loss += dice_loss(img[:, 0], gt_img[:, 0, 1, :, :])
            print(loss, "after dice, val")
        
        if self.eval_plots:
            if isinstance(self, DeepFinder_model_2D_MS):
                eval_plots_2D(batch_target, pred, mins_seeds, seeds, img, batch_data, gt_img.detach().cpu().numpy()[0,0,1], self.log_img_folder, idx=batch_idx)
            else:
                eval_plots_3D(batch_target, pred, mins_seeds, seeds, img, batch_data, gt_img.detach().cpu().numpy()[0,0,1], self.log_img_folder, idx=batch_idx)
        self.log('hp/val loss', loss)
        self.log('hp/val acc', torch.mean((((img > 0) * 1.0) == ((gt_img > 0) * 1.0)) * 1.0))
        self.epoch_acc_val.append(torch.mean((((img > 0) * 1.0) == ((gt_img > 0) * 1.0)) * 1.0))
        self.epoch_loss_val.append(loss)
        self.epoch_MS_loss_val.append(jic_losses[0])
        self.epoch_MS_loss_seeds_val.append(jic_losses[1])
        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        self.log('hp/train_loss_epoch', torch.mean(torch.Tensor(self.epoch_loss_train)))
        self.log('hp/train_acc_epoch', torch.mean(torch.Tensor(self.epoch_acc_train)))
        self.log('hp/val_loss_epoch', torch.mean(torch.Tensor(self.epoch_loss_val)))
        self.log('val_loss_epoch', torch.mean(torch.Tensor(self.epoch_loss_val)))
        self.log('val_MS_loss_epoch', torch.mean(torch.Tensor(self.epoch_MS_loss_val)))
        self.log('val_MS_loss_seeds_epoch', torch.mean(torch.Tensor(self.epoch_MS_loss_seeds_val)))
        self.log('val_MS_loss_combined', torch.mean(torch.Tensor(self.epoch_MS_loss_val)) + torch.mean(torch.Tensor(self.epoch_MS_loss_seeds_val)))
        self.log('hp/val_acc_epoch', torch.mean(torch.Tensor(self.epoch_acc_val)))
        
        print("END OF EPOCH")
        print("END OF EPOCH")
        print("Loss", torch.mean(torch.Tensor(self.epoch_loss_val)))
        print("MS Loss", torch.mean(torch.Tensor(self.epoch_MS_loss_val)))
        print("MS Loss seeds", torch.mean(torch.Tensor(self.epoch_MS_loss_seeds_val)))
        self.epoch_acc_train = []
        self.epoch_loss_train = []
        self.epoch_acc_val = []
        self.epoch_loss_val = []
        self.epoch_MS_loss_val = []
        self.epoch_MS_loss_seeds_val = []

    def on_train_start(self):
        # self.logger.log_hyperparams(self.hparams, {"hp/train loss": 0, "hp/mean_pred": 0})
        print(self.hparams)

    def forward(self, x, gt_centers):
        x_high = self.layer1(x)
        mid = self.layer2(x_high)
        x = self.layer3(mid)

        x = torch.cat((mid, x), dim=1)
        x = self.layer4(x)

        x = torch.cat((x_high, x), dim=1)
        x = self.layer5(x)


        x, means, seeds0 = self.ms_forwarder.mean_shift_forward(x, gt_centers)
        return x, means, seeds0


class DeepFinder_model_2D(DeepFinder_model):
    def __init__(self, Ncl, loss_fn, lr, weight_decay, Lrnd, bandwidth, num_seeds, max_iter, bce_loss=None, bce_fac=1.0, dice_loss=False, dice_fac=1.0, eval_plots=False):
        super().__init__(Ncl, loss_fn, lr, weight_decay, Lrnd, bandwidth, num_seeds, max_iter, bce_loss=bce_loss, bce_fac=bce_fac, dice_loss=dice_loss, dice_fac=dice_fac, eval_plots=eval_plots)
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
    def __init__(self, Ncl, loss_fn, lr, weight_decay, Lrnd, bandwidth, num_seeds, max_iter, bce_loss=None, bce_fac=1.0, dice_loss=False, dice_fac=1.0, eval_plots=False):
        super().__init__(Ncl, loss_fn, lr, weight_decay, Lrnd, bandwidth, num_seeds, max_iter, bce_loss=bce_loss, bce_fac=bce_fac, dice_loss=dice_loss, dice_fac=dice_fac, eval_plots=eval_plots)
        self.layer5 = self.layer5 = nn.Sequential(
            nn.Conv2d(48 + 32, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, self.Ncl, (1, 1), padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, gt_centers):
        x_high = self.layer1(x)
        mid = self.layer2(x_high)
        x = self.layer3(mid)
        x = torch.cat((mid, x), dim=1)
        x = self.layer4(x)
        x = torch.cat((x_high, x), dim=1)
        x = self.layer5(x)

        
        x, means, seeds0 = self.ms_forwarder.mean_shift_forward(x, gt_centers)
        return x, means, seeds0