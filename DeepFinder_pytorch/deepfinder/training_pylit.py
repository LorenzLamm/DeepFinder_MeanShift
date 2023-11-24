# ============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# ============================================================================================
# Copyright (c) 2019 - now
# Inria - Centre de Rennes Bretagne Atlantique, France
# Author: Emmanuel Moebel (serpico team); adapted by Lorenz Lamm
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# ============================================================================================


import h5py
import numpy as np
import time
from matplotlib import pyplot as plt

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from sklearn.metrics import precision_recall_fscore_support
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from .dataloader_pylit import DeepFinder_dataset, DeepFinder_dataset_2D, DeepFinder_dataset_shrec, DeepFinder_dataset_experimental
from . import model_pylit
from . import losses_pylit
from .utils import core
from .utils import common as cm
from argparse import ArgumentParser
from deepfinder.dataloader_pylit import mean_shift_collate


# # TODO: add method for resuming training. It should load existing weights and train_history. So when restarting, the plot curves show prececedent epochs
# class Train():
#     def __init__(self, Ncl, dim_in, lr, weight_decay, Lrnd, tensorboard_logdir, bandwidth, num_seeds, max_iter, two_D_test=False, three_D_test=False):

#         # Network parameters:
#         self.Ncl = Ncl  # Ncl
#         self.dim_in = dim_in  # /!\ has to a multiple of 4 (because of 2 pooling layers), so that dim_in=dim_out
#         self.tensorboard_logdir = tensorboard_logdir
        
#         self.loss_fn = losses_pylit.Tversky_loss(dim=((0,2,3) if two_D_test else (0,2,3,4)))
#         if two_D_test or three_D_test:
#             # self.loss_fn = losses_pylit.MeanShift_loss_directional()
#             self.loss_fn = losses_pylit.MeanShift_loss()
#             self.loss_fn_BCE = torch.nn.BCELoss()
#         self.Lrnd = Lrnd  # random shifts applied when sampling data- and target-patches (in voxels)
#         if two_D_test:
#             self.model = model_pylit.DeepFinder_model_2D_MS(self.Ncl, self.loss_fn, lr, weight_decay, Lrnd, bandwidth=bandwidth, num_seeds=num_seeds, max_iter=max_iter, bce_loss=self.loss_fn_BCE)
#         else:
#             self.model = model_pylit.DeepFinder_model(self.Ncl, self.loss_fn, lr, weight_decay, Lrnd, bce_loss=self.loss_fn_BCE)
#         self.label_list = []
#         for l in range(self.Ncl): self.label_list.append(l) # for precision_recall_fscore_support
#                                                             # (else bug if not all labels exist in batch)


#     def launch(self, path_data, path_target, objlist_train, objlist_valid, two_D_test=False, three_D_test=False,
#                two_D_data_train=(None, None), two_D_data_val=(None, None)):

#         if two_D_test or three_D_test:
#             train_set = DeepFinder_dataset_2D(two_D_data_train[0], two_D_data_train[1], two_D_data_train[2], max_len=1000)
#             # val_set = DeepFinder_dataset_2D(two_D_data_train[0], two_D_data_train[1], two_D_data_train[2], max_len=1)
#             val_set = DeepFinder_dataset_2D(two_D_data_val[0], two_D_data_val[1], two_D_data_val[2], max_len=10)
#         else:
#             train_set = DeepFinder_dataset(self.flag_direct_read, path_data, path_target, objlist_train, self.dim_in,
#                                            self.Ncl, self.Lrnd, self.h5_dset_name)
#             val_set = DeepFinder_dataset(self.flag_direct_read, path_data, path_target, objlist_valid, self.dim_in,
#                                            self.Ncl, self.Lrnd, self.h5_dset_name)

#         train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
#         val_loader = DataLoader(val_set, batch_size=1, shuffle=True)


#         logger = pl_loggers.TensorBoardLogger(self.tensorboard_logdir)
#         trainer = pl.Trainer(logger=logger, log_every_n_steps=1)#,gpus=0)
#         trainer.fit(self.model, train_loader, val_loader)



class Train:
    def __init__(self, Ncl, dim_in, lr, weight_decay, Lrnd, tensorboard_logdir, bandwidth, num_seeds, max_iter, use_bce=False, use_dice=False, use_MS_loss=True, bce_fac=1.0, dice_fac=1.0,
                 two_D_test=False, three_D_test=False, run_token="MS_DF", eval_plots=False):
        """
        Initialize the Train class.

        Parameters:
        - Ncl (int): Number of classes.
        - dim_in (int): Input dimension. Must be a multiple of 4.
        - lr (float): Learning rate.
        - weight_decay (float): Weight decay for regularization.
        - Lrnd (int): Random shifts applied when sampling data- and target-patches (in voxels).
        - tensorboard_logdir (str): Directory for TensorBoard logs.
        - bandwidth (float): Bandwidth parameter.
        - num_seeds (int): Number of seeds.
        - max_iter (int): Maximum number of iterations.
        - two_D_test (bool): Flag for 2D testing.
        - three_D_test (bool): Flag for 3D testing.
        """
        assert dim_in % 4 == 0, "Input dimension must be a multiple of 4."

        # Network parameters
        self.Ncl = Ncl
        self.dim_in = dim_in
        self.Lrnd = Lrnd
        self.tensorboard_logdir = tensorboard_logdir
        self.run_token = run_token
        self.eval_plots = eval_plots

        self.use_bce = use_bce
        self.use_dice = use_dice
        self.use_MS_loss = use_MS_loss
        self.bce_fac = bce_fac
        self.dice_fac = dice_fac

        self.two_D_test = two_D_test
        self.three_D_test = three_D_test
        # Initialize loss function
        self._init_loss_function(two_D_test, three_D_test)

        # Initialize model
        self._init_model(lr, weight_decay, bandwidth, num_seeds, max_iter, two_D_test)

        # Label list for precision_recall_fscore_support (avoids bug if not all labels exist in batch)
        self.label_list = list(range(self.Ncl))

    def _init_loss_function(self, two_D_test, three_D_test):
        """
        Initialize the loss function based on the testing mode.
        """
        if two_D_test or three_D_test:
            self.loss_fn = losses_pylit.MeanShift_loss(use_loss=self.use_MS_loss)
            if self.use_bce:
                self.loss_fn_BCE = torch.nn.BCELoss()
            else:
                self.loss_fn_BCE = None
        else:
            loss_dims = (0, 2, 3, 4) if not two_D_test else (0, 2, 3)
            self.loss_fn = losses_pylit.Tversky_loss(dim=loss_dims)
            self.loss_fn_BCE = None

    def _init_model(self, lr, weight_decay, bandwidth, num_seeds, max_iter, two_D_test):
        """
        Initialize the model based on the testing mode.
        """
        model_args = (self.Ncl, self.loss_fn, lr, weight_decay, self.Lrnd)
        if two_D_test:
            self.model = model_pylit.DeepFinder_model_2D_MS(*model_args, bandwidth=bandwidth, num_seeds=num_seeds,
                                                         max_iter=max_iter, bce_loss=self.loss_fn_BCE, bce_fac=self.bce_fac, dice_loss=self.use_dice, dice_fac=self.dice_fac, eval_plots=self.eval_plots)
        else:
            self.model = model_pylit.DeepFinder_model(*model_args, bandwidth=bandwidth, num_seeds=num_seeds,
                                                         max_iter=max_iter, bce_loss=self.loss_fn_BCE, bce_fac=self.bce_fac, dice_loss=self.use_dice, dice_fac=self.dice_fac, eval_plots=self.eval_plots)

    def launch(self, path_data, path_target, objlist_train, objlist_valid,
               two_D_data_train=(None, None, None), two_D_data_val=(None, None, None), shrec=False, experimental=False, experimental_sparse=False):
        """
        Launch training and validation.

        Parameters:
        - path_data (str): Path to the data.
        - path_target (str): Path to the target.
        - objlist_train (list): List of objects for training.
        - objlist_valid (list): List of objects for validation.
        - two_D_data_train (tuple): Tuple containing 2D training data information.
        - two_D_data_val (tuple): Tuple containing 2D validation data information.
        """
        # Initialize datasets
        if not shrec and not experimental and not experimental_sparse:
            if self.two_D_test or self.three_D_test:
                train_set = DeepFinder_dataset_2D(*two_D_data_train, max_len=1000)
                val_set = DeepFinder_dataset_2D(*two_D_data_val, max_len=100)
            else:
                train_set = DeepFinder_dataset(path_data, path_target, objlist_train, self.dim_in, self.Ncl, self.Lrnd)
                val_set = DeepFinder_dataset(path_data, path_target, objlist_valid, self.dim_in, self.Ncl, self.Lrnd)
        elif shrec:
            train_set = DeepFinder_dataset_shrec(path_data=two_D_data_train, model_nrs=range(6))
            val_set = DeepFinder_dataset_shrec(path_data=two_D_data_train, model_nrs=range(6,8))
            test_set = DeepFinder_dataset_shrec(path_data=two_D_data_train, model_nrs=range(8,10))

        elif experimental:
            train_set = DeepFinder_dataset_experimental(path_data=two_D_data_train, train_split="train")
            val_set = DeepFinder_dataset_experimental(path_data=two_D_data_train, train_split="val")
            test_set = DeepFinder_dataset_experimental(path_data=two_D_data_train, train_split="test")
        elif experimental_sparse:
            train_set = DeepFinder_dataset_experimental(path_data=two_D_data_train, train_split="train", sparse_points=True)
            val_set = DeepFinder_dataset_experimental(path_data=two_D_data_train, train_split="val", sparse_points=True)

        # Initialize data loaders
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=True)

        # Initialize logger and trainer
        # logger = pl_loggers.TensorBoardLogger(self.tensorboard_logdir)
        run_token = self.run_token
        # Setup model checkpointing

        if self.use_MS_loss:
            checkpoint_callback = ModelCheckpoint(
                monitor='val_MS_loss_combined',  # or another metric that you want to monitor
                dirpath='./checkpoints',  # change this to your desired checkpoint directory
                filename=run_token + '_{epoch}-{val_MS_loss_combined:.2f}',  # save files as epoch-number_val-loss
                save_top_k=1,  # only keep the top 3 checkpoints
                mode='min',  # for validation loss, 'min' is the mode we want to save in
                auto_insert_metric_name=False,  # to simplify the filename
            )
        else:
            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss_epoch',  # or another metric that you want to monitor
                dirpath='./checkpoints',  # change this to your desired checkpoint directory
                filename=run_token + '_{epoch}-{val_loss_epoch:.2f}',  # save files as epoch-number_val-loss
                save_top_k=1,  # only keep the top 3 checkpoints
                mode='min',  # for validation loss, 'min' is the mode we want to save in
                auto_insert_metric_name=False,  # to simplify the filename
            )

        trainer = pl.Trainer(log_every_n_steps=1, callbacks=[checkpoint_callback])

        # Start training
        trainer.fit(self.model, train_loader, val_loader)