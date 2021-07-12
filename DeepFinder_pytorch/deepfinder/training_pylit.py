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
from .dataloader_pylit import DeepFinder_dataset, DeepFinder_dataset_2D
from . import model_pylit
from . import losses_pylit
from .utils import core
from .utils import common as cm
from argparse import ArgumentParser
from deepfinder.dataloader_pylit import mean_shift_collate







class TargetBuilder(core.DeepFinder):
    def __init__(self):
        core.DeepFinder.__init__(self)

        self.remove_flag = False # if true, places '0' at object voxels, instead of 'lbl'.
                                 # Usefull in annotation tool, for removing objects from target

    # Generates segmentation targets from object list. Here macromolecules are annotated with their shape.
    # INPUTS
    #   objl: list of dictionaries. Needs to contain [phi,psi,the] Euler angles for orienting the shapes.
    #   target_array: 3D numpy array that initializes the training target. Allows to pass an array already containing annotated structures like membranes.
    #                 index order of array should be [z,y,x]
    #   ref_list: list of binary 3D arrays (expected to be cubic). These reference arrays contain the shape of macromolecules ('1' for 'is object' and '0' for 'is not object')
    #             The references order in list should correspond to the class label
    #             For ex: 1st element of list -> reference of class 1
    #                     2nd element of list -> reference of class 2 etc.
    # OUTPUT
    #   target_array: 3D numpy array. '0' for background class, {'1','2',...} for object classes.
    def generate_with_shapes(self, objl, target_array, ref_list):
        """Generates segmentation targets from object list. Here macromolecules are annotated with their shape.

        Args:
            objl (list of dictionaries): Needs to contain [phi,psi,the] Euler angles for orienting the shapes.
            target_array (3D numpy array): array that initializes the training target. Allows to pass an array already containing annotated structures like membranes.
                index order of array should be [z,y,x]
            ref_list (list of 3D numpy arrays): These reference arrays are expected to be cubic and to contain the shape of macromolecules ('1' for 'is object' and '0' for 'is not object')
                The references order in list should correspond to the class label.
                For ex: 1st element of list -> reference of class 1; 2nd element of list -> reference of class 2 etc.

        Returns:
            3D numpy array: Target array, where '0' for background class, {'1','2',...} for object classes.
        """
        self.check_arguments(objl, target_array, ref_list)

        N = len(objl)
        dim = target_array.shape
        for p in range(len(objl)):
            self.display('Annotating object ' + str(p + 1) + ' / ' + str(N) + ' ...')
            lbl = int(objl[p]['label'])
            x = int(objl[p]['x'])
            y = int(objl[p]['y'])
            z = int(objl[p]['z'])
            phi = objl[p]['phi']
            psi = objl[p]['psi']
            the = objl[p]['the']

            ref = ref_list[lbl - 1]
            centeroffset = np.int(np.floor(ref.shape[0] / 2)) # here we expect ref to be cubic

            # Rotate ref:
            if phi!=None and psi!=None and the!=None:
                ref = cm.rotate_array(ref, (phi, psi, the))
                ref = np.int8(np.round(ref))

            # Get the coordinates of object voxels in target_array
            obj_voxels = np.nonzero(ref == 1)
            x_vox = obj_voxels[2] + x - centeroffset #+1
            y_vox = obj_voxels[1] + y - centeroffset #+1
            z_vox = obj_voxels[0] + z - centeroffset #+1

            for idx in range(x_vox.size):
                xx = x_vox[idx]
                yy = y_vox[idx]
                zz = z_vox[idx]
                if xx >= 0 and xx < dim[2] and yy >= 0 and yy < dim[1] and zz >= 0 and zz < dim[0]:  # if in tomo bounds
                    if self.remove_flag:
                        target_array[zz, yy, xx] = 0
                    else:
                        target_array[zz, yy, xx] = lbl

        return np.int8(target_array)

    def check_arguments(self, objl, target_array, ref_list):
        self.is_list(objl, 'objl')
        self.is_3D_nparray(target_array, 'target_array')
        self.is_list(ref_list, 'ref_list')

    # Generates segmentation targets from object list. Here macromolecules are annotated with spheres.
    # This method does not require knowledge of the macromolecule shape nor Euler angles in the objl.
    # On the other hand, it can be that a network trained with 'sphere targets' is less accurate than with 'shape targets'
    # INPUTS
    #   objl: list of dictionaries.
    #   target_array: 3D numpy array that initializes the training target. Allows to pass an array already containing annotated structures like membranes.
    #                 index order of array should be [z,y,x]
    #   radius_list: list of sphere radii (in voxels).
    #             The radii order in list should correspond to the class label
    #             For ex: 1st element of list -> sphere radius for class 1
    #                     2nd element of list -> sphere radius for class 2 etc.
    # OUTPUT
    #   target_array: 3D numpy array. '0' for background class, {'1','2',...} for object classes.
    def generate_with_spheres(self, objl, target_array, radius_list):
        """Generates segmentation targets from object list. Here macromolecules are annotated with spheres.
        This method does not require knowledge of the macromolecule shape nor Euler angles in the objl.
        On the other hand, it can be that a network trained with 'sphere targets' is less accurate than with 'shape targets'.

        Args:
            objl (list of dictionaries)
            target_array (3D numpy array): array that initializes the training target. Allows to pass an array already containing annotated structures like membranes.
                index order of array should be [z,y,x]
            radius_list (list of int): contains sphere radii per class (in voxels).
                The radii order in list should correspond to the class label.
                For ex: 1st element of list -> sphere radius for class 1, 2nd element of list -> sphere radius for class 2 etc.

        Returns:
            3D numpy array: Target array, where '0' for background class, {'1','2',...} for object classes.
        """
        Rmax = max(radius_list)
        dim = [2*Rmax, 2*Rmax, 2*Rmax]
        ref_list = []
        for idx in range(len(radius_list)):
            ref_list.append(cm.create_sphere(dim, radius_list[idx]))
        target_array = self.generate_with_shapes(objl, target_array, ref_list)
        return target_array


# TODO: add method for resuming training. It should load existing weights and train_history. So when restarting, the plot curves show prececedent epochs
class Train(core.DeepFinder):
    def __init__(self, Ncl, dim_in, lr, weight_decay, Lrnd, tensorboard_logdir, two_D_test=False):
        core.DeepFinder.__init__(self)
        self.path_out = './'
        self.tensorboard_logdir = tensorboard_logdir
        self.h5_dset_name = 'dataset' # if training set is stored as .h5 file, specify here in which h5 dataset the arrays are stored

        # Network parameters:
        self.Ncl = Ncl  # Ncl
        self.dim_in = dim_in  # /!\ has to a multiple of 4 (because of 2 pooling layers), so that dim_in=dim_out
        # self.net = models.my_model(self.dim_in, self.Ncl)
        self.loss_fn = losses_pylit.Tversky_loss(dim=((0,2,3) if two_D_test else (0,2,3,4)))
        if two_D_test:
            self.loss_fn = losses_pylit.MeanShift_loss_directional()
            # self.loss_fn = losses_pylit.MeanShift_loss()
        # self.loss_fn = torch.nn.BCELoss()
        self.Lrnd = Lrnd  # random shifts applied when sampling data- and target-patches (in voxels)
        if two_D_test:
            self.model = model_pylit.DeepFinder_model_2D_MS(self.Ncl, self.loss_fn, lr, weight_decay, Lrnd)
        else:
            self.model = model_pylit.DeepFinder_model(self.Ncl, self.loss_fn, lr, weight_decay, Lrnd)
        self.label_list = []
        for l in range(self.Ncl): self.label_list.append(l) # for precision_recall_fscore_support
                                                            # (else bug if not all labels exist in batch)

        # Training parameters:
        self.batch_size = 25
        self.epochs = 100
        self.steps_per_epoch = 100
        self.steps_per_valid = 10  # number of samples for validation
        # self.optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        self.flag_direct_read = 1
        self.flag_batch_bootstrap = 0

        self.class_weight = None

        self.check_attributes()



    def check_attributes(self):
        self.is_positive_int(self.Ncl, 'Ncl')
        self.is_multiple_4_int(self.dim_in, 'dim_in')
        self.is_positive_int(self.batch_size, 'batch_size')
        self.is_positive_int(self.epochs, 'epochs')
        self.is_positive_int(self.steps_per_epoch, 'steps_per_epoch')
        self.is_positive_int(self.steps_per_valid, 'steps_per_valid')
        self.is_int(self.Lrnd, 'Lrnd')

    # This function launches the training procedure. For each epoch, an image is plotted, displaying the progression
    # with different metrics: loss, accuracy, f1-score, recall, precision. Every 10 epochs, the current network weights
    # are saved.
    # INPUTS:
    #   path_data     : a list containing the paths to data files (i.e. tomograms)
    #   path_target   : a list containing the paths to target files (i.e. annotated volumes)
    #   objlist_train : list of dictionaries containing information about annotated objects (e.g. class, position)
    #                   In particular, the tomo_idx should correspond to the index of 'path_data' and 'path_target'.
    #                   See utils/objl.py for more info about object lists.
    #                   During training, these coordinates are used for guiding the patch sampling procedure.
    #   objlist_valid : same as 'objlist_train', but objects contained in this list are not used for training,
    #                   but for validation. It allows to monitor the training and check for over/under-fitting. Ideally,
    #                   the validation objects should originate from different tomograms than training objects.
    # The network is trained on small 3D patches (i.e. sub-volumes), sampled from the larger tomograms (due to memory
    # limitation). The patch sampling is not realized randomly, but is guided by the macromolecule coordinates contained
    # in so-called object lists (objlist).
    # Concerning the loading of the dataset, two options are possible:
    #    flag_direct_read=0: the whole dataset is loaded into memory
    #    flag_direct_read=1: only the patches are loaded into memory, each time a training batch is generated. This is
    #                        usefull when the dataset is too large to load into memory. However, the transfer speed
    #                        between the data server and the GPU host should be high enough, else the procedure becomes
    #                        very slow.
    # TODO: delete flag_direct_read. Launch should detect if direct_read is desired by checking if input data_list and
    #       target_list contain str (path) or numpy array
    def launch(self, path_data, path_target, objlist_train, objlist_valid, two_D_test=False, two_D_data_train=(None, None), two_D_data_val=(None, None)):
        """This function launches the training procedure. For each epoch, an image is plotted, displaying the progression
        with different metrics: loss, accuracy, f1-score, recall, precision. Every 10 epochs, the current network weights
        are saved.

        Args:
            path_data (list of string): contains paths to data files (i.e. tomograms)
            path_target (list of string): contains paths to target files (i.e. annotated volumes)
            objlist_train (list of dictionaries): contains information about annotated objects (e.g. class, position)
                In particular, the tomo_idx should correspond to the index of 'path_data' and 'path_target'.
                See utils/objl.py for more info about object lists.
                During training, these coordinates are used for guiding the patch sampling procedure.
            objlist_valid (list of dictionaries): same as 'objlist_train', but objects contained in this list are not
                used for training, but for validation. It allows to monitor the training and check for over/under-fitting.
                Ideally, the validation objects should originate from different tomograms than training objects.

        Note:
            The function saves following files at regular intervals:
                net_weights_epoch*.h5: contains current network weights

                net_train_history.h5: contains arrays with all metrics per training iteration

                net_train_history_plot.png: plotted metric curves

        """
        self.check_attributes()
        self.check_arguments(path_data, path_target, objlist_train, objlist_valid)


        if two_D_test:
            train_set = DeepFinder_dataset_2D(two_D_data_train[0], two_D_data_train[1], two_D_data_train[2], max_len=256)
            val_set = DeepFinder_dataset_2D(two_D_data_val[0], two_D_data_val[1], two_D_data_val[2], max_len=4)
        else:
            train_set = DeepFinder_dataset(self.flag_direct_read, path_data, path_target, objlist_train, self.dim_in,
                                           self.Ncl, self.Lrnd, self.h5_dset_name)
            val_set = DeepFinder_dataset(self.flag_direct_read, path_data, path_target, objlist_valid, self.dim_in,
                                           self.Ncl, self.Lrnd, self.h5_dset_name)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=True)


        logger = pl_loggers.TensorBoardLogger(self.tensorboard_logdir)
        trainer = pl.Trainer(logger=logger, log_every_n_steps=1,gpus=1)
        trainer.fit(self.model, train_loader, val_loader)



    def check_arguments(self, path_data, path_target, objlist_train, objlist_valid):
        self.is_list(path_data, 'path_data')
        self.is_list(path_target, 'path_target')
        self.are_lists_same_length([path_data, path_target], ['data_list', 'target_list'])
        self.is_list(objlist_train, 'objlist_train')
        self.is_list(objlist_valid, 'objlist_valid')

