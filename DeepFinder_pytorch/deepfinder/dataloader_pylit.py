# ============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# ============================================================================================
# Copyright (c) 2019 - now
# Inria - Centre de Rennes Bretagne Atlantique, France
# Author: Emmanuel Moebel (serpico team); adapted by Lorenz Lamm
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# ============================================================================================

from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
from .utils import core
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


def transpose_to_channels_first(in_tensor):
    in_tensor = torch.transpose(in_tensor, 0, 3)
    in_tensor = torch.transpose(in_tensor, 1, 3)
    in_tensor = torch.transpose(in_tensor, 2, 3)
    return in_tensor

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    Args:
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes. If `None`, this would be inferred
          as the (largest number in `y`) + 1.
        dtype: The data type expected by the input. Default: `'float32'`.
    Returns:
        A binary matrix representation of the input. The classes axis is placed
        last.

    <<<<< Copied from tensorflow >>>>>
    """
    y = torch.tensor(y, dtype=torch.int64)
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
      input_shape = tuple(input_shape[:-1])
    y = y.view(-1)

    if not num_classes:
      num_classes = torch.max(y) + 1
    n = y.shape[0]
    categorical = torch.zeros((n, num_classes), dtype=torch.float32)
    categorical[torch.arange(n, dtype=torch.long), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class DeepFinder_dataset_2D(Dataset):
    def __init__(self, path_data, path_target, path_points, max_len=None):
        self.data = np.load(path_data)
        self.target = np.load(path_target)
        self.points = np.load(path_points)
        self.margin = 4.
        if max_len is not None:
            self.data = self.data[:max_len]
            self.target = self.target[:max_len]
            self.points = self.points[:max_len]
        self.shape = self.data.shape


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        points = self.points[idx]
        points = points[np.all((points[:, 0] > self.margin, points[:, 1] > self.margin), axis=0)]
        points = points[np.all((points[:, 0] < self.shape[1]-self.margin, points[:, 1] < self.shape[2]-self.margin), axis=0)]
        return torch.from_numpy(self.data[idx]).unsqueeze(0).float(), torch.from_numpy(points).float().squeeze(), \
               torch.from_numpy(self.target[idx]).unsqueeze(0).float()


def mean_shift_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    gt_imgs = [item[2] for item in batch]
    # target = torch.tensor(target)
    data = torch.stack(data)
    gt_imgs = torch.stack(gt_imgs)
    return data, target, gt_imgs


class DeepFinder_dataset(Dataset):
    def __init__(self, flag_direct_read, path_data, path_target, objlist, dim_in,
                 Ncl, Lrnd, h5_dset_name):
        self.flag_direct_read = flag_direct_read
        self.path_data = path_data
        self.path_target = path_target
        self.objlist = objlist
        self.dim_in = dim_in
        self.Ncl = Ncl
        self.p_in = np.int(np.floor(self.dim_in / 2))
        self.Lrnd = Lrnd
        if not self.flag_direct_read:
            self.data_list, self.target_list = core.load_dataset(self.path_data, self.path_target, h5_dset_name)

    def __len__(self):
        return len(self.objlist)

    def __getitem__(self, idx):
        if self.flag_direct_read:
            return self.generate_batch_direct_read(idx)
        else:
            return self.generate_batch_from_array(idx)





    def generate_batch_direct_read(self, index):
        batch_data = np.zeros((self.dim_in, self.dim_in, self.dim_in, 1))

        tomoID = int(self.objlist[index]['tomo_idx'])
        h5file = h5py.File(self.path_data[tomoID], 'r')
        tomodim = h5file['dataset'].shape  # get tomo dimensions without loading the array
        h5file.close()

        x, y, z = core.get_patch_position(tomodim, self.p_in, self.objlist[index], self.Lrnd)

        # Load data and target patches:
        h5file = h5py.File(self.path_data[tomoID], 'r')
        patch_data = h5file['dataset'][z-self.p_in:z+self.p_in, y-self.p_in:y+-self.p_in, x-self.p_in:x+self.p_in]
        h5file.close()

        h5file = h5py.File(-self.path_target[tomoID], 'r')
        patch_target = h5file['dataset'][z-self.p_in:z+self.p_in, y-self.p_in:y+-self.p_in, x-self.p_in:x+self.p_in]
        h5file.close()

        # Process the patches in order to be used by network:
        patch_data = (patch_data - np.mean(patch_data)) / np.std(patch_data)  # normalize
        patch_target_onehot = to_categorical(patch_target, self.Ncl)

        # Store into batch array:
        batch_data[:, :, :, 0] = patch_data
        batch_target = patch_target_onehot

        # Data augmentation (180degree rotation around tilt axis):
        if np.random.uniform() < 0.5:
            batch_data = np.rot90(batch_data, k=2, axes=(0, 2))
            batch_target = np.rot90(batch_target, k=2, axes=(0, 2))
        batch_data = transpose_to_channels_first(batch_data)
        batch_target = transpose_to_channels_first(batch_target)
        return torch.from_numpy(batch_data), torch.from_numpy(batch_target)

    # Generates batches for training and validation. In this version, the whole dataset has already been loaded into
    # memory, and batch is sampled from there. Apart from that does the same as above.
    # Is called when self.flag_direct_read=False
    # INPUTS:
    #   data: list of numpy arrays
    #   target: list of numpy arrays
    #   batch_size: int
    #   objlist: list of dictionnaries
    # OUTPUT:
    #   batch_data: numpy array [batch_idx, z, y, x, channel] in our case only 1 channel
    #   batch_target: numpy array [batch_idx, z, y, x, class_idx] is one-hot encoded
    def generate_batch_from_array(self, index):
        batch_data = torch.zeros((self.dim_in, self.dim_in, self.dim_in, 1))

        tomoID = int(self.objlist[index]['tomo_idx'])

        tomodim = self.data_list[tomoID].shape

        sample_data = self.data_list[tomoID]
        sample_target = self.target_list[tomoID]

        # Get patch position:
        x, y, z = core.get_patch_position(tomodim, self.p_in, self.objlist[index], self.Lrnd)

        # Get patch:
        patch_data = sample_data[z - self.p_in:z + self.p_in, y - self.p_in:y + self.p_in, x - self.p_in:x + self.p_in]
        patch_target = sample_target[z - self.p_in:z + self.p_in, y - self.p_in:y + self.p_in, x - self.p_in:x + self.p_in]

        # Process the patches in order to be used by network:
        patch_data = (patch_data - torch.mean(patch_data)) / torch.std(patch_data)  # normalize
        patch_target_onehot = to_categorical(patch_target, self.Ncl)

        # Store into batch array:
        batch_data[:, :, :, 0] = patch_data
        batch_target = patch_target_onehot

        # Data augmentation (180degree rotation around tilt axis):
        if np.random.uniform() < 0.5:
            batch_data = torch.rot90(batch_data, k=2, dims=(0, 2))
            batch_target = torch.rot90(batch_target, k=2, dims=(0, 2))
        batch_data = transpose_to_channels_first(batch_data)
        batch_target = transpose_to_channels_first(batch_target)
        return batch_data, batch_target

