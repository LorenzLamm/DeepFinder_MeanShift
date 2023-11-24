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
from deepfinder.utils.data_utils import load_tomogram, store_tomogram

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
    def __init__(self, path_data, path_target, path_points, max_len=None, return_also_marginal_points=False):
        self.data = np.load(path_data)
        self.target = np.load(path_target)
        self.points = np.load(path_points)
        self.margin = 4.
        if max_len is not None:
            self.data = self.data[:max_len]
            self.target = self.target[:max_len]
            self.points = self.points[:max_len]
        self.shape = self.data.shape
        self.return_also_marginal_points = return_also_marginal_points


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        points = self.points[idx]
        if len(self.shape) == 3:
            points = points[np.all((points[:, 0] > self.margin, points[:, 1] > self.margin), axis=0)]
            points = points[np.all((points[:, 0] < self.shape[1]-self.margin, points[:, 1] < self.shape[2]-self.margin), axis=0)]
        elif len(self.shape) == 4:
            points = points[np.all((points[:, 0] > self.margin, points[:, 1] > self.margin, points[:, 2] > self.margin), axis=0)]
            points = points[np.all((points[:, 0] < self.shape[1]-self.margin, points[:, 1] < self.shape[2]-self.margin, points[:, 2] < self.shape[3]-self.margin), axis=0)]
        else:
            raise IOError("WHats your sample shape??", self.shape)
        if not self.return_also_marginal_points:
            return torch.from_numpy(self.data[idx]).unsqueeze(0).float(), torch.from_numpy(points).float().squeeze(), \
                torch.from_numpy(self.target[idx]).unsqueeze(0).float()
        else:
            return torch.from_numpy(self.data[idx]).unsqueeze(0).float(), (torch.from_numpy(points).float().squeeze(), torch.from_numpy(self.points[idx]).float().squeeze()), \
                torch.from_numpy(self.target[idx]).unsqueeze(0).float()


    def store_sample_images(self, sample_folder, num_images):
        for i in range(num_images):
            sample = self[i]
            data = sample[0].numpy().squeeze()
            target = sample[2].numpy().squeeze()
            store_tomogram(os.path.join(sample_folder, "data_%i.mrc" % i), data)
            store_tomogram(os.path.join(sample_folder, "target_%i.mrc" % i), target[1])
            



class DeepFinder_dataset_shrec(DeepFinder_dataset_2D):
    def __init__(self, path_data, model_nrs, max_len=None, return_also_marginal_points=False, patch_shape=(52,52,52)):
        self.path_data = path_data
        self.model_nrs = model_nrs        
        self.patch_shape = patch_shape
        self.margin = 4.
        self.initialize_data()
        if max_len is not None:
            self.data = self.data[:max_len]
            self.target = self.target[:max_len]
            self.points = self.points[:max_len]
        self.shape = self.data.shape
        self.return_also_marginal_points = return_also_marginal_points

    def initialize_data(self):
        self.points = []
        self.data = []
        self.target = []
        for model_nr in self.model_nrs:
            model_folder = os.path.join(self.path_data, 'model_' + str(model_nr))
            tomo_file = os.path.join(model_folder, "reconstruction.mrc")
            occ_mask_file = os.path.join(model_folder, "occupancy_mask.mrc")
            particle_locations_file = os.path.join(model_folder, "particle_locations.txt")
            tomo = load_tomogram(tomo_file, normalize_data=True)
            occ_mask = load_tomogram(occ_mask_file, normalize_data=False)
            particle_locations, ignore_count = parse_particle_positions(particle_locations_file)
            occ_mask = occ_mask > ignore_count

            for x in range(0, tomo.shape[0] - self.patch_shape[0]-1, self.patch_shape[0]):
                for y in range(0, tomo.shape[1] - self.patch_shape[1]-1, self.patch_shape[1]):
                    for z in range(0, tomo.shape[2] - self.patch_shape[2]-1, self.patch_shape[2]):
                        # Reset the patch_pos for the current patch
                        current_patch_pos = []
                        # Check particle positions for the current patch
                        for position in particle_locations:
                            px, py, pz = position
                            if (x <= px < x + self.patch_shape[0] and
                                y <= py < y + self.patch_shape[1] and
                                z <= pz < z + self.patch_shape[2]):
                                current_patch_pos.append(position)
                        if len(current_patch_pos) < 2:
                            continue
                        current_patch_pos = np.stack(current_patch_pos, axis=0)
                        current_patch_pos[:, 0] -= x
                        current_patch_pos[:, 1] -= y
                        current_patch_pos[:, 2] -= z

                        cropped_current_patch_pos = current_patch_pos[np.all((current_patch_pos[:, 0] > self.margin, current_patch_pos[:, 1] > self.margin, current_patch_pos[:, 2] > self.margin), axis=0)]
                        cropped_current_patch_pos = cropped_current_patch_pos[np.all((cropped_current_patch_pos[:, 0] < self.patch_shape[0]-self.margin, cropped_current_patch_pos[:, 1] < self.patch_shape[1]-self.margin, cropped_current_patch_pos[:, 2] < self.patch_shape[2]-self.margin), axis=0)]
                        if cropped_current_patch_pos.shape[0] < 2:
                            continue

                        current_patch_pos = np.stack([current_patch_pos[:, 1], current_patch_pos[:, 0], current_patch_pos[:, 2]], axis=1)
                        
                        
                        self.points.append(current_patch_pos)
                        self.data.append(tomo[x:x+self.patch_shape[0],
                                              y:y+self.patch_shape[1],
                                              z:z+self.patch_shape[2]])
                        self.target.append(occ_mask[x:x+self.patch_shape[0],
                                              y:y+self.patch_shape[1],
                                              z:z+self.patch_shape[2]]*1.)
        self.data = np.stack(self.data, axis=0)
        self.target = np.stack(self.target, axis=0)
        self.target = np.expand_dims(self.target, axis=1)
        self.target = np.concatenate((1-self.target, self.target), axis=1)
                        


def draw_sphere_around_coord_from_grid(coords, radius, target, grid_coords):
    for coord in coords:
        x, y, z = coord
        target[(grid_coords[0] - x)**2 + (grid_coords[1] - y)**2 + (grid_coords[2] - z)**2 <= radius**2] = 1
    return target

class DeepFinder_dataset_experimental(DeepFinder_dataset_2D):
    def __init__(self, path_data, train_split, max_len=None, return_also_marginal_points=False, patch_shape=(52,52,52), sparse_points=False):
        self.path_data = path_data
        self.train_split = train_split # "train" "val" "test"
        self.patch_shape = patch_shape
        self.margin = 4.
        self.sphere_radius = 4.
        self.sparse_points = sparse_points
        self.initialize_data()
        
        if max_len is not None:
            self.data = self.data[:max_len]
            self.target = self.target[:max_len]
            self.points = self.points[:max_len]
        self.shape = self.data.shape
        self.return_also_marginal_points = return_also_marginal_points

    def initialize_data(self):
        self.points = []
        self.data = []
        self.target = []

        
        grid_coords = np.meshgrid(np.arange(self.patch_shape[0]), np.arange(self.patch_shape[0]), np.arange(self.patch_shape[0]))

        tomo_file = os.path.join(self.path_data, "IS002_291013_006_bin8.mrc")
        particle_locations_file = os.path.join(self.path_data, "IS002_291013_006.coords")
        
        tomo = load_tomogram(tomo_file, normalize_data=True)
        particle_locations = parse_particle_positions_experimental(particle_locations_file) / 8.
        
        for x in range(0, tomo.shape[0] - self.patch_shape[0]-1, self.patch_shape[0]):
            for y in range(0, tomo.shape[1] - self.patch_shape[1]-1, self.patch_shape[1]):
                for z in range(0, tomo.shape[2] - self.patch_shape[2]-1, self.patch_shape[2]):
                    # Reset the patch_pos for the current patch
                    current_patch_pos = []
                    # Check particle positions for the current patch
                    for position in particle_locations:
                        px, py, pz = position
                        if (x <= px < x + self.patch_shape[0] and
                            y <= py < y + self.patch_shape[1] and
                            z <= pz < z + self.patch_shape[2]):
                            current_patch_pos.append(position)
                    if len(current_patch_pos) < 2:
                        continue
                    current_patch_pos = np.stack(current_patch_pos, axis=0)
                    current_patch_pos[:, 0] -= x
                    current_patch_pos[:, 1] -= y
                    current_patch_pos[:, 2] -= z

                    cropped_current_patch_pos = current_patch_pos[np.all((current_patch_pos[:, 0] > self.margin, current_patch_pos[:, 1] > self.margin, current_patch_pos[:, 2] > self.margin), axis=0)]
                    cropped_current_patch_pos = cropped_current_patch_pos[np.all((cropped_current_patch_pos[:, 0] < self.patch_shape[0]-self.margin, cropped_current_patch_pos[:, 1] < self.patch_shape[1]-self.margin, cropped_current_patch_pos[:, 2] < self.patch_shape[2]-self.margin), axis=0)]
                    if cropped_current_patch_pos.shape[0] < 2:
                        continue

                    current_patch_pos = np.stack([current_patch_pos[:, 1], current_patch_pos[:, 0], current_patch_pos[:, 2]], axis=1)
                    cropped_current_patch_pos = np.stack([cropped_current_patch_pos[:, 1], cropped_current_patch_pos[:, 0], cropped_current_patch_pos[:, 2]], axis=1)

                    if self.sparse_points:
                        current_patch_pos = cropped_current_patch_pos[np.random.choice(cropped_current_patch_pos.shape[0], 1, replace=False)]
                        
                    target = np.zeros(self.patch_shape)
                    occ_mask = draw_sphere_around_coord_from_grid(current_patch_pos, self.sphere_radius, target, grid_coords)
                    
                    self.points.append(current_patch_pos)
                    self.data.append(tomo[x:x+self.patch_shape[0],
                                            y:y+self.patch_shape[1],
                                            z:z+self.patch_shape[2]])
                    self.target.append(occ_mask*1.)
        

        total_data_length = len(self.data)
        train_split = int(total_data_length*0.7)
        val_split = int(total_data_length*0.85)
        test_split = total_data_length

        print("Total data length: ", total_data_length)
        print("Train split: ", train_split)
        print("Val split: ", val_split)
        print("Test split: ", test_split)
        print("currently using: ", self.train_split)

        if self.train_split == 'train':
            self.data = self.data[:train_split]
            self.target = self.target[:train_split]
            self.points = self.points[:train_split]
        elif self.train_split == 'val':
            self.data = self.data[train_split:val_split]
            self.target = self.target[train_split:val_split]
            self.points = self.points[train_split:val_split]
        elif self.train_split == 'test':
            self.data = self.data[val_split:test_split]
            self.target = self.target[val_split:test_split]
            self.points = self.points[val_split:test_split]
        
        self.data = np.stack(self.data, axis=0)
        self.target = np.stack(self.target, axis=0)
        self.target = np.expand_dims(self.target, axis=1)
        self.target = np.concatenate((1-self.target, self.target), axis=1)

        print("This results in the following data shapes")
        print("Data: ", self.data.shape)
        print("Target: ", self.target.shape)
        print("Points: ", self.points[-1].shape)
            



def parse_particle_positions(file_path):
    # Initialize an empty list to store the particle positions
    particle_positions = []
    ignore_count = 0
    # Open the file for reading
    with open(file_path, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Split the line into components
            parts = line.split()
            # Check if the line corresponds to a particle
            if parts[0] not in ('vesicle', 'fiducial'):
                # Extract the particle positions (columns 2-4) and convert them to floats
                position = list(map(float, parts[1:4]))
                # Append the position to the list
                particle_positions.append(position)
            else:
                ignore_count += 1
    
    # Convert the list of positions to a numpy array
    particle_positions_array = np.array(particle_positions)
    
    return particle_positions_array, ignore_count


def parse_particle_positions_experimental(file_path):
    # Initialize an empty list to store the particle positions
    particle_positions = []
    # Open the file for reading
    with open(file_path, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Split the line into components
            parts = line.split()
            # Check if the line corresponds to a particle
            
            # Extract the particle positions (columns 2-4) and convert them to floats
            position = list(map(float, parts))
            # Append the position to the list
            particle_positions.append(position)
    
    # Convert the list of positions to a numpy array
    particle_positions_array = np.array(particle_positions)
    
    return particle_positions_array


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

