import numpy as np
from matplotlib import pyplot as plt
import torch
from time import time
import os
from deepfinder.model_pylit import DeepFinder_model_2D_MS

os.environ['KMP_DUPLICATE_LIB_OK']='True'

shape = [50, 50, 50, 50]
if len(shape) == 4:
    s_max1 = shape[2]
    s_max2 = shape[3]
else:
    raise IOError('Wrong input shape!')
coords_x = torch.linspace(0, s_max1-1, s_max1)
coords_y = torch.linspace(0, s_max2-1, s_max2)
coords_x = coords_x.repeat(1,s_max1).view(-1, s_max1).transpose(1,0)
coords_y = coords_y.repeat(1,s_max1).view(-1, s_max1)
coordinate_grid = torch.cat((coords_x.unsqueeze(2), coords_y.unsqueeze(2)), dim=2)
coordinate_grid = coordinate_grid.reshape((-1,2))
coordinate_grid = coordinate_grid.repeat((50, 1))

time_zero = time()
coordinate_grid = torch.cat((coords_x.unsqueeze(2), coords_y.unsqueeze(2)), dim=2)
coordinate_grid = coordinate_grid.reshape((-1, 2))
coordinate_grid = coordinate_grid.repeat((50, 1))

seed_points = torch.from_numpy(np.array([[2.,3.], [12., 29.], [40., 12.]])).repeat((30,1)).float()
a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
print(a.shape, b.shape, coordinate_grid.shape, seed_points.shape)
print(torch.cdist(a, b))
time_zero = time()
print(torch.cdist(coordinate_grid, seed_points))
print(time() - time_zero)
# print(seed_points.shape, coordinate_grid.shape)
coordinate_grid = coordinate_grid.repeat((seed_points.shape[0],1,1))
for i in range(coordinate_grid.shape[0]):
    coordinate_grid[i] -= seed_points[i]
# print(coordinate_grid.shape)
coordinate_grid = coordinate_grid**2
# print(coordinate_grid.shape)
# print(seed_points.shape, coordinate_grid.shape)
# print(torch.max(coordinate_grid), torch.min(coordinate_grid))
print(time() - time_zero)

plt.figure()
toy_points = torch.from_numpy(np.array([[1.,3.], [0.9, 2.9], [4.,5.], [3.,5.], [4.,4.], [4.,6.], [5.,5.], [2.,4.], [2.5,4.5],
                                        [3.5,4.5], [3.5,5.5], [4.5,4.5], [4.5,5.5]]))
nn_weights = torch.from_numpy(np.array([[1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]]))
seeds = torch.from_numpy(np.array([[1.,3.]]))



model_2D = DeepFinder_model_2D_MS(2, 2, 2, 2, 2)
S, p_num = model_2D.mean_shift_for_seeds(toy_points, nn_weights, seeds, bandwidth=2.)
plt.scatter(toy_points[:, 0],toy_points[:, 1])
plt.scatter(seeds[:, 0], seeds[:, 1], c='green')
plt.scatter(S[:, 0], S[:, 1], c='red')
plt.show()
print(S)
