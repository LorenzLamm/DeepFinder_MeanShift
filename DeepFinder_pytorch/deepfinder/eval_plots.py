"""Here are evaluation plots for mean shift clustering!"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def eval_plots_2D(batch_target, pred, mins_seeds, seeds, img, batch_data, gt_img, folder_path, idx=0):
    batch_target_np = to_numpy(batch_target[0])
    pred_np = to_numpy(pred[0])
    seeds_np = to_numpy(seeds)
    mins_seeds_np = to_numpy(mins_seeds[0])
    img_np = to_numpy(img[0, 0])
    batch_data_np = to_numpy(batch_data[0, 0])

    # Scatter plot for batch targets, predictions, and seeds
    plt.figure()
    plt.scatter(batch_target_np[:, 0], batch_target_np[:, 1], c='black', s=100)
    plt.scatter(pred_np[:, 0], pred_np[:, 1], c=mins_seeds_np)
    plt.scatter(seeds_np[:, 0], seeds_np[:, 1], c='yellow')
    plt.colorbar()
    plt.savefig(f'{folder_path}/loss_vals.png')

    # Scatter plots and images for predictions, seeds, batch targets, img, and ground truth
    plt.figure(figsize=(25, 10))


    plt.subplot(1, 5, 1)
    plt.imshow(batch_data_np, cmap='gray')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 5, 2)
    # plt.imshow(gt_img, cmap='gray')
    plt.imshow(np.zeros(gt_img.shape), cmap='gray')
    plt.scatter(batch_target_np[:, 0], batch_target_np[:, 1], c='yellow')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 5, 3)
    plt.imshow(img_np, cmap='gray')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 5, 4)
    # plt.scatter(pred_np[:, 0], pred_np[:, 1], c='red')
    plt.scatter(seeds_np[:, 0], seeds_np[:, 1], c='green')
    # plt.scatter(batch_target_np[:, 0], batch_target_np[:, 1], c='blue')
    plt.imshow(img_np, cmap='gray')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 5, 5)
    plt.scatter(pred_np[:, 0], pred_np[:, 1], c='red')
    plt.scatter(seeds_np[:, 0], seeds_np[:, 1], c='green')
    # plt.scatter(batch_target_np[:, 0], batch_target_np[:, 1], c='blue')
    plt.imshow(img_np, cmap='gray')
    plt.colorbar(fraction=0.046, pad=0.04)




    plt.savefig(f'{folder_path}/progress{idx}.png')


def eval_plots_3D(batch_target, pred, mins_seeds, seeds, img, batch_data, gt_img, folder_path, idx=0):
    batch_target_np = to_numpy(batch_target[0])
    pred_np = to_numpy(pred[0])
    seeds_np = to_numpy(seeds)
    mins_seeds_np = to_numpy(mins_seeds[0])
    img_np = to_numpy(img[0, 0])
    batch_data_np = to_numpy(batch_data[0, 0])
    # Scatter plot for batch targets, predictions, and seeds
    plt.figure()
    plt.scatter(batch_target_np[:, 0], batch_target_np[:, 1], c='black', s=100)
    plt.scatter(pred_np[:, 0], pred_np[:, 1], c=mins_seeds_np)
    plt.scatter(seeds_np[:, 0], seeds_np[:, 1], c='yellow')
    plt.colorbar()
    plt.savefig(f'{folder_path}/loss_vals.png')

    # Scatter plots and images for predictions, seeds, batch targets, img, and ground truth
    plt.figure(figsize=(20, 20))

    plt.subplot(1, 4, 1)
    plt.scatter(pred_np[:, 0], pred_np[:, 1], c='red')
    plt.scatter(seeds_np[:, 0], seeds_np[:, 1], c='green')
    plt.scatter(batch_target_np[:, 0], batch_target_np[:, 1], c='blue')
    plt.imshow(img_np[:, :, 25], cmap='gray')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 4, 2)
    plt.imshow(img_np[:, :, 25], cmap='gray')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 4, 3)
    plt.imshow(batch_data_np[:, :, 25], cmap='gray')
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 4, 4)
    # plt.imshow(gt_img[:, :, 25], cmap='gray')
    plt.imshow(np.sum(gt_img, axis=2), cmap='gray')
    plt.scatter(batch_target_np[:, 0], batch_target_np[:, 1], c='blue')

    plt.savefig(f'{folder_path}/progress3D{idx}.png')
    plt.close()

# Example usage
# eval_plots(batch_target, pred, mins_seeds, seeds, img, batch_data, gt_img, 'path/to/folder')