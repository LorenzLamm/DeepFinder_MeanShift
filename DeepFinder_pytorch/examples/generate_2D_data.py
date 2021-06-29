import numpy as np
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
import os


def sample_points(shape, num, min_dist):
    all_points = np.zeros((0, 2))
    for i in range(1000):
        rand_x, rand_y = np.random.randint(0, shape[0]), np.random.randint(0, shape[1])
        if i == 0:
            all_points = np.concatenate((all_points, np.expand_dims(np.array((rand_x, rand_y)), 0)), 0)
        else:
            dists = np.linalg.norm(all_points - np.array((rand_x, rand_y)), axis=1)
            if np.min(dists) > min_dist:
                all_points = np.concatenate((all_points, np.expand_dims(np.array((rand_x, rand_y)), 0)), 0)
        if all_points.shape[0] == num:
            break
    return all_points

def spheres_around_points(points, shape, radius):
    image_coord_x = np.tile(np.linspace(0, shape[0] - 1, shape[0]), (shape[1],1))
    image_coord_y = np.tile(np.linspace(0, shape[1] - 1, shape[1]), (shape[0],1)).transpose()
    positions = np.concatenate((np.expand_dims(image_coord_x, 2), np.expand_dims(image_coord_y, 2)), 2)
    positions = positions.reshape(shape[0] * shape[1], 2)
    dist_mat = distance_matrix(positions, points)
    dist_mat = np.min(dist_mat, axis=1)
    dist_mat = dist_mat.reshape(shape[0], shape[1])
    target = dist_mat < radius
    return target


def add_noise_to_image(target, std):
    noise = np.random.normal(0, std, target.shape)
    image = target + noise
    return image


def sample_images(out_dir, num_images, shape, num_points, min_dist, noise, train_val_test='train'):
    all_images, all_targets, all_points = [], [], []
    for i in range(num_images):
        points = sample_points(shape, num_points, min_dist)
        target = spheres_around_points(points, shape, 4)
        image = add_noise_to_image(target, noise)
        target_c1 = target == 0
        target_c2 = target == 1
        target = np.concatenate((np.expand_dims(target_c1, 0), np.expand_dims(target_c2, 0)), 0)
        all_images.append(image)
        all_targets.append(target)
        all_points.append(points)
    all_images = np.stack(all_images)
    all_targets = np.stack(all_targets)
    all_points = np.stack(all_points)
    points_out = os.path.join(out_dir, train_val_test + '_points.npy')
    images_out = os.path.join(out_dir, train_val_test + '_images.npy')
    targets_out = os.path.join(out_dir, train_val_test + '_targets.npy')
    np.save(points_out, all_points)
    np.save(images_out, all_images)
    np.save(targets_out, all_targets)



num_images_train = 512
num_images_val = 16
out_dir = '/Users/lorenz.lamm/PhD_projects/DeepFinder_MeanShift/2D_test_data/'
shape = [52, 52]
num_points = 10
min_dist = 12
noise = 0.5
sample_images(out_dir, num_images_train, shape, num_points, min_dist, noise, train_val_test='train')
sample_images(out_dir, num_images_val, shape, num_points, min_dist, noise, train_val_test='val')