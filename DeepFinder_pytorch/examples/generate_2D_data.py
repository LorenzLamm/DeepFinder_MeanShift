import numpy as np
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import cv2



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

def get_ellipse(img, center, width, height, angle):
    cv2.ellipse(img, center=(int(center[0]), int(center[1])), axes=(width,height), angle=angle, startAngle=0, endAngle=360, color=(1,1,1), thickness=-1)
    return img

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def get_moon_centered_at_arc(radius):
    radius = int(radius)
    image_coord_x = np.tile(np.linspace(-radius, radius, 2*radius + 1), (2*radius + 1, 1))
    image_coord_y = np.tile(np.linspace(-radius, radius, 2*radius + 1), (2*radius + 1, 1)).transpose()
    positions = np.concatenate((np.expand_dims(image_coord_x, 2), np.expand_dims(image_coord_y, 2)), 2)
    positions = positions.reshape((2*radius + 1)**2, 2)

    dist_mat = distance_matrix(positions, np.array([[0,0]]))
    mask = dist_mat<radius
    circle_pos = positions[mask[:,0]] + radius
    circle_pos = np.array(circle_pos, dtype=np.int)


    circle_pos2 = circle_pos + 0.5*radius
    circle_pos2 = np.array(circle_pos2, dtype=np.int)

    moon_coords = []
    for coord in circle_pos:
        coord = coord.tolist()
        if not coord in circle_pos2.tolist():
            moon_coords.append(coord)
    moon_coords = np.stack(moon_coords)

    plt.subplot(1, 4, 1)
    img = np.zeros((2 * radius + 1, 2 * radius + 1))
    for coord in moon_coords:
        img[tuple(coord)] = 1
    plt.imshow(img)

    moon_coords -= radius
    new_moon_coords = []
    angle = np.random.uniform(0, 2 * np.pi)

    for coord in moon_coords:
        new_x, new_y = rotate(np.array([0, 0]), coord, angle)
        new_moon_coords.append(np.array([new_x, new_y]))
    moon_coords = np.stack(new_moon_coords)
    dist_mat = distance_matrix(positions, np.array(moon_coords))
    dist_mat = np.min(dist_mat, axis=1)
    mask = dist_mat < 0.8
    print(positions)
    print(moon_coords)
    moon_coords = positions[mask]
    print(moon_coords)



    moon_coords += radius
    plt.subplot(1, 4, 2)
    img = np.zeros((2 * radius + 1, 2 * radius + 1))
    for coord in moon_coords:
        img[tuple(np.array(coord, dtype=np.int))] = 1
    plt.imshow(img)

    moon_coords -= radius
    new_moon_coords = []
    angle = np.random.uniform(0, 2 * np.pi)

    for coord in moon_coords:
        new_x, new_y = rotate(np.array([0, 0]), coord, angle)
        new_moon_coords.append(np.array([new_x, new_y]))
    moon_coords = np.stack(new_moon_coords)
    moon_coords += radius

    plt.subplot(1, 4, 3)
    img = np.zeros((2 * radius + 1, 2 * radius + 1))
    for coord in moon_coords:
        img[tuple(np.array(coord, dtype=np.int))] = 1
    plt.imshow(img)

    moon_coords -= radius
    new_moon_coords = []
    angle = np.random.uniform(0, 2 * np.pi)

    for coord in moon_coords:
        new_x, new_y = rotate(np.array([0, 0]), coord, angle)
        new_moon_coords.append(np.array([new_x, new_y]))
    moon_coords = np.stack(new_moon_coords)
    moon_coords += radius

    plt.subplot(1, 4, 4)
    img = np.zeros((2 * radius + 1, 2 * radius + 1))
    for coord in moon_coords:
        img[tuple(np.array(coord, dtype=np.int))] = 1
    plt.imshow(img)
    plt.show()

    moon_coords -= radius
    new_moon_coords = []
    for coord in moon_coords:
        angle = np.random.uniform(0, 2 * np.pi)
        new_x, new_y = rotate(np.array([0, 0]), coord, angle)
        new_moon_coords.append(np.array([new_x, new_y]))
    moon_coords = np.stack(new_moon_coords)
    moon_coords += radius

    exit()


def ellipses_around_points(points):
    img = np.zeros((52,52))
    for point in points:

        width = int(np.random.uniform(2,7))
        height = int(np.random.uniform(2,7))
        angle = int(np.random.uniform(0,360))
        img = get_ellipse(img, point, width, height, angle)
    return img


def moons_around_points(points, shape, radii):
    add_val = 2.
    image_coord_x = np.tile(np.linspace(0, shape[0] - 1, shape[0]), (shape[1],1))
    image_coord_y = np.tile(np.linspace(0, shape[1] - 1, shape[1]), (shape[0],1)).transpose()
    positions = np.concatenate((np.expand_dims(image_coord_x, 2), np.expand_dims(image_coord_y, 2)), 2)


    exit()
    moon = get_moon_centered_at_arc(5.)
    positions = positions.reshape(shape[0] * shape[1], 2)
    for i in range(positions.shape[0]):
        add_val_x, add_val_y = get_random_orientations(add_val)
        positions[i, 0] += add_val_x
        positions[i, 1] += add_val_y

    dist_mat = distance_matrix(positions, points)
    dist_mat = np.min(dist_mat, axis=1)
    dist_mat = dist_mat.reshape(shape[0], shape[1])
    target = dist_mat < radii[0]

    positions = np.concatenate((np.expand_dims(image_coord_x, 2), np.expand_dims(image_coord_y, 2)), 2)
    positions = positions.reshape(shape[0] * shape[1], 2)
    positions[:, 0] += add_val_x*1.5
    positions[:, 1] += add_val_y*1.5

    dist_mat = distance_matrix(positions, points)
    dist_mat = np.min(dist_mat, axis=1)
    dist_mat = dist_mat.reshape(shape[0], shape[1])
    target2 = dist_mat < radii[1]

    target = target*1.0 - target2*1.0
    target[target <0] = 0
    plt.scatter(points[:, 0], points[:,1])
    plt.imshow(target)
    plt.show()
    exit()
    return target


def add_noise_to_image(target, std):
    noise = np.random.normal(0, std, target.shape)
    image = target + noise
    return image


def sample_images(out_dir, num_images, shape, num_points, min_dist, noise, train_val_test='train', case='spheres'):
    all_images, all_targets, all_points = [], [], []
    for i in range(num_images):
        points = sample_points(shape, num_points, min_dist)
        if case == 'spheres':
            target = spheres_around_points(points, shape, 4)
        elif case == 'moons':
            target = moons_around_points(points, shape, (5, 4))
        elif case == 'ellipses':
            target = ellipses_around_points(points)
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
    points_out = os.path.join(out_dir, train_val_test + '_points_' +case + '.npy')
    images_out = os.path.join(out_dir, train_val_test + '_images_' +case + '.npy')
    targets_out = os.path.join(out_dir, train_val_test + '_targets_' +case + '.npy')
    np.save(points_out, all_points)
    np.save(images_out, all_images)
    np.save(targets_out, all_targets)



num_images_train = 512
num_images_val = 16
out_dir = '/fs/pool/pool-engel/Lorenz/DeepFinder_MeanShift_RNN/toy_images'
shape = [52, 52]
num_points = 15
min_dist = 6
noise = 0.5
case = 'ellipses'
sample_images(out_dir, num_images_train, shape, num_points, min_dist, noise, train_val_test='train', case=case)
sample_images(out_dir, num_images_val, shape, num_points, min_dist, noise, train_val_test='val', case=case)