import numpy as np
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import cv2
from pyellipsoid import drawing as pyel


def sample_points(shape, num, min_dist, dist_to_boundary=5):
    all_points = np.zeros((0, len(shape)))
    for i in range(1000):
        rand_x, rand_y = np.random.randint(dist_to_boundary, shape[0]-dist_to_boundary), np.random.randint(dist_to_boundary, shape[1]-dist_to_boundary)
        if len(shape) == 3: rand_z = np.random.randint(dist_to_boundary, shape[2]-dist_to_boundary)
        cur_point = np.array((rand_x, rand_y))
        if len(shape) == 3:
            cur_point = np.array((rand_x, rand_y, rand_z))
        if i == 0:
            all_points = np.concatenate((all_points, np.expand_dims(cur_point, 0)), 0)
        else:
            dists = np.linalg.norm(all_points - cur_point, axis=1)
            if np.min(dists) > min_dist:
                all_points = np.concatenate((all_points, np.expand_dims(cur_point, 0)), 0)
        if all_points.shape[0] == num:
            break
    return all_points

def spheres_around_points(points, shape, radius):
    image_coord_x = np.tile(np.linspace(0, shape[0] - 1, shape[0]), (shape[1],1))
    image_coord_y = np.tile(np.linspace(0, shape[1] - 1, shape[1]), (shape[0],1)).transpose()
    image_coord_x = np.linspace(0, shape[0]-1, shape[0])
    image_coord_y = np.linspace(0, shape[1]-1, shape[1])
    if len(shape) == 2:
        image_coord_x, image_coord_y = np.meshgrid(image_coord_x, image_coord_y)
        positions = np.concatenate((np.expand_dims(image_coord_x, 2), np.expand_dims(image_coord_y, 2)), 2)
        positions = positions.reshape(shape[0] * shape[1], 2)
    if len(shape) == 3:
        image_coord_z = np.linspace(0, shape[2]-1, shape[2])
        image_coord_x, image_coord_y, image_coord_z = np.meshgrid(image_coord_x, image_coord_y, image_coord_z)
        positions = np.concatenate((np.expand_dims(image_coord_x, 3), np.expand_dims(image_coord_y, 3), np.expand_dims(image_coord_z, 3)), 3)
        positions = positions.reshape(shape[0] * shape[1] * shape[2], 3)
    dist_mat = distance_matrix(positions, points)
    dist_mat = np.min(dist_mat, axis=1)
    dist_mat = dist_mat.reshape(*shape)
    target = dist_mat < radius
    return target

def get_ellipse(img, center, width, height, angle):
    cv2.ellipse(img, center=(int(center[0]), int(center[1])), axes=(width,height), angle=angle, startAngle=0, endAngle=360, color=(1,1,1), thickness=-1)
    return img

def get_ellipse_3D(img, center, width, height, depth, angles=(90., 90., 90.)):
    shape = img.shape
    ell_center = center
    ell_radii = (width, height, depth)
    ell_angles = np.deg2rad(angles)
    binary = pyel.make_ellipsoid_image(shape, ell_center, ell_radii, ell_angles)
    return binary

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


def ellipses_around_points(points, shape):
    img = np.zeros(shape)
    for point in points:
        width = int(np.random.uniform(2,7))
        height = int(np.random.uniform(2,7))
        if len(img.shape) == 2:
            angle = int(np.random.uniform(0,360))
            img = get_ellipse(img, point, width, height, angle)
        elif len(img.shape) == 3:
            angles = (int(np.random.uniform(0,360)), int(np.random.uniform(0,360)), int(np.random.uniform(0,360)))
            depth = int(np.random.uniform(2,7))
            img = get_ellipse_3D(img, point, width, height, depth, angles)
    return img


def sandclocks_around_points(points, shape, radii=(5,5), shift_val=4):
    possible_shifts_pairs = [(np.array([0, 0, shift_val]), np.array([0, 0, -shift_val])),
                        (np.array([0, shift_val, 0]), np.array([0, -shift_val, 0])),
                        (np.array([shift_val, 0, 0]), np.array([-shift_val, 0, 0]))
                        ]
    # possible_shifts_pairs = [(np.array([a, b, c]), np.array([-a, -b, -c])) if (abs(a)+abs(b)+abs(c))==shift_val else None for a in range(-shift_val, shift_val+1) for b in range(-shift_val, shift_val+1) for c in range(-shift_val, shift_val+1)]
    possible_shifts_pairs = [(np.array([a, b, c]), np.array([-a, -b, -c])) if np.abs(np.linalg.norm(np.array([a, b, c])) - shift_val) < .3 else None for a in range(-shift_val, shift_val+1) for b in range(-shift_val, shift_val+1) for c in range(-shift_val, shift_val+1)]
    possible_shifts_pairs = [x for x in possible_shifts_pairs if x is not None]

    shifts = [possible_shifts_pairs[np.random.randint(len(possible_shifts_pairs))] for _ in range(points.shape[0])]
    points_1 = np.stack([point + shift[0] for (point, shift) in zip(points, shifts)])
    points_2 = np.stack([point + shift[1] for (point, shift) in zip(points, shifts)])
    spheres1 = spheres_around_points(points_1, shape, radius=radii[0])
    spheres2 = spheres_around_points(points_2, shape, radius=radii[1])
    sandclock_mask = np.logical_or(spheres1, spheres2)
    return sandclock_mask


def moons_around_points(points, shape, radii=(5,4), shift_val=3):
    big_spheres = spheres_around_points(points, shape, radius=radii[0])
    if len(shape) == 3:
        possible_shifts = [np.array([0, 0, shift_val]),
                        np.array([0, 0, -shift_val]),
                        np.array([0, shift_val, 0]),
                        np.array([0, -shift_val, 0]),
                        np.array([shift_val, 0, 0]),
                        np.array([-shift_val, 0, 0]),
                            ]
    elif len(shape) == 2:
        possible_shifts = [np.array([0, shift_val]),
                        np.array([0, -shift_val]),
                        np.array([shift_val, 0]),
                        np.array([-shift_val, 0]),
                            ]
    shifts = np.stack([possible_shifts[np.random.randint(len(possible_shifts))] for _ in range(points.shape[0])], axis=0)
    points_shifted = points + shifts
    small_spheres = spheres_around_points(points_shifted, shape, radius=radii[1])
    moon_mask = np.logical_and(big_spheres, np.logical_not(small_spheres))
    return moon_mask


def moons_around_points_old(points, shape, radii):
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
        print(i, "/", num_images)
        points = sample_points(shape, num_points, min_dist)
        if case == 'spheres':
            target = spheres_around_points(points, shape, 4)
        elif case == 'moons':
            target = moons_around_points(points, shape, (5, 4))
        elif case == 'ellipses':
            target = ellipses_around_points(points, shape)
        elif case == 'sandclock':
            target = sandclocks_around_points(points, shape, (5, 4))
        elif case == 'experimental':
            target = experimental_around_points(points, shape, (5, 4))
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
    points_out = os.path.join(out_dir, train_val_test + ('_3D' if len(shape) == 3 else '') +  '_points_' +case + '.npy')
    images_out = os.path.join(out_dir, train_val_test + ('_3D' if len(shape) == 3 else '') +'_images_' +case + '.npy')
    targets_out = os.path.join(out_dir, train_val_test + ('_3D' if len(shape) == 3 else '') +'_targets_' +case + '.npy')
    np.save(points_out, all_points)
    np.save(images_out, all_images)
    np.save(targets_out, all_targets)
    



num_images_train = 100
num_images_val = 10
num_images_test = 25
out_dir = '/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MeanShift_Loss/2D_test_data'
# shape = [52, 52, 52]
shape = [52, 52]
num_points = 15
# min_dist = 7
min_dist = 9
noise = 0.7
case = 'spheres'
sample_images(out_dir, num_images_train, shape, num_points, min_dist, noise, train_val_test='train', case=case)
sample_images(out_dir, num_images_val, shape, num_points, min_dist, noise, train_val_test='val', case=case)
sample_images(out_dir, num_images_test, shape, num_points, min_dist, noise, train_val_test='test', case=case)