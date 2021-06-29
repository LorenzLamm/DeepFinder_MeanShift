import numpy as np
import lxml
from skimage import measure, filters
# from grow_membranes import load_tomogram, store_tomogram
import csv
from lxml.etree import Element
from lxml.etree import SubElement
from lxml.etree import ElementTree as ET
import sys
sys.path.append('/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/deepfinder/utils/')
import objl as ol


def get_connected_components(seg):
    all_labels = measure.label(seg, background=0)


def store_positions_in_xml_deepfinder(store_path, positions):
    """Store positions in a format that fits for DeepFinder
    :param store_path: specify wherre file should be stored
    :param positions: array of shape [N, 5], where N is the number of positions; for each position you need the following values:
            0: tomo_idx
            1: class_nr
            2-4: x,y,z coordinates in volume

    """

    root = Element('objlist')
    for pos in positions:
        obj = SubElement(root, 'object')
        obj.set('tomo_idx', str(int(pos[0])))
        obj.set('class_label', str(int(pos[1])))
        obj.set('x', str(pos[2]))
        obj.set('y', str(pos[3]))
        obj.set('z', str(pos[4]))
    tree = ET(root)
    tree.write(store_path, pretty_print=True)


def store_positions_in_csv(store_path, positions, out_del=','):
    with open(store_path, 'w') as csv_out_file:
        csv_writer = csv.writer(csv_out_file, delimiter=out_del)
        for i in range(positions.shape[0]):
            csv_writer.writerow(positions[i])


def convert_to_1_0_labelmask(in_file, out_file):
    """ If a segmentation mask has different values than one, all areas that are larger than zero will be labeled as 1"""
    seg = load_tomogram(in_file)
    seg = np.array(seg > 0, dtype=np.int8)
    store_tomogram(out_file, seg)


def extract_coords_for_file(in_seg, out_xml, tomo_idx, shrink_rate=100):
    print(in_seg)
    seg = load_tomogram(in_seg)
    print(np.unique(seg))
    coords = np.argwhere(seg)
    thres_coords = coords[np.array(range(coords.shape[0])) % shrink_rate == 0]
    tomo_idcs = np.ones((thres_coords.shape[0], 1)) * tomo_idx
    tomo_class_nrs = np.ones((thres_coords.shape[0], 1))
    thres_coords = np.concatenate((tomo_idcs, tomo_class_nrs, thres_coords), axis=1)
    store_positions_in_xml_deepfinder(out_xml, thres_coords)
    # store_positions_in_csv('/fs/pool/pool-engel/Lorenz/test_coords_mem.csv', thres_coords)


def mask_out_areas_of_tomogram(tomo_file, out_tomo, x_range=[(0, 927)], y_range=[(0,927)], z_range=[(0,463)]):
    tomo = load_tomogram(tomo_file)
    assert len(x_range) == len(y_range) and len(x_range) == len(z_range)
    for i in range(len(x_range)):
        cur_range_x = x_range[i]
        cur_range_y = y_range[i]
        cur_range_z = z_range[i]
        tomo[cur_range_x[0]:cur_range_x[1],
            cur_range_y[0]:cur_range_y[1],
            cur_range_z[0]: cur_range_z[1]] = 0


def get_hyperplane_from_points(x,y,z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    cp = np.cross(x-y, x-z)
    d = np.dot(cp, x)
    return np.array([cp[0], cp[1], cp[2], d])


def get_volume_mask_by_hyperplane(volume_shape, hyperplane_vec, upper=True):
    vol_mask = np.array(np.mgrid[[range(volume_shape[0]), range(volume_shape[1]), range(volume_shape[2])]])
    vol_mask = np.transpose(vol_mask, (1,2,3,0))
    # vol_mask = np.tile(vol_mask, volume_shape[2])

    vol_mask = np.dot(vol_mask, hyperplane_vec[:3])
    if upper:
        vol_mask = vol_mask > hyperplane_vec[3]
    else:
        vol_mask = vol_mask <= hyperplane_vec[3]
    return vol_mask




def mask_out_tomo_area(in_tomo, out_tomo, hyp_points_list, upper_list):
    tomo = load_tomogram(in_tomo)
    for i in range(len(hyp_points_list)):
        print(i, hyp_points_list[i])
        hyp = get_hyperplane_from_points(hyp_points_list[i][0], hyp_points_list[i][1], hyp_points_list[i][2])
        mask = get_volume_mask_by_hyperplane((928, 928, 464), hyp, upper=upper_list[i])
        tomo[mask] = 0
    # tomo = np.transpose(np.array(tomo), (2,1,0))
    store_tomogram(out_tomo, tomo)

def mask_out_hyperplanes():
    in_tomo = '/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/filled-volume_tomo17_binary.mrc'
    out_tomo = '/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/filled-volume_tomo17_binary_clean.mrc'
    upper_list = [False, False]
    hyperplane1 = [[(913, 927, 111), (1, 468, 219), (503, 133, 133)], [(423, 442, 283), (8, 751, 335), (917, 645, 214)]]
    mask_out_tomo_area(in_tomo, out_tomo, hyperplane1, upper_list)

# mask_out_hyperplanes()

def fuse_xml_files(xml_files, out_xml):
    all_objs = []
    for file in xml_files:
        print(file)
        cur_pos = ol.read_xml(file)

        all_objs += cur_pos
    ol.write_xml(all_objs, out_xml)


def reduce_points_to_fraction(in_file, out_file, keep_nth):
    cur_pos = np.array(ol.read_xml(in_file))
    mask = np.arange(cur_pos.shape[0])
    mask = [mask % keep_nth == 0]
    cur_pos = cur_pos[mask]
    cur_pos = cur_pos.tolist()
    ol.write_xml(cur_pos, out_file)




def extract_noise():
    ''' Subtracts clean tomogram from original one. This gives only the noisy version of the segmentation'''
    tomo1 = '/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/filled-volume_tomo17_binary.mrc'
    tomo2 = '/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/filled-volume_tomo17_binary_clean.mrc'
    tomo1 = load_tomogram(tomo1)
    tomo2 = load_tomogram(tomo2)
    tomo1 -= tomo2
    store_tomogram('/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/filled-volume_tomo17_binary_noise.mrc', tomo1)

# in_seg = '/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/tomo32_edges.labels.mrc'
# mask = convert_to_1_0_labelmask(in_seg, in_seg)
# exit()

# extract_coords_for_file('/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/tomo32_edges.labels_grow1.mrc',
#                       '/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/tomo_32_edges_pos.xml', tomo_idx=0)
# extract_coords_for_file('/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/filled-volume_tomo17_binary_clean.mrc',
#                         '/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/spinach17_filled_clean_pos.xml', tomo_idx=1)
# extract_coords_for_file('/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/filled-volume_tomo17_binary_noise.mrc',
#                         '/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/spinach17_filled_noise_pos.xml', tomo_idx=1, shrink_rate=1000)
# extract_coords_for_file('/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/tomo38_deepfinder_all_mbs.mrc',
#                         '/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/spinach38_pos.xml', tomo_idx=2, shrink_rate=100)
#
# in_xmls = ['/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/tomo_32_edges_pos.xml',
#            '/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/spinach17_filled_clean_pos.xml',
#            '/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/spinach17_filled_noise_pos.xml']
# out_xml = '/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/spinach17_32_fused_with_noise_pos.xml'
# fuse_xml_files(in_xmls, out_xml)

reduce_points_to_fraction('/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/spinach17_32_fused_with_noise_pos.xml',
                          '/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/spinach17_32_fused_with_noise_pos_reduced.xml', 1000)
reduce_points_to_fraction('/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/spinach38_pos.xml',
                          '/fs/pool/pool-engel/Lorenz/DeepFinder_membrane_segmentation/deep-finder-master/examples/training/in/spinach38_pos_reduced.xml', 1000)