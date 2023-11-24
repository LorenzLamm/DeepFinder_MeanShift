import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as eu_dist
import multiprocessing as mp
from numpy.linalg import norm



class MeanShift_weighted():
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
        self.cluster_centers = []
        self.max_it = 10000
        self.convergence_thres = 1e-3 * bandwidth
        self.min_score = 0.
        self.max_score = 1.
        self.feat_dim = 3

    def _reset_globals(self):
        self.cluster_centers = []
        self.min_score = 0.
        self.max_score = 1.

    def _get_process_points_masks(self, points, n_pr):
        masks = []
        step = points.shape[0] / n_pr
        for i in range(n_pr - 1):
            mask_i = np.zeros(points.shape[0])
            mask_i[int(i * step): int((i + 1) * step)] = 1
            mask_i = mask_i > 0.5
            masks.append(mask_i)
        mask_last = np.zeros(points.shape[0])
        mask_last[int((n_pr - 1) * step):] = 1
        mask_last = mask_last > 0
        masks.append(mask_last)
        return masks


    def compute_weights(self, neigh_scores, weighting):
        if weighting is None:
            return np.ones_like(neigh_scores)
        min_max_range = self.max_score - self.min_score + 1e-7
        weight = (neigh_scores - self.min_score) / min_max_range
        if weighting == 'simple':
            return weight
        if weighting == 'quadratic':
            return weight**2
        return np.zeros_like(neigh_scores)


    def compute_kernel(self, neigh_dists, kernel):
        if kernel == 'gauss':
            return np.exp(-1 * neigh_dists**2 / (2 * self.bandwidth) ** 2)
        else:
            raise IOError('No valid kernel specified: ', kernel)


    def compute_weighted_average(self, neighbors, neigh_dists, neigh_scores, kernel, weighting):
        kernels = self.compute_kernel(neigh_dists, kernel)
        weights = self.compute_weights(neigh_scores, weighting)
        weighted_kernels = weights * kernels
        denominator = np.sum(weighted_kernels) + 1e-7
        numerator = np.sum(neighbors * np.tile(np.expand_dims(weighted_kernels, 1), (1,self.feat_dim)), axis=0)
        return numerator / denominator

    def process_single_point(self, point, points, scores, bandwidth, weighting, kernel, center_list):
        converged = False
        it = 0
        cur_cen = point
        while not converged and it < self.max_it:
            it += 1
            cur_distances = eu_dist(np.expand_dims(cur_cen,0), points)
            cur_distances = cur_distances[0]
            mask = cur_distances < bandwidth
            neighbors = points[mask]
            neigh_dists = cur_distances[mask]
            neigh_scores = scores[mask]
            new_cen = self.compute_weighted_average(neighbors, neigh_dists, neigh_scores, kernel, weighting)
            if norm(new_cen - cur_cen) < self.convergence_thres:
                converged = True
            cur_cen = new_cen
        center_list.append(cur_cen)
        return center_list

    def process_points_and_scores(self, pr_id, return_dict, points, scores, mask, bandwidth, weighting, kernel):
        pr_points = points[mask]
        center_list = []
        for i in range(pr_points.shape[0]):
            center_list = self.process_single_point(pr_points[i], points, scores, bandwidth, weighting, kernel, center_list)
        return_dict[pr_id] = center_list

    def post_cleanup(self, all_centers, dist_thres=1):
        idcs = np.array(range(all_centers.shape[0]))
        cluster_idcs = -1 * np.ones_like(idcs)
        cluster_centers = np.array(all_centers)
        new_cluster_centers = np.zeros((0, self.feat_dim))
        while cluster_centers.shape[0] != 0:
            center = cluster_centers[0]
            close_centers_mask = (eu_dist(np.expand_dims(center, 0), cluster_centers) < dist_thres)
            close_centers_mask = close_centers_mask[0]
            cluster_idcs[idcs[close_centers_mask]] = new_cluster_centers.shape[0]
            close_centers = cluster_centers[close_centers_mask]
            close_centers_mask = 1 - close_centers_mask
            cluster_centers = cluster_centers[close_centers_mask > 0]
            idcs = idcs[close_centers_mask > 0]
            new_cluster_centers = np.concatenate((new_cluster_centers, np.expand_dims(np.mean(close_centers, axis=0), 0)))
        return new_cluster_centers, cluster_idcs

    def mean_shift(self, points, scores, weighting=None, kernel='gauss', n_pr=1, fuse_dist=40.):
        assert points.shape[0] == scores.shape[0]
        if n_pr > points.shape[0]:
            n_pr = points.shape[0]
        self._reset_globals()
        self.min_score = np.min(scores)
        self.max_score = np.max(scores)
        self.feat_dim = points.shape[1]
        process_points_masks = self._get_process_points_masks(points, n_pr)
        processes = []
        manager = mp.Manager()
        return_dict = manager.dict()
        queue = mp.Queue()
        queue.put([])
        for pr_id in range(n_pr):
            mask = process_points_masks[pr_id]
            pr = mp.Process(target=self.process_points_and_scores, args=(pr_id, return_dict, points, scores, mask, self.bandwidth, weighting, kernel))
            pr.start()
            processes.append(pr)
        for pr_id in range(n_pr):
            pr = processes[pr_id]
            pr.join()
        all_centers = np.concatenate([return_dict[pr_id] for pr_id in range(n_pr)], 0)
        cluster_centers, cluster_idcs = self.post_cleanup(all_centers, dist_thres=fuse_dist)
        return cluster_centers, cluster_idcs


# test_points = np.array([[2.,.5], [2.,1.], [3.,1.]])
# weights = np.array([0.5,0.8,0.7])
# ms = MeanShift_weighted(bandwidth=0.5)
# cluster_centers = ms.mean_shift(test_points, weights, fuse_dist=1.0)
# print(cluster_centers)
