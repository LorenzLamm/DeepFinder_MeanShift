# ============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# ============================================================================================
# Copyright (c) 2019 - now
# Inria - Centre de Rennes Bretagne Atlantique, France
# Author: Emmanuel Moebel (serpico team); adapted by Lorenz Lamm
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# ============================================================================================

import torch
from torch.nn.functional import normalize


class Tversky_loss(torch.nn.Module):
    def __init__(self, dim=(0, 2, 3, 4)):
        super(Tversky_loss, self).__init__()
        self.dim = dim

    def forward(self, y_true, y_pred):
        # alpha = torch.Tensor([0.5], device=('cuda' if torch.cuda.is_available() else 'cpu)')
        alpha = torch.empty((1), device=('cuda' if torch.cuda.is_available() else 'cpu'))
        alpha[0] = 0.5
        beta = torch.empty((1), device=('cuda' if torch.cuda.is_available() else 'cpu'))
        beta[0] = 0.5
        # beta = torch.Tensor([0.5])
        ones = torch.ones_like(y_true)
        p0 = y_pred
        p1 = ones - y_pred
        g0 = y_true
        g1 = ones - y_true


        num = torch.sum(p0 * g0, dim=self.dim)
        if torch.cuda.is_available():
            den = num.cuda() + alpha.cuda() * torch.sum(p0.cuda() * g1.cuda(), dim=self.dim) + beta.cuda() * torch.sum(p1.cuda() * g0.cuda(), dim=self.dim)
        else:
            den = num + alpha * torch.sum(p0 * g1, dim=self.dim) + beta * torch.sum(p1 * g0, dim=self.dim)

        T = torch.sum(num / den)

        # Ncl = torch.Tensor(y_true.shape[-1])
        Ncl = torch.empty((1), device=('cuda' if torch.cuda.is_available() else 'cpu'))
        Ncl[0] = y_true.shape[1]
        return Ncl - T


class MeanShift_loss(torch.nn.Module):
    def __init__(self):
        super(MeanShift_loss, self).__init__()

    def forward(self, true_pos, pred_pos):
        dists = torch.cdist(true_pos.float(), pred_pos.float())
        mins, _ = torch.min(dists, dim=2)
        mins_seeds, _ = torch.min(dists, dim=1)
        loss = torch.mean(mins)
        loss_seeds = torch.mean(mins_seeds)
        return loss + loss_seeds, mins_seeds

class MeanShift_loss_directional(torch.nn.Module):
    def __init__(self):
        super(MeanShift_loss_directional, self).__init__()

    def forward(self, true_pos, pred_pos, seed_points):
        dists = torch.cdist(true_pos.float(), seed_points.float())
        closest_centers = torch.argmin(dists, dim=1)
        closest_centers = true_pos[0, closest_centers]
        true_directions = closest_centers - seed_points
        true_directions = normalize(true_directions, dim=2)
        pred_directions = pred_pos - seed_points
        pred_directions = normalize(pred_directions, dim=2)
        dot_prod = pred_directions[0] * true_directions[0]  # values in range -1:1 --> 1 is good, -1 is bad
        dot_prod = torch.sum(dot_prod, axis=1)
        dot_prod = 1. - dot_prod
        dot_bkp = dot_prod.clone().detach().cpu()
        dists_pred = torch.cdist(true_pos.float(), pred_pos.float())
        dists_pred, _ = torch.min(dists_pred, dim=1)
        dists_pred = dists_pred[0]
        #dot_prod *= dists_pred
        dot_prod = torch.mean(dot_prod)

        # loss = dot_prod #+ torch.mean(dists_pred)
        loss = torch.mean(dists_pred)
        # loss = torch.mean(dists_pred)

        print(dot_prod, loss)


        return loss, dot_bkp.unsqueeze(0)
