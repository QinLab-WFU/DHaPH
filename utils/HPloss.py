import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
import utils.pmath as pmath


def _tensor_dot(x, y):
    res = torch.einsum("ij,kj->ik", (x, y))
    return res


def _mobius_addition_batch(x, y, c):
    xy = _tensor_dot(x, y)  # B x C
    x2 = x.pow(2).sum(-1, keepdim=True)  # B x 1
    y2 = y.pow(2).sum(-1, keepdim=True)  # C x 1
    num = 1 + 2 * c * xy + c * y2.permute(1, 0)  # B x C
    num = num.unsqueeze(2) * x.unsqueeze(1)
    num = num + (1 - c * x2).unsqueeze(2) * y  # B x C x D
    denom_part1 = 1 + 2 * c * xy  # B x C
    denom_part2 = c ** 2 * x2 * y2.permute(1, 0)
    denom = denom_part1 + denom_part2
    res = num / (denom.unsqueeze(2) + 1e-5)
    return res


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def _dist_matrix(x, y, c):
    sqrt_c = c ** 0.5
    return (
        2
        / sqrt_c
        * artanh(sqrt_c * torch.norm(_mobius_addition_batch(-x, y, c=c), dim=-1))
    )


def dist_matrix(x, y, c=1.0):
    c = torch.as_tensor(c).type_as(x)
    return _dist_matrix(x, y, c)


class ToPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Euclidean space
    to n-dim Poincare ball
    Also implements clipping from https://arxiv.org/pdf/2107.11472.pdf
    """

    def __init__(self, c, train_c=False, train_x=False, ball_dim=None, riemannian=True, clip_r=None):
        super(ToPoincare, self).__init__()
        if train_x:
            if ball_dim is None:
                raise ValueError(
                    "if train_x=True, ball_dim has to be integer, got {}".format(
                        ball_dim
                    )
                )
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        if train_c:
            self.c = nn.Parameter(torch.Tensor([c, ]))
        else:
            self.c = c

        self.train_x = train_x

        self.riemannian = pmath.RiemannianGradient
        self.riemannian.c = c

        self.clip_r = clip_r

        if riemannian:
            self.grad_fix = lambda x: self.riemannian.apply(x)
        else:
            self.grad_fix = lambda x: x

    def forward(self, x):
        if self.clip_r is not None:
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac = torch.minimum(
                torch.ones_like(x_norm),
                self.clip_r / x_norm
            )
            x = x * fac

        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return self.grad_fix(pmath.project(pmath.expmap(xp, x, c=self.c), c=self.c))
        return self.grad_fix(pmath.project(pmath.expmap0(x, c=self.c), c=self.c))

    def extra_repr(self):
        return "c={}, train_x={}".format(self.c, self.train_x)


class HPLoss(nn.Module):
    def __init__(self, nb_proxies, sz_embed, mrg=0.1, tau=0.1, hyp_c=0.1, clip_r=2.3):
        super().__init__()
        self.nb_proxies = nb_proxies
        self.sz_embed = sz_embed
        self.tau = tau
        self.hyp_c = hyp_c
        self.mrg = mrg
        self.clip_r = clip_r

        self.lcas = torch.randn(self.nb_proxies, self.sz_embed).to(0)
        self.lcas = self.lcas / math.sqrt(self.sz_embed) * clip_r * 0.9
        self.lcas = torch.nn.Parameter(self.lcas).to(0)
        self.to_hyperbolic = ToPoincare(c=hyp_c, ball_dim=sz_embed, riemannian=True, clip_r=clip_r, train_c=False)

        if hyp_c > 0:
            self.dist_f = lambda x, y: dist_matrix(x, y, c=hyp_c)
        else:
            self.dist_f = lambda x, y: 2 - 2 * F.linear(x, y)

    def compute_gHHC(self, z_s, lcas, dist_matrix, indices_tuple, sim_matrix):
        i, j, k = indices_tuple
        bs = len(z_s)

        cp_dist = dist_matrix

        max_dists_ij = torch.maximum(cp_dist[i], cp_dist[j])
        lca_ij_prob = F.gumbel_softmax(-max_dists_ij / self.tau, dim=1, hard=True)
        lca_ij_idx = lca_ij_prob.argmax(-1)

        max_dists_ijk = torch.maximum(cp_dist[k], max_dists_ij)
        lca_ijk_prob = F.gumbel_softmax(-max_dists_ijk / self.tau, dim=1, hard=True)
        lca_ijk_idx = lca_ijk_prob.argmax(-1)

        dist_i_lca_ij, dist_i_lca_ijk = (cp_dist[i] * lca_ij_prob).sum(1), (cp_dist[i] * lca_ijk_prob).sum(1)
        dist_j_lca_ij, dist_j_lca_ijk = (cp_dist[j] * lca_ij_prob).sum(1), (cp_dist[j] * lca_ijk_prob).sum(1)
        dist_k_lca_ij, dist_k_lca_ijk = (cp_dist[k] * lca_ij_prob).sum(1), (cp_dist[k] * lca_ijk_prob).sum(1)

        hc_loss = torch.relu(dist_i_lca_ij - dist_i_lca_ijk + self.mrg) \
                  + torch.relu(dist_j_lca_ij - dist_j_lca_ijk + self.mrg) \
                  + torch.relu(dist_k_lca_ijk - dist_k_lca_ij + self.mrg)

        hc_loss = hc_loss * (lca_ij_idx != lca_ijk_idx).float()
        loss = hc_loss.mean()

        return loss

    def get_reciprocal_triplets(self, sim_matrix, topk=20, t_per_anchor=100):
        anchor_idx, positive_idx, negative_idx = [], [], []

        topk_index = torch.topk(sim_matrix, topk)[1]
        nn_matrix = torch.zeros_like(sim_matrix).scatter_(1, topk_index, torch.ones_like(sim_matrix))
        sim_matrix = ((nn_matrix + nn_matrix.t()) / 2).float()
        sim_matrix = sim_matrix.fill_diagonal_(-1)

        for i in range(len(sim_matrix)):
            if len(torch.nonzero(sim_matrix[i] == 1)) <= 1:
                continue
            pair_idxs1 = np.random.choice(torch.nonzero(sim_matrix[i] == 1).squeeze().cpu().numpy(), t_per_anchor,
                                          replace=True)
            pair_idxs2 = np.random.choice(torch.nonzero(sim_matrix[i] < 1).squeeze().cpu().numpy(), t_per_anchor,
                                          replace=True)
            positive_idx.append(pair_idxs1)
            negative_idx.append(pair_idxs2)
            anchor_idx.append(np.ones(t_per_anchor) * i)
        anchor_idx = np.concatenate(anchor_idx)
        positive_idx = np.concatenate(positive_idx)
        negative_idx = np.concatenate(negative_idx)
        return anchor_idx, positive_idx, negative_idx

    def forward(self, z_s, t_s, y, topk=15):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        bs = len(z_s)

        Sim = torch.mm(y, y.T)
        hot_mat = (Sim > 0)

        lcas = self.to_hyperbolic(self.lcas)

        all_nodes = torch.cat([z_s, lcas])
        all_dist_matrix = self.dist_f(all_nodes, all_nodes)

        t_all_nodes = torch.cat([t_s, lcas])
        t_all_dist_matrix = self.dist_f(t_all_nodes, t_all_nodes)

        sim_matrix = torch.exp(-all_dist_matrix[:bs, :bs]).detach()
        sim_matrix[hot_mat] += 1
        sim_matrix2 = torch.exp(-all_dist_matrix[bs:, bs:]).detach()

        t_sim_matrix = torch.exp(-t_all_dist_matrix[:bs, :bs]).detach()
        t_sim_matrix[hot_mat] += 1
        t_sim_matrix2 = torch.exp(-t_all_dist_matrix[bs:, bs:]).detach()

        indices_tuple = self.get_reciprocal_triplets(sim_matrix, topk=topk, t_per_anchor=50)
        loss = self.compute_gHHC(z_s, lcas, all_dist_matrix[:bs, bs:], indices_tuple, sim_matrix)
        indices_tuple2 = self.get_reciprocal_triplets(sim_matrix2, topk=topk, t_per_anchor=50)
        loss += self.compute_gHHC(lcas, lcas, all_dist_matrix[bs:, bs:], indices_tuple2, sim_matrix2)

        t_indices_tuple = self.get_reciprocal_triplets(t_sim_matrix, topk=topk, t_per_anchor=50)
        t_loss = self.compute_gHHC(t_s, lcas, t_all_dist_matrix[:bs, bs:], t_indices_tuple, t_sim_matrix)
        t_indices_tuple2 = self.get_reciprocal_triplets(t_sim_matrix2, topk=topk, t_per_anchor=50)
        t_loss += self.compute_gHHC(lcas, lcas, t_all_dist_matrix[bs:, bs:], t_indices_tuple2, t_sim_matrix2)

        total_loss = loss + t_loss
        return total_loss