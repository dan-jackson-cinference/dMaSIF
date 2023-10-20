from __future__ import annotations

import torch
from pykeops.torch import LazyTensor
from torch import Tensor, nn


def knn_atoms(x, y, k):
    N, D = x.shape
    x_i = LazyTensor(x[:, None, :].contiguous())
    y_j = LazyTensor(y[None, :, :].contiguous())

    pairwise_distance_ij = ((x_i - y_j) ** 2).sum(-1)
    # pairwise_distance_ij.ranges = diagonal_ranges(x_batch, y_batch)

    # N.B.: KeOps doesn't yet support backprop through Kmin reductions...
    # dists, idx = pairwise_distance_ij.Kmin_argKmin(K=k,axis=1)
    # So we have to re-compute the values ourselves:
    idx = pairwise_distance_ij.argKmin(K=k, axis=1)  # (N, K)
    x_ik = y[idx.view(-1)].view(N, k, D)
    dists = ((x[:, None, :] - x_ik) ** 2).sum(-1)

    return idx, dists


def get_atom_features(x, y, y_atomtype, k=16):
    idx, dists = knn_atoms(x, y, k=k)  # (num_points, k)
    num_points, _ = idx.size()

    idx = idx.view(-1)
    dists = 1 / dists.view(-1, 1)
    _, num_dims = y_atomtype.size()

    feature = y_atomtype[idx, :]
    feature = torch.cat([feature, dists], dim=1)
    feature = feature.view(num_points, k, num_dims + 1)

    return feature


class AtomNet(nn.Module):
    def __init__(self, atom_dims: int):
        super().__init__()

        self.transform_types = nn.Sequential(
            nn.Linear(atom_dims, atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(atom_dims, atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(atom_dims, atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.embed = AtomEmbedding(atom_dims)

    def forward(self, xyz, atom_xyz, atom_types):
        # Run a DGCNN on the available information:
        atom_types = self.transform_types(atom_types)
        return self.embed(xyz, atom_xyz, atom_types)


class AtomEmbeddingMP(nn.Module):
    def __init__(self, atom_dims: int):
        super().__init__()
        self.k = 16
        self.n_layers = 3
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * atom_dims + 1, 2 * atom_dims + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * atom_dims + 1, atom_dims),
                )
                for _ in range(self.n_layers)
            ]
        )
        self.norm = nn.ModuleList(
            [nn.GroupNorm(2, atom_dims) for _ in range(self.n_layers)]
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atom_types):
        idx, dists = knn_atoms(x, y, k=self.k)  # N, 9, 7
        num_points = x.shape[0]
        num_dims = y_atom_types.shape[-1]

        point_emb = torch.ones_like(x[:, 0])[:, None].repeat(1, num_dims)
        for i in range(self.n_layers):
            features = y_atom_types[idx.reshape(-1), :]
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
            features = features.view(num_points, self.k, num_dims + 1)
            features = torch.cat(
                [point_emb[:, None, :].repeat(1, self.k, 1), features], dim=-1
            )  # N, 8, 13

            messages = self.mlp[i](features)  # N,8,6
            messages = messages.sum(1)  # N,6
            point_emb = point_emb + self.relu(self.norm[i](messages))

        return point_emb


class AtomAtomEmbeddingMP(nn.Module):
    def __init__(self, atom_dims: int):
        super().__init__()
        self.k = 17
        self.n_layers = 3

        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * atom_dims + 1, 2 * atom_dims + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * atom_dims + 1, atom_dims),
                )
                for _ in range(self.n_layers)
            ]
        )

        self.norm = nn.ModuleList(
            [nn.GroupNorm(2, atom_dims) for _ in range(self.n_layers)]
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atom_types):
        idx, dists = knn_atoms(x, y, k=self.k)  # N, 9, 7
        idx = idx[:, 1:]  # Remove self
        dists = dists[:, 1:]
        k = self.k - 1
        num_points = y_atom_types.shape[0]

        out = y_atom_types
        for i in range(self.n_layers):
            _, num_dims = out.size()
            features = out[idx.reshape(-1), :]
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
            features = features.view(num_points, k, num_dims + 1)
            features = torch.cat(
                [out[:, None, :].repeat(1, k, 1), features], dim=-1
            )  # N, 8, 13

            messages = self.mlp[i](features)  # N,8,6
            messages = messages.sum(1)  # N,6
            out = out + self.relu(self.norm[i](messages))

        return out


class AtomNetMP(nn.Module):
    def __init__(self, atom_dims: int):
        super().__init__()

        self.transform_types = nn.Sequential(
            nn.Linear(atom_dims, atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(atom_dims, atom_dims),
        )

        self.embed = AtomEmbeddingMP(atom_dims)
        self.atom_atom = AtomAtomEmbeddingMP(atom_dims)

    def forward(self, xyz: Tensor, atom_xyz: Tensor, atom_types: Tensor):
        atom_types = self.transform_types(atom_types)
        atom_types = self.atom_atom(atom_xyz, atom_xyz, atom_types)
        atom_types = self.embed(xyz, atom_xyz, atom_types)
        return atom_types


class AtomEmbedding(nn.Module):
    def __init__(self, atom_dims: int):
        super().__init__()

        self.k = 16
        self.D = atom_dims
        self.conv1 = nn.Linear(atom_dims + 1, atom_dims)
        self.conv2 = nn.Linear(atom_dims, atom_dims)
        self.conv3 = nn.Linear(2 * atom_dims, atom_dims)
        self.bn1 = nn.BatchNorm1d(atom_dims)
        self.bn2 = nn.BatchNorm1d(atom_dims)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atom_types):
        fx = get_atom_features(x, y, y_atom_types, k=self.k)
        fx = self.conv1(fx)
        fx = fx.view(-1, self.D)
        fx = self.bn1(self.relu(fx))
        fx = fx.view(-1, self.k, self.D)
        fx1 = fx.sum(dim=1, keepdim=False)

        fx = self.conv2(fx)
        fx = fx.view(-1, self.D)
        fx = self.bn2(self.relu(fx))
        fx = fx.view(-1, self.k, self.D)
        fx2 = fx.sum(dim=1, keepdim=False)
        fx = torch.cat((fx1, fx2), dim=-1)
        fx = self.conv3(fx)

        return fx
