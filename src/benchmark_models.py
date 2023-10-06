import torch
import torch.nn.functional as F
from pykeops.torch import LazyTensor
from torch import Tensor, nn
from torch_geometric.nn import (
    DynamicEdgeConv,
    PointNetConv,
    global_max_pool,
    knn_interpolate,
    radius,
)

from benchmark_layers import MyDynamicEdgeConv, MyXConv
from geometry_processing import dMaSIFConv, mesh_normals_areas, tangent_vectors
from helper import diagonal_ranges

DEConv = {"torch": DynamicEdgeConv, "keops": MyDynamicEdgeConv}

# Dynamic Graph CNNs ===========================================================
# Adapted from the PyTorch_geometric gallery to get a close fit to
# the original paper.


def MLP(channels: list[int], batch_norm: bool = True):
    """Multi-layer perceptron, with ReLU non-linearities and batch normalization."""
    return nn.Sequential(
        *[
            nn.Sequential(
                nn.Linear(channels[i - 1], channels[i]),
                nn.BatchNorm1d(channels[i]) if batch_norm else nn.Identity(),
                nn.LeakyReLU(negative_slope=0.2),
            )
            for i in range(1, len(channels))
        ]
    )


class DGCNN_seg(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_layers: int,
        knn: int = 40,
        aggr: str = "max",
        backend: str = "keops",
    ):
        super(DGCNN_seg, self).__init__()

        self.name = "DGCNN_seg_" + backend
        self.I, self.O = (
            in_channels + 3,
            out_channels,
        )  # Add coordinates to input channels
        self.n_layers = n_layers

        self.transform_1 = DEConv[backend](MLP([2 * 3, 64, 128]), knn, aggr)
        self.transform_2 = MLP([128, 1024])
        self.transform_3 = MLP([1024, 512, 256], batch_norm=False)
        self.transform_4 = nn.Linear(256, 3 * 3)

        self.conv_layers = nn.ModuleList(
            [DEConv[backend](MLP([2 * self.I, self.O, self.O]), knn, aggr)]
            + [
                DEConv[backend](MLP([2 * self.O, self.O, self.O]), knn, aggr)
                for _ in range(n_layers - 1)
            ]
        )

        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.O, self.O), nn.ReLU(), nn.Linear(self.O, self.O)
                )
                for _ in range(n_layers)
            ]
        )

        self.linear_transform = nn.ModuleList(
            [nn.Linear(self.I, self.O)]
            + [nn.Linear(self.O, self.O) for _ in range(n_layers - 1)]
        )

    def forward(self, positions, features, batch_indices):
        # Lab: (B,), Pos: (N, 3), Batch: (N,)
        pos, feat, batch = positions, features, batch_indices

        # TransformNet:
        x = pos  # Don't use the normals!

        x = self.transform_1(x, batch)  # (N, 3) -> (N, 128)
        x = self.transform_2(x)  # (N, 128) -> (N, 1024)
        x = global_max_pool(x, batch)  # (B, 1024)

        x = self.transform_3(x)  # (B, 256)
        x = self.transform_4(x)  # (B, 3*3)
        x = x[batch]  # (N, 3*3)
        x = x.view(-1, 3, 3)  # (N, 3, 3)

        # Apply the transform:
        x0 = torch.einsum("ni,nij->nj", pos, x)  # (N, 3)

        # Add features to coordinates
        x = torch.cat([x0, feat], dim=-1).contiguous()

        for i in range(self.n_layers):
            x_i = self.conv_layers[i](x, batch)
            x_i = self.linear_layers[i](x_i)
            x = self.linear_transform[i](x)
            x = x + x_i

        return x


# Reference PointNet models, from the PyTorch_geometric gallery =========================


class SAModule(torch.nn.Module):
    """Set abstraction module."""

    def __init__(self, ratio, r, nn, max_num_neighbors=64):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn)
        self.max_num_neighbors = max_num_neighbors

    def forward(self, x, pos, batch):
        # Subsample with Farthest Point Sampling:
        # idx = fps(pos, batch, ratio=self.ratio)  # Extract self.ratio indices TURN OFF FOR NOW
        idx = torch.arange(0, len(pos), device=pos.device)

        # For each "cluster", get the list of (up to 64) neighbors in a ball of radius r:
        row, col = radius(
            pos,
            pos[idx],
            self.r,
            batch,
            batch[idx],
            max_num_neighbors=self.max_num_neighbors,
        )

        # Applies the PointNet++ Conv:
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)

        # Return the features and sub-sampled point clouds:
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNet2_seg(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, radius: float, n_layers: int
    ):
        super(PointNet2_seg, self).__init__()

        self.name = "PointNet2"
        self.k = 10000  # We don't restrict the number of points in a patch

        # self.sa1_module = SAModule(1.0, self.radius, MLP([self.I+3, self.O, self.O]),self.k)
        self.layers = nn.ModuleList(
            [
                SAModule(
                    1.0,
                    radius,
                    MLP([in_channels + 3, out_channels, out_channels]),
                    self.k,
                )
            ]
            + [
                SAModule(
                    1.0,
                    radius,
                    MLP([out_channels + 3, out_channels, out_channels]),
                    self.k,
                )
                for _ in range(n_layers - 1)
            ]
        )

        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(out_channels, out_channels),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels),
                )
                for _ in range(n_layers)
            ]
        )

        self.linear_transform = nn.ModuleList(
            [nn.Linear(in_channels, out_channels)]
            + [nn.Linear(out_channels, out_channels) for _ in range(n_layers - 1)]
        )

    def forward(self, positions, features, batch_indices):
        x = (features, positions, batch_indices)
        for i, layer in enumerate(self.layers):
            x_i, pos, b_ind = layer(*x)
            x_i = self.linear_layers[i](x_i)
            x = self.linear_transform[i](x[0])
            x = x + x_i
            x = (x, pos, b_ind)

        return x[0]


## TangentConv benchmark segmentation


class DMaSIFConvBlock(torch.nn.Module):
    """Performs geodesic convolution on the point cloud"""

    def __init__(self, in_channels: int, out_channels: int, radius: float):
        super().__init__()
        self.linear_transform = nn.Linear(in_channels, out_channels)
        self.dmasif_conv = dMaSIFConv(in_channels, out_channels, radius, out_channels)
        self.linear_block = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, features: Tensor, points: Tensor, nuv: Tensor, ranges: Tensor):
        x = features
        x_i = self.dmasif_conv(points, nuv, x, ranges)
        x_i = self.linear_transform(features)
        x_i = self.linear_block(x_i)
        # x_i += x
        return x_i


class dMaSIFConv_seg(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, n_layers: int, radius: float
    ):
        super().__init__()

        self.name = "dMaSIFConv_seg_keops"
        self.radius = radius

        self.blocks = nn.ModuleList(
            [DMaSIFConvBlock(in_channels, out_channels, radius)]
            + [
                DMaSIFConvBlock(out_channels, out_channels, radius)
                for _ in range(n_layers - 1)
            ]
        )

    def forward(self, features: Tensor, points: Tensor, nuv: Tensor):
        # Lab: (B,), Pos: (N, 3), Batch: (N,)
        x = features

        for block in self.blocks:
            x = block(x, points, nuv, self.ranges)
        return x

    ### ORIGINAL CODE - THIS IS NOT DESCRIBED IN THE ARCHITECTURE IN THE PAPER
    # def forward(self, features: Tensor, nuv: Tensor):
    #     # Lab: (B,), Pos: (N, 3), Batch: (N,)
    #     x = features
    #     for i, layer in enumerate(self.layers):
    #         x_i = layer(self.points, nuv, x, self.ranges)
    #         x_i = self.linear_layers[i](x_i)
    #         x = self.linear_transform[i](x)
    #         x = x + x_i

    #     return x

    def load_mesh(
        self,
        xyz: Tensor | None,
        normals: Tensor | None = None,
        weights: Tensor | None = None,
        batch: Tensor | None = None,
    ) -> Tensor:
        """Loads the geometry of a triangle mesh.

        Input arguments:
        - xyz, a point cloud encoded as an (N, 3) Tensor.
        - weights, importance weights for the orientation estimation, encoded as an (N, 1) Tensor.
        - radius, the scale used to estimate the local normals.
        - a batch vector, following PyTorch_Geometric's conventions.

        The routine updates the model attributes:
        - points, i.e. the point cloud itself,
        - nuv, a local oriented basis in R^3 for every point,
        - ranges, custom KeOps syntax to implement batch processing.
        """

        # 1. Save the vertices for later use in the convolutions ---------------
        points = xyz
        # KeOps support for heterogeneous batch processing
        self.ranges = diagonal_ranges(batch)

        # 2. Estimate the normals and tangent frame ----------------------------
        # Normalize the scale:
        points = xyz / self.radius

        # Normals and local areas:
        tangent_bases = tangent_vectors(normals)  # Tangent basis (N, 2, 3)

        # 3. Steer the tangent bases according to the gradient of "weights" ----

        # 3.a) Encoding as KeOps LazyTensors:
        # Orientation scores:
        weights_j = LazyTensor(weights.view(1, -1, 1))  # (1, N, 1)
        # Vertices:
        x_i = LazyTensor(points[:, None, :])  # (N, 1, 3)
        x_j = LazyTensor(points[None, :, :])  # (1, N, 3)
        # Normals:
        n_i = LazyTensor(normals[:, None, :])  # (N, 1, 3)
        n_j = LazyTensor(normals[None, :, :])  # (1, N, 3)
        # Tangent basis:
        uv_i = LazyTensor(tangent_bases.view(-1, 1, 6))  # (N, 1, 6)

        # 3.b) Pseudo-geodesic window:
        # Pseudo-geodesic squared distance:
        rho2_ij = ((x_j - x_i) ** 2).sum(-1) * ((2 - (n_i | n_j)) ** 2)  # (N, N, 1)
        # Gaussian window:
        window_ij = (-rho2_ij).exp()  # (N, N, 1)

        # 3.c) Coordinates in the (u, v) basis - not oriented yet:
        X_ij = uv_i.matvecmult(x_j - x_i)  # (N, N, 2)

        # 3.d) Local average in the tangent plane:
        orientation_weight_ij = window_ij * weights_j  # (N, N, 1)
        orientation_vector_ij = orientation_weight_ij * X_ij  # (N, N, 2)

        # Support for heterogeneous batch processing:
        orientation_vector_ij.ranges = self.ranges  # Block-diagonal sparsity mask

        orientation_vector_i = orientation_vector_ij.sum(dim=1)  # (N, 2)
        orientation_vector_i = (
            orientation_vector_i + 1e-5
        )  # Just in case someone's alone...

        # 3.e) Normalize stuff:
        orientation_vector_i = F.normalize(orientation_vector_i, p=2, dim=-1)  #  (N, 2)
        ex_i, ey_i = (
            orientation_vector_i[:, 0][:, None],
            orientation_vector_i[:, 1][:, None],
        )  # (N,1)

        # 3.f) Re-orient the (u,v) basis:
        uv_i = tangent_bases  # (N, 2, 3)
        u_i, v_i = uv_i[:, 0, :], uv_i[:, 1, :]  # (N, 3)
        tangent_bases = torch.cat(
            (ex_i * u_i + ey_i * v_i, -ey_i * u_i + ex_i * v_i), dim=1
        ).contiguous()  # (N, 6)

        # 4. Store the local 3D frame as an attribute --------------------------
        return torch.cat((normals.view(-1, 1, 3), tangent_bases.view(-1, 2, 3)), dim=1)
