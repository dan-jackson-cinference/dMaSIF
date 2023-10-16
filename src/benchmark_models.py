from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from pykeops.torch import LazyTensor
from torch import Tensor, nn

from geometry_processing import tangent_vectors
from helper import diagonal_ranges


#  Fast tangent convolution layer ===============================================
class ContiguousBackward(torch.autograd.Function):
    """
    Function to ensure contiguous gradient in backward pass. To be applied after PyKeOps reduction.
    N.B.: This workaround fixes a bug that will be fixed in ulterior KeOp releases.
    """

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.contiguous()


class dMaSIFConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_units: int,
        out_channels: int,
        radius: float,
        cheap: bool = False,
    ):
        """Creates the KeOps convolution layer.

        I = in_channels  is the dimension of the input features
        O = out_channels is the dimension of the output features
        H = hidden_units is the dimension of the intermediate representation
        radius is the size of the pseudo-geodesic Gaussian window w_ij = W(d_ij)


        This affordable layer implements an elementary "convolution" operator
        on a cloud of N points (x_i) in dimension 3 that we decompose in three steps:

          1. Apply the MLP "net_in" on the input features "f_i". (N, I) -> (N, H)

          2. Compute H interaction terms in parallel with:
                  f_i = sum_j [ w_ij * conv(P_ij) * f_j ]
            In the equation above:
              - w_ij is a pseudo-geodesic window with a set radius.
              - P_ij is a vector of dimension 3, equal to "x_j-x_i"
                in the local oriented basis at x_i.
              - "conv" is an MLP from R^3 to R^H:
                 - with 1 linear layer if "cheap" is True;
                 - with 2 linear layers and C=8 intermediate "cuts" otherwise.
              - "*" is coordinate-wise product.
              - f_j is the vector of transformed features.

          3. Apply the MLP "net_out" on the output features. (N, H) -> (N, O)


        A more general layer would have implemented conv(P_ij) as a full
        (H, H) matrix instead of a mere (H,) vector... At a much higher
        computational cost. The reasoning behind the code below is that
        a given time budget is better spent on using a larger architecture
        and more channels than on a very complex convolution operator.
        Interactions between channels happen at steps 1. and 3.,
        whereas the (costly) point-to-point interaction step 2.
        lets the network aggregate information in spatial neighborhoods.

        Args:
            in_channels (int, optional): numper of input features per point. Defaults to 1.
            out_channels (int, optional): number of output features per point. Defaults to 1.
            radius (float, optional): deviation of the Gaussian window on the
                quasi-geodesic distance `d_ij`. Defaults to 1..
            hidden_units (int, optional): number of hidden features per point.
                Defaults to out_channels.
            cheap (bool, optional): shall we use a 1-layer deep Filter,
                instead of a 2-layer deep MLP? Defaults to False.
        """

        super().__init__()

        self.radius = radius
        self.cuts = 8  # Number of hidden units for the 3D MLP Filter.
        self.cheap = cheap

        # For performance reasons, we cut our "hidden" vectors
        # in n_heads "independent heads" of dimension 8.
        self.heads_dim = 8  # 4 is probably too small; 16 is certainly too big

        # We accept "Hidden" dimensions of size 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, ...
        if hidden_units < self.heads_dim:
            self.heads_dim = hidden_units

        if hidden_units % self.heads_dim != 0:
            raise ValueError(
                f"The dimension of the hidden units ({hidden_units})"
                + f"should be a multiple of the heads dimension ({self.heads_dim})."
            )
        else:
            self.n_heads = hidden_units // self.heads_dim

        # Transformation of the input features:
        self.net_in = nn.Sequential(
            nn.Linear(in_channels, hidden_units),  # (H, I) + (H,)
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_units, hidden_units),  # (H, H) + (H,)
            # nn.LayerNorm(self.Hidden),#nn.BatchNorm1d(self.Hidden),
            nn.LeakyReLU(negative_slope=0.2),
        )  #  (H,)
        self.norm_in = nn.GroupNorm(4, hidden_units)
        # self.norm_in = nn.LayerNorm(self.Hidden)
        # self.norm_in = nn.Identity()

        # 3D convolution filters, encoded as an MLP:
        if cheap:
            self.conv = nn.Sequential(
                nn.Linear(3, hidden_units), nn.ReLU()  # (H, 3) + (H,)
            )  # KeOps does not support well LeakyReLu
        else:
            self.conv = nn.Sequential(
                nn.Linear(3, self.cuts),  # (C, 3) + (C,)
                nn.ReLU(),  # KeOps does not support well LeakyReLu
                nn.Linear(self.cuts, hidden_units),
            )  # (H, C) + (H,)

        # Transformation of the output features:
        self.net_out = nn.Sequential(
            nn.Linear(hidden_units, out_channels),  # (O, H) + (O,)
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(out_channels, out_channels),  # (O, O) + (O,)
            # nn.LayerNorm(self.Output),#nn.BatchNorm1d(self.Output),
            nn.LeakyReLU(negative_slope=0.2),
        )  #  (O,)

        self.norm_out = nn.GroupNorm(4, out_channels)
        # self.norm_out = nn.LayerNorm(self.Output)
        # self.norm_out = nn.Identity()

        # Custom initialization for the MLP convolution filters:
        # we get interesting piecewise affine cuts on a normalized neighborhood.
        with torch.no_grad():
            nn.init.normal_(self.conv[0].weight)
            nn.init.uniform_(self.conv[0].bias)
            self.conv[0].bias *= 0.8 * (self.conv[0].weight ** 2).sum(-1).sqrt()

            if not cheap:
                nn.init.uniform_(
                    self.conv[2].weight,
                    a=-1 / np.sqrt(self.cuts),
                    b=1 / np.sqrt(self.cuts),
                )
                nn.init.normal_(self.conv[2].bias)
                self.conv[2].bias *= 0.5 * (self.conv[2].weight ** 2).sum(-1).sqrt()

    def forward(self, points: Tensor, nuv: Tensor, features: Tensor, ranges=None):
        """Performs a quasi-geodesic interaction step.

        points, local basis, in features  ->  out features
        (N, 3),   (N, 3, 3),    (N, I)    ->    (N, O)

        This layer computes the interaction step of Eq. (7) in the paper,
        in-between the application of two MLP networks independently on all
        feature vectors.

        Args:
            points (Tensor): (N,3) point coordinates `x_i`.
            nuv (Tensor): (N,3,3) local coordinate systems `[n_i,u_i,v_i]`.
            features (Tensor): (N,I) input feature vectors `f_i`.
            ranges (6-uple of integer Tensors, optional): low-level format
                to support batch processing, as described in the KeOps documentation.
                In practice, this will be built by a higher-level object
                to encode the relevant "batch vectors" in a way that is convenient
                for the KeOps CUDA engine. Defaults to None.

        Returns:
            (Tensor): (N,O) output feature vectors `f'_i`.
        """

        # 1. Transform the input features: -------------------------------------
        features = self.net_in(features)  # (N, I) -> (N, H)
        features = features.transpose(1, 0)[None, :, :]  # (1,H,N)
        features = self.norm_in(features)
        features = features[0].transpose(1, 0).contiguous()  # (1, H, N) -> (N, H)

        # 2. Compute the local "shape contexts": -------------------------------

        # 2.a Normalize the kernel radius:
        points = points / (sqrt(2.0) * self.radius)  # (N, 3)

        # 2.b Encode the variables as KeOps LazyTensors

        # Vertices:
        x_i = LazyTensor(points[:, None, :])  # (N, 1, 3)
        x_j = LazyTensor(points[None, :, :])  # (1, N, 3)

        # WARNING - Here, we assume that the normals are fixed:
        normals = (
            nuv[:, 0, :].contiguous().detach()
        )  # (N, 3) - remove the .detach() if needed

        # Local bases:
        nuv_i = LazyTensor(nuv.view(-1, 1, 9))  # (N, 1, 9)
        # Normals:
        n_i = nuv_i[:3]  # (N, 1, 3)

        n_j = LazyTensor(normals[None, :, :])  # (1, N, 3)

        # To avoid register spilling when using large embeddings, we perform our KeOps reduction
        # over the vector of length "self.Hidden = self.n_heads * self.heads_dim"
        # as self.n_heads reduction over vectors of length self.heads_dim (= "Hd" in the comments).
        head_out_features = []
        for head in range(self.n_heads):
            # Extract a slice of width Hd from the feature array
            head_start = head * self.heads_dim
            head_end = head_start + self.heads_dim
            head_features = features[
                :, head_start:head_end
            ].contiguous()  # (N, H) -> (N, Hd)

            # Features:
            f_j = LazyTensor(head_features[None, :, :])  # (1, N, Hd)

            # Convolution parameters:
            if self.cheap:
                # Extract a slice of Hd lines: (H, 3) -> (Hd, 3)
                A = self.conv[0].weight[head_start:head_end, :].contiguous()
                # Extract a slice of Hd coefficients: (H,) -> (Hd,)
                B = self.conv[0].bias[head_start:head_end].contiguous()
                AB = torch.cat((A, B[:, None]), dim=1)  # (Hd, 4)
                ab = LazyTensor(AB.view(1, 1, -1))  # (1, 1, Hd*4)
            else:
                A_1, B_1 = self.conv[0].weight, self.conv[0].bias  # (C, 3), (C,)
                # Extract a slice of Hd lines: (H, C) -> (Hd, C)
                A_2 = self.conv[2].weight[head_start:head_end, :].contiguous()
                # Extract a slice of Hd coefficients: (H,) -> (Hd,)
                B_2 = self.conv[2].bias[head_start:head_end].contiguous()
                a_1 = LazyTensor(A_1.view(1, 1, -1))  # (1, 1, C*3)
                b_1 = LazyTensor(B_1.view(1, 1, -1))  # (1, 1, C)
                a_2 = LazyTensor(A_2.view(1, 1, -1))  # (1, 1, Hd*C)
                b_2 = LazyTensor(B_2.view(1, 1, -1))  # (1, 1, Hd)

            # 2.c Pseudo-geodesic window:
            # Pseudo-geodesic squared distance:
            d2_ij = ((x_j - x_i) ** 2).sum(-1) * ((2 - (n_i | n_j)) ** 2)  # (N, N, 1)
            # Gaussian window:
            window_ij = (-d2_ij).exp()  # (N, N, 1)

            # 2.d Local MLP:
            # Local coordinates:
            X_ij = nuv_i.matvecmult(x_j - x_i)  # (N, N, 9) "@" (N, N, 3) = (N, N, 3)
            # MLP:
            if self.cheap:
                X_ij = ab.matvecmult(
                    X_ij.concat(LazyTensor(1))
                )  # (N, N, Hd*4) @ (N, N, 3+1) = (N, N, Hd)
                X_ij = X_ij.relu()  # (N, N, Hd)
            else:
                X_ij = a_1.matvecmult(X_ij) + b_1  # (N, N, C)
                X_ij = X_ij.relu()  # (N, N, C)
                X_ij = a_2.matvecmult(X_ij) + b_2  # (N, N, Hd)
                X_ij = X_ij.relu()

            # 2.e Actual computation:
            F_ij = window_ij * X_ij * f_j  # (N, N, Hd)
            F_ij.ranges = ranges  # Support for batches and/or block-sparsity

            head_out_features.append(
                ContiguousBackward().apply(F_ij.sum(dim=1))
            )  # (N, Hd)

        # Concatenate the result of our n_heads "attention heads":
        features = torch.cat(head_out_features, dim=1)  # n_heads * (N, Hd) -> (N, H)

        # 3. Transform the output features: ------------------------------------
        features = self.net_out(features)  # (N, H) -> (N, O)
        features = features.transpose(1, 0)[None, :, :]  # (1,O,N)
        features = self.norm_out(features)
        features = features[0].transpose(1, 0).contiguous()

        return features


class dMaSIFConvBlock(torch.nn.Module):
    """Performs geodesic convolution on the point cloud"""

    def __init__(self, in_channels: int, out_channels: int, radius: float):
        super().__init__()
        self.linear_transform = nn.Linear(in_channels, out_channels)
        self.dmasif_conv = dMaSIFConv(in_channels, out_channels, out_channels, radius)
        self.linear_block = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(
        self,
        features: Tensor,
        points: Tensor,
        nuv: Tensor,
        ranges: Tensor,
    ):
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
            [dMaSIFConvBlock(in_channels, out_channels, radius)]
            + [
                dMaSIFConvBlock(out_channels, out_channels, radius)
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
        xyz: Tensor,
        normals: Tensor,
        weights: Tensor,
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
