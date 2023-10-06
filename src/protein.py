from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pykeops.torch import LazyTensor
from torch import FloatTensor, LongTensor, Tensor

from geometry_processing import soft_distances, subsample


@dataclass
class Protein:
    """A class to hold protein features"""

    xyz: Tensor = None
    mesh_triangles: Tensor = None
    mesh_labels: Tensor = None
    chemical_features: Tensor = None
    normals: Tensor = None
    center_location: Tensor = None
    atom_coords: Tensor = None
    atom_types: Tensor = None
    num_nodes: Tensor = None
    rand_rot: Tensor | None = None
    surface_xyz: Tensor = field(init=False, repr=False)
    surface_normals: Tensor = field(init=False, repr=False)
    surface_labels: Tensor = field(init=False, repr=False)

    @classmethod
    def from_numpy(
        cls, pdb_id: str, data_dir: Path, center: bool = False, single_pdb: bool = False
    ) -> Protein:
        """Loads a protein surface mesh and its features"""
        # Load the data, and read the connectivity information:
        triangles = (
            None
            if single_pdb
            else torch.from_numpy(np.load(data_dir / (pdb_id + "_triangles.npy")))
            .type(torch.float32)
            .T
        )
        # Normalize the point cloud, as specified by the user:
        point_cloud = (
            None
            if single_pdb
            else torch.from_numpy(np.load(data_dir / (pdb_id + "_xyz.npy"))).type(
                torch.float32
            )
        )
        center_location = (
            None
            if single_pdb
            else torch.mean(point_cloud, axis=0, keepdims=True).type(torch.float32)
        )

        atom_coords = torch.from_numpy(
            np.load(data_dir / (pdb_id + "_atomxyz.npy"))
        ).type(torch.float32)
        atom_types = torch.from_numpy(
            np.load(data_dir / (pdb_id + "_atomtypes.npy"))
        ).type(torch.float32)

        if center:
            point_cloud = point_cloud - center_location
            atom_coords = atom_coords - center_location

        # Interface labels
        iface_labels = (
            None
            if single_pdb
            else torch.from_numpy(
                np.load(data_dir / (pdb_id + "_iface_labels.npy")).reshape((-1, 1))
            ).type(torch.float32)
        )

        # Features
        chemical_features = (
            None
            if single_pdb
            else torch.from_numpy(np.load(data_dir / (pdb_id + "_features.npy"))).type(
                torch.float32
            )
        )

        # Normals
        normals = (
            None
            if single_pdb
            else torch.from_numpy(np.load(data_dir / (pdb_id + "_normals.npy"))).type(
                torch.float32
            )
        )
        return cls(
            xyz=point_cloud,
            mesh_triangles=triangles,
            mesh_labels=iface_labels,
            chemical_features=chemical_features,
            normals=normals,
            center_location=center_location,
            num_nodes=None if single_pdb else point_cloud.shape[0],
            atom_coords=atom_coords,
            atom_types=atom_types,
        )

    def apply_random_rotation(self) -> None:
        """Apply a random rotation to the protein"""
        self.rand_rot = self.rand_rot.view(-1, 3, 3)[0]
        self.center_location = self.center_location.view(-1, 1, 3)[0]
        self.xyz = self.xyz - self.center_location
        self.xyz = (torch.matmul(self.rand_rot, self.xyz.T).T).contiguous()
        self.normals = (torch.matmul(self.rand_rot, self.normals.T).T).contiguous()

    def center_protein(self) -> None:
        """Center the protein on the origin"""
        self.rand_rot = torch.eye(3, device=self.xyz.device)
        self.atom_center = torch.zeros((1, 3), device=self.xyz.device)

    def atoms_to_points_normals(
        self,
        resolution: float = 1.0,
        sup_sampling: int = 20,
        distance: float = 1.05,
        smoothness: float = 0.5,
        nits: int = 4,
        variance: float = 0.1,
    ):
        """Turns a collection of atoms into an oriented point cloud.

        Sampling algorithm for protein surfaces, described in Fig. 3 of the paper.

        Args:
            atoms (Tensor): (N,3) coordinates of the atom centers `a_k`.
            resolution (float, optional): side length of the cubic cells in
                the final sub-sampling pass. Defaults to 1.0.
            distance (float, optional): value of the level set to sample from
                the smooth distance function. Defaults to 1.05.
            smoothness (float, optional): radii of the atoms, if atom types are
                not provided. Defaults to 0.5.
            nits (int, optional): number of iterations . Defaults to 4.


        Returns:
            (Tensor): (M,3) coordinates for the surface points `x_i`.
            (Tensor): (M,3) unit normals `n_i`.
            (integer Tensor): (M,) batch vector, as in PyTorch_geometric.
        """
        # a) Parameters for the soft distance function and its level set:
        n, d = self.atom_coords.shape

        # b) Draw N*B points at random in the neighborhood of our atoms
        z = self.atom_coords[:, None, :] + 10 * distance * torch.randn(
            n, sup_sampling, d
        ).type_as(self.atom_coords)
        z = z.view(-1, d)  # (N*B, D)
        # We don't want to backprop through a full network here!
        atoms = self.atom_coords.detach().contiguous()
        z = z.detach().contiguous()

        # N.B.: Test mode disables the autograd engine: we must switch it on explicitely.
        with torch.enable_grad():
            if z.is_leaf:
                z.requires_grad = True

            # c) Iterative loop: gradient descent along the potential
            # ".5 * (dist - T)^2" with respect to the positions z of our samples
            for _ in range(nits):
                dists = soft_distances(
                    atoms,
                    z,
                    smoothness=smoothness,
                    atomtypes=self.atom_types,
                )
                Loss = ((dists - distance) ** 2).sum()
                g = torch.autograd.grad(Loss, z)[0]
                z.data -= 0.5 * g

            # d) Only keep the points which are reasonably close to the level set:
            dists = soft_distances(
                atoms,
                z,
                smoothness=smoothness,
                atomtypes=self.atom_types,
            )
            margin = (dists - distance).abs()
            mask = margin < variance * distance

            # d') And remove the points that are trapped *inside* the protein:
            zz = z.detach()
            zz.requires_grad = True
            for _ in range(nits):
                dists = soft_distances(
                    atoms,
                    zz,
                    smoothness=smoothness,
                    atomtypes=self.atom_types,
                )
                Loss = (1.0 * dists).sum()
                g = torch.autograd.grad(Loss, zz)[0]
                normals = F.normalize(g, p=2, dim=-1)  # (N, 3)
                zz = zz + 1.0 * distance * normals

            dists = soft_distances(
                atoms,
                zz,
                smoothness=smoothness,
                atomtypes=self.atom_types,
            )
            mask = mask & (dists > 1.5 * distance)

            z = z[mask].contiguous().detach()

            # e) Subsample the point cloud:
            points = subsample(z, scale=resolution)

            # f) Compute the normals on this smaller point cloud:
            p = points.detach()
            p.requires_grad = True
            dists = soft_distances(
                atoms,
                p,
                smoothness=smoothness,
                atomtypes=self.atom_types,
            )
            Loss = (1.0 * dists).sum()
            g = torch.autograd.grad(Loss, p)[0]
            normals = F.normalize(g, p=2, dim=-1)  # (N, 3)
        points = points - 0.5 * normals
        return points.detach(), normals.detach()

    def project_iface_labels(self, threshold: float = 2.0):
        """We have to update the labels for the new point cloud representation"""
        queries = self.surface_xyz
        source = self.xyz
        labels = self.mesh_labels

        x_i = LazyTensor(queries[:, None, :])  # (N, 1, D)
        y_j = LazyTensor(source[None, :, :])  # (1, M, D)

        D_ij = ((x_i - y_j) ** 2).sum(-1).sqrt()  # (N, M)
        nn_i = D_ij.argmin(dim=1).view(-1)  # (N,)
        # If chain is not connected because of missing densities MaSIF cut out a part of the protein
        nn_dist_i = (D_ij.min(dim=1).view(-1, 1) < threshold).float()
        return labels[nn_i] * nn_dist_i

    def compute_surface_features(
        self, resolution: float, sup_sampling: int, distance: float
    ):
        """
        Compute a surface representation of a protein from a point cloud and
        surface normal vectors. This modifies protein.xyz, protein.normals and
        protein.mesh_labels
        """
        self.surface_xyz, self.surface_normals = self.atoms_to_points_normals(
            resolution, sup_sampling, distance
        )

        if self.mesh_labels is not None:
            self.surface_labels = self.project_iface_labels()

        if self.rand_rot is not None:
            self.xyz = torch.matmul(self.rand_rot.T, self.xyz.T).T + self.atom_center
            self.normals = torch.matmul(self.rand_rot.T, self.normals.T).T
