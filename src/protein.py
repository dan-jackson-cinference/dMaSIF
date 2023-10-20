from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from plyfile import PlyData
from pykeops.torch import LazyTensor
from torch import Tensor

from geometry_processing import soft_distances, subsample

ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}


def load_ply_file(ply_file: Path) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    "Load data from a .ply file, and return information about the pointcloud"
    plydata = PlyData.read(str(ply_file))

    triangles = torch.from_numpy(
        np.vstack(plydata["face"].data["vertex_indices"])
    ).T.type(torch.float32)

    # Normalize the point cloud, as specified by the user:
    points = torch.from_numpy(
        np.vstack([[v[0], v[1], v[2]] for v in plydata["vertex"]])
    ).type(torch.float32)

    nx = plydata["vertex"]["nx"]
    ny = plydata["vertex"]["ny"]
    nz = plydata["vertex"]["nz"]
    normals = torch.from_numpy(np.stack([nx, ny, nz]).T).type(torch.float32)

    # Interface labels
    iface_labels = (
        torch.from_numpy(plydata["vertex"]["iface"]).type(torch.float32).squeeze()
    )

    # Features
    charge = plydata["vertex"]["charge"]
    hbond = plydata["vertex"]["hbond"]
    hphob = plydata["vertex"]["hphob"]
    features = torch.from_numpy(np.stack([charge, hbond, hphob]).T).type(torch.float32)
    return points, triangles, iface_labels, features, normals


def load_pdb_file(pdb_file: Path) -> tuple[Tensor, Tensor]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", str(pdb_file))
    atoms = structure.get_atoms()

    coords = []
    types = []
    for atom in atoms:
        coords.append(atom.get_coord())
        types.append(ele2num[atom.element])

    coords = np.stack(coords)
    types_array = np.zeros((len(types), len(ele2num)))
    for i, t in enumerate(types):
        types_array[i, t] = 1.0

    return torch.from_numpy(coords).type(torch.float32), torch.from_numpy(
        types_array
    ).type(torch.float32)


@dataclass
class ProteinProtocol(Protocol):
    pdb_id: str
    surface_xyz: Tensor
    surface_normals: Tensor
    surface_labels: Tensor
    atom_coords: Tensor
    atom_types: Tensor
    split_idx: int = 0


@dataclass
class Protein:
    """A class to hold protein features"""

    pdb_id: str
    surface_xyz: Tensor
    surface_triangles: Tensor
    surface_normals: Tensor
    surface_labels: Tensor
    chemical_features: Tensor
    atom_coords: Tensor
    atom_types: Tensor
    num_nodes: int
    split_idx: int = 0

    @classmethod
    def from_ply_and_pdb(cls, pdb_id: str, surface_dir: Path, pdb_dir: Path) -> Protein:
        """Loads a .ply mesh to return a point cloud and connectivity."""
        ply_file = surface_dir / f"{pdb_id}.ply"
        pdb_file = pdb_dir / f"{pdb_id}.pdb"

        xyz, triangles, labels, features, normals = load_ply_file(ply_file)
        atom_coords, atom_types = load_pdb_file(pdb_file)

        return cls(
            pdb_id=pdb_id,
            surface_xyz=xyz,
            surface_triangles=triangles,
            surface_normals=normals,
            surface_labels=labels,
            chemical_features=features,
            atom_coords=atom_coords,
            atom_types=atom_types,
            num_nodes=xyz.shape[0],
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
        center = torch.mean(self.surface_xyz, dim=0, keepdims=True)
        self.surface_xyz -= center
        self.atom_coords -= center

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

    def project_iface_labels(self, threshold: float = 2.0) -> Tensor:
        """We have to update the labels for the new point cloud representation"""
        queries = self.surface_xyz
        source = self.xyz
        labels = self.mesh_labels[:, None]

        x_i = LazyTensor(queries[:, None, :])  # (N, 1, D)
        y_j = LazyTensor(source[None, :, :])  # (1, M, D)

        D_ij = ((x_i - y_j) ** 2).sum(-1).sqrt()  # (N, M)
        nn_i = D_ij.argmin(dim=1).view(-1)  # (N,)
        # If chain is not connected because of missing densities MaSIF cut out a part of the protein
        nn_dist_i = (D_ij.min(dim=1).view(-1, 1) < threshold).float()
        query_labels = (labels[nn_i] * nn_dist_i).squeeze()
        return query_labels


@dataclass
class PDBInferenceProtein:
    """
    A class representing a Protein used for inference, \
    where our starting information is only the atom types and coordinates
    """

    pdb_id: str
    atom_coords: Tensor
    atom_types: Tensor
    surface_xyz: Tensor = field(init=False, repr=False)
    surface_normals: Tensor = field(init=False, repr=False)
    surface_labels: Tensor = field(init=False, repr=False)

    @classmethod
    def from_pdb(cls, pdb_id: str, pdb_dir: Path) -> PDBInferenceProtein:
        "Load a PDB file into a Protein object"
        pdb_file = pdb_dir / f"{pdb_id}.pdb"

        atom_coords, atom_types = load_pdb_file(pdb_file)
        return cls(pdb_id, atom_coords, atom_types)

    def atoms_to_surface(
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

    def compute_surface_features(
        self, resolution: float = 1.0, sup_sampling: int = 20, distance: float = 1.05
    ):
        """
        Compute a surface representation of a protein from atom coordinates and types.
        Generates a surface point cloud and a normal vector for each point
        """
        self.surface_xyz, self.surface_normals = self.atoms_to_surface(
            resolution, sup_sampling, distance
        )


@dataclass
class ProteinPair:
    protein_1: Protein
    protein_2: Protein
    if_labels: Tensor

    @property
    def surface_xyz(self) -> Tensor:
        return torch.cat([self.protein_1.surface_xyz, self.protein_2.surface_xyz])

    @property
    def surface_normals(self) -> Tensor:
        return torch.cat(
            [self.protein_1.surface_normals, self.protein_2.surface_normals]
        )

    @property
    def atom_coords(self) -> Tensor:
        return torch.cat([self.protein_1.atom_coords, self.protein_2.atom_coords])

    @property
    def atom_types(self) -> Tensor:
        return torch.cat([self.protein_1.atom_types, self.protein_2.atom_types])

    @property
    def surface_labels(self) -> Tensor:
        return torch.cat([self.protein_1.surface_labels, self.protein_2.surface_labels])

    @property
    def split_idx(self) -> int:
        return len(self.protein_1.surface_xyz)
