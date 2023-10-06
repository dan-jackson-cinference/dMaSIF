import torch
from scipy.spatial.transform import Rotation
from torch_geometric.data import Data

tensor = torch.FloatTensor
inttensor = torch.LongTensor


class RandomRotationPairAtoms:
    r"""Randomly rotate a protein"""

    def __call__(self, data: Data):
        R1 = tensor(Rotation.random().as_matrix())
        R2 = tensor(Rotation.random().as_matrix())

        data.protein_1.atom_coords = torch.matmul(R1, data.protein_1.atom_coords.T).T
        data.protein_1.xyz = torch.matmul(R1, data.protein_1.xyz.T).T
        data.protein_1.normals = torch.matmul(R1, data.protein_1.normals.T).T

        data.protein_2.atom_coords_p2 = torch.matmul(R2, data.protein_2.atom_coords.T).T
        data.protein_2.xyz = torch.matmul(R2, data.protein_2.xyz.T).T
        data.protein_2.normals = torch.matmul(R2, data.protein_2.normals.T).T

        data.protein_1.rand_rot = R1
        data.protein_2.rand_rot = R2
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class CenterPairAtoms:
    r"""Centers a protein"""

    def __call__(self, data: Data):
        atom_center1 = data.protein_1.atom_coords.mean(dim=-2, keepdim=True)
        atom_center2 = data.protein_2.atom_coords.mean(dim=-2, keepdim=True)

        data.protein_1.atom_coords = data.protein_1.atom_coords - atom_center1
        data.protein_2.atom_coords = data.protein_2.atom_coords - atom_center2

        data.protein_1.xyz = data.protein_1.xyz - atom_center1
        data.protein_2.xyz = data.protein_2.xyz - atom_center2

        data.protein_1.atom_center = atom_center1
        data.protein_2.atom_center2 = atom_center2
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class NormalizeChemFeatures:
    r"""Centers a protein"""

    def __call__(self, data: Data):
        pb_upper = 3.0
        pb_lower = -3.0

        chem_p1 = data.protein_1.chemical_features
        chem_p2 = data.protein_2.chemical_features

        pb_p1 = chem_p1[:, 0]
        pb_p2 = chem_p2[:, 0]
        hb_p1 = chem_p1[:, 1]
        hb_p2 = chem_p2[:, 1]
        hp_p1 = chem_p1[:, 2]
        hp_p2 = chem_p2[:, 2]

        # Normalize PB
        pb_p1 = torch.clamp(pb_p1, pb_lower, pb_upper)
        pb_p1 = (pb_p1 - pb_lower) / (pb_upper - pb_lower)
        pb_p1 = 2 * pb_p1 - 1

        pb_p2 = torch.clamp(pb_p2, pb_lower, pb_upper)
        pb_p2 = (pb_p2 - pb_lower) / (pb_upper - pb_lower)
        pb_p2 = 2 * pb_p2 - 1

        # Normalize HP
        hp_p1 = hp_p1 / 4.5
        hp_p2 = hp_p2 / 4.5

        data.protein_1.chemical_features = torch.stack([pb_p1, hb_p1, hp_p1]).T
        data.protein_2.chemical_features = torch.stack([pb_p2, hb_p2, hp_p2]).T

        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)
