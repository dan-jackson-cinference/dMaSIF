from __future__ import annotations

import torch
from torch import Tensor, nn

from atomnet import AtomNetMP
from benchmark_models import DGCNN_seg, PointNet2_seg, dMaSIFConv_seg
from enums import Mode
from geometry_processing import curvatures
from helper import soft_dimension
from load_configs import ModelConfig, SearchModelConfig, SiteModelConfig
from protein import Protein


def combine_pair(
    p1: dict[str, Tensor | None], p2: dict[str, Tensor | None]
) -> dict[str, Tensor | None]:
    p1p2: dict[str, Tensor | None] = {}
    for key in p1:
        v1 = p1[key]
        v2 = p2[key]
        if v1 is None:
            continue

        if key in ("batch", "batch_atoms"):
            v1v2 = torch.cat([v1, v2 + v1[-1] + 1], dim=0)
        elif key == "triangles":
            # v1v2 = torch.cat([v1,v2],dim=1)
            continue
        else:
            v1v2 = torch.cat([v1, v2], dim=0)
        p1p2[key] = v1v2

    return p1p2


def split_pair(p1p2):
    batch_size = p1p2["batch_atoms"][-1] + 1
    p1_indices = p1p2["batch"] < batch_size // 2
    p2_indices = p1p2["batch"] >= batch_size // 2

    p1_atom_indices = p1p2["batch_atoms"] < batch_size // 2
    p2_atom_indices = p1p2["batch_atoms"] >= batch_size // 2

    p1 = {}
    p2 = {}
    for key in p1p2:
        v1v2 = p1p2[key]

        if key in ("rand_rot", "atom_center"):
            n = v1v2.shape[0] // 2
            p1[key] = v1v2[:n].view(-1, 3)
            p2[key] = v1v2[n:].view(-1, 3)
        elif "atom" in key:
            p1[key] = v1v2[p1_atom_indices]
            p2[key] = v1v2[p2_atom_indices]
        elif key == "triangles":
            continue
            # p1[key] = v1v2[:,p1_atom_indices]
            # p2[key] = v1v2[:,p2_atom_indices]
        else:
            p1[key] = v1v2[p1_indices]
            p2[key] = v1v2[p2_indices]

    p2["batch"] = p2["batch"] - batch_size + 1
    p2["batch_atoms"] = p2["batch_atoms"] - batch_size + 1

    return p1, p2


class BaseModel(nn.Module):
    def __init__(
        self,
        embedding_model: dMaSIFSiteEmbed | dMaSIFSearchEmbed,
        atom_dims: int,
        dropout: float,
        curvature_scales: list[float],
        no_chem: bool,
        no_geom: bool,
    ):
        super().__init__()

        self.atomnet = AtomNetMP(atom_dims)
        self.embedding_model = embedding_model
        self.dropout = nn.Dropout(dropout)
        self.curvature_scales = curvature_scales
        self.no_chem = no_chem
        self.no_geom = no_geom

    def features(
        self,
        surface_xyz: Tensor,
        surface_normals: Tensor,
        atom_coords: Tensor,
        atom_types: Tensor,
    ) -> Tensor:
        """Estimates geometric and chemical features from a protein surface or a cloud of atoms."""

        # Estimate the curvatures using the triangles or the estimated normals:
        protein_curvatures = curvatures(
            surface_xyz,
            normals=surface_normals,
            scales=self.curvature_scales,
        )

        # Compute chemical features on-the-fly:
        chem_feats = self.atomnet(surface_xyz, atom_coords, atom_types)

        if self.no_chem:
            chem_feats = 0.0 * chem_feats
        if self.no_geom:
            protein_curvatures = 0.0 * protein_curvatures

        # Concatenate our features:
        return torch.cat([protein_curvatures, chem_feats], dim=1).contiguous()

    def forward(
        self,
        surface_xyz: Tensor,
        surface_normals: Tensor,
        atom_coords: Tensor,
        atom_types: Tensor,
    ) -> tuple[float, float]:
        """Embeds all points of a protein in a high-dimensional vector space."""
        input_features = self.dropout(
            self.features(
                surface_xyz,
                surface_normals,
                atom_coords,
                atom_types,
            )
        )

        output_features = self.embedding_model(
            surface_xyz, surface_normals, input_features
        )

        return input_features, output_features


class SiteModel(BaseModel):
    """A model to find the binding site of a single protein"""

    def __init__(
        self,
        embedding_model: dMaSIFSiteEmbed,
        atom_dims: int,
        dropout: float,
        curvature_scales: list[float],
        no_chem: bool,
        no_geom: bool,
        emb_dims: int,
        post_units: int,
    ):
        super().__init__(
            embedding_model, atom_dims, dropout, curvature_scales, no_chem, no_geom
        )

        self.net_out = nn.Sequential(
            nn.Linear(emb_dims, post_units),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(post_units, post_units),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(post_units, 1),
        )

    @classmethod
    def from_config(
        cls,
        embedding_model: nn.Module,
        cfg: SiteModelConfig,
    ) -> SiteModel:
        """Instantiate a SiteModel object from a config"""
        return cls(
            embedding_model,
            cfg.atom_dims,
            cfg.dropout,
            cfg.curvature_scales,
            cfg.no_chem,
            cfg.no_geom,
            cfg.emb_dims,
            cfg.post_units,
        )

    def forward(
        self,
        surface_xyz: Tensor,
        surface_normals: Tensor,
        atom_coords: Tensor,
        atom_types: Tensor,
    ) -> tuple[Tensor, float, float]:
        """Compute embeddings of the point clouds"""
        input_features, output_embedding = super().forward(
            surface_xyz, surface_normals, atom_coords, atom_types
        )

        # Monitor the approximate rank of our representations:
        input_r_value = soft_dimension(input_features)
        conv_r_value = soft_dimension(output_embedding)
        interface_preds = self.net_out(output_embedding)

        return interface_preds, input_r_value, conv_r_value


class SearchModel(BaseModel):
    @classmethod
    def from_config(
        cls,
        embedding_model: dMaSIFSearchEmbed,
        cfg: SearchModelConfig,
    ) -> SearchModel:
        """Load a SearchModel from a config"""
        return cls(
            embedding_model,
            cfg.atom_dims,
            cfg.dropout,
            cfg.curvature_scales,
            cfg.no_chem,
            cfg.no_geom,
        )

    def forward(
        self,
        surface_xyz_1: Tensor,
        surface_normals_1: Tensor,
        atom_coords_1: Tensor,
        atom_types_1: Tensor,
        surface_xyz_2: Tensor,
        surface_normals_2: Tensor,
        atom_coords_2: Tensor,
        atom_types_2: Tensor,
    ):
        """Compute embeddings of the point clouds"""
        features_1 = self.features(
            surface_xyz_1, surface_normals_1, atom_coords_1, atom_types_1
        )
        features_1 = self.dropout(features_1)
        features_2 = self.features(
            surface_xyz_2, surface_normals_2, atom_coords_2, atom_types_2
        )
        features_2 = self.dropout(features_2)

        output_embedding = self.embedding_model(
            surface_xyz_1, surface_normals_1, surface_xyz_2, surface_normals_2
        )

        # Monitor the approximate rank of our representations:
        input_r_value = soft_dimension(input_features)
        conv_r_value = soft_dimension(output_embedding)

        return input_r_value, conv_r_value, output_embedding


class dMaSIFSiteEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        orientation_units: int,
        emb_dims: int,
        n_layers: int,
        radius: float,
    ):
        super().__init__()
        # Post-processing, without batch norm:
        self.orientation_scores = nn.Sequential(
            nn.Linear(in_channels, orientation_units),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(orientation_units, 1),
        )

        # Segmentation network:
        self.conv = dMaSIFConv_seg(
            in_channels=in_channels,
            out_channels=emb_dims,
            n_layers=n_layers,
            radius=radius,
        )

    @classmethod
    def from_config(cls, cfg: SiteModelConfig):
        "Create an instance of the DMasifSiteEmbed model from a config object"
        return cls(
            cfg.in_channels,
            cfg.orientation_units,
            cfg.emb_dims,
            cfg.n_layers,
            cfg.radius,
        )

    def forward(
        self, surface_xyz: Tensor, surface_normals: Tensor, features: Tensor
    ) -> Tensor:
        feature_scores = self.orientation_scores(features)
        nuv = self.conv.load_mesh(
            surface_xyz,
            surface_normals,
            weights=feature_scores,
        )
        return self.conv(features, surface_xyz, nuv)


class dMaSIFSearchEmbed(dMaSIFSiteEmbed):
    def __init__(
        self,
        in_channels: int,
        orientation_units: int,
        emb_dims: int,
        n_layers: int,
        radius: float,
    ):
        super().__init__(in_channels, orientation_units, emb_dims, n_layers, radius)
        self.orientation_scores2 = nn.Sequential(
            nn.Linear(in_channels, orientation_units),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(orientation_units, 1),
        )

        self.conv2 = dMaSIFConv_seg(
            in_channels=in_channels,
            out_channels=emb_dims,
            n_layers=n_layers,
            radius=radius,
        )

    def forward(
        self,
        surface_xyz_1: Tensor,
        surface_normals_1: Tensor,
        surface_xyz_2: Tensor,
        surface_normals_2: Tensor,
        features: Tensor,
    ):
        super().forward(surface_xyz_1, surface_normals_1, features)
        feature_scores_2 = self.orientation_scores2(features)
        nuv2 = self.conv2.load_mesh(
            surface_xyz_2,
            normals=surface_normals_2,
            weights=feature_scores_2,
        )
        return self.conv2(features, nuv2)


MODELS = {Mode.SEARCH: SearchModel, Mode.SITE: SiteModel}
EMBEDDING_MODELS = {
    Mode.SEARCH: dMaSIFSearchEmbed,
    Mode.SITE: dMaSIFSiteEmbed,
}


def load_model(mode: Mode, cfg: ModelConfig) -> BaseModel:
    """
    Choose the correct type of model depending on the desired mode
    and instantiate it from a ModelConfig
    """
    embedding_model = EMBEDDING_MODELS[mode].from_config(cfg)
    model = MODELS[mode].from_config(embedding_model, cfg)
    return model
