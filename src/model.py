from __future__ import annotations

import torch
from torch import Tensor, nn

from atomnet import AtomNetMP
from benchmark_models import dMaSIFConv_seg
from enums import Mode
from geometry_processing import curvatures
from helper import soft_dimension
from load_configs import ModelConfig, SearchModelConfig, SiteModelConfig


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
    ) -> tuple[Tensor, Tensor]:
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
        embedding_model: dMaSIFSiteEmbed,
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
        surface_xyz: Tensor,
        surface_normals: Tensor,
        atom_coords: Tensor,
        atom_types: Tensor,
    ):
        """Compute embeddings of the point clouds"""
        features = self.features(surface_xyz, surface_normals, atom_coords, atom_types)
        features = self.dropout(features)
        output_embedding_1, output_embedding_2 = self.embedding_model(
            surface_xyz, surface_normals, features
        )

        # Monitor the approximate rank of our representations:
        input_r_value = soft_dimension(features)
        conv_r_value = soft_dimension(output_embedding_1)

        return output_embedding_1, output_embedding_2, input_r_value, conv_r_value


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
    def from_config(cls, cfg: ModelConfig):
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

        embedding = self.conv(features, surface_xyz, nuv)
        return embedding


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
        surface_xyz: Tensor,
        surface_normals: Tensor,
        features: Tensor,
    ):
        output_embedding_1 = super().forward(surface_xyz, surface_normals, features)
        feature_scores_2 = self.orientation_scores2(features)
        nuv2 = self.conv2.load_mesh(
            surface_xyz,
            normals=surface_normals,
            weights=feature_scores_2,
        )
        return output_embedding_1, self.conv2(features, surface_xyz, nuv2)


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
