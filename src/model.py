from __future__ import annotations

from abc import abstractmethod

import torch
from pykeops.torch import LazyTensor
from torch import Tensor, nn

from atomnet import AtomNetMP
from benchmark_models import dMaSIFConv_seg
from geometry_processing import curvatures
from helper import soft_dimension
from load_configs import ModelConfig, SearchModelConfig, SiteModelConfig


def split_feature(feature: Tensor, split_idx: int) -> tuple[Tensor, Tensor]:
    return feature[:split_idx], feature[split_idx:]


def accumulate_predictions(embedding_1: Tensor, embedding_2: Tensor) -> Tensor:
    lazy_emb_1 = LazyTensor(embedding_1[:, None, :])
    lazy_emb_2 = LazyTensor(embedding_2[None, :, :])

    features = lazy_emb_1 | lazy_emb_2

    # We want to do take the max(), but there is currently no backpropagation
    # through LazyTensor.max()reduction so need to workaround for now:
    # preds_1 = features.max(1).squeeze()
    # preds_2 = features.max(0).squeeze()

    # Find the indexes of the feature that corresponds to the max value
    pred_arg_max_1 = features.argmax(1).squeeze()
    pred_arg_max_2 = features.argmax(0).squeeze()

    # Select those features
    target_features_1 = embedding_2.index_select(0, pred_arg_max_1)
    target_features_2 = embedding_1.index_select(0, pred_arg_max_2)

    # Do row by row dot product, using the fact that
    # diag(AB^T) = sum(A*B, axis=1) where * denotes the elementwise product (Hadamard product)
    preds_1 = (embedding_1 * target_features_1).sum(1)
    preds_2 = (embedding_2 * target_features_2).sum(1)

    return torch.cat([preds_1, preds_2])


class BaseModel(nn.Module):
    def __init__(
        self,
        atom_dims: int,
        dropout: float,
        curvature_scales: list[float],
        no_chem: bool,
        no_geom: bool,
    ):
        super().__init__()

        self.atomnet = AtomNetMP(atom_dims)
        self.dropout = nn.Dropout(dropout)
        self.curvature_scales = curvature_scales
        self.no_chem = no_chem
        self.no_geom = no_geom

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> BaseModel:
        "Create a SiteModel or a SearchModel from the ModelCfg"
        if cfg.mode == "site":
            return SiteModel.from_config(cfg)
        if cfg.mode == "search":
            return SearchModel.from_config(cfg)
        raise ValueError(f"Mode must either be 'search' or 'site', not '{cfg.mode}'")

    def compute_features(
        self,
        surface_xyz: Tensor,
        surface_normals: Tensor,
        atom_coords: Tensor,
        atom_types: Tensor,
    ) -> Tensor:
        """Estimates geometric and chemical features from a protein surface or a cloud of atoms."""

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

    @abstractmethod
    def compute_predictions(
        self,
        surface_xyz: Tensor,
        surface_normals: Tensor,
        atom_coords: Tensor,
        atom_types: Tensor,
        split_idx: int,
    ) -> Tensor:
        "Predict the class of each point in the point cloud"
        raise NotImplementedError

    @abstractmethod
    def compute_embeddings(
        self,
        surface_xyz: Tensor,
        surface_normals: Tensor,
        atom_coords: Tensor,
        atom_types: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    # def forward(
    #     self,
    #     surface_xyz: Tensor,
    #     surface_normals: Tensor,
    #     atom_coords: Tensor,
    #     atom_types: Tensor,
    #     split_idx: int,
    # ) -> Tensor:
    #     return self.compute_predictions(
    #         surface_xyz, surface_normals, atom_coords, atom_types, split_idx
    #     )


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
        super().__init__(atom_dims, dropout, curvature_scales, no_chem, no_geom)
        self.embedding_model = embedding_model

        self.net_out = nn.Sequential(
            nn.Linear(emb_dims, post_units),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(post_units, post_units),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(post_units, 1),
        )

    @classmethod
    def from_config(cls, cfg: SiteModelConfig) -> SiteModel:
        """Instantiate a SiteModel object from a config"""
        embedding_model = dMaSIFSiteEmbed.from_config(cfg)
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

    def compute_embeddings(
        self,
        surface_xyz: Tensor,
        surface_normals: Tensor,
        atom_coords: Tensor,
        atom_types: Tensor,
    ) -> Tensor:
        """Compute embeddings of the point clouds"""
        features = self.compute_features(
            surface_xyz, surface_normals, atom_coords, atom_types
        )
        features = self.dropout(features)
        embed = self.embedding_model(surface_xyz, surface_normals, features)
        return embed

    def compute_predictions(self, embedding: Tensor) -> Tensor:
        interface_preds = self.net_out(embedding)

        return interface_preds

    def forward(
        self,
        surface_xyz: Tensor,
        surface_normals: Tensor,
        atom_coords: Tensor,
        atom_types: Tensor,
    ):
        embedding = self.compute_embeddings(
            surface_xyz, surface_normals, atom_coords, atom_types
        )
        predictions = self.compute_predictions(embedding)
        return predictions


class SearchModel(BaseModel):
    """A model to predict the interaction site between two proteins"""

    def __init__(
        self,
        embedding_model: dMaSIFSearchEmbed,
        atom_dims: int,
        dropout: float,
        curvature_scales: list[float],
        no_chem: bool,
        no_geom: bool,
    ):
        super().__init__(atom_dims, dropout, curvature_scales, no_chem, no_geom)
        self.embedding_model = embedding_model

    @classmethod
    def from_config(cls, cfg: SearchModelConfig) -> SearchModel:
        """Load a SearchModel from a config"""
        embedding_model = dMaSIFSearchEmbed.from_config(cfg)
        return cls(
            embedding_model,
            cfg.atom_dims,
            cfg.dropout,
            cfg.curvature_scales,
            cfg.no_chem,
            cfg.no_geom,
        )

    def compute_embeddings(
        self,
        surface_xyz: Tensor,
        surface_normals: Tensor,
        atom_coords: Tensor,
        atom_types: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute embeddings of the point clouds"""
        features = self.compute_features(
            surface_xyz, surface_normals, atom_coords, atom_types
        )
        features = self.dropout(features)
        embed_1, embed_2 = self.embedding_model(surface_xyz, surface_normals, features)
        return embed_1, embed_2

    def compute_predictions(
        self,
        surface_xyz: Tensor,
        surface_normals: Tensor,
        atom_coords: Tensor,
        atom_types: Tensor,
        split_idx: int,
    ) -> Tensor:
        # embedding_1_1, embedding_2_1 = split_feature(embed_1, split_idx)
        # embedding_1_2, embedding_2_2 = split_feature(embed_2, split_idx)

        # preds = accumulate_predictions(embedding_1_1, embedding_2_1)
        return None

    def forward(
        self,
        surface_xyz: Tensor,
        surface_normals: Tensor,
        atom_coords: Tensor,
        atom_types: Tensor,
        split_idx: int,
    ) -> Tensor:
        embed_1, embed_2 = self.compute_embeddings(
            surface_xyz, surface_normals, atom_coords, atom_types
        )

        return embed_1, embed_2


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


class dMaSIFSearchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        orientation_units: int,
        emb_dims: int,
        n_layers: int,
        radius: float,
    ):
        super().__init__()
        self.orientation_scores = nn.Sequential(
            nn.Linear(in_channels, orientation_units),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(orientation_units, 1),
        )

        self.orientation_scores2 = nn.Sequential(
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

        self.conv2 = dMaSIFConv_seg(
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
        self,
        surface_xyz: Tensor,
        surface_normals: Tensor,
        features: Tensor,
    ):
        feature_scores = self.orientation_scores(features)
        nuv = self.conv.load_mesh(
            surface_xyz,
            surface_normals,
            weights=feature_scores,
        )
        embed_1 = self.conv(features, surface_xyz, nuv)

        feature_scores_2 = self.orientation_scores2(features)
        nuv2 = self.conv2.load_mesh(
            surface_xyz,
            normals=surface_normals,
            weights=feature_scores_2,
        )
        embed_2 = self.conv2(features, surface_xyz, nuv2)
        return embed_1, embed_2
