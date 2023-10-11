from __future__ import annotations

import torch
from torch import Tensor, nn

from benchmark_models import DGCNN_seg, PointNet2_seg, dMaSIFConv_seg
from enums import Mode
from atomnet import AtomNetMP
from helper import soft_dimension
from load_config import ModelCfg
from protein import Protein
from geometry_processing import curvatures


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
        embedding_model: nn.Module,
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
        xyz: Tensor,
        normals: Tensor,
        atom_xyz: Tensor,
        atom_types: Tensor,
    ) -> Tensor:
        """Estimates geometric and chemical features from a protein surface or a cloud of atoms."""

        # Estimate the curvatures using the triangles or the estimated normals:
        protein_curvatures = curvatures(
            xyz,
            normals=normals,
            scales=self.curvature_scales,
        )

        # Compute chemical features on-the-fly:
        chem_feats = self.atomnet(xyz, atom_xyz, atom_types)

        if self.no_chem:
            chem_feats = 0.0 * chem_feats
        if self.no_geom:
            protein_curvatures = 0.0 * protein_curvatures

        # Concatenate our features:
        return torch.cat([protein_curvatures, chem_feats], dim=1).contiguous()

    def embed(self, protein: Protein) -> tuple[float, float]:
        """Embeds all points of a protein in a high-dimensional vector space."""
        input_features = self.dropout(
            self.features(
                protein.surface_xyz,
                protein.surface_normals,
                protein.atom_coords,
                protein.atom_types,
            )
        )

        output_features = self.embedding_model(protein, input_features)

        return input_features, output_features


class SiteModel(BaseModel):
    """A model to find the binding site of a single protein"""

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        embedding_model: nn.Module,
        atom_dims: int,
        dropout: float,
        emb_dims: int,
        post_units: int,
    ):
        super().__init__(feature_extractor, embedding_model, atom_dims, dropout)

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
        feature_extractor: FeatureExtractor,
        embedding_model: nn.Module,
        cfg: ModelCfg,
    ) -> SiteModel:
        """Instantiate a SiteModel object from a config"""
        return cls(
            feature_extractor,
            embedding_model,
            cfg.atom_dims,
            cfg.dropout,
            cfg.emb_dims,
            cfg.post_units,
        )

    def forward(
        self,
        protein: Protein,
    ) -> tuple[Tensor, float, float]:
        """Compute embeddings of the point clouds"""

        input_features, output_embedding = self.embed(protein)

        # Monitor the approximate rank of our representations:
        input_r_value = soft_dimension(input_features)
        conv_r_value = soft_dimension(output_embedding)
        interface_preds = self.net_out(output_embedding)

        return interface_preds, input_r_value, conv_r_value


class SearchModel(BaseModel):
    @classmethod
    def from_config(
        cls,
        feature_extractor: FeatureExtractor,
        embedding_model: nn.Module,
        cfg: ModelCfg,
    ) -> SearchModel:
        """Load a SearchModel from a config"""
        return cls(
            feature_extractor,
            embedding_model,
            cfg.atom_dims,
            cfg.dropout,
        )

    def forward(
        self,
        use_mesh: bool,
        p1: dict[str, Tensor | None],
        p2: None | dict[str, Tensor | None] = None,
    ):
        """Compute embeddings of the point clouds"""
        if p2 is not None:
            p1p2 = combine_pair(p1, p2)
        else:
            p1p2 = p1

        conv_time, memory_usage = self.embed(p1p2, use_mesh)

        # Monitor the approximate rank of our representations:
        R_values = {}
        R_values["input"] = soft_dimension(p1p2["input_features"])
        R_values["conv"] = soft_dimension(p1p2["embedding_1"])

        if p2 is not None:
            p1, p2 = split_pair(p1p2)
        else:
            p1 = p1p2

        return {
            "p1": p1,
            "p2": p2,
            "R_values": R_values,
            "conv_time": conv_time,
            "memory_usage": memory_usage,
        }


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
    def from_config(cls, cfg: ModelCfg):
        "Create an instance of the DMasifSiteEmbed model from a config object"
        return cls(
            cfg.in_channels,
            cfg.orientation_units,
            cfg.emb_dims,
            cfg.n_layers,
            cfg.radius,
        )

    def forward(self, protein: Protein, features: Tensor) -> Tensor:
        feature_scores = self.orientation_scores(features)
        nuv = self.conv.load_mesh(
            protein.surface_xyz,
            protein.surface_normals,
            weights=feature_scores,
        )
        return self.conv(features, protein.surface_xyz, nuv)


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

    def forward(self, P: dict[str, Tensor | None], features: Tensor):
        super().forward(P, features)
        feature_scores_2 = self.orientation_scores2(features)
        nuv2 = self.conv2.load_mesh(
            P["xyz"],
            normals=P["normals"],
            weights=feature_scores_2,
            batch=P["batch"],
        )
        P["embedding_2"] = self.conv2(features, nuv2)


class DGCNNSiteEmbed(nn.Module):
    def __init__(self, in_channels: int, emb_dims: int, n_layers: int, knn: int):
        super().__init__()
        self.conv = DGCNN_seg(in_channels + 3, emb_dims, n_layers, knn)

    @classmethod
    def from_config(cls, cfg: ModelCfg):
        return cls(cfg.in_channels, cfg.emb_dims, cfg.n_layers, cfg.knn)

    def forward(self, P: dict[str, Tensor | None], features):
        features = torch.cat([features, P["xyz"]], dim=-1).contiguous()
        P["embedding_1"] = self.conv(P["xyz"], features, P["batch"])


class DGCNNSearchEmbed(DGCNNSiteEmbed):
    def __init__(self, in_channels: int, emb_dims: int, n_layers: int, knn: int):
        super().__init__(in_channels, emb_dims, n_layers, knn)
        self.conv2 = DGCNN_seg(in_channels + 3, emb_dims, n_layers, knn)

    def forward(self, P, features):
        super().forward(P, features)
        P["embedding_2"] = self.conv2(P["xyz"], features, P["batch"])


class PointNetSiteEmbed(nn.Module):
    def __init__(self, in_channels: int, emb_dims: int, radius: float, n_layers: int):
        self.conv = PointNet2_seg(in_channels, emb_dims, radius, n_layers)

    @classmethod
    def from_config(cls, cfg: ModelCfg):
        return cls(
            cfg.in_channels,
            cfg.emb_dims,
            cfg.radius,
            cfg.n_layers,
        )

    def forward(self, P: dict[str, Tensor | None], features):
        P["embedding_1"] = self.conv(P["xyz"], features, P["batch"])


class PointNetSearchEmbed(PointNetSiteEmbed):
    def __init__(self, in_channels: int, emb_dims: int, radius: float, n_layers: int):
        super().__init__(in_channels, emb_dims, radius, n_layers)
        self.conv2 = PointNet2_seg(in_channels, emb_dims, radius, n_layers)

    def forward(self, P: dict[str, Tensor | None], features):
        super().forward(P, features)
        P["embedding_2"] = self.conv2(P["xyz"], features, P["batch"])


MODELS = {Mode.SEARCH: SearchModel, Mode.SITE: SiteModel}
EMBEDDING_MODELS = {
    Mode.SEARCH: {
        "dMaSIF": dMaSIFSearchEmbed,
        "DCGNN": DGCNNSearchEmbed,
        "PointNet": PointNetSearchEmbed,
    },
    Mode.SITE: {
        "dMaSIF": dMaSIFSiteEmbed,
        "DCGNN": DGCNNSiteEmbed,
        "PointNet": PointNetSiteEmbed,
    },
}


def load_model(mode: Mode, cfg: ModelCfg, model_type: str = "dMaSIF") -> BaseModel:
    """
    Choose the correct type of model depending on the desired mode
    and instantiate it from a ModelConfig
    """
    embedding_model = EMBEDDING_MODELS[mode][model_type].from_config(cfg)
    model = MODELS[mode].from_config(embedding_model, cfg)
    return model
