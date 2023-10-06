from dataclasses import dataclass


@dataclass
class FeatureCfg:
    resolution: float
    sup_sampling: int
    distance: float
    no_chem: bool
    no_geom: bool
    curvature_scales: list[float]


@dataclass
class GeometryCfg:
    variance: float


@dataclass
class ModelCfg:
    atom_dims: int
    embedding_layer: str
    emb_dims: int
    in_channels: int
    orientation_units: int
    unet_hidden_channels: int
    post_units: int
    n_layers: int
    radius: float
    knn: int
    dropout: float
    use_mesh: bool


@dataclass
class TrainingCfg:
    n_epochs: int
    batch_size: int
    lr: float
    restart_training: str
    n_roc_auc_samples: int


@dataclass
class Cfg:
    model_path: str
    save_path: str
    site: bool
    search: bool
    single_pdb: str
    pdb_list: str
    use_mesh: bool
    profile: bool
    seed: int
    random_rotation: bool
    single_protein: bool
    validation_fraction: bool
    feature_cfg: FeatureCfg
    geometry_cfg: GeometryCfg
    model_cfg: ModelCfg
    training_cfg: TrainingCfg
