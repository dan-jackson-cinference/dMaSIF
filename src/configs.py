from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ModelConfig:
    mode: str = MISSING
    atom_dims: int = 6
    in_channels: int = 16
    orientation_units: int = 16
    n_layers: int = 3
    radius: float = 9.0
    knn: int = 40
    dropout: float = 0.1


@dataclass
class SiteModelConfig(ModelConfig):
    mode: str = "site"
    emb_dims: int = 8
    post_units: int = 8


@dataclass
class SearchModelConfig(ModelConfig):
    mode: str = "search"
    emb_dims: int = 16


@dataclass
class DataConfig:
    resolution: float = 1.0
    sup_sampling: int = 20
    distance: float = 1.05
    validation_fraction: float = 0.1
    random_rotation: bool = False


@dataclass
class TrainingConfig:
    n_epochs: int = 50
    batch_size: int = 1
    optimizer: str = "adam"
    lr: float = 0.0001
    lr_scheduler: Optional[str] = None
    restart_training: str = ""


@dataclass
class Config:
    model: ModelConfig = MISSING
    training: TrainingConfig = MISSING
    data: DataConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="dMaSIF", node=Config)
cs.store(group="model", name="base_search_model", node=SearchModelConfig)
cs.store(group="model", name="base_site_model", node=SiteModelConfig)
cs.store(group="data", name="base_data", node=DataConfig)
cs.store(group="training", name="base_training", node=TrainingConfig)
