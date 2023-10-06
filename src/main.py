import os
from pathlib import Path

import hydra
import lightning.pytorch as pl
import numpy as np
import torch
from hydra.core.config_store import ConfigStore

from data import Mode, load_training_data
from features import FeatureExtractor
from load_config import Cfg
from model import EMBEDDING_MODELS, MODELS
from pl_trainer import dMaSIFTrainer
from train import train

# Ensure reproducability:
torch.backends.cudnn.deterministic = True
cs = ConfigStore.instance()
cs.store(name="config", node=Cfg)


def set_seed(seed: int) -> None:
    "Set all the seeds to ensure reproducibility"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@hydra.main(
    version_base="1.3",
    config_path="/home/danjackson/repos/dMaSIF",
    config_name="dmasif_config",
)
def main(cfg: Cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feature_extractor = FeatureExtractor.from_config(
        cfg.feature_cfg, cfg.single_protein
    )
    mode = Mode.SITE

    embedding_model = EMBEDDING_MODELS[mode]["dMaSIF"].from_config(cfg.model_cfg)
    model = MODELS[mode].from_config(feature_extractor, embedding_model, cfg.model_cfg)

    if not os.path.exists(cfg.model_path):
        dataloaders = load_training_data(
            cfg.feature_cfg.resolution,
            cfg.feature_cfg.sup_sampling,
            cfg.feature_cfg.distance,
            cfg.random_rotation,
            mode,
            cfg.validation_fraction,
        )
        model = train(
            cfg.training_cfg,
            model,
            dataloaders,
            cfg.random_rotation,
            cfg.single_protein,
            mode,
            cfg.model_path,
            device,
        )
    else:
        model.load_state_dict(
            torch.load(cfg.model_path, map_location=device)["model_state_dict"]
        )
    # model = model.to(device)
    # d_masif = dMaSIFTrainer(
    #     model, feature_extractor, mode, random_rotation=True, save_path="/."
    # )
    # trainer = pl.Trainer()
    # trainer.fit(d_masif, train_dataloaders=dataloaders["traint"])


if __name__ == "__main__":
    main()
