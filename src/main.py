import os
from pathlib import Path

import hydra
import lightning.pytorch as pl
import numpy as np
import torch

from data import Mode, load_training_data
from model import load_model
from train import train
from configs import Config


def set_seed(seed: int) -> None:
    "Set all the seeds to ensure reproducibility"
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: Config):
    mode = cfg.model_cfg.mode
    set_seed(cfg.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_model(mode, cfg.model_cfg)

    if not os.path.exists(cfg.model_path):
        dataloaders = load_training_data(mode, cfg.data_cfg)
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
        model = load_trained_model(cfg.checkpoint_path)


if __name__ == "__main__":
    main()
