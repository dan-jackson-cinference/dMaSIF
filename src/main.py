import os

import hydra
import numpy as np
import torch

from data import load_training_data
from enums import Mode
from load_configs import Config, ModelConfig, TrainingConfig
from model import load_model
from pl_trainer import train as pl_train
from train import train


def set_seed(seed: int) -> None:
    "Set all the seeds to ensure reproducibility"
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def create_checkpoint(
    mode: Mode, model_cfg: ModelConfig, train_cfg: TrainingConfig
) -> str:
    return f"{mode.value}/dMaSIF_{model_cfg.n_layers}_layers_{model_cfg.radius}A_radius_{model_cfg.emb_dims}_emb_dim_{train_cfg.lr}_lr"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: Config):
    mode = cfg.model.mode
    set_seed(cfg.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = create_checkpoint(mode, cfg.model, cfg.training)

    model = load_model(mode, cfg.model)

    if not os.path.exists(checkpoint):
        dataloaders = load_training_data(mode, cfg.data)
        model = pl_train(
            model, cfg.training, dataloaders["train"], dataloaders["val"], checkpoint
        )
    else:
        pass
        # model = load_trained_model(checkpoint)


if __name__ == "__main__":
    main()
