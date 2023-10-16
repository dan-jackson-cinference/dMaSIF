import os

import hydra
import numpy as np
import torch

from dataset import create_dataloader, create_datasets
from enums import Mode
from load_configs import Config, ModelConfig, TrainingConfig
from model import load_model
from pl_trainer import train as pl_train
from process_data import PROCESSORS


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
    checkpoint = create_checkpoint(mode, cfg.model, cfg.training)

    model = load_model(mode, cfg.model)

    if not os.path.exists(checkpoint):
        processor = PROCESSORS[mode].from_config(root_dir="surface_data", cfg=cfg.data)
        train_data, test_data = processor.load_processed_data()
        datasets = create_datasets(train_data, test_data, cfg.data.validation_fraction)
        dataloaders = {
            split: create_dataloader(dataset, mode=mode, batch_size=1, split=split)
            for split, dataset in datasets.items()
        }

        model = pl_train(
            mode,
            model,
            cfg.training,
            dataloaders["train"],
            dataloaders["val"],
            checkpoint,
        )
    else:
        pass
        # model = load_trained_model(checkpoint)


if __name__ == "__main__":
    main()
