import os
from pathlib import Path

import hydra
import numpy as np
import torch
from lightning.pytorch import Trainer

from dataset import ProteinDataset, create_dataloader, create_datasets
from load_configs import Config, ModelConfig
from model import BaseModel, SearchModel
from pl_trainer import dMaSIFBaseModule, dMaSIFSearchModule, embeddings_to_labels, train
from process_data import PROCESSORS
from protein import PDBInferenceProtein, ProteinPair

torch.set_float32_matmul_precision("medium")


def set_seed(seed: int) -> None:
    "Set all the seeds to ensure reproducibility"
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def create_checkpoint(model_cfg: ModelConfig) -> str:
    return f"dMaSIF_{model_cfg.n_layers}_layers_{model_cfg.radius}A_radius_{model_cfg.emb_dims}_emb_dims"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: Config):
    set_seed(cfg.seed)

    model = BaseModel.from_config(cfg.model)
    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    checkpoint = create_checkpoint(cfg.model)
    checkpoint_path = os.path.join(out_dir, checkpoint)

    if not os.path.exists(checkpoint_path):
        processor = PROCESSORS[cfg.model.mode].from_config(
            root_dir="surface_data", cfg=cfg.data
        )
        train_data, test_data = processor.load_processed_data()
        datasets = create_datasets(train_data, test_data, cfg.data.validation_fraction)
        dataloaders = {
            split: create_dataloader(dataset, batch_size=1, split=split)
            for split, dataset in datasets.items()
        }

        model = train(
            model, cfg.training, dataloaders["train"], dataloaders["val"], out_dir
        )
    else:
        checkpoint = torch.load(checkpoint_path)
        model = BaseModel.from_config(cfg.model)
        pl_model = dMaSIFSearchModule(model, 0.001)
        pl_model.load_state_dict(checkpoint["state_dict"])
        data_dir = Path("/home/danjackson/repos/dMaSIF/")
        pdb_id_1 = "1f6m_F"
        pdb_id_2 = "1f6m_H"
        protein_1 = PDBInferenceProtein.from_pdb(pdb_id_1, data_dir)
        protein_2 = PDBInferenceProtein.from_pdb(pdb_id_2, data_dir)
        protein_1.compute_surface_features()
        protein_2.compute_surface_features()
        protein_pair = ProteinPair(protein_1, protein_2)
        inference_dataset = ProteinDataset([protein_pair])
        inference_dataloader = create_dataloader(
            inference_dataset, 1, "inference", "inference"
        )
        print(len(protein_1.surface_xyz))
        print(len(protein_2.surface_xyz))

        trainer = Trainer()
        trainer.predict(pl_model, inference_dataloader)


if __name__ == "__main__":
    main()
