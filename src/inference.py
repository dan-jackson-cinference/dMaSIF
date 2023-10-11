from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from data import Mode
from data_iteration import iterate
from atomnet import FeatureExtractor
from model import BaseModel


def inference(
    model: BaseModel,
    feature_extractor: FeatureExtractor,
    dataloader: DataLoader,
    single_protein: bool,
    random_rotation: bool,
    mode: Mode,
    save_predictions_path: Path,
    device: torch.device,
):
    # Perform one pass through the data:
    info = iterate(
        model,
        feature_extractor,
        dataloader=dataloader,
        device=device,
        single_protein=single_protein,
        random_rotation=random_rotation,
        mode=mode,
        save_path=save_predictions_path,
        pdb_ids=test_pdb_ids,
    )
    return info

    # np.save(f"timings/{args.experiment_name}_convtime.npy", info["conv_time"])
    # np.save(f"timings/{args.experiment_name}_memoryusage.npy", info["memory_usage"])
