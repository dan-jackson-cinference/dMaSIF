from typing import Optional

import numpy as np
import torch
from pykeops.torch import LazyTensor
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data import Mode
from geometry_processing import save_vtk
from loss import compute_site_loss
from model import BaseModel


def numpy(x: Tensor):
    return x.detach().cpu().numpy()


def save_protein_batch_single(protein_pair_id, P, save_path, pdb_idx):
    protein_pair_id = protein_pair_id.split("_")
    pdb_id = protein_pair_id[0] + "_" + protein_pair_id[pdb_idx]

    batch = P["batch"]

    xyz = P["xyz"]

    inputs = P["input_features"]

    embedding = P["embedding_1"] if pdb_idx == 1 else P["embedding_2"]
    emb_id = 1 if pdb_idx == 1 else 2

    predictions = (
        torch.sigmoid(P["iface_preds"])
        if "iface_preds" in P.keys()
        else 0.0 * embedding[:, 0].view(-1, 1)
    )

    labels = P["labels"].view(-1, 1) if P["labels"] is not None else 0.0 * predictions

    coloring = torch.cat([inputs, embedding, predictions, labels], axis=1)

    save_vtk(str(save_path / pdb_id) + f"_pred_emb{emb_id}", xyz, values=coloring)
    np.save(str(save_path / pdb_id) + "_predcoords", numpy(xyz))
    np.save(str(save_path / pdb_id) + f"_predfeatures_emb{emb_id}", numpy(coloring))


def generate_matchinglabels(
    p1: dict[str, Tensor | None], p2: dict[str, Tensor | None], random_rotation: bool
) -> None:
    if random_rotation:
        p1["xyz"] = torch.matmul(p1["rand_rot"].T, p1["xyz"].T).T + p1["atom_center"]
        p2["xyz"] = torch.matmul(p2["rand_rot"].T, p2["xyz"].T).T + p2["atom_center"]
    xyz1_i = LazyTensor(p1["xyz"][:, None, :].contiguous())
    xyz2_j = LazyTensor(p2["xyz"][None, :, :].contiguous())

    xyz_dists = ((xyz1_i - xyz2_j) ** 2).sum(-1).sqrt()
    xyz_dists = (1.0 - xyz_dists).step()

    p1_iface_labels = (xyz_dists.sum(1) > 1.0).float().view(-1)
    p2_iface_labels = (xyz_dists.sum(0) > 1.0).float().view(-1)

    p1["labels"] = p1_iface_labels
    p2["labels"] = p2_iface_labels


def iterate(
    model: BaseModel,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: Optional[Optimizer] = None,
    summary_writer: Optional[SummaryWriter] = None,
    epoch_number: Optional[int] = None,
) -> dict[str, list[float]]:
    """Goes through one epoch of the dataset, returns information for Tensorboard."""
    if optimizer is None:
        model.eval()
        torch.set_grad_enabled(False)
    else:
        model.train()
        torch.set_grad_enabled(True)

    # Statistics and fancy graphs to summarize the epoch:
    losses = []
    roc_aucs = []
    input_r_values = []
    conv_r_values = []

    # Loop over one epoch:
    for protein_batch in tqdm(dataloader):
        protein_batch.to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        for protein in protein_batch.protein_1:
            interface_preds, input_r_value, conv_r_value = model(protein)

            loss = compute_site_loss(interface_preds, protein.surface_labels)
            if torch.isnan(loss).sum() > 0:
                continue

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            roc_auc = roc_auc_score(
                numpy(protein.surface_labels), numpy(interface_preds)
            )

            losses.append(loss.item())
            roc_aucs.append(roc_auc)
            input_r_values.append(input_r_value)
            conv_r_values.append(conv_r_value)

    return losses, roc_aucs, input_r_values, conv_r_values
