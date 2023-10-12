from typing import Optional

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import roc_auc_score
from torch import Tensor, optim
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from data import Protein
from load_configs import TrainingConfig
from loss import compute_site_loss
from model import BaseModel


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


def numpy(x: Tensor):
    return x.detach().cpu().numpy()


class dMaSIFTrainer(LightningModule):
    def __init__(
        self,
        model: BaseModel,
        learning_rate: float,
        save_path: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.save_path = save_path
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate, amsgrad=True)

    def forwards(
        self,
        surface_xyz: Tensor,
        surface_normals: Tensor,
        atom_coords: Tensor,
        atom_types: Tensor,
    ):
        """Run a forwards pass through the network for a single protein"""
        interface_preds, input_r_value, conv_r_value = self.model(
            surface_xyz, surface_normals, atom_coords, atom_types
        )

        return interface_preds

    def training_step(self, batch: Batch, batch_idx):
        surface_xyz, surface_normals, atom_coords, atom_types, surface_labels = batch
        preds = self.forwards(surface_xyz, surface_normals, atom_coords, atom_types)
        loss = compute_site_loss(preds, surface_labels)
        if torch.isnan(loss):
            return None
        roc_auc = roc_auc_score(numpy(surface_labels), numpy(preds))
        self.log("loss/train", loss, prog_bar=True)
        self.log("ROC_AUC/train", roc_auc)
        return loss

    def validation_step(self, batch: Batch, batch_idx):
        surface_xyz, surface_normals, atom_coords, atom_types, surface_labels = batch
        preds = self.forwards(surface_xyz, surface_normals, atom_coords, atom_types)
        loss = compute_site_loss(preds, surface_labels)
        if not torch.isnan(loss):
            roc_auc = roc_auc_score(numpy(surface_labels), numpy(preds))
            self.log("loss/val", loss, prog_bar=True)
            self.log("ROC_AUC/val", roc_auc)

    def test_step(self, batch: Batch, batch_idx):
        surface_xyz, surface_normals, atom_coords, atom_types, surface_labels = batch
        preds = self.forwards(surface_xyz, surface_normals, atom_coords, atom_types)
        loss = compute_site_loss(preds, surface_labels)
        if not torch.isnan(loss):
            roc_auc = roc_auc_score(numpy(surface_labels), numpy(preds))
        return roc_auc


def train(
    model: BaseModel,
    train_cfg: TrainingConfig,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    checkpoint: str,
):
    dmasif_model = dMaSIFTrainer(model, train_cfg.lr, checkpoint)
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        max_epochs=50,
        callbacks=[EarlyStopping(monitor="loss/val", mode="min", patience=5)],
    )
    trainer.fit(
        dmasif_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
