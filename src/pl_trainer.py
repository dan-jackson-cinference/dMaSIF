from typing import Optional

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import roc_auc_score
from torch import Tensor, optim
from torch_geometric.data import Batch

from data import Protein
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
        random_rotation: bool,
        learning_rate: float,
        save_path: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.save_path = save_path
        self.learning_rate = learning_rate
        self.random_rotation = random_rotation

    @classmethod
    def from_config(cls, cfg):
        pass

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate, amsgrad=True)

    def forwards(self, protein: Protein):
        """Run a forwards pass through the network for a single protein"""
        interface_preds, input_r_value, conv_r_value = self.model(protein)

        return interface_preds

    def training_step(self, batch: Batch, batch_idx):
        for protein in batch.protein_1:
            preds = self.forwards(protein)
            loss = compute_site_loss(preds, protein.surface_labels)
            if torch.isnan(loss):
                return None
            roc_auc = roc_auc_score(numpy(protein.surface_labels), numpy(preds))
        self.log("loss/train", loss, prog_bar=True)
        self.log("ROC_AUC/train", roc_auc)
        return loss

    def validation_step(self, batch: Batch, batch_idx):
        for protein in batch.protein_1:
            preds = self.forwards(protein)
            loss = compute_site_loss(preds, protein.surface_labels)
        if not torch.isnan(loss):
            roc_auc = roc_auc_score(numpy(protein.surface_labels), numpy(preds))
            self.log("loss/val", loss, prog_bar=True)
            self.log("ROC_AUC/val", roc_auc)

    def test_step(self, batch: Batch, batch_idx):
        for protein in batch.protein_1:
            preds = self.forwards(protein)
            loss = compute_site_loss(preds, protein.surface_labels)
        if not torch.isnan(loss):
            roc_auc = roc_auc_score(numpy(protein.surface_labels), numpy(preds))
        return roc_auc


def train(model: BaseModel, train_dataloader: DataLoader, val_dataloader: DataLoader):
    dmasif_model = dMaSIFTrainer(model)
    trainer = Trainer(
        callbacks=[EarlyStopping(monitor="loss/val", mode="min", patience=5)]
    )
    trainer.fit(
        dmasif_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
