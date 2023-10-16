from typing import Optional

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import roc_auc_score
from torch import Tensor, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from enums import Mode
from load_configs import TrainingConfig
from loss import compute_search_loss, compute_site_loss
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
        mode: Mode,
        model: BaseModel,
        learning_rate: float,
        save_path: Optional[str] = None,
    ):
        super().__init__()
        self.mode = mode
        self.model = model
        self.save_path = save_path
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, amsgrad=True
        )
        scheduler = ReduceLROnPlateau(optimizer, patience=8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "loss/val",
        }

    def site_forwards(
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

    def search_forwards(
        self,
        surface_xyz: Tensor,
        surface_normals: Tensor,
        atom_coords: Tensor,
        atom_types: Tensor,
    ):
        embedding_1, embedding_2, input_r_values, conv_r_value = self.model(
            surface_xyz, surface_normals, atom_coords, atom_types
        )
        return embedding_1, embedding_2

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx
    ):
        (
            surface_xyz,
            surface_normals,
            atom_coords,
            atom_types,
            surface_labels,
            # split_idx,
        ) = batch
        if self.mode == Mode.SITE:
            preds = self.site_forwards(
                surface_xyz, surface_normals, atom_coords, atom_types
            )
            loss = compute_site_loss(preds, surface_labels)
        if self.mode == Mode.SEARCH:
            embedding_1, embedding_2 = self.search_forwards(
                surface_xyz, surface_normals, atom_coords, atom_types
            )
            split_idx = 0
            loss = compute_search_loss(surface_xyz, embedding_1, embedding_2, split_idx)

        if torch.isnan(loss):
            return None
        roc_auc = roc_auc_score(numpy(surface_labels), numpy(preds))
        self.log(
            "loss/train",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        self.log(
            "ROC_AUC/train",
            roc_auc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
        )
        return loss

    def validation_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_idx,
    ):
        surface_xyz, surface_normals, atom_coords, atom_types, surface_labels = batch
        if self.mode == Mode.SITE:
            preds = self.site_forwards(
                surface_xyz, surface_normals, atom_coords, atom_types
            )
            loss = compute_site_loss(preds, surface_labels)
        if self.mode == Mode.SEARCH:
            embedding_1, embedding_2 = self.search_forwards(
                surface_xyz,
                surface_normals,
                atom_coords,
                atom_types,
            )
            loss = compute_search_loss(
                surface_xyz,
                surface_labels,
                embedding_1,
                embedding_2,
                len_surface_1,
                len_atoms_1,
            )
        if not torch.isnan(loss):
            roc_auc = roc_auc_score(numpy(surface_labels), numpy(preds))
            self.log("loss/val", loss, prog_bar=True, batch_size=1)
            self.log("ROC_AUC/val", roc_auc, batch_size=1)

    def test_step(self, batch, batch_idx):
        surface_xyz, surface_normals, atom_coords, atom_types, surface_labels = batch
        preds = self.forwards(surface_xyz, surface_normals, atom_coords, atom_types)
        loss = compute_site_loss(preds, surface_labels)
        if not torch.isnan(loss):
            roc_auc = roc_auc_score(numpy(surface_labels), numpy(preds))
        return roc_auc


def train(
    mode: Mode,
    model: BaseModel,
    train_cfg: TrainingConfig,
    train_dataloader: DataLoader[Tensor],
    val_dataloader: DataLoader[Tensor],
    checkpoint: str,
):
    wandb_logger = TensorBoardLogger("dMaSIF_logs", name="my_model")
    dmasif_model = dMaSIFTrainer(mode, model, train_cfg.lr, checkpoint)
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        logger=wandb_logger,
        max_epochs=train_cfg.n_epochs,
        callbacks=[
            EarlyStopping(monitor="loss/val", mode="min", patience=10),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.fit(
        dmasif_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
