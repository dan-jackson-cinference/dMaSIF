from typing import Callable, Optional

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pykeops.torch import LazyTensor
from sklearn.metrics import roc_auc_score
from torch import Tensor, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from load_configs import TrainingConfig
from loss import (
    compute_search_loss,
    compute_site_loss,
    compute_search_loss_small,
    split_feature,
)
from model import BaseModel, SearchModel, SiteModel


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


def embeddings_to_labels(embed_1: Tensor, embed_2: Tensor) -> tuple[Tensor, Tensor]:
    lazy_1 = LazyTensor(embed_1[:, None, :])
    lazy_2 = LazyTensor(embed_2[None, :, :])

    labels_1 = (lazy_1 | lazy_2).max(1)
    labels_2 = (lazy_1 | lazy_2).max(0)

    return labels_1, labels_2


class dMaSIFBaseModule(LightningModule):
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
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, amsgrad=True
        )
        scheduler = ReduceLROnPlateau(optimizer, patience=8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "loss/val",
        }


class dMaSIFSearchModule(dMaSIFBaseModule):
    def __init__(self, model: SearchModel, learning_rate: float):
        super().__init__(model, learning_rate)
        self.loss_fn = compute_search_loss_small

    def training_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int, Tensor],
        batch_idx: int,
        *args,
        **kwargs,
    ) -> Tensor | None:
        xyz, normals, atom_coords, atom_types, labels, split_idx, if_labels = batch
        embed_1, embed_2 = self.model(xyz, normals, atom_coords, atom_types, split_idx)
        loss, preds = self.loss_fn(if_labels, labels, embed_1, embed_2, split_idx)
        # losses = self.all_gather(loss)
        # if any(torch.isnan(loss) for loss in losses):
        #     return None
        roc_auc = roc_auc_score(numpy(labels.view(-1)), numpy(preds.view(-1)))
        self.log(
            "loss/train",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "ROC_AUC/train",
            roc_auc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int, Tensor],
        batch_idx: int,
    ):
        xyz, normals, atom_coords, atom_types, labels, split_idx, if_labels = batch
        embed_1, embed_2 = self.model(xyz, normals, atom_coords, atom_types, split_idx)
        loss, preds = self.loss_fn(if_labels, labels, embed_1, embed_2, split_idx)

        if not torch.isnan(loss):
            roc_auc = roc_auc_score(numpy(labels.view(-1)), numpy(preds.view(-1)))
            self.log("loss/val", loss, prog_bar=True, batch_size=1, sync_dist=True)
            self.log("ROC_AUC/val", roc_auc, batch_size=1, sync_dist=True)

    def test_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int], batch_idx: int
    ):
        xyz, normals, atom_coords, atom_types, labels, split_idx = batch
        preds = self.model(xyz, normals, atom_coords, atom_types, split_idx)
        loss = self.loss_fn(preds, labels)
        if not torch.isnan(loss):
            roc_auc = roc_auc_score(numpy(labels), numpy(preds))
        return loss

    def predict_step(self, batch: list[Tensor, Tensor, Tensor, Tensor, int]):
        xyz, normals, atom_coords, atom_types, split_idx = batch
        embed_1, embed_2 = self.model(xyz, normals, atom_coords, atom_types, split_idx)

        p1_embed_1, p2_embed_1 = split_feature(embed_1, split_idx)
        p1_embed_2, p2_embed_2 = split_feature(embed_2, split_idx)

        labels_1, labels_2 = embeddings_to_labels(p1_embed_1, p2_embed_2)
        print(labels_1.shape)
        print(labels_2.shape)
        print(labels_1.max())
        print(labels_1.min())
        print(labels_1.mean())

        # return preds_1, preds_2


class dMaSIFSiteModule(dMaSIFBaseModule):
    def __init__(
        self,
        model: SiteModel,
        learning_rate: float,
    ):
        super().__init__(model, learning_rate)
        self.loss_fn = compute_site_loss

    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor | None:
        xyz, normals, atom_coords, atom_types, labels = batch
        logits = self.model(xyz, normals, atom_coords, atom_types)
        loss = self.loss_fn(logits, labels)

        if torch.isnan(loss):
            return None
        roc_auc = roc_auc_score(numpy(labels.view(-1)), numpy(logits.view(-1)))
        self.log(
            "loss/train",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "ROC_AUC/train",
            roc_auc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=1,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: list[Tensor], batch_idx: int) -> None:
        xyz, normals, atom_coords, atom_types, labels = batch
        preds = self.model(xyz, normals, atom_coords, atom_types)
        loss = self.loss_fn(preds, labels)

        if not torch.isnan(loss):
            roc_auc = roc_auc_score(numpy(labels.view(-1)), numpy(preds.view(-1)))
            self.log("loss/val", loss, prog_bar=True, batch_size=1, sync_dist=True)
            self.log("ROC_AUC/val", roc_auc, batch_size=1, sync_dist=True)

    def test_step(self, batch: list[Tensor], batch_idx: int) -> None:
        xyz, normals, atom_coords, atom_types, labels = batch
        logits = self.model(xyz, normals, atom_coords, atom_types)
        loss = self.loss_fn(logits, labels)
        if not torch.isnan(loss):
            roc_auc = roc_auc_score(numpy(labels), numpy(logits))
            self.log(
                "loss/test",
                loss,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )
            self.log(
                "ROC_AUC/test",
                roc_auc,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

    def predict_step(self, batch: list[Tensor]) -> Tensor:
        xyz, normals, atom_coords, atom_types = batch
        logits = self.model(xyz, normals, atom_coords, atom_types)
        preds = logits > 0
        return preds


def train(
    model: BaseModel,
    cfg: TrainingConfig,
    train_dataloader: DataLoader[Tensor],
    val_dataloader: DataLoader[Tensor],
    output_dir: str,
):
    """
    Train the PyTorchLightning Module!
    """

    if isinstance(model, SiteModel):
        pl_model = dMaSIFSiteModule(model, cfg.lr)
    elif isinstance(model, SearchModel):
        pl_model = dMaSIFSearchModule(model, cfg.lr)
    else:
        raise TypeError(
            f"model must either be a 'SearchModel' or a 'SiteModel'. It cannot be a {type(model)}"
        )

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        # profiler="simple",
        logger=TensorBoardLogger(output_dir, name="", version=""),
        accumulate_grad_batches=16,
        max_epochs=cfg.n_epochs,
        callbacks=[
            EarlyStopping(monitor="loss/val", mode="min", patience=10),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.fit(pl_model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    a = torch.tensor([[1, 2], [1, 2], [3, 4]], dtype=torch.float32)
    b = torch.tensor([[1, 2], [1, 2], [3, 4]], dtype=torch.float32)

    c = embeddings_to_labels(a, b)
