import torch
import torch.nn.functional as F
from torch import Tensor

from data import Mode


def compute_search_preds(p1, p2, *args, **kwargs):
    pos_xyz1 = p1["xyz"][p1["labels"] == 1]
    pos_xyz2 = p2["xyz"][p2["labels"] == 1]
    pos_descs1 = p1["embedding_1"][p1["labels"] == 1]
    pos_descs2 = p2["embedding_2"][p2["labels"] == 1]

    pos_xyz_dists = ((pos_xyz1[:, None, :] - pos_xyz2[None, :, :]) ** 2).sum(-1).sqrt()
    pos_desc_dists = torch.matmul(pos_descs1, pos_descs2.T)

    pos_preds = pos_desc_dists[pos_xyz_dists < 1.0]
    pos_labels = torch.ones_like(pos_preds)

    n_desc_sample = 100
    sample_desc2 = torch.randperm(len(p2["embedding_2"]))[:n_desc_sample]
    sample_desc2 = p2["embedding_2"][sample_desc2]
    neg_preds = torch.matmul(pos_descs1, sample_desc2.T).view(-1)
    neg_labels = torch.zeros_like(neg_preds)

    # For symmetry
    pos_descs1_2 = p1["embedding_2"][p1["labels"] == 1]
    pos_descs2_2 = p2["embedding_1"][p2["labels"] == 1]

    pos_desc_dists2 = torch.matmul(pos_descs2_2, pos_descs1_2.T)
    pos_preds2 = pos_desc_dists2[pos_xyz_dists.T < 1.0]
    pos_preds = torch.cat([pos_preds, pos_preds2], dim=0)
    pos_labels = torch.ones_like(pos_preds)

    sample_desc1_2 = torch.randperm(len(p1["embedding_2"]))[:n_desc_sample]
    sample_desc1_2 = p1["embedding_2"][sample_desc1_2]
    neg_preds_2 = torch.matmul(pos_descs2_2, sample_desc1_2.T).view(-1)

    neg_preds = torch.cat([neg_preds, neg_preds_2], dim=0)
    neg_labels = torch.zeros_like(neg_preds)
    return pos_preds, pos_labels, neg_preds, neg_labels


def compute_site_loss(predictions: Tensor, labels: Tensor) -> Tensor:
    n_total = len(labels)
    n_positive = labels.sum()
    n_negative = n_total - n_positive
    positive_weight = n_negative / n_positive
    loss = F.binary_cross_entropy_with_logits(
        predictions, labels, pos_weight=positive_weight
    )

    classification = predictions > 0

    TP = torch.sum(torch.logical_and(classification == 1, labels == 1))
    TN = torch.sum(torch.logical_and(classification == 0, labels == 0))
    FP = torch.sum(torch.logical_and(classification == 1, labels == 0))
    FN = torch.sum(torch.logical_and(classification == 0, labels == 1))

    precision = TP / (TP + FP) * 100

    recall = TP / (TP + FN) * 100

    return loss


LOSS_FNS = {Mode.SEARCH: compute_search_preds, Mode.SITE: compute_site_loss}
