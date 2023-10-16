import torch
import torch.nn.functional as F
from pykeops.torch import LazyTensor
from torch import Tensor

from enums import Mode


def split_feature(feature: Tensor, split_idx: int) -> tuple[Tensor, Tensor]:
    return feature[:split_idx], feature[split_idx:]


def lazy_pairwise_distance(coords_1: Tensor, coords_2: Tensor) -> Tensor:
    lazy_coords_1 = LazyTensor(coords_1[:, None, :])
    lazy_coords_2 = LazyTensor(coords_2[None, :, :])
    return ((lazy_coords_1 - lazy_coords_2) ** 2).sum(-1).sqrt()


def pairwise_distances(coords_1: Tensor, coords_2: Tensor) -> Tensor:
    return ((coords_1[:, None, :] - coords_2[None, :, :]) ** 2).sum(-1).sqrt()


def generate_interface_labels(
    surface_1: Tensor, surface_2: Tensor, threshold: float = 1.0
) -> tuple[Tensor, Tensor]:
    """Only use positive labels for points on the surface that are within a given distance of each other"""
    coords_1 = LazyTensor(surface_1[:, None, :].contiguous())
    coords_2 = LazyTensor(surface_2[None, :, :].contiguous())
    pairwise_dists = ((coords_1 - coords_2) ** 2).sum(-1).sqrt()
    interface = (threshold - pairwise_dists).step()

    interface_labels_1 = interface.max(1)
    interface_labels_2 = interface.max(0)

    return interface_labels_1.squeeze(), interface_labels_2.squeeze()


def compute_search_loss(
    surface_xyz: Tensor,
    embedding_1: Tensor,
    embedding_2: Tensor,
    len_surface: int,
    sample_neg_points: int = 1000,
):
    surface_xyz_1, surface_xyz_2 = split_feature(surface_xyz, len_surface)

    interface_labels_1, interface_labels_2 = generate_interface_labels(
        surface_xyz_1, surface_xyz_2
    )
    interface_points_1 = surface_xyz_1[interface_labels_1 == 1]
    interface_points_2 = surface_xyz_2[interface_labels_2 == 1]

    interface_labels = torch.outer(
        interface_labels_1[interface_labels_1 == 1],
        interface_labels_2[interface_labels_2 == 1],
    )
    dists = pairwise_distances(interface_points_1, interface_points_2)
    interface_labels[dists > 1.0] = 0

    embedding_1_1, embedding_2_1 = split_feature(embedding_1, len_surface)
    embedding_1_2, embedding_2_2 = split_feature(embedding_2, len_surface)

    pos_features_1_1 = embedding_1_1[interface_labels_1 == 1]
    pos_features_2_1 = embedding_2_1[interface_labels_2 == 1]
    interface_preds_1 = torch.matmul(pos_features_1_1, pos_features_2_1.T)

    pos_features_1_2 = embedding_1_2[interface_labels_1 == 1]
    pos_features_2_2 = embedding_2_2[interface_labels_2 == 1]
    interface_preds_2 = torch.matmul(pos_features_1_2, pos_features_2_2.T)

    print(interface_labels.shape)
    print(interface_labels.sum())
    print(interface_preds_1.shape)
    print(interface_preds_2.shape)

    neg_features_1_1 = embedding_1_1[interface_labels_1 == 0]
    neg_features_2_1 = embedding_2_1[interface_labels_2 == 0]
    samples_idxs_1_1 = torch.randint(
        low=0, high=len(neg_features_1_1), size=(sample_neg_points,)
    )
    samples_idxs_2_1 = torch.randint(
        low=0, high=len(neg_features_2_1), size=(sample_neg_points,)
    )
    sampled_neg_features_1_1 = neg_features_1_1[samples_idxs_1_1]
    sampled_neg_features_2_1 = neg_features_1_1[samples_idxs_2_1]

    neg_interface_preds = torch.matmul(
        sampled_neg_features_1_1, sampled_neg_features_2_1.T
    )

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


LOSS_FNS = {Mode.SEARCH: compute_search_loss, Mode.SITE: compute_site_loss}
