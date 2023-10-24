import torch
import torch.nn.functional as F
from pykeops.torch import LazyTensor
from torch import Tensor


def split_feature(feature: Tensor, split_idx: int) -> tuple[Tensor, Tensor]:
    return feature[:split_idx], feature[split_idx:]


def lazy_pairwise_distance(coords_1: Tensor, coords_2: Tensor) -> Tensor:
    lazy_coords_1 = LazyTensor(coords_1[:, None, :])
    lazy_coords_2 = LazyTensor(coords_2[None, :, :])
    return ((lazy_coords_1 - lazy_coords_2) ** 2).sum(-1).sqrt()


def pairwise_distances(coords_1: Tensor, coords_2: Tensor) -> Tensor:
    return ((coords_1[:, None, :] - coords_2[None, :, :]) ** 2).sum(-1).sqrt()


def compute_search_loss_test(
    surface_xyz: Tensor,
    surface_labels: Tensor,
    embedding_1: Tensor,
    embedding_2: Tensor,
    len_surface: int,
    sample_neg_points: int = 100,
    threshold: float = 2.0,
):
    surface_xyz_1, surface_xyz_2 = split_feature(surface_xyz, len_surface)
    labels_1, labels_2 = split_feature(surface_labels, len_surface)
    p1_embed_1, p2_embed_1 = split_feature(embedding_1, len_surface)
    p1_embed_2, p2_embed_2 = split_feature(embedding_2, len_surface)

    if_points_1 = surface_xyz_1[labels_1 == 2]
    if_points_2 = surface_xyz_2[labels_2 == 2]
    if_dists = pairwise_distances(if_points_1, if_points_2)
    if_labels = torch.heaviside(
        threshold - if_dists, values=torch.tensor([0.0], device=if_dists.device)
    )

    p1_if_feats_1 = p1_embed_1[labels_1 == 2]
    p2_if_feats_1 = p2_embed_1[labels_2 == 2]
    p1_if_feats_2 = p1_embed_2[labels_1 == 2]
    p2_if_feats_2 = p2_embed_2[labels_2 == 2]

    if_preds_1 = torch.matmul(p1_if_feats_1, p2_if_feats_2.T)
    if_preds_2 = torch.matmul(p1_if_feats_2, p2_if_feats_1.T)
    if_preds = torch.cat([if_preds_1, if_preds_2]).view(-1)
    if_labels = if_labels.tile((2,)).view(-1)

    p1_cand_feats_1 = p1_embed_1[labels_1 == 1]
    p2_cand_feats_1 = p2_embed_1[labels_2 == 1]
    p1_cand_feats_2 = p1_embed_2[labels_1 == 1]
    p2_cand_feats_2 = p2_embed_2[labels_2 == 1]

    cand_preds_1 = torch.matmul(p1_cand_feats_1, p2_cand_feats_2.T)
    cand_preds_2 = torch.matmul(p1_cand_feats_2, p2_cand_feats_1.T)

    cand_preds = torch.cat([cand_preds_1, cand_preds_2]).view(-1)
    cand_labels = torch.zeros_like(cand_preds)

    p1_neg_feats_1 = p1_embed_1[labels_1 == 0]
    p2_neg_feats_1 = p2_embed_1[labels_2 == 0]
    p1_neg_feats_2 = p1_embed_2[labels_1 == 0]
    p2_neg_feats_2 = p2_embed_2[labels_2 == 0]

    if (
        len(p1_neg_feats_1) < sample_neg_points
        or len(p2_neg_feats_2) < sample_neg_points
    ):
        sample_neg_points = min(len(p1_neg_feats_1), len(p2_neg_feats_2))

    p1_samples_idxs_1 = torch.randperm(
        len(p1_neg_feats_1), device=p1_neg_feats_1.device
    )[:sample_neg_points]
    p2_samples_idxs_1 = torch.randperm(
        len(p2_neg_feats_1), device=p2_neg_feats_1.device
    )[:sample_neg_points]
    p1_samples_idxs_2 = torch.randperm(
        len(p1_neg_feats_2), device=p1_neg_feats_2.device
    )[:sample_neg_points]
    p2_samples_idxs_2 = torch.randperm(
        len(p2_neg_feats_2), device=p2_neg_feats_2.device
    )[:sample_neg_points]

    p1_sampled_neg_feats_1 = p1_neg_feats_1[p1_samples_idxs_1]
    p2_sampled_neg_feats_1 = p2_neg_feats_1[p2_samples_idxs_1]
    p1_sampled_neg_feats_2 = p1_neg_feats_2[p1_samples_idxs_2]
    p2_sampled_neg_feats_2 = p2_neg_feats_1[p2_samples_idxs_2]

    neg_preds_1 = torch.matmul(p1_sampled_neg_feats_1, p2_sampled_neg_feats_2.T)
    neg_preds_2 = torch.matmul(p1_sampled_neg_feats_2, p2_sampled_neg_feats_1.T)

    neg_preds = torch.cat([neg_preds_1, neg_preds_2]).view(-1)

    neg_labels = torch.zeros_like(neg_preds)

    preds = torch.cat([if_preds, cand_preds, neg_preds])
    labels = torch.cat([if_labels, cand_labels, neg_labels])

    n_total = len(labels)
    n_positive = labels.sum()
    n_negative = n_total - n_positive
    positive_weight = n_negative / n_positive

    loss = F.binary_cross_entropy_with_logits(labels, preds, pos_weight=positive_weight)
    return loss


def compute_site_loss(predictions: Tensor, labels: Tensor) -> Tensor:
    n_total = len(labels)
    n_positive = labels.sum()
    n_negative = n_total - n_positive
    positive_weight = n_negative / n_positive

    loss = F.binary_cross_entropy_with_logits(
        predictions.squeeze(), labels, pos_weight=positive_weight
    )

    classification = predictions > 0

    # TP = torch.sum(torch.logical_and(classification == 1, labels == 1))
    # TN = torch.sum(torch.logical_and(classification == 0, labels == 0))
    # FP = torch.sum(torch.logical_and(classification == 1, labels == 0))
    # FN = torch.sum(torch.logical_and(classification == 0, labels == 1))

    # precision = TP / (TP + FP) * 100

    # recall = TP / (TP + FN) * 100

    return loss


def compute_search_loss(
    if_labels: Tensor,
    surface_labels: Tensor,
    embedding_1: Tensor,
    embedding_2: Tensor,
    split_idx: int,
):
    labels_1, labels_2 = split_feature(surface_labels, split_idx)
    p1_embed_1, p2_embed_1 = split_feature(embedding_1, split_idx)
    p1_embed_2, p2_embed_2 = split_feature(embedding_2, split_idx)

    n_sample = 100
    if_feats_1 = p1_embed_1[labels_1 == 1]
    neg_feats_1 = p1_embed_1[labels_1 == 0]
    neg_samples_1 = torch.randperm(len(neg_feats_1))[:n_sample]
    sample_feats_1 = p1_embed_1[neg_samples_1]
    feats_1 = torch.cat([if_feats_1, sample_feats_1])

    if_feats_2 = p2_embed_2[labels_2 == 1]
    neg_feats_2 = p2_embed_2[labels_2 == 0]
    neg_samples_2 = torch.randperm(len(neg_feats_2))[:n_sample]
    sample_feats_2 = p2_embed_2[neg_samples_2]
    feats_2 = torch.cat([if_feats_2, sample_feats_2])

    feats = torch.matmul(feats_1, feats_2.T)

    pairwise_labels = torch.zeros_like(feats)
    pairwise_labels[: len(if_feats_1), : len(if_feats_2)] = if_labels

    n_total = pairwise_labels.numel()
    n_pos = pairwise_labels.sum()
    n_neg = n_total - n_pos
    weighting = n_neg / n_pos

    loss = F.binary_cross_entropy_with_logits(
        feats, pairwise_labels, pos_weight=weighting
    )

    lazy_1 = LazyTensor(p1_embed_1[:, None, :])
    lazy_2 = LazyTensor(p2_embed_2[None, :, :])

    feats = lazy_1 | lazy_2
    logits_1 = feats.max(1).squeeze()
    logits_2 = feats.max(0).squeeze()

    return loss, torch.cat([logits_1, logits_2])


def compute_search_loss_small(
    if_labels: Tensor,
    surface_labels: Tensor,
    embedding_1: Tensor,
    embedding_2: Tensor,
    split_idx: int,
):
    labels_1, labels_2 = split_feature(surface_labels, split_idx)
    p1_embed_1, p2_embed_1 = split_feature(embedding_1, split_idx)
    p1_embed_2, p2_embed_2 = split_feature(embedding_2, split_idx)

    n_sample = 64
    if_feats_1 = p1_embed_1[labels_1 == 1]
    if_feats_2 = p2_embed_2[labels_2 == 1]

    if_samples_1 = torch.randperm(len(if_feats_1), device=if_feats_1.device)[:n_sample]
    if_samples_2 = torch.randperm(len(if_feats_2), device=if_feats_2.device)[:n_sample]

    if_sample_feats_1 = if_feats_1[if_samples_1]
    if_sample_feats_2 = if_feats_2[if_samples_2]

    labels_sample = if_labels.index_select(0, if_samples_1).index_select(
        1, if_samples_2
    )

    neg_feats_1 = p1_embed_1[labels_1 == 0]
    neg_samples_1 = torch.randperm(len(neg_feats_1), device=neg_feats_1.device)[
        :n_sample
    ]
    neg_sample_feats_1 = p1_embed_1[neg_samples_1]
    feats_1 = torch.cat([if_sample_feats_1, neg_sample_feats_1])

    neg_feats_2 = p2_embed_2[labels_2 == 0]
    neg_samples_2 = torch.randperm(len(neg_feats_2), device=neg_feats_2.device)[
        :n_sample
    ]
    neg_sample_feats_2 = p2_embed_2[neg_samples_2]
    feats_2 = torch.cat([if_sample_feats_2, neg_sample_feats_2])

    feats = torch.matmul(feats_1, feats_2.T)

    pairwise_labels = torch.zeros_like(feats)
    pairwise_labels[: len(if_sample_feats_1), : len(if_sample_feats_2)] = labels_sample

    n_total = pairwise_labels.numel()
    n_pos = pairwise_labels.sum()
    n_neg = n_total - n_pos
    weighting = n_neg / n_pos

    loss = F.binary_cross_entropy_with_logits(
        feats, pairwise_labels, pos_weight=weighting
    )
    if torch.isnan(loss):
        # print(feats.sum())
        print(pairwise_labels.sum())
        exit()

    lazy_1 = LazyTensor(p1_embed_1[:, None, :])
    lazy_2 = LazyTensor(p2_embed_2[None, :, :])

    feats = lazy_1 | lazy_2
    logits_1 = feats.max(1).squeeze()
    logits_2 = feats.max(0).squeeze()

    return loss, torch.cat([logits_1, logits_2])


LOSS_FNS = {"search": compute_search_loss, "site": compute_site_loss}
