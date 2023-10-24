from typing import TypeVar

from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from protein import Protein, ProteinPair, ProteinProtocol

T = TypeVar("T")


class ProteinDataset(Dataset[T]):
    def __init__(self, data: list[T]):
        self.data = data

    def __getitem__(self, index: int) -> T:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


def create_datasets(
    train_data: list[ProteinProtocol],
    test_data: list[ProteinProtocol],
    validation_fraction: float = 0.1,
):
    train_dataset = ProteinDataset(train_data)
    train_nsamples = len(train_dataset)
    val_nsamples = int(train_nsamples * validation_fraction)
    train_nsamples = train_nsamples - val_nsamples
    train_dataset, val_dataset = random_split(
        train_dataset, [train_nsamples, val_nsamples]
    )
    test_dataset = ProteinDataset(test_data)

    return {"train": train_dataset, "val": val_dataset, "test": test_dataset}


def collate_proteins(
    data: list[Protein],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    protein_pair = data[0]

    return (
        protein_pair.surface_xyz,
        protein_pair.surface_normals,
        protein_pair.atom_coords,
        protein_pair.atom_types,
        protein_pair.surface_labels,
    )


def collate_protein_pairs(
    data: list[ProteinPair],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int, Tensor]:
    protein_pair = data[0]

    return (
        protein_pair.surface_xyz,
        protein_pair.surface_normals,
        protein_pair.atom_coords,
        protein_pair.atom_types,
        protein_pair.surface_labels,
        protein_pair.split_idx,
        protein_pair.if_labels,
    )


def collate_inference_proteins(
    data: list[ProteinProtocol],
) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    protein_data = data[0]
    return (
        protein_data.surface_xyz,
        protein_data.surface_normals,
        protein_data.atom_coords,
        protein_data.atom_types,
        protein_data.split_idx,
    )


def create_dataloader(
    dataset: ProteinDataset, batch_size: int, split: str, mode: str = "training"
) -> DataLoader[Tensor]:
    return DataLoader(
        dataset,
        batch_size,
        shuffle=split == "train",
        collate_fn=collate_protein_pairs
        if mode == "training"
        else collate_inference_proteins,
        num_workers=4,
    )
