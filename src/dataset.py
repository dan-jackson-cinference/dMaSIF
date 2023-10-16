import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from enums import Mode
from protein import Protein


class ProteinDataset(Dataset[Protein]):
    def __init__(self, data: list[Protein]):
        self.data = data

    def __getitem__(self, index: int) -> Protein:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


def create_datasets(
    train_data: list[Protein],
    test_data: list[Protein],
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
    protein = data[0]
    return (
        protein.surface_xyz,
        protein.surface_normals,
        protein.atom_coords,
        protein.atom_types,
        protein.surface_labels,
    )


def collate_protein_pairs(
    data: list[tuple[Protein, Protein]]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, float]:
    protein_1, protein_2 = data[0]

    return (
        torch.cat([protein_1.surface_xyz, protein_2.surface_xyz]),
        torch.cat([protein_1.surface_normals, protein_2.surface_normals]),
        torch.cat([protein_1.atom_coords, protein_2.atom_coords]),
        torch.cat([protein_1.atom_types, protein_2.atom_types]),
        torch.cat([protein_1.surface_labels, protein_2.surface_labels]),
        len(protein_1.surface_xyz),
    )


def create_dataloader(
    dataset: ProteinDataset, mode: Mode, batch_size: int, split: str
) -> DataLoader[Tensor]:
    return DataLoader(
        dataset,
        batch_size,
        shuffle=split == "train",
        collate_fn=COLLATE_FNS[mode],
        num_workers=4,
    )


COLLATE_FNS = {Mode.SEARCH: collate_protein_pairs, Mode.SITE: collate_proteins}
