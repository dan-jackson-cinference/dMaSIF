import torch
from torch.utils.data import DataLoader, Dataset, random_split

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


def collate_proteins(data: list[Protein]):
    print(data)
    exit()


def create_dataloader(
    dataset: ProteinDataset, batch_size: int, split: str
) -> DataLoader:
    return DataLoader(
        dataset, batch_size, shuffle=split == "train", collate_fn=collate_proteins
    )
