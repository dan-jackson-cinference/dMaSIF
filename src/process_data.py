from __future__ import annotations

import csv
import os
import pickle
import tarfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests
import torch
from pykeops.torch import LazyTensor
from torch import Tensor
from tqdm import tqdm

from load_configs import DataConfig
from protein import Protein, ProteinPair, ProteinProtocol


def protein_labels_valid(protein: Protein) -> bool:
    """Check that the labels for the interaction site are reasonable"""
    labels = protein.surface_labels.reshape(-1)
    return (labels.sum() < 0.75 * len(labels)).item() and (labels.sum() > 30).item()


def download(url: Path, target_dir: Path):
    response = requests.get(str(url), stream=True, timeout=20)
    if response.status_code == 200:
        with open(target_dir, "wb") as f:
            f.write(response.raw.read())


def untar(file: Path, target_dir: Path):
    with tarfile.open(file) as tar:
        tar.extractall(target_dir)


def load_csv(csv_path: Path) -> list[dict[str, str]]:
    """Load a csv file into a list of dicts representing each line"""
    data: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for line in reader:
            data.append(line)
    return data


def pickle_dump(data: Any, save_path: Path):
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file: Path) -> list[ProteinProtocol]:
    with open(file, "rb") as handle:
        b = pickle.load(handle)
    return b


def pairwise_distance(coords_1: LazyTensor, coords_2: LazyTensor) -> LazyTensor:
    return ((coords_1[:, None, :] - coords_2[None, :, :]) ** 2).sum(-1).sqrt()


def update_interface_labels(
    p_1: Protein,
    p_2: Protein,
    threshold: float,
) -> Tensor:
    """
    Generate 3 classes of label:
    0: This is not a candidate binding site (given a label of 0 by MaSIF)
    1: This is a candidate binding site (given a label of 1 by MaSIF), but \
        is not in the interface
    2: This is a candidate binding site and is in the interface \
        (as defined by a distance to a vertex on the other protein < threshold)
    """
    coords_1 = LazyTensor(p_1.surface_xyz[:, None, :].contiguous())
    coords_2 = LazyTensor(p_2.surface_xyz[None, :, :].contiguous())
    pairwise_dists = ((coords_1 - coords_2) ** 2).sum(-1).sqrt()
    interface = (threshold - pairwise_dists).step()

    interface_labels_1 = interface.max(1).squeeze()
    interface_labels_2 = interface.max(0).squeeze()

    # Binary classification
    p_1.surface_labels = interface_labels_1
    p_2.surface_labels = interface_labels_2

    # Multiclass cassification
    # p_1.surface_labels[interface_labels_1 == 1] = 2
    # p_2.surface_labels[interface_labels_2 == 1] = 2

    p_1.surface_labels.squeeze()
    p_2.surface_labels.squeeze()

    if_coords_1 = p_1.surface_xyz[interface_labels_1 == 1][:, None, :]
    if_coords_2 = p_2.surface_xyz[interface_labels_2 == 1][None, :, :]

    if_dists = ((if_coords_1 - if_coords_2) ** 2).sum(-1).sqrt()
    if_labels = torch.heaviside(threshold - if_dists, torch.tensor([0.0]))
    return if_labels


class SurfaceProcessor(ABC):
    def __init__(
        self,
        root: str = "surface_data",
        resolution: float = 1.0,
        sup_sampling: int = 20,
        distance: float = 1.05,
        debug: bool = False,
    ):
        super().__init__()
        if debug:
            root += "_debug"
        self.debug = debug
        self.raw_data_dir = Path(root) / "raw"
        self.processed_data_dir = Path(root) / "processed"
        self.tar_file = (
            self.raw_data_dir / "masif_site_masif_search_pdbs_and_ply_files.tar.gz"
        )
        self.surface_dir = self.raw_data_dir / "01-benchmark_surfaces"
        self.pdb_dir = self.raw_data_dir / "01-benchmark_pdbs"

        self.resolution = resolution
        self.sup_sampling = sup_sampling
        self.distance = distance

    @classmethod
    def from_config(cls, root_dir: str, cfg: DataConfig) -> SurfaceProcessor:
        cwd = os.getcwd()
        return cls(
            os.path.join(cwd, root_dir),
            cfg.resolution,
            cfg.sup_sampling,
            cfg.distance,
            cfg.debug,
        )

    def download(self):
        if not self.tar_file.exists():
            print("DOWNLOADING TAR FILE")
            download(self.tar_file, self.raw_data_dir)
        else:
            print("TAR FILE ALREADY DOWNLOADED")

    def extract(self):
        """Untar the ply and pdb files and convert them to numpy files"""

        if not (self.pdb_dir.exists() and self.surface_dir.exists()):
            print("EXTRACTING TAR FILE")
            untar(self.tar_file, self.raw_data_dir)
        else:
            print("TAR FILE ALREADY EXTRACTED")

    @abstractmethod
    def split(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_processed_data(
        self,
    ) -> tuple[list[ProteinProtocol], list[ProteinProtocol]]:
        raise NotImplementedError


class SiteProcessor(SurfaceProcessor):
    def split(self):
        """Load Protein objects from the preprocessed data and split by dataset"""
        print("LOADING DATASETS")
        lists_dir = Path("./lists")
        data_split = load_csv(lists_dir / "site_data_split.csv")

        train_data: list[Protein] = []
        test_data: list[Protein] = []
        for data in tqdm(data_split):
            try:
                protein = Protein.from_ply_and_pdb(
                    f"{data['pdb_id']}_{data['chains']}",
                    self.surface_dir,
                    self.pdb_dir,
                )
                if not protein_labels_valid(protein):
                    continue
            except FileNotFoundError:
                continue
            if data["split"] == "train":
                train_data.append(protein)
            elif data["split"] == "test":
                test_data.append(protein)

        pickle_dump(train_data, self.processed_data_dir / "site_train_data.pickle")
        pickle_dump(test_data, self.processed_data_dir / "site_test_data.pickle")

    def load_processed_data(
        self,
    ) -> tuple[list[ProteinProtocol], list[ProteinProtocol]]:
        if not (self.processed_data_dir / "site_train_data.pickle").exists():
            self.processed_data_dir.mkdir(exist_ok=True)
            self.download()
            self.extract()
            self.split()
        train_data = pickle_load(self.processed_data_dir / "site_train_data.pickle")
        test_data = pickle_load(self.processed_data_dir / "site_test_data.pickle")
        return train_data, test_data


class SearchProcessor(SurfaceProcessor):
    def split(self):
        """Load Protein objects from the preprocessed data and split by dataset"""
        print("LOADING DATASETS")
        lists_dir = Path("./lists")
        split_file = (
            "search_data_split_debug.csv" if self.debug else "search_data_split.csv"
        )
        data_split = load_csv(lists_dir / split_file)

        train_data: list[ProteinPair] = []
        test_data: list[ProteinPair] = []
        for data in tqdm(data_split):
            try:
                protein_1 = Protein.from_ply_and_pdb(
                    f"{data['pdb_id']}_{data['chain_a']}",
                    self.surface_dir,
                    self.pdb_dir,
                )

                protein_2 = Protein.from_ply_and_pdb(
                    f"{data['pdb_id']}_{data['chain_b']}",
                    self.surface_dir,
                    self.pdb_dir,
                )

                if not protein_labels_valid(protein_1) or not protein_labels_valid(
                    protein_2
                ):
                    continue

                if_labels = update_interface_labels(
                    protein_1,
                    protein_2,
                    threshold=2.0,
                )

                protein_1.center_protein()
                protein_2.center_protein()

                protein_pair = ProteinPair(protein_1, protein_2, if_labels)
            except FileNotFoundError:
                print(f"PDB id {data['pdb_id']} not found")
                continue

            if data["split"] == "train":
                train_data.append(protein_pair)
            elif data["split"] == "test":
                test_data.append(protein_pair)

        train_file = (
            "search_train_data_debug.pickle"
            if self.debug
            else "search_train_data.pickle"
        )
        test_file = (
            "search_test_data_debug.pickle" if self.debug else "search_test_data.pickle"
        )

        pickle_dump(train_data, self.processed_data_dir / train_file)
        pickle_dump(test_data, self.processed_data_dir / test_file)

    def load_processed_data(
        self,
    ) -> tuple[list[ProteinProtocol], list[ProteinProtocol]]:
        train_file = (
            "search_train_data_debug.pickle"
            if self.debug
            else "search_train_data.pickle"
        )
        test_file = (
            "search_test_data_debug.pickle" if self.debug else "search_test_data.pickle"
        )
        if not (self.processed_data_dir / train_file).exists():
            self.processed_data_dir.mkdir(exist_ok=True)
            self.download()
            self.extract()
            self.split()
        train_data = pickle_load(self.processed_data_dir / train_file)
        test_data = pickle_load(self.processed_data_dir / test_file)
        return train_data, test_data


PROCESSORS = {"search": SearchProcessor, "site": SiteProcessor}
