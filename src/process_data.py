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

from data_preprocessing.convert_pdb2npy import convert_pdbs
from data_preprocessing.convert_ply2npy import convert_plys
from enums import Mode
from load_configs import DataConfig
from protein import Protein


def protein_labels_valid(protein: Protein) -> bool:
    """Check that the labels for the interaction site are reasonable"""
    labels = protein.mesh_labels.reshape(-1)
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


def pickle_load(file: Path) -> list[Protein] | list[tuple[Protein, Protein]]:
    with open(file, "rb") as handle:
        b = pickle.load(handle)
    return b


def pairwise_distance(coords_1: LazyTensor, coords_2: LazyTensor) -> LazyTensor:
    return ((coords_1[:, None, :] - coords_2[None, :, :]) ** 2).sum(-1).sqrt()


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
        self.features_dir = self.raw_data_dir / "01-benchmark_surfaces_npy"
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

    def preprocess(self):
        """Untar the ply and pdb files and convert them to numpy files"""

        if not (self.pdb_dir.exists() and self.surface_dir.exists()):
            print("EXTRACTING TAR FILE")
            untar(self.tar_file, self.raw_data_dir)
        else:
            print("TAR FILE ALREADY EXTRACTED")

        if not self.features_dir.exists():
            self.features_dir.mkdir(parents=False, exist_ok=False)
            convert_plys(self.surface_dir, self.features_dir)
            convert_pdbs(self.pdb_dir, self.features_dir)

    @abstractmethod
    def split(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_processed_data(self) -> list[Protein]:
        raise NotImplementedError


class SiteProcessor(SurfaceProcessor):
    def split(self, precompute_surface_features: bool = True):
        """Load Protein objects from the preprocessed data and split by dataset"""
        print("LOADING DATASETS")
        lists_dir = Path("./lists")
        data_split = load_csv(lists_dir / "site_data_split.csv")

        train_data: list[Protein] = []
        test_data: list[Protein] = []
        for data in tqdm(data_split):
            try:
                protein = Protein.from_numpy(
                    data["pdb_id"], data["chains"], self.features_dir
                )
                if not protein_labels_valid(protein):
                    continue
            except FileNotFoundError:
                continue
            if precompute_surface_features:
                protein.compute_surface_features(
                    self.resolution, self.sup_sampling, self.distance
                )
            if data["split"] == "train":
                train_data.append(protein)
            elif data["split"] == "test":
                test_data.append(protein)

        pickle_dump(train_data, self.processed_data_dir / "site_train_data")
        pickle_dump(test_data, self.processed_data_dir / "site_test_data")

    def load_processed_data(self):
        if not (self.processed_data_dir / "site_train_data.pickle").exists():
            self.processed_data_dir.mkdir()
            self.download()
            self.preprocess()
            self.split()
        train_data = pickle_load(self.processed_data_dir / "site_train_data.pickle")
        test_data = pickle_load(self.processed_data_dir / "site_test_data.pickle")
        return train_data, test_data


class SearchProcessor(SurfaceProcessor):
    def split(self, precompute_surface_features: bool = True):
        """Load Protein objects from the preprocessed data and split by dataset"""
        print("LOADING DATASETS")
        lists_dir = Path("./lists")
        split_file = (
            "search_data_split_debug.csv" if self.debug else "search_data_split.csv"
        )
        data_split = load_csv(lists_dir / split_file)

        train_data: list[tuple[Protein, Protein]] = []
        test_data: list[tuple[Protein, Protein]] = []
        for data in tqdm(data_split):
            try:
                protein_1 = Protein.from_numpy(
                    data["pdb_id"], data["chain_a"], self.features_dir
                )
                protein_2 = Protein.from_numpy(
                    data["pdb_id"], data["chain_b"], self.features_dir
                )
                if not protein_labels_valid(protein_1) or not protein_labels_valid(
                    protein_2
                ):
                    continue

            except FileNotFoundError:
                continue
            if precompute_surface_features:
                protein_1.compute_surface_features(
                    self.resolution, self.sup_sampling, self.distance
                )
                protein_2.compute_surface_features(
                    self.resolution, self.sup_sampling, self.distance
                )

            if data["split"] == "train":
                train_data.append((protein_1, protein_2))
            elif data["split"] == "test":
                test_data.append((protein_1, protein_2))

        train_file = (
            "search_train_data_clean_debug.pickle"
            if self.debug
            else "search_train_data_clean.pickle"
        )
        test_file = (
            "search_test_data_clean_debug.pickle"
            if self.debug
            else "search_test_data_clean.pickle"
        )

        pickle_dump(train_data, self.processed_data_dir / train_file)
        pickle_dump(test_data, self.processed_data_dir / test_file)

    def load_processed_data(self):
        train_file = (
            "search_train_data_clean_debug.pickle"
            if self.debug
            else "search_train_data_clean.pickle"
        )
        test_file = (
            "search_test_data_clean_debug.pickle"
            if self.debug
            else "search_test_data_clean.pickle"
        )
        if not (self.processed_data_dir / train_file).exists():
            print(self.processed_data_dir)
            self.processed_data_dir.mkdir(exist_ok=True)
            self.download()
            self.preprocess()
            self.split()
        train_data = pickle_load(self.processed_data_dir / train_file)
        test_data = pickle_load(self.processed_data_dir / test_file)
        return train_data, test_data


PROCESSORS = {Mode.SEARCH: SearchProcessor, Mode.SITE: SiteProcessor}
