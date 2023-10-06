from __future__ import annotations

import tarfile
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from data_preprocessing.convert_pdb2npy import convert_pdbs
from data_preprocessing.convert_ply2npy import convert_plys
from features import FeatureExtractor
from geometry_processing import atoms_to_points_normals
from protein import Protein
from transforms import CenterPairAtoms, NormalizeChemFeatures, RandomRotationPairAtoms


def numpy(x):
    return x.detach().cpu().numpy()


def iface_valid_filter(protein_pair: PairData):
    labels1 = protein_pair.protein_1.mesh_labels.reshape(-1)
    labels2 = protein_pair.protein_2.mesh_labels.reshape(-1)
    valid1 = (
        (torch.sum(labels1) < 0.75 * len(labels1))
        and (torch.sum(labels1) > 30)
        and (torch.sum(labels1) > 0.01 * labels2.shape[0])
    )
    valid2 = (
        (torch.sum(labels2) < 0.75 * len(labels2))
        and (torch.sum(labels2) > 30)
        and (torch.sum(labels2) > 0.01 * labels1.shape[0])
    )

    return valid1 and valid2


class PairData(Data):
    def __init__(self, p1: Protein = None, p2: Optional[Protein] = None):
        super().__init__()
        self.protein_1 = p1
        self.protein_2 = p2

    def __inc__(self, key: str, value, *args, **kwargs):
        if key == "face_p1":
            return self.protein_1.xyz.size(0)
        if key == "face_p2":
            return self.protein_2.xyz.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        if ("index" in key) or ("face" in key):
            return 1
        return 0


def load_protein_pair(
    pdb_id: str,
    data_dir: Path,
    resolution: float,
    sup_sampling: int,
    distance: float,
    single_pdb: bool = False,
    precompute_surface_features: bool = False,
):
    """Loads a protein surface mesh and its features"""

    pspl = pdb_id.split("_")
    p1_id = pspl[0] + "_" + pspl[1]
    p2_id = pspl[0] + "_" + pspl[2]

    protein_1 = Protein.from_numpy(p1_id, data_dir, center=False, single_pdb=single_pdb)
    protein_2 = Protein.from_numpy(p2_id, data_dir, center=False, single_pdb=single_pdb)

    if precompute_surface_features:
        protein_1.compute_surface_features(resolution, sup_sampling, distance)
        protein_2.compute_surface_features(resolution, sup_sampling, distance)

    return PairData(protein_1, protein_2)


class Mode(Enum):
    SITE = "site"
    SEARCH = "search"


FILES = {
    Mode.SITE: [
        "training_pairs_data.pt",
        "testing_pairs_data.pt",
        "training_pairs_data_ids.npy",
        "testing_pairs_data_ids.npy",
    ],
    Mode.SEARCH: [
        "training_pairs_data_ppi.pt",
        "testing_pairs_data_ppi.pt",
        "training_pairs_data_ids_ppi.npy",
        "testing_pairs_data_ids_ppi.npy",
    ],
}

PDB_IDS = {
    Mode.SITE: "surface_data/processed/testing_pairs_data_ids.npy",
    Mode.SEARCH: "surface_data/processed/testing_pairs_data_ids_ppi.npy",
}


class ProteinPairsSurfaces(InMemoryDataset):
    url = ""

    def __init__(
        self,
        root: str,
        resolution: float,
        sup_sampling: int,
        distance: float,
        mode: Mode,
        train: bool = True,
        precompute_surface_features: bool = False,
        transform=None,
        pre_transform=None,
    ):
        self.resolution = resolution
        self.sup_sampling = sup_sampling
        self.distance = distance
        self.mode = mode
        self.precompute_surface_features = precompute_surface_features

        super().__init__(root, transform, pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]

        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return "masif_site_masif_search_pdbs_and_ply_files.tar.gz"

    @property
    def processed_file_names(self):
        return [
            "training_pairs_data.pt",
            "testing_pairs_data.pt",
            "training_pairs_data_ids.npy",
            "testing_pairs_data_ids.npy",
        ]

    def download(self):
        url = "https://zenodo.org/record/2625420/files/masif_site_masif_search_pdbs_and_ply_files.tar.gz"
        target_path = self.raw_paths[0]
        response = requests.get(url, stream=True, timeout=20)
        if response.status_code == 200:
            with open(target_path, "wb") as f:
                f.write(response.raw.read())

        # raise RuntimeError(
        #    "Dataset not found. Please download {} from {} and move it to {}".format(
        #        self.raw_file_names, self.url, self.raw_dir
        #    )
        # )

    def process(self):
        pdb_dir = Path(self.root) / "raw" / "01-benchmark_pdbs"
        surf_dir = Path(self.root) / "raw" / "01-benchmark_surfaces"
        protein_dir = Path(self.root) / "raw" / "01-benchmark_surfaces_npy"
        lists_dir = Path("./lists")

        # Untar surface files
        if not (pdb_dir.exists() and surf_dir.exists()):
            with tarfile.open(self.raw_paths[0]) as tar:
                tar.extractall(self.raw_dir)

        if not protein_dir.exists():
            protein_dir.mkdir(parents=False, exist_ok=False)
            convert_plys(surf_dir, protein_dir)
            convert_pdbs(pdb_dir, protein_dir)

        with open(lists_dir / "training.txt") as f_tr, open(
            lists_dir / "testing.txt"
        ) as f_ts:
            training_list = sorted(f_tr.read().splitlines())
            testing_list = sorted(f_ts.read().splitlines())

        with open(lists_dir / "training_ppi.txt") as f_tr, open(
            lists_dir / "testing_ppi.txt"
        ) as f_ts:
            training_pairs_list = sorted(f_tr.read().splitlines())
            testing_pairs_list = sorted(f_ts.read().splitlines())
            pairs_list = sorted(training_pairs_list + testing_pairs_list)

        if self.mode == Mode.SITE:
            training_pairs_list: list[str] = []
            for p in pairs_list:
                pspl = p.split("_")
                p1 = pspl[0] + "_" + pspl[1]
                p2 = pspl[0] + "_" + pspl[2]

                if p1 in training_list:
                    training_pairs_list.append(p)
                if p2 in training_list:
                    training_pairs_list.append(pspl[0] + "_" + pspl[2] + "_" + pspl[1])

            testing_pairs_list: list[str] = []
            for p in pairs_list:
                pspl = p.split("_")
                p1 = pspl[0] + "_" + pspl[1]
                p2 = pspl[0] + "_" + pspl[2]
                if p1 in testing_list:
                    testing_pairs_list.append(p)
                if p2 in testing_list:
                    testing_pairs_list.append(pspl[0] + "_" + pspl[2] + "_" + pspl[1])

        # # Read data into huge `Data` list.
        training_pairs_data: list[PairData] = []
        training_pairs_data_ids: list[str] = []
        for pdb_id in training_pairs_list:
            try:
                protein_pair = load_protein_pair(
                    pdb_id,
                    protein_dir,
                    self.resolution,
                    self.sup_sampling,
                    self.distance,
                    precompute_surface_features=self.precompute_surface_features,
                )
            except FileNotFoundError:
                continue
            training_pairs_data.append(protein_pair)
            training_pairs_data_ids.append(pdb_id)

        testing_pairs_data: list[PairData] = []
        testing_pairs_data_ids: list[str] = []
        for pdb_id in testing_pairs_list:
            try:
                protein_pair = load_protein_pair(
                    pdb_id,
                    protein_dir,
                    self.resolution,
                    self.sup_sampling,
                    self.distance,
                    precompute_surface_features=self.precompute_surface_features,
                )
            except FileNotFoundError:
                continue
            testing_pairs_data.append(protein_pair)
            testing_pairs_data_ids.append(pdb_id)

        if self.pre_filter is not None:
            training_pairs_data = [
                data for data in training_pairs_data if self.pre_filter(data)
            ]
            testing_pairs_data = [
                data for data in testing_pairs_data if self.pre_filter(data)
            ]

        if self.pre_transform is not None:
            training_pairs_data = [
                self.pre_transform(data) for data in training_pairs_data
            ]
            testing_pairs_data = [
                self.pre_transform(data) for data in testing_pairs_data
            ]

        training_pairs_data, training_pairs_slices = self.collate(training_pairs_data)
        torch.save(
            (training_pairs_data, training_pairs_slices), self.processed_paths[0]
        )
        np.save(self.processed_paths[2], training_pairs_data_ids)
        testing_pairs_data, testing_pairs_slices = self.collate(testing_pairs_data)
        torch.save((testing_pairs_data, testing_pairs_slices), self.processed_paths[1])
        np.save(self.processed_paths[3], testing_pairs_data_ids)


def load_test_data(
    random_rotation: bool,
    single_pdb: str,
    pdb_list: str,
    mode: Mode,
    batch_size: int,
) -> DataLoader:
    """Load the data you want to test. This can either be the test dataset or a single pdbfile"""
    transformations = (
        Compose([NormalizeChemFeatures(), CenterPairAtoms(), RandomRotationPairAtoms()])
        if random_rotation
        else Compose([NormalizeChemFeatures()])
    )

    if single_pdb != "":
        single_data_dir = Path(
            "/home/danjackson/repos/original_repos/dMaSIF/data_preprocessing/npys"
        )
        test_dataset = [load_protein_pair(single_pdb, single_data_dir, single_pdb=True)]
        test_pdb_ids = [single_pdb]
    elif pdb_list != "":
        with open(pdb_list) as f:
            pdb_list = f.read().splitlines()
        single_data_dir = Path("./data_preprocessing/npys/")
        test_dataset = [
            load_protein_pair(pdb, single_data_dir, single_pdb=True) for pdb in pdb_list
        ]
        test_pdb_ids = list(pdb_list)
    else:
        test_dataset = ProteinPairsSurfaces(
            "surface_data", train=False, mode=mode, transform=transformations
        )
        test_pdb_ids = np.load(PDB_IDS[mode])

        test_dataset = [
            (data, pdb_id)
            for data, pdb_id in zip(test_dataset, test_pdb_ids)
            if iface_valid_filter(data)
        ]
        test_dataset, test_pdb_ids = list(zip(*test_dataset))

    # PyTorch geometric expects an explicit list of "batched variables":
    batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, follow_batch=batch_vars
    )
    return test_loader


def load_training_data(
    resolution: float,
    sup_sampling: int,
    distance: float,
    random_rotation: bool,
    mode: Mode,
    validation_fraction: float,
) -> dict[str, DataLoader]:
    transformations = (
        Compose([NormalizeChemFeatures(), CenterPairAtoms(), RandomRotationPairAtoms()])
        if random_rotation
        else Compose([NormalizeChemFeatures()])
    )

    if not Path("models/").exists():
        Path("models/").mkdir(exist_ok=False)

    # PyTorch geometric expects an explicit list of "batched variables":
    batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]
    # Load the train dataset:
    train_dataset = ProteinPairsSurfaces(
        "surface_data",
        resolution,
        sup_sampling,
        distance,
        mode=mode,
        precompute_surface_features=True,
        train=True,
        transform=transformations,
    )
    train_dataset = [data for data in train_dataset if iface_valid_filter(data)]

    # # Train/Validation split:
    train_nsamples = len(train_dataset)
    val_nsamples = int(train_nsamples * validation_fraction)
    train_nsamples = train_nsamples - val_nsamples
    train_dataset, val_dataset = random_split(
        train_dataset, [train_nsamples, val_nsamples]
    )

    # PyTorch_geometric data loaders:
    train_loader = DataLoader(
        train_dataset, batch_size=1, follow_batch=batch_vars, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, follow_batch=batch_vars)

    return {"train": train_loader, "val": val_loader}
