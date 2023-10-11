import torch
import requests
from pathlib import Path
import tarfile
import numpy as np
from protein import Protein
import csv
from data_preprocessing import convert_pdb2npy, convert_ply2npy


def download(url: str, target_dir: str):
    response = requests.get(url, stream=True, timeout=20)
    if response.status_code == 200:
        with open(target_dir, "wb") as f:
            f.write(response.raw.read())


def untar(file: str, target_dir: str):
    with tarfile.open(file) as tar:
        tar.extractall(target_dir)


def load_csv(csv_path: Path) -> list[dict[str, str]]:
    data: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for line in reader:
            data.append(line)
    return data


class SurfaceProcessor:
    def __init__(
        self,
        root: str = "surface_data",
        resolution: float = 1.0,
        sup_sampling: int = 20,
        distance: float = 1.05,
        precompute_surface_features: bool = False,
    ):
        super().__init__()
        self.root = root
        self.raw_file_names = "masif_site_masif_search_pdbs_and_ply_files.tar.gz"
        self.resolution = resolution
        self.sup_sampling = sup_sampling
        self.distance = distance
        self.precompute_surface_features = precompute_surface_features
        self.protein_dir = Path(self.root) / "raw" / "01-benchmark_surfaces_npy"

    @property
    def processed_file_names(self):
        return [
            "training_pairs_data.pt",
            "testing_pairs_data.pt",
            "training_pairs_data_ids.npy",
            "testing_pairs_data_ids.npy",
        ]

    def process(self):
        pdb_dir = Path(self.root) / "raw" / "01-benchmark_pdbs"
        surf_dir = Path(self.root) / "raw" / "01-benchmark_surfaces"

        # Untar surface files
        if not (pdb_dir.exists() and surf_dir.exists()):
            untar()

        if not self.protein_dir.exists():
            self.protein_dir.mkdir(parents=False, exist_ok=False)
            convert_plys(surf_dir, self.protein_dir)
            convert_pdbs(pdb_dir, self.protein_dir)

    def split(self):
        lists_dir = Path("./lists")

        data_split = load_csv(lists_dir / "single_protein_data.csv")

        training_data: list[Protein] = []
        testing_data: list[Protein] = []
        for data in data_split:
            protein = Protein.from_numpy(data["pdb_id"], self.protein_dir)
            if self.precompute_surface_features:
                protein.compute_surface_features(
                    self.resolution, self.sup_sampling, self.distance
                )
            if data["split"] == "train":
                training_data.append(protein)
            elif data["split"] == "test":
                testing_data.append(protein)

        np.save(self.processed_paths[2], training_pairs_data_ids)
        testing_pairs_data, testing_pairs_slices = self.collate(testing_pairs_data)
        torch.save((testing_pairs_data, testing_pairs_slices), self.processed_paths[1])
        np.save(self.processed_paths[3], testing_pairs_data_ids)
