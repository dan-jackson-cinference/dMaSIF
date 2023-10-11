from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from tqdm import tqdm

ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}


def load_structure_np(fname, center):
    """Loads a .ply mesh to return a point cloud and connectivity."""
    # Load the data
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    coords = []
    types = []
    for atom in atoms:
        coords.append(atom.get_coord())
        types.append(ele2num[atom.element])

    coords = np.stack(coords)
    types_array = np.zeros((len(types), len(ele2num)))
    for i, t in enumerate(types):
        types_array[i, t] = 1.0

    # Normalize the coordinates, as specified by the user:
    if center:
        coords = coords - np.mean(coords, axis=0, keepdims=True)

    return {"xyz": coords, "types": types_array}


def convert_pdbs(pdb_dir: Path, npy_dir: Path):
    print("Converting PDBs")
    for p in tqdm(pdb_dir.glob("*.pdb")):
        protein = load_structure_np(p, center=False)
        np.save(npy_dir / (p.stem + "_atomxyz.npy"), protein["xyz"])
        np.save(npy_dir / (p.stem + "_atomtypes.npy"), protein["types"])


def convert_single_pdb(pdb_file: str, output_dir: str) -> None:
    print("Converting PDBs")
    protein = load_structure_np(pdb_file, center=False)
    output_path = Path(output_dir)
    pdb_path = Path(pdb_file)
    np.save(output_path / (pdb_path.stem + "_atomxyz.npy"), protein["xyz"])
    np.save(output_path / (pdb_path.stem + "_atomtypes.npy"), protein["types"])


if __name__ == "__main__":
    convert_single_pdb(
        "/home/danjackson/repos/original_repos/dMaSIF/1f6m_F.pdb",
        "data_preprocessing/npys",
    )
