#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kiba dataset preprocessing
"""

import os
import json
import warnings
import argparse
import logging
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

class PDBProtein:
    AA_NAME_SYM = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }

    AA_NAME_NUMBER = {k: i + 1 for i, (k, _) in enumerate(AA_NAME_SYM.items())}
    BACKBONE_NAMES = ["CA", "C", "N", "O", "OXT"]

    def __init__(self, data, mode="auto"):
        if (data[-4:].lower() == ".pdb" and mode == "auto") or mode == "path":
            with open(data, "r") as f:
                self.block = f.read()
        else:
            self.block = data

        # import rdkit only when necessary
        from rdkit import Chem

        self.ptable = Chem.GetPeriodicTable()

        # attributes
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom2residue = []

        self.residues = []
        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line.startswith("HEADER"):
                yield {"type": "HEADER", "value": line[10:].strip()}
                continue
            if not line.startswith("ATOM"):
                yield {"type": "others", "value": line}
                continue
            yield {
                "type": "ATOM",
                "atom_id": int(line[6:11]),
                "atom_name": line[12:16].strip(),
                "res_name": line[17:20].strip(),
                "chain": line[21:22].strip(),
                "res_id": int(line[22:26]),
                "res_insert_id": line[26:27].strip(),
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
                "occupancy": float(line[54:60]) if line[54:60].strip() else 1.0,
                "temp_factor": float(line[60:66]) if line[60:66].strip() else 0.0,
                "element_symb": line[76:78].strip(),
                "line": line.strip(),
            }

    def _parse(self):
        residues_tmp = {}
        num_residue = -1
        for atom in self._enum_formatted_atom_lines():
            if atom["type"] != "ATOM":
                continue
            if atom["atom_name"].startswith("H"):
                continue

            self.atoms.append(atom)
            anum = self.ptable.GetAtomicNumber(atom["element_symb"])
            self.element.append(anum)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(anum))
            self.pos.append(np.array([atom["x"], atom["y"], atom["z"]], dtype=np.float32))
            self.atom_name.append(atom["atom_name"])
            self.is_backbone.append(atom["atom_name"] in self.BACKBONE_NAMES)

            res_key = (atom["chain"], atom["res_id"], atom["res_insert_id"])
            if res_key not in residues_tmp:
                num_residue += 1
                residues_tmp[res_key] = {
                    "name": atom["res_name"],
                    "atoms": [],
                    "chain": atom["chain"],
                    "id": atom["res_id"],
                    "insert_id": atom["res_insert_id"],
                }
            residues_tmp[res_key]["atoms"].append(len(self.element) - 1)
            self.atom2residue.append(num_residue)

        self.residues = list(residues_tmp.values())
        for residue in self.residues:
            total_mass = 0.0
            total_pos = np.zeros(3)
            for idx in residue["atoms"]:
                mass = self.atomic_weight[idx]
                total_pos += self.pos[idx] * mass
                total_mass += mass
            residue["center_of_mass"] = total_pos / total_mass
            for idx in residue["atoms"]:
                name = self.atom_name[idx]
                if name in self.BACKBONE_NAMES:
                    residue[f"pos_{name}"] = self.pos[idx]

    def query_residues_ligand(self, ligand_center, radius):
        keep = []
        for res in self.residues:
            if np.linalg.norm(res["center_of_mass"] - ligand_center) <= radius:
                keep.append(res)
        return keep


def smiles_to_3d_coords(smiles: str, seed: int = 42):
    """return (atomic_numbers, coords) or (None, None)"""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
            return None, None
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)

        nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
        conf = mol.GetConformer()
        coords = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
        return np.array(nums, dtype=np.int32), np.array(coords, dtype=np.float32)
    except Exception as e:
        logging.warning(f"SMILES parse error: {smiles} -> {e}")
        return None, None


def process_one(entry: Tuple[int, str, str, float], pdb_dir: str, pocket_radius: float, seed: int):
    idx, smiles, protein, affinity = entry
    pdb_path = os.path.join(pdb_dir, f"{protein}.pdb")
    if not os.path.isfile(pdb_path):
        return None, f"missing_pdb:{protein}"

    l_nums, l_coords = smiles_to_3d_coords(smiles, seed=seed)
    if l_nums is None:
        return None, f"bad_smiles:{smiles}"

    lig_center = l_coords.mean(axis=0)
    try:
        prot = PDBProtein(pdb_path, mode="path")
    except Exception as e:
        return None, f"pdb_parse_error:{protein}:{e}"

    residues = prot.query_residues_ligand(lig_center, pocket_radius)
    if not residues:
        return None, f"empty_pocket:{protein}"

    p_nums, p_coords = [], []
    for res in residues:
        for atom_idx in res["atoms"]:
            p_nums.append(prot.element[atom_idx])
            p_coords.append(prot.pos[atom_idx])

    p_nums = np.array(p_nums, dtype=np.int32)
    p_coords = np.array(p_coords, dtype=np.float32)
    l_nums = l_nums.astype(np.int32)
    l_coords = l_coords.astype(np.float32)

    charges = np.concatenate([p_nums, l_nums])
    positions = np.concatenate([p_coords, l_coords])

    return {
        "index": idx,
        "num_atoms": len(charges),
        "pocket_atoms": len(p_nums),
        "ligand_atoms": len(l_nums),
        "charges": charges,
        "positions": positions,
        "neglog_aff": float(affinity),
        "smiles": smiles,
        "protein_name": protein,
    }, None


def build_dict(processed: List[Dict[str, Any]], max_atoms_clip=None):
    if not processed:
        return {}
    
    max_atoms = max(p["num_atoms"] for p in processed)
    if max_atoms_clip:
        max_atoms = min(max_atoms, max_atoms_clip)

    data = {
        "index": [],
        "num_atoms": [],
        "pocket_atoms": [],
        "ligand_atoms": [],
        "charges": [],
        "positions": [],
        "neglog_aff": [],
        "smiles": [],
        "protein_names": [],
    }
    for p in processed:
        n = p["num_atoms"]
        data["index"].append(p["index"])
        data["num_atoms"].append(n)
        data["pocket_atoms"].append(p["pocket_atoms"])
        data["ligand_atoms"].append(p["ligand_atoms"])
        data["neglog_aff"].append(p["neglog_aff"])
        data["smiles"].append(p["smiles"])
        data["protein_names"].append(p["protein_name"])

        ch = np.zeros(max_atoms, dtype=np.int32)
        ch[:n] = p["charges"]
        pos = np.zeros((max_atoms, 3), dtype=np.float32)
        pos[:n] = p["positions"]

        data["charges"].append(ch)
        data["positions"].append(pos)

    # list -> np.ndarray
    for k in ("index", "num_atoms", "pocket_atoms", "ligand_atoms", "neglog_aff"):
        data[k] = np.array(data[k])
    data["charges"] = np.array(data["charges"])
    data["positions"] = np.array(data["positions"])
    return data


def group_split(df: pd.DataFrame, group_col: str, seed: int):
    from sklearn.model_selection import GroupShuffleSplit

    groups = df[group_col].values
    idx = np.arange(len(df))
    gss1 = GroupShuffleSplit(train_size=0.8, random_state=seed)
    train_idx, tmp_idx = next(gss1.split(idx, groups=groups))
    gss2 = GroupShuffleSplit(train_size=0.5, random_state=seed)
    valid_idx, test_idx = next(gss2.split(tmp_idx, groups=groups[tmp_idx]))
    return train_idx, tmp_idx[valid_idx], tmp_idx[test_idx]


def main():
    parser = argparse.ArgumentParser(description="Kiba dataset preprocessing")
    parser.add_argument("--csv_path", default="./full.csv")
    parser.add_argument("--pdb_dir", default="./prot_3d_for_Kiba")
    parser.add_argument("--output_dir", default="./data")
    parser.add_argument("--pocket_radius", type=float, default=8.0)
    parser.add_argument("--max_atoms", type=int, help="clip max atoms per sample")
    parser.add_argument("--num_workers", type=int, default=min(cpu_count(), 8))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)

    if not os.path.isfile(args.csv_path):
        raise FileNotFoundError(args.csv_path)

    df = pd.read_csv(args.csv_path)
    logging.info(f"Loaded {len(df)} rows from {args.csv_path}")

    entries = [(i, row["ligand"], row["protein"], row["label"]) for i, row in df.iterrows()]

    process_fn = partial(
        process_one,
        pdb_dir=args.pdb_dir,
        pocket_radius=args.pocket_radius,
        seed=args.seed,
    )

    success, fails = [], []
    with Pool(args.num_workers) as pool:
        for res, err in tqdm(
            pool.imap(process_fn, entries), total=len(entries), desc="processing"
        ):
            if res:
                success.append(res)
            else:
                print(err)
                fails.append(err)

    logging.info(f"Success: {len(success)}  Fail: {len(fails)}")
    if fails:
        fail_log = os.path.join(args.output_dir, "failed_samples.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(fail_log, "w") as f:
            json.dump(fails, f, indent=2)
        logging.warning(f"Failed samples saved to {fail_log}")

    data = build_dict(success, max_atoms_clip=args.max_atoms)
    
    success_df = pd.DataFrame([{
        "index": item["index"],
        "protein": item["protein_name"],
        "ligand": item["smiles"],
        "label": item["neglog_aff"]
    } for item in success])
    
    train_idx, valid_idx, test_idx = group_split(success_df, group_col="protein", seed=args.seed)
    
    splits = {
        "train": {k: v[train_idx] if k not in ("smiles", "protein_names") else [v[i] for i in train_idx] for k, v in data.items()},
        "valid": {k: v[valid_idx] if k not in ("smiles", "protein_names") else [v[i] for i in valid_idx] for k, v in data.items()},
        "test": {k: v[test_idx] if k not in ("smiles", "protein_names") else [v[i] for i in test_idx] for k, v in data.items()},
    }

    os.makedirs(args.output_dir, exist_ok=True)
    for split, d in splits.items():
        path = os.path.join(args.output_dir, f"data_{split}.npy")
        np.save(path, d, allow_pickle=True)
        logging.info(f"{split}: {len(d['index'])} samples -> {path}")

    logging.info("All done.")


if __name__ == "__main__":
    main()