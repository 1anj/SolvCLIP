#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : data_preparation.py
# @Time    : 2021/07/25 16:11:32
import math
import os
import pickle as pkl
import lmdb
import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

FILE_PATH         = "./BioLiP_nr.txt"
LMDB_PATH         = "./lmdb_atom3dPocket"
MAP_SIZE          = 100_995_116_277_76
N_READ_WORKERS    = min(cpu_count(), 8)
N_PROC_WORKERS    = min(cpu_count(), 32)
PROC_CHUNK_LINES  = 2_000
ERR_FILE          = "./no_atom3d_line.txt"


def _read_chunk(path: str, offset: int, size: int):
    lines = []
    with open(path, 'r', buffering=1 << 20) as f:
        f.seek(offset)
        if offset != 0:
            f.readline() 
        while True:
            pos = f.tell()
            if pos >= offset + size:
                break
            line = f.readline()
            if not line:        # EOF
                break
            lines.append(line)
    return lines

def parallel_read_lines(path: str, n_workers=None, chunk_size=None):
    file_size = os.path.getsize(path)
    if n_workers is None:
        n_workers = min(cpu_count(), 8)
    if chunk_size is None:
        chunk_size = max(1, file_size // n_workers)

    offsets = list(range(0, file_size, chunk_size))
    read_partial = partial(_read_chunk, path, chunk_size)
    all_lines = []
    with Pool(n_workers) as pool:
        for chunk_lines in tqdm(pool.imap(read_partial, offsets),
                                total=len(offsets),
                                desc="Reading file"):
            all_lines.extend(chunk_lines)
    return all_lines

def process_one_line_pack(args):
    idx, line = args
    try:
        items = line.strip().split('\t')
        if len(items) < 7:
            return idx, None

        protein_name, protein_chain = items[0], items[1]
        ligand_name, ligand_chain, lig_serial = items[4], items[5], items[6]
        smi = items[2]

        protein_pdb = f"./BioLiP/receptor/{protein_name}{protein_chain}.pdb"
        ligand_pdb  = (f"./BioLiP/ligand/"
                       f"{protein_name}_{ligand_name}_{ligand_chain}_{lig_serial}.pdb")

        lig_coords, lig_atoms = [], []
        with open(ligand_pdb) as f:
            for ln in f:
                if ln.startswith('TER'):
                    continue
                x, y, z = map(float, [ln[30:38], ln[38:46], ln[46:54]])
                sym = ln[76:78].strip()
                lig_coords.append([x, y, z])
                lig_atoms.append(sym)
        lig_coords = np.array(lig_coords, dtype=np.float32)
        lig_center = lig_coords.mean(axis=0)
        lig_radius = np.sqrt(((lig_coords - lig_center) ** 2).sum(axis=1)).mean()

        prot_coords, prot_atoms, resi_nums = [], [], []
        ca_info = []                       # (resi_num, dist)
        with open(protein_pdb) as f:
            for ln in f:
                if ln.startswith('TER'):
                    continue
                resi_num = int(ln[22:26])
                x, y, z = map(float, [ln[30:38], ln[38:46], ln[46:54]])
                sym = ln[76:78].strip()
                if ln[12:16].strip() == 'CA':
                    ca = np.array([x, y, z], dtype=np.float32)
                    dist = np.linalg.norm(ca - lig_center)
                    # <= 8 AA
                    if dist <= 8 + lig_radius:
                        ca_info.append((resi_num, dist))
                resi_nums.append(resi_num)
                prot_coords.append([x, y, z])
                prot_atoms.append(sym)

        prot_coords = np.array(prot_coords, dtype=np.float32)

        # atom nums <= 1500
        retain_resi = {r[0] for r in ca_info}
        pocket_idx = np.where(pd.Series(resi_nums).isin(retain_resi))[0]
        while len(pocket_idx) + len(lig_coords) > 1500 and ca_info:
            ca_info = sorted(ca_info, key=lambda x: x[1])[:-1]
            retain_resi = {r[0] for r in ca_info}
            pocket_idx = np.where(pd.Series(resi_nums).isin(retain_resi))[0]

        if len(pocket_idx) == 0:
            return idx, None

        pocket_coords = prot_coords[pocket_idx]
        pocket_atoms  = list(np.array(prot_atoms)[pocket_idx])

        data = dict(pocket_atoms=pocket_atoms,
                    pocket_coordinates=pocket_coords,
                    lig_atoms_real=lig_atoms,
                    atoms=lig_atoms,
                    protein_name=protein_name,
                    lig_coord_real=lig_coords.tolist(),
                    smi=smi)
        return idx, pkl.dumps(data)

    except Exception:
        return idx, None


def main():
    os.makedirs(os.path.dirname(ERR_FILE), exist_ok=True)

    lines = parallel_read_lines(FILE_PATH)
    if not lines:
        print("File read error: ", FILE_PATH)
        return

    env = lmdb.open(LMDB_PATH, map_size=MAP_SIZE, lock=False)
    with env.begin(write=True) as txn, open(ERR_FILE, 'w') as err_f:
        indexed_lines = list(enumerate(lines))
        with ProcessPoolExecutor(max_workers=N_PROC_WORKERS) as exe:
            results = exe.map(process_one_line_pack,
                              indexed_lines,
                              chunksize=PROC_CHUNK_LINES)
            for idx, serialized in tqdm(results, total=len(lines), desc="Parsing"):
                if serialized:
                    txn.put(str(idx).encode(), serialized)
                else:
                    err_f.write(lines[idx])

    env.close()
    print("✅ All done!")

if __name__ == "__main__":
    main()
