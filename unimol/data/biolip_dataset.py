import random
import numpy as np
from functools import lru_cache
import torch
from torch_geometric.data import (InMemoryDataset, Data)
from unicore.data import BaseWrapperDataset
import lmdb
import pickle as pk

class BiolipDataset(InMemoryDataset):
    def __init__(self, file_path, max_num=512):
        self.env = lmdb.open(
            file_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        self.txn = self.env.begin()
        keys = list(self.txn.cursor().iternext(values=False))
        self.length = len(keys)
        self.max_num = max_num

        # Store the actual keys for debugging
        self.keys = [key.decode() for key in keys]
        print(f"Biolip Data initialized with {self.length} entries")
        if self.length > 0:
            print(f"First few keys: {self.keys[:min(5, len(self.keys))]}")
            print(f"Last few keys: {self.keys[-min(5, len(self.keys)):]}")
    
    def __len__(self) -> int:
        return self.length

    
    def __getitem__(self, idx):

        data = Data()
        data.idx = idx
        ky = f'{idx}'.encode()
        datapoint_pickled = self.txn.get(ky)
        if datapoint_pickled is None:
            return self.__getitem__(random.randint(0, self.length - 1))

        data_item = pk.loads(datapoint_pickled)

        data.atoms = data_item['atoms']
        data.coordinates = data_item['lig_coord_real'] # rdkit pose
        data.pocket_atoms = data_item['pocket_atoms']
        data.pocket_coordinates = data_item['pocket_coordinates']
        data.smi = data_item['smi']
        data.pocket_name = data_item.get('protein_name', data_item.get('uniprot_id'))
        data.solvent_coordinates = data_item['solvent_coordinates']
        
        lig_coord_real = data_item['lig_coord_real'] # real pose
        
        # random pick one 
        real_len = len(lig_coord_real)
        real_idx = random.randint(0, real_len - 1)
        data.lig_coord_real = lig_coord_real[real_idx]
        
        if len(data.pocket_atoms) > self.max_num:
            org_len = len(data.pocket_atoms)
            random_idx = random.sample(range(org_len), self.max_num)
            data.pocket_atoms = np.array(data_item['pocket_atoms'])[random_idx]
            data.pocket_coordinates = data_item['pocket_coordinates'][random_idx]
        
        return data
    

class ExtractCPConformerDataset(BaseWrapperDataset):
    def __init__(self, dataset, smi, atoms, coordinates, mask_feat=False, mask_ratio=0.8):
        self.dataset = dataset
        self.smi = smi
        self.atoms = atoms
        self.coordinates = coordinates
        self.set_epoch(None)

        self.mask_feat = mask_feat
        self.mask_ratio = mask_ratio

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        data = self.dataset[index]
            
        pocket_atoms = data.pocket_atoms
        pocket_coordinates = torch.tensor(data.pocket_coordinates)
        
        if len(data.solvent_coordinates) != len(data.coordinates):
            data.solvent_coordinates = data.coordinates
        lig_atoms = data.atoms
        lig_coordinates = torch.tensor(data.coordinates)
        ori_lig_coordinates = torch.tensor(data.coordinates)
        solvent_lig_coordinates = torch.tensor(data.solvent_coordinates)
        ori_solvent_lig_coordinates = torch.tensor(data.solvent_coordinates)
        all_coordinates = torch.tensor(np.concatenate((pocket_coordinates, lig_coordinates), axis=0))
        pocket_len = len(pocket_atoms)
        lig_len = len(lig_atoms)
        idx = data.idx
        smi = data.smi

        res = {"smi": smi, "atoms": pocket_atoms, "coordinates": pocket_coordinates, 'lig_atoms': lig_atoms, "lig_coordinates": lig_coordinates, "ori_lig_coordinates": ori_lig_coordinates, "solvent_lig_coordinates": solvent_lig_coordinates, "ori_solvent_lig_coordinates": ori_solvent_lig_coordinates, 'all_coordinates': all_coordinates, "pocket_idx": idx, "pocket_len": pocket_len, "lig_len": lig_len}

        if self.mask_feat:
            sample_size = int(lig_len * self.mask_ratio + 1)
            masked_atom_indices = random.sample(range(lig_len), sample_size)
            mask_array = np.zeros(lig_len, dtype=np.float32)
            mask_array[masked_atom_indices] = 1
            res['mask_array'] = torch.tensor(mask_array)

        return res

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)