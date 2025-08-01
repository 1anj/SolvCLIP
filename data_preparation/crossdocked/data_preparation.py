import os
import argparse
import multiprocessing as mp
import pickle
import shutil
from functools import partial

import numpy as np
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import lmdb

ATOM_TYPES = [
    '', 'N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]

NUM_ATOMS = [4, 5, 11, 8, 8, 6, 9, 9, 4, 10, 8, 8, 9, 8, 11, 7, 6, 7, 14, 12, 7]

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']

ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}

class PDBProtein(object):
    AA_NAME_SYM = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
                   'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
                   'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

    AA_NAME_NUMBER = {
        k: i + 1 for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, mode='auto'):
        super().__init__()
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        self.atom2residue = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.amino_idx = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []
        self.residue_natoms = []
        self.seq = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break  # Some PDBs have more than 1 model.
            else:
                yield {
                    'type': 'others'
                }

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        num_residue = -1
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            if atom['type'] == 'others':
                continue
            if atom['atom_name'][0] == 'H' or atom['atom_name'] == 'OXT':
                continue
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            next_ptr = len(self.element)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                num_residue += 1
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                    'res_id': atom['res_id'],
                    'full_seq_idx': num_residue,
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)
            self.atom2residue.append(num_residue)

        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass
            self.residue_natoms.append(len(residue['atoms']))
            assert len(residue['atoms']) <= NUM_ATOMS[self.AA_NAME_NUMBER[residue['name']]]

            # Process backbone atoms of residues
            self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            self.amino_idx.append(residue['res_id'])
            self.seq.append(self.AA_NAME_SYM[residue['name']])
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name  # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

        # convert atom_name to number
        self.atom_name = np.array([ATOM_TYPES.index(atom) for atom in self.atom_name])
        self.pos = np.array(self.pos, dtype=np.float32)

    def to_dict_atom(self):
        return {
            'element': np.array(self.element, dtype=np.longlong),
            'molecule_name': self.title,
            'pos': self.pos,
            'is_backbone': np.array(self.is_backbone, dtype=bool),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.longlong),
            'atom2residue': np.array(self.atom2residue, dtype=np.longlong)
        }

    def to_dict_residue(self):
        return {
            'seq': self.seq,
            'res_idx': np.array(self.amino_idx, dtype=np.longlong),
            'amino_acid': np.array(self.amino_acid, dtype=np.longlong),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
            'residue_natoms': np.array(self.residue_natoms, dtype=np.longlong),
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(self, ligand, radius=3.5, selected_residue=None, return_mask=True):
        selected = []
        sel_idx = set()
        selected_mask = np.zeros(len(self.residues), dtype=bool)
        full_seq_idx = set()
        if selected_residue is None:
            selected_residue = self.residues
        # The time-complexity is O(mn).
        for i, residue in enumerate(selected_residue):
            for center in ligand['pos']:
                distance = np.min(np.linalg.norm(self.pos[residue['atoms']] - center, ord=2, axis=1))
                if distance <= radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
                    full_seq_idx.add(residue['full_seq_idx'])
                    break
        selected_mask[list(sel_idx)] = 1
        if return_mask:
            return list(full_seq_idx), selected_mask
        return list(full_seq_idx), selected

    # can be used for select pocket residues

    def residues_to_pdb_block(self, residues, name='POCKET'):
        block = "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block

    def return_residues(self):
        return self.residues, self.atoms


def load_item(item, path):
    pdb_path = os.path.join(path, item[0])
    sdf_path = os.path.join(path, item[1])
    with open(pdb_path, 'r') as f:
        pdb_block = f.read()
    with open(sdf_path, 'r') as f:
        sdf_block = f.read()
    return pdb_block, sdf_block

def parse_sdf_file(path, feat=True):
    mol = Chem.MolFromMolFile(path, sanitize=False)
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    rd_num_atoms = mol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.longlong)
    if feat:
        rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=True)))
        for feat in factory.GetFeaturesForMol(rdmol):
            feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    with open(path, 'r') as f:
        sdf = f.read()

    sdf = sdf.splitlines()
    num_atoms, num_bonds = map(int, [sdf[3][0:3], sdf[3][3:6]])
    assert num_atoms == rd_num_atoms

    ptable = Chem.GetPeriodicTable()
    element, pos = [], []
    accum_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    accum_mass = 0.0
    for atom_line in map(lambda x: x.split(), sdf[4:4 + num_atoms]):
        x, y, z = map(float, atom_line[:3])
        symb = atom_line[3]
        atomic_number = ptable.GetAtomicNumber(symb.capitalize())
        element.append(atomic_number)
        pos.append([x, y, z])

        atomic_weight = ptable.GetAtomicWeight(atomic_number)
        accum_pos += np.array([x, y, z]) * atomic_weight
        accum_mass += atomic_weight

    center_of_mass = np.array(accum_pos / accum_mass, dtype=np.float32)

    pos = np.array(pos, dtype=np.float32)

    BOND_TYPES = {t: i for i, t in enumerate(rdchem.BondType.names.values())}
    bond_type_map = {
        1: BOND_TYPES[rdchem.BondType.SINGLE],
        2: BOND_TYPES[rdchem.BondType.DOUBLE],
        3: BOND_TYPES[rdchem.BondType.TRIPLE],
        4: BOND_TYPES[rdchem.BondType.AROMATIC],
    }
    row, col, edge_type = [], [], []
    for bond_line in sdf[4 + num_atoms:4 + num_atoms + num_bonds]:
        start, end = int(bond_line[0:3]) - 1, int(bond_line[3:6]) - 1
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond_type_map[int(bond_line[6:9])]]

    edge_index = np.array([row, col], dtype=np.longlong)
    edge_type = np.array(edge_type, dtype=np.longlong)

    perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    neighbor_dict = {}
    # used in rotation angle prediction
    for i, atom in enumerate(mol.GetAtoms()):
        neighbor_dict[i] = [n.GetIdx() for n in atom.GetNeighbors()]

    data = {
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'neighbors': neighbor_dict
    }
    return data

def process_item(item, args):
    try:
        pdb_block, sdf_block = load_item(item, args.source)
        protein = PDBProtein(pdb_block)
        seq = ''.join(protein.to_dict_residue()['seq'])
        # ligand = parse_sdf_block(sdf_block)
        ligand = parse_sdf_file(os.path.join(args.source, item[1]))

        r10_idx, r10_residues = protein.query_residues_ligand(ligand, args.radius, selected_residue=None,
                                                              return_mask=False)
        assert len(r10_idx) == len(r10_residues)

        pdb_block_pocket = protein.residues_to_pdb_block(r10_residues)

        full_seq_idx, _ = protein.query_residues_ligand(ligand, radius=3.5, selected_residue=r10_residues,
                                                        return_mask=False)

        ligand_fn = item[1]
        pocket_fn = ligand_fn[:-4] + '_pocket%d.pdb' % args.radius
        ligand_dest = os.path.join(args.dest, ligand_fn)
        pocket_dest = os.path.join(args.dest, pocket_fn)
        os.makedirs(os.path.dirname(ligand_dest), exist_ok=True)

        shutil.copyfile(
            src=os.path.join(args.source, ligand_fn),
            dst=os.path.join(args.dest, ligand_fn)
        )
        with open(pocket_dest, 'w') as f:
            f.write(pdb_block_pocket)

        return pocket_fn, ligand_fn, item[0], item[2], seq, full_seq_idx, r10_idx  # item[0]: original protein filename; item[2]: rmsd.

    except Exception:
        print('Exception occurred.', item)
        return None


def process_item_for_lmdb(item, args):
    """
    Returns data dict with keys: 'protein_name', 'pocket_atoms', 'smi', 'lig_coord_real', etc.
    """
    try:
        pdb_block, sdf_block = load_item(item, args.source)
        protein = PDBProtein(pdb_block)
        ligand = parse_sdf_file(os.path.join(args.source, item[1]))

        # Get pocket residues within specified radius
        r10_idx, r10_residues = protein.query_residues_ligand(ligand, args.radius, selected_residue=None,
                                                              return_mask=False)

        # Extract pocket atoms and coordinates
        pocket_atoms = []
        pocket_coordinates = []
        for residue in r10_residues:
            for atom_idx in residue['atoms']:
                pocket_atoms.append(protein.element[atom_idx])  # atomic number
                pocket_coordinates.append(protein.pos[atom_idx])

        # Keep pocket_atoms as list (same as atoms), only convert coordinates to numpy array
        pocket_coordinates = np.array(pocket_coordinates, dtype=np.float32)

        # Get SMILES from ligand file
        mol = Chem.MolFromMolFile(os.path.join(args.source, item[1]))
        smi = Chem.MolToSmiles(mol) if mol else ""

        # Prepare data dict in format expected by CrossDockData
        data_dict = {
            'protein_name': item[0],  # original protein filename
            'pocket_atoms': pocket_atoms,
            'pocket_coordinates': pocket_coordinates,
            'smi': smi,
            'atoms': ligand['element'],  # ligand atoms
            'lig_coord_real': ligand['pos'],  # RDKit pose (current coordinates)
            'coordinates': [ligand['pos']],  # real pose coordinates (as list for multiple conformations)
        }

        return data_dict

    except Exception as e:
        print(f'Exception occurred processing {item}: {e}')
        return None


def create_lmdb_dataset(args):
    """
    Create LMDB dataset from processed crossdocked data for use with CrossDockData class.
    """
    # Load index
    with open(os.path.join(args.source, 'index.pkl'), 'rb') as f:
        index = pickle.load(f)

    # Create LMDB environment
    lmdb_path = args.dest
    env = lmdb.open(lmdb_path, map_size=1024**4)  # 1TB max size

    # Process items and store in LMDB
    pool = mp.Pool(args.num_workers)
    processed_count = 0

    with env.begin(write=True) as txn:
        for i, data_dict in enumerate(tqdm(
            pool.imap_unordered(partial(process_item_for_lmdb, args=args), index),
            total=len(index),
            desc="Creating LMDB dataset"
        )):
            if data_dict is not None:
                # Store in LMDB with index as key
                key = f'{processed_count}'.encode()
                value = pickle.dumps(data_dict)
                txn.put(key, value)
                processed_count += 1

    pool.close()
    env.close()

    print(f'Created LMDB dataset with {processed_count} entries at {lmdb_path}')
    return processed_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/crossdocked_v1.1_rmsd1.0')
    parser.add_argument('--dest', type=str, required=True, default='data/crossdocked')
    parser.add_argument('--radius', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--create_lmdb', action='store_true',
                        help='Create LMDB dataset for CrossDockData class instead of file-based processing')
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    if args.create_lmdb:
        # Create LMDB dataset for CrossDockData class
        processed_count = create_lmdb_dataset(args)
        print(f'Created LMDB dataset with {processed_count} protein-ligand pairs.')
    else:
        # Original file-based processing
        with open(os.path.join(args.source, 'index.pkl'), 'rb') as f:
            index = pickle.load(f)

        pool = mp.Pool(args.num_workers)
        for item_pocket in tqdm(pool.imap_unordered(partial(process_item, args=args), index), total=len(index)):
            pass
        pool.close()

        print('Done. %d protein-ligand pairs in total.' % len(index))