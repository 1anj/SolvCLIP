# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
import torch

from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    TokenizeDataset,
    RightPadDataset2D,
    EpochShuffleDataset,
)

from unimol.data import (
    KeyDataset,
    DistanceDataset,
    EdgeTypeDataset,
    MaskPointsDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    BiolipDataset,
    ExtractCPConformerDataset,
)

from unicore import checkpoint_utils
from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)

DEFAULT_MASK_PROB = 0.15
DEFAULT_LEAVE_UNMASKED_PROB = 0.05
DEFAULT_RANDOM_TOKEN_PROB = 0.05
DEFAULT_NOISE = 1.0
DEFAULT_MAX_ATOMS = 256
DEFAULT_DICT_NAME = "dict_protein.txt"
DEFAULT_LIG_DICT_NAME = "dict_ligand.txt"
DEFAULT_RUN_NAME = "solclip-pretrain"

@register_task("solvclip")
class SolvclipTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="Colon separated path to data directories list, "
                 "will be iterated upon during epochs in round-robin manner",
        )

        # Masking strategy parameters
        parser.add_argument(
            "--mask-prob",
            default=DEFAULT_MASK_PROB,
            type=float,
            help="Probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=DEFAULT_LEAVE_UNMASKED_PROB,
            type=float,
            help="Probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=DEFAULT_RANDOM_TOKEN_PROB,
            type=float,
            help="Probability of replacing a token with a random token",
        )

        # Noise configuration
        parser.add_argument(
            "--noise-type",
            default="uniform",
            choices=["trunc_normal", "uniform", "normal", "none"],
            help="Noise type in coordinate noise",
        )
        parser.add_argument(
            "--noise",
            default=DEFAULT_NOISE,
            type=float,
            help="Coordinate noise for masked atoms",
        )

        # Hydrogen handling options
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="Remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="Remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only polar hydrogen; -1: all hydrogen; 0: remove all hydrogen",
        )

        # Model configuration
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=DEFAULT_MAX_ATOMS,
            help="Selected maximum number of atoms in a molecule",
        )

        # Dictionary files
        parser.add_argument(
            "--dict-name",
            default=DEFAULT_DICT_NAME,
            help="Protein dictionary file",
        )
        parser.add_argument(
            "--ligdict-name",
            default=DEFAULT_LIG_DICT_NAME,
            help="Ligand dictionary file",
        )

        # Experiment configuration
        parser.add_argument(
            "--run-name",
            default=DEFAULT_RUN_NAME,
            type=str,
            help="Wandb run name",
        )
        parser.add_argument(
            "--remove-lba-casf",
            default=0,
            type=int,
            help="Remove the overlap with DAVIS and CASF datasets",
        )
        parser.add_argument(
            "--gce-std-dev",
            default=1.5,
            type=float,
            help="Standard deviation for atom similarity distribution in GCE loss",
        )
        parser.add_argument(
            "--mask-feature",
            type=int,
            default=0,
            help="Enable feature masking (1: enable, 0: disable)",
        )
        parser.add_argument(
            "--mask-ratio",
            type=float,
            default=0.8,
            help="Ratio of features to mask during training",
        )

    def __init__(self, args, dictionary, lig_dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.lig_dictionary = lig_dictionary
        self.seed = args.seed
        self.mask_feature = self.args.mask_feature
        self.mask_ratio = self.args.mask_ratio

        # Add mask tokens to both dictionaries
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        lig_dictionary.add_symbol("[MASK]", is_special=True)


        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif self.args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True

    @classmethod
    def setup_task(cls, args, **kwargs):
        dict_path = os.path.join(args.data, args.dict_name)
        lig_dict_path = os.path.join(args.data, args.ligdict_name)

        dictionary = Dictionary.load(dict_path)
        lig_dictionary = Dictionary.load(lig_dict_path)

        logger.info(f"Loaded protein dictionary with {len(dictionary)} types")
        logger.info(f"Loaded ligand dictionary with {len(lig_dictionary)} types")

        return cls(args, dictionary, lig_dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """

        raw_dataset = self.load_raw_dataset(split)
        net_input, target = self.process_dataset(
            raw_dataset,
            mask_seed=self.args.seed
        )

        dataset = NestedDictionaryDataset({
            "net_input": net_input,
            "target": target
        })
        
        if split in ["train", "train.small"]:
            dataset = EpochShuffleDataset(dataset, len(dataset), self.args.seed)

        self.datasets[split] = dataset

    def process_dataset(self, raw_dataset, mask_seed):
        dataset = raw_dataset
        if hasattr(self.args, 'mode') and self.args.mode == 'train':
            dataset = ExtractCPConformerDataset(
                raw_dataset, "smi", "atoms", "coordinates",
                mask_feat=self.mask_feature,
                mask_ratio=self.args.mask_ratio
            )

        max_atoms = self.args.max_atoms
        dataset = CroppingDataset(
            dataset, self.seed, "atoms", "coordinates", max_atoms
        )

        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        dataset = NormalizeDataset(dataset, "lig_coordinates", normalize_coord=True)
        dataset = NormalizeDataset(dataset, "solvent_lig_coordinates", normalize_coord=True)
        if self.mask_feature:
            dataset = NormalizeDataset(dataset, "ori_lig_coordinates", normalize_coord=True)
            dataset = NormalizeDataset(dataset, "ori_solvent_lig_coordinates", normalize_coord=True)

        token_dataset = KeyDataset(dataset, "atoms")
        token_dataset = TokenizeDataset(
            token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )


        # Extract coordinate
        coord_dataset = KeyDataset(dataset, "coordinates")
        expand_dataset = MaskPointsDataset(
            token_dataset,
            coord_dataset,
            self.dictionary,
            pad_idx=self.dictionary.pad(),
            mask_idx=self.mask_idx,
            noise_type=self.args.noise_type,
            noise=self.args.noise,
            seed=mask_seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob
        )

        def prepend_and_append(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        # Extract basic datasets
        encoder_token_dataset = KeyDataset(expand_dataset, "atoms")
        encoder_coord_dataset = KeyDataset(expand_dataset, "coordinates")

        src_dataset = prepend_and_append(
            encoder_token_dataset, self.dictionary.bos(), self.dictionary.eos()
        )

        # Prepare coordinate dataset
        encoder_coord_dataset = prepend_and_append(encoder_coord_dataset, 0.0, 0.0)
        encoder_distance_dataset = DistanceDataset(encoder_coord_dataset)

        # Create edge type dataset
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))

        net_input = {
            "src_tokens": RightPadDataset(src_dataset, pad_idx=self.dictionary.pad()),
            "src_coord": RightPadDatasetCoord(encoder_coord_dataset, pad_idx=0),
            "src_distance": RightPadDataset2D(encoder_distance_dataset, pad_idx=0),
            "src_edge_type": RightPadDataset2D(edge_type, pad_idx=0),
            "pocket_len": KeyDataset(dataset, "pocket_len"),
            "lig_len": KeyDataset(dataset, "lig_len")
        }

        lig_encoder_token_dataset = KeyDataset(dataset, "lig_atoms")
        lig_encoder_token_dataset = TokenizeDataset(
            lig_encoder_token_dataset, self.lig_dictionary, max_seq_len=self.args.max_seq_len
        )
        lig_src_dataset = prepend_and_append(
            lig_encoder_token_dataset, self.lig_dictionary.bos(), self.lig_dictionary.eos()
        )
        net_input['lig_tokens'] = RightPadDataset(
            lig_src_dataset, pad_idx=self.lig_dictionary.pad()
        )

        # Process ligand coordinates
        lig_encoder_coord_dataset = KeyDataset(dataset, "lig_coordinates")
        lig_encoder_coord_dataset = prepend_and_append(lig_encoder_coord_dataset, 0.0, 0.0)
        net_input['lig_coordinates'] = RightPadDatasetCoord(lig_encoder_coord_dataset, pad_idx=0)

        # Process ligand distances and edge types
        lig_encoder_distance_dataset = DistanceDataset(lig_encoder_coord_dataset)
        lig_edge_type = EdgeTypeDataset(lig_src_dataset, len(self.lig_dictionary))

        net_input['lig_distance'] = RightPadDataset2D(lig_encoder_distance_dataset, pad_idx=0)
        net_input['lig_edge_type'] = RightPadDataset2D(lig_edge_type, pad_idx=0)

        # Add masked feature information if enabled
        if self.mask_feature:
            # Original ligand coordinates
            lig_encoder_coord_dataset_org = KeyDataset(dataset, "ori_lig_coordinates")
            lig_encoder_coord_dataset_org = prepend_and_append(lig_encoder_coord_dataset_org, 0.0, 0.0)
            net_input['ori_lig_coordinates'] = RightPadDatasetCoord(lig_encoder_coord_dataset_org, pad_idx=0)

            # Original ligand distances
            lig_encoder_distance_dataset_org = DistanceDataset(lig_encoder_coord_dataset_org)
            net_input['ori_lig_distance'] = RightPadDataset2D(lig_encoder_distance_dataset_org, pad_idx=0)

            # Feature masking indices
            masking_idx = KeyDataset(dataset, "mask_array")
            masking_idx = prepend_and_append(masking_idx, 0, 0)
            net_input['masking_idx'] = RightPadDataset(masking_idx, pad_idx=0)


        solv_lig_encoder_coord_dataset = KeyDataset(dataset, "solvent_lig_coordinates")
        solv_lig_encoder_coord_dataset = prepend_and_append(solv_lig_encoder_coord_dataset, 0.0, 0.0)
        net_input['solvent_lig_coordinates'] = RightPadDatasetCoord(solv_lig_encoder_coord_dataset, pad_idx=0)
        # Process ligand distances and edge types
        lig_encoder_distance_dataset = DistanceDataset(lig_encoder_coord_dataset)

        net_input['solvent_lig_distance'] = RightPadDataset2D(lig_encoder_distance_dataset, pad_idx=0)

        # Add masked feature information if enabled
        if self.mask_feature:
            lig_encoder_coord_dataset_org = KeyDataset(dataset, "ori_solvent_lig_coordinates")
            lig_encoder_coord_dataset_org = prepend_and_append(lig_encoder_coord_dataset_org, 0.0, 0.0)
            net_input['ori_solvent_lig_coordinates'] = RightPadDatasetCoord(lig_encoder_coord_dataset_org, pad_idx=0)

            # Original ligand distances
            lig_encoder_distance_dataset_org = DistanceDataset(lig_encoder_coord_dataset_org)
            net_input['ori_solvent_lig_distance'] = RightPadDataset2D(lig_encoder_distance_dataset_org, pad_idx=0)

            encoder_target_dataset = KeyDataset(expand_dataset, "targets")
            tgt_dataset = prepend_and_append(
                encoder_target_dataset, self.dictionary.pad(), self.dictionary.pad()
            )

        target = {
            "all_coordinates": RightPadDatasetCoord(KeyDataset(dataset, "all_coordinates"), pad_idx=0),
            "tokens_target": RightPadDataset(tgt_dataset, pad_idx=self.dictionary.pad()),
            "pocket_len": KeyDataset(dataset, "pocket_len"),
            "lig_len": KeyDataset(dataset, "lig_len")
        }

        return net_input, target

    def load_raw_dataset(self, split):
        dataset_path = f'{self.args.data}/{split}.lmdb'
        return BiolipDataset(dataset_path, max_num=DEFAULT_MAX_ATOMS)
        
    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        logger.info(f"Built model with {sum(p.numel() for p in model.parameters())} parameters")
        return model
