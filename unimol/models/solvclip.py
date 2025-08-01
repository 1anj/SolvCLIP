# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Optional, Tuple, Union
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params, TransformerEncoderLayer
from .transformer_encoder_with_pair import TransformerEncoderWithPair
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unimol import UniMolModel, base_architecture

logger = logging.getLogger(__name__)
DEFAULT_ENCODER_LAYERS = 15
DEFAULT_EMBED_DIM = 512
DEFAULT_FFN_EMBED_DIM = 2048
DEFAULT_ATTENTION_HEADS = 64
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_DROPOUT = 0.1
DEFAULT_GAUSSIAN_K = 128

@register_model("solvclip")
class SolvclipModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Pretrained model paths
        parser.add_argument(
            "--lig-pretrained",
            type=str,
            default="",
            help="Path to pretrained ligand model",
        )
        parser.add_argument(
            "--pocket-pretrained",
            type=str,
            default="",
            help="Path to pretrained protein model",
        )

        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
            help="Model operation mode",
        )
        parser.add_argument(
            "--x-norm-loss",
            type=float,
            metavar="D",
            help="Weight for x norm loss",
        )
        parser.add_argument(
            "--delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="Weight for delta encoder pair representation norm loss",
        )
        parser.add_argument(
            "--recycling",
            type=int,
            default=3,
            help="Number of recycling iterations for decoder",
        )

    def __init__(self, args, dictionary, lig_dictionary: Optional[Any] = None):
        super().__init__()
        solvclip_architecture(args)

        self.args = args
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
        )

        K = DEFAULT_GAUSSIAN_K
        n_edge_type = len(dictionary) * len(dictionary)

        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)


        # Ligand token embeddings
        self.lig_embed_tokens = nn.Embedding(
            len(lig_dictionary), args.encoder_embed_dim, self.padding_idx
        )

        # Ligand transformer encoder
        self.lig_encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
        )

        # Ligand Gaussian features
        self.lig_gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        lig_n_edge_type = len(lig_dictionary) * len(lig_dictionary)
        self.lig_gbf = GaussianLayer(K, lig_n_edge_type)

        self.classification_heads = nn.ModuleDict()


        self.concat_decoder = TransformerEncoderWithPair(
                encoder_layers=4,
                embed_dim=args.encoder_embed_dim,
                ffn_embed_dim=args.encoder_ffn_embed_dim,
                attention_heads=args.encoder_attention_heads,
                emb_dropout=0.1,
                dropout=0.1,
                attention_dropout=0.1,
                activation_dropout=0.0,
                activation_fn="gelu",
            )

        self.lm_head = MaskLMHead(
                embed_dim=args.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=None,
            )
        self.mask_token_embedding = nn.Embedding(1, args.encoder_embed_dim)
        self.reconstruct_mask_feat_head = NonLinearHead(
            args.encoder_embed_dim, args.encoder_embed_dim, "relu"
        )
        input_dim = args.encoder_embed_dim * 2 + args.encoder_attention_heads
        self.cross_distance_project = NonLinearHead(
            input_dim, 1, "relu"
        )
        self.holo_distance_project = DistanceHead(
            args.encoder_embed_dim + args.encoder_attention_heads, "relu"
        )

        self.apply(init_bert_params)

        self._load_pretrained_models(args)


    def _load_pretrained_models(self, args):
        """Load pretrained models if specified."""
        self.load_pretrained_model(args.lig_pretrained)

        self.load_pocket_pretrained_model(args.pocket_pretrained)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary, task.lig_dictionary)



    def decoder_forward(self, encoder_rep, encoder_pair_rep, padding_mask, lig_encoder_rep, lig_graph_attn_bias, lig_padding_mask, lig_encoder_pair_rep):
        pocket_encoder_rep = encoder_rep
        pocket_encoder_pair_rep = encoder_pair_rep
        pocket_padding_mask = padding_mask

        mol_encoder_rep = lig_encoder_rep
        mol_graph_attn_bias = lig_graph_attn_bias
        mol_padding_mask = lig_padding_mask
        mol_encoder_pair_rep = lig_encoder_pair_rep

        mol_sz = lig_encoder_rep.size(1)
        pocket_sz = pocket_encoder_rep.size(1)

        concat_rep = torch.cat(
            [mol_encoder_rep, pocket_encoder_rep], dim=-2
        )  # [batch, mol_sz+pocket_sz, hidden_dim]
        concat_mask = torch.cat(
            [mol_padding_mask, pocket_padding_mask], dim=-1
        )  # [batch, mol_sz+pocket_sz]
        attn_bs = mol_graph_attn_bias.size(0)

        concat_attn_bias = torch.zeros(
            attn_bs, mol_sz + pocket_sz, mol_sz + pocket_sz
        ).type_as(
            concat_rep
        )  # [batch, mol_sz+pocket_sz, mol_sz+pocket_sz]
        concat_attn_bias[:, :mol_sz, :mol_sz] = (
            mol_encoder_pair_rep.permute(0, 3, 1, 2)
            .reshape(-1, mol_sz, mol_sz)
            .contiguous()
        )
        concat_attn_bias[:, -pocket_sz:, -pocket_sz:] = (
            pocket_encoder_pair_rep.permute(0, 3, 1, 2)
            .reshape(-1, pocket_sz, pocket_sz)
            .contiguous()
        )

        decoder_rep = concat_rep
        decoder_pair_rep = concat_attn_bias
        for i in range(self.args.recycling):
            decoder_outputs = self.concat_decoder(
                decoder_rep, padding_mask=concat_mask, attn_mask=decoder_pair_rep
            )
            decoder_rep = decoder_outputs[0]
            decoder_pair_rep = decoder_outputs[1]
            if i != (self.args.recycling - 1):
                decoder_pair_rep = decoder_pair_rep.permute(0, 3, 1, 2).reshape(
                    -1, mol_sz + pocket_sz, mol_sz + pocket_sz
                )

        mol_decoder = decoder_rep[:, :mol_sz]
        pocket_decoder = decoder_rep[:, mol_sz:]

        return mol_decoder, pocket_decoder, decoder_pair_rep

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_distance: torch.Tensor,
        src_coord: torch.Tensor,
        src_edge_type: torch.Tensor,
        masked_tokens: Optional[torch.Tensor] = None,
        classification_head_name: Optional[str] = None,
        lig_feat_input: Optional[torch.Tensor] = None,
        lig_len: Optional[torch.Tensor] = None,
        pocket_len: Optional[torch.Tensor] = None,
        lig_tokens: Optional[torch.Tensor] = None,
        lig_distance: Optional[torch.Tensor] = None,
        lig_edge_type: Optional[torch.Tensor] = None,
        ori_lig_distance: Optional[torch.Tensor] = None,
        solvent_lig_distance: Optional[torch.Tensor] = None,
        ori_solvent_lig_distance: Optional[torch.Tensor] = None,
        masking_idx: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:

        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        ligand_components = [
            self.lig_encoder,
            self.lig_embed_tokens,
            self.lig_gbf_proj,
            self.lig_gbf
        ]

        for component in ligand_components:
            if hasattr(self, component.__class__.__name__.lower()) and component.training:
                component.eval()
        
        protein_components = [
            self.embed_tokens,
            self.gbf_proj,
            self.gbf,
            self.encoder
        ]

        for component in protein_components:
            if component.training:
                component.eval()


        def compute_gbf_features(gbf_layer, gbf_proj_layer, dist, et):
            n_node = dist.size(-1)
            gbf_feature = gbf_layer(dist, et)
            gbf_result = gbf_proj_layer(gbf_feature)
            # Use contiguous view for better memory layout
            return gbf_result.permute(0, 3, 1, 2).contiguous().view(-1, n_node, n_node)

        # Pre-compute distance features to avoid redundant calculations
        graph_attn_bias = compute_gbf_features(self.gbf, self.gbf_proj, src_distance, src_edge_type)

        # Compute embeddings and encoder outputs
        x = self.embed_tokens(src_tokens)

        with torch.no_grad():
            (
                encoder_rep,
                encoder_pair_rep,
                delta_encoder_pair_rep,
                x_norm,
                delta_encoder_pair_rep_norm,
            ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)


        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

        encoder_distance = None
        encoder_coord = None

        # Pre-compute ligand distance features to avoid redundant calculations
        lig_graph_attn_bias = compute_gbf_features(self.lig_gbf, self.lig_gbf_proj, lig_distance, lig_edge_type)
        solvent_lig_graph_attn_bias = compute_gbf_features(self.lig_gbf, self.lig_gbf_proj, solvent_lig_distance, lig_edge_type)

        with torch.no_grad():
            lig_x = self.lig_embed_tokens(lig_tokens)
            lig_padding_mask = lig_tokens.eq(self.padding_idx)

            # Compute ligand encoder outputs
            (
                lig_encoder_rep,
                lig_encoder_pair_rep,
                lig_delta_encoder_pair_rep,
                lig_x_norm,
                lig_delta_encoder_pair_rep_norm,
            ) = self.lig_encoder(lig_x, padding_mask=lig_padding_mask, attn_mask=lig_graph_attn_bias)

            # Compute solvent-aware ligand encoder outputs
            (
                solvent_lig_encoder_rep,
                solvent_lig_encoder_pair_rep,
                solvent_lig_delta_encoder_pair_rep,
                solvent_lig_x_norm,
                solvent_lig_delta_encoder_pair_rep_norm,
            ) = self.lig_encoder(lig_x, padding_mask=lig_padding_mask, attn_mask=solvent_lig_graph_attn_bias)

            # Handle feature masking if needed
            if ori_lig_distance is not None:
                lig_graph_attn_bias_org = compute_gbf_features(self.lig_gbf, self.lig_gbf_proj, ori_lig_distance, lig_edge_type)

                (
                    lig_encoder_rep_org,
                    lig_encoder_pair_rep_org,
                    _,
                    _,
                    _,
                ) = self.lig_encoder(lig_x, padding_mask=lig_padding_mask, attn_mask=lig_graph_attn_bias_org)

                if masking_idx is not None:
                    masking_idx = masking_idx.to(torch.bool)
                    lig_encoder_rep_unmask = lig_encoder_rep.clone()
                    mask_token = self.mask_token_embedding(torch.tensor(0, device=lig_encoder_rep.device))
                    lig_encoder_rep[masking_idx] = mask_token
                    lig_feature_reg_target = lig_encoder_rep_org[masking_idx]

                    if ori_solvent_lig_distance is not None:
                        solvent_lig_graph_attn_bias_org = compute_gbf_features(self.lig_gbf, self.lig_gbf_proj, ori_solvent_lig_distance, lig_edge_type)

                        (
                            solvent_lig_encoder_rep_org,
                            solvent_lig_encoder_pair_rep_org,
                            _,
                            _,
                            _,
                        ) = self.lig_encoder(lig_x, padding_mask=lig_padding_mask, attn_mask=solvent_lig_graph_attn_bias_org)

                        solvent_lig_encoder_rep_unmask = solvent_lig_encoder_rep.clone()
                        solvent_lig_encoder_rep[masking_idx] = mask_token
                        solvent_lig_feature_reg_target = solvent_lig_encoder_rep_org[masking_idx]


        all_padding_mask = torch.cat([padding_mask, lig_padding_mask], dim=1)
        # NOTE cls and sep for the ligand

        all_feat_x = torch.cat([encoder_rep, lig_encoder_rep], dim=1)
        
        if self.args.mask_feature:
            mask_mol_decoder, _, _ = self.decoder_forward(encoder_rep, encoder_pair_rep, padding_mask, lig_encoder_rep, lig_graph_attn_bias, lig_padding_mask, lig_encoder_pair_rep)
            mask_solvent_lig_decoder, _, _ = self.decoder_forward(encoder_rep, encoder_pair_rep, padding_mask, solvent_lig_encoder_rep, solvent_lig_graph_attn_bias, lig_padding_mask, solvent_lig_encoder_pair_rep)
            reconstruct_feat = self.reconstruct_mask_feat_head(mask_mol_decoder[masking_idx])
            reconstruct_solvent_lig_feat = self.reconstruct_mask_feat_head(mask_solvent_lig_decoder[masking_idx])

            mol_decoder, pocket_decoder, decoder_pair_rep = self.decoder_forward(encoder_rep, encoder_pair_rep, padding_mask, lig_encoder_rep_unmask, lig_graph_attn_bias, lig_padding_mask, lig_encoder_pair_rep)
            # Add solvent decoder computation for unmasked features
            solvent_mol_decoder, solvent_pocket_decoder, solvent_decoder_pair_rep = self.decoder_forward(encoder_rep, encoder_pair_rep, padding_mask, solvent_lig_encoder_rep_unmask, solvent_lig_graph_attn_bias, lig_padding_mask, solvent_lig_encoder_pair_rep)
        else:
            mol_decoder, pocket_decoder, decoder_pair_rep = self.decoder_forward(encoder_rep, encoder_pair_rep, padding_mask, lig_encoder_rep, lig_graph_attn_bias, lig_padding_mask, lig_encoder_pair_rep)
            # Add solvent decoder computation for regular features
            solvent_mol_decoder, solvent_pocket_decoder, solvent_decoder_pair_rep = self.decoder_forward(encoder_rep, encoder_pair_rep, padding_mask, solvent_lig_encoder_rep, solvent_lig_graph_attn_bias, lig_padding_mask, solvent_lig_encoder_pair_rep)

        mol_sz = mol_decoder.size(1)
        pocket_sz = pocket_decoder.size(1)

        mol_pair_decoder_rep = decoder_pair_rep[:, :mol_sz, :mol_sz, :]
        mol_pocket_pair_decoder_rep = (
            decoder_pair_rep[:, :mol_sz, mol_sz:, :]
            + decoder_pair_rep[:, mol_sz:, :mol_sz, :].transpose(1, 2)
        ) * 0.5
        mol_pocket_pair_decoder_rep[mol_pocket_pair_decoder_rep == float("-inf")] = 0

        # Extract solvent-related pair representations
        solvent_mol_pocket_pair_decoder_rep = (
            solvent_decoder_pair_rep[:, :mol_sz, mol_sz:, :]
            + solvent_decoder_pair_rep[:, mol_sz:, :mol_sz, :].transpose(1, 2)
        ) * 0.5
        solvent_mol_pocket_pair_decoder_rep[solvent_mol_pocket_pair_decoder_rep == float("-inf")] = 0

        cross_rep = torch.cat(
            [
                mol_pocket_pair_decoder_rep,
                mol_decoder.unsqueeze(-2).repeat(1, 1, pocket_sz, 1),
                pocket_decoder.unsqueeze(-3).repeat(1, mol_sz, 1, 1),
            ],
            dim=-1,
        )  # [batch, mol_sz, pocket_sz, 4*hidden_size]

        solvent_cross_rep = torch.cat(
            [
                solvent_mol_pocket_pair_decoder_rep,
                solvent_mol_decoder.unsqueeze(-2).repeat(1, 1, pocket_sz, 1),
                solvent_pocket_decoder.unsqueeze(-3).repeat(1, mol_sz, 1, 1),
            ],
            dim=-1,
        )  # [batch, mol_sz, pocket_sz, 4*hidden_size]

        cross_distance_predict = (
        F.elu(self.cross_distance_project(cross_rep).squeeze(-1)) + 1.0
        )  # batch, mol_sz, pocket_sz
        solvent_cross_distance_predict = (
        F.elu(self.cross_distance_project(solvent_cross_rep).squeeze(-1)) + 1.0
        )

        dis_cls_logits = cross_distance_predict # regression target
        # For regression mode, solvent distance logits are the same as regular
        solvent_dis_cls_logits = solvent_cross_distance_predict
        
        if masked_tokens is not None:
            logits = self.lm_head(encoder_rep, masked_tokens)
        else:
            logits = None

        if self.args.mask_feature:
            return all_feat_x, all_padding_mask, dis_cls_logits, solvent_dis_cls_logits, x_norm, delta_encoder_pair_rep_norm, (reconstruct_feat, lig_feature_reg_target), (reconstruct_solvent_lig_feat, solvent_lig_feature_reg_target), logits

        return all_feat_x, all_padding_mask, dis_cls_logits, solvent_dis_cls_logits, x_norm, delta_encoder_pair_rep_norm, logits


    def load_pretrained_model(self, lig_pretrained: str):
        """
        Load pretrained weights for the ligand encoder.

        Args:
            lig_pretrained: Path to pretrained ligand model checkpoint
        """
        if not lig_pretrained:
            return

        logger.info(f"Loading pretrained ligand weights from {lig_pretrained}")

        try:
            state_dict = torch.load(
                lig_pretrained,
                map_location=lambda storage, _: storage
            )

            # Load token embeddings
            token_weight_dict = {
                'weight': state_dict['model']['embed_tokens.weight']
            }
            self.lig_embed_tokens.load_state_dict(token_weight_dict, strict=True)

            # Load GBF projection weights
            gbf_proj_weight_dict = {
                'linear1.weight': state_dict['model']['gbf_proj.linear1.weight'],
                'linear1.bias': state_dict['model']['gbf_proj.linear1.bias'],
                'linear2.weight': state_dict['model']['gbf_proj.linear2.weight'],
                'linear2.bias': state_dict['model']['gbf_proj.linear2.bias']
            }
            self.lig_gbf_proj.load_state_dict(gbf_proj_weight_dict, strict=True)

            # Load GBF weights with dimension checking
            pretrained_mul_weight = state_dict['model']['gbf.mul.weight']
            pretrained_bias_weight = state_dict['model']['gbf.bias.weight']

            # Check if dimensions match current model
            current_edge_types = self.lig_gbf.mul.num_embeddings
            pretrained_edge_types = pretrained_mul_weight.size(0)

            if current_edge_types != pretrained_edge_types:
                logger.warning(
                    f"Edge type dimension mismatch: current model expects {current_edge_types}, "
                    f"but pretrained model has {pretrained_edge_types}. "
                    f"Resizing embeddings to match pretrained weights."
                )

                # Resize the embeddings to match pretrained weights
                self.lig_gbf.mul = nn.Embedding(pretrained_edge_types, 1)
                self.lig_gbf.bias = nn.Embedding(pretrained_edge_types, 1)

                # Re-initialize the resized embeddings
                nn.init.constant_(self.lig_gbf.bias.weight, 0)
                nn.init.constant_(self.lig_gbf.mul.weight, 1)

            gbf_weight_dict = {
                'means.weight': state_dict['model']['gbf.means.weight'],
                'stds.weight': state_dict['model']['gbf.stds.weight'],
                'mul.weight': pretrained_mul_weight,
                'bias.weight': pretrained_bias_weight
            }
            self.lig_gbf.load_state_dict(gbf_weight_dict, strict=True)

            # Load encoder weights
            model_dict = {
                k.replace('encoder.', ''): v
                for k, v in state_dict['model'].items()
            }
            missing_keys, not_matched_keys = self.lig_encoder.load_state_dict(
                model_dict, strict=False
            )

            logger.info(f"Loaded ligand model weights")
            logger.info(f"Missing keys: {missing_keys}")
            logger.info(f"Not matched keys: {not_matched_keys}")

            # Freeze ligand encoder parameters
            self._freeze_ligand_components()

        except Exception as e:
            logger.error(f"Failed to load ligand pretrained model: {e}")
            raise
    def _freeze_ligand_components(self):
        """Freeze ligand encoder components."""
        components_to_freeze = [
            self.lig_embed_tokens,
            self.lig_gbf_proj,
            self.lig_gbf,
            self.lig_encoder
        ]

        for component in components_to_freeze:
            if hasattr(self, component.__class__.__name__.lower()):
                self.freeze_params(component)

    def freeze_params(self, model: nn.Module):
        """
        Freeze parameters of a given model.

        Args:
            model: PyTorch module to freeze
        """
        for param in model.parameters():
            param.requires_grad = False


    def load_pocket_pretrained_model(self, poc_pretrained: str):
        """
        Load pretrained weights for the protein pocket encoder.

        Args:
            poc_pretrained: Path to pretrained protein model checkpoint
        """
        if not poc_pretrained:
            return

        logger.info(f"Loading pocket pretrained weights from {poc_pretrained}")

        try:
            poc_state_dict = torch.load(
                poc_pretrained,
                map_location=lambda storage, _: storage
            )
            missing_keys, not_matched_keys = self.load_state_dict(
                poc_state_dict['model'], strict=False
            )

            # Filter out ligand-related missing keys for cleaner logging
            filter_lig_keys = [k for k in missing_keys if not k.startswith('lig_')]

            logger.info(f"Loaded pocket model weights")
            logger.info(f"Missing keys (excluding ligand): {filter_lig_keys}")
            logger.info(f"Not matched keys: {not_matched_keys}")

            # Freeze protein encoder parameters if specified
            if getattr(self.args, 'proc_freeze', False):
                self._freeze_protein_components()

        except Exception as e:
            logger.error(f"Failed to load pocket pretrained model: {e}")
            raise

    def _freeze_protein_components(self):
        """Freeze protein encoder components."""
        components_to_freeze = [
            self.embed_tokens,
            self.gbf_proj,
            self.gbf,
            self.encoder
        ]

        for component in components_to_freeze:
            if hasattr(self, component.__class__.__name__.lower().replace('embedding', 'embed_tokens')):
                self.freeze_params(component)


    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates

class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@torch.jit.script
def gaussian(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Compute Gaussian function values.

    Args:
        x: Input tensor
        mean: Mean values
        std: Standard deviation values

    Returns:
        Gaussian function values
    """
    PI = 3.14159
    a = (2 * PI) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    """
    Gaussian basis function layer for distance encoding.

    This layer encodes distances using Gaussian basis functions, which is
    essential for the model to understand spatial relationships in molecular
    structures.

    Args:
        K: Number of Gaussian basis functions
        edge_types: Number of edge types in the molecular graph
    """

    def __init__(self, K: int = DEFAULT_GAUSSIAN_K, edge_types: int = 1024):
        super().__init__()
        self.K = K

        # Gaussian parameters
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)

        # Edge-specific scaling parameters
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize Gaussian layer parameters."""
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)
    
@register_model_architecture("solvclip", "solvclip")
def solvclip_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)

    base_architecture(args)
