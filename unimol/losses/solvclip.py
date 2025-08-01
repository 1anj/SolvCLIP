import math
import os
import torch
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from typing import Dict, Any, List, Optional, Tuple, Union
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from torch import distributions
from torch.distributions import Normal, kl_divergence
import logging
logger = logging.getLogger(__name__)

DEFAULT_DIST_MEAN = 6.312581655060595
DEFAULT_DIST_STD = 3.3899264663911888

def _get_local_rank() -> int:
    try:
        return int(os.environ.get('LOCAL_RANK', '0'))
    except (ValueError, TypeError):
        return 0
    
def create_atom_similarity_distribution(vocab_size: int, std_dev: float = 1.5) -> torch.Tensor:
    pdf_matrix = torch.zeros(vocab_size, vocab_size)
    pdf_matrix[0, 0] = 1.0

    for atom_idx in range(1, vocab_size):
        atom_indices = torch.arange(1, vocab_size, dtype=torch.float)
        normal_dist = distributions.Normal(float(atom_idx), std_dev)

        probabilities = normal_dist.log_prob(atom_indices).exp()
        probabilities[atom_idx - 1] *= 2.0
        probabilities = probabilities / probabilities.sum()
        pdf_matrix[atom_idx, 1:] = probabilities

    return pdf_matrix

@register_loss("solvclip")
class SolvclipLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.seed = task.seed

        self.dist_mean = DEFAULT_DIST_MEAN
        self.dist_std = DEFAULT_DIST_STD

        self.mask_feature = getattr(task.args, 'mask_feature', False)
        self.gce_std_dev = getattr(task.args, 'gce_std_dev', 1.5)

        # Initialize atom similarity distribution for GCE loss
        vocab_size = len(task.dictionary)
        self.atom_similarity_dist = create_atom_similarity_distribution(
            vocab_size, self.gce_std_dev
        )
        logger.info(f"Initialized GCE loss with vocabulary size {vocab_size} and std_dev {self.gce_std_dev}")

    def compute_gce_masked_token_loss(
        self,
        logits_encoder: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        if not hasattr(self, 'atom_similarity_dist_device') or self.atom_similarity_dist_device != logits_encoder.device:
            self.atom_similarity_dist = self.atom_similarity_dist.to(logits_encoder.device)
            self.atom_similarity_dist_device = logits_encoder.device

        vocab_size = logits_encoder.size(-1)

        # Create non-padding mask efficiently
        non_padding_mask = target.ne(self.padding_idx)

        if non_padding_mask.sum() == 0:
            return torch.tensor(0.0, device=logits_encoder.device, requires_grad=True)

        # Use advanced indexing for better performance
        valid_targets = target[non_padding_mask]
        valid_logits = logits_encoder[non_padding_mask]

        # Create soft labels more efficiently using vectorized operations
        soft_labels = torch.zeros_like(valid_logits)

        # Use broadcasting to avoid loops
        for atom_idx in torch.unique(valid_targets):
            if atom_idx > 0:  # Skip padding tokens
                atom_mask = (valid_targets == atom_idx)
                soft_labels[atom_mask] = self.atom_similarity_dist[atom_idx].to(soft_labels.dtype)

        # Compute log probabilities efficiently
        log_probs = F.log_softmax(valid_logits, dim=-1)

        # Compute GCE loss with better numerical stability
        gce_loss = -torch.sum(soft_labels * log_probs) / valid_targets.numel()

        return gce_loss


    def forward(self, model, sample, reduce=True):
        input_key = "net_input"
        target_key = "target"
        if "tokens_target" in sample[target_key]:
            masked_tokens = sample[target_key]["tokens_target"].ne(self.padding_idx)
            sample_size = masked_tokens.long().sum()
            sample[input_key]['masked_tokens'] = masked_tokens

        model_outputs = self.get_complex_model_outputs(model, sample[input_key])
        loss, logging_output = self.compute_complex_loss(model_outputs, sample)

        return loss, sample_size, logging_output



    def compute_complex_loss(
        self,
        model_outputs,
        sample
    ):
        """Compute loss for complex pretraining mode."""
        all_feat_x = model_outputs['all_feat_x']
        dis_cls_logits = model_outputs['dis_cls_logits']
        x_norm = model_outputs['x_norm']
        delta_encoder_pair_rep_norm = model_outputs['delta_encoder_pair_rep_norm']
        logits = model_outputs["logits"]
        target = sample["target"]["tokens_target"]
        if sample["net_input"]["masked_tokens"] is not None:
            target = target[sample["net_input"]["masked_tokens"]]


        masked_token_loss = self.compute_gce_masked_token_loss(logits, target)
        # Extract target information
        pocket_len = sample['target']['pocket_len']
        lig_len = sample['target']['lig_len']
        batch_size = sample['target']['all_coordinates'].shape[0]

        # Initialize logging output
        logging_output = {
            "sample_size": 1,
            "bsz": batch_size,
            "seq_len": all_feat_x.shape[1],
            "pocket_max_len": pocket_len.max().item(),
            "lig_max_len": lig_len.max().item()
        }

        # Compute distance loss
        distance_loss = self.compute_distance_loss(
            sample, dis_cls_logits, pocket_len, lig_len, batch_size
        )

        # Log appropriate distance loss type
        logging_output["idmp_loss"] = distance_loss
        loss = distance_loss

        # Add feature masking loss if enabled
        if self.mask_feature and 'mask_pred_target_feat' in model_outputs:
            

            mask_feat_loss = self.compute_mask_feature_loss(
                model_outputs['mask_pred_target_feat']
            )

            if 'mask_sovent_pred_target_feat' in model_outputs:
                mask_solvent_feat_loss = self.compute_mask_feature_loss(
                    model_outputs['mask_sovent_pred_target_feat']
                )
                mask_loss = mask_feat_loss + mask_solvent_feat_loss
            else:
                mask_loss = mask_feat_loss
            logging_output["mmr_loss"] = mask_loss + masked_token_loss
            loss += mask_loss

        # Add contrastive learning loss
        contrastive_loss = self.compute_contrastive_loss(
            model_outputs, pocket_len, lig_len, batch_size
        )
        if contrastive_loss > 0:
            loss += contrastive_loss
            logging_output["cl_loss"] = contrastive_loss

        # Add regularization losses
        loss, logging_output = self.add_regularization_losses(
            loss, logging_output, x_norm, delta_encoder_pair_rep_norm
        )

        logging_output["loss"] = loss.data
        return loss, logging_output
    
    def add_regularization_losses(
        self,
        loss: torch.Tensor,
        logging_output: Dict[str, Any],
        x_norm: Optional[torch.Tensor],
        delta_encoder_pair_rep_norm: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Add regularization losses to the main loss."""
        # Add x norm loss
        if (hasattr(self.args, 'x_norm_loss') and
            self.args.x_norm_loss > 0 and
            x_norm is not None):
            loss = loss + self.args.x_norm_loss * x_norm
            logging_output["x_norm_loss"] = x_norm.data

        # Add delta pair representation norm loss
        if (hasattr(self.args, 'delta_pair_repr_norm_loss') and
            self.args.delta_pair_repr_norm_loss > 0 and
            delta_encoder_pair_rep_norm is not None):
            loss = loss + self.args.delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm
            logging_output["delta_pair_repr_norm_loss"] = delta_encoder_pair_rep_norm.data

        return loss, logging_output
    def compute_mask_feature_loss(self, mask_pred_target_feat: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute feature masking loss."""
        pred_feat, target_feat = mask_pred_target_feat
        return F.mse_loss(
            pred_feat.float(),
            target_feat.float(),
            reduction="mean"
        )
    
    def get_complex_model_outputs(self, model, net_input):
        """Get model outputs for complex pretraining mode."""
        if self.mask_feature:
            (all_feat_x, all_padding_mask, dis_cls_logits, solvent_dis_cls_logits,
             x_norm, delta_encoder_pair_rep_norm, mask_pred_target_feat, mask_sovent_pred_target_feat, logits) = model(**net_input)
            return {
                'all_feat_x': all_feat_x,
                'all_padding_mask': all_padding_mask,
                'dis_cls_logits': dis_cls_logits,
                'solvent_dis_cls_logits': solvent_dis_cls_logits,
                'x_norm': x_norm,
                'delta_encoder_pair_rep_norm': delta_encoder_pair_rep_norm,
                'mask_pred_target_feat': mask_pred_target_feat,
                'mask_sovent_pred_target_feat': mask_sovent_pred_target_feat,
                "logits": logits,
                'encoder_coord': None,
                'encoder_distance': None,
            }
        else:
            (all_feat_x, all_padding_mask, dis_cls_logits, solvent_dis_cls_logits,
             x_norm, delta_encoder_pair_rep_norm, logits), = model(**net_input)
            return {
                'all_feat_x': all_feat_x,
                'all_padding_mask': all_padding_mask,
                'dis_cls_logits': dis_cls_logits,
                'solvent_dis_cls_logits': solvent_dis_cls_logits,
                'x_norm': x_norm,
                'delta_encoder_pair_rep_norm': delta_encoder_pair_rep_norm,
                "logits": logits,
                'encoder_coord': None,
                'encoder_distance': None,
            }


    def compute_distance_loss(
        self,
        sample: Dict[str, Any],
        dis_cls_logits: torch.Tensor,
        pocket_len: torch.Tensor,
        lig_len: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """Optimized distance-based loss computation for protein-ligand interactions."""
        # Pre-allocate tensors for better memory efficiency
        distance_losses = []

        # Extract all positions once to avoid repeated indexing
        all_positions = sample['target']['all_coordinates']

        for idx in range(batch_size):
            proc_num = pocket_len[idx].item()
            lig_num = lig_len[idx].item()

            all_pos = all_positions[idx]
            proc_pos = all_pos[:proc_num]
            lig_pos = all_pos[proc_num:proc_num + lig_num]

            lig_proc_distance = torch.cdist(lig_pos, proc_pos, p=2)

            lig_proc_distance_pred = dis_cls_logits[idx, 1:1+lig_num, 1:1+proc_num]

            distance_loss = self.compute_regression_distance_loss(
                lig_proc_distance, lig_proc_distance_pred
            )

            if torch.isnan(distance_loss):
                distance_loss = torch.tensor(0.0, device=distance_loss.device, dtype=distance_loss.dtype)

            distance_losses.append(distance_loss)

        if distance_losses:
            return torch.stack(distance_losses).sum() / batch_size
        else:
            return torch.tensor(0.0, device=dis_cls_logits.device)
        

    def compute_regression_distance_loss(
        self,
        lig_proc_distance: torch.Tensor,
        lig_proc_distance_pred: torch.Tensor
    ) -> torch.Tensor:
        """Optimized regression-based distance loss computation."""
        distance_mask = lig_proc_distance.ne(0)  # 0 is padding

        if hasattr(self.args, 'dist_threshold') and self.args.dist_threshold > 0:
            distance_mask = distance_mask & (lig_proc_distance < self.args.dist_threshold)

        if not distance_mask.any():
            return torch.tensor(0.0, device=lig_proc_distance.device, dtype=lig_proc_distance.dtype)

        distance_predict = lig_proc_distance_pred[distance_mask]
        distance_target = lig_proc_distance[distance_mask]

        return F.mse_loss(
            distance_predict.float(),
            distance_target.float(),
            reduction="mean"
        )

    def compute_contrastive_loss(
        self,
        model_outputs,
        pocket_len,
        lig_len,
        batch_size
    ):
        """
        Optimized SimCLR contrastive learning loss computation.
        """
        z1 = model_outputs.get('dis_cls_logits')            # [B, *]
        z2 = model_outputs.get('solvent_dis_cls_logits')    # [B, *]

        if z1 is None or z2 is None:
            return torch.tensor(0.0, device=pocket_len.device)

        # Optimize tensor reshaping and normalization
        z1_flat = z1.view(batch_size, -1)
        z2_flat = z2.view(batch_size, -1)

        # Use more efficient normalization
        z1_norm = F.normalize(z1_flat, p=2, dim=1, eps=1e-8)
        z2_norm = F.normalize(z2_flat, p=2, dim=1, eps=1e-8)

        # Optimize feature concatenation
        features = torch.cat([z1_norm, z2_norm], dim=0)     # [2B, D]

        sim_matrix = torch.mm(features, features.T) / 0.1  # [2B, 2B]

        # Create mask more efficiently
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
        sim_matrix.masked_fill_(mask, float('-inf'))

        # Create labels more efficiently
        labels = torch.arange(batch_size, device=features.device)
        pos_labels = torch.cat([labels + batch_size, labels])  # [2B]

        # Apply contrastive weight if specified
        loss = F.cross_entropy(sim_matrix, pos_labels)

        return loss
    
    @staticmethod
    def reduce_metrics(logging_outputs: List[Dict[str, Any]], split: str = "valid") -> None:
        """
        Aggregate logging outputs from data parallel training.

        This method collects and aggregates metrics from multiple workers
        in distributed training, computing averages and logging them for
        monitoring and evaluation purposes.

        Args:
            logging_outputs: List of logging dictionaries from each worker
            split: Dataset split name (train/valid/test)
        """
        if not logging_outputs:
            return

        # Aggregate basic metrics
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)

        if sample_size > 0:
            metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        if bsz > 0:
            metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)

        distance_loss_value = sum(
            log.get("idmp_loss", 0) for log in logging_outputs
        )
        if distance_loss_value > 0 and sample_size > 0:
            metrics.log_scalar(
                "idmp_loss",
                distance_loss_value / sample_size,
                sample_size,
                round=3,
            )

        mask_feat_loss = sum(log.get("mmr_loss", 0) for log in logging_outputs)
        if mask_feat_loss > 0 and sample_size > 0:
            metrics.log_scalar(
                "mmr_loss",
                mask_feat_loss / sample_size,
                sample_size,
                round=3,
            )
        
        contrastive_loss = sum(log.get("cl_loss", 0) for log in logging_outputs)
        if contrastive_loss > 0 and sample_size > 0:
            metrics.log_scalar(
                "cl_loss",
                contrastive_loss / sample_size,
                sample_size,
                round=3,
            )


    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train