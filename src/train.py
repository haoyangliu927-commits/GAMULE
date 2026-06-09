import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.loss import directed_hyperedge_inclusion_loss, weak_pos_neg_loss


def _move_mask_to_device(mask, device):
    if mask is None:
        return None
    return mask.to(device)


def _format_supervision_log(loss_items, relation_targets):
    log_parts = []
    for name, target in relation_targets.items():
        loss_key = f"loss_{name}"
        if loss_key not in loss_items:
            continue
        if target <= 0:
            log_parts.append(f"{name}(->{target:g}): {loss_items[loss_key].item():.9f}")
        else:
            log_parts.append(f"{name}(>= {target:g}): {loss_items[loss_key].item():.9f}")
    return ", ".join(log_parts)


def train_embedding_weakpos0(
    init_partition,
    init_hyperedge_emb,
    pos_mask=None,
    neg_mask=None,
    epochs=200,
    lr=1e-2,
    device="cuda",
    pos_threshold=0.5,
    alpha=0.1,   # partition 熵正则
    supervision_mode="binary",
    full_pos_mask=None,
    partial_pos_mask=None,
    relation_masks=None,
    relation_targets=None,
    full_pos_threshold=1.0,
    partial_pos_threshold=0.5,
    neg_target=0.0,
    entropy_gene_mask=None,
    directed_inclusion_mask=None,
    hierarchy_strength=0.0,
    hierarchy_margin=0.0,
    hierarchy_min_direction_weight=0.0,
    ):
    if isinstance(init_partition, np.ndarray):
        init_partition = torch.from_numpy(init_partition).float()
    if isinstance(init_hyperedge_emb, np.ndarray):
        init_hyperedge_emb = torch.from_numpy(init_hyperedge_emb).float()

    partition = nn.Parameter(init_partition.clone().detach().to(device))
    hyperedge_emb = nn.Parameter(init_hyperedge_emb.clone().detach().to(device))

    pos_mask = _move_mask_to_device(pos_mask, device)
    neg_mask = _move_mask_to_device(neg_mask, device)
    full_pos_mask = _move_mask_to_device(full_pos_mask, device)
    partial_pos_mask = _move_mask_to_device(partial_pos_mask, device)
    directed_inclusion_mask = _move_mask_to_device(directed_inclusion_mask, device)
    if relation_masks is not None:
        relation_masks = {
            name: _move_mask_to_device(mask, device)
            for name, mask in relation_masks.items()
        }
    entropy_gene_mask = _move_mask_to_device(entropy_gene_mask, device)

    optimizer = torch.optim.Adam([partition, hyperedge_emb], lr=lr)
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        partition_used = F.softmax(partition, dim=1)      # (N, K)
        hyperedge_emb_norm = F.normalize(hyperedge_emb, p=2, dim=1)   # (K, D)
        gene_emb = partition_used @ hyperedge_emb_norm    # (N, D)
        sim_matrix = gene_emb @ gene_emb.T

        # 主损失：弱正 + 负到0
        loss_main, loss_items = weak_pos_neg_loss(
            sim_matrix=sim_matrix,
            pos_mask=pos_mask,
            neg_mask=neg_mask,
            pos_threshold=pos_threshold,
            supervision_mode=supervision_mode,
            full_pos_mask=full_pos_mask,
            partial_pos_mask=partial_pos_mask,
            relation_masks=relation_masks,
            relation_targets=relation_targets,
            full_pos_threshold=full_pos_threshold,
            partial_pos_threshold=partial_pos_threshold,
            neg_target=neg_target,
        )
        
        entropy = -torch.sum(partition_used * torch.log(partition_used + 1e-8), dim=1)
        if entropy_gene_mask is None:
            loss_entropy = entropy.mean()
        else:
            entropy_gene_mask = entropy_gene_mask.bool()
            if entropy_gene_mask.any():
                loss_entropy = entropy[entropy_gene_mask].mean()
            else:
                loss_entropy = entropy.mean()

        hierarchy_enabled = directed_inclusion_mask is not None and hierarchy_strength > 0.0
        if hierarchy_enabled:
            loss_hierarchy = directed_hyperedge_inclusion_loss(
                partition_soft=partition_used,
                neg_mask=neg_mask,
                directed_inclusion_mask=directed_inclusion_mask,
                margin=hierarchy_margin,
                min_direction_weight=hierarchy_min_direction_weight,
            )
            loss_items["loss_hierarchy_directed"] = loss_hierarchy
        else:
            loss_hierarchy = torch.zeros((), device=partition_used.device, dtype=partition_used.dtype)

        loss = loss_main + hierarchy_strength * loss_hierarchy + alpha * loss_entropy
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0 or epoch == 0:
            resolved_targets = relation_targets
            if resolved_targets is None:
                if supervision_mode == "ternary":
                    resolved_targets = {
                        "full_pos": full_pos_threshold,
                        "partial_pos": partial_pos_threshold,
                        "neg": neg_target,
                    }
                else:
                    resolved_targets = {
                        "pos": pos_threshold,
                        "neg": neg_target,
                    }
            hierarchy_log = ""
            if hierarchy_enabled:
                hierarchy_log = f", hierarchy_directed: {loss_hierarchy.item():.9f}"
            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.9f}, "
                f"{_format_supervision_log(loss_items, resolved_targets)}"
                f"{hierarchy_log}, Entropy: {loss_entropy.item():.4f}"
            )

    with torch.no_grad():
        partition_final = F.softmax(partition, dim=1)
        hyperedge_emb_norm = F.normalize(hyperedge_emb, p=2, dim=1)
        gene_emb_final = partition_final @ hyperedge_emb_norm

    return partition_final.detach(), hyperedge_emb.detach(), gene_emb_final.detach(), losses
