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


def _scheduled_entropy_weight(
    *,
    alpha: float,
    epoch: int,
    epochs: int,
    entropy_schedule: str,
    entropy_warmup_start_fraction: float,
    entropy_warmup_end_fraction: float,
) -> float:
    if entropy_schedule == "constant":
        return float(alpha)
    if entropy_schedule not in {"delayed_linear", "linear"}:
        raise ValueError("entropy_schedule must be one of {'constant', 'linear', 'delayed_linear'}.")

    if entropy_schedule == "linear":
        start_fraction = 0.0
    else:
        start_fraction = float(entropy_warmup_start_fraction)
    end_fraction = float(entropy_warmup_end_fraction)
    if end_fraction <= start_fraction:
        return float(alpha) if (epoch + 1) / max(1, epochs) >= end_fraction else 0.0

    progress = (epoch + 1) / max(1, epochs)
    if progress <= start_fraction:
        return 0.0
    if progress >= end_fraction:
        return float(alpha)
    return float(alpha) * (progress - start_fraction) / (end_fraction - start_fraction)


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
    entropy_schedule="constant",
    entropy_warmup_start_fraction=0.5,
    entropy_warmup_end_fraction=1.0,
    directed_inclusion_mask=None,
    hierarchy_strength=0.0,
    hierarchy_margin=0.0,
    hierarchy_min_direction_weight=0.0,
    garbage_hyperedge_index=None,
    ambiguity_weight=None,
    garbage_strength=0.0,
    clean_repel_strength=0.0,
    garbage_margin_strength=0.0,
    garbage_margin=0.1,
    exclude_garbage_from_relation_loss=False,
    garbage_eps=1e-8,
    return_loss_history=False,
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
    if ambiguity_weight is not None:
        ambiguity_weight = torch.as_tensor(ambiguity_weight, dtype=torch.float32, device=device)
        if ambiguity_weight.ndim != 1 or ambiguity_weight.shape[0] != init_partition.shape[0]:
            raise ValueError(
                "ambiguity_weight must be a 1D tensor/array with one value per gene."
            )
    if relation_masks is not None:
        relation_masks = {
            name: _move_mask_to_device(mask, device)
            for name, mask in relation_masks.items()
        }
    entropy_gene_mask = _move_mask_to_device(entropy_gene_mask, device)

    optimizer = torch.optim.Adam([partition, hyperedge_emb], lr=lr)
    losses = []
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        partition_used = F.softmax(partition, dim=1)      # (N, K)
        hyperedge_emb_norm = F.normalize(hyperedge_emb, p=2, dim=1)   # (K, D)
        gene_emb = partition_used @ hyperedge_emb_norm    # (N, D)
        relation_partition = partition_used
        relation_hyperedge_emb = hyperedge_emb_norm
        if (
            exclude_garbage_from_relation_loss
            and garbage_hyperedge_index is not None
        ):
            garbage_idx = int(garbage_hyperedge_index)
            if garbage_idx < 0 or garbage_idx >= partition_used.shape[1]:
                raise ValueError(
                    f"garbage_hyperedge_index={garbage_idx} is out of bounds for "
                    f"{partition_used.shape[1]} hyperedges."
                )
            normal_indices = [
                idx for idx in range(partition_used.shape[1]) if idx != garbage_idx
            ]
            relation_partition = partition_used[:, normal_indices]
            relation_hyperedge_emb = hyperedge_emb_norm[normal_indices]

        gene_emb_relation = relation_partition @ relation_hyperedge_emb
        sim_matrix = gene_emb_relation @ gene_emb_relation.T

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
                partition_soft=relation_partition,
                neg_mask=neg_mask,
                directed_inclusion_mask=directed_inclusion_mask,
                margin=hierarchy_margin,
                min_direction_weight=hierarchy_min_direction_weight,
            )
            loss_items["loss_hierarchy_directed"] = loss_hierarchy
        else:
            loss_hierarchy = torch.zeros((), device=partition_used.device, dtype=partition_used.dtype)

        garbage_enabled = (
            garbage_hyperedge_index is not None
            and ambiguity_weight is not None
            and garbage_strength > 0.0
        )
        if garbage_enabled:
            garbage_idx = int(garbage_hyperedge_index)
            if garbage_idx < 0 or garbage_idx >= partition_used.shape[1]:
                raise ValueError(
                    f"garbage_hyperedge_index={garbage_idx} is out of bounds for "
                    f"{partition_used.shape[1]} hyperedges."
                )
            p_garbage = partition_used[:, garbage_idx].clamp(garbage_eps, 1.0 - garbage_eps)
            normal_indices = [
                idx for idx in range(partition_used.shape[1]) if idx != garbage_idx
            ]
            max_normal_p = partition_used[:, normal_indices].max(dim=1).values
            weight = ambiguity_weight.to(device=partition_used.device, dtype=partition_used.dtype)
            loss_garbage_push = -(weight * torch.log(p_garbage + garbage_eps)).mean()
            loss_items["loss_garbage_push"] = loss_garbage_push
            if clean_repel_strength > 0.0:
                loss_clean_repel = -(
                    (1.0 - weight) * torch.log(1.0 - p_garbage + garbage_eps)
                ).mean()
            else:
                loss_clean_repel = torch.zeros(
                    (),
                    device=partition_used.device,
                    dtype=partition_used.dtype,
                )
            loss_items["loss_clean_repel"] = loss_clean_repel
            if garbage_margin_strength > 0.0:
                loss_garbage_margin = (
                    weight * F.relu(float(garbage_margin) + max_normal_p - p_garbage)
                ).mean()
            else:
                loss_garbage_margin = torch.zeros(
                    (),
                    device=partition_used.device,
                    dtype=partition_used.dtype,
                )
            loss_items["loss_garbage_margin"] = loss_garbage_margin
        else:
            loss_garbage_push = torch.zeros(
                (),
                device=partition_used.device,
                dtype=partition_used.dtype,
            )
            loss_clean_repel = torch.zeros(
                (),
                device=partition_used.device,
                dtype=partition_used.dtype,
            )
            loss_garbage_margin = torch.zeros(
                (),
                device=partition_used.device,
                dtype=partition_used.dtype,
            )

        entropy_weight = _scheduled_entropy_weight(
            alpha=alpha,
            epoch=epoch,
            epochs=epochs,
            entropy_schedule=entropy_schedule,
            entropy_warmup_start_fraction=entropy_warmup_start_fraction,
            entropy_warmup_end_fraction=entropy_warmup_end_fraction,
        )

        loss = (
            loss_main
            + hierarchy_strength * loss_hierarchy
            + entropy_weight * loss_entropy
            + garbage_strength * loss_garbage_push
            + clean_repel_strength * loss_clean_repel
            + garbage_margin_strength * loss_garbage_margin
        )
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        loss_history.append(
            {
                "epoch": int(epoch + 1),
                "loss_total": float(loss.item()),
                "loss_main": float(loss_main.item()),
                "loss_entropy": float(loss_entropy.item()),
                "entropy_weight": float(entropy_weight),
                "loss_hierarchy_directed": float(loss_hierarchy.item()),
                "loss_garbage_push": float(loss_garbage_push.item()),
                "loss_clean_repel": float(loss_clean_repel.item()),
                "loss_garbage_margin": float(loss_garbage_margin.item()),
            }
        )

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
            garbage_log = ""
            if garbage_enabled:
                garbage_log = (
                    f", garbage_push: {loss_garbage_push.item():.9f}, "
                    f"clean_repel: {loss_clean_repel.item():.9f}, "
                    f"garbage_margin: {loss_garbage_margin.item():.9f}"
                )
            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.9f}, "
                f"{_format_supervision_log(loss_items, resolved_targets)}"
                f"{hierarchy_log}{garbage_log}, Entropy: {loss_entropy.item():.4f}"
                f", EntropyWeight: {entropy_weight:.6f}"
            )

    with torch.no_grad():
        partition_final = F.softmax(partition, dim=1)
        hyperedge_emb_norm = F.normalize(hyperedge_emb, p=2, dim=1)
        gene_emb_final = partition_final @ hyperedge_emb_norm

    outputs = (partition_final.detach(), hyperedge_emb.detach(), gene_emb_final.detach(), losses)
    if return_loss_history:
        return (*outputs, loss_history)
    return outputs
