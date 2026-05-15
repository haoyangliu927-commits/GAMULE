import torch
import torch.nn.functional as F


def resolve_relation_supervision(
    *,
    supervision_mode: str = "binary",
    pos_mask: torch.Tensor | None = None,
    neg_mask: torch.Tensor | None = None,
    full_pos_mask: torch.Tensor | None = None,
    partial_pos_mask: torch.Tensor | None = None,
    relation_masks: dict[str, torch.Tensor] | None = None,
    relation_targets: dict[str, float] | None = None,
    pos_threshold: float = 0.5,
    full_pos_threshold: float = 1.0,
    partial_pos_threshold: float = 0.5,
    neg_target: float = 0.0,
) -> tuple[dict[str, torch.Tensor], dict[str, float], list[str]]:
    """
    统一整理监督关系定义，兼容：
    1. 旧接口：binary -> pos_mask / neg_mask
    2. 新接口：ternary -> full_pos_mask / partial_pos_mask / neg_mask
    3. 自定义接口：custom -> relation_masks / relation_targets

    返回:
    - relation_masks: {name: bool mask}
    - relation_targets: {name: target value}
    - direct_separation_names: 默认用于 partition 分离惩罚的关系名
    """
    if relation_masks is not None:
        if relation_targets is None:
            raise ValueError("When relation_masks is provided, relation_targets must also be provided.")
        if set(relation_masks.keys()) != set(relation_targets.keys()):
            raise ValueError("relation_masks and relation_targets must have the same keys.")
        resolved_masks = {name: mask.bool() for name, mask in relation_masks.items()}
        resolved_targets = {name: float(target) for name, target in relation_targets.items()}
        direct_separation_names = [name for name, target in resolved_targets.items() if target <= 0.0]
    else:
        mode = supervision_mode.lower()
        if mode == "binary":
            if pos_mask is None or neg_mask is None:
                raise ValueError("binary mode requires both pos_mask and neg_mask.")
            resolved_masks = {
                "pos": pos_mask.bool(),
                "neg": neg_mask.bool(),
            }
            resolved_targets = {
                "pos": float(pos_threshold),
                "neg": float(neg_target),
            }
            direct_separation_names = ["neg"]
        elif mode == "ternary":
            if full_pos_mask is None or partial_pos_mask is None or neg_mask is None:
                raise ValueError(
                    "ternary mode requires full_pos_mask, partial_pos_mask, and neg_mask."
                )
            resolved_masks = {
                "full_pos": full_pos_mask.bool(),
                "partial_pos": partial_pos_mask.bool(),
                "neg": neg_mask.bool(),
            }
            resolved_targets = {
                "full_pos": float(full_pos_threshold),
                "partial_pos": float(partial_pos_threshold),
                "neg": float(neg_target),
            }
            direct_separation_names = ["neg"]
        else:
            raise ValueError(
                "supervision_mode must be one of {'binary', 'ternary'} unless relation_masks is provided."
            )

    mask_names = list(resolved_masks.keys())
    for idx, name_i in enumerate(mask_names):
        for name_j in mask_names[idx + 1 :]:
            overlap = resolved_masks[name_i] & resolved_masks[name_j]
            if overlap.any():
                raise ValueError(
                    f"Supervision masks '{name_i}' and '{name_j}' overlap on {int(overlap.sum().item())} entries."
                )
    return resolved_masks, resolved_targets, direct_separation_names


def cme_reconstruction_loss(cme_pred, pos_mask, neg_mask):
    """MSE重建损失：正位置(pos_mask)目标=1，负位置(neg_mask)目标=0"""
    loss = pos_mask.float() * (cme_pred - 1) ** 2 + neg_mask.float() * cme_pred ** 2
    return loss.mean()


def weak_pos_neg_loss(
    sim_matrix: torch.Tensor,
    pos_mask: torch.Tensor | None = None,
    neg_mask: torch.Tensor | None = None,
    pos_threshold: float = 0.5,
    supervision_mode: str = "binary",
    full_pos_mask: torch.Tensor | None = None,
    partial_pos_mask: torch.Tensor | None = None,
    relation_masks: dict[str, torch.Tensor] | None = None,
    relation_targets: dict[str, float] | None = None,
    full_pos_threshold: float = 1.0,
    partial_pos_threshold: float = 0.5,
    neg_target: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    relation_masks, relation_targets, _ = resolve_relation_supervision(
        supervision_mode=supervision_mode,
        pos_mask=pos_mask,
        neg_mask=neg_mask,
        full_pos_mask=full_pos_mask,
        partial_pos_mask=partial_pos_mask,
        relation_masks=relation_masks,
        relation_targets=relation_targets,
        pos_threshold=pos_threshold,
        full_pos_threshold=full_pos_threshold,
        partial_pos_threshold=partial_pos_threshold,
        neg_target=neg_target,
    )

    loss_total = torch.zeros((), device=sim_matrix.device, dtype=sim_matrix.dtype)
    loss_items: dict[str, torch.Tensor] = {}

    for name, mask in relation_masks.items():
        pred = sim_matrix[mask]
        target = relation_targets[name]
        if pred.numel() == 0:
            loss_part = torch.zeros((), device=sim_matrix.device, dtype=sim_matrix.dtype)
            print(f"Warning: No '{name}' pairs in this batch.")
        elif target <= 0.0:
            loss_part = F.mse_loss(pred, torch.full_like(pred, target))
        else:
            loss_part = (F.relu(target - pred) ** 2).mean()

        loss_items[f"loss_{name}"] = loss_part
        loss_total = loss_total + loss_part

    loss_items["loss_total"] = loss_total
    return loss_total, loss_items


def maxpool_bidirectional_cme_loss(
    sim_heads: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    margin: float = 0.2,
    pos_min_target: float = 0.7,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Subspace Max/Min-Pooling 双向约束损失:
    - Positive: max -> 1, min -> pos_min_target
    - Negative: min -> 0, max < (1 - margin)
    """
    pos_mask = pos_mask.bool()
    neg_mask = neg_mask.bool()

    sim_max, _ = torch.max(sim_heads, dim=0)
    sim_min, _ = torch.min(sim_heads, dim=0)

    pred_pos_max = sim_max[pos_mask]
    loss_pos_max = F.mse_loss(pred_pos_max, torch.ones_like(pred_pos_max))

    pred_pos_min = sim_min[pos_mask]
    loss_pos_min = F.mse_loss(pred_pos_min, torch.full_like(pred_pos_min, pos_min_target))

    pred_neg_min = sim_min[neg_mask]
    loss_neg_min = F.mse_loss(pred_neg_min, torch.zeros_like(pred_neg_min))

    pred_neg_max = sim_max[neg_mask]
    loss_neg_max = F.relu(pred_neg_max - (1 - margin)).mean()

    loss_cme = loss_pos_max + loss_pos_min + loss_neg_min + loss_neg_max
    # loss_cme = loss_pos_max + loss_neg_min + loss_neg_max
    loss_items = {
        "loss_pos_max": loss_pos_max,
        "loss_pos_min": loss_pos_min,
        "loss_neg_min": loss_neg_min,
        "loss_neg_max": loss_neg_max,
        "loss_cme": loss_cme,
    }
    return loss_cme, loss_items



def neg_pair_partition_separation_loss(
    partition_soft: torch.Tensor,
    neg_mask: torch.Tensor,
    power: float = 2.0,
) -> torch.Tensor:
    """
    对 neg_mask 中基因对的“同超边概率”进行惩罚。
    overlap_ij = sum_k p(i->k) * p(j->k)，值越大表示越倾向同超边。
    """
    neg_mask = neg_mask.bool()
    overlap = partition_soft @ partition_soft.T
    overlap_neg = overlap[neg_mask]
    if overlap_neg.numel() == 0:
        print("Warning: No negative pairs in this batch.")
        return torch.zeros((), device=partition_soft.device, dtype=partition_soft.dtype)
    return (overlap_neg ** power).mean()


def build_relation_inconsistency_mask(
    pos_mask: torch.Tensor | None = None,
    neg_mask: torch.Tensor | None = None,
    include_direct_neg: bool = True,
    supervision_mode: str = "binary",
    full_pos_mask: torch.Tensor | None = None,
    partial_pos_mask: torch.Tensor | None = None,
    relation_masks: dict[str, torch.Tensor] | None = None,
    relation_targets: dict[str, float] | None = None,
    pos_threshold: float = 0.5,
    full_pos_threshold: float = 1.0,
    partial_pos_threshold: float = 0.5,
    neg_target: float = 0.0,
    direct_separation_names: list[str] | None = None,
) -> torch.Tensor:
    """
    构建“监督关系不一致”掩码：
    - 若 relation(i, k) 与 relation(j, k) 属于不同监督类别，则 i/j 不一致
    - 未标注位置自动忽略，不参与冲突计数
    - direct_separation_names 指定哪些“直接监督”也应加入 partition 分离惩罚
    """
    relation_masks, _, default_direct_separation_names = resolve_relation_supervision(
        supervision_mode=supervision_mode,
        pos_mask=pos_mask,
        neg_mask=neg_mask,
        full_pos_mask=full_pos_mask,
        partial_pos_mask=partial_pos_mask,
        relation_masks=relation_masks,
        relation_targets=relation_targets,
        pos_threshold=pos_threshold,
        full_pos_threshold=full_pos_threshold,
        partial_pos_threshold=partial_pos_threshold,
        neg_target=neg_target,
    )

    relation_names = list(relation_masks.keys())
    relation_floats = {
        name: mask.float()
        for name, mask in relation_masks.items()
    }
    first_mask = next(iter(relation_masks.values()))
    conflict_count = torch.zeros(first_mask.shape, device=first_mask.device, dtype=torch.float32)
    for idx, name_i in enumerate(relation_names):
        for name_j in relation_names[idx + 1 :]:
            mask_i = relation_floats[name_i]
            mask_j = relation_floats[name_j]
            conflict_count = conflict_count + mask_i @ mask_j.T + mask_j @ mask_i.T
    incons_mask = conflict_count > 0

    if direct_separation_names is None:
        direct_separation_names = default_direct_separation_names if include_direct_neg else []
    for name in direct_separation_names:
        if name not in relation_masks:
            raise ValueError(f"Unknown direct separation relation name: {name}")
        incons_mask = incons_mask | relation_masks[name]

    # 去掉对角线
    eye = torch.eye(incons_mask.shape[0], dtype=torch.bool, device=incons_mask.device)
    incons_mask = incons_mask & (~eye)
    return incons_mask


def inconsistent_pair_partition_separation_loss(
    partition_soft: torch.Tensor,
    pos_mask: torch.Tensor | None = None,
    neg_mask: torch.Tensor | None = None,
    power: float = 2.0,
    include_direct_neg: bool = True,
    supervision_mode: str = "binary",
    full_pos_mask: torch.Tensor | None = None,
    partial_pos_mask: torch.Tensor | None = None,
    relation_masks: dict[str, torch.Tensor] | None = None,
    relation_targets: dict[str, float] | None = None,
    pos_threshold: float = 0.5,
    full_pos_threshold: float = 1.0,
    partial_pos_threshold: float = 0.5,
    neg_target: float = 0.0,
    direct_separation_names: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对“监督关系不一致”的基因对加同超边惩罚。
    返回: (loss, inconsistency_mask)
    """
    incons_mask = build_relation_inconsistency_mask(
        pos_mask=pos_mask,
        neg_mask=neg_mask,
        include_direct_neg=include_direct_neg,
        supervision_mode=supervision_mode,
        full_pos_mask=full_pos_mask,
        partial_pos_mask=partial_pos_mask,
        relation_masks=relation_masks,
        relation_targets=relation_targets,
        pos_threshold=pos_threshold,
        full_pos_threshold=full_pos_threshold,
        partial_pos_threshold=partial_pos_threshold,
        neg_target=neg_target,
        direct_separation_names=direct_separation_names,
    )
    overlap = partition_soft @ partition_soft.T
    overlap_incons = overlap[incons_mask]
    if overlap_incons.numel() == 0:
        loss = torch.zeros((), device=partition_soft.device, dtype=partition_soft.dtype)
    else:
        loss = (overlap_incons ** power).mean()
    return loss, incons_mask


def maxpool_partition_total_loss(
    sim_heads: torch.Tensor,
    partition_soft: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    alpha: float = 0.1,
    beta: float = 1.0,
    margin: float = 0.2,
    pos_min_target: float = 0.7,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """组合损失: CME双向约束 + partition熵正则 + neg分离惩罚。"""
    loss_cme, loss_items = maxpool_bidirectional_cme_loss(
        sim_heads=sim_heads,
        pos_mask=pos_mask,
        neg_mask=neg_mask,
        margin=margin,
        pos_min_target=pos_min_target,
    )

    entropy = -torch.sum(partition_soft * torch.log(partition_soft + 1e-8), dim=1)
    loss_entropy = entropy.mean()

    loss_partition_neg_sep = neg_pair_partition_separation_loss(
        partition_soft=partition_soft,
        neg_mask=neg_mask,
    )

    loss_total = loss_cme + alpha * loss_entropy + beta * loss_partition_neg_sep
    loss_items.update(
        {
            "loss_entropy": loss_entropy,
            "loss_partition_neg_sep": loss_partition_neg_sep,
            "loss_total": loss_total,
        }
    )
    return loss_total, loss_items
