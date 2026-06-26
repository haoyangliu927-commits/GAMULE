from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass
class ModuleInclusionResult:
    module_ids: list[int]
    module_labels: list[str]
    mean_score_matrix: np.ndarray
    max_score_matrix: np.ndarray
    directed_fraction_matrix: np.ndarray
    directed_count_matrix: np.ndarray
    total_pair_count_matrix: np.ndarray
    edge_table: list[dict[str, object]]
    selected_edge_table: list[dict[str, object]]
    stats: dict[str, object]


def _to_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array)


def genes_by_hyperedge(
    partition,
    *,
    null_hyperedge_index: int | None = None,
    meaningful_hyperedge_indices: Sequence[int] | None = None,
    assignment: str = "argmax",
    membership_threshold: float = 0.5,
) -> dict[int, np.ndarray]:
    partition_np = _to_numpy(partition)
    if partition_np.ndim != 2:
        raise ValueError(f"partition must be 2D, got shape {partition_np.shape}.")

    num_hyperedges = partition_np.shape[1]
    if meaningful_hyperedge_indices is None:
        meaningful_hyperedge_indices = [
            idx for idx in range(num_hyperedges) if idx != null_hyperedge_index
        ]
    module_ids = [int(idx) for idx in meaningful_hyperedge_indices]

    if assignment == "argmax":
        assigned = np.argmax(partition_np, axis=1)
        return {
            module_idx: np.flatnonzero(assigned == module_idx).astype(np.int64)
            for module_idx in module_ids
        }

    if assignment == "threshold":
        return {
            module_idx: np.flatnonzero(partition_np[:, module_idx] >= membership_threshold).astype(
                np.int64
            )
            for module_idx in module_ids
        }

    raise ValueError("assignment must be one of {'argmax', 'threshold'}.")


def aggregate_module_inclusion_from_genes(
    result,
    inclusion_score_matrix: np.ndarray,
    *,
    directed_inclusion_mask: np.ndarray | None = None,
    min_mean_score: float = 0.1,
    min_directed_fraction: float = 0.0,
) -> ModuleInclusionResult:
    scores = np.asarray(inclusion_score_matrix, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"inclusion_score_matrix must be square, got shape {scores.shape}.")

    directed = None
    if directed_inclusion_mask is not None:
        directed = np.asarray(directed_inclusion_mask, dtype=bool)
        if directed.shape != scores.shape:
            raise ValueError(
                "directed_inclusion_mask must have the same shape as inclusion_score_matrix."
            )

    module_ids = [int(module_id) for module_id in result.module_ids]
    module_labels = [
        result.node_labels[idx] if idx < len(result.node_labels) else f"H{module_id}"
        for idx, module_id in enumerate(module_ids)
    ]
    n_modules = len(module_ids)
    mean_scores = np.zeros((n_modules, n_modules), dtype=np.float32)
    max_scores = np.zeros((n_modules, n_modules), dtype=np.float32)
    directed_counts = np.zeros((n_modules, n_modules), dtype=np.int64)
    total_counts = np.zeros((n_modules, n_modules), dtype=np.int64)
    directed_fractions = np.zeros((n_modules, n_modules), dtype=np.float32)

    edge_table: list[dict[str, object]] = []
    for src_pos, src_module in enumerate(module_ids):
        src_genes = np.asarray(result.module_gene_indices[src_module], dtype=np.int64)
        for dst_pos, dst_module in enumerate(module_ids):
            if src_pos == dst_pos:
                continue
            dst_genes = np.asarray(result.module_gene_indices[dst_module], dtype=np.int64)
            total = int(src_genes.size * dst_genes.size)
            total_counts[src_pos, dst_pos] = total
            if total == 0:
                continue

            block = scores[np.ix_(src_genes, dst_genes)]
            mean_score = float(block.mean())
            max_score = float(block.max())
            mean_scores[src_pos, dst_pos] = mean_score
            max_scores[src_pos, dst_pos] = max_score

            directed_count = 0
            if directed is not None:
                directed_count = int(directed[np.ix_(src_genes, dst_genes)].sum())
                directed_counts[src_pos, dst_pos] = directed_count
                directed_fractions[src_pos, dst_pos] = directed_count / total

            directed_fraction = float(directed_fractions[src_pos, dst_pos])
            if mean_score >= min_mean_score and directed_fraction >= min_directed_fraction:
                edge_table.append(
                    {
                        "source_module_pos": int(src_pos),
                        "target_module_pos": int(dst_pos),
                        "source_hyperedge": int(src_module),
                        "target_hyperedge": int(dst_module),
                        "source_label": module_labels[src_pos],
                        "target_label": module_labels[dst_pos],
                        "mean_inclusion_score": mean_score,
                        "max_inclusion_score": max_score,
                        "directed_gene_pairs": directed_count,
                        "total_gene_pairs": total,
                        "directed_fraction": directed_fraction,
                    }
                )

    edge_table.sort(
        key=lambda row: (
            -float(row["mean_inclusion_score"]),
            -float(row["directed_fraction"]),
            int(row["source_hyperedge"]),
            int(row["target_hyperedge"]),
        )
    )

    parent_of: dict[int, int] = {}
    selected_edges: list[dict[str, object]] = []

    def would_create_cycle(parent: int, child: int) -> bool:
        cursor = parent
        seen = {child}
        while cursor in parent_of:
            if cursor in seen:
                return True
            seen.add(cursor)
            cursor = parent_of[cursor]
        return cursor in seen

    for row in edge_table:
        parent = int(row["source_module_pos"])
        child = int(row["target_module_pos"])
        if child in parent_of:
            continue
        if would_create_cycle(parent, child):
            continue
        parent_of[child] = parent
        selected_edges.append(row)
        if len(selected_edges) >= max(0, n_modules - 1):
            break

    stats = {
        "num_modules": int(n_modules),
        "candidate_edges": int(len(edge_table)),
        "selected_edges": int(len(selected_edges)),
        "min_mean_score": float(min_mean_score),
        "min_directed_fraction": float(min_directed_fraction),
    }

    return ModuleInclusionResult(
        module_ids=module_ids,
        module_labels=module_labels,
        mean_score_matrix=mean_scores,
        max_score_matrix=max_scores,
        directed_fraction_matrix=directed_fractions,
        directed_count_matrix=directed_counts,
        total_pair_count_matrix=total_counts,
        edge_table=edge_table,
        selected_edge_table=selected_edges,
        stats=stats,
    )


def plot_module_inclusion_heatmaps(
    result: ModuleInclusionResult,
    *,
    save_path: str | Path | None = None,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    labels = [f"H{module_id}" for module_id in result.module_ids]
    matrices = [
        (result.mean_score_matrix, "Mean module inclusion score"),
        (result.directed_fraction_matrix, "Directed gene-pair fraction"),
    ]
    for ax, (matrix, title) in zip(axes, matrices):
        im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=max(1e-6, float(np.nanmax(matrix))))
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Target / child candidate")
        ax.set_ylabel("Source / parent candidate")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig


def plot_module_inclusion_hierarchy(
    result: ModuleInclusionResult,
    *,
    save_path: str | Path | None = None,
):
    import matplotlib.pyplot as plt

    n_modules = len(result.module_ids)
    edges = [
        (int(row["source_module_pos"]), int(row["target_module_pos"]), row)
        for row in result.selected_edge_table
    ]
    children_by_parent: dict[int, list[int]] = {idx: [] for idx in range(n_modules)}
    has_parent = set()
    for parent, child, _ in edges:
        children_by_parent[parent].append(child)
        has_parent.add(child)
    roots = [idx for idx in range(n_modules) if idx not in has_parent]

    positions: dict[int, tuple[float, float]] = {}
    next_x = 0.0

    def place(node: int, depth: int) -> float:
        nonlocal next_x
        children = children_by_parent.get(node, [])
        if not children:
            x = next_x
            next_x += 1.0
        else:
            child_x = [place(child, depth + 1) for child in children]
            x = float(np.mean(child_x))
        positions[node] = (x, -float(depth))
        return x

    for root in roots:
        place(root, 0)
        next_x += 0.5

    width = max(8.0, n_modules * 1.2)
    fig, ax = plt.subplots(figsize=(width, 5.0))
    for parent, child, row in edges:
        x0, y0 = positions[parent]
        x1, y1 = positions[child]
        ax.plot([x0, x1], [y0, y1], color="#6b7280", linewidth=1.5, zorder=1)
        ax.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            f"{float(row['mean_inclusion_score']):.2f}",
            fontsize=8,
            color="#374151",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
        )

    for idx, label in enumerate(result.module_labels):
        x, y = positions.get(idx, (float(idx), 0.0))
        ax.scatter([x], [y], s=1500, color="#7c3aed", edgecolor="white", zorder=2)
        ax.text(x, y, label, ha="center", va="center", fontsize=8, color="white", zorder=3)

    ax.set_title("Module Hierarchy From Directed Inclusion")
    ax.axis("off")
    ax.margins(x=0.08, y=0.18)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig
