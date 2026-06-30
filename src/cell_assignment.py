from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass
class PrototypeCellAssignment:
    table: pd.DataFrame
    cell_embedding: np.ndarray
    prototypes: np.ndarray
    probabilities: np.ndarray
    leaf_positions: list[int]
    coarse_positions: list[int]
    stats: dict[str, object]


def _column_mean(x, gene_indices: Sequence[int]) -> np.ndarray:
    if len(gene_indices) == 0:
        return np.zeros(x.shape[0], dtype=np.float32)
    values = x[:, list(gene_indices)].mean(axis=1)
    return np.asarray(values).reshape(-1).astype(np.float32)


def _minmax_columns(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    lo = x.min(axis=0, keepdims=True)
    hi = x.max(axis=0, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, eps)


def _hierarchy_maps(module_inclusion) -> tuple[dict[int, int], dict[int, list[int]]]:
    parent_of: dict[int, int] = {}
    children_of: dict[int, list[int]] = {
        idx: [] for idx in range(len(module_inclusion.module_ids))
    }
    for row in module_inclusion.selected_edge_table:
        parent = int(row["source_module_pos"])
        child = int(row["target_module_pos"])
        parent_of[child] = parent
        children_of.setdefault(parent, []).append(child)
        children_of.setdefault(child, [])
    return parent_of, children_of


def _path_to_root(node: int, parent_of: Mapping[int, int]) -> list[int]:
    path = [node]
    seen = {node}
    while path[-1] in parent_of:
        parent = int(parent_of[path[-1]])
        if parent in seen:
            raise ValueError("module hierarchy contains a cycle.")
        path.append(parent)
        seen.add(parent)
    return path


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def build_leaf_prototypes(module_inclusion) -> tuple[np.ndarray, list[int], list[int]]:
    n_nodes = len(module_inclusion.module_ids)
    parent_of, children_of = _hierarchy_maps(module_inclusion)
    leaves = [idx for idx in range(n_nodes) if not children_of.get(idx)]
    prototypes = np.zeros((len(leaves), n_nodes), dtype=np.float32)
    coarse_positions: list[int] = []

    for proto_idx, leaf in enumerate(leaves):
        path = _path_to_root(leaf, parent_of)
        prototypes[proto_idx, path] = 1.0
        coarse_positions.append(path[-1])

    return prototypes, leaves, coarse_positions


def assign_cells_by_hierarchy_prototypes(
    expression_cell_by_gene,
    module_gene_indices: Mapping[int, Sequence[int]],
    module_inclusion,
    *,
    cell_names: Sequence[str] | None = None,
    temperature: float = 0.25,
) -> PrototypeCellAssignment:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    module_ids = [int(module_id) for module_id in module_inclusion.module_ids]
    raw_scores = np.column_stack(
        [
            _column_mean(expression_cell_by_gene, module_gene_indices.get(module_id, []))
            for module_id in module_ids
        ]
    )
    cell_embedding = _minmax_columns(raw_scores)
    prototypes, leaf_positions, coarse_positions = build_leaf_prototypes(module_inclusion)

    distances = ((cell_embedding[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
    probabilities = _softmax(-distances / float(temperature))
    assigned_proto = probabilities.argmax(axis=1)
    assigned_leaf_pos = np.asarray([leaf_positions[idx] for idx in assigned_proto], dtype=np.int64)
    assigned_coarse_pos = np.asarray([coarse_positions[idx] for idx in assigned_proto], dtype=np.int64)

    labels = [str(label).splitlines()[0] for label in module_inclusion.module_labels]
    names = list(cell_names) if cell_names is not None else [f"cell_{idx}" for idx in range(raw_scores.shape[0])]
    table = pd.DataFrame(
        {
            "cell_index": np.arange(raw_scores.shape[0], dtype=np.int64),
            "cell_name": names,
            "leaf_node_pos": assigned_leaf_pos,
            "leaf_hyperedge": [module_ids[idx] for idx in assigned_leaf_pos],
            "leaf_label": [labels[idx] for idx in assigned_leaf_pos],
            "coarse_node_pos": assigned_coarse_pos,
            "coarse_hyperedge": [module_ids[idx] for idx in assigned_coarse_pos],
            "coarse_label": [labels[idx] for idx in assigned_coarse_pos],
            "prototype_probability": probabilities.max(axis=1),
        }
    )

    return PrototypeCellAssignment(
        table=table,
        cell_embedding=cell_embedding,
        prototypes=prototypes,
        probabilities=probabilities,
        leaf_positions=leaf_positions,
        coarse_positions=coarse_positions,
        stats={
            "num_cells": int(raw_scores.shape[0]),
            "num_hierarchy_nodes": int(len(module_ids)),
            "num_leaf_prototypes": int(len(leaf_positions)),
            "leaf_labels": [labels[idx] for idx in leaf_positions],
        },
    )


def _demo() -> None:
    from types import SimpleNamespace

    hierarchy = SimpleNamespace(
        module_ids=[0, 1, 2],
        module_labels=["H0", "H1", "H2"],
        selected_edge_table=[
            {"source_module_pos": 0, "target_module_pos": 1},
            {"source_module_pos": 0, "target_module_pos": 2},
        ],
    )
    result = assign_cells_by_hierarchy_prototypes(
        np.asarray([[10, 9, 0], [10, 0, 9]], dtype=np.float32),
        {0: [0], 1: [1], 2: [2]},
        hierarchy,
        cell_names=["a", "b"],
    )
    assert result.table["leaf_label"].tolist() == ["H1", "H2"]
    assert result.table["coarse_label"].tolist() == ["H0", "H0"]
    assert result.prototypes.shape == (2, 3)
    assert result.prototypes[0, 0] == result.prototypes[1, 0] == 1.0


if __name__ == "__main__":
    _demo()
