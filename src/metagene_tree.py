from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from typing import Mapping, Sequence

import numpy as np

from src.cme_supervision import compute_cme_matrix


@dataclass
class MetageneTreeResult:
    module_gene_indices: dict[int, np.ndarray]
    module_gene_names: dict[int, list[str]]
    module_ids: list[int]
    node_labels: list[str]
    metagene_matrix: np.ndarray
    metagene_cme: np.ndarray
    mutual_exclusion_adjacency: np.ndarray
    mutual_exclusion_adjacency_with_empty: np.ndarray
    complement_adjacency: np.ndarray
    tree_edges: list[tuple[int, int]]
    root_index: int
    empty_node_index: int
    complement_degrees: np.ndarray
    module_assignment_table: list[dict[str, object]]
    stats: dict[str, object]


@dataclass
class CellTypeScoringResult:
    metagene_cell_type_table: list[dict[str, object]]
    cell_assignment_table: list[dict[str, object]]
    cell_type_score_matrix: np.ndarray
    cell_type_labels: list[str]
    predicted_cell_types: np.ndarray
    best_metagene_indices: np.ndarray
    stats: dict[str, object]


@dataclass
class CellHierarchyScoringResult:
    hierarchy_relation_table: list[dict[str, object]]
    cell_assignment_table: list[dict[str, object]]
    predicted_parent_types: np.ndarray
    reference_parent_types: np.ndarray
    stats: dict[str, object]


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
    """Map each meaningful hyperedge to the genes assigned to it."""
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


def build_metagene_matrix(
    expression_gene_by_cell: np.ndarray,
    module_gene_indices: Mapping[int, Sequence[int]],
    *,
    aggregation: str = "sum",
) -> np.ndarray:
    """Aggregate the genes in each module into one metagene expression vector."""
    expression = np.asarray(expression_gene_by_cell, dtype=np.float32)
    if expression.ndim != 2:
        raise ValueError(f"expression_gene_by_cell must be 2D, got shape {expression.shape}.")

    metagenes = []
    for gene_indices in module_gene_indices.values():
        gene_indices = np.asarray(gene_indices, dtype=np.int64)
        if gene_indices.size == 0:
            metagenes.append(np.zeros(expression.shape[1], dtype=np.float32))
            continue

        module_expression = expression[gene_indices]
        if aggregation == "sum":
            metagenes.append(module_expression.sum(axis=0, dtype=np.float32))
        elif aggregation == "mean":
            metagenes.append(module_expression.mean(axis=0, dtype=np.float32))
        else:
            raise ValueError("aggregation must be one of {'sum', 'mean'}.")

    if not metagenes:
        raise ValueError("At least one meaningful module is required.")
    return np.vstack(metagenes).astype(np.float32)


def compare_module_assignments(
    module_gene_indices: Mapping[int, Sequence[int]],
    reference_gene_modules: Sequence[object],
    *,
    exclude_reference_labels: Sequence[object] = ("NA", ""),
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Compare inferred hyperedge modules to reference labels by majority vote."""
    reference = list(reference_gene_modules)
    excluded = {str(label) for label in exclude_reference_labels}
    rows: list[dict[str, object]] = []
    total_correct = 0
    total_evaluable = 0

    for node_index, (module_idx, gene_indices) in enumerate(module_gene_indices.items()):
        label_counts: dict[str, int] = {}
        missing_reference = 0
        for gene_idx in np.asarray(gene_indices, dtype=np.int64):
            if int(gene_idx) >= len(reference):
                missing_reference += 1
                continue
            label = str(reference[int(gene_idx)])
            if label in excluded:
                continue
            label_counts[label] = label_counts.get(label, 0) + 1

        if label_counts:
            identified_module, identified_count = sorted(
                label_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[0]
            evaluable_genes = int(sum(label_counts.values()))
            accuracy = identified_count / evaluable_genes
        else:
            identified_module = None
            identified_count = 0
            evaluable_genes = 0
            accuracy = np.nan

        total_correct += int(identified_count)
        total_evaluable += int(evaluable_genes)
        rows.append(
            {
                "node_index": int(node_index),
                "hyperedge": int(module_idx),
                "num_genes": int(len(gene_indices)),
                "evaluable_genes": evaluable_genes,
                "identified_module": identified_module,
                "identified_count": int(identified_count),
                "accuracy": float(accuracy),
                "missing_reference": int(missing_reference),
                "label_counts": label_counts,
            }
        )

    weighted_accuracy = total_correct / total_evaluable if total_evaluable > 0 else np.nan
    summary = {
        "assignment_total_correct": int(total_correct),
        "assignment_total_evaluable": int(total_evaluable),
        "weighted_assignment_accuracy": float(weighted_accuracy),
    }
    return rows, summary


def _reference_labels_from_adata(adata, *, column: str, num_genes: int) -> list[object]:
    if adata is None:
        return []
    if column not in adata.var:
        raise ValueError(f"adata.var does not contain column {column!r}.")
    labels = adata.var[column]
    if hasattr(labels, "iloc"):
        return labels.iloc[:num_genes].tolist()
    return list(labels)[:num_genes]


def _reference_cell_labels_from_adata(adata, *, column: str, num_cells: int) -> list[object]:
    if adata is None:
        return []
    if column not in adata.obs:
        raise ValueError(f"adata.obs does not contain column {column!r}.")
    labels = adata.obs[column]
    if hasattr(labels, "iloc"):
        return labels.iloc[:num_cells].tolist()
    return list(labels)[:num_cells]


def _format_accuracy(value: object) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if np.isnan(value):
        return "NA"
    return f"{value:.0%}"


def _format_node_labels(
    module_gene_indices: Mapping[int, Sequence[int]],
    module_assignment_table: Sequence[Mapping[str, object]],
) -> list[str]:
    assignment_by_module = {
        int(row["hyperedge"]): row for row in module_assignment_table
    }
    labels = []
    for module_idx, gene_indices in module_gene_indices.items():
        label = f"H{module_idx}\nn={len(gene_indices)}"
        assignment = assignment_by_module.get(int(module_idx))
        if assignment is not None and assignment.get("identified_module") is not None:
            label += (
                f"\nid={assignment['identified_module']} "
                f"({_format_accuracy(assignment.get('accuracy'))})"
            )
        labels.append(label)
    return labels


def leaf_node_indices_from_tree(result: MetageneTreeResult) -> list[int]:
    child_nodes = {int(child) for _, child in result.tree_edges}
    parent_nodes = {int(parent) for parent, _ in result.tree_edges}
    leaves = sorted(child_nodes - parent_nodes)
    return [
        node_idx
        for node_idx in leaves
        if node_idx != result.empty_node_index and node_idx < result.metagene_matrix.shape[0]
    ]


def score_cell_types_from_metagenes(
    metagene_matrix: np.ndarray,
    *,
    adata=None,
    reference_cell_types: Sequence[object] | None = None,
    obs_column: str = "cell_type",
    module_ids: Sequence[int] | None = None,
    candidate_node_indices: Sequence[int] | None = None,
    normalize: str = "none",
    exclude_reference_labels: Sequence[object] = ("NA", ""),
) -> CellTypeScoringResult:
    """Infer cell types from metagene scores and compare them to reference cell labels."""
    scores = np.asarray(metagene_matrix, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"metagene_matrix must be 2D, got shape {scores.shape}.")

    if normalize == "none":
        scored = scores.copy()
    elif normalize == "zscore_by_metagene":
        mean = scores.mean(axis=1, keepdims=True)
        std = scores.std(axis=1, keepdims=True)
        scored = (scores - mean) / np.where(std > 0, std, 1.0)
    else:
        raise ValueError("normalize must be one of {'none', 'zscore_by_metagene'}.")

    num_metagenes, num_cells = scored.shape
    if module_ids is None:
        module_ids = list(range(num_metagenes))
    module_ids = [int(module_id) for module_id in module_ids]
    if len(module_ids) != num_metagenes:
        raise ValueError("module_ids length must match the number of metagenes.")
    if candidate_node_indices is None:
        candidate_node_indices = list(range(num_metagenes))
    candidate_node_indices = [int(idx) for idx in candidate_node_indices]
    if not candidate_node_indices:
        raise ValueError("At least one candidate node is required for cell type scoring.")
    if min(candidate_node_indices) < 0 or max(candidate_node_indices) >= num_metagenes:
        raise ValueError("candidate_node_indices contains an index outside metagene_matrix.")

    if reference_cell_types is None and adata is not None:
        reference_cell_types = _reference_cell_labels_from_adata(
            adata,
            column=obs_column,
            num_cells=num_cells,
        )
    if reference_cell_types is None:
        raise ValueError("Either adata or reference_cell_types must be provided.")

    reference = np.asarray([str(label) for label in list(reference_cell_types)[:num_cells]], dtype=object)
    if reference.shape[0] != num_cells:
        raise ValueError(
            f"reference_cell_types has length {reference.shape[0]}, expected {num_cells}."
        )

    excluded = {str(label) for label in exclude_reference_labels}
    evaluable_mask = np.asarray([label not in excluded for label in reference], dtype=bool)
    cell_type_labels = sorted(set(reference[evaluable_mask].tolist()))
    if not cell_type_labels:
        raise ValueError("No evaluable cell type labels found.")

    candidate_scores = scored[candidate_node_indices]
    score_matrix = np.zeros((len(candidate_node_indices), len(cell_type_labels)), dtype=np.float32)
    for type_idx, cell_type in enumerate(cell_type_labels):
        mask = reference == cell_type
        score_matrix[:, type_idx] = candidate_scores[:, mask].mean(axis=1)

    best_type_indices = np.argmax(score_matrix, axis=1)
    metagene_cell_type_table: list[dict[str, object]] = []
    candidate_max_scores = candidate_scores.max(axis=0)
    for row_idx, node_idx in enumerate(candidate_node_indices):
        module_id = module_ids[node_idx]
        identified_type = cell_type_labels[int(best_type_indices[row_idx])]
        assigned_cells = candidate_scores[row_idx] == candidate_max_scores
        assigned_evaluable = assigned_cells & evaluable_mask
        assigned_reference = reference[assigned_evaluable]
        correct_assigned = int((assigned_reference == identified_type).sum())
        evaluable_assigned = int(assigned_evaluable.sum())
        assigned_accuracy = (
            correct_assigned / evaluable_assigned if evaluable_assigned > 0 else np.nan
        )
        metagene_cell_type_table.append(
            {
                "node_index": int(node_idx),
                "hyperedge": int(module_id),
                "identified_cell_type": identified_type,
                "mean_score": float(score_matrix[row_idx, best_type_indices[row_idx]]),
                "assigned_cells": int(assigned_cells.sum()),
                "evaluable_assigned_cells": evaluable_assigned,
                "correct_assigned_cells": correct_assigned,
                "assigned_accuracy": float(assigned_accuracy),
            }
        )

    best_candidate_rows = np.argmax(candidate_scores, axis=0).astype(np.int64)
    best_metagene_indices = np.asarray(
        [candidate_node_indices[int(row_idx)] for row_idx in best_candidate_rows],
        dtype=np.int64,
    )
    predicted_cell_types = np.asarray(
        [cell_type_labels[int(best_type_indices[int(row_idx)])] for row_idx in best_candidate_rows],
        dtype=object,
    )

    correct_mask = (predicted_cell_types == reference) & evaluable_mask
    total_correct = int(correct_mask.sum())
    total_evaluable = int(evaluable_mask.sum())
    overall_accuracy = total_correct / total_evaluable if total_evaluable > 0 else np.nan

    cell_assignment_table = [
        {
            "cell_index": int(cell_idx),
            "best_hyperedge": int(module_ids[int(best_metagene_indices[cell_idx])]),
            "best_node_index": int(best_metagene_indices[cell_idx]),
            "predicted_cell_type": str(predicted_cell_types[cell_idx]),
            "reference_cell_type": str(reference[cell_idx]),
            "correct": bool(correct_mask[cell_idx]),
        }
        for cell_idx in range(num_cells)
    ]
    stats = {
        "cell_type_total_correct": total_correct,
        "cell_type_total_evaluable": total_evaluable,
        "cell_type_accuracy": float(overall_accuracy),
        "num_metagenes": int(num_metagenes),
        "num_candidate_nodes": int(len(candidate_node_indices)),
        "candidate_node_indices": candidate_node_indices,
        "num_cells": int(num_cells),
        "normalize": normalize,
        "obs_column": obs_column,
    }

    return CellTypeScoringResult(
        metagene_cell_type_table=metagene_cell_type_table,
        cell_assignment_table=cell_assignment_table,
        cell_type_score_matrix=score_matrix,
        cell_type_labels=cell_type_labels,
        predicted_cell_types=predicted_cell_types,
        best_metagene_indices=best_metagene_indices,
        stats=stats,
    )


def score_cell_types_from_metagene_tree(
    result: MetageneTreeResult,
    *,
    adata=None,
    reference_cell_types: Sequence[object] | None = None,
    obs_column: str = "cell_type",
    normalize: str = "none",
    exclude_reference_labels: Sequence[object] = ("NA", ""),
) -> CellTypeScoringResult:
    leaf_indices = leaf_node_indices_from_tree(result)
    return score_cell_types_from_metagenes(
        result.metagene_matrix,
        adata=adata,
        reference_cell_types=reference_cell_types,
        obs_column=obs_column,
        module_ids=result.module_ids,
        candidate_node_indices=leaf_indices,
        normalize=normalize,
        exclude_reference_labels=exclude_reference_labels,
    )


def infer_parent_cell_type(cell_type: object) -> str:
    label = str(cell_type)
    match = re.match(r"^[A-Za-z]+", label)
    if match:
        return match.group(0)
    for separator in ("_", "-", "/"):
        if separator in label:
            return label.split(separator, 1)[0]
    return label


def load_cell_type_parent_map_from_xml(
    tree_xml_path: str | Path,
    *,
    include_root_children: bool = True,
) -> dict[str, str]:
    """Read explicit child -> parent cell-type relationships from a simulator XML tree."""
    root = ET.parse(tree_xml_path).getroot()
    parent_map: dict[str, str] = {}

    def visit(node: ET.Element, parent_name: str | None = None) -> None:
        if node.tag != "celltype":
            for child in node:
                visit(child, parent_name)
            return

        name = node.attrib.get("name")
        if not name:
            raise ValueError(f"Found a celltype without a name in {tree_xml_path}.")
        if parent_name is not None and (include_root_children or parent_name != "Root"):
            parent_map[str(name)] = str(parent_name)

        for child in node:
            visit(child, str(name))

    visit(root)
    return parent_map


def make_xml_parent_extractor(
    parent_map: Mapping[str, str],
    *,
    fallback=infer_parent_cell_type,
):
    """Create a parent extractor that uses XML relationships before falling back to labels."""
    normalized = {str(child): str(parent) for child, parent in parent_map.items()}

    def parent_extractor(cell_type: object) -> str:
        label = str(cell_type)
        if label in normalized:
            return normalized[label]
        return fallback(label)

    return parent_extractor


def score_cell_hierarchy_from_cell_types(
    cell_type_result: CellTypeScoringResult,
    *,
    parent_extractor=infer_parent_cell_type,
) -> CellHierarchyScoringResult:
    predicted = np.asarray(cell_type_result.predicted_cell_types, dtype=object)
    reference = np.asarray(
        [row["reference_cell_type"] for row in cell_type_result.cell_assignment_table],
        dtype=object,
    )
    if predicted.shape[0] != reference.shape[0]:
        raise ValueError("Predicted and reference cell type arrays must have the same length.")

    predicted_parent = np.asarray([parent_extractor(label) for label in predicted], dtype=object)
    reference_parent = np.asarray([parent_extractor(label) for label in reference], dtype=object)
    correct_mask = predicted_parent == reference_parent

    relation_items: dict[tuple[str, str], dict[str, object]] = {}
    for pred_type, pred_parent, ref_parent, is_correct in zip(
        predicted,
        predicted_parent,
        reference_parent,
        correct_mask,
    ):
        key = (str(pred_type), str(pred_parent))
        if key not in relation_items:
            relation_items[key] = {
                "predicted_cell_type": str(pred_type),
                "predicted_parent_type": str(pred_parent),
                "assigned_cells": 0,
                "correct_parent_cells": 0,
                "reference_parent_counts": {},
            }
        item = relation_items[key]
        item["assigned_cells"] = int(item["assigned_cells"]) + 1
        item["correct_parent_cells"] = int(item["correct_parent_cells"]) + int(is_correct)
        counts = item["reference_parent_counts"]
        counts[str(ref_parent)] = counts.get(str(ref_parent), 0) + 1

    relation_table = []
    for item in relation_items.values():
        assigned_cells = int(item["assigned_cells"])
        correct_parent_cells = int(item["correct_parent_cells"])
        relation_table.append(
            {
                **item,
                "parent_accuracy": (
                    correct_parent_cells / assigned_cells if assigned_cells > 0 else np.nan
                ),
            }
        )
    relation_table.sort(key=lambda row: (row["predicted_parent_type"], row["predicted_cell_type"]))

    cell_assignment_table = [
        {
            "cell_index": int(row["cell_index"]),
            "predicted_cell_type": str(predicted[idx]),
            "reference_cell_type": str(reference[idx]),
            "predicted_parent_type": str(predicted_parent[idx]),
            "reference_parent_type": str(reference_parent[idx]),
            "correct_parent": bool(correct_mask[idx]),
        }
        for idx, row in enumerate(cell_type_result.cell_assignment_table)
    ]

    total_correct = int(correct_mask.sum())
    total_evaluable = int(correct_mask.shape[0])
    stats = {
        "hierarchy_total_correct": total_correct,
        "hierarchy_total_evaluable": total_evaluable,
        "hierarchy_accuracy": total_correct / total_evaluable if total_evaluable > 0 else np.nan,
        "parent_rule": "leading letters, fallback split on _, -, /",
    }

    return CellHierarchyScoringResult(
        hierarchy_relation_table=relation_table,
        cell_assignment_table=cell_assignment_table,
        predicted_parent_types=predicted_parent,
        reference_parent_types=reference_parent,
        stats=stats,
    )


def compute_metagene_cme(
    metagene_matrix: np.ndarray,
    *,
    normalize: bool = False,
    use_numba: bool = True,
) -> np.ndarray:
    return compute_cme_matrix(metagene_matrix, normalize=normalize, use_numba=use_numba)


def cme_to_mutual_exclusion_graph(cme_matrix: np.ndarray, *, cme_threshold: float) -> np.ndarray:
    cme = np.asarray(cme_matrix, dtype=np.float32)
    if cme.ndim != 2 or cme.shape[0] != cme.shape[1]:
        raise ValueError(f"cme_matrix must be square, got shape {cme.shape}.")
    adjacency = cme > cme_threshold
    adjacency = adjacency | adjacency.T
    np.fill_diagonal(adjacency, False)
    return adjacency.astype(bool)


def add_empty_node(
    adjacency: np.ndarray,
    *,
    node_labels: Sequence[str] | None = None,
    empty_label: str = "EMPTY",
) -> tuple[np.ndarray, list[str], int]:
    adjacency = np.asarray(adjacency, dtype=bool)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be square, got shape {adjacency.shape}.")

    num_nodes = adjacency.shape[0]
    with_empty = np.zeros((num_nodes + 1, num_nodes + 1), dtype=bool)
    with_empty[:num_nodes, :num_nodes] = adjacency
    labels = list(node_labels) if node_labels is not None else [str(idx) for idx in range(num_nodes)]
    labels.append(empty_label)
    return with_empty, labels, num_nodes


def complement_graph(adjacency: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adjacency, dtype=bool)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be square, got shape {adjacency.shape}.")
    complement = ~adjacency
    np.fill_diagonal(complement, False)
    return complement.astype(bool)


def choose_root_by_degree(
    adjacency: np.ndarray,
    *,
    preferred_root_index: int | None = None,
) -> int:
    degrees = np.asarray(adjacency, dtype=bool).sum(axis=1)
    max_degree = degrees.max()
    max_nodes = np.flatnonzero(degrees == max_degree)
    if preferred_root_index is not None and preferred_root_index in set(max_nodes.tolist()):
        return int(preferred_root_index)
    return int(max_nodes[0])


def degree_priority_tree_edges(
    adjacency: np.ndarray,
    *,
    root_index: int | None = None,
    preferred_root_index: int | None = None,
    child_strategy: str = "max_degree_ties",
) -> list[tuple[int, int]]:
    """Build a spanning tree from graph edges, expanding high-degree neighbors first."""
    graph = np.asarray(adjacency, dtype=bool)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError(f"adjacency must be square, got shape {graph.shape}.")
    if child_strategy not in {"max_degree_ties", "all_neighbors"}:
        raise ValueError("child_strategy must be one of {'max_degree_ties', 'all_neighbors'}.")

    if root_index is None:
        root_index = choose_root_by_degree(graph, preferred_root_index=preferred_root_index)

    degrees = graph.sum(axis=1)
    visited = {int(root_index)}
    active = [int(root_index)]
    edges: list[tuple[int, int]] = []

    def select_children(parent: int) -> list[int]:
        neighbors = [int(idx) for idx in np.flatnonzero(graph[parent]) if int(idx) not in visited]
        neighbors.sort(key=lambda idx: (-int(degrees[idx]), idx))
        if child_strategy == "all_neighbors" or not neighbors:
            return neighbors
        max_degree = degrees[neighbors[0]]
        return [idx for idx in neighbors if degrees[idx] == max_degree]

    while len(visited) < graph.shape[0]:
        next_active: list[int] = []
        for parent in active:
            for child in select_children(parent):
                visited.add(child)
                edges.append((parent, child))
                next_active.append(child)

        if next_active:
            active = next_active
            continue

        fallback_parent = None
        for parent in sorted(visited, key=lambda idx: (-int(degrees[idx]), idx)):
            if any(int(idx) not in visited for idx in np.flatnonzero(graph[parent])):
                fallback_parent = int(parent)
                break

        if fallback_parent is None:
            missing = sorted(set(range(graph.shape[0])) - visited)
            raise ValueError(f"Complement graph is disconnected from root; missing nodes: {missing}.")
        active = [fallback_parent]

    return edges


def validate_metagene_tree_result(result: MetageneTreeResult) -> dict[str, int | bool]:
    num_nodes = len(result.node_labels)
    tree_edge_count = len(result.tree_edges)
    tree_nodes = {result.root_index}
    for parent, child in result.tree_edges:
        tree_nodes.add(parent)
        tree_nodes.add(child)

    return {
        "num_nodes": num_nodes,
        "tree_edges": tree_edge_count,
        "tree_has_all_nodes": len(tree_nodes) == num_nodes,
        "tree_edge_count_ok": tree_edge_count == num_nodes - 1,
        "root_is_empty_node": result.root_index == result.empty_node_index,
        "mutex_graph_is_symmetric": bool(
            np.array_equal(
                result.mutual_exclusion_adjacency_with_empty,
                result.mutual_exclusion_adjacency_with_empty.T,
            )
        ),
        "complement_graph_is_symmetric": bool(
            np.array_equal(result.complement_adjacency, result.complement_adjacency.T)
        ),
    }


def build_metagene_tree_from_result(
    result,
    expression_gene_by_cell: np.ndarray,
    *,
    adata=None,
    gene_names: Sequence[str] | None = None,
    reference_gene_modules: Sequence[object] | None = None,
    reference_column: str = "gene_module",
    exclude_reference_labels: Sequence[object] = ("NA", ""),
    cme_threshold: float = 0.66,
    assignment: str = "argmax",
    membership_threshold: float = 0.5,
    aggregation: str = "sum",
    empty_label: str = "EMPTY",
    child_strategy: str = "max_degree_ties",
    use_numba: bool = True,
) -> MetageneTreeResult:
    module_gene_indices = genes_by_hyperedge(
        result.partition,
        null_hyperedge_index=result.null_hyperedge_index,
        meaningful_hyperedge_indices=range(result.num_gene_modules),
        assignment=assignment,
        membership_threshold=membership_threshold,
    )
    module_ids = list(module_gene_indices.keys())

    if gene_names is None:
        module_gene_names = {
            module_idx: [str(idx) for idx in gene_indices]
            for module_idx, gene_indices in module_gene_indices.items()
        }
    else:
        names = list(gene_names)
        module_gene_names = {
            module_idx: [names[int(idx)] for idx in gene_indices]
            for module_idx, gene_indices in module_gene_indices.items()
        }

    num_genes = np.asarray(expression_gene_by_cell).shape[0]
    if reference_gene_modules is None and adata is not None:
        reference_gene_modules = _reference_labels_from_adata(
            adata,
            column=reference_column,
            num_genes=num_genes,
        )
    if reference_gene_modules is None:
        module_assignment_table: list[dict[str, object]] = []
        assignment_summary: dict[str, object] = {}
    else:
        reference_gene_modules = list(reference_gene_modules)[:num_genes]
        module_assignment_table, assignment_summary = compare_module_assignments(
            module_gene_indices,
            reference_gene_modules,
            exclude_reference_labels=exclude_reference_labels,
        )

    module_labels = _format_node_labels(
        module_gene_indices,
        module_assignment_table,
    )
    metagene_matrix = build_metagene_matrix(
        expression_gene_by_cell,
        module_gene_indices,
        aggregation=aggregation,
    )
    metagene_cme = compute_metagene_cme(
        metagene_matrix,
        normalize=False,
        use_numba=use_numba,
    )
    mutex_graph = cme_to_mutual_exclusion_graph(metagene_cme, cme_threshold=cme_threshold)
    mutex_with_empty, node_labels, empty_node_index = add_empty_node(
        mutex_graph,
        node_labels=module_labels,
        empty_label=f"{empty_label}\nn=0",
    )
    complement = complement_graph(mutex_with_empty)
    root_index = choose_root_by_degree(complement, preferred_root_index=empty_node_index)
    tree_edges = degree_priority_tree_edges(
        complement,
        root_index=root_index,
        child_strategy=child_strategy,
    )
    degrees = complement.sum(axis=1).astype(np.int64)

    upper = np.triu_indices(mutex_graph.shape[0], k=1)
    stats = {
        "num_modules": len(module_ids),
        "num_nodes_with_empty": len(node_labels),
        "empty_node_index": int(empty_node_index),
        "root_index": int(root_index),
        "root_is_empty_node": bool(root_index == empty_node_index),
        "cme_threshold": float(cme_threshold),
        "mutex_edges": int(mutex_graph[upper].sum()),
        "complement_edges": int(np.triu(complement, k=1).sum()),
        "tree_edges": len(tree_edges),
        "empty_modules": int(sum(len(indices) == 0 for indices in module_gene_indices.values())),
        "assignment": assignment,
        "aggregation": aggregation,
        "child_strategy": child_strategy,
    }
    stats.update(assignment_summary)

    return MetageneTreeResult(
        module_gene_indices=module_gene_indices,
        module_gene_names=module_gene_names,
        module_ids=module_ids,
        node_labels=node_labels,
        metagene_matrix=metagene_matrix,
        metagene_cme=metagene_cme,
        mutual_exclusion_adjacency=mutex_graph,
        mutual_exclusion_adjacency_with_empty=mutex_with_empty,
        complement_adjacency=complement,
        tree_edges=tree_edges,
        root_index=root_index,
        empty_node_index=empty_node_index,
        complement_degrees=degrees,
        module_assignment_table=module_assignment_table,
        stats=stats,
    )


def aggregate_module_inclusion_from_genes(
    result: MetageneTreeResult,
    inclusion_score_matrix: np.ndarray,
    *,
    directed_inclusion_mask: np.ndarray | None = None,
    min_mean_score: float = 0.1,
    min_directed_fraction: float = 0.0,
) -> ModuleInclusionResult:
    """
    Aggregate directed gene-level inclusion into directed module-module scores.

    mean_score_matrix[i, j] summarizes module i -> module j, where i is the
    coarser candidate module and j is the subtype candidate module.
    """
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


def _tree_layout(
    num_nodes: int,
    tree_edges: Sequence[tuple[int, int]],
    root_index: int,
) -> dict[int, tuple[float, float]]:
    children: dict[int, list[int]] = {idx: [] for idx in range(num_nodes)}
    for parent, child in tree_edges:
        children[int(parent)].append(int(child))

    positions: dict[int, tuple[float, float]] = {}
    next_x = 0.0

    def place(node: int, depth: int) -> float:
        nonlocal next_x
        if not children[node]:
            x = next_x
            next_x += 1.0
        else:
            child_x = [place(child, depth + 1) for child in children[node]]
            x = float(np.mean(child_x))
        positions[node] = (x, -float(depth))
        return x

    place(int(root_index), 0)
    return positions


def plot_metagene_tree(
    result: MetageneTreeResult,
    *,
    ax=None,
    save_path: str | Path | None = None,
    show_degrees: bool = False,
):
    import matplotlib.pyplot as plt

    if ax is None:
        width = max(8.0, len(result.node_labels) * 1.3)
        _, ax = plt.subplots(figsize=(width, 5.0))

    positions = _tree_layout(
        len(result.node_labels),
        result.tree_edges,
        result.root_index,
    )

    for parent, child in result.tree_edges:
        x0, y0 = positions[parent]
        x1, y1 = positions[child]
        ax.plot([x0, x1], [y0, y1], color="#5f6b7a", linewidth=1.4, zorder=1)

    for node_idx, label in enumerate(result.node_labels):
        x, y = positions[node_idx]
        is_empty = node_idx == result.empty_node_index
        color = "#8f9aa8" if is_empty else "#2f80ed"
        ax.scatter([x], [y], s=1300 if is_empty else 1550, color=color, edgecolor="white", zorder=2)
        degree_text = f"\ndeg={int(result.complement_degrees[node_idx])}" if show_degrees else ""
        ax.text(
            x,
            y,
            f"{label}{degree_text}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            zorder=3,
        )

    ax.set_title("Metagene CME Complement Tree")
    ax.axis("off")
    ax.margins(x=0.08, y=0.18)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=180, bbox_inches="tight")

    return ax.figure


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
