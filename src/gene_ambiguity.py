from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class GeneAmbiguityResult:
    table: pd.DataFrame
    ambiguity_weight: np.ndarray
    raw_ambiguity_score: np.ndarray
    robust_z: np.ndarray


def _as_bool_matrix(matrix, *, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    values = np.asarray(matrix, dtype=bool)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {values.shape}.")
    if shape is not None and values.shape != shape:
        raise ValueError(f"{name} shape {values.shape} does not match expected {shape}.")
    return values


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def compute_gene_ambiguity(
    positive_mask,
    negative_mask,
    partial_pos_mask=None,
    directed_inclusion_mask=None,
    gene_names: Sequence[str] | None = None,
    *,
    robust_center_z: float = 3.0,
    robust_temperature: float = 0.75,
    eps: float = 1e-8,
) -> GeneAmbiguityResult:
    """Score genes whose relation evidence is spread across multiple relation types."""
    positive = _as_bool_matrix(positive_mask, name="positive_mask")
    negative = _as_bool_matrix(negative_mask, name="negative_mask", shape=positive.shape)
    partial = (
        np.zeros_like(positive, dtype=bool)
        if partial_pos_mask is None
        else _as_bool_matrix(partial_pos_mask, name="partial_pos_mask", shape=positive.shape)
    )
    directed = (
        np.zeros_like(positive, dtype=bool)
        if directed_inclusion_mask is None
        else _as_bool_matrix(
            directed_inclusion_mask,
            name="directed_inclusion_mask",
            shape=positive.shape,
        )
    )

    np.fill_diagonal(positive, False)
    np.fill_diagonal(negative, False)
    np.fill_diagonal(partial, False)
    np.fill_diagonal(directed, False)

    positive_degree = positive.sum(axis=1).astype(np.float64)
    negative_degree = negative.sum(axis=1).astype(np.float64)
    partial_degree = partial.sum(axis=1).astype(np.float64)
    directed_out_degree = directed.sum(axis=1).astype(np.float64)
    directed_in_degree = directed.sum(axis=0).astype(np.float64)

    counts = np.vstack(
        [
            positive_degree,
            negative_degree,
            partial_degree,
            directed_out_degree,
            directed_in_degree,
        ]
    ).T
    total_relation_degree = counts.sum(axis=1)
    proportions = np.divide(
        counts,
        total_relation_degree[:, None],
        out=np.zeros_like(counts, dtype=np.float64),
        where=total_relation_degree[:, None] > 0,
    )

    num_relation_types = counts.shape[1]
    entropy = -np.sum(proportions * np.log(proportions + eps), axis=1) / np.log(
        num_relation_types
    )
    entropy = np.where(total_relation_degree > 0, entropy, 0.0)
    conflict_score = np.log1p(total_relation_degree) * entropy
    raw_score = conflict_score.copy()

    direction_total = directed_out_degree + directed_in_degree
    direction_balance = np.divide(
        np.minimum(directed_out_degree, directed_in_degree),
        np.maximum(directed_out_degree, directed_in_degree),
        out=np.zeros_like(direction_total, dtype=np.float64),
        where=direction_total > 0,
    )

    median = float(np.median(raw_score))
    mad = float(np.median(np.abs(raw_score - median)))
    robust_z = (raw_score - median) / (1.4826 * mad + eps)
    ambiguity_weight = _sigmoid((robust_z - robust_center_z) / robust_temperature)

    if gene_names is None:
        names = [str(idx) for idx in range(positive.shape[0])]
    else:
        names = [str(name) for name in list(gene_names)[: positive.shape[0]]]

    table = pd.DataFrame(
        {
            "gene_index": np.arange(positive.shape[0], dtype=int),
            "gene_name": names,
            "positive_degree": positive_degree.astype(int),
            "negative_degree": negative_degree.astype(int),
            "partial_degree": partial_degree.astype(int),
            "directed_inclusion_out_degree": directed_out_degree.astype(int),
            "directed_inclusion_in_degree": directed_in_degree.astype(int),
            "total_relation_degree": total_relation_degree.astype(int),
            "relation_type_entropy": entropy,
            "relation_type_conflict_score": conflict_score,
            "direction_balance": direction_balance,
            "raw_ambiguity_score": raw_score,
            "robust_z": robust_z,
            "ambiguity_weight": ambiguity_weight,
        }
    )
    return GeneAmbiguityResult(
        table=table,
        ambiguity_weight=ambiguity_weight.astype(np.float32),
        raw_ambiguity_score=raw_score.astype(np.float32),
        robust_z=robust_z.astype(np.float32),
    )
